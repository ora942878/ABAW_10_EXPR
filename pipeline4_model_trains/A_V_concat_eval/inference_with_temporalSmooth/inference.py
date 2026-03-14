
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import csv

from configs.paths import PATH
from pipeline4_model_trains.A_V_concat_eval.build import build_model, load_cfg, apply_runtime_defaults
from pipeline4_model_trains.common.set_seed import set_seed
from pipeline4_model_trains.common.macro_f1_from_cm import macro_f1_from_cm
from pipeline4_model_trains.common.confusion_matrix_np import confusion_matrix_np
from pipeline4_model_trains.A_V_concat_eval.eval_5fold import load_fold_csv, split_fold_txtids

# ==========================================
FIXED_WINDOW = 101
STRATEGIES = ["Mean", "Median", "Gaussian", "Hard_Vote"]
MODEL_NAME = "gate"
FOLD_CSV = PATH.EXPR_5FOLD

SINGLE_RUN_DIR = PATH.PROJECT_ROOT / "pipeline4_model_trains" / "A_V_concat_eval" / "runs" / "av_concat_gate"
CV5_RUN_DIR = PATH.PROJECT_ROOT / "pipeline4_model_trains" / "A_V_concat_eval" / "runs" / "av_concat_gate_cv5"


# ==========================================
def read_txtid_to_videoid_map(csv_path: str):
    mp = {}
    if not os.path.exists(csv_path): return mp
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t, v = str(row.get("txtid", "")).strip(), str(row.get("videoid", "")).strip()
            if t and v: mp[t] = v
    return mp


def read_expr_txt_labels(txt_path: str):
    labels = []
    if not os.path.exists(txt_path): return np.array([])
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s == "":
                labels.append(-1)
            else:
                try:
                    labels.append(int(float(s)))
                except:
                    labels.append(-1)
    return np.asarray(labels, dtype=np.int64)


def load_feat_dict_pt(pt_path: str):
    if not os.path.exists(pt_path): return None
    try:
        obj = torch.load(pt_path, map_location="cpu", weights_only=False)
        frames = obj["frames"]
        feats = obj["feats"].detach().cpu().float().numpy()
        return {int(fr): feats[i] for i, fr in enumerate(frames)}
    except:
        return None


# ==========================================
def run_inference_on_txtids(model, txtids, stats_info, cfg, device, txt2vid):
    v_mean, v_std, a_mean, a_std = stats_info
    vis_roots = [str(PATH.FACE09_V_DINOV2_FT2), str(PATH.FACE12_V_DINOV2_FT2), str(PATH.FACE15_V_DINOV2_FT2)]
    aud_root = str(PATH.OFFAUDIO_WAV2VEC2_PT_aligned_window50)

    cms = {name: np.zeros((8, 8), dtype=np.int64) for name in STRATEGIES}
    cms["Baseline"] = np.zeros((8, 8), dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for txtid in tqdm(txtids, desc="Inferring", leave=False):
            txt_path = Path(PATH.EXPR_VALID_ABAW10th) / f"{txtid}.txt"
            if not txt_path.exists(): txt_path = Path(PATH.EXPR_TRAIN_ABAW10th) / f"{txtid}.txt"
            if not txt_path.exists(): continue

            y_all = read_expr_txt_labels(str(txt_path))
            seq_len = len(y_all)
            if seq_len == 0: continue

            aud_map = load_feat_dict_pt(os.path.join(aud_root, f"{txt2vid.get(txtid, txtid)}.pt"))
            vis_maps = [load_feat_dict_pt(os.path.join(r, f"{txtid}.pt")) for r in vis_roots]
            vis_maps = [v for v in vis_maps if v is not None]

            if not aud_map or not vis_maps: continue

            V = torch.zeros((seq_len, cfg.VIS_DIM), device=device)
            A = torch.zeros((seq_len, cfg.AUD_DIM), device=device)
            mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

            valid_frs = []
            for i in range(seq_len):
                fr = i + 1
                if fr in aud_map and any(fr in vm for vm in vis_maps):
                    valid_frs.append(i)
                    A[i] = torch.from_numpy(aud_map[fr]).to(device)
                    V[i] = torch.from_numpy(np.mean(np.stack([vm[fr] for vm in vis_maps if fr in vm]), 0)).to(device)
                    if 0 <= y_all[i] < 8: mask[i] = True

            if not valid_frs: continue
            for i in range(seq_len):
                if i not in valid_frs:
                    idx = min(valid_frs, key=lambda x: abs(x - i))
                    V[i], A[i] = V[idx], A[idx]

            if getattr(cfg, "DO_BRANCH_L2_NORM", False):
                V, A = F.normalize(V, p=2, dim=-1), F.normalize(A, p=2, dim=-1)
            V, A = (V - v_mean) / v_std, (A - a_mean) / a_std

            lg = model(V, A)
            if isinstance(lg, tuple): lg = lg[0]
            y_true = y_all[mask.cpu().numpy()]
            cms["Baseline"] += confusion_matrix_np(y_true, lg[mask].argmax(-1).cpu().numpy(), 8)

            lg_T = lg.T.unsqueeze(0)
            w = FIXED_WINDOW
            pad = w // 2

            lg_padded = F.pad(lg_T, (pad, pad), mode='replicate')
            cms["Mean"] += confusion_matrix_np(y_true, F.avg_pool1d(lg_padded, w, 1).squeeze(0).T[mask].argmax(
                -1).cpu().numpy(), 8)
            cms["Median"] += confusion_matrix_np(y_true,
                                                 lg_padded.unfold(2, w, 1).median(dim=-1)[0].squeeze(0).T[mask].argmax(
                                                     -1).cpu().numpy(), 8)

            gx = torch.arange(-pad, pad + 1, device=device).float()
            gw = torch.exp(-0.5 * (gx / (w / 4.0)) ** 2);
            gw /= gw.sum()
            cms["Gaussian"] += confusion_matrix_np(y_true, F.conv1d(lg_padded, gw.view(1, 1, w).repeat(8, 1, 1),
                                                                    groups=8).squeeze(0).T[mask].argmax(
                -1).cpu().numpy(), 8)

            vh = F.pad(lg.argmax(-1).float().view(1, 1, -1), (pad, pad), mode='replicate').unfold(2, w, 1).squeeze()
            cms["Hard_Vote"] += confusion_matrix_np(y_true, torch.mode(vh, dim=-1)[0][mask].cpu().numpy(), 8)

    return {n: macro_f1_from_cm(cms[n]) for n in ["Baseline"] + STRATEGIES}


# ==========================================
def main():
    set_seed(3407)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = apply_runtime_defaults(load_cfg(MODEL_NAME))
    txt2vid = read_txtid_to_videoid_map(str(PATH.EXPR_VIDEO_INDEX_CSV))

    print("\n[INFO] Evaluating Single Split (Standard Val)...")
    model = build_model(cfg).to(device)

    ckpt = torch.load(SINGLE_RUN_DIR / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])

    stats_s = [torch.from_numpy(np.load(SINGLE_RUN_DIR / f"train_{m}.npy")).float().to(device) for m in
               ["vis_mean", "vis_std", "aud_mean", "aud_std"]]
    stats_s[1] = torch.clamp(stats_s[1], min=1e-6);
    stats_s[3] = torch.clamp(stats_s[3], min=1e-6)

    va_ids_s = [t[:-4] for t in os.listdir(PATH.EXPR_VALID_ABAW10th) if t.endswith(".txt") and t[:-4] in txt2vid]
    res_s = run_inference_on_txtids(model, va_ids_s, stats_s, cfg, device, txt2vid)

    print("\n[INFO] Evaluating 5-Fold Cross-Validation...")
    fold_df = load_fold_csv(FOLD_CSV)
    cv_results = {n: [] for n in ["Baseline"] + STRATEGIES}

    for f_id in range(1, 6):
        f_dir = CV5_RUN_DIR / f"fold_{f_id}"
        if not f_dir.exists(): continue
        ckpt_f = torch.load(f_dir / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt_f["model"])

        stats_f = [torch.from_numpy(np.load(f_dir / f"train_{m}.npy")).float().to(device) for m in
                   ["vis_mean", "vis_std", "aud_mean", "aud_std"]]
        stats_f[1] = torch.clamp(stats_f[1], min=1e-6);
        stats_f[3] = torch.clamp(stats_f[3], min=1e-6)

        _, va_rows = split_fold_txtids(fold_df, f_id)
        va_ids_f = va_rows["txtid"].astype(str).tolist()
        res_f = run_inference_on_txtids(model, va_ids_f, stats_f, cfg, device, txt2vid)
        for n in cv_results: cv_results[n].append(res_f[n])

    # ==========================================
    print("\n" + "=" * 85)
    print(f"{'Strategy (101-Frame Window)':<30} | {'Single Val F1':<15} | {'5-Fold Mean ± Std':<20}")
    print("-" * 85)
    for n in ["Baseline"] + STRATEGIES:
        s_val = res_s[n]
        cv_m, cv_s = np.mean(cv_results[n]), np.std(cv_results[n], ddof=0)
        print(f"{n:<30} | {s_val:.4f}          | {cv_m:.4f} ± {cv_s:.4f}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()