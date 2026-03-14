
import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from configs.paths import PATH
from pipeline4_model_trains.A_V_concat_eval.build import build_model, load_cfg, apply_runtime_defaults
from pipeline4_model_trains.common.set_seed import set_seed
from pipeline4_model_trains.common.macro_f1_from_cm import macro_f1_from_cm
from pipeline4_model_trains.common.confusion_matrix_np import confusion_matrix_np


# ==========================================
def read_txtid_to_videoid_map(csv_path: str):
    fallback = "/mnt/data/expr_video_index.csv"
    if not os.path.exists(csv_path) and os.path.exists(fallback): csv_path = fallback
    mp = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        cols_lower = [c.lower() for c in cols]

        def pick_col(cands):
            for c in cands:
                if c in cols_lower: return cols[cols_lower.index(c)]
            return None

        txt_col = pick_col(["txtid", "txt_id", "txt", "sample", "sampleid", "sample_id", "id"])
        vid_col = pick_col(["videoid", "video_id", "video", "vid", "vidid"])
        for row in reader:
            t, v = str(row.get(txt_col, "")).strip(), str(row.get(vid_col, "")).strip()
            if t and v: mp[t] = v
    return mp


def read_expr_txt_labels(txt_path: str):
    labels = []
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


def load_feat_pt(pt_path: str):
    if not os.path.exists(pt_path): return None
    try:
        obj = torch.load(pt_path, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict) or "frames" not in obj or "feats" not in obj: return None
        return {int(fr): feat for fr, feat in zip(obj["frames"], obj["feats"].detach().cpu().float().numpy())}
    except:
        return None


# ==========================================
SF_CKPT_PATH = PATH.PROJECT_ROOT / "pipeline4_model_trains" / "A_V_concat_eval" / "runs" / "av_concat_gate" / "best.pt"
MODEL_NAME = "gate"
MAX_WINDOW = 205
STEP = 2


def main():
    set_seed(3407)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    cfg = apply_runtime_defaults(load_cfg(MODEL_NAME))
    model = build_model(cfg).to(device)
    ckpt = torch.load(SF_CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sf_dir = SF_CKPT_PATH.parent
    v_mean = torch.from_numpy(np.load(sf_dir / "train_vis_mean.npy")).float().to(device)
    v_std = torch.clamp(torch.from_numpy(np.load(sf_dir / "train_vis_std.npy")).float(), min=1e-6).to(device)
    a_mean = torch.from_numpy(np.load(sf_dir / "train_aud_mean.npy")).float().to(device)
    a_std = torch.clamp(torch.from_numpy(np.load(sf_dir / "train_aud_std.npy")).float(), min=1e-6).to(device)

    txt2vid = read_txtid_to_videoid_map(str(PATH.EXPR_VIDEO_INDEX_CSV))
    txtids = [t[:-4] for t in os.listdir(PATH.EXPR_VALID_ABAW10th) if t.endswith(".txt") and t[:-4] in txt2vid]

    vis_roots = [str(PATH.FACE09_V_DINOV2_FT2), str(PATH.FACE12_V_DINOV2_FT2), str(PATH.FACE15_V_DINOV2_FT2)]
    aud_root = str(PATH.OFFAUDIO_WAV2VEC2_PT_aligned_window50)

    print("\n[INFO] Step 1: Fast Logits Extraction for all videos...")
    video_records = []

    with torch.no_grad():
        for txtid in tqdm(txtids, desc="Processing Videos"):
            y_all = read_expr_txt_labels(os.path.join(PATH.EXPR_VALID_ABAW10th, f"{txtid}.txt"))
            seq_len = len(y_all)
            if seq_len == 0: continue

            aud_map = load_feat_pt(os.path.join(aud_root, f"{txt2vid.get(txtid, txtid)}.pt"))
            vis_maps = [load_feat_pt(os.path.join(r, f"{txtid}.pt")) for r in vis_roots]
            vis_maps = [v for v in vis_maps if v is not None]

            if not aud_map or not vis_maps: continue

            V = torch.zeros((seq_len, cfg.VIS_DIM), dtype=torch.float32, device=device)
            A = torch.zeros((seq_len, cfg.AUD_DIM), dtype=torch.float32, device=device)
            valid_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)


            valid_indices = []
            for i in range(seq_len):
                fr = i + 1
                if fr in aud_map and any(fr in vm for vm in vis_maps):
                    valid_indices.append(i)
                    A[i] = torch.from_numpy(aud_map[fr]).to(device)
                    xs = [vm[fr] for vm in vis_maps if fr in vm]
                    V[i] = torch.from_numpy(np.mean(np.stack(xs, axis=0), axis=0)).to(device)
                    if 0 <= y_all[i] < 8:
                        valid_mask[i] = True

            if not valid_indices: continue

            for i in range(seq_len):
                if i not in valid_indices:
                    nearest = min(valid_indices, key=lambda x: abs(x - i))
                    V[i] = V[nearest]
                    A[i] = A[nearest]

            if getattr(cfg, "DO_BRANCH_L2_NORM", False):
                V, A = F.normalize(V, p=2, dim=-1), F.normalize(A, p=2, dim=-1)
            V, A = (V - v_mean) / v_std, (A - a_mean) / a_std

            chunks = 2048
            logits_list = []
            for start_idx in range(0, seq_len, chunks):
                v_chunk = V[start_idx:start_idx + chunks]
                a_chunk = A[start_idx:start_idx + chunks]
                lg = model(v_chunk, a_chunk)
                if isinstance(lg, tuple): lg = lg[0]
                logits_list.append(lg)

            logits_full = torch.cat(logits_list, dim=0)  # [L, 8]
            labels_full = torch.from_numpy(y_all).to(device)
            video_records.append((logits_full, labels_full, valid_mask))

    print(f"\n[INFO] Logits extracted for {len(video_records)} videos. Ready for lightning-fast smoothing!")


    windows = list(range(3, MAX_WINDOW + 1, STEP))
    strategies = ["mean", "median", "gaussian", "hard_vote"]
    cm_dict = {w: {s: np.zeros((8, 8), dtype=np.int64) for s in strategies} for w in windows}
    cm_center = np.zeros((8, 8), dtype=np.int64)

    for logits, labels, mask in video_records:
        if mask.sum() == 0: continue
        cm_center += confusion_matrix_np(labels[mask].cpu().numpy(), logits[mask].argmax(-1).cpu().numpy(), 8)
    mf1_center = macro_f1_from_cm(cm_center)
    print(f"[INFO] Baseline Center Frame MF1: {mf1_center:.4f}")

    for w in tqdm(windows, desc="Testing Window Sizes"):
        pad = w // 2

        gauss_x = torch.arange(-pad, pad + 1, dtype=torch.float32, device=device)
        gauss_w = torch.exp(-0.5 * (gauss_x / (w / 4.0)) ** 2)
        gauss_kernel = (gauss_w / gauss_w.sum()).view(1, 1, w).repeat(8, 1, 1)

        for logits, labels, mask in video_records:
            if mask.sum() == 0: continue
            y_true = labels[mask].cpu().numpy()

            L = logits.shape[0]
            L_tensor = logits.T.unsqueeze(0)

            padded = F.pad(L_tensor, (pad, pad), mode='replicate')  # [1, 8, L + 2*pad]

            # 1. Mean Smoothing
            out_mean = F.avg_pool1d(padded, kernel_size=w, stride=1).squeeze(0).T  # [L, 8]
            cm_dict[w]["mean"] += confusion_matrix_np(y_true, out_mean[mask].argmax(-1).cpu().numpy(), 8)

            # 2. Gaussian Weighting
            out_gauss = F.conv1d(padded, gauss_kernel, groups=8).squeeze(0).T  # [L, 8]
            cm_dict[w]["gaussian"] += confusion_matrix_np(y_true, out_gauss[mask].argmax(-1).cpu().numpy(), 8)

            # 3. Median Filter
            # unfold: [1, 8, L, w] -> median在最后一个维度求
            unfolded = padded.unfold(2, w, 1)
            out_median = unfolded.median(dim=-1)[0].squeeze(0).T  # [L, 8]
            cm_dict[w]["median"] += confusion_matrix_np(y_true, out_median[mask].argmax(-1).cpu().numpy(), 8)

            # 4. Hard Voting
            preds = logits.argmax(-1).float().unsqueeze(0).unsqueeze(0)  # [1, 1, L]
            padded_preds = F.pad(preds, (pad, pad), mode='replicate')
            unfolded_preds = padded_preds.unfold(2, w, 1).squeeze(0).squeeze(0).long()  # [L, w]
            out_hard = torch.mode(unfolded_preds, dim=-1)[0]  # [L]
            cm_dict[w]["hard_vote"] += confusion_matrix_np(y_true, out_hard[mask].cpu().numpy(), 8)

    # ==========================================
    results_mf1 = {s: [] for s in strategies}
    for w in windows:
        for s in strategies:
            results_mf1[s].append(macro_f1_from_cm(cm_dict[w][s]))

    plt.figure(figsize=(12, 7), dpi=300)
    plt.axhline(y=mf1_center, color='black', linestyle='--', linewidth=2,
                label=f'Center Frame Baseline ({mf1_center:.4f})')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    labels_map = ['Mean Smoothing', 'Median Filter', 'Gaussian Weighting', 'Hard Voting']

    best_overall_mf1 = -1
    best_config = ""

    for i, s in enumerate(strategies):
        mf1s = results_mf1[s]
        best_w_idx = np.argmax(mf1s)
        best_mf1 = mf1s[best_w_idx]
        best_w = windows[best_w_idx]

        if best_mf1 > best_overall_mf1:
            best_overall_mf1 = best_mf1
            best_config = f"{labels_map[i]} @ {best_w} Frames"

        plt.plot(windows, mf1s, marker=markers[i], markersize=4, color=colors[i], linewidth=2,
                 label=f'{labels_map[i]} (Best: {best_mf1:.4f} @ {best_w})')

    plt.xlabel("Temporal Window Size (Frames)", fontsize=12, fontweight='bold')
    plt.ylabel("Macro F1 Score", fontsize=12, fontweight='bold')
    # plt.title("Ablation:Temporal Smoothing", fontsize=15, fontweight='bold')
    plt.xticks(np.arange(min(windows) - 3, max(windows) + 1, 16))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()

    plot_path = os.path.join(PATH.RUN_ROOT, "ultimate_temporal_ablation.png")
    plt.savefig(plot_path)

    print("\n" + "=" * 60)
    print(f"[WINNER] The Absolute Best Config: {best_config}")
    print(f"[SCORE]  Maximum MF1 Score     : {best_overall_mf1:.4f}")
    print(f"[PLOT]   Saved to: {plot_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()