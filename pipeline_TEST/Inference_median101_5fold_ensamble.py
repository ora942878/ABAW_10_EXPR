
import os
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from configs.paths import PATH
from pipeline4_model_trains.common.set_seed import set_seed
from pipeline4_model_trains.A_V_concat_eval.build import (
    build_model,
    load_cfg,
    apply_runtime_defaults,
)

# ============================ config =============================
MODEL_NAME = "gate"
FIXED_WINDOW = 101
BATCH_SIZE = 4096
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CV5_RUN_DIR = (
    PATH.PROJECT_ROOT
    / "pipeline4_model_trains"
    / "A_V_concat_eval"
    / "runs"
    / "av_concat_gate_cv5"
)


VIS_ROOTS_TEST = [
    PATH.FACE09_V_DINOV2_FT2_TEST,
    PATH.FACE12_V_DINOV2_FT2_TEST,
    PATH.FACE15_V_DINOV2_FT2_TEST,
]
AUD_ROOT_TEST = PATH.OFFAUDIO_WAV2VEC2_PT_aligned_window50_TEST


TEST_VIDEO_INDEX_CSV = PATH.EXPR_VIDEO_INDEX_CSV_TEST
TEST_TEMPLATE_TXT = PATH.frame_ids_txt


OUTPUT_TXT = PATH.METADATA_ROOT_TEST / "predictions_median101_cv5.txt"


# ============================ tools =============================
def read_txtid_to_videoid_map(csv_path):
    mp = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txtid = str(row.get("txtid", "")).strip()
            videoid = str(row.get("videoid", "")).strip()
            if txtid and videoid:
                mp[txtid] = videoid
    return mp


def load_feat_dict_pt(pt_path):
    pt_path = str(pt_path)
    if not os.path.exists(pt_path):
        return None

    try:
        obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[WARN] torch.load failed: {pt_path} | {e}")
        return None

    frames = None
    feats = None

    if isinstance(obj, dict):
        if "frames" in obj and "feats" in obj:
            frames = obj["frames"]
            feats = obj["feats"]
        elif "frame_ids" in obj and "features" in obj:
            frames = obj["frame_ids"]
            feats = obj["features"]
        else:
            try:
                if all(str(k).isdigit() for k in obj.keys()):
                    out = {}
                    for k, v in obj.items():
                        out[int(k)] = (
                            v.detach().cpu().float().numpy()
                            if torch.is_tensor(v) else np.asarray(v, dtype=np.float32)
                        )
                    return out
            except Exception:
                pass

    if frames is None or feats is None:
        print(f"[WARN] unknown pt format: {pt_path}")
        return None

    if torch.is_tensor(feats):
        feats = feats.detach().cpu().float().numpy()
    else:
        feats = np.asarray(feats, dtype=np.float32)

    out = {}
    for i, fr in enumerate(frames):
        try:
            out[int(fr)] = feats[i].astype(np.float32, copy=False)
        except Exception:
            continue
    return out


def parse_submission_template(template_txt):
    with open(template_txt, "r", encoding="utf-8-sig") as f:
        lines = [x.rstrip("\n\r") for x in f]

    if len(lines) == 0:
        raise RuntimeError(f"empty template file: {template_txt}")

    header = lines[0]
    rows = []

    for line in lines[1:]:
        s = line.strip()
        if not s:
            continue

        image_location = s.split(",")[0].strip()
        if "/" not in image_location:
            raise ValueError(f"bad image_location line: {line}")

        txtid, frame_name = image_location.split("/", 1)
        frame_stem = Path(frame_name).stem
        frame_id = int(frame_stem)

        rows.append({
            "image_location": image_location,
            "txtid": txtid,
            "frame": frame_id,
        })

    return header, rows


def group_rows_by_txtid(rows):
    mp = defaultdict(list)
    for i, row in enumerate(rows):
        mp[row["txtid"]].append((i, row))
    return mp


def find_nearest_key(sorted_keys, target):
    if len(sorted_keys) == 0:
        return None

    import bisect
    pos = bisect.bisect_left(sorted_keys, target)

    cand = []
    if pos < len(sorted_keys):
        cand.append(sorted_keys[pos])
    if pos - 1 >= 0:
        cand.append(sorted_keys[pos - 1])

    if not cand:
        return None
    return min(cand, key=lambda x: abs(x - target))


def aggregate_visual_feature_at_frame(vis_maps, frame_id):
    feats = []
    for vm in vis_maps:
        if vm is not None and frame_id in vm:
            feats.append(vm[frame_id])

    if len(feats) == 0:
        return None, False

    if len(feats) == 1:
        return feats[0].astype(np.float32, copy=False), True

    return np.mean(np.stack(feats, axis=0), axis=0).astype(np.float32), True


def aggregate_visual_feature_with_fallback(vis_maps, frame_id):
    exact_feat, exact_exists = aggregate_visual_feature_at_frame(vis_maps, frame_id)
    if exact_feat is not None:
        return exact_feat, exact_exists

    feats = []
    for vm in vis_maps:
        if vm is None or len(vm) == 0:
            continue
        keys = sorted(vm.keys())
        nn = find_nearest_key(keys, frame_id)
        if nn is not None:
            feats.append(vm[nn])

    if len(feats) == 0:
        return None, False

    if len(feats) == 1:
        return feats[0].astype(np.float32, copy=False), False

    return np.mean(np.stack(feats, axis=0), axis=0).astype(np.float32), False


def get_audio_feature_with_fallback(aud_map, frame_id):
    if aud_map is None or len(aud_map) == 0:
        return None, False

    if frame_id in aud_map:
        return aud_map[frame_id].astype(np.float32, copy=False), True

    keys = sorted(aud_map.keys())
    nn = find_nearest_key(keys, frame_id)
    if nn is None:
        return None, False

    return aud_map[nn].astype(np.float32, copy=False), False


def load_test_video_features(txtid, videoid):
    vis_maps = []
    for root in VIS_ROOTS_TEST:
        vis_maps.append(load_feat_dict_pt(root / f"{txtid}.pt"))
    vis_maps = [x for x in vis_maps if x is not None]

    aud_map = load_feat_dict_pt(AUD_ROOT_TEST / f"{videoid}.pt")
    return vis_maps, aud_map


def build_dense_features_for_rows(rows_this_txtid, vis_maps, aud_map, vis_dim, aud_dim):
    n = len(rows_this_txtid)
    V = np.zeros((n, vis_dim), dtype=np.float32)
    A = np.zeros((n, aud_dim), dtype=np.float32)
    raw_valid_mask = np.zeros(n, dtype=bool)
    usable_mask = np.zeros(n, dtype=bool)

    for j, (_, row) in enumerate(rows_this_txtid):
        fr = row["frame"]

        v_feat, exact_vis = aggregate_visual_feature_with_fallback(vis_maps, fr)
        a_feat, exact_aud = get_audio_feature_with_fallback(aud_map, fr)

        if v_feat is not None and a_feat is not None:
            V[j] = v_feat
            A[j] = a_feat
            usable_mask[j] = True
            raw_valid_mask[j] = bool(exact_vis and exact_aud)

    return V, A, raw_valid_mask, usable_mask


def normalize_features_np(V, A, v_mean, v_std, a_mean, a_std, do_l2):
    Vt = torch.from_numpy(V).to(DEVICE)
    At = torch.from_numpy(A).to(DEVICE)

    if do_l2:
        Vt = F.normalize(Vt, p=2, dim=-1)
        At = F.normalize(At, p=2, dim=-1)

    Vt = (Vt - v_mean) / v_std
    At = (At - a_mean) / a_std
    return Vt, At


def forward_in_batches(model, Vt, At, batch_size=BATCH_SIZE):
    outs = []
    model.eval()
    with torch.no_grad():
        n = Vt.shape[0]
        for st in range(0, n, batch_size):
            ed = min(st + batch_size, n)
            lg = model(Vt[st:ed], At[st:ed])
            if isinstance(lg, tuple):
                lg = lg[0]
            outs.append(lg.detach())
    return torch.cat(outs, dim=0)


def median_smooth_logits_excluding_imputed(logits, raw_valid_mask, window=101):
    if torch.is_tensor(logits):
        x = logits.detach().cpu().numpy()
    else:
        x = np.asarray(logits, dtype=np.float32)

    n, c = x.shape
    out = np.empty_like(x)
    pad = window // 2

    global_valid_idx = np.flatnonzero(raw_valid_mask)

    for i in range(n):
        l = max(0, i - pad)
        r = min(n, i + pad + 1)

        local_valid = raw_valid_mask[l:r]
        if np.any(local_valid):
            vals = x[l:r][local_valid]
            out[i] = np.median(vals, axis=0)
        else:
            if len(global_valid_idx) > 0:
                nn = global_valid_idx[np.argmin(np.abs(global_valid_idx - i))]
                out[i] = x[nn]
            else:
                out[i] = x[i]

    return out


# ========================= load models ================================
def load_cv5_models_and_stats(cfg):
    bundle = []
    for fold_id in range(1, 6):
        fold_dir = CV5_RUN_DIR / f"fold_{fold_id}"
        if not fold_dir.exists():
            print(f"[WARN] fold dir not found: {fold_dir}")
            continue

        model = build_model(cfg).to(DEVICE)
        ckpt = torch.load(fold_dir / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)

        v_mean = torch.from_numpy(np.load(fold_dir / "train_vis_mean.npy")).float().to(DEVICE)
        v_std  = torch.from_numpy(np.load(fold_dir / "train_vis_std.npy")).float().to(DEVICE)
        a_mean = torch.from_numpy(np.load(fold_dir / "train_aud_mean.npy")).float().to(DEVICE)
        a_std  = torch.from_numpy(np.load(fold_dir / "train_aud_std.npy")).float().to(DEVICE)

        v_std = torch.clamp(v_std, min=1e-6)
        a_std = torch.clamp(a_std, min=1e-6)

        bundle.append({
            "fold_id": fold_id,
            "model": model,
            "v_mean": v_mean,
            "v_std": v_std,
            "a_mean": a_mean,
            "a_std": a_std,
        })

    if len(bundle) == 0:
        raise RuntimeError(f"no fold checkpoints found under: {CV5_RUN_DIR}")

    print(f"[INFO] loaded {len(bundle)} fold models from {CV5_RUN_DIR}")
    return bundle


def predict_logits_for_video_cv5(cfg, rows_this_txtid, vis_maps, aud_map, model_bundle):
    V, A, raw_valid_mask, usable_mask = build_dense_features_for_rows(
        rows_this_txtid=rows_this_txtid,
        vis_maps=vis_maps,
        aud_map=aud_map,
        vis_dim=cfg.VIS_DIM,
        aud_dim=cfg.AUD_DIM,
    )

    if not np.any(usable_mask):
        return None, raw_valid_mask, usable_mask

    usable_idx = np.flatnonzero(usable_mask)
    for i in range(len(usable_mask)):
        if not usable_mask[i]:
            nn = usable_idx[np.argmin(np.abs(usable_idx - i))]
            V[i] = V[nn]
            A[i] = A[nn]

    logits_sum = None

    for item in model_bundle:
        Vt, At = normalize_features_np(
            V=V,
            A=A,
            v_mean=item["v_mean"],
            v_std=item["v_std"],
            a_mean=item["a_mean"],
            a_std=item["a_std"],
            do_l2=getattr(cfg, "DO_BRANCH_L2_NORM", False),
        )
        logits = forward_in_batches(item["model"], Vt, At, batch_size=BATCH_SIZE)

        if logits_sum is None:
            logits_sum = logits
        else:
            logits_sum = logits_sum + logits

    logits_mean = logits_sum / float(len(model_bundle))
    return logits_mean, raw_valid_mask, usable_mask


def save_predictions_txt(output_path, header, rows, preds):
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write(header + "\n")
        for row, pred in zip(rows, preds):
            f.write(f"{row['image_location']},{int(pred)}\n")



def main():
    set_seed(3407)
    cfg = apply_runtime_defaults(load_cfg(MODEL_NAME))

    print(f"[INFO] DEVICE = {DEVICE}")
    print(f"[INFO] TEST_TEMPLATE_TXT = {TEST_TEMPLATE_TXT}")
    print(f"[INFO] TEST_VIDEO_INDEX_CSV = {TEST_VIDEO_INDEX_CSV}")
    print(f"[INFO] OUTPUT_TXT = {OUTPUT_TXT}")
    print(f"[INFO] FIXED_WINDOW = {FIXED_WINDOW}")
    print(f"[INFO] CV5_RUN_DIR = {CV5_RUN_DIR}")

    txt2vid = read_txtid_to_videoid_map(TEST_VIDEO_INDEX_CSV)
    header, all_rows = parse_submission_template(TEST_TEMPLATE_TXT)
    rows_by_txtid = group_rows_by_txtid(all_rows)

    print(f"[INFO] total rows = {len(all_rows)}")
    print(f"[INFO] total txtids = {len(rows_by_txtid)}")

    model_bundle = load_cv5_models_and_stats(cfg)
    final_preds = np.zeros(len(all_rows), dtype=np.int64)

    total_raw_valid = 0
    total_usable = 0
    total_rows_count = 0
    missing_txtid_count = 0
    missing_feature_count = 0

    pbar = tqdm(rows_by_txtid.items(), desc="CV5 Test inference by txtid")
    for txtid, rows_this_txtid in pbar:
        total_rows_count += len(rows_this_txtid)

        if txtid not in txt2vid:
            missing_txtid_count += 1
            print(f"[WARN] txtid not found in csv: {txtid}")
            for global_idx, _ in rows_this_txtid:
                final_preds[global_idx] = 0
            continue

        videoid = txt2vid[txtid]
        vis_maps, aud_map = load_test_video_features(txtid, videoid)

        if len(vis_maps) == 0 or aud_map is None or len(aud_map) == 0:
            missing_feature_count += 1
            print(f"[WARN] missing feature file(s): txtid={txtid}, videoid={videoid}")
            for global_idx, _ in rows_this_txtid:
                final_preds[global_idx] = 0
            continue

        logits, raw_valid_mask, usable_mask = predict_logits_for_video_cv5(
            cfg=cfg,
            rows_this_txtid=rows_this_txtid,
            vis_maps=vis_maps,
            aud_map=aud_map,
            model_bundle=model_bundle,
        )

        if logits is None:
            print(f"[WARN] no usable logits: txtid={txtid}")
            for global_idx, _ in rows_this_txtid:
                final_preds[global_idx] = 0
            continue

        total_raw_valid += int(raw_valid_mask.sum())
        total_usable += int(usable_mask.sum())

        smooth_logits = median_smooth_logits_excluding_imputed(
            logits=logits,
            raw_valid_mask=raw_valid_mask,
            window=FIXED_WINDOW,
        )

        pred_cls = np.argmax(smooth_logits, axis=1).astype(np.int64)

        for local_i, (global_idx, _) in enumerate(rows_this_txtid):
            final_preds[global_idx] = pred_cls[local_i]

        pbar.set_postfix({
            "txtid": txtid,
            "rows": len(rows_this_txtid),
            "raw_valid": int(raw_valid_mask.sum()),
            "usable": int(usable_mask.sum()),
        })

    save_predictions_txt(
        output_path=OUTPUT_TXT,
        header=header,
        rows=all_rows,
        preds=final_preds,
    )

    print("\n" + "=" * 90)
    print("[DONE] prediction file written.")
    print(f"OUTPUT: {OUTPUT_TXT.resolve()}")
    print(f"TOTAL_ROWS            : {total_rows_count}")
    print(f"TOTAL_USABLE_ROWS     : {total_usable}")
    print(f"TOTAL_RAW_VALID_ROWS  : {total_raw_valid}")
    print(f"MISSING_TXID_COUNT    : {missing_txtid_count}")
    print(f"MISSING_FEATURE_COUNT : {missing_feature_count}")
    print("=" * 90)


if __name__ == "__main__":
    main()