"""
Input:
  - video meta csv: PATH.METADATA_ROOT / "expr_video_meta_fps_frames_audioDur.csv"
      columns: videoid, fps, n_frames, video_duration_sec, audio_duration_sec
  - raw audio feats: PATH.OFFAUDIO_WAV2VEC2_PT / "<videoid>.pt"
      expected keys: feats [T_a, 1024], t_sec [T_a], duration_sec ...

Output:
  - aligned audio feats: OUT_ROOT / "<videoid>.pt"
      keys: folderid, frames(1..n_frames), feats [n_frames, 1024], fps, align_method, win_sec, ...

Notes:
  - Align method default: window_mean with prefix-sum (fast), fallback to nearest when window empty.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from configs.paths import PATH

# =========================== configs ===========================
STORE_FP16 = True

# alignment method: "window_mean" (recommended) or "nearest"
ALIGN_METHOD = "window_mean"
WIN_SEC = 0.50               # window half-size for mean, seconds (only for window_mean)
FALLBACK_EMPTY = "nearest"   # when window has no audio steps: "nearest" or "zeros"

# raw audio root (your extracted wav2vec2 last4mean native ts)
RAW_AUDIO_ROOT = Path(PATH.OFFAUDIO_WAV2VEC2_PT)

# output aligned audio root (create a subdir, avoid overwriting raw)
OUT_ROOT = Path(PATH.OFFAUDIO_WAV2VEC2_PT_aligned)

# meta csv
META_CSV = Path(PATH.METADATA_ROOT) / "expr_video_meta.csv"

# ==============================================================
def out_checkpoint_is_valid(pt_path: Path, expect_dim: int = 1024) -> bool:
    if not pt_path.exists():
        return False
    try:
        obj = torch.load(pt_path, map_location="cpu")
        if not isinstance(obj, dict):
            return False
        required = {"folderid", "frames", "feats", "fps", "align_method"}
        if not required.issubset(obj.keys()):
            return False
        feats = obj["feats"]
        frames = obj["frames"]
        if (not torch.is_tensor(feats)) or feats.ndim != 2 or feats.shape[1] != expect_dim:
            return False
        if (not isinstance(frames, list)) or len(frames) != int(feats.shape[0]):
            return False
        if float(obj["fps"]) <= 0:
            return False
        return True
    except Exception:
        return False


def load_raw_audio_pt(videoid: str) -> Optional[Dict]:
    p = RAW_AUDIO_ROOT / f"{videoid}.pt"
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu")


def _nearest_indices_sorted(t_sec: np.ndarray, t_v: np.ndarray) -> np.ndarray:
    """
    t_sec: [T_a] sorted ascending
    t_v:   [T_v]
    returns idx: [T_v] nearest index in t_sec
    """
    idx = np.searchsorted(t_sec, t_v, side="left")
    idx = np.clip(idx, 0, len(t_sec) - 1)
    idx_left = np.clip(idx - 1, 0, len(t_sec) - 1)
    choose_left = np.abs(t_sec[idx_left] - t_v) <= np.abs(t_sec[idx] - t_v)
    return np.where(choose_left, idx_left, idx).astype(np.int64)


def align_audio_to_frames_window_mean(
    feats: np.ndarray,   # [T_a, D]
    t_sec: np.ndarray,   # [T_a] sorted
    fps: float,
    n_frames: int,
    win_sec: float,
    fallback: str = "nearest",
) -> np.ndarray:
    """
    returns aligned: [n_frames, D]
    Uses prefix-sum to compute mean in [tv-win, tv+win] efficiently.
    """
    D = feats.shape[1]
    # frame center times, 0-based frame index i -> (i+0.5)/fps
    i = np.arange(n_frames, dtype=np.float32)
    t_v = (i + 0.5) / float(fps)  # [n_frames]

    # window bounds
    l_t = t_v - float(win_sec)
    r_t = t_v + float(win_sec)

    # indices
    l = np.searchsorted(t_sec, l_t, side="left")
    r = np.searchsorted(t_sec, r_t, side="right")
    cnt = (r - l).astype(np.int32)  # [n_frames]

    # prefix sum (Ta+1, D)
    # use float32 prefix for stability
    pref = np.zeros((feats.shape[0] + 1, D), dtype=np.float32)
    np.cumsum(feats.astype(np.float32), axis=0, out=pref[1:])

    sum_lr = pref[r] - pref[l]  # [n_frames, D]
    aligned = np.zeros((n_frames, D), dtype=np.float32)

    non_empty = cnt > 0
    if np.any(non_empty):
        aligned[non_empty] = sum_lr[non_empty] / cnt[non_empty].reshape(-1, 1).astype(np.float32)

    empty = ~non_empty
    if np.any(empty):
        if fallback == "zeros":
            pass
        else:
            nn_idx = _nearest_indices_sorted(t_sec, t_v[empty])
            aligned[empty] = feats[nn_idx].astype(np.float32)

    return aligned


def align_audio_to_frames_nearest(
    feats: np.ndarray,
    t_sec: np.ndarray,
    fps: float,
    n_frames: int,
) -> np.ndarray:
    D = feats.shape[1]
    i = np.arange(n_frames, dtype=np.float32)
    t_v = (i + 0.5) / float(fps)
    nn_idx = _nearest_indices_sorted(t_sec, t_v)
    return feats[nn_idx].astype(np.float32).reshape(n_frames, D)


def main():
    assert META_CSV.exists(), f"Missing meta csv: {META_CSV}"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(META_CSV)
    need_cols = {"videoid", "fps", "n_frames"}
    assert need_cols.issubset(df.columns), f"Meta csv must contain {need_cols}, got {list(df.columns)}"

    # unique by videoid
    df = df.drop_duplicates("videoid").reset_index(drop=True)

    skips = []
    saved = 0

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Align audio -> video frames"):
        vid = str(getattr(row, "videoid"))
        fps = float(getattr(row, "fps"))
        n_frames = int(getattr(row, "n_frames"))

        out_pt = OUT_ROOT / f"{vid}.pt"
        if out_checkpoint_is_valid(out_pt, expect_dim=1024):
            continue

        if fps <= 0 or n_frames <= 0:
            skips.append({"videoid": vid, "reason": "bad_fps_or_nframes"})
            continue

        raw = load_raw_audio_pt(vid)
        if raw is None:
            skips.append({"videoid": vid, "reason": "raw_audio_pt_missing"})
            continue

        if not isinstance(raw, dict) or ("feats" not in raw) or ("t_sec" not in raw):
            skips.append({"videoid": vid, "reason": "raw_audio_pt_bad_format"})
            continue

        feats_t = raw["feats"]
        tsec_t = raw["t_sec"]

        if not torch.is_tensor(feats_t) or feats_t.ndim != 2:
            skips.append({"videoid": vid, "reason": "raw_feats_invalid"})
            continue
        if not torch.is_tensor(tsec_t) or tsec_t.ndim != 1:
            skips.append({"videoid": vid, "reason": "raw_tsec_invalid"})
            continue
        if int(tsec_t.numel()) != int(feats_t.shape[0]):
            skips.append({"videoid": vid, "reason": "raw_len_mismatch"})
            continue

        feats = feats_t.detach().cpu().float().numpy()  # [T_a, D]
        t_sec = tsec_t.detach().cpu().float().numpy()   # [T_a]

        if feats.shape[1] != 1024:
            skips.append({"videoid": vid, "reason": f"raw_dim_not_1024({feats.shape[1]})"})
            continue

        # ensure sorted by time
        if np.any(np.diff(t_sec) < 0):
            order = np.argsort(t_sec)
            t_sec = t_sec[order]
            feats = feats[order]

        try:
            if ALIGN_METHOD == "nearest":
                aligned = align_audio_to_frames_nearest(feats, t_sec, fps=fps, n_frames=n_frames)
            else:
                aligned = align_audio_to_frames_window_mean(
                    feats, t_sec, fps=fps, n_frames=n_frames, win_sec=WIN_SEC, fallback=FALLBACK_EMPTY
                )
        except Exception as e:
            skips.append({"videoid": vid, "reason": "align_failed", "detail": str(e)[:300]})
            continue

        # save
        aligned_t = torch.from_numpy(aligned)
        if STORE_FP16:
            aligned_t = aligned_t.half()

        obj = {
            "folderid": vid,
            "frames": list(range(1, n_frames + 1)),  # 1-based, matches your label/txt style
            "feats": aligned_t,                      # [n_frames, 1024]
            "fp16": bool(STORE_FP16),

            # alignment meta (no absolute paths)
            "fps": float(fps),
            "align_method": str(ALIGN_METHOD),
            "win_sec": float(WIN_SEC) if ALIGN_METHOD != "nearest" else 0.0,
            "fallback_empty": str(FALLBACK_EMPTY) if ALIGN_METHOD != "nearest" else "",
            "raw_model": str(raw.get("model", "")),
            "raw_audio_sr": int(raw.get("audio_sr", 0)) if isinstance(raw.get("audio_sr", 0), (int, float)) else 0,
            "raw_duration_sec": float(raw.get("duration_sec", 0.0)) if isinstance(raw.get("duration_sec", 0.0), (int, float)) else 0.0,
        }

        torch.save(obj, out_pt)
        saved += 1

        # free
        del raw, feats_t, tsec_t, feats, t_sec, aligned, aligned_t
        gc.collect()

    # skip report
    if skips:
        out_skip = Path(PATH.METADATA_ROOT) / "expr_audio_align_skips.csv"
        pd.DataFrame(skips).to_csv(out_skip, index=False, encoding="utf-8-sig")
        print("[OK] saved skip report:", out_skip)

    print(f"[DONE] saved aligned pts: {saved} / {len(df)}")
    print("[OUT_ROOT]", OUT_ROOT)


if __name__ == "__main__":
    main()