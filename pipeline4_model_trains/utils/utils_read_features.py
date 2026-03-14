# pipeline4_model_trains/utils/utils_read_features.py
import os
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import torch

from configs.paths import PATH
from pipeline4_model_trains.utils.utils_id_mapper import IDMapper

MAPPER = IDMapper()


def load_feat_pt(pt_path: Union[str, Path]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    pt_path = Path(pt_path)
    if not pt_path.exists():
        return None

    try:
        obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[WARN] failed load {pt_path}: {e}")
        return None

    if not isinstance(obj, dict):
        return None
    if "frames" not in obj or "feats" not in obj:
        return None

    frames = obj["frames"]
    feats = obj["feats"]

    if not isinstance(frames, list):
        return None
    if not torch.is_tensor(feats) or feats.ndim != 2:
        return None

    frames_np = np.asarray(frames, dtype=np.int64)
    feats_np = feats.detach().cpu().float().numpy()

    if len(frames_np) != feats_np.shape[0]:
        return None

    return frames_np, feats_np


def intersect_frames(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    common = sorted(set(map(int, a.tolist())) & set(map(int, b.tolist())))
    return np.asarray(common, dtype=np.int64)


def gather_by_common_frames(
    frames_src: np.ndarray,
    values_src: np.ndarray,
    common_frames: np.ndarray,
) -> np.ndarray:
    pos = {int(fr): i for i, fr in enumerate(frames_src.tolist())}
    idx = np.asarray([pos[int(fr)] for fr in common_frames], dtype=np.int64)
    return values_src[idx]


def load_visual_mean_091215(txtid: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    p09 = Path(PATH.FACE09_V_DINOV2_FT2) / f"{txtid}.pt"
    p12 = Path(PATH.FACE12_V_DINOV2_FT2) / f"{txtid}.pt"
    p15 = Path(PATH.FACE15_V_DINOV2_FT2) / f"{txtid}.pt"

    pack09 = load_feat_pt(p09)
    pack12 = load_feat_pt(p12)
    pack15 = load_feat_pt(p15)
    if pack09 is None or pack12 is None or pack15 is None:
        return None

    f09, x09 = pack09
    f12, x12 = pack12
    f15, x15 = pack15

    common = intersect_frames(intersect_frames(f09, f12), f15)
    if len(common) == 0:
        return None

    x09c = gather_by_common_frames(f09, x09, common)
    x12c = gather_by_common_frames(f12, x12, common)
    x15c = gather_by_common_frames(f15, x15, common)

    if not (x09c.shape == x12c.shape == x15c.shape):
        return None

    x = (x09c + x12c + x15c) / 3.0
    return common, x.astype(np.float32)


def load_audio_aligned(videoid: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    p = Path(PATH.OFFAUDIO_WAV2VEC2_PT_aligned_window50) / f"{videoid}.pt"
    return load_feat_pt(p)


def get_av_features(txtid: str) -> Tuple[
    Optional[Tuple[np.ndarray, np.ndarray]],
    Optional[Tuple[np.ndarray, np.ndarray]]
]:
    """
    return:
        vis_pack = (vis_frames, vis_feats)  # 091215 mean
        aud_pack = (aud_frames, aud_feats)  # aligned audio
    """
    vis_pack = load_visual_mean_091215(txtid)

    videoid = MAPPER.get_videoid(txtid)
    if videoid is None or len(str(videoid).strip()) == 0:
        return vis_pack, None

    aud_pack = load_audio_aligned(str(videoid))
    return vis_pack, aud_pack
if __name__ == "__main__":
    from configs.paths import PATH
    v,a = get_av_features("10-60-1280x720_right")
    print(v)
    print(a)
