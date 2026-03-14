
from __future__ import annotations

import os
import csv
import gc
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio
from tqdm.auto import tqdm

from configs.paths import PATH


# =========================== env safety ===========================
if "OMP_NUM_THREADS" in os.environ:
    v = os.environ.get("OMP_NUM_THREADS", "")
    if (not v.isdigit()) or int(v) <= 0:
        os.environ.pop("OMP_NUM_THREADS", None)

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# =========================== fixed configs ===========================
STORE_FP16 = True
AUDIO_SR = 16000

MODEL_NAME = "wav2vec2_fairseq_large_lv60k_asr_ls960_last4mean_native_ts_test"
CKPT_NAME = "wav2vec2_fairseq_large_lv60k_asr_ls960.pth"
VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"]

TORCH_HOME = Path(PATH.PIPELINE_3_ROOT) / "torch_cache"
WAV_DUMP_ROOT = Path(PATH.OFFAUDIO_WAV2VEC2_PT2_TEST)
OUT_ROOT = Path(PATH.OFFAUDIO_WAV2VEC2_PT_TEST)

# chunking
CHUNK_SEC = 20.0
MIN_TAIL_SEC = 1.0


# =========================== io helpers ===========================
def checkpoint_is_valid(pt_path: Path, expect_dim: int = 1024) -> bool:
    if not pt_path.exists():
        return False
    try:
        obj = torch.load(pt_path, map_location="cpu")
        required = {"folderid", "frames", "feats", "model", "fp16"}
        if not required.issubset(obj.keys()):
            return False

        feats = obj["feats"]
        if (not torch.is_tensor(feats)) or feats.ndim != 2:
            return False
        if int(feats.shape[1]) != expect_dim:
            return False

        frames = obj["frames"]
        if (not isinstance(frames, list)) or len(frames) != int(feats.shape[0]):
            return False

        if "t_sec" in obj:
            tsec = obj["t_sec"]
            if (not torch.is_tensor(tsec)) or tsec.ndim != 1 or int(tsec.numel()) != int(feats.shape[0]):
                return False

        return True
    except Exception:
        return False


def ensure_w2v_ckpt(torch_home: Path) -> Path:
    ckpt = torch_home / "hub" / "checkpoints" / CKPT_NAME
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    return ckpt


def read_expr_test_video_ids(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing csv: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        cand_cols = ["videoid", "folderid", "video_id", "vid", "id", "name", "video", "txtid"]
        use_col = next((c for c in cand_cols if c in fieldnames), None)
        if use_col is None:
            use_col = fieldnames[0] if fieldnames else None

        assert use_col is not None, f"Cannot determine id column from csv header: {fieldnames}"

        ids = []
        for row in reader:
            v = (row.get(use_col) or "").strip()
            if v:
                ids.append(Path(v).stem)

    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def find_video_by_id(folderid: str) -> Optional[Path]:
    roots = []

    if hasattr(PATH, "VIDEO_TEST_ABAW10th"):
        roots.append(Path(PATH.VIDEO_TEST_ABAW10th))

    roots.extend([
        Path(PATH.VIDEO_batch1_ABAW10th),
        Path(PATH.VIDEO_batch2_ABAW10th),
        Path(PATH.VIDEO_batch3_ABAW10th),
    ])

    for r in roots:
        if not r.exists():
            continue
        for ext in VIDEO_EXTS:
            p = r / f"{folderid}{ext}"
            if p.is_file():
                return p

    for r in roots:
        if not r.exists():
            continue
        for ext in VIDEO_EXTS:
            for p in r.rglob(f"*{ext}"):
                if p.stem == folderid:
                    return p

    return None


def ffmpeg_extract_wav(video_path: Path, wav_path: Path) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", str(AUDIO_SR),
        "-f", "wav",
        str(wav_path),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {video_path}:\n{p.stderr}")


# =========================== core extraction ===========================
@torch.inference_mode()
def extract_last4mean_with_timestamps_chunked(
    model,
    waveform_cpu: torch.Tensor,   # [1, T] on CPU
    sr: int,
    device: torch.device,
    chunk_sec: float,
    min_tail_sec: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return:
      feats_cpu: float32 [T_total, 1024]
      t_sec_cpu: float32 [T_total]
    """
    assert waveform_cpu.device.type == "cpu"
    assert waveform_cpu.ndim == 2 and waveform_cpu.size(0) == 1

    total = int(waveform_cpu.size(1))
    chunk = max(int(round(chunk_sec * sr)), sr)  # >= 1 sec
    min_tail = int(round(min_tail_sec * sr))

    starts = list(range(0, total, chunk))
    if len(starts) >= 2:
        last_len = total - starts[-1]
        if last_len < min_tail:
            starts.pop(-1)

    feats_all: List[torch.Tensor] = []
    t_all: List[torch.Tensor] = []

    use_amp = (device.type == "cuda")

    for i, s in enumerate(starts):
        e = total if (i == len(starts) - 1) else min(s + chunk, total)
        seg = waveform_cpu[:, s:e].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            feats_list, _ = model.extract_features(seg, num_layers=4)
            layers = [x[0] for x in feats_list[-4:]]   # each [t', 1024]
            feat_mean = torch.stack(layers, dim=0).mean(dim=0)

        feat_mean_cpu = feat_mean.detach().float().cpu()
        feats_all.append(feat_mean_cpu)

        t0 = float(s) / float(sr)
        seg_dur = float(e - s) / float(sr)
        Tn = int(feat_mean_cpu.size(0))
        if Tn <= 0:
            tsec = torch.empty((0,), dtype=torch.float32)
        else:
            j = torch.arange(Tn, dtype=torch.float32)
            tsec = t0 + (j + 0.5) * (seg_dur / float(Tn))
        t_all.append(tsec)

        del seg, feats_list, feat_mean, layers, feat_mean_cpu, tsec
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    feats_cat = torch.cat(feats_all, dim=0) if len(feats_all) else torch.empty((0, 1024), dtype=torch.float32)
    t_cat = torch.cat(t_all, dim=0) if len(t_all) else torch.empty((0,), dtype=torch.float32)
    return feats_cat, t_cat


def load_wav2vec2_model_from_local_ckpt(device):
    from pathlib import Path
    import torch
    from torchaudio.models import wav2vec2_model

    ckpt_path = Path(PATH.PIPELINE_3_ROOT) / "torch_cache" / "hub" / "checkpoints" / CKPT_NAME
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    try:
        state = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"[BAD CKPT] cannot load checkpoint:\n{ckpt_path}\n\n{e}")

    model = wav2vec2_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=[
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        extractor_conv_bias=True,
        encoder_embed_dim=1024,
        encoder_projection_dropout=0.1,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_attention_dropout=0.1,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=0.0,
        encoder_dropout=0.0,
        encoder_layer_norm_first=True,
        encoder_layer_drop=0.0,
        aux_num_out=32,
    )

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[INFO] missing keys:", len(missing))
    print("[INFO] unexpected keys:", len(unexpected))
    if len(missing) > 0:
        print("[INFO] first missing keys:", missing[:10])
    if len(unexpected) > 0:
        print("[INFO] first unexpected keys:", unexpected[:10])

    model = model.to(device).eval()
    return model
# =========================== main ===========================
@torch.inference_mode()
def main():
    print("[BOOT] step5_get_features_audio_raw_last4mean_ts_test.py starting.", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device, flush=True)

    ckpt = ensure_w2v_ckpt(TORCH_HOME)
    print("[INFO] TORCH_HOME:", TORCH_HOME, flush=True)
    print("[INFO] wav2vec2 ckpt (expected):", ckpt, flush=True)
    print("[INFO] CHUNK_SEC:", CHUNK_SEC, "MIN_TAIL_SEC:", MIN_TAIL_SEC, flush=True)

    model = load_wav2vec2_model_from_local_ckpt(device)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    WAV_DUMP_ROOT.mkdir(parents=True, exist_ok=True)

    ids = read_expr_test_video_ids(Path(PATH.EXPR_VIDEO_INDEX_CSV_TEST))
    print(f"[INFO] Loaded TEST video ids: {len(ids)} from {PATH.EXPR_VIDEO_INDEX_CSV_TEST}", flush=True)

    miss = 0
    done = 0

    for folderid in tqdm(ids, desc="Total [TEST AUDIO wav2vec2 last4mean nativeTS]"):
        out_pt = OUT_ROOT / f"{folderid}.pt"
        if checkpoint_is_valid(out_pt, expect_dim=1024):
            continue

        vpath = find_video_by_id(folderid)
        if vpath is None:
            miss += 1
            print(f"[WARN] video not found: {folderid}", flush=True)
            continue

        wav_path = WAV_DUMP_ROOT / f"{folderid}.wav"
        if (not wav_path.exists()) or wav_path.stat().st_size < 1024:
            try:
                ffmpeg_extract_wav(vpath, wav_path)
            except Exception as e:
                print(f"[WARN] ffmpeg failed: {folderid} -> {e}", flush=True)
                continue

        waveform, sr = torchaudio.load(str(wav_path))
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != AUDIO_SR:
            waveform = torchaudio.transforms.Resample(sr, AUDIO_SR)(waveform)
            sr = AUDIO_SR

        duration_sec = float(waveform.size(1)) / float(sr)

        try:
            feats_cpu, t_sec_cpu = extract_last4mean_with_timestamps_chunked(
                model=model,
                waveform_cpu=waveform.cpu(),
                sr=sr,
                device=device,
                chunk_sec=CHUNK_SEC,
                min_tail_sec=MIN_TAIL_SEC,
            )
        except torch.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            print(f"[WARN] OOM on {folderid}. Retrying with smaller chunk.", flush=True)
            feats_cpu, t_sec_cpu = extract_last4mean_with_timestamps_chunked(
                model=model,
                waveform_cpu=waveform.cpu(),
                sr=sr,
                device=device,
                chunk_sec=max(5.0, CHUNK_SEC / 2.0),
                min_tail_sec=MIN_TAIL_SEC,
            )

        feats_save = feats_cpu.half() if STORE_FP16 else feats_cpu
        frames = list(range(1, int(feats_save.size(0)) + 1))

        torch.save(
            {
                "folderid": folderid,
                "frames": frames,
                "feats": feats_save,
                "model": MODEL_NAME,
                "fp16": bool(STORE_FP16),

                "t_sec": t_sec_cpu,
                "audio_sr": int(sr),
                "duration_sec": float(duration_sec),
                "video_path": str(vpath),
                "wav_path": str(wav_path),
                "chunk_sec": float(CHUNK_SEC),
                "min_tail_sec": float(MIN_TAIL_SEC),
            },
            out_pt,
        )
        done += 1

        del waveform, feats_cpu, t_sec_cpu, feats_save
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"[DONE] saved pts: {done}", flush=True)
    print(f"[DONE] miss videos: {miss}", flush=True)
    print(f"[OUT] pt root : {OUT_ROOT}", flush=True)
    print(f"[OUT] wav root: {WAV_DUMP_ROOT}", flush=True)


if __name__ == "__main__":
    main()