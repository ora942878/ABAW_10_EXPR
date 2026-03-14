
import os
import shutil
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from typing import List, Optional

from configs.paths import PATH

INSIGHTFACE_HOME = Path(PATH.INSIGHTFACE_ROOT)  # e.g. .../pipeline2_face_extract/insightface
MODELS_DIR = INSIGHTFACE_HOME / "models"
BUFFALO_DIR = MODELS_DIR / "buffalo_l"
ZIP_PATH = MODELS_DIR / "buffalo_l.zip"

# Download sources (GitHub first, mirror fallback)
CANDIDATE_URLS = [
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download",
]

# Minimal sanity checks
MIN_ZIP_SIZE_BYTES = 10_000_000  # 10MB


def _human(n: int) -> str:
    v = float(n)
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if v < 1024.0:
            if u == "B":
                return f"{int(v)}{u}"
            return f"{v:.1f}{u}"
        v /= 1024.0
    return f"{v:.1f}PB"


def _download_file(url: str, dst: Path, timeout: int = 120) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] downloading: {url}")
    print(f"[INFO] to: {dst}")

    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as r, open(dst, "wb") as f:
        total = r.headers.get("Content-Length")
        total_int: Optional[int] = int(total) if total is not None else None

        done = 0
        chunk = 1024 * 1024  # 1MB
        while True:
            b = r.read(chunk)
            if not b:
                break
            f.write(b)
            done += len(b)
            if total_int:
                pct = done * 100.0 / total_int
                print(f"\r[DL] {pct:6.2f}%  {_human(done)}/{_human(total_int)}", end="")
            else:
                print(f"\r[DL] {_human(done)}", end="")
        print("")


def _safe_rmtree(p: Path) -> None:
    try:
        shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass


def _safe_unlink(p: Path) -> None:
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


def _list_onnx(p: Path) -> List[Path]:
    return [x for x in p.rglob("*.onnx") if x.is_file()]


def _extract_zip(zip_path: Path, dst_dir: Path) -> None:
    print(f"[INFO] extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            # Guard against zip slip
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"Unsafe zip member path: {member.filename}")
        z.extractall(dst_dir)


def _normalize_to_buffalo_dir(models_dir: Path, buffalo_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    buffalo_dir.mkdir(parents=True, exist_ok=True)

    # Case A: already good
    if any(buffalo_dir.glob("*.onnx")) or any(buffalo_dir.rglob("*.onnx")):
        return buffalo_dir

    flat_onnx = [p for p in models_dir.glob("*.onnx") if p.is_file()]
    if flat_onnx:
        print(f"[INFO] detected flat extraction: {len(flat_onnx)} onnx directly under {models_dir}")
        for f in flat_onnx:
            target = buffalo_dir / f.name
            if target.exists():
                _safe_unlink(f)  # keep existing
            else:
                f.replace(target)
        return buffalo_dir

    all_onnx = _list_onnx(models_dir)
    if not all_onnx:
        raise RuntimeError(f"No .onnx found under: {models_dir}")

    buffalo_candidates = []
    for f in all_onnx:
        for parent in f.parents:
            if parent.name.lower() == "buffalo_l":
                buffalo_candidates.append(parent)
                break

    src_dir: Optional[Path] = None
    if buffalo_candidates:
        src_dir = min(set(buffalo_candidates), key=lambda p: len(p.parts))
    else:
        counts = {}
        for f in all_onnx:
            d = f.parent
            counts[d] = counts.get(d, 0) + 1
        src_dir = max(counts.items(), key=lambda kv: kv[1])[0]

    if src_dir is None or not src_dir.exists():
        raise RuntimeError("Failed to locate a source directory for ONNX files.")

    print(f"[INFO] detected model dir: {src_dir}")

    moved = 0
    for f in _list_onnx(src_dir):
        target = buffalo_dir / f.name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        moved += 1

    if moved == 0 and not any(buffalo_dir.glob("*.onnx")):
        raise RuntimeError(
            f"Detected source dir {src_dir} but did not populate {buffalo_dir} with onnx."
        )

    return buffalo_dir


def ensure_buffalo_l() -> Path:
    # 0) DIR
    if BUFFALO_DIR.exists() and any(BUFFALO_DIR.rglob("*.onnx")):
        print(f"[OK] buffalo_l already exists: {BUFFALO_DIR}")
        return BUFFALO_DIR

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Download zip if missing
    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size < MIN_ZIP_SIZE_BYTES:
        last_err = None
        for url in CANDIDATE_URLS:
            try:
                _safe_unlink(ZIP_PATH)
                _download_file(url, ZIP_PATH)
                if ZIP_PATH.exists() and ZIP_PATH.stat().st_size >= MIN_ZIP_SIZE_BYTES:
                    last_err = None
                    break
            except (HTTPError, URLError, TimeoutError, OSError) as e:
                last_err = e
                print(f"[WARN] download failed: {e}")
                continue
        if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size < MIN_ZIP_SIZE_BYTES:
            raise RuntimeError(
                f"Failed to download buffalo_l.zip to {ZIP_PATH}. Last error: {last_err}"
            )
    else:
        print(f"[OK] zip already exists: {ZIP_PATH} ({_human(ZIP_PATH.stat().st_size)})")

    # 2) Extract to MODELS_DIR
    # Optionally clean stale buffalo_l dir first
    if BUFFALO_DIR.exists():
        print(f"[INFO] removing old buffalo_l dir: {BUFFALO_DIR}")
        _safe_rmtree(BUFFALO_DIR)

    _extract_zip(ZIP_PATH, MODELS_DIR)

    # 3) Normalize structure to models/buffalo_l/*.onnx
    out_dir = _normalize_to_buffalo_dir(MODELS_DIR, BUFFALO_DIR)

    # 4) Verify
    onnx_files = _list_onnx(out_dir)
    if not onnx_files:
        tops = sorted([p.name for p in MODELS_DIR.iterdir()])
        raise RuntimeError(
            f"Extraction finished but ONNX files not found under: {out_dir}\n"
            f"models/ contains: {tops}"
        )

    print(f"[OK] buffalo_l ready: {out_dir}  (onnx={len(onnx_files)})")

    return out_dir


def main():
    os.environ["INSIGHTFACE_HOME"] = str(INSIGHTFACE_HOME)
    print("[INFO] INSIGHTFACE_HOME:", os.environ["INSIGHTFACE_HOME"])
    print("[INFO] MODELS_DIR:", MODELS_DIR)

    out = ensure_buffalo_l()

    print("[INFO] sample files:")
    for p in sorted(out.glob("*.onnx"))[:10]:
        print("  -", p.name)

if __name__ == "__main__":
    main()