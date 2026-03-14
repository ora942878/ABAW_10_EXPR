import os
import requests
from pathlib import Path
from tqdm import tqdm

URL = "https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_large_lv60k_asr_ls960.pth"

def download_with_resume(url, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    file_size = 0

    if save_path.exists():
        file_size = save_path.stat().st_size
        headers["Range"] = f"bytes={file_size}-"

    with requests.get(url, headers=headers, stream=True) as r:
        if r.status_code not in (200, 206):
            raise RuntimeError(f"Download failed, status code: {r.status_code}")

        total_size = int(r.headers.get("Content-Length", 0)) + file_size

        mode = "ab" if file_size > 0 else "wb"

        with open(save_path, mode) as f, tqdm(
            total=total_size,
            initial=file_size,
            unit="B",
            unit_scale=True,
            desc="Downloading"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print("[OK] Download completed:", save_path)


if __name__ == "__main__":
    from configs.paths import PATH

    ckpt_path = Path(PATH.PIPELINE_3_ROOT) / "torch_cache" /'hub' /"checkpoints"/ \
                "wav2vec2_fairseq_large_lv60k_asr_ls960.pth"

    download_with_resume(URL, ckpt_path)