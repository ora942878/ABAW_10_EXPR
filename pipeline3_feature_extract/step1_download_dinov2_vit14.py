from configs.paths import PATH
import torch

DINO_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"


def main_download_pth():
    weight_dir = PATH.VIT_WEIGHTS_ROOT
    weight_dir.mkdir(parents=True, exist_ok=True)

    save_path = weight_dir / "dinov2_vitl14.pth"

    if save_path.exists():
        print(f"[OK] already exists: {save_path}")
        return

    print("Downloading DINOv2 ViT-L/14 weights...")
    print("URL:", DINO_URL)
    print("Saving to:", save_path)

    torch.hub.download_url_to_file(
        DINO_URL,
        str(save_path),
        progress=True
    )

    print("\nDownload finished.")

    ckpt = torch.load(save_path, map_location="cpu")
    if isinstance(ckpt, dict):
        print("Loaded checkpoint keys:", list(ckpt.keys())[:10])
    else:
        print("Checkpoint loaded (tensor/state_dict).")

    print("[DONE]")



if __name__ == "__main__":
    main_download_pth()
    import torch

    ckpt = torch.load(PATH.VIT_WEIGHTS_ROOT / 'dinov2_vitl14.pth', map_location="cpu")
    print(type(ckpt))
    print(len(ckpt))

