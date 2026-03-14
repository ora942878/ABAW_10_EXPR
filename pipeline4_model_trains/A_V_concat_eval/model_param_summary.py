from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

_THIS = Path(__file__).resolve()
for p in [_THIS.parent, *_THIS.parents]:
    if (p / 'pipeline4_model_trains').exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break

from pipeline4_model_trains.A_V_concat_eval.build import (
    load_cfg,
    apply_runtime_defaults,
    build_model,
)

try:
    from torchinfo import summary
except Exception as e:
    raise RuntimeError(
        "please pip install torchinfo"
    ) from e


MODEL_NAMES = [
    'linear',
    'mlp',
    'gate',
    'dynamic',
    'bilinear',
    'crossattn',
    'moe',
]


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_int(x: int) -> str:
    return f"{x:,}"


def get_cfg(model_name: str) -> Any:
    cfg = load_cfg(model_name)
    cfg = apply_runtime_defaults(cfg)
    return cfg


def build_one(model_name: str):
    cfg = get_cfg(model_name)
    model = build_model(cfg)
    return cfg, model


def run_torchinfo(model_name: str, cfg, model: torch.nn.Module):
    model = model.cpu()
    model.eval()

    vis_dim = int(getattr(cfg, 'VIS_DIM'))
    aud_dim = int(getattr(cfg, 'AUD_DIM'))

    return summary(
        model,
        input_size=[(1, vis_dim), (1, aud_dim)],
        dtypes=[torch.float32, torch.float32],
        col_names=['input_size', 'output_size', 'num_params', 'trainable'],
        depth=4,
        verbose=0,
        device='cpu',
    )


def main():
    rows: list[dict[str, Any]] = []

    print('=' * 120)
    print('Model parameter summary (torchinfo)')
    print('=' * 120)

    for name in MODEL_NAMES:
        print(f"\n[MODEL] {name}")
        cfg, model = build_one(name)
        total, trainable = count_params(model)

        ok = True
        err_msg = ''
        info_text = ''

        try:
            info = run_torchinfo(name, cfg, model)
            info_text = str(info)
            print(info_text)
        except Exception as e:
            ok = False
            err_msg = repr(e)
            print(f"[WARN] torchinfo failed for {name}: {err_msg}")
            print(f"[FALLBACK] total_params={format_int(total)}, trainable_params={format_int(trainable)}")

        rows.append(
            {
                'model': name,
                'head_type': getattr(cfg, 'HEAD_TYPE', name),
                'vis_dim': int(getattr(cfg, 'VIS_DIM', -1)),
                'aud_dim': int(getattr(cfg, 'AUD_DIM', -1)),
                'total_params': total,
                'trainable_params': trainable,
                'torchinfo_ok': ok,
                'torchinfo_error': err_msg,
            }
        )

    print('\n' + '=' * 120)
    print('Compact parameter table')
    print('=' * 120)
    print(f"{'model':<12} {'total_params':>16} {'trainable_params':>18} {'torchinfo':>10}")
    print('-' * 120)
    for r in rows:
        print(
            f"{r['model']:<12} {format_int(r['total_params']):>16} {format_int(r['trainable_params']):>18} {str(r['torchinfo_ok']):>10}"
        )


if __name__ == '__main__':
    main()
