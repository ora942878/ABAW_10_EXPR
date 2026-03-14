
import random
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF


class RandomBlackPadShift:
    def __init__(
        self,
        p: float = 0.6,
        max_area_frac: float = 0.20,
        min_bar_frac: float = 0.03,
        max_shift_frac: float = 0.02,
        allow_L: bool = True,
    ):
        self.p = float(p)
        self.max_area_frac = float(max_area_frac)
        self.min_bar_frac = float(min_bar_frac)
        self.max_shift_frac = float(max_shift_frac)
        self.allow_L = bool(allow_L)

    def _sample_L_fracs(self) -> Tuple[float, float]:
        for _ in range(50):
            v = random.uniform(self.min_bar_frac, self.max_area_frac)
            h = random.uniform(self.min_bar_frac, self.max_area_frac)
            total = v + h - v * h
            if total <= self.max_area_frac:
                return v, h
        # fallback
        v = min(self.max_area_frac * 0.6, 0.15)
        h = min(self.max_area_frac * 0.6, 0.15)
        return v, h

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        w, h = img.size
        if w < 8 or h < 8:
            return img

        patterns = ["left", "right", "top", "bottom"]
        if self.allow_L:
            patterns += ["L_lt", "L_lb", "L_rt", "L_rb"]

        pat = random.choice(patterns)

        max_shift_px = int(round(self.max_shift_frac * min(w, h)))
        max_shift_px = max(1, min(max_shift_px, 10))  # “超级小幅度”
        dx = 0
        dy = 0

        if pat in ["left", "right"]:
            v_frac = random.uniform(self.min_bar_frac, self.max_area_frac)
            t = int(round(v_frac * w))
            t = max(1, min(t, w - 1))
            if pat == "left":
                dx = +random.randint(1, max_shift_px)
                bars = [("left", t)]
            else:
                dx = -random.randint(1, max_shift_px)
                bars = [("right", t)]

        elif pat in ["top", "bottom"]:
            h_frac = random.uniform(self.min_bar_frac, self.max_area_frac)
            t = int(round(h_frac * h))
            t = max(1, min(t, h - 1))
            if pat == "top":
                dy = +random.randint(1, max_shift_px)
                bars = [("top", t)]
            else:
                dy = -random.randint(1, max_shift_px)
                bars = [("bottom", t)]

        else:
            v_frac, h_frac = self._sample_L_fracs()
            tv = int(round(v_frac * w))
            th = int(round(h_frac * h))
            tv = max(1, min(tv, w - 1))
            th = max(1, min(th, h - 1))

            if pat == "L_lt":
                dx = +random.randint(1, max_shift_px)
                dy = +random.randint(1, max_shift_px)
                bars = [("left", tv), ("top", th)]
            elif pat == "L_lb":
                dx = +random.randint(1, max_shift_px)
                dy = -random.randint(1, max_shift_px)
                bars = [("left", tv), ("bottom", th)]
            elif pat == "L_rt":
                dx = -random.randint(1, max_shift_px)
                dy = +random.randint(1, max_shift_px)
                bars = [("right", tv), ("top", th)]
            else:  # L_rb
                dx = -random.randint(1, max_shift_px)
                dy = -random.randint(1, max_shift_px)
                bars = [("right", tv), ("bottom", th)]

        if dx != 0 or dy != 0:
            img = TF.affine(img, angle=0.0, translate=(dx, dy), scale=1.0, shear=(0.0, 0.0), fill=0)

        draw = ImageDraw.Draw(img)
        for side, t in bars:
            if side == "left":
                draw.rectangle([0, 0, t - 1, h], fill=(0, 0, 0))
            elif side == "right":
                draw.rectangle([w - t, 0, w, h], fill=(0, 0, 0))
            elif side == "top":
                draw.rectangle([0, 0, w, t - 1], fill=(0, 0, 0))
            elif side == "bottom":
                draw.rectangle([0, h - t, w, h], fill=(0, 0, 0))

        return img