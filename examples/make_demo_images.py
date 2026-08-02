"""Generate two demo images so the pipeline can be tried with zero setup.

  demo_a.png — the target: concentric rainbow rings on a dark sky (320x320)
  demo_b.png — the source: a warm sunset gradient with a sun disk (480x480,
               deliberately larger so the downsizing step has work to do)
"""

from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent


def demo_a(size: int = 320) -> Image.Image:
    y, x = np.mgrid[0:size, 0:size].astype(np.float64) / size - 0.5
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    hue = (r * 6.0 + theta / (2 * np.pi)) % 1.0
    rings = 0.5 + 0.5 * np.cos(r * 40.0)
    img = np.stack(
        [
            rings * (0.5 + 0.5 * np.cos(2 * np.pi * (hue + s)))
            for s in (0.0, 1 / 3, 2 / 3)
        ],
        axis=-1,
    )
    img *= np.clip(1.2 - 1.6 * r, 0.05, 1.0)[..., None]
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def demo_b(size: int = 480) -> Image.Image:
    y, x = np.mgrid[0:size, 0:size].astype(np.float64) / size
    sky_r = 0.95 - 0.55 * y
    sky_g = 0.45 + 0.15 * (1 - y) - 0.25 * y
    sky_b = 0.30 + 0.45 * y
    img = np.stack([sky_r, sky_g, sky_b], axis=-1)
    sun = np.sqrt((x - 0.5) ** 2 + (y - 0.42) ** 2)
    disk = np.clip(1.0 - sun / 0.16, 0, 1)[..., None] ** 0.5
    img = img * (1 - disk) + disk * np.array([1.0, 0.92, 0.62])
    img[y > 0.78] = [0.08, 0.07, 0.12]  # ground silhouette
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


if __name__ == "__main__":
    demo_a().save(HERE / "demo_a.png")
    demo_b().save(HERE / "demo_b.png")
    print(f"wrote {HERE / 'demo_a.png'} (320x320) and {HERE / 'demo_b.png'} (480x480)")
