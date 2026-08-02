"""Image loading and size normalization.

Rule: if one image is bigger than the other, the bigger one is downsized to the
smaller one's dimensions. Upscaling never happens. Both images are then reduced
to a square working grid (default 64x64) so the assignment problem stays exact
and tractable: N*N pixels means an (N*N) x (N*N) cost matrix for the Hungarian
solver, which is O(n^3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class PreparedImages:
    """Both images at the shared full size, plus the working-grid versions."""

    a_full: Image.Image      # image A at the shared (post-downsize) size
    b_full: Image.Image      # image B at the shared (post-downsize) size
    a_grid: np.ndarray       # (grid, grid, 3) float32 in [0, 1]
    b_grid: np.ndarray       # (grid, grid, 3) float32 in [0, 1]
    grid: int


def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def match_sizes(a: Image.Image, b: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Downsize the larger image so both share the smaller image's dimensions."""
    area_a = a.width * a.height
    area_b = b.width * b.height
    if area_a == area_b and a.size == b.size:
        return a, b
    if area_a <= area_b:
        return a, b.resize(a.size, Image.LANCZOS)
    return a.resize(b.size, Image.LANCZOS), b


def to_grid(img: Image.Image, grid: int) -> np.ndarray:
    small = img.resize((grid, grid), Image.LANCZOS)
    return np.asarray(small, dtype=np.float32) / 255.0


def prepare(path_a: str, path_b: str, grid: int = 64) -> PreparedImages:
    a, b = match_sizes(load_rgb(path_a), load_rgb(path_b))
    return PreparedImages(
        a_full=a,
        b_full=b,
        a_grid=to_grid(a, grid),
        b_grid=to_grid(b, grid),
        grid=grid,
    )
