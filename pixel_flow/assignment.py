"""Optimal pixel assignment: where should every pixel of B go to best match A?

Pixels of B may only MOVE — their colors are never altered. So the question is
a pure assignment problem: pair each source pixel i of B with a distinct grid
position j such that the total perceptual color mismatch against A is minimal.

    minimize  sum_i || Lab(B[i]) - Lab(A[dest(i)]) ||^2
    subject to dest being a permutation (every position used exactly once)

The Hungarian algorithm (scipy's linear_sum_assignment) solves this exactly —
the result is the mathematically optimal rearrangement, not a heuristic.
Costs are computed in CIELAB space so "mismatch" tracks human perception
rather than raw RGB distance.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert (..., 3) sRGB in [0, 1] to CIELAB (D65)."""
    rgb = np.clip(rgb, 0.0, 1.0)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = linear @ m.T
    xyz /= np.array([0.95047, 1.0, 1.08883])  # D65 white point

    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)

    lab = np.empty_like(xyz)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab.astype(np.float32)


def solve_assignment(b_grid: np.ndarray, a_grid: np.ndarray) -> np.ndarray:
    """Return dest indices: pixel i of flattened B belongs at flat position dest[i].

    b_grid, a_grid: (N, N, 3) float arrays in [0, 1].
    """
    n = b_grid.shape[0] * b_grid.shape[1]
    lab_b = srgb_to_lab(b_grid).reshape(n, 3)
    lab_a = srgb_to_lab(a_grid).reshape(n, 3)

    # cost[i, j] = ||lab_b[i] - lab_a[j]||^2, expanded to avoid an (n, n, 3) temp
    sq_b = np.einsum("ij,ij->i", lab_b, lab_b)[:, None]
    sq_a = np.einsum("ij,ij->i", lab_a, lab_a)[None, :]
    cost = (sq_b + sq_a - 2.0 * (lab_b @ lab_a.T)).astype(np.float32)

    rows, cols = linear_sum_assignment(cost)
    dest = np.empty(n, dtype=np.int64)
    dest[rows] = cols
    return dest


def apply_assignment(b_grid: np.ndarray, dest: np.ndarray) -> np.ndarray:
    """Rearrange B's pixels to their assigned positions. Colors are untouched."""
    n, _, c = b_grid.shape
    flat = b_grid.reshape(-1, c)
    out = np.empty_like(flat)
    out[dest] = flat
    return out.reshape(n, n, c)
