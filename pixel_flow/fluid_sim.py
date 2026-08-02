"""Fluid simulation that carries every pixel of B to its assigned destination.

Each pixel is a particle in a 2D incompressible fluid on the unit square:

  * every pixel has its OWN mission: its own departure time, its own arrival
    deadline, its own curved flight path (per-pixel arc curvature and wobble
    rhythm) — so pixels stream individually instead of moving in grouped waves;
  * a background incompressible fluid (semi-Lagrangian advection + FFT
    Helmholtz projection + decaying curl noise) adds gentle shared ambience,
    deliberately too weak to herd pixels into groups;
  * per-pixel guidance (target - pos) / own_time_remaining plus a terminal
    blend pins each pixel to its exact destination at its own arrival time.

Every arrival time is <= 10 seconds: whatever the fluid does, each pixel's
schedule guarantees it lands on target by the deadline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_DURATION = 10.0


@dataclass
class SimConfig:
    duration: float = 10.0        # seconds; clamped to MAX_DURATION
    fps: int = 20
    field_res: int = 48           # fluid grid resolution
    swirl_strength: float = 0.45  # curl-noise amplitude of the background fluid at t = 0
    dissipation: float = 0.985    # per-step velocity retention
    relax: float = 6.0            # particle velocity relaxation rate (1/s); higher = less inertia
    substeps: int = 4             # physics substeps per rendered frame (keeps CFL low)
    bounce: float = 0.4           # velocity kept (and reversed) when a pixel hits the frame wall
    seed: int = 0

    def __post_init__(self) -> None:
        self.duration = min(float(self.duration), MAX_DURATION)


@dataclass
class SimResult:
    positions: list[np.ndarray] = field(default_factory=list)  # per frame (n, 2) in [0,1]
    colors: np.ndarray = None                                   # (n, 3) float in [0,1]
    grid: int = 0
    fps: int = 0
    # per-pixel mission data (so other renderers can replay the same missions)
    depart: np.ndarray = None        # (n,) departure time in seconds
    arrive: np.ndarray = None        # (n,) arrival deadline in seconds, all <= 10
    arc: np.ndarray = None           # (n,) signed arc curvature of the flight path
    wobble_freq: np.ndarray = None   # (n,) personal wiggle frequency
    wobble_phase: np.ndarray = None  # (n,) personal wiggle phase


def _cell_centers(grid: int) -> np.ndarray:
    """(grid*grid, 2) positions of cell centers in [0,1]^2, row-major (x, y)."""
    idx = np.arange(grid * grid)
    ys = (idx // grid + 0.5) / grid
    xs = (idx % grid + 0.5) / grid
    return np.stack([xs, ys], axis=1).astype(np.float64)


def _fft_lowpass_noise(res: int, cutoff: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth random potential field via FFT low-pass of white noise."""
    noise = rng.standard_normal((res, res))
    f = np.fft.fft2(noise)
    kx = np.fft.fftfreq(res)[None, :]
    ky = np.fft.fftfreq(res)[:, None]
    mask = np.exp(-((kx**2 + ky**2) / (2 * cutoff**2)))
    psi = np.real(np.fft.ifft2(f * mask))
    psi /= np.abs(psi).max() + 1e-12
    return psi


def _curl(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divergence-free velocity from a scalar potential: (d psi/dy, -d psi/dx)."""
    dpdy = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) * 0.5
    dpdx = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) * 0.5
    return dpdy, -dpdx


def _project(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove the divergent component (Helmholtz projection, periodic FFT)."""
    uh, vh = np.fft.fft2(u), np.fft.fft2(v)
    res = u.shape[0]
    kx = np.fft.fftfreq(res)[None, :]
    ky = np.fft.fftfreq(res)[:, None]
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0
    div = kx * uh + ky * vh
    uh -= kx * div / k2
    vh -= ky * div / k2
    return np.real(np.fft.ifft2(uh)), np.real(np.fft.ifft2(vh))


def _bilinear_sample(f: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Sample grid field f (res, res) at positions (n, 2) in [0,1]^2, periodic."""
    res = f.shape[0]
    gx = pos[:, 0] * res - 0.5
    gy = pos[:, 1] * res - 0.5
    x0 = np.floor(gx).astype(int)
    y0 = np.floor(gy).astype(int)
    fx = gx - x0
    fy = gy - y0
    x0m, x1m = x0 % res, (x0 + 1) % res
    y0m, y1m = y0 % res, (y0 + 1) % res
    return (
        f[y0m, x0m] * (1 - fx) * (1 - fy)
        + f[y0m, x1m] * fx * (1 - fy)
        + f[y1m, x0m] * (1 - fx) * fy
        + f[y1m, x1m] * fx * fy
    )


def _advect(f: np.ndarray, u: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
    """Semi-Lagrangian advection of grid field f by velocity (u, v), periodic."""
    res = f.shape[0]
    ys, xs = np.mgrid[0:res, 0:res]
    px = (xs + 0.5) / res - u * dt
    py = (ys + 0.5) / res - v * dt
    pos = np.stack([px.ravel(), py.ravel()], axis=1)
    return _bilinear_sample(f, pos).reshape(res, res)


def _smoothstep(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def simulate(
    b_grid: np.ndarray,
    dest: np.ndarray,
    config: SimConfig | None = None,
) -> SimResult:
    """Run the fluid sim. Returns per-frame particle positions plus colors.

    b_grid: (N, N, 3) source image (the moving pixels).
    dest:   flat destination index for each flat source pixel (from assignment).
    """
    cfg = config or SimConfig()
    grid = b_grid.shape[0]
    n = grid * grid
    rng = np.random.default_rng(cfg.seed)

    centers = _cell_centers(grid)
    pos = centers.copy()                 # start: B's own layout
    target = centers[dest]               # end: assigned positions in A's layout
    colors = b_grid.reshape(-1, 3).astype(np.float32)

    # ---- per-pixel missions: every pixel gets its own schedule and flight plan
    T = cfg.duration
    depart = rng.uniform(0.0, 0.35, n) * T            # when this pixel leaves B
    arrive = rng.uniform(0.45, 1.0, n) * T            # when it must be in place
    dist = np.linalg.norm(target - pos, axis=1)
    min_window = 0.20 * T + 0.25 * T * dist / max(dist.max(), 1e-9)
    arrive = np.clip(np.maximum(arrive, depart + min_window), None, T)
    depart = np.minimum(depart, arrive - 0.15 * T)    # every mission gets airtime
    arc = rng.uniform(-1.0, 1.0, n) * rng.uniform(0.3, 1.0, n)  # own curve dir/amount
    wobble_freq = rng.uniform(2.0, 5.0, n)            # own wiggle rhythm
    wobble_phase = rng.uniform(0.0, 2.0 * np.pi, n)

    res = cfg.field_res
    u = np.zeros((res, res))
    v = np.zeros((res, res))
    psi1 = _fft_lowpass_noise(res, 0.06, rng)
    psi2 = _fft_lowpass_noise(res, 0.06, rng)

    n_frames = max(2, int(round(cfg.duration * cfg.fps)))
    n_steps = n_frames * cfg.substeps
    dt = cfg.duration / n_steps
    vel = np.zeros_like(pos)

    # solid frame: pixel centers stay a half-pixel inside so blocks never poke out
    wall_lo = 0.5 / grid
    wall_hi = 1.0 - wall_lo

    result = SimResult(
        colors=colors, grid=grid, fps=cfg.fps,
        depart=depart, arrive=arrive, arc=arc,
        wobble_freq=wobble_freq, wobble_phase=wobble_phase,
    )
    result.positions.append(pos.copy())

    for step in range(n_steps):
        t = (step + 1) * dt
        tau = t / cfg.duration

        flying = (t >= depart) & (t < arrive)
        landed = t >= arrive
        frac = np.clip((t - depart) / np.maximum(arrive - depart, 1e-9), 0.0, 1.0)

        # guidance: each pixel flies to make ITS OWN deadline, not a global one
        t_left = np.maximum(arrive - t, dt)[:, None]
        guide = (target - pos) / t_left
        speed = np.linalg.norm(guide, axis=1, keepdims=True)
        vmax = 3.0
        guide *= np.minimum(1.0, vmax / np.maximum(speed, 1e-9))

        # own curved detour: arc sideways, strongest mid-flight, plus a personal wobble
        heading = guide / np.maximum(speed, 1e-9)
        perp = np.stack([-heading[:, 1], heading[:, 0]], axis=1)
        wig = arc + 0.35 * np.sin(wobble_freq * 2.0 * np.pi * frac + wobble_phase)
        detour = perp * (wig * np.sin(np.pi * frac) * speed[:, 0] * 0.8)[:, None]

        # gentle background fluid (ambience only — too weak to herd pixels into groups)
        u0, v0 = u, v
        u = _advect(u0, u0, v0, dt) * cfg.dissipation
        v = _advect(v0, u0, v0, dt) * cfg.dissipation
        swirl = cfg.swirl_strength * (1.0 - _smoothstep(tau * 1.2)) ** 2
        if swirl > 1e-4:
            omega = 2.0 * np.pi * 0.35 * t
            psi = np.cos(omega) * psi1 + np.sin(omega) * psi2
            cu, cv = _curl(psi)
            u += cu * swirl * res * dt * 2.0
            v += cv * swirl * res * dt * 2.0
        u, v = _project(u, v)
        # no-flux walls: the frame is solid, so kill normal flow at the edges
        u[:, 0] = u[:, -1] = 0.0
        v[0, :] = v[-1, :] = 0.0
        fluid_vel = np.stack(
            [_bilinear_sample(u, pos), _bilinear_sample(v, pos)], axis=1
        )

        desired = guide + detour + 0.15 * fluid_vel
        desired[~flying] = 0.0                        # waiting at home or landed
        vel += (desired - vel) * min(1.0, cfg.relax * dt)
        vel[landed] = 0.0
        pos = pos + vel * dt

        # per-pixel terminal pin: exact landing at each pixel's OWN arrival time
        pin = _smoothstep((frac - 0.8) / 0.2)[:, None]
        pos = pos + (target - pos) * pin
        pos[landed] = target[landed]

        # solid walls: pixels squish past each other freely, but bounce off the
        # frame — reflect the position and reverse (damped) the normal velocity
        for ax in range(2):
            below = pos[:, ax] < wall_lo
            above = pos[:, ax] > wall_hi
            pos[below, ax] = 2.0 * wall_lo - pos[below, ax]
            pos[above, ax] = 2.0 * wall_hi - pos[above, ax]
            hit = below | above
            vel[hit, ax] *= -cfg.bounce
        np.clip(pos, wall_lo, wall_hi, out=pos)  # safety net for extreme overshoots

        if (step + 1) % cfg.substeps == 0:
            result.positions.append(pos.copy())

    result.positions[-1] = target.copy()  # exact, by construction
    return result


def render_frames(
    result: SimResult,
    render_size: int = 448,
    background: float = 0.0,
) -> list[np.ndarray]:
    """Rasterize each frame: every particle is a square block of its color."""
    n = result.colors.shape[0]
    block = max(2, render_size // result.grid)
    offs = np.arange(block) - block // 2
    oy, ox = np.meshgrid(offs, offs, indexing="ij")
    oy, ox = oy.ravel(), ox.ravel()

    frames = []
    for pos in result.positions:
        px = pos[:, 0] * render_size
        py = pos[:, 1] * render_size
        acc = np.zeros((render_size, render_size, 3), dtype=np.float32)
        cnt = np.zeros((render_size, render_size, 1), dtype=np.float32)
        for dy, dx in zip(oy, ox):
            ix = np.clip((px + dx).astype(int), 0, render_size - 1)
            iy = np.clip((py + dy).astype(int), 0, render_size - 1)
            np.add.at(acc, (iy, ix), result.colors)
            np.add.at(cnt, (iy, ix), 1.0)
        img = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), background)
        frames.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
    return frames
