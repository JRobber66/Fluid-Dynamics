"""Output writers: PDF report, animated GIF, standalone HTML fluid player."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .fluid_sim import SimResult


def _labeled(img: Image.Image, label: str, size: int = 512) -> Image.Image:
    """Upscale to a common page size and stamp a label strip on top."""
    page = Image.new("RGB", (size, size + 36), "white")
    page.paste(img.resize((size, size), Image.NEAREST if img.width < size else Image.LANCZOS), (0, 36))
    draw = ImageDraw.Draw(page)
    draw.rectangle([0, 0, size, 36], fill=(24, 24, 28))
    draw.text((12, 10), label, fill="white")
    return page


def write_pdf(
    path: str | Path,
    a_full: Image.Image,
    b_full: Image.Image,
    rearranged: np.ndarray,
) -> None:
    """Multi-page PDF: target A, source B, and B's pixels optimally rearranged."""
    pages = [
        _labeled(a_full, "Image A - target"),
        _labeled(b_full, "Image B - source (after size matching)"),
        _labeled(
            Image.fromarray((np.clip(rearranged, 0, 1) * 255).astype(np.uint8)),
            "Image B pixels, optimally rearranged to match A (moved only, never recolored)",
        ),
    ]
    pages[0].save(str(path), save_all=True, append_images=pages[1:], resolution=120)


def write_gif(path: str | Path, frames: list[np.ndarray], fps: int, hold_last: int = 20) -> None:
    imgs = [Image.fromarray(f) for f in frames]
    imgs.extend([imgs[-1]] * hold_last)
    imgs[0].save(
        str(path),
        save_all=True,
        append_images=imgs[1:],
        duration=int(round(1000 / fps)),
        loop=0,
    )


_HTML_TEMPLATE = """<!-- pixel_flow interactive player -->
<title>pixel flow — B into A</title>
<style>
  html, body { margin: 0; min-height: 100vh; display: grid; place-items: center;
               background: #0d0e12; color: #e8e8ee; font: 14px system-ui, sans-serif; }
  #wrap { text-align: center; padding: 16px; }
  canvas { max-width: min(92vw, 640px); width: 100%; border-radius: 10px;
           background: #000; image-rendering: pixelated; }
  button { margin-top: 12px; padding: 8px 22px; font-size: 14px; border: 0;
           border-radius: 8px; background: #3d6df2; color: white; cursor: pointer; }
  button:hover { background: #5583ff; }
  #meta { margin-top: 8px; opacity: 0.65; }
</style>
<div id="wrap">
  <canvas id="c" width="__RS__" height="__RS__"></canvas><br>
  <button id="replay">Replay</button>
  <div id="meta">__N__ pixels of image B flowing to their optimal positions in image A — __DUR__ s</div>
</div>
<script>
const DATA = __DATA__;
const DUR = __DUR__ * 1000, RS = __RS__, G = DATA.grid, N = DATA.colors.length;
const ctx = document.getElementById('c').getContext('2d');
const block = Math.max(2, Math.floor(RS / G));
const sx = DATA.start.map(p => p[0]), sy = DATA.start.map(p => p[1]);
const tx = DATA.target.map(p => p[0]), ty = DATA.target.map(p => p[1]);
const px = new Float32Array(N), py = new Float32Array(N);
const vx = new Float32Array(N), vy = new Float32Array(N);
// solid frame: pixel centers stay a half-pixel inside so blocks never poke out
const LO = 0.5 / G, HI = 1 - 0.5 / G, BOUNCE = 0.4;

// per-particle swirl phase so the flow field varies smoothly across space
const styles = DATA.colors.map(c => `rgb(${c[0]},${c[1]},${c[2]})`);
function curl(x, y, t) {
  // analytic divergence-free field: curl of a sum of drifting sine potentials
  const a = 1.9, b = 2.6, w = 0.00045;
  const p1x = Math.sin(a * 6.28 * y + w * t) * Math.cos(b * 6.28 * x - w * t);
  const p1y = -Math.sin(a * 6.28 * x - w * t * 1.3) * Math.cos(b * 6.28 * y + w * t);
  return [p1x, p1y];
}
const smooth = x => { x = Math.min(1, Math.max(0, x)); return x * x * (3 - 2 * x); };

let t0 = null;
function reset() { for (let i = 0; i < N; i++) { px[i] = sx[i]; py[i] = sy[i]; vx[i] = vy[i] = 0; } t0 = null; requestAnimationFrame(tick); }
let last = 0;
function tick(ts) {
  if (t0 === null) { t0 = ts; last = ts; }
  const t = Math.min(ts - t0, DUR), tau = t / DUR;
  const dt = Math.min((ts - last) / 1000, 0.05); last = ts;
  const tRem = Math.max((DUR - t) / 1000, 0.016);
  const swirl = 0.45 * Math.pow(1 - smooth(tau * 1.35), 2);
  const beta = 0.25 + 0.75 * smooth((tau - 0.15) / 0.7);
  const pin = smooth((tau - 0.72) / 0.28);
  for (let i = 0; i < N; i++) {
    let gx = (tx[i] - px[i]) / tRem, gy = (ty[i] - py[i]) / tRem;
    const s = Math.hypot(gx, gy); if (s > 2.5) { gx *= 2.5 / s; gy *= 2.5 / s; }
    const [cx, cy] = curl(px[i], py[i], t);
    const dx = (1 - beta) * cx * swirl * 3 + gx * beta + cx * swirl;
    const dy = (1 - beta) * cy * swirl * 3 + gy * beta + cy * swirl;
    vx[i] += (dx - vx[i]) * Math.min(1, 6 * dt);
    vy[i] += (dy - vy[i]) * Math.min(1, 6 * dt);
    px[i] += vx[i] * dt; py[i] += vy[i] * dt;
    if (pin > 0) { px[i] += (tx[i] - px[i]) * pin; py[i] += (ty[i] - py[i]) * pin; }
    // walls are solid: reflect off the frame, pixels only squish past each other
    if (px[i] < LO) { px[i] = 2 * LO - px[i]; vx[i] *= -BOUNCE; }
    else if (px[i] > HI) { px[i] = 2 * HI - px[i]; vx[i] *= -BOUNCE; }
    if (py[i] < LO) { py[i] = 2 * LO - py[i]; vy[i] *= -BOUNCE; }
    else if (py[i] > HI) { py[i] = 2 * HI - py[i]; vy[i] *= -BOUNCE; }
    px[i] = Math.min(HI, Math.max(LO, px[i])); py[i] = Math.min(HI, Math.max(LO, py[i]));
  }
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, RS, RS);
  for (let i = 0; i < N; i++) {
    ctx.fillStyle = styles[i];
    ctx.fillRect(px[i] * RS - block / 2, py[i] * RS - block / 2, block, block);
  }
  if (t < DUR) requestAnimationFrame(tick);
}
document.getElementById('replay').onclick = reset;
reset();
</script>
"""


def write_html(path: str | Path, result: SimResult, duration: float, render_size: int = 512) -> None:
    """Standalone HTML player: re-runs the flow in-browser with a replay button."""
    start = result.positions[0]
    target = result.positions[-1]
    data = {
        "grid": result.grid,
        "start": [[round(float(x), 4), round(float(y), 4)] for x, y in start],
        "target": [[round(float(x), 4), round(float(y), 4)] for x, y in target],
        "colors": (result.colors * 255).astype(int).tolist(),
    }
    html = (
        _HTML_TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
        .replace("__DUR__", str(duration))
        .replace("__RS__", str(render_size))
        .replace("__N__", str(len(data["colors"])))
    )
    Path(path).write_text(html)
