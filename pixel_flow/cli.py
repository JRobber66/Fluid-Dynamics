"""Command-line entry point.

    python -m pixel_flow imageA.png imageB.png -o out/

Steps, in order:
  1. Load A and B; downsize the bigger one to the smaller one's dimensions.
  2. Write a PDF containing A, B, and B-optimally-rearranged.
  3. Hungarian algorithm: mathematically optimal destination for every B pixel
     (pixels only move — colors never change).
  4. Incompressible fluid sim carries each pixel to its destination in <= 10 s.
     Outputs: animated GIF + standalone interactive HTML player.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .assignment import apply_assignment, solve_assignment
from .fluid_sim import MAX_DURATION, SimConfig, render_frames, simulate
from .outputs import write_gif, write_html, write_pdf
from .preprocess import prepare


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="pixel_flow",
        description="Fluid-simulate the pixels of image B flowing into the optimal arrangement matching image A.",
    )
    p.add_argument("image_a", help="target image (what B's pixels should look like)")
    p.add_argument("image_b", help="source image (the pixels that move)")
    p.add_argument("-o", "--out", default="out", help="output directory (default: out/)")
    p.add_argument("--grid", type=int, default=64,
                   help="working grid N: N*N pixels are assigned and simulated (default 64)")
    p.add_argument("--duration", type=float, default=10.0,
                   help=f"animation length in seconds, capped at {MAX_DURATION:.0f} (default 10)")
    p.add_argument("--fps", type=int, default=20, help="GIF frame rate (default 20)")
    p.add_argument("--render-size", type=int, default=448, help="GIF resolution (default 448)")
    p.add_argument("--seed", type=int, default=0, help="fluid randomness seed")
    args = p.parse_args(argv)

    if args.duration > MAX_DURATION:
        print(f"duration {args.duration}s exceeds the cap; clamping to {MAX_DURATION:.0f}s")
        args.duration = MAX_DURATION

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] loading + size-matching {args.image_a} and {args.image_b} ...")
    prep = prepare(args.image_a, args.image_b, grid=args.grid)
    print(f"      shared size {prep.a_full.size}, working grid {args.grid}x{args.grid} "
          f"({args.grid * args.grid} pixels)")

    print(f"[2/4] solving optimal assignment (Hungarian, CIELAB cost) ...")
    t0 = time.time()
    dest = solve_assignment(prep.b_grid, prep.a_grid)
    rearranged = apply_assignment(prep.b_grid, dest)
    print(f"      solved in {time.time() - t0:.1f}s — globally optimal, pixels moved only")

    pdf_path = out / "report.pdf"
    write_pdf(pdf_path, prep.a_full, prep.b_full, rearranged)
    print(f"[3/4] PDF written -> {pdf_path}")

    print(f"[4/4] fluid simulation ({args.duration:.1f}s, {args.fps} fps) ...")
    t0 = time.time()
    cfg = SimConfig(duration=args.duration, fps=args.fps, seed=args.seed)
    result = simulate(prep.b_grid, dest, cfg)
    frames = render_frames(result, render_size=args.render_size)
    gif_path = out / "flow.gif"
    html_path = out / "flow.html"
    write_gif(gif_path, frames, fps=args.fps)
    write_html(html_path, result, duration=cfg.duration)
    print(f"      simulated + rendered in {time.time() - t0:.1f}s")
    print(f"      GIF  -> {gif_path}")
    print(f"      HTML -> {html_path}  (open in a browser, click Replay)")


if __name__ == "__main__":
    main()
