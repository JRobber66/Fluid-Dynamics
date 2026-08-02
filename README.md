# pixel_flow — image B flows into image A

Takes two images, **A** (the target) and **B** (the source), and:

1. **Size-matches** them — if one is bigger than the other, the bigger one is
   downsized to the smaller one's dimensions (never upscaled).
2. **Writes a PDF** (`report.pdf`) containing A, B, and the optimal rearrangement.
3. **Finds the mathematically best place for every pixel of B** so the result
   matches A as closely as possible — pixels can only *move*, never be
   recolored. This is solved *exactly* as an assignment problem with the
   Hungarian algorithm (`scipy.optimize.linear_sum_assignment`), using
   squared CIELAB distance as the cost so "closeness" is perceptual. The
   result is the globally optimal permutation, not a heuristic.
4. **Runs a fluid simulation** in which every pixel of B flies **its own
   mission**: its own departure time, its own arrival deadline, its own curved
   flight path (per-pixel arc curvature and wobble), on top of a gentle
   background incompressible fluid (semi-Lagrangian advection + FFT pressure
   projection + decaying curl noise). Pixels squish and push past each other
   freely, bounce off the solid frame, and each one is hard-pinned to its
   exact destination at its own arrival time — every arrival is guaranteed
   within **at most 10 seconds**.

Outputs an animated **GIF** and a standalone interactive **HTML player**
(open in any browser, click Replay).

## Usage

```bash
pip install -r requirements.txt

python -m pixel_flow imageA.png imageB.png -o out/
```

Options:

| flag | default | meaning |
|---|---|---|
| `--grid N` | 64 | working grid: N×N pixels are assigned & simulated (Hungarian is O(n³), 64 → 4096 pixels ≈ 25 s to solve) |
| `--duration S` | 10 | animation length in seconds, hard-capped at 10 |
| `--fps` | 20 | GIF frame rate |
| `--render-size` | 448 | GIF resolution in pixels |
| `--seed` | 0 | fluid turbulence seed |

## Demo

No images handy? Generate the built-in demo pair (a sunset that flows into
concentric rings) and run:

```bash
python examples/make_demo_images.py
python -m pixel_flow examples/demo_a.png examples/demo_b.png -o examples/out
```

## How the pieces fit

```
preprocess.py   load, downsize larger → smaller, reduce to working grid
assignment.py   sRGB→CIELAB, exact Hungarian assignment (a true permutation:
                every color of B used exactly once, none altered)
fluid_sim.py    per-pixel missions (own departure, deadline, arc, wobble) over
                a background incompressible fluid on a 48² grid; solid frame
                walls; each pixel pinned to its target at its own arrival time
outputs.py      PDF report, animated GIF, self-contained HTML player
cli.py          orchestrates the four steps
```

Invariants (all machine-checked):
- the destination map is a permutation — every grid position receives exactly one pixel;
- the multiset of colors after rearrangement is identical to B's — moved, never changed;
- final particle positions equal assigned targets exactly, at t = duration ≤ 10 s.
