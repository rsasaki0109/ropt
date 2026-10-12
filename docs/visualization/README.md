# Reproduce the README animation

The checked-in GIF visualizes the deterministic, noise-free Bundle Adjustment
fixture used by the Rust regression tests: 4 camera poses, 12 world points and
48 pixel observations. Two known camera poses are fixed to anchor the world
frame and metric scale. Intrinsics use an undistorted pinhole model.

## Export actual solver states

From the repository root, with the pinned Rust toolchain:

```sh
cargo run -p ropt --example export_ba_trace --locked -- docs/assets/ba_trace.json
```

The exporter first solves with LM + block Schur, then reruns from the same initial
state with iteration caps 1 through 8. Each resulting checkpoint is checked
against the original solve's cost. This records the real optimization trajectory
without adding per-iteration state allocations to the library solver. Projection
points and camera geometry come from those states. The checked-in trace contains
9 checkpoints, including the initial state.

## Render

Python 3.12+ with Matplotlib, NumPy and Pillow is needed only to regenerate the
visual, not to build or use ropt. The renderer was validated with the exact versions
in `requirements.txt`. Install them in a virtual environment:

```sh
python3 -m venv .venv-viz
.venv-viz/bin/python -m pip install -r docs/visualization/requirements.txt
.venv-viz/bin/python docs/visualization/render_ba.py \
  docs/assets/ba_trace.json docs/assets/ba-convergence.gif
```

Rendering is headless and needs no external assets, fonts or services. It writes
Matplotlib/font caches under the system temporary directory when those paths are
not already configured. An optional `--preview /tmp/ropt-ba-preview.png` writes a
contact sheet for visual inspection. The GIF is 1100×700, loops for about 6 seconds,
and is about 1.6 MB. Pillow coalesces repeated identical frames during encoding.

Geometry and displayed projections are interpolated between checkpoints to make
playback legible. Numeric readouts and the convergence curve use actual solver
checkpoints. The world view displays X/Z/Y, so optical depth lies across the ground
plane. Camera glyphs use a 640×480 image and the fixture's calibrated intrinsics,
with a 0.5-unit visualization depth.

RMS is the square root of the mean **2D** squared reprojection distance across
all 48 observations, in pixels. The final value is about 5.9e-12 pixels for this
noise-free synthetic fixture. This is not validation on a public BA dataset,
a real-time playback recording, or a speed comparison with another solver.
