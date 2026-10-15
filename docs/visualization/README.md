# Reproduce the README gallery

The gallery uses actual recorded pose estimates, measured benchmark samples and
deterministic solver checkpoints. It includes three GIFs and static trajectory
and comparison plots; no renderer invents timing samples or solver states.

## Recorded Intel hero and Rosenbrock

From the repository root:

```sh
cargo run -p ropt --example export_rosenbrock_trace --locked -- docs/assets/rosenbrock_trace.csv
python3 -m venv .venv-viz
.venv-viz/bin/python -m pip install -r docs/visualization/requirements.txt
.venv-viz/bin/python docs/visualization/render_showcase.py
```

The renderer produces `ropt-hero.gif`, its PNG still, `intel943-trajectory.png`,
`isam2-comparison.png`, `rosenbrock-convergence.gif` and its PNG still under
`docs/assets`. [showcase_manifest.json](../assets/showcase_manifest.json) captures
input/renderer/output SHA-256 checksums, sizes and rendering versions.

The Intel hero selects 80 **actual** prefixes from the archived primary ropt
output. It replays all 943 poses / 1,837 measured constraints in an inferred
pose-ID arrival order. It has no interpolation and is not real-time footage.
Its original-energy chart shows the cost of the growing graph, which can increase
as observations arrive; it is not the cost of iterations on a fixed objective.
The comparison plot retains all six paired samples in each of two independent
campaigns, with complete update/replay timings rather than per-frame latency.
Both campaigns use common accuracy gates and explicit full-refresh policies.
The historical MIT1228 attempt was ineligible; a later
[Cartesian/precision comparison](../../benchmarks/PRECISE_REAL_ISAM2.md) passes
the unchanged gates. The Intel hero retains its original archived inputs.
See [the comparison contract](../../benchmarks/REAL_ISAM2.md).

## Recorded CSAIL comparison

The additional static figure uses a different recorded graph, CSAIL: 1,045 poses
and 1,172 original constraints. Its left panel reads the actual archived final
ropt poses and the shared seeds integrated from original odometry; its right
panel shows every measured pair in the primary and confirmation campaigns.
No optimized or truth-based initializer is used.

```sh
MPLCONFIGDIR=/tmp/ropt-viz-config XDG_CACHE_HOME=/tmp/ropt-viz-cache \
  python3 benchmarks/plot_csail_comparison.py
```

This produces `docs/assets/csail-comparison.png` and
[csail-comparison.json](../assets/csail-comparison.json), with source, input,
renderer and image hashes. It leaves the historical showcase inputs/outputs
untouched. The [CSAIL comparison contract](../../benchmarks/CSAIL_REAL_ISAM2.md)
documents the source artifact, unchanged quality gates and full-refresh scope.

Rosenbrock uses the same AD residual and initial state as the standard example.
The exporter reruns LM with every iteration cap and checks each actual state cost
against the full solve's accepted/rejected trial record. The independent renderer
also checks those costs against `0.5 * (100*(y-x*x)^2 + (1-x)^2)` before plotting.
All 32 trials are shown without interpolation. Rejected trials keep the accepted
state. This synthetic demo is not a performance benchmark.

## Bundle adjustment

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
