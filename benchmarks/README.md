# Reproducible small pose-graph benchmark

This measures complete optimization calls on the pinned public GTSAM
`w100_30.g2o` sample: 30 poses, 59 constraints, 177 residuals, 87 active tangent
coordinates, vertex 0 fixed exactly at its input pose, squared loss. The residual
is the translation and principal angle of `Z^-1*(Xi^-1*Xj)`, with objective
`0.5*sum(r^T*Omega*r)`. See the [dataset validation](../docs/validation/pose_graph_2d.md)
for provenance and information-matrix ordering. The dataset remains external.

A later [exact-Jacobian C++ GTSAM comparison](gtsam_exact/README.md) now passes
the common quality gate at 30, 100 and 300 poses. This page preserves the earlier
wheel baseline and its failed gate; the two configurations must not be conflated.
The subsequent [PCG workspace comparison](BUFFERED_PCG.md) measures a checked
implementation improvement and preserves both the earlier and updated runs.
The subsequent [sparse LDLᵀ comparison](SPARSE_LDL.md) adds a direct backend
and repeats the common quality gates without replacing these historical runs.
The latest [reusable LDL and analytic SE2 comparison](REUSED_LDL.md) includes
alternating measurements against native GTSAM QR and Cholesky configurations.
The [large-graph validation](LARGE_POSE_GRAPH.md) extends eligible measurements
to 1,000 poses and retains the failed 3,000-pose accuracy case.
The subsequent [stable-objective validation](STABLE_OBJECTIVE.md) resolves that
accuracy failure while preserving the earlier diagnostics.

The sample is too small to establish scalability or SOTA. Memory is not measured.
This adds a way to measure solver configurations and reject inaccurate results;
it does not establish a performance advantage over GTSAM, Ceres or g2o.

## Reproduce

Run from the repository root. Rust 1.99.0 and Python 3.12 were used. No new Rust
dependency or C++ runtime is required by ropt. GTSAM is an optional external
benchmark dependency, installed in a separate virtual environment.

```sh
curl -fL https://raw.githubusercontent.com/borglab/gtsam/a74146dbea788ea25828041065d82674d6430ada/examples/Data/w100_30.g2o -o /tmp/w100_30.g2o
cargo build -p ropt --example benchmark_pose_graph --release --locked
python3 -m venv /tmp/ropt-benchmark-venv
/tmp/ropt-benchmark-venv/bin/python -m pip install -r benchmarks/requirements.txt
benchmark_cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c "$benchmark_cpu" \
  /tmp/ropt-benchmark-venv/bin/python benchmarks/pose_graph.py \
  --input /tmp/w100_30.g2o --ropt target/release/examples/benchmark_pose_graph \
  --output /tmp/ropt-benchmark-results --repeats 30 --warmup 5
```

The runner verifies SHA256
`69d9959b6e6c530ad268b4d9999383acbd262b54319fd830e1d3075142c73343`
before evaluating input. `taskset` is optional on systems without Linux affinity;
the actual affinity is recorded. Output must be a new directory outside the
checkout; use a different path on subsequent runs. The Rust timing executable
refuses debug builds and existing output directories.

With the pinned GTSAM wheel and settings below, the full runner exits **1** after
saving its report: the native reference fails the common quality gate. This is
an observed reference-quality limitation, not a missing dependency or discarded
measurement. `report.json` marks it ineligible. Do not interpret its raw timing
as an equal-accuracy speed comparison.

To measure only ropt, add `--without-gtsam`; then the Python runner needs only
the standard library and can be called with `python3`. For an offline functional
check using the ropt-authored four-pose loop, run:

```sh
python3 benchmarks/check_smoke.py target/release/examples/benchmark_pose_graph
```

CI builds this release example and runs the offline check. It checks the four
default backends plus both explicit sparse-LDL orderings, comparing
solutions against scalar objective/central-difference calculations, sample counts,
fixed poses and protection against overwriting existing results. It does not
download public data, install GTSAM, or assert timing thresholds.

## Measurement and accuracy rules

- Each ropt backend uses the same graph and initial state. Five complete warmup
  solves per backend are discarded, followed by 30 measured solves. The first
  backend rotates each round. An extra untimed dense solve supplies a consistency
  reference. All graphs are constructed before timing.
- `Instant` measures `solve_local`, including initial evaluation, forward AD,
  whitening, matrix assembly, linear solves, damping trials, preconditioner
  construction and report construction. Initial-state cloning, input parsing,
  graph construction, output serialization and independent validation are
  excluded. Final report destruction is excluded; intermediate solver cleanup
  is part of the call. This is a solve-call benchmark, not just a linear solve.
- ropt uses default LM: 100 maximum trials, initial damping 1e-3, gradient/step
  tolerances 1e-10 and cost tolerance 1e-12. Both CG modes use 10,000 maximum
  iterations, relative tolerance 1e-8 and absolute tolerance 1e-10. True linear
  residuals are checked. `ropt/options.txt` records every backend's full options.
- GTSAM uses native `BetweenFactorPose2`, full Gaussian information matrices,
  exact `NonlinearEqualityPose2` at vertex 0, COLAMD multifrontal QR and LM with
  initial damping 1e-3, 100 maximum iterations, absolute/relative error-change
  tolerances 1e-12 and total-error threshold 0. Its other defaults are recorded
  in `gtsam-options.txt`. No Python custom factor or numerical Jacobian is used
  in its optimization. Optimizer construction and `optimize()` are timed together;
  copying initial `Values`, graph construction, validation and serialization are
  excluded. Python/native binding overhead in these calls is included.
- GTSAM and ropt have different retractions, damping policies, stopping rules,
  derivatives and linear-solver implementations. GTSAM optimizer iterations are
  reported separately from ropt LM trial counts. The libraries run sequentially,
  rather than in an interleaved paired experiment. Even passing all quality
  gates would establish only these particular configured runs.
- Scalar Python arithmetic independently evaluates the original input edges.
  Each first solution must preserve the anchor exactly, have finite coordinates,
  lie within 1e-9 of the previously validated cost 0.08290751089989, have absolute
  stored-coordinate central-difference gradient infinity norm at most 1e-5
  (step 1e-6, anchor excluded), and differ from the dense solution by at most
  1e-6 per pose in translation/wrapped-angle norm. The cost reference and dense
  pose agreement are regression checks, not an independent global-optimality
  oracle. The independent gradient adds a stationarity check.
- Every measured ropt solve is checked for convergence, cost consistency,
  anchor preservation and pose agreement outside the timer. GTSAM repeated
  solutions are also checked outside the timer. Raw timing samples are saved
  with no outlier deletion. Statistics include median, mean, standard deviation
  and minimum/maximum; no speed ratios are generated.

`metadata.json` and `report.json` record CPU, affinity/quota, OS, thread-variable
settings, Rust/Python/package versions, dataset checksum, release-binary hash,
GTSAM native-module hash, Git HEAD/dirty state, a fingerprint of Rust sources and
Cargo manifests/lockfile, and the Python driver hash. Shared-cloud scheduling and
CPU frequency are uncontrolled. Record a fresh result when changing code or
conditions; do not reuse these timings as stable machine-independent numbers.

## Recorded run

The [report](results/w100_30/report.json),
[ropt raw samples](results/w100_30/ropt-samples.csv),
[GTSAM raw samples](results/w100_30/gtsam-samples.csv) and both option files
record the 2026-10-12 run on an AMD EPYC 9V74, Debian 13 / Linux 6.18.44,
Rust 1.99.0, default Cargo release profile. CPU affinity was `{0}` and the
three recorded thread variables were set to 1. The checkout was dirty; the
report identifies the measured source fingerprint rather than pretending the
old HEAD alone reproduces the implementation. Optimized poses remain external
under the generated output directory; no third-party dataset is vendored here.

| ropt backend | Median solve ms | LM trials | Total CG iterations | Quality gate |
| --- | ---: | ---: | ---: | --- |
| Dense Cholesky | 3.4422 | 7 | 0 | Pass |
| Dense QR | 11.5485 | 7 | 0 | Pass |
| Scalar Jacobi CG | 4.3528 | 7 | 443 | Pass |
| Block Jacobi CG | 5.0173 | 7 | 406 | Pass |

All four costs were 0.08290751089989, independent gradient infinity norms were
about 1.5e-9, and maximum pose difference from dense was at most 2.12e-11.
In this small sample, block Jacobi reduced CG work but increased median solve
time. This demonstrates why iteration counts alone do not establish speed.

The GTSAM 4.2.1 wheel's native run preserved the anchor and matched the same
scalar objective both initially and finally, but ended at cost
0.08291675444042 with independent gradient infinity norm 3.5769e-3 and maximum
pose difference from dense 1.0793e-3. It fails the cost, gradient and pose gates.
Its five optimizer iterations cannot be equated with seven ropt LM trials.

An untimed diagnostic compares this wheel's native factor linearization with
central differences of that factor's own `unwhitenedError` under GTSAM Pose2
retractions. Rotated, off-solution probes show maximum Jacobian entry difference
1.247194. Residual-coordinate probes agree to 1e-12. This identifies a derivative
discrepancy in this exact native baseline; it does not establish behavior for
other GTSAM builds, factor types or problems, or isolate a sole cause of its
stopping behavior. No cross-library speed conclusion is drawn from this run.

The exact-Jacobian native configuration now has validated derivatives and a
passing common quality gate; see its comparison above. Remaining work includes
larger complete public graphs/BA datasets, representative
robust/noisy cases, peak-memory measurement and comparable Ceres/g2o runs.

## Historical kernel reproduction

The [recorded Intel replay](REAL_ISAM2.md) uses complete pinned real-data graphs,
an explicitly shared column-normalized gradient criterion and all-prefix native
pose/cost checks. It has a separate adapter and runner; its stopping measure is
different from the historical synthetic absolute-gradient contract below.

The archived [grouped QR rebuild](QR_REBUILD.md) beats the verified native iSAM2
configuration at the unchanged common-accuracy target on the 100/300-arrival
synthetic streams, with a second independent 300 campaign. This is scoped evidence,
not a general SLAM or native default-policy claim.
The preceding [iSAM local-coordinate reuse](ANCHOR_COORDINATES.md) compares the
cached policy, with unchanged independent quality checks and exact full results.
Its approximate cached prefixes do not support a native speed comparison.
The [contiguous QR column checks](QR_COLUMNS.md) retain an earlier archived
before/after/native timing campaign. The preceding [aligned QR reductions](QR_REDUCTIONS.md)
remain an archived comparison. Timing claims in older documents refer to archived
kernels, not arbitrary comparisons against the latest checkout. Restore both
adjacent source stages and verify their fingerprints against the saved metadata
before attributing a timing difference to one historical change. The baseline
helper reconstructs earlier source kernels by reversing later patches first.
Older-stage source checks do not imply that every old kernel was rebuilt in the
latest task; each campaign states which binaries were built and tested.

For example, the archived Schur update-cursor after kernel is restored by
`--before-qr-reductions`, and its before kernel by `--before-update-cursor`.
The rejected normal-pattern helper accepts an explicit predecessor with
`--source`; its documented comparison driver hashes both supplied source trees.
Recorded data and timing gates are preserved rather than replaced by new runs.
