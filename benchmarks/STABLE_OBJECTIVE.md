# Stable true-loss reductions on large pose graphs

The previous [large-graph sweep](LARGE_POSE_GRAPH.md) fixed an LDL residual
failure but retained a 3,000-pose accuracy failure: cost stopping left pose
disagreement 9.10e-6 against a native QR consistency reference. Tightening the
cost tolerance to 1e-15 instead produced `NoProgress`.

An iteration trace exposed subtraction of rounded aggregate objectives as a
separate numerical problem. Near convergence, reductions around 1e-17 could be
reported as about 1e-13, producing gain ratios in the thousands, spurious
acceptance/rejection and increased damping. This change computes reductions
from individual true losses without subtracting large aggregate costs.

## Computation and contracts

Objective values and signed reductions use Neumaier compensated summation.
Linearization retains one pre-IRLS norm per scalar fallback term or explicit
joint-loss block. For each old/new norm `u,v`:

- Squared loss changes by `(u-v)*(u+v)/2`.
- Huber changes use the linear difference outside the quadratic region and
  separate quadratic/linear pieces across the threshold.
- Cauchy changes use `log1p` of the normalized squared-norm change. The larger
  norm is placed first so a negative ratio cannot round to -1. In the small-norm
  regime the scale squared is canceled algebraically, using the continuous
  `log1p(z)/z = 1` value at zero to avoid intermediate underflow at large scales.

These are algebraic forms of the existing true losses. They are not frozen-IRLS
objective changes or first-order approximations. Acceptance still requires
positive actual reduction and positive gain ratio; the predicted model,
Jacobians, damping policy, fixed state, rejection behavior and tolerance checks
retain their existing meanings. `Iteration::actual_reduction` exposes the value
used for gain/stopping decisions. Rounded `cost` and `trial_cost` can be identical
despite a nonzero reduction; their direct difference is no longer the gain
numerator. GN also uses these true-loss reductions.

Each live model stores one extra f64 per effective loss term and makes one
additional allocation per evaluation. Norm arrays are not retained in iteration
history. This is an explicit memory/precision tradeoff, not a zero-allocation
full-solver claim. Prepared sparse factor loops keep their existing buffers.

Regressions verify accepted scalar and joint squared/Huber/Cauchy improvements
hidden by large constant costs; rejection with unchanged state for increasing
true losses; GN acceptance with identical rounded totals; Huber thresholds;
Cauchy close norms, extreme ratios and scales; compensated sums and nonfinite
errors. Existing correlated whitening, true robust gain-ratio, fixed-lag,
covariance and manifold tests pass. No Rust dependency, C++ runtime, unsafe
library code or numerical-derivative fallback is added.

## Accuracy and configured measurement

The input is the existing derived 3,000-pose / 16,834-edge between subgraph,
SHA256 `b9e62bab6b28d75b3fc845033cdd84496c5c81fdb7faca670896bcfc0da347c4`.
The converter explicitly omits its 1,684 within-prefix EQUIV records; this is
not the complete original `w10000.graph` problem.

ropt uses analytic SE2 factors, block minimum degree of size three, reused LDL
with unchanged 1e-10/1e-12 original-system residual bounds, explicit cost
tolerance **1e-15**, and existing gradient/step tolerances 1e-10. Default cost
stopping is not silently changed. GTSAM uses its existing exact native
BetweenFactor build, COLAMD, Cholesky and absolute/relative error tolerances
1e-12. Both iteration limits are 300. A separately executed native QR solve
supplies the pose-consistency reference and is excluded from timing samples.
There is no dense/global-optimality reference.

The unchanged gates require an exact anchor, cost agreement within
`1e-9*max(1,abs(reference_cost))`, scalar stored-coordinate central-difference
gradient infinity norm <=1e-5 and pose agreement <=1e-6. Reported objectives and
termination/iteration limits are also checked independently.

The corrected solve uses 25 nonlinear trials, finishes with cost stopping, and
has independent objective 36.75010834401441, numerical gradient about 2.19e-10
and pose difference about **1.01e-7**. All accuracy gates pass. Native QR and
Cholesky use 148 optimizer iterations; these counts are different units of work.
The prior 37-trial `NoProgress` trace and failed accuracy reports remain retained.

The timing sweep compares ropt with Cholesky; QR only supplies the independently
executed consistency reference. Each process discards one complete solve before
measuring one solve. Eight balanced rotated/reversed rounds put both candidates
in both positions four times. All 16 samples are retained without outlier
deletion. Parsing, graph construction, initial cloning, validation and output
are untimed; evaluation, loss arithmetic, factorization, nonlinear trials and
report creation are timed. Native optimizer construction is timed. No other
solver or compiler overlaps the sweep.

The environment remains AMD EPYC 9V74, CPU 0 affinity, single-thread settings,
Rust release defaults and native GTSAM 4.2.1 exact Jacobians without
native-architecture/TBB/MKL flags. Frequency and cloud co-tenancy are
uncontrolled. Peak/live/RSS memory is not measured. Build/source/executable
hashes, options, data provenance, individual quality checks and dispersion are
retained. No external graphs or optimized poses are vendored. This is a
configured comparison on this derived graph, not a general GTSAM/SOTA claim or
an iSAM2/fixed-lag performance benchmark.

## Recorded comparison

Recorded medians: **3,097.63 ms** for ropt versus **12,504.05 ms** for native
GTSAM Cholesky, about **4.04x** on this configured derived graph. All sixteen
individual solutions pass every unchanged gate. See the
[report](results/stable_objective/poses3000/report.json),
[raw samples](results/stable_objective/poses3000/samples.csv) and recorded options.
QR's excluded reference solve is not part of this timing comparison.

| Configuration | Median | Min–max |
| --- | ---: | ---: |
| ropt analytic / block LDL, cost tolerance 1e-15 | 3,097.63 ms | 2,982.17–3,261.07 ms |
| Native exact GTSAM Cholesky | 12,504.05 ms | 11,918.38–13,005.84 ms |

The retained [iteration diagnostics](results/stable_objective/diagnostics/metadata.json)
include before/after CSV and summaries, without optimized poses. They are not
timing measurements. Compared norms, residual callbacks and block-norm
evaluation remain subject to f64 precision.

## Reproduce

Prepare the pinned input and optional native reference/manifest using the
[reference workflow](gtsam_exact/README.md), then:

```sh
cargo build -p ropt --example benchmark_pose_graph --example trace_pose_graph --release --locked
benchmark_cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c "$benchmark_cpu" \
  python3 benchmarks/paired_large.py --input /tmp/ropt-prefix3000.g2o \
  --ropt target/release/examples/benchmark_pose_graph \
  --gtsam /tmp/ropt-gtsam-reference/reference/gtsam_exact \
  --build-manifest /tmp/ropt-gtsam-reference/manifest.json \
  --output /tmp/ropt-stable-results --rounds 8 --native-solvers cholesky \
  --max-iterations 300 --cost-tolerance 1e-15
```

The driver still defaults to both QR and Cholesky for smaller sweeps. Rounds
must be a positive multiple of twice the candidate count. Output paths must be
new and outside the checkout. Run timings separately from compilers, tests,
diagnostics and other solver processes.

The production example accepts the same explicit configuration:

```sh
cargo run -p ropt --example pose_graph_g2o --release --locked -- \
  --input /tmp/ropt-prefix3000.g2o --ldl --analytic --block-ordering \
  --max-iterations 300 --cost-tolerance 1e-15
```

`trace_pose_graph input.g2o NEW_DIRECTORY 1e-15` exports iteration CSV, poses
and summary even for unsuccessful termination. It is a diagnostic, not a
benchmark, and records rounded difference alongside actual reduction.
`make_ldl_baseline.py --before-objective --output /tmp/ropt-before-objective`
reverses `stable_objective.patch` in an isolated benchmark-only snapshot with
the recorded predecessor's source fingerprint. The existing default and
`--before-refinement` stages reverse that patch first, preserving older sources.
