# Sparse LDLᵀ and checked complete-solve measurements

This page preserves the initial map-based implementation and its recorded run.
The later [reusable LDL and analytic SE2 comparison](REUSED_LDL.md) replaces
per-trial ordering/maps with cached structures and records new matched runs.

The buffered-PCG sweep left substantial time in matrix-free linear iterations.
An instrumented solve of one frozen initial normal system at 300 poses spent
18.94 of 25.30 ms in scalar-CG operator calls, or 16.03 of 26.69 ms in block-CG
operator calls. This diagnostic excludes graph evaluation and preconditioner
construction, uses damping 1e-9 and the artificial RHS `cos(0.13*k)`, and does
not represent an entire LM solve. Its [raw counters](results/sparse_ldl/diagnostics/frozen_linear.csv)
motivate trying direct factorization rather than relaxing accuracy criteria.

## Backend and failure contracts

`LinearSolver::SparseLdl(SparseLdlOptions::default())` provides LM with a CSR
Jacobian, sparse lower-triangle normal assembly, deterministic greedy scalar
minimum-degree ordering, unpivoted right-looking LDLᵀ and triangular solves.
The ordering is exact greedy degree selection on the evolving elimination graph,
not AMD or COLAMD. Natural ordering is available explicitly. No dense global
Jacobian/normal matrix, inverse, added jitter, CG fallback or C++ dependency is
introduced. Local factor Jacobians may still be dense. Existing backend defaults
are unchanged, and sparse GN remains unsupported.

The damped matrix is `JᵀJ + lambda*diag(max(diag(JᵀJ),1))`, after the same
whitening and first-order robust IRLS weights as other LM backends. Nonlinear
gain ratios, rejection behavior and convergence rules remain shared. Direct
trials record zero `linear_iterations`; this denotes no iterative linear solver,
not zero factorization work.

The LM path calls `sparse_direct::solve_normal`, which recomputes the true linear
residual with the original matrix-free operator and requires
`||rhs-A*x|| <= max(absolute_tolerance, relative_tolerance*||rhs||)`. Defaults
are 1e-12 absolute and 1e-10 relative. The reusable `SparseLdlFactor::solve`
alone solves the computed factors without checking the original system.
Nonpositive pivots, nonfinite arithmetic, excessive residuals and fill-budget
exhaustion return explicit errors; LM propagates them. Undamped singular systems
can fail even when a damped system succeeds.

`max_entries`, default 1,000,000, bounds assembled lower-triangle entries and
symbolic/numeric factor fill, including diagonals and structural zeros. Fill is
never pruned using a numerical threshold. The budget counts entries, not bytes,
live/peak memory or RSS, and does not bound the input Jacobian. Maps, adjacency
sets, heap and intermediate storage add overhead. Ordering and numeric factors
are rebuilt on every trial; there is no cached symbolic analysis, supernodal
execution, parallel factorization, indefinite pivoting or iterative refinement.
Normal equations still square the Jacobian's condition number, and a small
linear residual alone does not certify forward accuracy.

Tests compare both orderings against independent dense Cholesky, repeated RHS
solves, singular/damped systems, sparse-only custom problems, true-residual
failure, assembly/fill budgets, fixed variables and correlated robust losses.
Shared-calibration BA also checks LDLᵀ alongside the five existing backends.

## Reproduce

Prepare the pinned input files and exact-Jacobian native GTSAM executable using
the [external reference instructions](gtsam_exact/README.md). The 100/300 inputs
are explicitly derived **between-factor subgraphs**, omitting source `EQUIV`
constraints, not complete original SLAM datasets. No third-party graph is
vendored. Source/derived checksums and transformation metadata accompany each
report. Build with the locked dependencies and default release profile:

```sh
cargo build -p ropt --example benchmark_pose_graph --example sparse_ldl --example profile_linear --release --locked
benchmark_cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c "$benchmark_cpu" \
  python3 benchmarks/exact_pose_graph.py \
  --input /tmp/w100-prefix300.g2o \
  --ropt target/release/examples/benchmark_pose_graph \
  --gtsam /tmp/ropt-native-reference-build/gtsam_exact \
  --build-manifest /tmp/ropt-gtsam-exact-manifest.json \
  --backends dense,sparse_ldl,scalar_cg,block_cg \
  --output /tmp/ropt-ldl-results300 --repeats 10 --warmup 3
target/release/examples/sparse_ldl /tmp/w100-prefix300.g2o
taskset -c "$benchmark_cpu" target/release/examples/profile_linear /tmp/w100-prefix300.g2o
```

Repeat the runner with the original 30-pose and derived 100-pose inputs, each
using a new output directory. `sparse_ldl_natural` is another explicit backend
selector for diagnostic comparison; the recorded sweep uses minimum degree.
An untimed dense solve supplies a consistency reference. After three discarded
warmups, each candidate has ten measured solves; the first ropt backend rotates
each round. Native GTSAM runs sequentially after ropt, not interleaved with it.
The three sizes produce 150 recorded measurements. No sample is discarded.

The timer covers `solve_local`, including evaluations/AD, whitening, assembly,
ordering/factorization, damping trials and report construction. Parsing, graph
construction, initial cloning, serialization and independent validation are
outside it. GTSAM times native optimizer construction and optimization. All
options, machine/thread settings and source/binary fingerprints are saved.
The cloud CPU is AMD EPYC 9V74; affinity is one logical CPU. Frequency and
co-tenancy are uncontrolled, and no peak memory is measured.

The unchanged quality gates require exact anchors, independent cost agreement
within `1e-9*max(1,abs(reference_cost))`, stored-coordinate central-difference
gradient infinity norm <=1e-5 and per-pose translation/wrapped-angle difference
<=1e-6. The dense consistency reference is not a global-optimality certificate.
CG uses relative/absolute linear tolerances 1e-10/1e-12, matching the checked
LDL defaults. GTSAM 4.2.1 uses its exact native between-factor Jacobians with
`GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR=ON`, COLAMD multifrontal QR and the
recorded LM settings. Retraction, damping and stopping policies differ.

## Recorded results

Median complete-solve times in milliseconds; all five candidates pass every
common quality gate at every size:

| Poses / edges | ropt dense | ropt LDLᵀ | ropt scalar CG | ropt block CG | Native GTSAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 30 / 59 | 3.4009 | 3.3497 | 3.0430 | 3.2781 | 1.8648 |
| 100 / 300, derived | 63.7211 | 26.9477 | 29.1294 | 29.1693 | 7.5563 |
| 300 / 721, derived | 2173.8531 | 120.9451 | 395.8322 | 396.8063 | 94.3395 |

The [30](results/sparse_ldl/poses30/report.json),
[100](results/sparse_ldl/poses100/report.json) and
[300](results/sparse_ldl/poses300/report.json) reports include raw CSV samples,
dispersion, quality checks and provenance. The final objectives are
0.08290751089989, 0.56891259146098 and 1.05375815353753. At 300 poses LDLᵀ has
independent gradient infinity norm 1.89e-9 and maximum pose difference from
dense 1.39e-12, with 16 nonlinear trials. Scalar/block CG use 17/16 trials and
11,055/9,144 linear iterations. These counts describe different algorithms,
not identical units of work.

LDLᵀ is about 3.3 times faster than either CG backend on this configured
300-pose subgraph, while CG is slightly faster at 30 poses. Native GTSAM remains
faster at all three sizes; LDLᵀ takes about 1.28 times as long at 300 poses.
These measurements do not establish general speed superiority or SOTA, and
comparisons with earlier historical sweeps are not paired improvement estimates.

For the **initial** 300-pose Jacobian (897 tangent columns, 8,587 CSR entries),
the assembled normal lower triangle has 6,750 stored entries. With damping
1e-3, factor storage including the diagonal is 36,480 entries in natural order
versus 12,908 with minimum degree. Both complete example solves converge to
the same cost in 16 trials. The [storage diagnostic](results/sparse_ldl/diagnostics/initial_factor_storage.csv)
counts entries at one linearization, not peak memory throughout optimization.
An independent star-graph regression checks ordering benefits and ensures the
same budget can accept minimum-degree fill while rejecting natural fill.

Next priorities include reusable symbolic analysis, complete larger datasets,
BA/robust performance cases, peak-memory measurements and Ceres/g2o references.
