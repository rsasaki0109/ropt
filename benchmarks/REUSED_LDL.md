# Reusable LDL and explicit analytic SE2 performance

The initial sparse-direct run rebuilt maps and elimination order on each LM
trial. Profiling the 300-pose workload showed about 51 ms in graph evaluation
out of 133 ms total. Two changes address these separate costs: reusable symbolic
and numeric sparse LDL, and an explicitly selected analytic SE2 derivative path.
The AD path and default solver/derivative selections remain available unchanged.

## Implementation and accuracy contracts

`SparseLdlSymbolic` builds ordering and lower-triangle fill for the exact stored
CSR shape, row offsets and column indices. `SparseLdlWorkspace` retains that
analysis and continuous numeric/scratch arrays. LM creates one workspace per
optimization call. Every linear trial assembles **current** weighted values and
refactors with current damping. An exact structural change triggers new analysis;
no hash collision, stale numeric factor or global/shared cache is used. Rejected
nonlinear trials still preserve the accepted model/state.

Normal assembly and right-looking Schur updates can use cached destination
indices. Their combined optional index storage is bounded by `max_entries`.
If an index plan would exceed that budget, that phase uses searches in the same
stored factor pattern. This is an exact implementation choice, not fill pruning,
a tolerance relaxation or a different matrix. Assembly and factor fill retain
their separate lower-triangle entry limit, including diagonals/structural zeros.
The limits do not bound peak bytes, the input Jacobian or all container overhead.
Arithmetic/update order is preserved; a regression compares planned and search
paths bit for bit. A failed numeric refactor invalidates unchecked factor solves.

`BlockMinimumDegree { block_size }` performs greedy ordering on contiguous
equal-size tangent chunks, then keeps each chunk's scalar coordinates together.
For this SE2 graph the chunks are three coordinates per active pose. It reduces
ordering graph size and can change fill. It is not AMD/COLAMD, does not infer
heterogeneous layouts and requires a positive size dividing active dimension.
Scalar minimum degree remains the default. Tests compare block ordering with
dense Cholesky, enforce contiguous coordinates, demonstrate lower star-graph
fill, and check invalid dimensions/sizes.

`RelativePose2::linearize` provides exact analytic right-tangent derivatives of
the same g2o translation/principal-angle residual. It includes both pose blocks
and the local chart chain rule. Tests compare all six columns against forward AD
and manifold central differences after correlated whitening, including principal
angle wrap neighborhoods away from the cut. Squared/joint-Huber graph solves
agree with AD across dense, QR and sparse LDL. The production selector is
`PoseGraph2::build_with_jacobians(loss, PoseJacobian::Analytic)`; `build` stays AD.

No inverse, extra jitter, numerical differentiation fallback, altered loss or
looser stopping rule is introduced. The checked solve still verifies the true
original `rhs-(JᵀJ+lambda*D²)*step` residual, with relative/absolute tolerances
1e-10/1e-12. Normal-equation conditioning, positive unpivoted pivots and scalar
numeric factorization remain limitations. There is no supernodal/parallel solver.

## Reproduce

Prepare inputs and native GTSAM using the [exact-reference workflow](gtsam_exact/README.md).
The new native executable accepts an optional final `qr` or `cholesky` argument;
its default remains multifrontal QR. Both use exact native BetweenFactor
Jacobians, COLAMD, the same LM settings and exactly fixed anchors. Its options
and diagnostics record the selected backend. Rebuild the executable and generate
a new manifest when using the updated reference source.

```sh
cargo build -p ropt --example benchmark_pose_graph --example profile_pose_graph --example profile_ldl --release --locked
python3 benchmarks/make_ldl_baseline.py --output /tmp/ropt-ldl-before
benchmark_cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c "$benchmark_cpu" \
  python3 benchmarks/paired_ldl.py \
  --before /tmp/ropt-ldl-before/target/release/examples/benchmark_pose_graph \
  --before-source /tmp/ropt-ldl-before \
  --after target/release/examples/benchmark_pose_graph \
  --gtsam /tmp/ropt-native-solvers-build/gtsam_exact \
  --build-manifest /tmp/ropt-native-solvers-manifest.json \
  --input /tmp/w100-prefix300.g2o --output /tmp/ropt-ldl-paired300 \
  --rounds 12 --include-cholesky
```

Use new directories outside the checkout. Repeat with the original 30-pose and
derived 100-pose inputs. The baseline builder copies a benchmark-only snapshot
and reverses the [saved patch](ldl_reuse.patch), restoring the original map-based
LDL and AD-only benchmark path. It does not alter user files or create a Git
worktree. Offline baseline creation was exercised end to end, and its source
fingerprint exactly matched the measured baseline. It is a benchmark snapshot,
not a complete historical release or a separately validated documentation build.

Six candidates run in each of twelve rounds. Order rotates through all six
positions for the first six rounds, then reverses those rotations for the next
six. Each candidate occupies every position twice. A ropt process makes one
untimed consistency-reference solve; a native process discards one warmup solve.
Each then measures one complete optimization call. A separate dense AD process
supplies the independent pose-consistency reference and is excluded from the
comparison samples. Across three sizes, **216 comparison samples** are retained,
without outlier deletion. Graph parsing/construction, initial cloning, validation
and output are outside the timer. Evaluation, whitening, normal assembly,
ordering/factorization, nonlinear trials and report construction are inside.
Native optimizer construction and optimization are timed together.

Every recorded solution must preserve the exact anchor, agree with dense cost
within `1e-9*max(1,abs(reference_cost))`, have independent stored-coordinate
central-difference gradient infinity norm <=1e-5, and agree with dense poses
within 1e-6 in translation/wrapped-angle norm. Reported initial/final objectives
are checked against scalar arithmetic on original input edges. Derivative probes
also check the native exact-Jacobian build. The dense reference is not an
independent global-optimality oracle. Different retractions, damping/stopping
policies and linear methods remain part of the configured comparison.

Recorded machine: AMD EPYC 9V74, one logical CPU affinity, single-threaded
environment settings, default Rust release profile, GTSAM 4.2.1 built with
`GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR=ON` and no native-architecture/TBB/MKL
flags. Frequency and cloud co-tenancy are uncontrolled. Dataset provenance,
build/source/driver/executable fingerprints, options and dispersion are saved.

## Recorded comparison

Median complete-solve times in milliseconds. All twelve solutions per candidate
pass every independent quality gate at every size:

| Poses / edges | Original LDL + AD | Reused LDL + AD | Reused LDL + analytic | Block ordering + analytic | GTSAM QR | GTSAM Cholesky |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 / 59 | 2.9343 | 1.9937 | 0.8952 | **0.7247** | 1.5979 | 1.2844 |
| 100 / 300, derived | 26.5758 | 13.5567 | 6.2483 | **4.7335** | 7.3652 | 5.8144 |
| 300 / 721, derived | 121.5904 | 55.7507 | 23.0718 | **20.3039** | 85.1885 | 66.7478 |

The [30](results/reused_ldl/poses30/report.json),
[100](results/reused_ldl/poses100/report.json) and
[300](results/reused_ldl/poses300/report.json) reports include all raw samples,
per-round quality checks and source/build identity. Final objectives are
0.08290751089989, 0.56891259146098 and 1.05375815353753. All four ropt candidates
use the same 7/8/16 nonlinear trials at these sizes; native QR and Cholesky use
6/6/19 optimizer iterations. Those counts are not identical units of work.

Keeping AD and scalar ordering, reuse reduces median solve time about 32%, 49%
and 54% at the three sizes. That isolates the implementation improvement from
the derivative/backend choices. AD reuse still takes longer than GTSAM at 30
and 100 poses. Explicit analytic derivatives and block ordering then produce
lower medians than **both configured native references** at all three sizes:
about 1.77x, 1.23x and 3.29x faster than the faster native configuration.
This establishes these configured runs, not general GTSAM superiority or SOTA.

The 100/300 graphs are explicitly derived between-factor subgraphs, retaining
only prefix vertices/within-prefix edges and omitting source `EQUIV` constraints.
They are not complete original SLAM tasks. Their transformation/checksums are
recorded and no third-party graphs/optimized poses are vendored. BA, robust
performance, iSAM, fixed-lag, complete large datasets and other GTSAM solver
settings remain outside this comparison. No peak/live/RSS memory is measured.

The current explicit fast g2o configuration is:

```sh
cargo run -p ropt --example pose_graph_g2o --release --locked -- \
  --input input.g2o --ldl --analytic --block-ordering
```

## Allocation/phase diagnostics

`profile_ldl` prepares one symbolic/numeric workspace, then refactors and solves
100 times with alternating damping 1e-3/2e-3 and unchanged Jacobian. These
prepared `refactor`/`solve_factored_into` calls make **zero allocation requests**
and only one symbolic analysis. An original-system residual check follows outside
the counter. The unchecked triangular solve is not presented as a residual
certificate; full checked solves still allocate the original operator's diagonal.
Graph evaluation, changed-pattern analysis and owned output vectors also allocate.

```sh
taskset -c "$benchmark_cpu" target/release/examples/profile_ldl /tmp/w100-prefix300.g2o
taskset -c "$benchmark_cpu" target/release/examples/profile_pose_graph /tmp/w100-prefix300.g2o
```

On the initial 300-pose matrix, scalar ordering stores 12,908 factor entries,
23,533 assembly destination indices and 111,776 Schur destination indices.
Those plans fit the shared one-million-index budget. The
[instrumented diagnostics](results/reused_ldl/diagnostics) record request counts
and phase times separately from the formal benchmark. Requested bytes include
full successful realloc requests, not freed/live/peak memory. The System allocator
wrapper remains isolated to diagnostic examples; no unsafe code or allocator is
added to the library. CI checks the zero-allocation prepared loop on the authored
offline fixture and independently checks all LDL orderings/analytic selection.

Earlier [map-based LDL](SPARSE_LDL.md) and [buffered PCG](BUFFERED_PCG.md) reports
remain historical records. Comparing their separate runs is not a paired
improvement estimate; use the alternating measurements above for this change.
