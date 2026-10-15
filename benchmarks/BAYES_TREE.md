# Selective Gaussian Bayes-tree updates

`GaussianBayesTree` and `IsamSmoother::with_bayes_tree()` provide working
fixed-order, scalar-frontal Gaussian partial re-elimination. New observations
rebuild touched conditionals and their ancestor paths. Unaffected conditional
rows and symbolic separators stay shared through immutable `Arc`s. This extends
cached iSAM updates. This document describes the original append-only backend
and its frozen measurements.
The later [threshold-selected factor mode](SELECTIVE_RELINEARIZATION.md) adds
incident-factor reevaluation while rebuilding the entire Gaussian QR/tree.
The subsequent [editable Bayes-tree mode](EDITABLE_BAYES_TREE.md) adds cached
separator messages, original-factor replacement/deletion, and an explicit
selected nonlinear policy with affected Gaussian clique re-elimination.

## Algorithm and guarantees

The forest is constructed from sparse Givens QR. Each scalar clique stores one
frontal row, its right-hand side and a symbolic separator. The first separator
column is its parent. Symbolic fill enforces running intersection even when
numeric cancellation removed an R coefficient. Regressions test such a
cancellation, the actual shared identity of an orphan, and independent original
model stationarity.

Each nonzero input column marks its clique and ancestors. The selected old
conditional rows and new measurement rows form a smaller sparse QR problem,
retaining elimination order. Its rows replace only the marked conditionals;
constant residual energy is added to the retained constant. Retaining conditional
energy avoids replaying or double-counting unaffected subtree observations. Only
affected symbolic separators are extended. Old symbolic fill is conservatively
retained so rounded cancellations cannot remove dependencies. This preserves
the full quadratic objective, not just its minimizer.

There is one scalar frontal per clique; variable blocks and supernodes are not
amalgamated. QR is unpivoted with the existing relative R-diagonal rank cutoff,
not a rank-revealing certificate. `max_entries` applies separately to numeric R
and scalar-frontals-plus-symbolic-fill; it is not a peak-memory bound. Symbolic
fill may exhaust the budget before numeric R.

Back-substitution, rank checks and coordinate/clone bookkeeping still visit the
full dimension. A late constraint may affect the whole forest. Gaussian append
does not replace old factors or change anchors. Dynamic reordering, row removal,
selective nonlinear factor relinearization, incremental fixed-lag marginalization
and wildfire back-substitution remain unfinished.

The default iSAM backend remains incremental QR. Selecting the tree preserves
threshold, interval and forced **full batch nonlinear refreshes**, rank checking,
optional batch ordering/refinement and transactional state commit. A successful
refresh consumes its QR into a fresh tree. Cached appends evaluate only new
factors; failed updates preserve owned state.

```rust
use ropt::isam::{IsamOptions, IsamSmoother};
let smoother = IsamSmoother::new(IsamOptions::default())?.with_bayes_tree()?;
```

`IsamReport::bayes_tree_update` is `None` for the default backend. Tree reports
count scalar `reeliminated_cliques` and `reused_cliques`. `input_rows` counts
selected old conditional rows plus new rows; construction from a QR snapshot
counts its triangular rows, not batch callbacks or original observations.

## Independent checks

Gaussian forest tests match independent SVD solutions and full original energies
at scales `1e-80`, `1`, `1e80`, including late cross-branch constraints and
constant residuals. Local observations rebuild 2 of 12 cliques; a cross-branch
observation rebuilds 4 of 12. Tests cover permuted/extended columns,
nonsequential keys, missing rank, overflow, dimensions, numeric/symbolic fill
limits and successful updates after failures. Even a tree conversion failure
after batch optimization cannot commit variables or factors.

The **300-arrival nonlinear SE2** validation retains the existing noisy
measurements, analytic manifold Jacobians, delayed old-to-old loops, independent
central-difference stationarity/original-energy checks and AD/dense-QR reference
solutions. All gates pass. Cached updates remain approximate; checkpoints follow
full nonlinear refreshes. See [all-prefix quality](results/bayes_tree/se2_quality.csv)
and [summary](results/bayes_tree/se2_summary.txt).

Complete default Intel943 and MIT1228 replays preserve every estimate and
non-timing report field exactly against their immediate predecessors.
[Default regression](results/bayes_tree/default_regression.json).
Historical recorded-data speed campaigns and media remain frozen.

## Matched Gaussian measurement

The comparator uses the **same current-build executable** in QR and tree modes.
Existing QR already rotates sparse pivot paths; it is not a full-rebuild baseline.
Both modes time public atomic append, full solve and cached cost. Original-model
checks and output are excluded; allocation instrumentation is disabled.

The deterministic fixture has four branches with 256 leaves and one root each
(**1,028 scalar columns**), and 120 updates: local observations, constant residuals
and cross-branch measurements every 13th update. Each process performs one
complete untimed warm-up and one timed stream. Six AB/BA pairs per campaign run
serially on CPU 0 with one thread after all builds/tests finish.

The original absolute gradient gate is `1e-8`; reported original cost agreement
is `1e-10 * max(1,cost)`. Complete finite estimates agree across modes within
`2e-9` in infinity norm at every prefix, and original costs within
`1e-10 * max(1,cost)`. These scalar Gaussian gates differ from the recorded SE2
normalized-gradient contract.

Both campaigns retain complete compressed estimates, raw samples, environment,
source/executable fingerprints and all-prefix agreement:
[primary](results/bayes_tree/primary/report.json),
[confirmation](results/bayes_tree/confirmation/report.json).

| Campaign | QR median | Tree median | Median paired QR/tree | Tree wins |
|---|---:|---:|---:|---:|
| Primary | 5.040 ms | 3.460 ms | 1.481x | 6/6 |
| Confirmation | 5.230 ms | 3.587 ms | 1.454x | 6/6 |

The tree re-eliminates a median **6 of 1,028** scalar cliques (range 2–14).
Every original-model and paired gate passes; maximum estimate gap is
`3.331e-16`, original cost gap `1.777e-15`. These are short Gaussian kernel
streams, so their timing ratios should not be extrapolated to end-to-end systems.

This synthetic Gaussian measurement does not establish speedups on nonlinear
SE2, 3D, VIO, fixed-lag smoothing, native GTSAM or production systems. Peak memory
is not measured.

```sh
cargo test -p ropt --test bayes_tree --test isam_bayes_tree --locked --offline
cargo build -p ropt --example benchmark_bayes_tree \
  --example validate_isam_se2 --release --locked --offline
target/release/examples/validate_isam_se2 300 --bayes-tree
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_bayes_tree.py \
    --executable target/release/examples/benchmark_bayes_tree \
    --output /tmp/bayes-tree-primary --branches 4 --leaves 256 --updates 120 --rounds 6
```

[bayes_tree.patch](bayes_tree.patch) captures the production change.
`make_ldl_baseline.py --before-bayes-tree` restores the immediate QR-only
predecessor. All **28** zero-fuzz source rollback stages match archived
fingerprints; older builds were skipped in these source-only checks.
[Rollback records](results/bayes_tree/rollback_stages.json),
[verification](results/bayes_tree/validation_checks.json).

Verification passes **260 Rust unit/integration tests + 18 doctests**, 18 Python
tests, formatting, Clippy and rustdoc with warnings denied.

Subjective maturity stays **75%**. Gaussian partial re-elimination is usable;
selective nonlinear iSAM2 and field validation remain open.
