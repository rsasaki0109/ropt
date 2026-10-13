# Accepted Gaussian model and factor input reuse

This follows the [sorted QR / native iSAM2 comparison](NATIVE_ISAM2.md).
The solver still performs full nonlinear refreshes and undamped QR rebuilds;
this change removes redundant evaluation and input allocation, not Bayes-tree work.

The later [explicit damping reuse policy](DAMPING_REUSE.md) has its own matched
sweep. The measurements below remain the historical same-policy comparison.

## Diagnosis

A separate instrumented 300-arrival run of the previous kernel attributed
about 65% of whole update time to batch optimization and 26% to QR insertion.
Factor evaluation accounted for about 21%, across batch and post-solve passes;
checked LDL solves accounted for 22% within batch optimization, with symbolic
analysis nested within LDL.
Graph construction was only about 1%. These inclusive scopes overlap and must
not be added. The run was unwarmed, includes timer/counter overhead, and is
diagnostic evidence rather than a primary timing comparison. Its means match
the unmodified predecessor exactly. [Raw profile](results/model_reuse/diagnostics/profile.json).

## Changes and preserved contracts

`FactorGraph` now validates each ambient block through a borrowed vector view,
including disconnected/fixed blocks, instead of allocating a temporary owned
copy. Validation dimensions, finite checks and quaternion norm arithmetic are
unchanged. Each graph evaluation retains reusable owned vectors by callback
slot, resizing only when the ambient dimension changes. Each callback receives
only its current arity; numerical differentiation clones just that slice.
Storage is local to one evaluation, with no shared mutable cache or lifetime
change. Public callback signatures and owned outputs are unchanged.

An internal solver path returns the final **accepted** linearization along with
the ordinary report. iSAM consumes its whitened residual/Jacobian for QR rather
than invoking all factor callbacks again at the same state. This path rejects
robust loss metadata: only squared global/factor losses have unit IRLS weights
and give the required Gaussian model. Sparse Jacobians move directly; dense
ones are converted to canonical CSR with exact-zero removal. The cached factor
contains no damping or frozen robust surrogate. Public solver reports/options,
LM acceptance, termination and ordering remain unchanged.

Callbacks are still evaluated during optimization; there is no promised callback
count for general nonlinear solves. The unnecessary final callback pass has
been removed. Functions must provide consistent residuals/Jacobians for a supplied
state; external callback side effects are not rolled back. The old injected
final-pass callback failure test is replaced by a real batch-trial failure test.
Final undamped rank/fill checks still run after a successful batch solve and
preserve owned smoother state on failure.

New tests alternate callback arities/dimensions across analytic and explicit
numeric factors, compare dense/CSR assembly, and reject invalid fixed/disconnected
manifold blocks before any callback. Dense-QR and sparse-LDL refresh tests prove
that an already stationary graph needs just one evaluation, preserves noise
scaling, and supplies the correct cached posterior for the next
update. Nonlinear overshoot/rejection tests verify subsequent cached solves use
the converged model, not the initial or rejected trial. A stationary coupled
model exceeding final QR fill still fails atomically; the next update confirms
cache, key/history and interval recovery. Existing mixed SO3/SE3 layout and
Gaussian/robust/numerical derivative tests remain passing.

## Same-condition measurements

The input, seeds, root conditioning, algorithms, output scope and all-prefix
accuracy gates are exactly those of the prior native comparison. Both ropt
candidates use the same benchmark template and solver options; native uses the
same verified GTSAM 4.2.1 executable/library. Six rotated/reversed rounds per
size each discard one complete warm-up stream and measure one stream.
There are **36 measured streams / 7,200 measured arrivals**; every original
gradient, mean/cost agreement, fixed-root and reported-cost gate passes.
Before/after ropt means and original costs are identical at every prefix.

Median sum of full update/report-call times in milliseconds:

| Added poses | Sorted-QR predecessor | Model/input reuse | Native iSAM2 |
| --- | ---: | ---: | ---: |
| 100 | 111.797 | **99.611** | 45.184 |
| 300 | 1,393.241 | **1,238.583** | 431.180 |

The new change reduces aggregate time by **10.9% / 11.1%** (about 1.12x).
Native iSAM2 remains **2.20x / 2.87x faster** in this accuracy-conditioned
synthetic family. It is not either library's default incremental policy,
a bounded-window or real-time guarantee, or general SLAM superiority evidence.
The full-history/QR rebuild architecture remains a material gap.

Timing uses CPU 0 and single-thread settings with no concurrent compiler/test/
solver. Allocator counting is disabled, and validation, output and report
destruction are outside clocks. Cloud frequency/co-tenancy remain uncontrolled.
Old and new timing sweeps are retained separately, not pooled or substituted.
Reports preserve every-prefix quality, raw samples/update times, exact input,
losslessly compressed first-round full means, options and all source/template/
executable/library fingerprints:
[100](results/model_reuse/frames100/report.json),
[300](results/model_reuse/frames300/report.json).

A separate 300-arrival allocation diagnostic reduces requests from **8,937,704
to 6,574,152** (26.4%) and requested bytes from **2,212,836,126 to 1,953,361,086**
(11.7%), with identical means. These are allocation traffic, not live/peak/RSS
memory or native memory comparisons. Its instrumented times are excluded.
[Allocation evidence](results/model_reuse/diagnostics/allocations.json).

## Reproduce

Use the same optional native executable/verified manifest workflow as
[NATIVE_ISAM2.md](NATIVE_ISAM2.md). Rust remains C++-independent.

```sh
cargo build -p ropt --example benchmark_isam_se2 --release --locked --offline
python3 benchmarks/make_ldl_baseline.py --before-model-reuse \
  --output /tmp/ropt-before-model-reuse
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_isam2.py \
  --before /tmp/ropt-before-model-reuse/target/release/examples/benchmark_isam_se2 \
  --before-source /tmp/ropt-before-model-reuse \
  --after target/release/examples/benchmark_isam_se2 \
  --native YOUR_NATIVE_ISAM2_BINARY --native-manifest YOUR_VERIFIED_MANIFEST.json \
  --frames 300 --rounds 6 --output /tmp/model-reuse-comparison
```

Finish builds before timing, choose a permitted CPU, and use new output paths.
The helper reverses the later `qr_rebuild.patch`, then `anchor_coordinates.patch`, then `qr_columns.patch`, then `qr_reductions.patch`, then `update_cursor.patch`, then `qr_norm.patch`, then `symbolic_sorted_merge.patch`, then `qr_aligned.patch`, then `normal_pattern_rebind.patch`, then `ordering_adjacency.patch`, then `whitening_buffers.patch`, then `qr_inplace.patch`, then `graph_retraction.patch`, then `canonical_csr.patch`, then
`qr_merge_buffers.patch`, then
`symbolic_pattern.patch`, then
`isam_damping_reuse.patch`, then
`isam_model_reuse.patch` before older stage patches,
preserving historical kernel fingerprints. For the optional before-only stage
diagnostic, first replace the snapshot's `crates/ropt/examples/benchmark_isam_se2.rs`
with `benchmarks/templates/benchmark_isam_se2_original.rs` from this checkout.
This retains the original monolithic runner used by the saved profile patch;
the current runner delegates to shared code. Then apply
`isam_refresh_profile.patch` to that separate before-model-reuse snapshot,
rebuild the example and run a single stream without warm-up. It
prints inclusive stage counts/times to stderr. Never use the instrumented
executable as a primary candidate. CI retains offline correctness checks,
not timing thresholds.
