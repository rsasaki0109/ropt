# Direct canonical CSR assembly

This follows [QR processing](QR_MERGE_BUFFERS.md). Both candidates retain those
changes, compact symbolic columns and explicit adaptive LM damping reuse. This
change replaces graph/dense-model triplet assembly while preserving algorithms,
update policies, tolerances and the optimization trajectory.

## Assembly and contracts

`FactorGraph::evaluate_sparse` previously emitted row/column/value triplets in
callback-block order, sorted the entire matrix, built another triplet vector to
combine duplicates, then copied into CSR arrays. Registration rejects repeated
variable IDs, active tangent ranges are disjoint, and each factor occupies its
own residual rows, so these graph entries never overlap.

Assembly now keeps a local reusable list of active callback slots, sorts those
slots by variable index and emits each residual row directly into column/value
arrays. The tangent layout follows variable insertion order, including after
fixed flags change. Callback arguments and Jacobian block positions retain the
caller's order; whitening runs on the original blocks before assembly. Fixed and
disconnected values and every callback block remain validated. Fixed blocks are
omitted only from the assembled columns.

The internal checked `CsrMatrix::from_canonical` handoff validates array lengths,
row count, first/last and monotone offsets, column bounds, strictly increasing
indices within each row, finite values and removal of exact zeros. Empty rows
and zero-column matrices are supported. It exposes no unchecked public constructor
and uses no unsafe code. Public `from_triplets` retains its original sorting,
duplicate-sum order, cancellation and error behavior.

The accepted Gaussian model's dense-to-CSR conversion also emits row-major arrays
directly. Its whitening, squared-loss guard and final undamped QR rank/fill/solve
checks remain unchanged. Only exact zeros are omitted: tiny nonzero derivatives
remain stored even when their squared norms underflow. No layout cache persists
between evaluations.

Two graph regressions compare every CSR entry and residual with dense assembly
plus the original public triplet constructor. They cover reversed callback IDs,
varying arity/dimensions, SO3 ambient/tangent dimensions, correlated whitening,
fixed-layout changes, disconnected columns, all-fixed graphs, signed zero,
changing exact-zero derivatives and `1e-200` coefficients. An internal handoff
regression checks independent products and rejects malformed storage, unordered/
duplicate/out-of-range indices, zeros and nonfinite values. There are **194
unit/integration tests + 18 doctests**; Clippy, formatting and nine Python
quality-oracle tests pass. Existing dense/sparse Gaussian handoff, analytic/AD/
numeric derivatives, robust losses and transactional failures remain covered.

## Matched measurements

The [native comparison](NATIVE_ISAM2.md) specifies the near-truth synthetic SE2
input and verified GTSAM 4.2.1 configuration. Before/after candidates use identical
input bytes, shared runner sources, sparse LDL/block ordering and explicit adaptive
LM damping reuse. Every arrival forces a full nonlinear refresh. Initial damping
is `1e-3`, gradient tolerance `5e-7`, cost tolerance `1e-15`, step tolerance `1e-10`
and the trial limit 200. Native executable/library hashes are unchanged. This is
an accuracy-conditioned workload, not either library's default incremental policy.

Twelve rotated/reversed rounds place each candidate in each position four times.
Each process performs one discarded warm-up stream and one measured stream.
There are **72 measured streams / 14,400 timed arrivals**. All-prefix original
gradients must be at most `1e-6`; the fixed root must match exactly. Native pose
agreement stays within mixed metre/radian norm `2e-6`, original costs within
relative `1e-9` and reported costs within `1e-10`. No gate is relaxed and no
failed sample or timing outlier is removed.

The driver also requires identical before/after means, original/reported costs,
batch trial counts, damping telemetry and QR entry counts at every prefix.
Both take **213 / 631 LM trials** at 100 / 300 arrivals. First-round full mean
CSVs match byte for byte. Maximum original gradients remain `2.780e-7 / 2.964e-7`,
and native pose differences `1.422e-9 / 3.442e-8`.

Median sum of complete update/report-call times, milliseconds:

| Added poses | Triplet assembly | Direct CSR | Native iSAM2 |
| --- | ---: | ---: | ---: |
| 100 | 58.355 | **52.353** | 44.170 |
| 300 | 823.417 | **761.220** | 431.156 |

Observed median reductions are **10.3% / 7.6%**. Native remains **1.19x / 1.77x
faster**. All twelve paired rounds improve at 100; one is 0.9% slower after the
change at 300. Standard deviations over retained samples (denominator N) are
2.9 / 2.0 ms before/after at 100 and 16.0 / 30.1 ms at 300. These observations
do not establish per-update, real-time or general-workload guarantees. Historical
sweeps are retained separately and not pooled.

Timing includes complete updates and owned reports; native extra updates,
stationarity feedback and full means are included. Parsing, recipe creation,
initialization, telemetry formatting, validation, serialization and report
destruction are outside clocks. Allocation counting is disabled. CPU affinity
is 0 with single-thread settings and no concurrent compiler/test/other solver.
Cloud frequency and co-tenancy remain uncontrolled.

Raw samples, every-prefix time/quality/telemetry, exact input, options, compressed
first-round means and source/runner/executable/linked-library fingerprints:

- [100 report](results/canonical_csr/frames100/report.json),
  [metadata](results/canonical_csr/frames100/metadata.json),
  [samples](results/canonical_csr/frames100/samples.csv).
- [300 report](results/canonical_csr/frames300/report.json),
  [metadata](results/canonical_csr/frames300/metadata.json),
  [updates](results/canonical_csr/frames300/updates.csv),
  [quality](results/canonical_csr/frames300/quality.csv).

## Allocation and cached-update diagnostics

A separate warmed 300-arrival stream instruments complete update/report calls.
Requests fall from **3,418,789 to 3,400,782** (0.5%), and requested bytes from
**956,249,913 to 668,581,337** (30.1%). Removing large triplet buffers mostly
reduces allocation volume rather than callback allocation count. Means and reports
match their primary-run counterparts exactly. Instrumented times are excluded
from timing comparisons. These are cumulative allocation traffic, not live/peak/
RSS or native memory comparisons.
[Allocation evidence](results/canonical_csr/diagnostics/allocations.json).

A separate default-policy 300-arrival validation includes 258 cached updates,
30 interval refreshes and 12 forced loop refreshes. Original-energy checks and
twelve AD/dense-QR reference checkpoints pass; complete before/after traces and
summaries match byte for byte. Cached estimates remain approximate between
refreshes, with maximum original gradient about `0.0100`; refreshed gradients
stay below `4.894e-7`. This correctness workload is separate from forced-refresh
timing. [Record](results/canonical_csr/diagnostics/cached_validation.json),
[trace](results/canonical_csr/diagnostics/cached_trace.csv),
[summary](results/canonical_csr/diagnostics/cached_summary.txt).

## Reproduce

Build the optional native executable/verified manifest following
[NATIVE_ISAM2.md](NATIVE_ISAM2.md). Rust remains C++-independent.

```sh
cargo build -p ropt --example benchmark_warm_isam_se2 \
  --example validate_isam_se2 --release --locked --offline
python3 benchmarks/make_ldl_baseline.py --before-canonical-csr \
  --output /tmp/ropt-before-canonical-csr
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_canonical_csr.py \
  --before /tmp/ropt-before-canonical-csr/target/release/examples/benchmark_warm_isam_se2 \
  --before-source /tmp/ropt-before-canonical-csr \
  --after target/release/examples/benchmark_warm_isam_se2 \
  --native YOUR_NATIVE_ISAM2_BINARY --native-manifest YOUR_VERIFIED_MANIFEST.json \
  --frames 300 --rounds 12 --output /tmp/canonical-csr-comparison
```

Finish builds before timing, choose a permitted CPU and fresh output directories.
The helper now reverses the later `qr_rebuild.patch`, then `anchor_coordinates.patch`, then `qr_columns.patch`, then `qr_reductions.patch`, then `update_cursor.patch`, then `qr_norm.patch`, then `symbolic_sorted_merge.patch`, then `qr_aligned.patch`, then `normal_pattern_rebind.patch`, then `ordering_adjacency.patch`, then `whitening_buffers.patch`, then `qr_inplace.patch`, then `graph_retraction.patch`, then
`canonical_csr.patch`, retaining prior QR/symbolic/damping
changes. Its rebuilt source fingerprint matches the frozen predecessor, and its
stream smoke passes against native. Older stages reverse this patch first.
All eleven stages apply in source-only checks and six match archived kernel
fingerprints. Other stages skip compilation in that diagnostic; the latest stage
is built and tested separately.
[Rollback record](results/canonical_csr/diagnostics/rollback_stages.json).

Run each candidate's `validate_isam_se2 300` outside primary timing for cached
correctness. Allocation diagnostics append `alloc` to the runner arguments;
their times are excluded from primary samples. CI checks correctness without
timing thresholds. Full batch/QR rebuilding, retained history, missing Bayes-tree/
selective updates and field validation remain substantial gaps.

The subsequent [graph retraction change](GRAPH_RETRACTION.md) is measured separately.
