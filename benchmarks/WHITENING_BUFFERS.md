# Reusable matrix-whitening buffers

This follows [in-place QR updates](QR_INPLACE.md). Matrix whitening previously
allocated each `W * residual` and `W * Jacobian_block` result separately. A private
workspace now belongs to one `FactorGraph` evaluation and keeps one residual
buffer and one matrix buffer. Matching shapes reuse storage; changed shapes are
resized explicitly. Results swap into callback-owned outputs, and their old raw
storage becomes scratch for the next product. Callbacks keep the owned API and
original argument/block order. No workspace persists between evaluations.

Both owned multiplication and the new call use nalgebra's GEMM kernel with
`alpha=1, beta=0`. Scratch is completely overwritten without reading old values;
there is no diagonal approximation, dropped correlation or reordered product.
Unit/isotropic/diagonal whitening, noise constructors, finite checks, robust losses
and fixed-block validation are unchanged. Storage reuse does not guarantee zero
allocation for changing shapes or underlying large-matrix kernels. The change
introduces no unsafe code, dependency, public API or solver-policy change.

Two internal regressions compare complete products **bit for bit** with ordinary
owned multiplication. They cover small and larger dynamic matrices, poisoned NaN
scratch with matching sizes, changing row counts/widths, zero-width blocks, signed
zero, zero semidefinite weights, overflow and recovery using the same workspace.
A graph regression alternates correlated/unit noise, factor arity and block widths,
including fixed/all-fixed layouts, comparing independent weighted matrices and
complete dense/sparse assembly. Existing covariance/information, analytic/AD/
numeric derivatives, robust loss, fixed-lag and iSAM regressions remain covered.
**203 unit/integration tests + 18 doctests**, Clippy, formatting and nine Python
quality-oracle tests pass.

## Matched complete-update measurements

The near-truth synthetic SE2 input and native GTSAM 4.2.1 configuration follow
[NATIVE_ISAM2.md](NATIVE_ISAM2.md). Both candidates retain direct CSR, graph
retraction, in-place QR, compact symbolic columns and explicit adaptive LM damping
reuse. Input bytes, shared runners, sparse LDL/block ordering and forced full
nonlinear refresh after every arrival are identical. Initial damping is `1e-3`,
gradient tolerance `5e-7`, cost tolerance `1e-15`, step tolerance `1e-10` and trial
limit 200. This is accuracy-conditioned timing, not either default cached policy.

Twelve rotated/reversed rounds put each of three candidates in each position four
times. Each process discards one warm-up stream and measures one full stream:
**72 measured streams / 14,400 timed arrivals** over 100 and 300 added poses.
Original gradients remain at most `1e-6`, the fixed root matches exactly, native
world pose differences stay within mixed metre/radian norm `2e-6`, original costs
within relative `1e-9` and reported costs within `1e-10`. Every prefix passes,
with identical before/after means, costs, trial counts, damping telemetry and QR
entries. Both use **213 / 631 LM trials** at 100 / 300. First-round complete mean
CSVs are byte-identical. Original gradient maxima are `2.780e-7 / 2.964e-7`, and
native pose gaps `1.422e-9 / 3.442e-8`. No failed sample or outlier is removed.

Median sum of complete update/report-call times, milliseconds:

| Added poses | Allocated products | Reused/swapped storage | Native iSAM2 |
| --- | ---: | ---: | ---: |
| 100 | 50.806 | 50.816 | 44.714 |
| 300 | 708.455 | 673.132 | 433.986 |

At 100, the median is effectively unchanged (**0.02% slower**). At 300, the
observed median is **5.0% lower**, with ten of twelve pairs faster. Nine of twelve
pairs improve at 100 despite the slightly slower median. Standard deviations
(denominator N) are 4.7 / 1.7 ms before/after at 100 and 31.4 / 17.9 ms at 300.
Timing variability remains; these observations do not establish statistical
significance, per-update/real-time guarantees or general SLAM superiority.
Native remains **1.14x / 1.55x faster**. Historical sweeps are not pooled.

Timing includes complete updates and owned reports/full means; native additional
updates and stationarity feedback are included. Parsing, initialization, recipe
creation, telemetry formatting, external validation, serialization and report
destruction are outside clocks. Allocation counting is disabled. CPU affinity
is 0, thread settings one and no compiler/test/other solver overlaps primary
samples. Cloud frequency/co-tenancy remain uncontrolled. The driver verifies
native executable and actual resolved `libgtsam` hashes.

Raw samples, every-prefix time/quality/telemetry, input, options, compressed means
and source/runner/executable/library fingerprints:

- [100 report](results/whitening_buffers/frames100/report.json),
  [metadata](results/whitening_buffers/frames100/metadata.json),
  [samples](results/whitening_buffers/frames100/samples.csv).
- [300 report](results/whitening_buffers/frames300/report.json),
  [metadata](results/whitening_buffers/frames300/metadata.json),
  [updates](results/whitening_buffers/frames300/updates.csv),
  [quality](results/whitening_buffers/frames300/quality.csv).

## Separate allocation and cached correctness checks

A warmed 300-arrival allocation diagnostic counts complete update/report calls.
Requests fall from **3,101,115 to 2,651,399 (14.5%)**, and requested bytes from
**652,868,425 to 627,669,433 (3.9%)**. Means and non-timing reports match primary
runs exactly. These are cumulative allocation traffic, not live/peak/RSS memory
or native memory comparisons; instrumented times are excluded from primary timing.
[Record](results/whitening_buffers/diagnostics/allocations.json).

Separate default-policy 300-arrival validation contains 258 cached updates,
30 interval refreshes, 12 forced refreshes and twelve AD/dense-QR loop reference
checkpoints. Complete before/after traces and summaries are byte-identical.
Cached estimates remain approximate between refreshes, with maximum original
gradient about `0.0100` (unchanged). These checks validate correctness, not cached
performance. [Record](results/whitening_buffers/diagnostics/cached_validation.json),
[trace](results/whitening_buffers/diagnostics/cached_trace.csv),
[summary](results/whitening_buffers/diagnostics/cached_summary.txt).

## Retained first version

The first implementation copied products back into raw outputs. Its separate
same-policy sweep observed median reductions of 1.7% / 1.3% at 100 / 300, with
six / seven paired rounds faster and substantial timing variability. Those
samples and diagnostics are retained separately; no comparison pools the sweeps
or treats their timing difference as a same-condition copy-versus-swap result.
The final code removes that output copy through ownership-safe buffer swaps.
[First 100 report](results/whitening_buffers/copyback/frames100/report.json),
[first 300 report](results/whitening_buffers/copyback/frames300/report.json),
[first allocation record](results/whitening_buffers/copyback/diagnostics/allocations.json),
[saved first patch](whitening_copyback.patch).
The first metadata's rollback digest refers to that saved patch; the canonical
`whitening_buffers.patch` now describes the final implementation.
To recreate the first kernel, apply `whitening_copyback.patch` forward to an
isolated `--before-whitening-buffers` snapshot and rebuild the runner. Its
source-only reconstruction matches the archived first kernel fingerprint.

## Reproduce

Build the optional verified native executable/manifest following [NATIVE_ISAM2.md](NATIVE_ISAM2.md).
Rust still has no C++ dependency.

```sh
cargo build -p ropt --example benchmark_warm_isam_se2 \
  --example validate_isam_se2 --release --locked --offline
python3 benchmarks/make_ldl_baseline.py --before-whitening-buffers \
  --output /tmp/ropt-before-whitening-buffers
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_whitening.py \
  --before /tmp/ropt-before-whitening-buffers/target/release/examples/benchmark_warm_isam_se2 \
  --before-source /tmp/ropt-before-whitening-buffers \
  --after target/release/examples/benchmark_warm_isam_se2 \
  --native YOUR_NATIVE_ISAM2_BINARY --native-manifest YOUR_VERIFIED_MANIFEST.json \
  --frames 300 --rounds 12 --output /tmp/whitening-buffer-comparison
```

Finish builds first, choose a permitted CPU and fresh output directories. The
helper now reverses the later `qr_rebuild.patch`, then `anchor_coordinates.patch`, then `qr_columns.patch`, then `qr_reductions.patch`, then `update_cursor.patch`, then `qr_norm.patch`, then `symbolic_sorted_merge.patch`, then `qr_aligned.patch`, then `normal_pattern_rebind.patch`, then `ordering_adjacency.patch`, then
`whitening_buffers.patch` for this predecessor;
older stages reverse it first, then their existing patches. The rebuilt latest
predecessor matches the frozen kernel and passes six-round 25-arrival native smoke.
All fourteen stages apply in source-only checks and match archived fingerprints;
older-stage compilation is skipped while the newest predecessor is built separately.
[Rollback record](results/whitening_buffers/diagnostics/rollback_stages.json).

Run both candidates' `validate_isam_se2 300` outside timing for cached correctness.
Allocation diagnostics append `alloc` to the runner; exclude their times. CI
checks correctness without timing thresholds. Bayes-tree/selective updates,
retained history and field validation remain gaps.

The subsequent [ordering adjacency change](ORDERING_ADJACENCY.md) is measured separately.
