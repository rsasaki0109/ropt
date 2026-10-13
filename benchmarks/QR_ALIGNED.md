# Reuse aligned lower rows in sparse Givens QR

This follows [exact normal-graph rebinding](NORMAL_PATTERN_REBIND.md). A separate
warmed predecessor classifier finds **2,998,044 Givens rotations**, including
**1,424,980 populated upper rows with exactly equal live suffix columns** and
**19,850,690 paired suffix entries**, in the synthetic 300-arrival stream.
These counts describe this topology, not a general sparsity guarantee.

## Implementation and numerical contract

Before each rotation, equal suffix lengths and every column index are checked.
Exactly aligned rows rotate directly in upper and lower storage. A cursor consumes
the lower diagonal without shifting the row. Only exact-zero cancellations require
compaction. A different column pattern uses the previous sorted merge, with the
cursor selecting the live suffix. Equal widths alone never select direct rotation.

Arithmetic stays in ascending column order with unchanged multiplication/addition,
`hypot` normalization, exact-zero removal, positive diagonal/rank heuristic,
finite checks, RHS rotation and constant-cost accumulation. Matched columns can
only retain or remove existing upper entries; additions and transient fill checks
still use the sorted path. Public append and smoother updates commit candidates
only on success. Overflow after cursor advance cannot change committed state.
There is no unsafe code, new API/dependency, tolerance or refresh-policy change.

Three new regressions compare dense aligned suffixes and cancellation followed by
equal-width/different-column fallback with independent SVD and the complete
residual objective. Batch and row-by-row solves/costs are bit-identical. Late
coefficient/RHS overflow checks rollback and recovery. Existing transient-fill,
rank, empty/constant-row, signed-zero, underflow, column-extension, manifold,
iSAM and fixed-lag regressions pass: **211 unit/integration tests + 18 doctests**,
nine Python quality-oracle tests, Clippy and formatting checks.

## Matched complete-update comparison

The synthetic near-truth SE2 input and verified native GTSAM 4.2.1 configuration
follow [NATIVE_ISAM2.md](NATIVE_ISAM2.md). Both ropt candidates use sparse LDL
with three-coordinate block minimum-degree ordering, explicit adaptive LM damping
reuse and full nonlinear refresh at every arrival. Initial damping is `1e-3`,
gradient tolerance `5e-7`, cost tolerance `1e-15`, step tolerance `1e-10` and trial
limit 200. Input bytes, shared runners and all earlier changes are unchanged.
This is accuracy-conditioned timing, not default cached-update performance.

Twelve rotated/reversed rounds place each candidate in each position four times.
Each process discards one warm-up stream and measures one complete stream:
**72 measured streams / 14,400 timed arrivals** across 100 and 300 added poses.
Every prefix passes original-gradient `1e-6`, exact fixed-root, native world-pose
mixed metre/radian norm `2e-6`, relative original-cost `1e-9` and relative reported-
cost `1e-10` gates. Before/after estimates, costs, LM trials, damping telemetry and
QR entries match exactly. First-round full mean CSVs are byte-identical. Both
use **213 / 631 trials**; original gradient maxima stay `2.780e-7 / 2.964e-7`,
and native pose gaps `1.422e-9 / 3.442e-8`. No failure or timing outlier is removed.

Median sum of complete update/report-call times, milliseconds:

| Added poses | Copied lower rows | Aligned row reuse | Native iSAM2 |
| --- | ---: | ---: | ---: |
| 100 | 46.498 | 45.042 | 43.763 |
| 300 | 639.667 | 626.852 | 437.957 |

Observed median reductions are **3.1% / 2.0%**; ten of twelve paired rounds are
faster at each size. Population standard deviations before/after are 3.6 / 1.8 ms
at 100 and 19.0 / 15.9 ms at 300. Variation exceeds these small median differences;
no statistical significance or dependable general speedup is established.
Native remains **1.03x / 1.43x faster**. Historical sweeps are separate, not pooled.
There is no real-time, per-update or general SLAM superiority claim.

Timing includes complete updates and owned reports/full means, including native
additional updates and stationarity feedback. Parsing, initialization, recipe
construction, telemetry formatting, external validation, serialization and report
destruction are excluded. Allocation counting is disabled. CPU affinity is 0 and
thread settings one, with no compiler/test/other solver overlap during primary
samples. Cloud frequency/co-tenancy remain uncontrolled. The driver verifies the
native executable and actual resolved `libgtsam` hashes against its manifest.

Raw samples, every-prefix time/quality/telemetry, input, options, compressed means
and source/runner/executable/library fingerprints are retained:

- [100 report](results/qr_aligned/frames100/report.json),
  [metadata](results/qr_aligned/frames100/metadata.json),
  [samples](results/qr_aligned/frames100/samples.csv).
- [300 report](results/qr_aligned/frames300/report.json),
  [metadata](results/qr_aligned/frames300/metadata.json),
  [updates](results/qr_aligned/frames300/updates.csv),
  [quality](results/qr_aligned/frames300/quality.csv).

## Separate diagnostics

The classifier uses local counters aggregated once per append and reset before
the measured stream. Its estimates match primary predecessor means byte for byte
and pass original-energy gates. Instrumented times are excluded. Its forward
patch reproduces the instrumented kernel and runner in a source-only check.
[Structure record](results/qr_aligned/diagnostics/structure_classifier.json),
[counts](results/qr_aligned/diagnostics/structure_counts.csv).

A separate warmed 300-arrival allocation run counts complete update/report calls.
Both candidates request **2,403,726 allocations / 568,194,510 bytes**: cumulative
allocation traffic is unchanged. This optimization reduces merge/copy work, not
allocation counts. Means and every non-timing report field match primary results.
Instrumented times are excluded; live/peak/RSS and native memory are unmeasured.
[Allocation record](results/qr_aligned/diagnostics/allocations.json).

The 300-arrival cached-policy validator includes 258 cached updates, 30 interval
refreshes, 12 forced loop refreshes and twelve AD/dense-QR reference checkpoints.
Before/after traces and summaries are byte-identical. Cached estimates remain
approximate between refreshes, with maximum original gradient about `0.0100`
(unchanged). These are correctness checks, not cached-update timings.
[Record](results/qr_aligned/diagnostics/cached_validation.json),
[trace](results/qr_aligned/diagnostics/cached_trace.csv),
[summary](results/qr_aligned/diagnostics/cached_summary.txt).

## Reproduce

Build the optional verified native executable/manifest following
[NATIVE_ISAM2.md](NATIVE_ISAM2.md). Rust has no C++ dependency.

```sh
cargo build -p ropt --example benchmark_warm_isam_se2 \
  --example validate_isam_se2 --release --locked --offline
python3 benchmarks/make_ldl_baseline.py --before-qr-aligned \
  --output /tmp/ropt-before-qr-aligned
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_qr_aligned.py \
  --before /tmp/ropt-before-qr-aligned/target/release/examples/benchmark_warm_isam_se2 \
  --before-source /tmp/ropt-before-qr-aligned \
  --after target/release/examples/benchmark_warm_isam_se2 \
  --native YOUR_NATIVE_ISAM2_BINARY --native-manifest YOUR_VERIFIED_MANIFEST.json \
  --frames 300 --rounds 12 --output /tmp/qr-aligned-comparison
```

Finish builds first, choose a permitted CPU and fresh output directories. The
helper reverses the later `qr_rebuild.patch`, then `anchor_coordinates.patch`, then `qr_columns.patch`, then `qr_reductions.patch`, then `update_cursor.patch`, then `qr_norm.patch`, then `symbolic_sorted_merge.patch`, then `qr_aligned.patch`
for this predecessor; older stages
reverse it first, then their existing patches. The rebuilt predecessor matches
the frozen kernel and passes six-round 25-arrival native smoke. All seventeen
stages apply and match archived kernels in source-only checks; older-stage
compilation is skipped while the latest predecessor is built separately.
[Rollback record](results/qr_aligned/diagnostics/rollback_stages.json).

For cached correctness run both candidates' `validate_isam_se2 300` separately.
Allocation diagnostics append `alloc` to the runner and exclude their timings.
For structure counts apply `qr_aligned_classifier.patch` forward to an isolated
predecessor, rebuild `benchmark_warm_isam_se2`, then run the 300-arrival input with
arguments `1 1`. Retain the instrumented binary separately. CI checks correctness
without timing thresholds. Selective Bayes-tree updates, retained history and
field validation remain gaps.

The subsequent [sorted symbolic fill and CSR-row reuse](SYMBOLIC_MERGE.md) is measured separately.
