# Row buffer reuse and simpler Givens merges

This follows [compact symbolic LDL preparation](SYMBOLIC_PATTERN.md). Both
ropt candidates keep that improvement and explicitly reuse adaptive LM damping.
Only incremental QR processing changes; no solver/update policy or tolerance changes.

The later [direct CSR assembly change](CANONICAL_CSR.md) keeps this improvement
and records a separate matched sweep. The measurements here retain their kernels.

## Implementation and numerical contracts

`IncrementalQr::append_in_place` now retains its incoming-row vector across
residual rows, alongside the existing upper/lower merge buffers. Clearing and
refilling that vector avoids repeated creation/destruction of row storage.
All scratch remains local to one append call; no global or mutable shared cache
is introduced. Three buffers can retain capacity until the append returns.
Allocation traffic and peak/live memory are different quantities.

The sorted-column union now handles paired columns with one comparison, then
visits the remaining upper or lower tail. Ascending traversal, explicit zero
operands, Givens arithmetic, finite checks and exact-zero removal are preserved.
Entry accounting still follows each column in order: a temporary new entry can
exceed the limit **before a later cancellation frees a slot**. The diagonal is
counted in its original position. Public `append` remains transactional through
its owned candidate; smoother state commits only after all checks succeed.

`cost` skips scanning R when **every supplied delta component is exactly zero**.
Finite R entries contribute only signed zeros there. It accumulates the same
constant and RHS-square terms in row order, with the same dimension, finite-input
and overflow checks. Nonzero deltas use the ordinary full matrix traversal.
No epsilon or norm threshold is used: a nonzero vector's norm can underflow to
zero even when its product with a large R entry has substantial cost. Undamped
rank checks and full finite back-substitution remain in place.

Three new regressions compare varying-width/gapped rows and constant rows with
independent batch SVD solutions and full objectives, including split batches and
empty appends. A five-entry factor rejects transient six-entry fill even though
the completed factor would return to five entries; its solution/cost remain
unchanged and it accepts a later valid append. Exact-zero tests cover signed
zero, cost overflow, nonfinite input and a nonzero delta whose norm underflows.
Existing mixed-manifold, nonlinear rejection, rank/fill and rollback tests pass.
There are **191 unit/integration tests + 18 doctests**; Clippy, formatting and the
nine Python quality-oracle tests pass.

## Matched measurements

The input, root conditioning and native configuration are described in the
[native iSAM2 comparison](NATIVE_ISAM2.md). The same verified GTSAM 4.2.1 binary
and linked library are used. Before/after Rust candidates use identical shared
runner sources, sparse LDL/block ordering and explicit adaptive damping reuse.
Every arrival forces a complete nonlinear refresh. Initial damping is `1e-3`,
gradient tolerance `5e-7`, cost tolerance `1e-15`, step tolerance `1e-10` and the
trial limit 200. This synthetic near-truth SE2 workload is accuracy-conditioned;
it is not either library's default incremental policy.

Twelve rotated/reversed rounds put each candidate in each position four times.
Each process discards one full warm-up stream and measures one stream. The final
comparison contains **72 measured streams / 14,400 timed arrivals**. All original
prefix gradients must be at most `1e-6`, the fixed root must match exactly, native
pose agreement stays within mixed metre/radian norm `2e-6`, original-cost
agreement within relative `1e-9`, and reported-cost agreement within `1e-10`.
No gate is relaxed and no failed sample or timing outlier is discarded.

The driver additionally requires identical before/after means, original costs,
reported costs, batch trial counts, damping telemetry and QR entry counts at
every prefix. Both candidates take **213 / 631 LM trials** at 100 / 300 arrivals;
first-round full mean CSVs also match byte for byte. Maximum original gradients
are unchanged at `2.780e-7 / 2.964e-7`, and native pose differences remain
`1.422e-9 / 3.442e-8`.

Median sum of complete update/report-call times, milliseconds:

| Added poses | Previous QR processing | Reused buffers/merge/origin cost | Native iSAM2 |
| --- | ---: | ---: | ---: |
| 100 | 63.175 | **61.899** | 45.238 |
| 300 | 854.415 | **826.721** | 444.117 |

Observed median reductions are **2.0% / 3.2%**. Native remains **1.37x / 1.86x
faster**. Timing differences are small: before/after standard deviations over
the retained samples (denominator N)
are 3.0 / 3.1 ms at 100 and 35.7 / 18.7 ms at 300. Four of twelve paired rounds
are slower after the change at 100, and one is 2.2% slower at 300. These observations
do not establish a per-update, real-time or general-workload speed guarantee.

Timing includes each complete update and owned report. Native extra updates,
stationarity feedback and complete means are included. Parsing, recipe creation,
initialization, telemetry formatting, quality checks, serialization and report
destruction are outside clocks. Allocation counting is disabled. CPU affinity
is 0 with single-thread settings; no compiler/test/other solver overlaps the
primary sweep. Cloud frequency and co-tenancy remain uncontrolled.

Reports retain raw samples, every-prefix times/quality/telemetry, input bytes,
options, compressed first-round full means and source/runner/executable/linked-library
fingerprints:

- [100-arrival report](results/qr_merge_buffers/frames100/report.json),
  [metadata](results/qr_merge_buffers/frames100/metadata.json),
  [samples](results/qr_merge_buffers/frames100/samples.csv).
- [300-arrival report](results/qr_merge_buffers/frames300/report.json),
  [metadata](results/qr_merge_buffers/frames300/metadata.json),
  [updates](results/qr_merge_buffers/frames300/updates.csv),
  [quality](results/qr_merge_buffers/frames300/quality.csv).

## Retained first version and diagnostics

The first version included only row reuse and paired/tail merging. Its separate
six-round sweep observed 0.5% / 2.5% median reductions, with mixed paired rounds.
That small/noisy result prompted the exact-origin cost improvement. The first
version is retained with its original driver and rollback patch, not pooled with
or replaced by the final measurement:
[100](results/qr_merge_buffers/merge_only/frames100/report.json),
[300](results/qr_merge_buffers/merge_only/frames300/report.json),
[driver](results/qr_merge_buffers/merge_only/driver.py),
[patch](results/qr_merge_buffers/merge_only/rollback.patch).
Source/executable fingerprints distinguish the two implementations. Native and
previous-kernel timing also vary between sweeps; comparing absolute medians across
them would confound the code change with measurement conditions.

A separate warmed 300-arrival allocation diagnostic instruments only complete
update/report calls. Requests fall from **3,594,324 to 3,418,789** (4.9%), and
requested bytes from **977,028,313 to 956,249,913** (2.1%). Means and reports match
their primary-run counterparts exactly. Instrumented times are excluded from
timing comparisons. These are allocation traffic, not live/peak/RSS memory or
native memory comparisons.
[Allocation evidence](results/qr_merge_buffers/diagnostics/allocations.json).

Default cached updates, interval refreshes and loop refreshes are separately
checked through 300 arrivals against independent original-energy and AD/dense-QR
references. The complete before/after trace and summary match byte for byte.
Those cached estimates remain approximate between rebuilds; that validation is
separate from the forced-refresh timing workload.
[Cached validation](results/qr_merge_buffers/diagnostics/cached_validation.json),
[trace](results/qr_merge_buffers/diagnostics/cached_trace.csv),
[summary](results/qr_merge_buffers/diagnostics/cached_summary.txt).

## Reproduce

Build the optional native executable and verified manifest following
[NATIVE_ISAM2.md](NATIVE_ISAM2.md). The Rust library remains C++-independent.

```sh
cargo build -p ropt --example benchmark_warm_isam_se2 \
  --example validate_isam_se2 --release --locked --offline
python3 benchmarks/make_ldl_baseline.py --before-qr-buffers \
  --output /tmp/ropt-before-qr-buffers
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_qr_buffers.py \
  --before /tmp/ropt-before-qr-buffers/target/release/examples/benchmark_warm_isam_se2 \
  --before-source /tmp/ropt-before-qr-buffers \
  --after target/release/examples/benchmark_warm_isam_se2 \
  --native YOUR_NATIVE_ISAM2_BINARY --native-manifest YOUR_VERIFIED_MANIFEST.json \
  --frames 300 --rounds 12 --output /tmp/qr-buffer-comparison
```

Finish builds before timing, choose a permitted CPU and fresh output directories.
The helper reverses the later `qr_rebuild.patch`, then `anchor_coordinates.patch`, then `qr_columns.patch`, then `qr_reductions.patch`, then `update_cursor.patch`, then `qr_norm.patch`, then `symbolic_sorted_merge.patch`, then `qr_aligned.patch`, then `normal_pattern_rebind.patch`, then `ordering_adjacency.patch`, then `whitening_buffers.patch`, then `qr_inplace.patch`, then `graph_retraction.patch`, then `canonical_csr.patch`, then
`qr_merge_buffers.patch`, retaining compact symbolic columns
and damping reuse. Its rebuilt source fingerprint matches the frozen predecessor.
Rollback stages are now an ordered list with an explicit stopping stage. All ten
stages apply in source-only checks; five stages match archived kernel fingerprints.
Other stages skip compilation in that diagnostic; the latest stage is built and
smoke-tested separately.
[Rollback record](results/qr_merge_buffers/diagnostics/rollback_stages.json).

For cached correctness, run each candidate's `validate_isam_se2 300` outside the
timing sweep. For allocation traffic only, append `alloc` to the stream runner
arguments; do not use instrumented times as primary samples. CI checks correctness
and protection of existing output paths without timing thresholds.

Full batch/undamped QR rebuilds, retained all-history storage and missing Bayes-tree
or selective updates remain substantial architectural gaps. No iSAM2 equivalence
or general OSS superiority is claimed.
