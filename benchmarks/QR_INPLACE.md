# In-place updates of existing sparse QR columns

This follows [slice-based graph retraction](GRAPH_RETRACTION.md). Sparse Givens
insertion previously rebuilt every upper row into a separate vector, even when
its existing columns could be updated without insertion. Existing upper entries
now rotate and compact in their own row storage. Writes never reach unread
entries. After consuming that row, new trailing columns can append directly.
At the first new interior column, a reusable buffer copies the compacted prefix
and the original sorted merge processes the remaining suffix. Processed values
are never replayed. Initially empty rows use the existing buffered path.

All paths retain ascending column arithmetic, explicit zero operands, exact-zero
removal, `hypot` normalization, positive diagonal/rank policy, finite checks,
RHS/constant-cost accumulation and transient entry-limit checks. A prefix
cancellation can free a slot; an interior/trailing insertion must still fit at
that point, before a later cancellation. Public `append` and `IsamSmoother::update`
work on candidates and commit only on success. Errors after in-place writes,
including coefficient/RHS overflow, preserve the committed factor/estimates.
There is no new API, unsafe code, tolerance or incremental-policy change.

Three regressions compare mixed prefix cancellation/interior insertion/trailing
growth against independent dense SVD and the complete residual objective. They
check bit-identical one-batch/row-by-row solves and costs, transient fill after
prefix compaction, recovery after failure and rollback after coefficient/RHS
overflow. Existing constant/empty-row, signed-zero, tiny-coefficient, disjoint
pattern, rank, column-extension and iSAM/fixed-lag tests remain covered.
**200 unit/integration tests + 18 doctests**, Clippy, formatting and nine Python
quality-oracle tests pass.

## Structural diagnostic

An isolated instrumented predecessor processes one 300-arrival stream with no
warm-up. It counts **2,998,044 Givens rotations** and **41,152,491 existing
non-diagonal upper entries**. All those existing entries can be processed before
any new interior column in this particular stream; **1,966,104 rotations** have
an existing row and no remaining lower columns. Other rotations include empty
rows and trailing additions. This describes the synthetic topology, not a general
sparsity guarantee. Interior-fill fallback has separate regression coverage.

Instrumented means are byte-identical to the uninstrumented predecessor and
pass original-energy checks. Profiling counts/times are excluded from primary
timing. [Record](results/qr_inplace/diagnostics/structure_profile.json),
[counts](results/qr_inplace/diagnostics/structure_counts.csv),
[isolated instrumentation patch](qr_inplace_profile.patch).

## Matched complete-update comparison

Both candidates retain direct CSR, graph retraction, QR buffers, compact symbolic
columns and explicit adaptive LM damping reuse. They use identical input bytes,
shared runners, sparse LDL/block ordering and full nonlinear refresh at every
arrival. Initial damping is `1e-3`, gradient tolerance `5e-7`, cost tolerance
`1e-15`, step tolerance `1e-10` and trial limit 200. The near-truth synthetic SE2
input and verified native GTSAM 4.2.1 configuration follow [NATIVE_ISAM2.md](NATIVE_ISAM2.md).
This is an accuracy-conditioned comparison, not default cached-update timing.

Twelve rotated/reversed rounds place each of three candidates in each position
four times. Each process discards one full warm-up stream and measures one stream:
**72 measured streams / 14,400 timed arrivals** at 100 and 300 added poses.
The original gradient gate stays `1e-6`, fixed root exact, native mixed metre/
radian pose difference at most `2e-6`, relative original cost difference `1e-9`
and reported cost difference `1e-10`. Every prefix passes. Before/after means,
costs, LM trials, damping telemetry and QR entry counts match exactly; both use
**213 / 631 trials**. First-round full mean CSVs match byte for byte.
Maximum original gradients remain `2.780e-7 / 2.964e-7`; native pose gaps are
`1.422e-9 / 3.442e-8`. No failed sample or timing outlier is removed.

Median sum of complete update/report-call times, milliseconds:

| Added poses | Rebuilt upper rows | In-place existing columns | Native iSAM2 |
| --- | ---: | ---: | ---: |
| 100 | 56.073 | 55.453 | 50.110 |
| 300 | 772.182 | 702.126 | 430.193 |

Observed median reductions are **1.1% / 9.1%**, with 8 / 11 of twelve paired
rounds faster at 100 / 300. Standard deviations (denominator N) are 6.0 / 5.8 ms
before/after at 100 and 49.2 / 40.4 ms at 300. Timing remains variable; the small
100-arrival difference does not demonstrate a dependable speedup. No statistical
significance, per-update/real-time or general-workload guarantee is claimed.
Native remains **1.11x / 1.63x faster**. Historical sweeps stay separate, not pooled.

Timing includes complete updates and owned reports/full means. Native additional
updates and stationarity feedback are included. Parsing, initialization, recipe
creation, telemetry formatting, external validation, output and report destruction
are outside clocks. Allocation counting is disabled. CPU affinity is 0 and thread
settings one; no compiler/test/other solver overlaps primary timing. Cloud
frequency/co-tenancy remain uncontrolled. Native executable and actual resolved
`libgtsam` hashes are verified by the driver.

Raw every-prefix time/quality/telemetry, samples, input, options, compressed means
and source/runner/executable/linked-library fingerprints:

- [100 report](results/qr_inplace/frames100/report.json),
  [metadata](results/qr_inplace/frames100/metadata.json),
  [samples](results/qr_inplace/frames100/samples.csv).
- [300 report](results/qr_inplace/frames300/report.json),
  [metadata](results/qr_inplace/frames300/metadata.json),
  [updates](results/qr_inplace/frames300/updates.csv),
  [quality](results/qr_inplace/frames300/quality.csv).

## Separate allocation and cached correctness checks

A warmed 300-arrival allocation diagnostic measures complete update/report calls.
Requests decrease from **3,115,073 to 3,101,115 (0.45%)**, and requested bytes from
**660,198,857 to 652,868,425 (1.1%)**. Means and non-timing reports match primary
runs exactly. Instrumented times are excluded; these are cumulative allocation
traffic, not live/peak/RSS memory or native memory comparisons.
[Record](results/qr_inplace/diagnostics/allocations.json).

The default-policy 300-arrival validator includes 258 cached updates, 30 interval
refreshes, 12 forced loop refreshes and twelve AD/dense-QR reference checkpoints.
Before/after complete traces and summaries are byte-identical. Original-energy
checks pass; cached estimates remain approximate between refreshes, with maximum
original gradient about `0.0100` (unchanged). These are correctness checks, not
cached-update timing. [Record](results/qr_inplace/diagnostics/cached_validation.json),
[trace](results/qr_inplace/diagnostics/cached_trace.csv),
[summary](results/qr_inplace/diagnostics/cached_summary.txt).

## Reproduce

Build the optional verified native binary/manifest using [NATIVE_ISAM2.md](NATIVE_ISAM2.md).
The Rust library has no C++ dependency.

```sh
cargo build -p ropt --example benchmark_warm_isam_se2 \
  --example validate_isam_se2 --release --locked --offline
python3 benchmarks/make_ldl_baseline.py --before-qr-inplace \
  --output /tmp/ropt-before-qr-inplace
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_qr_inplace.py \
  --before /tmp/ropt-before-qr-inplace/target/release/examples/benchmark_warm_isam_se2 \
  --before-source /tmp/ropt-before-qr-inplace \
  --after target/release/examples/benchmark_warm_isam_se2 \
  --native YOUR_NATIVE_ISAM2_BINARY --native-manifest YOUR_VERIFIED_MANIFEST.json \
  --frames 300 --rounds 12 --output /tmp/qr-inplace-comparison
```

Finish builds first, choose a permitted CPU and fresh output directories. The
helper now reverses the later `qr_rebuild.patch`, then `anchor_coordinates.patch`, then `qr_columns.patch`, then `qr_reductions.patch`, then `update_cursor.patch`, then `qr_norm.patch`, then `symbolic_sorted_merge.patch`, then `qr_aligned.patch`, then `normal_pattern_rebind.patch`, then `ordering_adjacency.patch`, then `whitening_buffers.patch`, then `qr_inplace.patch`
for this predecessor. Older stages
reverse it first, then their existing patches. The rebuilt latest predecessor
matches the frozen kernel fingerprint and passes the six-round 25-arrival native
smoke. All thirteen stages apply in source-only checks and match archived kernels;
older stages skip compilation, while the newest predecessor is built separately.
[Rollback record](results/qr_inplace/diagnostics/rollback_stages.json).

For cached correctness run both candidates' `validate_isam_se2 300` separately.
For allocation diagnostics append `alloc` to the runner; exclude those times.
For structure counts apply `qr_inplace_profile.patch` to an isolated
`--before-qr-inplace` snapshot, rebuild `benchmark_warm_isam_se2`, then run the
300-arrival input with runner arguments `1 0`. Keep that instrumented binary
separate from primary candidates. CI checks correctness without timing thresholds.
Bayes-tree/selective updates, retained full history and field evidence remain gaps.

The subsequent [whitening buffer change](WHITENING_BUFFERS.md) is measured separately.
