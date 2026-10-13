# Slice-based factor-graph retraction

This follows [direct CSR assembly](CANONICAL_CSR.md). The change removes owned
per-variable value, tangent and updated-value buffers from `FactorGraph::retract`.
The graph clones its complete output once, reads input/tangent slices and writes
the updated ambient slices using existing fixed-size Lie operations. Fixed values
retain their original bits. There is no persistent scratch state or new public API.

Every input block is still validated in variable order, including disconnected
and fixed variables. Active steps and updated values retain finite-value checks;
quaternions retain the unit-norm check and the same normalization/retraction.
An earlier invalid step still precedes a later invalid fixed quaternion. Failure
returns an error without modifying caller inputs or returning partial output.
Private slices use the graph's validated ambient/tangent ranges; no unsafe code
or unchecked public constructor is introduced. The owned `VariableKind::retract`
remains the independent regression reference, with unchanged arithmetic.

Three regressions cover all five manifold kinds in a mixed graph, changing fixed
layouts including all-fixed, zero/tiny/large increments, signed zero, angle wrap,
negative slightly nonunit accepted quaternions, disconnected variables, dimension
errors, NaN/Inf, overflow and failure precedence. Every successful ambient component
matches the public owned retraction **bit for bit**. Full verification passes
**197 unit/integration tests + 18 doctests**, Clippy, formatting and nine Python
quality-oracle tests. Existing AD, manifold, dense/sparse, fixed-lag and transactional
iSAM tests remain covered.

## Matched complete-update measurements

Both candidates retain direct CSR, QR buffers, compact symbolic columns and
explicit adaptive LM damping reuse. Input bytes, shared runners, sparse LDL/block
ordering, solver tolerances, full-refresh policy and LM trajectories are unchanged.
The workload is the near-truth synthetic SE2 stream from [native iSAM2 comparison](NATIVE_ISAM2.md),
not either library's default incremental policy or a real-world SLAM dataset.
Native GTSAM 4.2.1 executable and the actual resolved `libgtsam` are hash-verified.

Twelve rotated/reversed rounds place each of three candidates in each position
four times. Each process discards one full warm-up stream and measures one stream:
**72 measured streams / 14,400 timed arrivals** across 100 and 300 added poses.
Timing includes complete update calls and owned reports/full means. Native extra
updates and stationarity feedback are included. Parsing, initialization, recipe
creation, external validation, serialization and report destruction are outside
clocks. Allocation counting is disabled in primary timing. CPU affinity is 0 and
thread settings are one; no compiler/test/other solver runs concurrently. Cloud
frequency/co-tenancy remain uncontrolled.

Every prefix must satisfy the original-energy gradient bound `1e-6`, exact fixed
root, world pose difference at most `2e-6` (mixed metre/radian norm), relative
original cost difference `1e-9` and reported cost difference `1e-10`. These gates
are unchanged. Before/after means, costs, LM trials, damping telemetry and QR
entry counts must match exactly. Both use **213 / 631 trials** at 100 / 300;
first-round full mean CSVs are byte-identical. Maximum original gradients are
`2.780e-7 / 2.964e-7`; native pose gaps are `1.422e-9 / 3.442e-8`.
No failed sample or timing outlier is removed.

Median sum of complete update/report-call times, milliseconds:

| Added poses | Owned block retraction | Slice-based retraction | Native iSAM2 |
| --- | ---: | ---: | ---: |
| 100 | 51.886 | 52.248 | 42.206 |
| 300 | 770.891 | 743.634 | 423.950 |

At 100, the median is **0.7% slower**; there is no demonstrated runtime improvement.
At 300, the observed median is **3.5% lower**, with 10 of 12 paired rounds faster.
Seven of twelve pairs improve at 100. Standard deviations (denominator N) are
1.3 / 2.3 ms before/after at 100 and 45.5 / 36.8 ms at 300. The effect is small
relative to timing variability and does not establish statistical significance,
per-update/real-time guarantees or general superiority. Native remains **1.24x /
1.75x faster**. Historical sweeps are retained separately and are not pooled.

Raw samples, every-prefix quality/telemetry/timing, full input, options, compressed
first-round means and source/runner/executable/library fingerprints:

- [100 report](results/graph_retraction/frames100/report.json),
  [metadata](results/graph_retraction/frames100/metadata.json),
  [samples](results/graph_retraction/frames100/samples.csv).
- [300 report](results/graph_retraction/frames300/report.json),
  [metadata](results/graph_retraction/frames300/metadata.json),
  [updates](results/graph_retraction/frames300/updates.csv),
  [quality](results/graph_retraction/frames300/quality.csv).

## Allocation and cached-update diagnostics

A separate warmed 300-arrival stream counts complete update/report allocations.
Requests fall from **3,400,782 to 3,115,073 (8.4%)**, and requested bytes from
**668,581,337 to 660,198,857 (1.3%)**. Means and all non-timing report fields match
primary runs exactly. Instrumented times are excluded from primary measurements.
These are cumulative allocation traffic, not live/peak/RSS memory or a native
memory comparison. [Allocation evidence](results/graph_retraction/diagnostics/allocations.json).

Separate default-policy 300-arrival cached validation checks original energies
and twelve AD/dense-QR loop reference checkpoints. Complete before/after traces
and summaries are byte-identical, including 258 cached updates, 30 interval
refreshes and 12 forced refreshes. Cached estimates remain approximate between
refreshes (maximum original gradient about `0.0100`, unchanged); these checks
are correctness evidence, not cached performance timing.
[Record](results/graph_retraction/diagnostics/cached_validation.json),
[trace](results/graph_retraction/diagnostics/cached_trace.csv),
[summary](results/graph_retraction/diagnostics/cached_summary.txt).

## Reproduce

Build the optional verified native executable/manifest following [NATIVE_ISAM2.md](NATIVE_ISAM2.md).
Rust still has no C++ dependency.

```sh
cargo build -p ropt --example benchmark_warm_isam_se2 \
  --example validate_isam_se2 --release --locked --offline
python3 benchmarks/make_ldl_baseline.py --before-graph-retraction \
  --output /tmp/ropt-before-graph-retraction
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_graph_retraction.py \
  --before /tmp/ropt-before-graph-retraction/target/release/examples/benchmark_warm_isam_se2 \
  --before-source /tmp/ropt-before-graph-retraction \
  --after target/release/examples/benchmark_warm_isam_se2 \
  --native YOUR_NATIVE_ISAM2_BINARY --native-manifest YOUR_VERIFIED_MANIFEST.json \
  --frames 300 --rounds 12 --output /tmp/graph-retraction-comparison
```

Finish builds before timing, choose a permitted CPU and fresh output directories.
The helper now reverses the later `qr_rebuild.patch`, then `anchor_coordinates.patch`, then `qr_columns.patch`, then `qr_reductions.patch`, then `update_cursor.patch`, then `qr_norm.patch`, then `symbolic_sorted_merge.patch`, then `qr_aligned.patch`, then `normal_pattern_rebind.patch`, then `ordering_adjacency.patch`, then `whitening_buffers.patch`, then `qr_inplace.patch`, then `graph_retraction.patch`
for this predecessor;
older stages reverse it before their existing patches. The rebuilt latest
predecessor matches the frozen previous kernel fingerprint and passes the
six-round 25-arrival smoke against native. All twelve rollback stages apply in
source-only checks, and all eleven old stages match their previously saved
fingerprints. Those older-stage builds are skipped; only the latest predecessor
is compiled and tested separately.
[Rollback record](results/graph_retraction/diagnostics/rollback_stages.json).

For cached correctness run both candidates' `validate_isam_se2 300` outside
primary timing. Allocation diagnostics append `alloc` to the stream runner;
exclude their times. CI retains correctness checks without timing thresholds.
Bayes-tree/selective updates, full-history growth and field validation remain gaps.

The subsequent [in-place QR change](QR_INPLACE.md) is measured separately.
