# Schur update destination cursor

This follows [bounded QR normalization](QR_NORM.md). A warmed predecessor
classifier observes **4,462,692 update destinations**, including **975,879
known diagonals** and **4,111,485 consecutive destination positions** (diagonals
included) across the synthetic 300-arrival stream. Thus **92.1% of full-column
binary lookups can be avoided**, leaving 351,207 searches. This describes this
factor pattern, not a general sparsity guarantee.

## Cache contract

Optional Schur-update plan construction now starts each source suffix at its
known destination-column diagonal. Subsequent source rows are strictly ascending.
If the next stored row matches, the cursor advances directly. At a gap, binary
search runs only over the remaining destination-column suffix. The checked slice
never crosses into another column; required clique entries remain guaranteed by
symbolic fill. Missing entries retain an internal invariant failure.

The plan contains exactly the same indices in the same order. Numeric assembly,
right-looking factor updates, damping, permutation/fill, entry counts, shared
optional-plan budgets and search fallback are unchanged. Exact CSR matching,
normal-graph rebinding, analysis counters, residual/refinement checks and failed-
refactor readiness remain unchanged. No floating arithmetic, tolerance, API,
dependency, unsafe code or retained storage is added.

A private regression compares every destination with the old full-column binary
lookup for sparse gapped graphs and complete cliques at dimensions 6, 9 and 72,
with natural, scalar minimum-degree and block minimum-degree orders. Diagonal-only
patterns retain an empty valid plan. Two independent Cholesky/full-normal tests
check exact planned/search solution, residual and refinement agreement across
three damping levels and all orders. The dense six-column plan fits at 35 entries
and falls back at 34, retaining the same numeric destinations and budgets.
Existing rebinding, rank, fill, overflow, robust, fixed-lag and iSAM tests pass:
**221 unit/integration tests + 18 doctests**, nine Python quality-oracle tests,
Clippy, formatting and documentation checks.

## Matched complete-update comparison

The near-truth synthetic SE2 input and verified native GTSAM 4.2.1 configuration
follow [NATIVE_ISAM2.md](NATIVE_ISAM2.md). Both ropt kernels retain bounded QR
normalization, sparse LDL with three-coordinate block minimum-degree ordering,
explicit adaptive LM damping reuse and full nonlinear refresh at every arrival.
Initial damping is `1e-3`, gradient tolerance `5e-7`, cost tolerance `1e-15`, step
tolerance `1e-10`, trial limit 200. Input, shared runners and policies are identical.
This is accuracy-conditioned timing, not default cached-update performance.

Twelve rotated/reversed rounds put each candidate in each position four times.
Each process discards one warm-up stream and measures one complete stream:
**72 measured streams / 14,400 timed arrivals** across 100 and 300 added poses.
Every prefix passes unchanged original-gradient `1e-6`, exact fixed-root, native
world-pose mixed metre/radian norm `2e-6`, relative original-cost `1e-9` and relative
reported-cost `1e-10` gates. Before/after estimates, original/reported costs,
LM trials, damping telemetry and QR entries match exactly. Both use **213 / 631
trials**. First-round full mean CSVs are byte-identical; gradient maxima remain
`2.780e-7 / 2.964e-7`, native pose gaps `1.422e-9 / 3.442e-8`. No failed sample or
timing outlier is removed.

Median sum of complete update/report-call times, milliseconds:

| Added poses | Full-column searches | Destination cursor | Native iSAM2 |
| --- | ---: | ---: | ---: |
| 100 | 44.080 | 44.244 | 48.595 |
| 300 | 586.496 | 561.809 | 461.785 |

At 300 the observed median is **4.2% lower**, with ten of twelve pairs faster.
At 100 it is **0.4% higher**, with only five pairs faster: no small-workload gain
is established. Population standard deviations before/after are 2.7 / 3.9 ms
at 100 and 28.6 / 35.9 ms at 300. Variation exceeds the median differences;
no statistical-significance, dependable general speedup or real-time guarantee
is claimed. Native remains **1.22x faster at 300**. The current 100-arrival median
is below native in this campaign; that is not a general superiority/SOTA claim.
Historical sweeps stay separate and are not pooled or compared across conditions.

Timing includes complete updates and owned reports/full means, including native
additional updates and stationarity feedback. Parsing, initialization, recipe
construction, telemetry formatting, external validation, serialization and report
destruction are excluded. Allocation counting is disabled. CPU affinity is 0 and
thread settings one; no compiler/test/other solver overlaps primary samples.
Cloud frequency/co-tenancy remain uncontrolled. The driver verifies native
executable and actual resolved `libgtsam` hashes against the pinned manifest.

- [100 report](results/update_cursor/frames100/report.json),
  [metadata](results/update_cursor/frames100/metadata.json),
  [samples](results/update_cursor/frames100/samples.csv).
- [300 report](results/update_cursor/frames300/report.json),
  [metadata](results/update_cursor/frames300/metadata.json),
  [updates](results/update_cursor/frames300/updates.csv),
  [quality](results/update_cursor/frames300/quality.csv).

## Separate diagnostics

A separate warmed profile observes 300 update-plan constructions: **25.529 ms
before / 10.074 ms after**. Both kernels have all fifteen expected counters and
match every primary non-timing report field. Inclusive scopes overlap; these
single-stream times are excluded from primary comparisons and not interpreted
as additive/causal timing percentages.
[Before stages](results/update_cursor/diagnostics/stages_before.json),
[after stages](results/update_cursor/diagnostics/stages_after.json),
[destination classifier](results/update_cursor/diagnostics/destination_classifier.json),
[counts](results/update_cursor/diagnostics/destination_counts.csv),
[patch reproduction](results/update_cursor/diagnostics/profile_patch_reproduction.json).
All instrumented estimates pass original-energy gates and match primary means.

Complete update/report allocation traffic is unchanged at **2,403,726 requests /
568,194,510 requested bytes** in a separate warmed 300-arrival diagnostic. Means
and non-timing reports match primary runs. Instrumented times are excluded;
this is cumulative traffic, not live/peak/RSS or native memory.
[Record](results/update_cursor/diagnostics/allocations.json).

The 300-arrival cached-policy validator retains 258 cached updates, 30 interval
refreshes, twelve forced loops and twelve AD/dense-QR reference checkpoints.
Before/after traces and summaries are byte-identical. The maximum refreshed
gradient is `4.905e-7`, checkpoint pose gap `8.267e-7`; cached original gradient
is still about `0.0100`. Cached estimates remain approximate between refreshes.
These are correctness checks, not cached-performance timings.
[Record](results/update_cursor/diagnostics/cached_validation.json),
[trace](results/update_cursor/diagnostics/cached_trace.csv),
[summary](results/update_cursor/diagnostics/cached_summary.txt).

## Reproduce

Build optional verified native executable/manifest following [NATIVE_ISAM2.md](NATIVE_ISAM2.md).
Rust has no C++ dependency.

```sh
cargo build -p ropt --example benchmark_warm_isam_se2 \
  --example validate_isam_se2 --release --locked --offline
python3 benchmarks/make_ldl_baseline.py --before-update-cursor \
  --output /tmp/ropt-before-update-cursor
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_update_cursor.py \
  --before /tmp/ropt-before-update-cursor/target/release/examples/benchmark_warm_isam_se2 \
  --before-source /tmp/ropt-before-update-cursor \
  --after target/release/examples/benchmark_warm_isam_se2 \
  --native YOUR_NATIVE_ISAM2_BINARY --native-manifest YOUR_VERIFIED_MANIFEST.json \
  --frames 300 --rounds 12 --output /tmp/update-cursor-comparison
```

Finish builds first, choose a permitted CPU and fresh output directories. The
helper first reverses the later `qr_rebuild.patch`, then `anchor_coordinates.patch`, then `qr_columns.patch`, then `qr_reductions.patch`, then `update_cursor.patch`
for this archived predecessor. Restore the archived after kernel as well when
repeating this historical campaign. The rebuilt predecessor matches the frozen source and
passes six-round 25-arrival native smoke. All twenty stages apply and match archived
kernels in source-only checks; the newest restored kernel is built while older-
stage compilation is skipped. [Record](results/update_cursor/diagnostics/rollback_stages.json).

For cached correctness run both `validate_isam_se2 300` executables separately.
Append `alloc` for allocation diagnostics and exclude timings. Apply
`symbolic_merge_stage_profile.patch` to isolated before/after sources for stage
counts, or `update_cursor_classifier.patch` to the predecessor for destinations.
Use fresh Cargo targets, retain separate binaries, run the 300-arrival input with
arguments `1 1`, and verify counters/non-timing reports before retaining diagnostics.
CI has no timing thresholds. Bayes-tree selective updates, retained history and
field validation remain gaps; maturity stays at the subjective 75%.
