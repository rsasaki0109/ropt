# Sorted fill insertion and repeated CSR-row planning

This follows [aligned QR row reuse](QR_ALIGNED.md). A separate warmed
300-arrival predecessor profile spent about 43 ms building the factor pattern,
21 ms building the normal pattern and 21 ms preparing assembly destinations.
These are instrumented inclusive stages, not primary timings or additive shares.

## Structural and numerical contracts

Short symbolic columns now insert a strictly ascending fill suffix using an
advancing cursor. This replaces a binary search for every destination while
keeping ascending insertion, duplicate detection and per-new-entry budget checks.
Columns promote from vector to tree at the same 64-entry threshold; tree insertions
retain their existing path. The cumulative fill budget does not refund entries
when columns are eliminated. Ordering, tie rules, fill and stored indices are unchanged.

Consecutive CSR rows with exactly equal column indices need no repeated normal-
pattern insertion. When an assembly plan fits, their previous destination block
is copied, without repeating permutation/entry lookups. Every residual retains
its own complete destination block and operation count. This does **not** deduplicate
numeric rows: every product is accumulated in its original row/column order.
Empty-row boundaries and equal widths with different column indices are distinct.
The optional update/assembly plans share their unchanged entry budget and fall
back to searches when they do not fit.

Exact CSR matching, normal-graph rebinding, public analysis/entry counters,
numeric refactoring, damping and original-operator residual/refinement checks
stay unchanged. Failures still invalidate numeric readiness and smoother updates
still commit only on success. There is no unsafe code, new API/dependency,
retained cache, tolerance or refresh-policy change.

Two new regressions compare repeated supports with independent dense Cholesky,
including empty boundaries, different supports of equal width, changed numeric
values and all three orderings. Planned and search-based factors give exactly
equal solutions/residuals/refinement counts; all 126 per-row assembly entries
remain present. A 72-column star with natural ordering exercises vector-to-tree
promotion during clique fill, exact factor budget and one-entry-too-small failure.
Existing normal-rebinding/plan-restoration, overflow/readiness, robust, rank,
large-star, fixed-lag and iSAM tests pass: **213 unit/integration tests + 18 doctests**,
nine Python quality-oracle tests, Clippy, formatting and documentation checks.

## Matched complete-update comparison

The near-truth synthetic SE2 input and verified native GTSAM 4.2.1 configuration
follow [NATIVE_ISAM2.md](NATIVE_ISAM2.md). Both ropt candidates use sparse LDL
with three-coordinate block minimum-degree ordering, explicit adaptive LM damping
reuse and full nonlinear refresh after every arrival. Initial damping is `1e-3`,
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

| Added poses | Individual insertion/lookups | Sorted insertion/row reuse | Native iSAM2 |
| --- | ---: | ---: | ---: |
| 100 | 43.614 | 42.657 | 43.156 |
| 300 | 625.043 | 591.656 | 445.572 |

Observed median reductions are **2.2% / 5.3%**; nine / eleven of twelve paired
rounds are faster at 100 / 300. Population standard deviations before/after are
0.8 / 1.8 ms at 100 and 26.2 / 24.0 ms at 300. Timing remains variable; the small
100-arrival difference does not establish a dependable gain. The current 100-
arrival median is 1.2% below native, but means/variation establish no superiority.
Native remains **1.33x faster at 300**. Historical sweeps are separate, not pooled.
There is no statistical-significance, real-time, per-update or general SLAM claim.

Timing includes complete updates and owned reports/full means, including native
additional updates and stationarity feedback. Parsing, initialization, recipe
construction, telemetry formatting, external validation, serialization and report
destruction are excluded. Allocation counting is disabled. CPU affinity is 0 and
thread settings one, with no compiler/test/other solver overlap during primary
samples. Cloud frequency/co-tenancy remain uncontrolled. The driver verifies the
native executable and actual resolved `libgtsam` hashes against its manifest.

Raw samples, every-prefix time/quality/telemetry, input, options, compressed means
and source/runner/executable/library fingerprints are retained:

- [100 report](results/symbolic_merge/frames100/report.json),
  [metadata](results/symbolic_merge/frames100/metadata.json),
  [samples](results/symbolic_merge/frames100/samples.csv).
- [300 report](results/symbolic_merge/frames300/report.json),
  [metadata](results/symbolic_merge/frames300/metadata.json),
  [updates](results/symbolic_merge/frames300/updates.csv),
  [quality](results/symbolic_merge/frames300/quality.csv).

## Separate diagnostics

A separate warmed final profile uses the same fifteen scopes and identical call
counts: 300 full preparations, 183 normal-graph rebindings and 631 numeric factors.
Selected inclusive stage times, milliseconds:

| Stage | Calls | Before | After |
| --- | ---: | ---: | ---: |
| Normal pattern | 483 | 20.958 | 13.837 |
| Factor pattern | 300 | 43.284 | 26.217 |
| Assembly plan | 483 | 20.893 | 13.633 |
| Update plan | 300 | 23.655 | 23.880 |

Both instrumented means match primary estimates byte for byte and pass original-
energy checks. Scopes overlap; these single-stream diagnostics are not summed,
pooled with primary samples or interpreted as causal timing percentages.
[Before stages](results/symbolic_merge/diagnostics/stages_before.csv),
[after stages](results/symbolic_merge/diagnostics/stages_after.csv),
[before provenance](results/symbolic_merge/diagnostics/stages_before.json),
[after provenance](results/symbolic_merge/diagnostics/stages_after.json),
[forward-patch reproduction](results/symbolic_merge/diagnostics/profile_patch_reproduction.json).

A separate warmed 300-arrival allocation run counts complete update/report calls.
Both candidates request **2,403,726 allocations / 568,194,510 bytes**: cumulative
allocation traffic is unchanged. This reduces structural search work. Means and
every non-timing report field match primary results. Instrumented times are
excluded; live/peak/RSS and native memory are unmeasured.
[Allocation record](results/symbolic_merge/diagnostics/allocations.json).

The 300-arrival cached-policy validator includes 258 cached updates, 30 interval
refreshes, 12 forced loop refreshes and twelve AD/dense-QR reference checkpoints.
Before/after traces and summaries are byte-identical. Cached estimates remain
approximate between refreshes, with maximum original gradient about `0.0100`
(unchanged). These are correctness checks, not cached-update timings.
[Record](results/symbolic_merge/diagnostics/cached_validation.json),
[trace](results/symbolic_merge/diagnostics/cached_trace.csv),
[summary](results/symbolic_merge/diagnostics/cached_summary.txt).

## Reproduce

Build the optional verified native executable/manifest following
[NATIVE_ISAM2.md](NATIVE_ISAM2.md). Rust has no C++ dependency.

```sh
cargo build -p ropt --example benchmark_warm_isam_se2 \
  --example validate_isam_se2 --release --locked --offline
python3 benchmarks/make_ldl_baseline.py --before-symbolic-merge \
  --output /tmp/ropt-before-symbolic-merge
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_symbolic_merge.py \
  --before /tmp/ropt-before-symbolic-merge/target/release/examples/benchmark_warm_isam_se2 \
  --before-source /tmp/ropt-before-symbolic-merge \
  --after target/release/examples/benchmark_warm_isam_se2 \
  --native YOUR_NATIVE_ISAM2_BINARY --native-manifest YOUR_VERIFIED_MANIFEST.json \
  --frames 300 --rounds 12 --output /tmp/symbolic-merge-comparison
```

Finish builds first, choose a permitted CPU and fresh output directories. The
helper reverses the later `qr_rebuild.patch`, then `anchor_coordinates.patch`, then `qr_columns.patch`, then `qr_reductions.patch`, then `update_cursor.patch`, then `qr_norm.patch`, then `symbolic_sorted_merge.patch`
for this predecessor;
older stages reverse it first, then their existing patches. The rebuilt restored
kernel matches the frozen predecessor and passes six-round 25-arrival native
smoke. All eighteen stages apply and match archived kernels in source-only checks;
older-stage compilation is skipped while the latest restored kernel is built separately.
[Rollback record](results/symbolic_merge/diagnostics/rollback_stages.json).

For cached correctness run both candidates' `validate_isam_se2 300` separately.
Allocation diagnostics append `alloc` to the runner and exclude their timings.
For stage times apply `symbolic_merge_stage_profile.patch` forward to isolated
before/after snapshots, rebuild `benchmark_warm_isam_se2`, then run the 300-arrival
input with arguments `1 1`. Keep instrumented binaries separate. CI checks
correctness without timing thresholds. Selective Bayes-tree updates, retained
history and field validation remain gaps.

The subsequent [bounded QR normalization](QR_NORM.md) is measured separately.
