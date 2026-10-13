# Aligned QR finite and zero reductions

This follows the retained [Schur update cursor](UPDATE_CURSOR.md) and the
[rejected normal-pattern experiments](NORMAL_PATTERN_EXPERIMENTS.md).
A warmed predecessor classifier observes **2,998,044 rotations**, including
**1,424,980 aligned suffixes / 19,850,690 aligned entries** in the synthetic
300-arrival stream. The existing index/length guard remains unchanged.
[Classifier](results/qr_reductions/diagnostics/suffix_classifier.json),
[length distribution](results/qr_reductions/diagnostics/suffix_counts.csv).

## Numeric and failure contract

Each aligned column still computes `a = c*upper + s*lower` and
`v = -s*upper + c*lower` in the same order. Finite validity, the number of upper
zeros and the presence of lower zeros are collected as integer/boolean
reductions. The independent floating operations have no cross-column floating
reduction. Norms, rotation coefficients, RHS operations, exact-zero compaction,
fallback merges, transient fill checks and rank tolerances are unchanged.

Finite validation now occurs after each aligned suffix. A failure can therefore
process more entries of that suffix before returning `NonFinite`. Validation
completes before compaction, the next rotation or RHS update. Public QR append
and smoother updates discard the private candidate on failure, preserving the
committed factor, estimates, RHS, costs and entry counts. No new allocation,
retained storage, public API, dependency, unsafe code or worker thread is added.

In this x86-64 release build, the compiler emits packed floating operations for
the new loop without manual SIMD or `target-cpu=native`. Static disassembly counts
for the complete append function increase from one to nine `mulpd` instructions
and from zero to two each of `addpd`/`subpd`. These are static instruction counts,
not executed operation counts or a speedup estimate; other targets/toolchains
may make different choices. [Compiler record](results/qr_reductions/diagnostics/compiler.json).

Three new public regressions cover mixed cancellations at widths 2, 3, 7, 16 and
65, exact batch/split solution and cost bits, and independent dense SVD/objective
agreement. Overflow at the first, middle and last suffix entries rolls back at
the exact fill limit and permits a later successful append. Scales `1e-290`,
`1` and `1e290` retain rank, zero solutions and scaled objectives.
[Tests](../crates/ropt/tests/qr_reductions.rs).
The final suite passes **226 unit/integration tests + 18 doctests**, nine Python
oracle tests, Clippy with warnings denied, formatting and documentation checks.
[Final verification](results/qr_reductions/diagnostics/validation_checks.json).

## Matched complete-update comparison

Both ropt kernels explicitly reuse LM damping and force nonlinear refresh at
every arrival. They use sparse LDL with three-coordinate block minimum-degree
ordering, initial damping `1e-3`, gradient tolerance `5e-7`, cost tolerance `1e-15`,
step tolerance `1e-10` and 200 trials. Input and shared runners are identical.
This is accuracy-conditioned timing, not default cached-policy performance.
The native GTSAM 4.2.1 configuration is pinned in [NATIVE_ISAM2.md](NATIVE_ISAM2.md);
the driver checks executable and actual resolved library hashes against its manifest.

Twelve rotated/reversed rounds at each size put every variant in every position
four times. Each process discards one warm-up stream and measures one complete
stream: **72 measured streams / 14,400 timed arrivals**. Six-round 25-arrival
native smoke is separate from primary results.

| Added poses | Before median ms | After median ms | Native median ms | After faster pairs |
| --- | ---: | ---: | ---: | ---: |
| 100 | 43.312 | 43.981 | 47.253 | 6 / 12 |
| 300 | 574.886 | 528.683 | 467.253 | 11 / 12 |

The 300-arrival median decreases **8.0%**. At 100 it increases **1.5%**, with no
dependable benefit demonstrated. Standard deviations are 49.2 / 32.5 ms at 300
and 6.79 / 4.96 ms at 100; cloud frequency and co-tenancy remain uncontrolled.
Native remains **1.13x faster at 300** in this campaign. These synthetic results
do not establish general SLAM superiority or mature iSAM2 equivalence.

Every prefix passes unchanged original-gradient `1e-6`, exact fixed-root, native
pose-gap `2e-6`, original-cost relative `1e-9` and reported-cost relative `1e-10`
gates. Before/after means, original and QR costs, entry counts, LM trials and
damping sequences match exactly in every round. Maximum ropt gradients are
`2.780e-7` / `2.964e-7`; maximum native pose gaps are `1.422e-9` / `3.442e-8` at
100 / 300. LM trial totals remain 213 / 631.

Timers include complete accuracy-target updates and owned means/report creation,
including native empty updates, stationarity feedback and full cost. Parsing,
initialization, recipe construction, external validation, serialization and
report destruction are excluded. Allocation counting is disabled. CPU affinity
is 0 with one-thread settings; no compiler, test or other solver overlaps primary samples.

- [100 report](results/qr_reductions/frames100/report.json),
  [metadata](results/qr_reductions/frames100/metadata.json),
  [samples](results/qr_reductions/frames100/samples.csv).
- [300 report](results/qr_reductions/frames300/report.json),
  [metadata](results/qr_reductions/frames300/metadata.json),
  [updates](results/qr_reductions/frames300/updates.csv),
  [quality](results/qr_reductions/frames300/quality.csv).

Each directory also retains the input, options and compressed full means.

## Separate diagnostics

A final serial warmed profile observes 301 QR appends: **228.179 ms before /
220.370 ms after**. All fifteen counters and each kernel's non-timing reports
match primary runs. Inclusive scopes overlap and instrumentation perturbs the
executable; these single-stream times are excluded from primary comparisons and
cannot explain an additive or causal fraction of the overall timing difference.
An earlier profile pair that could overlap a separate historical smoke check is
excluded from retained timings.
[Before stages](results/qr_reductions/diagnostics/stages_before.json),
[after stages](results/qr_reductions/diagnostics/stages_after.json),
[patch proof](results/qr_reductions/diagnostics/profile_patch_reproduction.json).

Complete update/report allocation traffic is unchanged at **2,403,726 requests /
568,194,510 requested bytes** in a separate 300-arrival diagnostic. Means and
non-timing reports match primary runs. This is cumulative traffic, not live/peak/
RSS or native memory. [Record](results/qr_reductions/diagnostics/allocations.json).

The 300-arrival cached validator has byte-identical before/after traces and
summaries: 258 cached updates, 30 interval refreshes, twelve forced loops and
twelve AD/dense-QR checkpoints. Maximum refreshed gradient is `4.905e-7` and
checkpoint pose gap `8.267e-7`. Cached original gradient remains about `0.0100`:
estimates between refreshes are approximate. These are correctness checks,
not cached-performance timings.
[Record](results/qr_reductions/diagnostics/cached_validation.json),
[trace](results/qr_reductions/diagnostics/cached_trace.csv),
[summary](results/qr_reductions/diagnostics/cached_summary.txt).

## Reproduce

Build the optional verified native executable/manifest following
[NATIVE_ISAM2.md](NATIVE_ISAM2.md). Rust has no C++ dependency.

```sh
cargo build -p ropt --example benchmark_warm_isam_se2 \
  --example validate_isam_se2 --release --locked --offline
python3 benchmarks/make_ldl_baseline.py --before-qr-reductions \
  --output /tmp/ropt-before-qr-reductions
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_qr_reductions.py \
  --before /tmp/ropt-before-qr-reductions/target/release/examples/benchmark_warm_isam_se2 \
  --before-source /tmp/ropt-before-qr-reductions \
  --after target/release/examples/benchmark_warm_isam_se2 \
  --native YOUR_NATIVE_ISAM2_BINARY --native-manifest YOUR_VERIFIED_MANIFEST.json \
  --frames 300 --rounds 12 --output /tmp/qr-reductions-comparison
```

Finish builds first, choose a permitted CPU and fresh output directories.
The helper now reverses the later `qr_rebuild.patch`, then `anchor_coordinates.patch`, then `qr_columns.patch` first, then
`qr_reductions.patch` for this archived predecessor. Its archived after kernel
is restored by `--before-qr-columns`; restore both adjacent sources for a
historical comparison. All 21 source-only rollback stages match archived kernels;
only the newest predecessor is built and smoke-tested here.
[Rollback record](results/qr_reductions/diagnostics/rollback_stages.json).
For an archived campaign, restore its after source as well as its before source
and verify both against that campaign's metadata before attributing its timings
to a single change. The rejected normal-pattern helper accepts `--source` for
this purpose; both archived candidates are rebuilt and smoke-tested.
[Historical proof](results/qr_reductions/diagnostics/historical_normal_reproduction.json).

Run both `validate_isam_se2 300` binaries for cached correctness. Append `alloc`
to the warm runner for separate allocation diagnostics, excluding their times.
Apply `symbolic_merge_stage_profile.patch` to isolated before/after sources for
stage counts, or `qr_reductions_classifier.patch` to the predecessor for suffix
counts. Use fresh targets; verify counters, estimates and non-timing reports.
[Classifier patch proof](results/qr_reductions/diagnostics/classifier_patch_reproduction.json).
CI has no timing thresholds. Bayes-tree selective updates, retained history and
field validation remain gaps; subjective maturity stays at 75%.
