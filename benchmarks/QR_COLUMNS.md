# Exact contiguous QR column matching

This follows [aligned QR reductions](QR_REDUCTIONS.md). The numeric rotation
loop is unchanged. A private alignment predicate uses the existing invariant
that both live suffixes contain strictly ascending, unique column indices.

## Structural proof and contract

Different lengths fail immediately. Empty and single-column suffixes are handled
directly. For larger suffixes, first and last columns must match. If
`last_column - first_column == length - 1`, strict ordering and uniqueness prove
that both suffixes contain every integer in that range, so no interior scan is
needed. Gapped ranges still compare every interior index; equal lengths and
endpoints alone cannot imply equality. Two-column gapped suffixes have no interior.
Subtraction avoids an overflowing `first + length` calculation near `usize::MAX`.

CSR construction, sorted Givens merges and stable zero compaction already
maintain the ordering invariant. No numeric arithmetic, finite/rank checks,
exact-zero policy, fallback merge, transient fill budget, public API, dependency,
allocation or retained storage changes. Both choices select exactly the same
aligned/fallback branch. Failure transactions remain unchanged.

Two private tests compare all pairs of seven-column subset masks at bases zero
and `usize::MAX-6`: **32,768 pairs**, with an independent mask-equality oracle.
They cover gaps, different widths and matching endpoints with different interiors.
Additional widths 63, 64, 65, 127, 128 and 257 cover matching contiguous/gapped
supports and mutations of endpoints/interiors. Existing independent SVD/objective,
overflow, fill-budget and batch/split-bit tests pass. The final suite has
**228 unit/integration tests + 18 doctests**, nine Python oracle tests, Clippy,
formatting and documentation validation.
[Final verification](results/qr_columns/diagnostics/validation_checks.json).

## Actual comparison work

A warmed predecessor classifier checks the proposed predicate against the
original decision for every equal-width suffix in the 300-arrival stream:

| Counter | Count |
| --- | ---: |
| Rotations | 2,998,044 |
| Equal-width candidate suffixes | 1,654,144 |
| Aligned suffixes | 1,424,980 |
| Nonempty contiguous aligned suffixes | 382,166 |
| Gapped aligned suffixes | 1,032,842 |
| Original index comparisons | 20,538,182 |
| Proposed index comparisons | 19,846,214 |

Only about 26.8% of aligned suffixes are nonempty contiguous ranges. The avoided
691,968 index comparisons are **3.4%** of the original comparisons, not a large
reduction in overall QR work. Most wide suffixes remain gapped. This describes
this input, not a general sparsity guarantee or a causal timing percentage.
Instrumentation is excluded from primary timings and reproduces every primary
mean and non-timing report field.
[Classifier](results/qr_columns/diagnostics/column_classifier.json),
[counts](results/qr_columns/diagnostics/column_counts.csv),
[forward patch proof](results/qr_columns/diagnostics/classifier_patch_reproduction.json).

## Matched complete-update comparison

Both ropt kernels force nonlinear refresh at every arrival and explicitly reuse
LM damping. They retain sparse LDL with block minimum-degree size three, initial
damping `1e-3`, gradient tolerance `5e-7`, cost tolerance `1e-15`, step tolerance
`1e-10` and 200 trials. This is accuracy-conditioned complete-stream timing,
not default cached-update timing. Verified native GTSAM 4.2.1 follows
[NATIVE_ISAM2.md](NATIVE_ISAM2.md); executable and actual resolved library hashes
are checked against its manifest.

Twelve rotated/reversed rounds at each size place each variant in each position
four times. Every process discards one warm-up stream and measures one full
stream: **72 measured streams / 14,400 timed arrivals**. Separate six-round
25-arrival native smoke passes.

| Added poses | Before total median ms | After total median ms | Native total median ms | After faster pairs |
| --- | ---: | ---: | ---: | ---: |
| 100 | 40.201 | 40.638 | 44.856 | 6 / 12 |
| 300 | 542.757 | 533.088 | 438.150 | 10 / 12 |

The 300-arrival median is **1.8% lower**; the 100-arrival median is **1.1% higher**.
Standard deviations at 300 are 39.3 / 27.3 ms, larger than the observed gap.
At 100 they are 10.4 / 3.56 ms, including a retained 77.7 ms predecessor sample.
No dependable general speedup or small-workload benefit is established.
Native is **1.22x faster at 300** in this campaign. Compare each kernel with its
matched predecessor; differences between campaigns are not causal regressions
or improvements. These synthetic observations do not establish general SLAM superiority.

Every prefix passes unchanged original-gradient `1e-6`, exact fixed-root, native
pose-gap `2e-6`, original-cost relative `1e-9` and reported-cost relative `1e-10`
gates. Before/after means, original and QR costs, stored entries, LM trials and
damping sequences match exactly. Maximum ropt gradients are `2.780e-7` / `2.964e-7`
and native pose gaps `1.422e-9` / `3.442e-8` at 100 / 300. LM trials remain 213 / 631.

Timers include complete accuracy-target updates and owned means/report creation,
including native empty updates, stationarity feedback and full cost. Parsing,
initialization, recipe construction, external validation, serialization and
report destruction are excluded. Allocation counting is disabled. CPU affinity
is 0 and thread settings one; no compiler, test or other solver overlaps primary
samples. Cloud frequency and co-tenancy remain uncontrolled.

- [100 report](results/qr_columns/frames100/report.json),
  [metadata](results/qr_columns/frames100/metadata.json),
  [samples](results/qr_columns/frames100/samples.csv).
- [300 report](results/qr_columns/frames300/report.json),
  [metadata](results/qr_columns/frames300/metadata.json),
  [updates](results/qr_columns/frames300/updates.csv),
  [quality](results/qr_columns/frames300/quality.csv).

Input, options and compressed full means are retained alongside these records.

## Separate diagnostics

A serial warmed profile observes 301 QR appends: **220.148 ms before /
208.624 ms after**. All fifteen counters and every primary non-timing report
field are verified. Scopes overlap and instrumentation perturbs code generation;
single-stream inclusive times are excluded from primary comparisons and cannot
explain an additive or causal fraction of the overall difference.
[Before stages](results/qr_columns/diagnostics/stages_before.json),
[after stages](results/qr_columns/diagnostics/stages_after.json),
[patch proof](results/qr_columns/diagnostics/profile_patch_reproduction.json).

Complete update/report allocation traffic is unchanged at **2,403,726 requests /
568,194,510 requested bytes**. Instrumented means/non-timing reports match primary
runs. This is cumulative traffic, not live/peak/RSS or native memory.
[Record](results/qr_columns/diagnostics/allocations.json).

The cached 300-arrival trace and summary are byte-identical: 258 cached updates,
30 interval refreshes, twelve forced loops and twelve AD/dense-QR checkpoints.
Maximum refreshed gradient is `4.905e-7`, checkpoint pose gap `8.267e-7`.
The cached original gradient remains about `0.0100`; estimates between refreshes
are approximate. These are correctness checks, not cached-performance timings.
[Record](results/qr_columns/diagnostics/cached_validation.json),
[trace](results/qr_columns/diagnostics/cached_trace.csv),
[summary](results/qr_columns/diagnostics/cached_summary.txt).

## Reproduce

Build the optional verified native executable/manifest following
[NATIVE_ISAM2.md](NATIVE_ISAM2.md). Rust has no C++ dependency.

```sh
python3 benchmarks/make_ldl_baseline.py --before-anchor-coordinates \
  --output /tmp/ropt-qr-columns-archived-after
python3 benchmarks/make_ldl_baseline.py --before-qr-columns \
  --output /tmp/ropt-before-qr-columns
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_qr_columns.py \
  --before /tmp/ropt-before-qr-columns/target/release/examples/benchmark_warm_isam_se2 \
  --before-source /tmp/ropt-before-qr-columns \
  --after /tmp/ropt-qr-columns-archived-after/target/release/examples/benchmark_warm_isam_se2 \
  --after-source /tmp/ropt-qr-columns-archived-after \
  --native YOUR_NATIVE_ISAM2_BINARY --native-manifest YOUR_VERIFIED_MANIFEST.json \
  --frames 300 --rounds 12 --output /tmp/qr-columns-comparison
```

Finish builds first, choose an allowed CPU and fresh output directories. The
helper now reverses the later `qr_rebuild.patch`, then `anchor_coordinates.patch` before `qr_columns.patch`
for this predecessor. `--before-anchor-coordinates` restores this archived after kernel.
Both fingerprints must match the saved campaign metadata. In the original
campaign, all 22 source-only rollback stages matched archived kernels; only the
newest predecessor was built and smoke-tested. Older compilation was skipped.
[Record](results/qr_columns/diagnostics/rollback_stages.json).
For an archived campaign restore both adjacent kernels and verify both source
fingerprints against its saved metadata before attributing timings to one change.

Run both `validate_isam_se2 300` binaries for cached correctness. Append `alloc`
to the warm runner for separate allocation diagnostics and exclude their times.
Apply `symbolic_merge_stage_profile.patch` to isolated before/after sources for
stage counts, or `qr_columns_classifier.patch` to the predecessor for comparisons.
Use fresh targets and verify counters, estimates and non-timing reports.
CI has no timing thresholds. Bayes-tree selective updates and field validation
remain gaps; subjective maturity stays at 75%.
