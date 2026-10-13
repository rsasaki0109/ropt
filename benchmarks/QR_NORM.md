# Bounded squared-sum normalization for Givens QR

This follows [sorted symbolic fill and CSR-row planning](SYMBOLIC_MERGE.md).
For maximum input magnitude in `[2^-450, 2^450]`, sparse QR now computes the
Givens norm as `sqrt(a*a+b*b)`. This keeps the aggregate squared sum far from
overflow or underflow to zero. Outside that range it retains standard `hypot`,
including extreme scales, subnormals and nonfinite inputs. Every norm/rotated
coefficient/RHS retains its finite checks. Positive diagonal/rank policy, column
traversal, exact-zero removal, fill budgets and transactional commits are unchanged.
There is no new option, dependency, unsafe code, diagonal perturbation or tolerance.

## Numerical validation

The direct path uses ordinary IEEE rounding and is **not bit-identical to hypot**.
Two private tests compare against standard hypot across 33,568 deterministic
pairs spanning exponents -1074 through 1023, randomized significands and both
signs, plus boundary-adjacent values, extreme magnitudes, zeros, NaN and infinity.
The sampled finite results differ by at most two ULPs. This is a sampled check,
not a universal correctly-rounded certificate. Extreme/nonfinite categories
retain hypot behavior. Existing independent SVD/full-objective, rank, scaled-row,
RHS/coefficient overflow, iSAM and fixed-lag regressions pass.

Three additional regressions retained from the discarded prefix experiment
exercise unequal suffixes, divergence after cancellation, transient fill/recovery
and late norm overflow. In total **218 unit/integration tests + 18 doctests**,
nine Python quality-oracle tests, Clippy, formatting and documentation pass.

## Matched complete-update comparison

The near-truth synthetic SE2 input and verified native GTSAM 4.2.1 configuration
follow [NATIVE_ISAM2.md](NATIVE_ISAM2.md). Both ropt candidates use sparse LDL with
three-coordinate block minimum-degree ordering, explicit adaptive LM damping
reuse and full nonlinear refresh at every arrival. Initial damping is `1e-3`,
gradient tolerance `5e-7`, cost tolerance `1e-15`, step tolerance `1e-10`, trial
limit 200. Input, runners, nonlinear solver and all other changes are identical.
This is accuracy-conditioned timing, not default cached-update performance.

Twelve rotated/reversed rounds place each candidate in each position four times.
Each process discards one warm-up stream and measures one complete stream:
**72 measured streams / 14,400 timed arrivals** across 100 and 300 added poses.
Every prefix passes the unchanged original-gradient `1e-6`, exact fixed-root,
native world-pose mixed metre/radian norm `2e-6`, relative original-cost `1e-9`
and relative reported-cost `1e-10` gates. Before/after full-refresh estimates,
original costs, LM trials, damping telemetry and QR entries match exactly. Both
use **213 / 631 trials**. First-round full mean CSVs are byte-identical; original
gradient maxima remain `2.780e-7 / 2.964e-7`, native pose gaps `1.422e-9 / 3.442e-8`.

Reported QR costs may differ after this arithmetic change: their maximum before/
after absolute gaps are **3.331e-16 / 1.110e-15**, within the unchanged `1e-10`
relative gate. The driver records the distinction explicitly instead of requiring
bit identity for that field. Independent original-energy checks remain required.
No failed sample or timing outlier is removed.

Median sum of complete update/report-call times, milliseconds:

| Added poses | Standard hypot | Bounded squared sum | Native iSAM2 |
| --- | ---: | ---: | ---: |
| 100 | 43.829 | 40.742 | 43.044 |
| 300 | 585.094 | 541.843 | 420.596 |

Observed median reductions are **7.0% / 7.4%**, with eleven / all twelve pairs
faster at 100 / 300. Population standard deviations before/after are 1.9 / 1.3 ms
and 22.3 / 13.6 ms. At 100, ropt's median is **5.3% lower than native** in this
campaign; native remains **1.29x faster at 300**. This small synthetic workload
establishes no general superiority, SOTA, real-time or statistical-significance
claim. Historical and discarded campaigns are retained separately, not pooled.

Timing includes complete updates and owned reports/full means, including native
additional updates and stationarity feedback. Parsing, initialization, recipe
construction, telemetry formatting, external validation, serialization and report
destruction are excluded. Allocation counting is disabled. CPU affinity is 0 and
thread settings one, with no compiler/test/other solver overlap during primary
samples. Cloud frequency/co-tenancy remain uncontrolled. The driver checks the
native executable and actual resolved `libgtsam` hashes against its manifest.

- [100 report](results/qr_norm/frames100/report.json),
  [metadata](results/qr_norm/frames100/metadata.json),
  [samples](results/qr_norm/frames100/samples.csv).
- [300 report](results/qr_norm/frames300/report.json),
  [metadata](results/qr_norm/frames300/metadata.json),
  [updates](results/qr_norm/frames300/updates.csv),
  [quality](results/qr_norm/frames300/quality.csv).

## Separate diagnostics and discarded approaches

A separate warmed profile records 301 QR append calls: about **252 / 235 ms**
before/after. Both instrumented kernels have all fifteen expected counters and
match every primary non-timing report field. Inclusive scopes overlap; these
single-stream times are excluded from primary comparisons, not summed or used
as causal percentages. [Before](results/qr_norm/diagnostics/stages_before.json),
[after](results/qr_norm/diagnostics/stages_after.json),
[patch reproduction](results/qr_norm/diagnostics/profile_patch_reproduction.json).
An earlier diagnostic with missing counters was excluded and rebuilt; it supplies
no timing evidence. Use fresh Cargo target directories for isolated snapshots.

A separate warmed complete-update/report allocation diagnostic gives **2,403,726
requests / 568,194,510 bytes** for both candidates. Means and reports match each
kernel's own primary run. This is unchanged cumulative traffic; peak/live/RSS
and native memory are unmeasured. [Record](results/qr_norm/diagnostics/allocations.json).

Both 300-arrival cached-policy validators pass twelve AD/dense-QR checkpoints,
with 258 cached updates, 30 interval refreshes, twelve forced loops and no
threshold refreshes. Reasons and QR entry counts match. Floating traces differ:
maximum before/after nonlinear-cost difference is `5.152e-14`, original-gradient
difference `6.729e-9`. The new maximum checkpoint pose gap is `8.267e-7`, refreshed
gradient `4.905e-7`; cached original gradient is still about `0.0100`. Cached
estimates remain approximate between refreshes. These are correctness checks,
not cached performance timing. [Record](results/qr_norm/diagnostics/cached_validation.json),
[before summary](results/qr_norm/diagnostics/cached_before_summary.txt),
[after summary](results/qr_norm/diagnostics/cached_after_summary.txt).

The earlier one-sided prefix candidate had a 4.5% slowdown at 100 and only a 1.7%
reduction at 300, with three / seven faster pairs. It was reverted. Its raw
campaign, fingerprint and forward patch are preserved in the
[decision record](results/qr_prefix/attempt_one_sided/decision.json). Reproduce it
by applying `qr_prefix.patch` forward to a standard-hypot predecessor snapshot.
It is benchmark-only and is not part of the production rollback chain.

Simulated per-append direct-mapped norm-input caches with 64, 256 and 1024 slots
had zero hits in 2,998,044 rotations, including 135,450 zero diagonals. No cache
was introduced. This does not prove global input uniqueness.
[Classifier](results/qr_norm/diagnostics/norm_cache_classifier.json),
[counts](results/qr_norm/diagnostics/norm_cache_counts.csv).

## Reproduce

Build the optional verified native executable/manifest following
[NATIVE_ISAM2.md](NATIVE_ISAM2.md); Rust has no C++ dependency.

```sh
cargo build -p ropt --example benchmark_warm_isam_se2 \
  --example validate_isam_se2 --release --locked --offline
python3 benchmarks/make_ldl_baseline.py --before-qr-norm \
  --output /tmp/ropt-before-qr-norm
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_qr_norm.py \
  --before /tmp/ropt-before-qr-norm/target/release/examples/benchmark_warm_isam_se2 \
  --before-source /tmp/ropt-before-qr-norm \
  --after target/release/examples/benchmark_warm_isam_se2 \
  --native YOUR_NATIVE_ISAM2_BINARY --native-manifest YOUR_VERIFIED_MANIFEST.json \
  --frames 300 --rounds 12 --output /tmp/qr-norm-comparison
```

Finish builds first, choose a permitted CPU and fresh output directories. The
helper reverses the later `qr_rebuild.patch`, then `anchor_coordinates.patch`, then `qr_columns.patch`, then `qr_reductions.patch`, then `update_cursor.patch`, then `qr_norm.patch` for this
predecessor, and reverses those first for older stages. The restored kernel matches the frozen predecessor and
passes six-round 25-arrival native smoke. All nineteen stages apply and match
archived kernels in source-only checks; the latest restored kernel is built while
older-stage compilation is skipped. [Record](results/qr_norm/diagnostics/rollback_stages.json).

For cached correctness run both `validate_isam_se2 300` executables separately.
Append `alloc` for allocation diagnostics and exclude their timings. Apply
`symbolic_merge_stage_profile.patch` to isolated before/after sources for stage
counts, or `givens_cache_classifier.patch` to the predecessor for cache simulation;
build in fresh target directories and retain separate binaries. Run the 300-arrival
input with arguments `1 1` and verify counters/reports before retaining diagnostics.
CI has no timing thresholds. Bayes-tree selective updates, retained history and
field validation remain gaps; maturity stays at the subjective 75%.

The subsequent [Schur update destination cursor](UPDATE_CURSOR.md) is measured separately.
