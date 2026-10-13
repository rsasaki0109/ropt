# Rebind CSR assembly for an exactly equal normal graph

This follows [compact ordering adjacency](ORDERING_ADJACENCY.md). Reprofiling
showed sparse LDL spending substantially more time preparing structure than
performing numeric factorization. In a separate warmed 300-arrival predecessor
run, 631 numeric factorizations took about 23 ms while 483 full preparations took
about 174 ms. These are instrumented inclusive stages, not primary timings or
additive percentages. A separate classifier found **300 initial analyses and
183 changed CSR patterns with the same normal graph; none changed that graph**.

## Cache contract

Symbolic analysis now retains an exact original-coordinate lower normal graph
as offsets and sorted row indices. When the CSR shape/offsets/indices change,
the workspace rebuilds the current normal graph under the same assembly budget
and compares every index, without hashes or collision assumptions. Equal graphs
reuse the same permutation, elimination fill, Schur update destinations and
numeric buffers. Only assembly destinations and the exact CSR descriptor are
refreshed. Different column counts or normal edges take the full analysis path.
This can repeat normal-pattern assembly on the changed-graph fallback; no general
speedup for frequently changing connectivity is claimed.

`SparseLdlSymbolic::matches_pattern` and standalone `factor_normal` retain exact
CSR matching and still reject a different descriptor. Only the owning workspace
performs the checked rebinding. `analyses()` continues counting every successful
CSR-pattern analysis, including rebindings, so its observable count is preserved.
Values/damping are always numerically refactored, and original-operator residual
checks/refinement are unchanged. Failed refactors invalidate numeric readiness.

Assembly/update destination budgets retain their existing shared entry limit.
A refreshed assembly plan that does not fit uses the same search path and
arithmetic; later fitting patterns can restore the plan. Normal graph signatures
add `dimension+1` offsets and `normal_entries` row indices per symbolic object.
This is extra retained storage; cumulative allocation traffic is not peak memory.
There is no unsafe code, tolerance/policy change, public option or new dependency.

Three regressions compare rebound solves and residual/refinement statistics
**exactly** against fresh LDL and independently against dense Cholesky. They
cover row count/order changes, repeated rows, assembly-plan fit/search/restoration,
unchanged factor/update-buffer addresses, equal edge counts with different edges,
dimension changes, strict standalone matching, numeric overflow after rebinding,
readiness invalidation and recovery. Existing zero-pattern, ordering, robust,
rank, fill, fixed-lag and transactional tests pass. There are **208 unit/integration
tests + 18 doctests**, nine Python quality-oracle tests, and passing Clippy,
formatting and documentation checks.

## Matched complete-update measurements

The near-truth synthetic SE2 input and verified native GTSAM 4.2.1 configuration
follow [NATIVE_ISAM2.md](NATIVE_ISAM2.md). Both candidates retain compact adjacency,
whitening reuse, in-place QR, direct CSR, graph retraction and explicit adaptive
LM damping reuse. Shared runners, input bytes, sparse LDL/block ordering and
full nonlinear refresh after every arrival are unchanged. Initial damping is
`1e-3`, gradient tolerance `5e-7`, cost tolerance `1e-15`, step tolerance `1e-10`
and trial limit 200. This is accuracy-conditioned, not default-policy timing.

Twelve rotated/reversed rounds place each candidate in each position four times.
Each process discards one warm-up stream and measures one complete stream:
**72 measured streams / 14,400 timed arrivals** at 100 and 300 added poses.
Original gradients stay at most `1e-6`, the fixed root matches exactly, native
world poses within mixed metre/radian norm `2e-6`, original costs within relative
`1e-9` and reported costs within `1e-10`. Every prefix passes unchanged gates.
Before/after estimates, costs, LM trials, damping telemetry and QR entries match
exactly. Both use **213 / 631 trials**, and first-round full mean CSVs match byte
for byte. Original gradient maxima remain `2.780e-7 / 2.964e-7`; native pose gaps
`1.422e-9 / 3.442e-8`. No failed sample or timing outlier is removed.

Median sum of complete update/report-call times, milliseconds:

| Added poses | Full preparation on CSR change | Normal-graph rebinding | Native iSAM2 |
| --- | ---: | ---: | ---: |
| 100 | 49.729 | 48.908 | 46.171 |
| 300 | 702.703 | 649.558 | 432.178 |

Observed median reductions are **1.7% / 7.6%**, with nine / all twelve paired
rounds faster at 100 / 300. Standard deviations (denominator N) are 3.2 / 2.6 ms
before/after at 100 and 48.7 / 29.9 ms at 300. Timing remains variable; these
observations establish no statistical-significance, real-time, per-update or
general SLAM guarantee. Native remains **1.06x / 1.50x faster**. Historical
sweeps are retained separately and not pooled.

Timing includes complete updates and owned reports/full means, including native
additional updates and stationarity feedback. Parsing, initialization, recipe
creation, telemetry formatting, external validation, serialization and report
destruction are outside clocks. Allocation counting is disabled. CPU affinity
is 0 and thread settings one; no compiler/test/other solver overlaps primary
samples. Cloud frequency/co-tenancy remain uncontrolled. Native executable and
actual resolved `libgtsam` hashes are verified by the driver.

Raw samples, every-prefix time/quality/telemetry, input, options, compressed means
and source/runner/executable/library fingerprints:

- [100 report](results/normal_pattern_rebind/frames100/report.json),
  [metadata](results/normal_pattern_rebind/frames100/metadata.json),
  [samples](results/normal_pattern_rebind/frames100/samples.csv).
- [300 report](results/normal_pattern_rebind/frames300/report.json),
  [metadata](results/normal_pattern_rebind/frames300/metadata.json),
  [updates](results/normal_pattern_rebind/frames300/updates.csv),
  [quality](results/normal_pattern_rebind/frames300/quality.csv).

## Separate profiling, allocation and cached correctness

A separate warmed final profile records **300 full preparations plus 183
rebindings**, with the same 631 numeric factorizations. Full ordering/fill
preparation calls decrease by 37.9%; public `analyses()` semantics remain unchanged.
Instrumented estimates match primary means exactly and pass original-energy
checks. Stage times overlap and are excluded from primary comparisons.
[Predecessor stages](results/normal_pattern_rebind/diagnostics/stages_before.csv),
[classifier](results/normal_pattern_rebind/diagnostics/normal_classifier.csv),
[final stages](results/normal_pattern_rebind/diagnostics/stages_after.csv).
Their JSON records retain source/patch/executable/input hashes and quality gates.

A separate warmed 300-arrival allocation run counts complete update/report calls.
Requests fall from **2,583,592 to 2,403,726 (7.0%)**, and requested bytes from
**622,117,529 to 568,194,510 (8.7%)**. Means and non-timing reports match primary
runs exactly. Instrumented times are excluded; these are cumulative allocation
traffic, not live/peak/RSS or native memory comparisons.
[Allocation record](results/normal_pattern_rebind/diagnostics/allocations.json).

The default-policy 300-arrival validator includes 258 cached updates, 30 interval
refreshes, 12 forced loop refreshes and twelve AD/dense-QR reference checkpoints.
Complete before/after traces and summaries are byte-identical. Cached estimates
remain approximate between refreshes, with maximum original gradient about
`0.0100` (unchanged). These are correctness checks, not cached performance timing.
[Record](results/normal_pattern_rebind/diagnostics/cached_validation.json),
[trace](results/normal_pattern_rebind/diagnostics/cached_trace.csv),
[summary](results/normal_pattern_rebind/diagnostics/cached_summary.txt).

## Reproduce

Build the optional verified native executable/manifest following [NATIVE_ISAM2.md](NATIVE_ISAM2.md).
Rust still has no C++ dependency.

```sh
cargo build -p ropt --example benchmark_warm_isam_se2 \
  --example validate_isam_se2 --release --locked --offline
python3 benchmarks/make_ldl_baseline.py --before-normal-rebind \
  --output /tmp/ropt-before-normal-rebind
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  python3 benchmarks/paired_normal_rebind.py \
  --before /tmp/ropt-before-normal-rebind/target/release/examples/benchmark_warm_isam_se2 \
  --before-source /tmp/ropt-before-normal-rebind \
  --after target/release/examples/benchmark_warm_isam_se2 \
  --native YOUR_NATIVE_ISAM2_BINARY --native-manifest YOUR_VERIFIED_MANIFEST.json \
  --frames 300 --rounds 12 --output /tmp/normal-rebind-comparison
```

Finish builds first, choose a permitted CPU and fresh output directories. The
helper reverses the later `qr_rebuild.patch`, then `anchor_coordinates.patch`, then `qr_columns.patch`, then `qr_reductions.patch`, then `update_cursor.patch`, then `qr_norm.patch`, then `symbolic_sorted_merge.patch`, then `qr_aligned.patch`, then `normal_pattern_rebind.patch`
for this predecessor;
older stages reverse it first, then their existing patches. Its rebuilt source
matches the frozen predecessor and passes six-round 25-arrival native smoke.
All sixteen stages apply and match archived kernels in source-only checks;
older-stage compilation is skipped while the latest predecessor is built separately.
[Rollback record](results/normal_pattern_rebind/diagnostics/rollback_stages.json).

For cached correctness run both candidates' `validate_isam_se2 300` separately.
Allocation diagnostics append `alloc` to the runner; exclude their times.
Profiling patches are benchmark-only and must be applied to isolated snapshots:
`normal_rebind_stage_profile.patch` and `normal_rebind_classifier.patch` apply
forward to the predecessor, while `normal_rebind_after_profile.patch` applies
to the final kernel. Rebuild `benchmark_warm_isam_se2`, then run a 300-arrival
input with arguments `1 1`. The optional patches reset counters before the
measured stream; retain instrumented binaries separately from primary candidates.
[Forward-patch reproduction record](results/normal_pattern_rebind/diagnostics/profile_patch_reproduction.json)
checks all three instrumented kernels and runners against their built snapshots.
CI checks correctness without timing thresholds. Selective Bayes-tree updates,
retained history and field validation remain gaps.

The subsequent [aligned QR row reuse](QR_ALIGNED.md) is measured separately.
