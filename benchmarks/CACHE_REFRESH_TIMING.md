# Complete-stream timing of scheduled factor-cache clone elision

**No complete-stream speed improvement is established.** Primary takes 0.9% less
median time; confirmation takes 0.5% more. Both have only 2/6 faster pairs and fail
the unchanged campaign criterion. The preceding allocation-traffic reduction
remains verified; it does not establish a runtime reduction.

| Campaign | Before median | After median | Time change | Faster pairs | Meets criterion |
| --- | ---: | ---: | ---: | ---: | --- |
| Primary | 5.045 s | 5.000 s | 0.9% less | 2/6 | No |
| Confirmation | 4.908 s | 4.933 s | 0.5% more | 2/6 | No |

All **25,056 timed arrivals** pass original-information and archived-native gates,
with exact before/after pose bytes and non-timing work fields. Maximum normalized
gradient is `4.6772858743494114e-8`. Each stream retains 129 forced recoveries,
1,173 nonlinear probes and 298,053 committed scalar-clique re-eliminations.

Arrival-latency changes also disagree across campaigns. In particular, primary's
maximum observed latency increases despite its slightly lower stream median.

| Campaign / arrival statistic | Before | After |
| --- | ---: | ---: |
| Primary median | 2.168 ms | 2.106 ms |
| Primary P95 | 30.729 ms | 30.077 ms |
| Primary P99 | 52.392 ms | 50.924 ms |
| Primary maximum | 76.144 ms | 100.977 ms |
| Confirmation median | 2.091 ms | 2.102 ms |
| Confirmation P95 | 29.252 ms | 30.215 ms |
| Confirmation P99 | 50.440 ms | 51.311 ms |
| Confirmation maximum | 83.770 ms | 69.817 ms |

This extends the [allocation/correctness study](CACHE_REFRESH.md) with complete
CSAIL timings. Both binaries already use consecutive CSR factor capture. The
only kernel change skips cloning old factor-row references on forced/interval
refresh; automatic threshold-selected updates still retain their cached rows.
The timed caller disables interval refresh, so this study measures the forced
recovery path. Interval behavior is covered by the preceding correctness tests.

The graph retains 1,045 poses, 1,172 original constraints and their information,
with 586 AD / 586 analytic factors, chunk 6 and stride 2. Both binaries use the
same explicit requested cap 0 and nonlinear targets. The previously measured
policy change and CSR capture change are therefore held constant.

Two campaigns each run six pairs in alternating order, pinned to CPU 0 with one
thread. Every fresh process discards one complete warmup and times one complete
stream. Builds, tests, profiling and separate archival/audit work are paused
throughout both campaigns. The runner's own checking/compression occur between
timed processes. Updates, probes, recovery, rejected/failed work and owned reports
are timed; parsing, factor recipes, independent checking and serialization are
excluded. Allocation/phase instrumentation is disabled.

Quality gates remain original-information normalized gradient `1e-7`, internal
target `8e-8`, reported-cost tolerance `1e-10 * max(1,cost)` and exact fixed root.
Native agreement uses archived, hashed precision-corrected GTSAM poses/costs with
world-pose norm `2e-6` and original-cost tolerance `1e-9 * max(1,cost)`; no fresh
native timings are measured. Each pair additionally requires exact pose bytes
and every non-timing update field. All pairs are retained.

The pre-existing speed criterion requires median stream time to improve and at
least 75% faster pairs **in each campaign**. Pooled arrival distributions contain
6,264 observations per variant/campaign; P95/P99 use nearest rank. These are
observed distributions, not paired confidence estimates or real-time guarantees.

The auditor checks all raw timing/work/quality records and derives summaries from
them. It also recomputes original-information costs/gradients and native agreement
from each campaign's retained after trajectory; before poses are byte-identical.
The complete reference inputs, first-pair poses, raw updates, quality results,
options and source/driver/executable fingerprints are retained.

## Reproduction

Use the prepared toolchain/cache and recorded source. Rebuild an isolated baseline
if the original frozen executable is unavailable, and keep other work paused
during timed campaigns.

```sh
python3 benchmarks/make_ldl_baseline.py --before-cache-refresh --source-only --output /tmp/ropt-refresh-before
cargo build --manifest-path /tmp/ropt-refresh-before/Cargo.toml --release --locked --offline --example benchmark_ad_selective_se2_isam
cargo build --release --locked --offline --example benchmark_ad_selective_se2_isam
python3 benchmarks/paired_cache_refresh.py --binary target/release/examples/benchmark_ad_selective_se2_isam --before-binary /tmp/ropt-refresh-before/target/release/examples/benchmark_ad_selective_se2_isam --before-source /tmp/ropt-refresh-before --input benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o --reference benchmarks/results/csail_real_isam2/primary --output /tmp/ropt-refresh-primary --rounds 6 --cpu 0 --ad-chunk 6 --ad-stride 2 --selected-cap 0
python3 benchmarks/verify_cache_refresh_timing.py --current-source
```

Repeat in a fresh confirmation directory. [Records](results/cache_refresh_timing/)
retain both campaigns and a 51-file hash manifest. Before source is
`8c04f9ad7802899dee48b1114765b34f3fb4a475202d5d9a5b6ebf1b2926caad`, after
`10a41ce417d42667132cd2ceaab6c916a88855b8e83aaaf0920ce421141667e3`.
Omit `--current-source` after further development. This timing extension changes
no Rust code or defaults. Historical native figures and informal maturity stay
unchanged at 75/100.
