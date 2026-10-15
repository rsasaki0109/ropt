# Complete-stream timing of factor-cache append reservation

**Candidate adoption is withdrawn.** Both observed stream medians increase by
about 2%, with only 2/6 and 3/6 faster pairs. Neither meets the pre-existing speed
criterion. Although allocation traffic decreases, that evidence does not justify
a speed claim. Production is restored to the exact preceding source fingerprint
`10a41ce417d42667132cd2ceaab6c916a88855b8e83aaaf0920ce421141667e3`.
These observations do not establish a universal causal slowdown; restoring the
preceding implementation is the conservative choice for the runtime objective.

| Campaign | Before median | Candidate median | Time increase | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Primary | 4.936 s | 5.033 s | 2.0% | 2/6 |
| Confirmation | 4.912 s | 5.013 s | 2.0% | 3/6 |

All **25,056 timed arrivals** pass unchanged original-information and archived-
native gates with exact before/after pose bytes and non-timing work fields.
Maximum normalized gradient is `4.6772858743494114e-8`. Each stream retains 129
recoveries, 1,173 nonlinear probes and 298,053 committed scalar-clique
re-eliminations.

| Campaign / pooled arrival statistic | Before | Candidate |
| --- | ---: | ---: |
| Primary median | 2.101 ms | 2.120 ms |
| Primary P95 | 29.849 ms | 30.090 ms |
| Primary P99 | 51.785 ms | 51.890 ms |
| Primary maximum | 70.655 ms | 76.986 ms |
| Confirmation median | 2.083 ms | 2.127 ms |
| Confirmation P95 | 29.542 ms | 30.125 ms |
| Confirmation P99 | 51.095 ms | 51.001 ms |
| Confirmation maximum | 73.862 ms | 74.986 ms |

This follows the [allocation/correctness study](CACHE_RESERVE.md). Before source
uses exact-length reference-vector clones; after source reserves incoming-factor
capacity first. Consecutive CSR capture and scheduled-refresh clone elision are
already enabled in both binaries. No Rust code is changed during this timing
study, and caller policy remains explicit requested cap 0, retaining automatic
threshold-selected passes.

The complete CSAIL stream contains 1,045 poses and 1,172 original constraints,
586 AD / 586 analytic factors, chunk 6 and stride 2. Two six-pair campaigns
alternate before/after order, CPU 0, one thread. Every fresh process discards one
complete warmup before timing one complete stream. Builds, tests, profiling and
separate archival/audit work are paused during the campaigns; the runner checks
and compresses between timed processes. Timers include updates, probes, recovery,
rejected/failed work and owned reports. Parsing, factor recipes, external checking
and serialization are excluded; allocation/phase instrumentation is disabled.

Original-information normalized gradient `1e-7`, internal target `8e-8`, exact
fixed root and reported-cost tolerance `1e-10 * max(1,cost)` remain unchanged.
Hashed archived-native pose/cost agreement uses `2e-6` and `1e-9 * max(1,cost)`.
No fresh native timing is measured. Every pair also requires exact pose bytes
and all non-timing work fields. All pairs are retained.

The pre-existing qualification requires a lower median stream time AND at least
75% faster pairs in each campaign. Pooled arrival statistics use 6,264 timed
observations per variant/campaign and nearest-rank P95/P99; they are not paired
confidence estimates or real-time guarantees. The auditor derives summaries from
raw records and recomputes costs/gradients and native agreement from both retained
after trajectories; their before counterparts are byte-identical.

## Reproduction

Use the recorded sources and prepared toolchain/cache. The baseline was rebuilt
in an isolated source snapshot using the same compiler/options and identical
driver. Keep builds, tests and audits outside timed campaigns. Because production
has been restored, the timing commands below run from a separate checkout made
from `results/cache_reserve_timing/candidate_source`, with the repository's
benchmark scripts, patches and retained result directories copied into their
usual relative paths. In that checkout install the saved
`candidate_baseline_builder.py` as `benchmarks/make_ldl_baseline.py`; it preserves
the candidate's `--before-cache-reserve` reconstruction recipe. That flag is
removed from the live production builder.

```sh
python3 benchmarks/make_ldl_baseline.py --before-cache-reserve --source-only --output /tmp/ropt-reserve-before
cargo build --manifest-path /tmp/ropt-reserve-before/Cargo.toml --release --locked --offline --example benchmark_ad_selective_se2_isam
cargo build --release --locked --offline --example benchmark_ad_selective_se2_isam
python3 benchmarks/paired_cache_reserve.py --binary target/release/examples/benchmark_ad_selective_se2_isam --before-binary /tmp/ropt-reserve-before/target/release/examples/benchmark_ad_selective_se2_isam --before-source /tmp/ropt-reserve-before --input benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o --reference benchmarks/results/csail_real_isam2/primary --output /tmp/ropt-reserve-primary --rounds 6 --cpu 0 --ad-chunk 6 --ad-stride 2 --selected-cap 0
python3 benchmarks/verify_cache_reserve_timing.py --current-source
```

Audit the retained data and the restored production checkout here with:

```sh
python3 benchmarks/verify_cache_reserve_timing.py --restored-production
```

Repeat in a fresh confirmation directory. [Records](results/cache_reserve_timing/)
retain both campaigns, poses, raw updates/quality, source/driver/executable hashes
and an artifact hash manifest. Before source:
`10a41ce417d42667132cd2ceaab6c916a88855b8e83aaaf0920ce421141667e3`;
after `939279f7aa51cd59af04323230a1433620751c75df1625fbf3c62d26243bc1de`.
Omit `--current-source` after further development. Native figures and informal
maturity remain unchanged at 75/100.
