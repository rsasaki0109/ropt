# Explicit requested-pass cap versus immediate full recovery

The complete-stream phase diagnosis found that explicit selected passes often
precede full recovery. This study compares requested cap **1 → 0** using the
**same uninstrumented executable and source**. Automatic threshold-selected passes
inside `update` remain enabled. After the current nonlinear probe misses the
unchanged `8e-8` target, cap 0 goes directly to the existing full nonlinear update.
The configured batch solver, refinement, observations, information and seeds
are unchanged. This is a caller-policy comparison, not a solver-kernel change.

**The additional isolated CSAIL campaign takes 7.2% less total update time**, with
5/6 faster pairs, passing the unchanged criterion. The earlier campaigns also
qualify, with their concurrent archival work disclosed below. All 37,584 timed
arrivals across the three campaigns pass independent original-information and
archived-native gates, including agreement between policies at the common gate.

| Campaign | Cap 1 median | Cap 0 median | Time reduction | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Primary, archival work overlapped | 5.263 s | 5.042 s | 4.2% | 5/6 |
| Confirmation, archival work overlapped | 5.518 s | 5.000 s | 9.4% | 6/6 |
| Additional isolated campaign | 5.514 s | 5.117 s | 7.2% | 5/6 |

The timing tradeoff is visible in pooled arrival distributions from the isolated
campaign (6,264 arrivals per variant). Median arrival time is **2.061 → 2.193 ms**,
6.4% higher, while P95 is **36.582 → 30.430 ms** and P99 **58.311 → 53.513 ms**.
Maximum observed arrival time is **84.554 → 76.829 ms**. Thus lower complete-stream
time does not imply that every arrival becomes faster. These are observed pooled
distributions, not independent paired confidence estimates or real-time guarantees.

## Functional policy experiment

Fresh cold cap-0 replays pass all original-information and archived-native gates
on complete CSAIL and MIT. This experiment's timers are noncomparative. Unlike
the instrumented diagnostic, different policies need not produce bit-identical
poses or work; their accuracy is checked against common tolerances.

| Work per complete stream | CSAIL cap 1 | CSAIL cap 0 | MIT cap 1 | MIT cap 0 |
| --- | ---: | ---: | ---: | ---: |
| Arrivals | 1,044 | 1,044 | 1,227 | 1,227 |
| Explicit requested passes | 130 | 0 | 111 | 0 |
| Full recoveries | 103 | 129 | 111 | 111 |
| Nonlinear probes | 1,277 | 1,173 | 1,449 | 1,338 |

CSAIL cap-0 independent maximum normalized gradient is
`4.6772858743494114e-8`, with archived-native pose/cost gaps
`3.772476536257072e-7` / `4.263256414560601e-13`. MIT values are
`7.793438497199398e-8`, `2.0375466499172747e-7` / `7.574385563202668e-12`.
The fixed root remains exact. Complete original information is retained,
including the MIT edge 160→161 near `1e12` under AD. Subsequent
[complete MIT timing campaigns](MIT_RECOVERY_CAPS.md) confirm 12.4% / 10.7% lower
stream totals (6/6 faster pairs each), with a small median-arrival increase and
mixed maximum latency. Those separate timings retain the same binary and gates.

CSAIL committed scalar-clique re-eliminations decrease from **404,223 to 298,053**
per stream, while full recoveries increase. Fewer probes or committed cliques
alone do not establish a runtime improvement or whole-stream memory traffic;
failed work and model-building work remain included in the timers.

## Comparative timing protocol

Complete CSAIL uses 1,045 poses, 1,172 edges, 586 AD / 586 analytic factors, chunk
6 and stride 2. Each fresh process discards one complete warmup stream and times
one stream. Six before/after pairs alternate order, CPU 0, one thread. Parsing,
factor recipes, independent checking and serialization are excluded. Updates,
probes, requested passes, rejected work, recoveries and owned reports are included.
Allocation and phase instrumentation are disabled in the timed executable.

The quality gates are original-information normalized gradient `1e-7`, internal
target `8e-8`, reported-cost tolerance `1e-10 * max(1,cost)`, exact root, and common
policy/native pose and cost tolerances `2e-6` / `1e-9 * max(1,cost)`. Neither
observations nor information are pruned or relaxed. Qualification remains median
ratio > 1 AND at least 75% faster pairs in a campaign. Every round is retained.

Record verification and compression overlapped portions of the first two
campaigns. An additional campaign runs with archival/verification work paused
to check the result with that source of CPU load removed. Earlier records remain
available; they are not replaced by the additional run.

The study does not change library defaults or the original historical benchmark
recipes. Its result applies to this explicit stationarity policy and dataset.
No fresh native timings or GTSAM speed claim are made.

## Reproduction

Use the recorded production kernel and prepared toolchain/cache. Supply the same
executable for both variants and its source checkout. Caps are fixed to 1/0 by
this study's runner; the inherited `--selected-cap` argument stays at its default.
Repeat in fresh directories, keeping builds, tests, profiling and archival work
outside comparative timing.

```sh
python3 benchmarks/paired_recovery_caps.py --binary target/release/examples/benchmark_ad_selective_se2_isam --before-binary target/release/examples/benchmark_ad_selective_se2_isam --before-source /workspace/ropt --input benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o --reference benchmarks/results/csail_real_isam2/primary --output /tmp/ropt-caps-primary --rounds 6 --ad-chunk 6 --ad-stride 2 --cpu 0
python3 benchmarks/verify_recovery_caps.py --current-source
```

Cold functional reproduction uses the retained complete input streams and
requested cap 0:

```sh
ROPT_BENCH_AD_CHUNK=6 ROPT_BENCH_AD_STRIDE=2 taskset -c 0 target/release/examples/benchmark_ad_selective_se2_isam benchmarks/results/recovery_caps/functional/mit/input.stream /tmp/ropt-cap0-mit 1 0 editable 0
python3 benchmarks/check_real_replay.py --input benchmarks/results/real_isam2/intel-mit1228-original.g2o --candidate /tmp/ropt-cap0-mit --output /tmp/ropt-cap0-mit-checks
```

Source: `408cdc1c2b4394120a8172592e4a20fd74b61188a5496ee826af733f331ee413`.
Omit `--current-source` when auditing after further development. Informal maturity
remains 75/100.
