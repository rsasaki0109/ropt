# Complete MIT: explicit requested-pass cap 1 versus 0

This study extends the [CSAIL recovery-cap comparison](RECOVERY_CAPS.md) to the
complete MIT graph. Both policies use the same uninstrumented production binary
and kernel. Automatic threshold-selected passes within `update` remain enabled.
Only the explicit requested-pass cap changes from 1 to 0: when the nonlinear
probe misses the target, cap 0 proceeds directly to the existing full recovery.
Library defaults are unchanged.

Complete-stream time decreases **12.4% / 10.7%** in the two campaigns, with
**6/6 faster pairs each**. All **29,448 timed arrivals** pass the unchanged common
original-information and archived-native accuracy gates. Both campaigns qualify
under the pre-existing speed criterion. This result applies to this caller policy
and workload; native speed figures remain historical.

| Campaign | Cap 1 stream median | Cap 0 stream median | Time reduction | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Primary | 7.248 s | 6.351 s | 12.4% | 6/6 |
| Confirmation | 7.174 s | 6.408 s | 10.7% | 6/6 |

The arrival-latency tradeoff persists: pooled median latency rises approximately
**1%**, while P95/P99 decrease in both campaigns. The maximum observed latency
decreases in the primary campaign but increases in confirmation. The lower stream
total therefore does not establish a universal per-arrival improvement.

| Campaign / pooled arrival statistic | Cap 1 | Cap 0 |
| --- | ---: | ---: |
| Primary median | 2.778 ms | 2.811 ms |
| Primary P95 | 32.666 ms | 26.825 ms |
| Primary P99 | 68.951 ms | 57.623 ms |
| Primary maximum | 120.537 ms | 109.642 ms |
| Confirmation median | 2.793 ms | 2.820 ms |
| Confirmation P95 | 32.346 ms | 26.693 ms |
| Confirmation P99 | 67.323 ms | 56.710 ms |
| Confirmation maximum | 123.898 ms | 131.090 ms |

Work is deterministic across all recorded streams within each policy:

| Work per complete stream | Cap 1 | Cap 0 |
| --- | ---: | ---: |
| Explicit requested passes | 111 | 0 |
| Full recoveries | 111 | 111 |
| Nonlinear probes | 1,449 | 1,338 |
| Committed scalar-clique re-eliminations | 442,122 | 242,634 |

The maximum independent normalized gradient is `7.793438497199398e-8` for both
policies. Across all policy/native comparisons the maximum world-pose gap is
`2.0375466499172747e-7`, and original-cost gap `7.574385563202668e-12`. Fewer
committed cliques describe solver work; they do not measure peak memory or total
memory traffic. The 51-file hash manifest and full records are retained in
[results/mit_recovery_caps](results/mit_recovery_caps).

## Protocol and retained evidence

The original graph contains 1,228 poses, 1,483 constraints and 1,227 arrivals.
Every constraint and its information matrix is retained, including the near
`1e12` information on edge 160→161, selected for AD. Chunk size 6 and stride 2
give 742 AD and 741 analytic factors. Initial values come from original vertices.

Two independent campaigns each contain six pairs in alternating order, CPU 0,
one thread. Each fresh process discards one complete warmup and times one complete
stream. Builds, tests, profiling and separate archival/audit work are paused
during both campaigns. The runner's own validation and serialization occur
between timed processes. Parsing, factor recipes, serialization and independent
checking are excluded from timers; updates, nonlinear probes, requested passes,
recovery, rejected work and owned reports are included.

All 29,448 timed arrivals are checked using original-information normalized
gradient `1e-7`, internal target `8e-8`, reported-cost tolerance
`1e-10 * max(1,cost)` and an exact fixed root. Policy/native pose and original-cost
tolerances are `2e-6` and `1e-9 * max(1,cost)`. Native poses/costs are a hashed
archived precision-corrected reference; native timing is not measured here.
Changing policies may change work and poses within these common accuracy gates.

Qualification requires median complete-stream time to improve and at least 75%
faster pairs in each campaign. Every pair is retained. Arrival distributions pool
7,362 timed arrivals per variant per campaign. P95/P99 use nearest rank; pooled
percentiles are observations rather than paired confidence estimates or real-time
guarantees. Complete-stream totals and arrival latency answer different questions.

The archives preserve raw samples, every timed update and independent quality
result, first-pair complete poses for both policies, options, dataset/stream and
source/driver/executable hashes. The auditor derives timing/work/latency summaries
from raw records and recomputes original-information quality and native/policy
agreement for both retained trajectories in each campaign.

## Reproduction

Use the recorded kernel/toolchain and the same executable for both variants.
Repeat in fresh output directories with other build, test and audit work paused.
The frozen runner fixes caps to 1/0; leave its inherited `--selected-cap` argument
at the default.

```sh
python3 benchmarks/paired_recovery_caps.py --binary target/release/examples/benchmark_ad_selective_se2_isam --before-binary target/release/examples/benchmark_ad_selective_se2_isam --before-source /workspace/ropt --input benchmarks/results/real_isam2/intel-mit1228-original.g2o --reference benchmarks/results/precise_real_isam2/primary --output /tmp/ropt-mit-caps-primary --rounds 6 --ad-chunk 6 --ad-stride 2 --cpu 0
python3 benchmarks/verify_mit_recovery_caps.py --current-source
```

Source fingerprint:
`408cdc1c2b4394120a8172592e4a20fd74b61188a5496ee826af733f331ee413`.
Omit `--current-source` after further kernel/driver development. Historical native
timings and informal maturity remain unchanged at 75/100.
