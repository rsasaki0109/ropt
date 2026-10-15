# Stack input wrappers for tangent-space AD

One-/two-block AD factors now pass their retracted input blocks through fixed
arrays of borrowed slices instead of allocating an outer vector of block vectors
and a second vector of slice references on every derivative chunk. The retracted
jet vectors themselves remain owned allocations. Three or more blocks, and
internal calls with mismatched value counts, retain the preceding generic path.
Current-estimate probes, graph factors and incremental updates share this adapter.
The preceding direct block-Jacobian writer is unchanged.

**Repeatable complete-update speedup remains unestablished.** CSAIL primary's
median improves 1.1%, but only 2/6 pairs are faster. Confirmation is 0.04% slower,
also with 2/6 faster pairs. Both miss the existing criterion. The supported speed
improvement is confined to the qualifying synthetic probe cases below; allocation
reduction is retained with all complete-stream latency records.

| Stream campaign | Heap wrappers before | Stack wrappers after | Time reduction | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Primary | 5.650 s | 5.590 s | 1.1% | 2/6 |
| Confirmation | 5.626 s | 5.628 s | -0.04% | 2/6 |

## Measured probe latency

Two independent campaigns cover 512/2,048 factors, 50/100% AD and chunk 1/8,
with both gradient norms. Each campaign measures six alternating before/after
pairs, 128 queries per norm, four query warmups, fresh processes, CPU 0 and one
thread. Setup, optimization and independent checking are outside query timers;
probes and their owned reports are timed. Allocation counting is disabled.

Every case retains exact cost/gradient/per-variable-gradient/estimate bits and
callback/chunk counts across kernels and repeats: **384 norm samples** total.
**14/16 norm cases qualify in each campaign; 12/16 qualify in both.** Qualification
requires median ratio > 1 AND at least 75% faster pairs in each campaign. The
remaining four cases are retained, with no repeatable speed claim for them.
These are synthetic query measurements, not whole optimization or GTSAM timings.

ColumnNormalized results; percentages are reductions in median query time:

| Factors / AD / chunk | Primary reduction (faster pairs) | Confirmation reduction (faster pairs) | Qualifies in both |
| --- | ---: | ---: | --- |
| 512 / 50% / 1 | 6.2% (5/6) | 9.6% (6/6) | Yes |
| 512 / 50% / 8 | 6.5% (4/6) | 3.7% (6/6) | No |
| 512 / 100% / 1 | 14.2% (6/6) | 14.5% (6/6) | Yes |
| 512 / 100% / 8 | 11.4% (5/6) | 2.5% (4/6) | No |
| 2,048 / 50% / 1 | 12.3% (6/6) | 12.6% (6/6) | Yes |
| 2,048 / 50% / 8 | 1.6% (3/6) | 4.2% (5/6) | No |
| 2,048 / 100% / 1 | 16.8% (6/6) | 16.3% (6/6) | Yes |
| 2,048 / 100% / 8 | 8.0% (6/6) | 7.1% (6/6) | Yes |

## Allocation diagnostic

An independent diagnostic covers 32/512 factors, 0/50/100% AD, chunk 1/8 and both
norms: two balanced pairs, one successful query per norm after four warmups,
**96 norm samples**. Results and callback work match bitwise, and allocation totals
repeat exactly. Analytic-only totals are unchanged. All measured two-block AD
callbacks remove **2 allocation requests / 128 requested bytes per invocation**.
For the three-tangent-coordinate fixture, chunk 1 invokes each AD callback three
times, so the savings are tripled. Fixed blocks retain their derivative chunks.
Callback slot order alternates in the fixture.

Examples for ColumnNormalized, chunk 8:

| Factors / AD | Requests before → after | Requested bytes before → after |
| --- | ---: | ---: |
| 32 / 0% | 265 → 265 | 10,536 → 10,536 |
| 32 / 50% | 553 → 521 | 26,792 → 24,744 |
| 32 / 100% | 841 → 777 | 43,048 → 38,952 |
| 512 / 0% | 4,184 → 4,184 | 164,904 → 164,904 |
| 512 / 50% | 8,792 → 8,280 | 425,000 → 392,232 |
| 512 / 100% | 13,400 → 12,376 | 685,096 → 619,560 |

A successful 512-factor all-AD, chunk-8 query saves **1,024 requests / 65,536
requested bytes** beyond the preceding direct-block-Jacobian implementation.
These are allocation requests, not live/peak memory measurements. One-block
allocation traffic is not measured here. The diagnostic's times are
noncomparative; setup and independent checks are excluded.

## Contracts and candidate selection

Input storage/tangent conventions, fixed-slot chunks, retraction expressions,
callback slot order, callback counts, finite checks and primal consistency checks
are unchanged. Each retraction still runs in caller slot order before the callback.
Arrays borrow locally owned jet vectors only for the callback invocation. Errors
retain their precedence and stop further callbacks as before. No public API,
dependency, unsafe code, derivative shortcut or persistent cache is introduced.

Two tests compare complete input value/partial traces and residual/Jacobian bits
with the original heap-wrapper expression across one, two and three blocks,
Euclidean, SE(2) Cartesian and SO(3) values and chunks 1/2/6/32. They also check
first-/second-slot validation and later callback errors. Existing manifold and
solver regressions remain enabled. Rust **331 library/integration/doctests + 2
benchmark tests**, Python **23**, all-target Clippy, formatting and Rustdoc with
warnings denied pass. All **45** source-only rollback fingerprints match with
zero-fuzz patches; historical kernels are not rebuilt here.

Before adoption, two isolated candidates were compared against the preceding
measured binary. A checked strided-write/constant-row-skip candidate qualified in
only **2/16** probe cases and was rejected. The stack-wrapper candidate qualified
in **14/16** initial cases and proceeded to final-source measurement after adding
regression tests. Both candidates' sources, build/unit logs and all raw timing
records are retained under [trials](results/inline_ad_inputs/trials/). Rejected
strided writes are absent from production and have no full-stream measurement.

## Complete-stream protocol

Complete CSAIL retains 1,045 poses, 1,172 original edges and 1,044 arrivals, with
586 AD / 586 analytic factors, chunk 6 and stride 2. Original observations,
information, initialization and editable-QR/GN stationarity policy are unchanged.
Each fresh process discards one complete warmup stream and measures one stream;
six pairs alternate order, CPU 0, one thread. Builds, tests and allocation/probe
campaigns finish before comparative stream timing. Parsing/factor recipes,
serialization and independent checking are excluded; updates, nonlinear probes,
requested passes, failed attempts, full recoveries and owned reports are timed.

Original-information gradient gate `1e-7`, internal target `8e-8`, exact root,
reported-cost tolerance `1e-10 * max(1,cost)` and archived-native pose/cost gates
`2e-6` / `1e-9 * max(1,cost)` are unchanged. Native poses/costs serve as a hashed
archived agreement reference; no fresh native timing or GTSAM speed claim is made.

All **25,056 timed arrivals** pass unchanged quality gates and archived-native
agreement. Complete pose bytes and all non-timing update fields match in every
paired round. Independent maximum normalized gradient remains
`5.8075741199751337e-8`. Per kernel/campaign, work remains 780 requested passes,
618 full recoveries, 7,662 probes and 2,425,338 committed re-eliminated scalar
cliques. Failed work is timed but not fully represented by successful-work
counters; they cannot establish complete-stream allocation traffic.

The complete [MIT functional regression](results/inline_ad_inputs/mit_functional/)
uses a fresh cold candidate against the frozen preceding direct-block replay.
All 1,227 arrivals pass original-information and archived-native gates, with exact
poses/non-timing work, 742 AD / 741 analytic factors and unchanged AD on
ill-conditioned edge 160→161 near `1e12`. These timers are noncomparative.
Archived-native maximum pose/cost gaps remain `1.945092214966316e-7` /
`6.423306331271306e-12`.

## Reproduction

Use the prepared toolchain/cache and distinct Cargo targets for source snapshots.
Before executables are the actual preceding measured binaries with archived hash
checks; before/after benchmark driver hashes match. Fresh baseline builds:

```sh
python3 benchmarks/make_ldl_baseline.py --before-inline-ad-inputs --source-only --output /tmp/ropt-before-ad-layout
cargo build --manifest-path /tmp/ropt-before-ad-layout/Cargo.toml --release --locked --offline --example benchmark_autodiff_probe --example benchmark_ad_selective_se2_isam
cargo build --release --locked --offline --example benchmark_autodiff_probe --example benchmark_ad_selective_se2_isam
python3 benchmarks/paired_ad_layout_probe.py --binary target/release/examples/benchmark_autodiff_probe --before-binary /tmp/ropt-before-ad-layout/target/release/examples/benchmark_autodiff_probe --before-source /tmp/ropt-before-ad-layout --output /tmp/ropt-input-probe-primary --rounds 6 --queries 128 --cpu 0
python3 benchmarks/paired_inline_ad_inputs_allocations.py --binary target/release/examples/benchmark_autodiff_probe --before-binary /tmp/ropt-before-ad-layout/target/release/examples/benchmark_autodiff_probe --before-source /tmp/ropt-before-ad-layout --output /tmp/ropt-input-allocations --rounds 2 --queries 1 --allocations --cpu 0
python3 benchmarks/paired_inline_ad_inputs.py --binary target/release/examples/benchmark_ad_selective_se2_isam --before-binary /tmp/ropt-before-ad-layout/target/release/examples/benchmark_ad_selective_se2_isam --before-source /tmp/ropt-before-ad-layout --input benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o --reference benchmarks/results/csail_real_isam2/primary --output /tmp/ropt-input-stream-primary --rounds 6 --ad-chunk 6 --ad-stride 2 --cpu 0
python3 benchmarks/verify_inline_ad_inputs.py --current-source
```

Repeat probe and stream campaigns in fresh directories for confirmation. The
diagnostic trial command additionally accepts `--source` for an isolated snapshot.
The retained functional recipe checks exact pose/work bytes and archived-native
agreement. A cold MIT replay with the preceding functional archive is available:

```sh
ROPT_BENCH_AD_CHUNK=6 ROPT_BENCH_AD_STRIDE=2 taskset -c 0 target/release/examples/benchmark_ad_selective_se2_isam benchmarks/results/inline_ad_inputs/mit_functional/input.stream /tmp/ropt-inline-mit 1 0 editable 1
python3 benchmarks/check_real_replay.py --input benchmarks/results/real_isam2/intel-mit1228-original.g2o --candidate /tmp/ropt-inline-mit --reference benchmarks/results/block_jacobian/mit_functional/after --output /tmp/ropt-inline-mit-checks
```

Omit `--current-source` to audit records after further development. Before:
`a4794dd43ee88f82c6155932cf56303cc41ff9048b760edbbd391dec67320080`.
After: `408cdc1c2b4394120a8172592e4a20fd74b61188a5496ee826af733f331ee413`.
Drivers are hashed separately. Informal maturity remains 75/100.
