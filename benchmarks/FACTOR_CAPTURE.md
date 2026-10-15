# Consecutive Gaussian factor capture from canonical CSR

Full refreshes and selective updates split a whitened residual/Jacobian model
into owned Gaussian factor blocks. The preceding implementation pushes each row's
nonzeros into growing column/value vectors. The new implementation copies each
factor's contiguous coefficient slices once and normalizes its CSR row offsets.
The global column numbers, coefficient and residual bits, empty rows and factor
order remain unchanged. Each vector is allocated at its required length.

Dimension checks, the original `residuals + nnz` fill limit, canonical matrix
validation and residual-finiteness checks remain in place. Cache mutations still
occur after successful capture; failed reset/append retain the old blocks.
The public API and library defaults are unchanged.

## Allocation traffic

The separate diagnostic measures complete forced refreshes through the public
`IsamSmoother` API. Each independent variable has one Gaussian factor, with an
observable 1D/3D block and optional constant-only rows. The initial state is the
exact optimum, so batch trials are zero. Four refreshes warm up each fresh
process; one refresh is counted. Two before/after pairs alternate order, CPU 0,
one thread. Physical costs, zero gradients, estimate bits, cached costs and work
match across kernels. Callback/graph/QR/tree/report allocations are included.

| 512 factors, residual rows per factor | Fewer allocation requests | Fewer requested bytes |
| --- | ---: | ---: |
| 1 | 512 | 36,864 |
| 3 | 2,560 | 159,744 |
| 16, including constant-only rows | 4,096 | 335,872 |

The 3-row case decreases **54,623 → 52,063** successful allocation/reallocation
requests and **4,325,904 → 4,166,160** requested bytes. The 32-factor cases have
the same per-factor savings. All six workloads retain identical counts between
their two pairs. Counts describe allocation traffic, **not live or peak memory**;
instrumented diagnostic timers make no comparative speed claim.

## Complete-stream protocol and regression

Complete-stream medians are **2.5% / 4.6% lower**, with **4/6 / 5/6 faster pairs**.
Only confirmation meets the pre-existing campaign criterion; a repeatable
whole-update speedup is **not established**. The change is retained for its
verified allocation-traffic reduction and unchanged numeric/error behavior.
All **25,056 timed arrivals** pass the unchanged original-information and
archived-native gates, with exact before/after pose bytes and every non-timing
work field. Maximum normalized gradient is `4.6772858743494114e-8`.

| Campaign | Before stream median | After stream median | Time reduction | Faster pairs | Meets criterion |
| --- | ---: | ---: | ---: | ---: | --- |
| Primary | 5.111 s | 4.985 s | 2.5% | 4/6 | No |
| Confirmation | 5.128 s | 4.892 s | 4.6% | 5/6 | Yes |

Pooled arrival medians rise slightly, while P95/P99 decrease. Maximum observed
latency improves in primary but worsens in confirmation. These observed pooled
distributions contain 6,264 arrivals per kernel/campaign; nearest-rank percentiles
are not paired confidence estimates or real-time guarantees.

| Campaign / arrival statistic | Before | After |
| --- | ---: | ---: |
| Primary median | 2.118 ms | 2.152 ms |
| Primary P95 | 31.249 ms | 30.261 ms |
| Primary P99 | 53.228 ms | 51.388 ms |
| Primary maximum | 80.027 ms | 70.740 ms |
| Confirmation median | 2.117 ms | 2.137 ms |
| Confirmation P95 | 31.253 ms | 29.910 ms |
| Confirmation P99 | 54.043 ms | 51.104 ms |
| Confirmation maximum | 78.028 ms | 85.227 ms |

The uninstrumented CSAIL comparison retains 1,045 poses and 1,172 constraints,
586 AD / 586 analytic factors, chunk 6 and stride 2. Both kernels use the
previously measured explicit requested cap 0, keeping automatic threshold passes.
The library change is compared at identical caller policy, avoiding a mixture of
the earlier policy improvement and this cache change.

Two six-pair campaigns alternate order on CPU 0, one thread. Each fresh process
discards one full warmup and times one complete stream. Builds, tests, profiling
and separate archival/audit work are paused during both campaigns. Timers include
updates, probes, failed/rejected work, recoveries and owned reports; parsing,
factor recipes, external checking and serialization are excluded. The runner's
own checking and compression take place between timed processes.

Qualification requires a lower median stream total and at least 75% faster pairs
in **each** campaign. All rounds are retained. Original-information gradient
gate `1e-7`, internal target `8e-8`, exact fixed root and reported-cost tolerance
`1e-10 * max(1,cost)` are unchanged. Hashed archived-native pose/cost tolerances
are `2e-6` / `1e-9 * max(1,cost)`; native timings are not measured.

Complete MIT has an additional cold functional replay, including edge 160→161
near `1e12` information under AD. All 1,227 arrivals pass the original gates and
retain exact pose bytes and every non-timing work field versus the frozen
preceding kernel with cap 0. Maximum normalized gradient is
`7.793438497199398e-8`; native pose/cost gaps are
`2.0375466499172747e-7` / `7.574385563202668e-12`. Its timers are noncomparative.

Two new tests compare captured blocks with the previous row-iterator assembly,
cover empty rows/columns, wide global column indices, extreme coefficients and
signed residual zero, and check dimension/fill/nonfinite errors and cache
preservation. Rust **333 library/integration/doc + 2 benchmark tests**, Python
**23**, all-target Clippy, formatting and warning-free Rustdoc pass. All **46**
source-only historical rollback stages restore their recorded fingerprints with
zero fuzz. A launcher typo on the final no-stage baseline was corrected; the log
and correction are retained.

## Reproduction

Use the prepared Rust toolchain/cache. Build the recorded source and a separate
baseline; keep builds, tests and audit work outside timed campaigns.

```sh
python3 benchmarks/make_ldl_baseline.py --before-factor-capture --source-only --output /tmp/ropt-capture-before
cargo build --manifest-path /tmp/ropt-capture-before/Cargo.toml --release --locked --offline --example benchmark_factor_cache --example benchmark_ad_selective_se2_isam
cargo build --release --locked --offline --example benchmark_factor_cache --example benchmark_ad_selective_se2_isam
python3 benchmarks/paired_factor_capture_allocations.py --binary target/release/examples/benchmark_factor_cache --before-binary /tmp/ropt-capture-before/target/release/examples/benchmark_factor_cache --before-source /tmp/ropt-capture-before --output /tmp/ropt-capture-allocations
python3 benchmarks/paired_factor_capture.py --binary target/release/examples/benchmark_ad_selective_se2_isam --before-binary /tmp/ropt-capture-before/target/release/examples/benchmark_ad_selective_se2_isam --before-source /tmp/ropt-capture-before --input benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o --reference benchmarks/results/csail_real_isam2/primary --output /tmp/ropt-capture-primary --rounds 6 --ad-chunk 6 --ad-stride 2 --cpu 0 --selected-cap 0
python3 benchmarks/verify_factor_capture.py --current-source
```

Repeat timing in a fresh confirmation directory. The
[108-file hash manifest and records](results/factor_capture/) preserve raw updates/quality/poses, allocation
outputs, recipes, tests, rollback logs, metadata and an artifact hash manifest.
Source fingerprints are before
`408cdc1c2b4394120a8172592e4a20fd74b61188a5496ee826af733f331ee413`, after
`8c04f9ad7802899dee48b1114765b34f3fb4a475202d5d9a5b6ebf1b2926caad`.
Omit `--current-source` when auditing after further development. Historical native
timings and informal maturity remain unchanged at 75/100.
