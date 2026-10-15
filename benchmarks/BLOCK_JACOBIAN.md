# Direct block Jacobians for tangent-space automatic differentiation

Factor AD now writes each validated derivative chunk directly into the final
parameter-block Jacobians. It no longer allocates the flat residual-by-tangent
Jacobian, copies its column blocks, or allocates a zero tangent input vector.
The shared chunk engine seeds the same jets at positive zero and invokes the
same manifold retractions and residual callbacks in the same order. It serves
graph factors, incremental updates and current-estimate probes; the public
`DynamicAutoDiff` adapter still fills its caller-supplied flat matrix.

The change removes **two allocation requests per successful AD factor evaluation**
in every measured configuration. For the synthetic workload (three residuals,
three total tangent coordinates, including a fixed one-coordinate block), each
AD factor saves **104 requested bytes**. A successful 512-factor all-AD query
saves **1,024 requests / 53,248 bytes** beyond the preceding dense-index kernel.
Analytic-only cases retain exactly the same allocation totals.

**Repeatable complete-update speedup is not established.** CSAIL primary is
2.3% slower (3/6 faster pairs), while confirmation is 1.8% faster (4/6). Neither
passes the unchanged criterion: median ratio > 1 AND at least 75% faster pairs
in each campaign. The implementation is retained for its allocation/storage
reduction, with both positive and negative latency records preserved.

| Campaign | Flat Jacobian before | Direct blocks after | Time reduction | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Primary | 5.625 s | 5.754 s | -2.3% | 3/6 |
| Confirmation | 5.748 s | 5.645 s | 1.8% | 4/6 |

These are allocation requests, not live/peak memory measurements. The eliminated
flat Jacobian and seed storage contain `8 * parameters * (residuals + 1)` bytes;
the measured total also reflects allocation capacity differences between copied
matrix views and directly allocated blocks. Derivative jets, callback outputs,
manifold intermediates and final block matrices still allocate.

## Separate allocation diagnostic

Two balanced before/after pairs cover 32/512 factors, 0/50/100% AD, chunk 1/8 and
Absolute/ColumnNormalized norms: **96 norm samples**. Each query follows four
warmups, with setup and independent checks excluded. Fixed blocks retain their
derivative chunks and callback slot order alternates. Cost, gradient, per-variable
gradient and estimate bits, and callback/chunk counts match exactly across
kernels and repeats. Allocation totals repeat exactly. Diagnostic timers are
noncomparative; allocation counting is absent from comparative stream timing.

Examples for ColumnNormalized, chunk 8:

| Factors / AD | Requests before → after | Requested bytes before → after |
| --- | ---: | ---: |
| 32 / 0% | 265 → 265 | 10,536 → 10,536 |
| 32 / 50% | 585 → 553 | 28,456 → 26,792 |
| 32 / 100% | 905 → 841 | 46,376 → 43,048 |
| 512 / 0% | 4,184 → 4,184 | 164,904 → 164,904 |
| 512 / 50% | 9,304 → 8,792 | 451,624 → 425,000 |
| 512 / 100% | 14,424 → 13,400 | 738,344 → 685,096 |

## Numerical and error contract

Chunk widths, fixed-block derivative dimensions, positive-zero seeds, constant
broadcasting and row/column write order are unchanged. Each output row is checked
for a carried error and derivative width before its primal value is accepted.
Primal values must match bitwise across chunks, including signed zero. The first
callback or output error stops evaluation as before. The public flat adapter
retains input/output dimension checks, finite-input validation and row-wise
partial-write behavior on failure. New internal block evaluation validates the
sum of block widths, including overflow, before invoking callbacks.

Two regression tests compare block and flat result bits across chunk sizes
1/2/4/6/7/32, zero-width blocks, chunks crossing block boundaries, constant and
signed-zero outputs, nonfinite outputs, invalid derivative widths, nondeterministic
primal values and later callback failures. Existing manifold tests and full-stream
before/after comparison cover nonlinear retractions and mixed AD/analytic factors.
No public API, dependency, unsafe code, numerical-differentiation fallback or
persistent cache is introduced.

Rust **329 library/integration/doctests + 2 benchmark tests**, Python **23**, all-
target Clippy, formatting and Rustdoc with warnings denied pass. All **44** source-
only rollback fingerprints match with zero-fuzz patches. Historical kernels are
not rebuilt in this turn.

## Stream protocol

The comparison uses complete CSAIL: 1,045 poses, 1,172 unchanged original edges,
1,044 arrivals, 586 AD / 586 analytic factors, AD chunk 6 and stride 2. Before/
after drivers, observations, original information, initialization and editable
QR/GN stationarity policy match. Each process discards one complete warmup stream
and measures one complete stream. Six pairs alternate order, CPU 0, one thread.
Builds, tests and allocation diagnostics finish before comparative timing.
Parsing/factor recipes, serialization and independent checks are excluded;
updates, probes, requested passes, failed attempts, full recovery and owned
reports are included.

The speed criterion remains median ratio > 1 AND at least 75% faster pairs in
each campaign. All rounds are retained. Original-information gradient gate
`1e-7`, internal target `8e-8`, exact fixed root, reported-cost tolerance
`1e-10 * max(1,cost)` and archived-native pose/cost gates `2e-6` /
`1e-9 * max(1,cost)` are unchanged. Native poses/costs are an archived agreement
reference; no fresh native timing or GTSAM speed claim is made.

All **25,056 timed arrivals** pass these gates and archived-native agreement.
Complete before/after pose bytes and non-timing update fields match in every
round. Independent maximum normalized gradient remains `5.8075741199751337e-8`.
Per kernel/campaign, work remains 780 requested passes, 618 full recoveries,
7,662 probes and 2,425,338 committed re-eliminated scalar cliques. Failed work
is timed but not fully represented in successful-work counters; these counters
cannot establish whole-stream allocation traffic.

The complete [MIT functional regression](results/block_jacobian/mit_functional/)
uses a fresh cold candidate against the frozen preceding dense-index replay.
All 1,227 arrivals pass original-information and archived-native quality gates,
with exact complete poses and non-timing work, 742 AD / 741 analytic factors
and unchanged AD on ill-conditioned edge 160→161 near `1e12`. Its timers are
noncomparative. Maximum archived-native pose/cost gaps remain
`1.945092214966316e-7` / `6.423306331271306e-12`.

## Reproduction

Use the prepared toolchain/cache and independent Cargo targets per source snapshot.
This run uses actual preceding measured executables with archived hash checks.
Before/after benchmark driver hashes match. A fresh baseline rebuild is supported:

```sh
python3 benchmarks/make_ldl_baseline.py --before-block-jacobian --source-only --output /tmp/ropt-before-block-jacobian
cargo build --manifest-path /tmp/ropt-before-block-jacobian/Cargo.toml --release --locked --offline -p ropt --example benchmark_autodiff_probe --example benchmark_ad_selective_se2_isam
cargo build --release --locked --offline -p ropt --example benchmark_autodiff_probe --example benchmark_ad_selective_se2_isam
python3 benchmarks/paired_block_jacobian_allocations.py --binary target/release/examples/benchmark_autodiff_probe --before-binary /tmp/ropt-before-block-jacobian/target/release/examples/benchmark_autodiff_probe --before-source /tmp/ropt-before-block-jacobian --output /tmp/ropt-block-allocations --rounds 2 --queries 1 --allocations --cpu 0
python3 benchmarks/paired_block_jacobian.py --binary target/release/examples/benchmark_ad_selective_se2_isam --before-binary /tmp/ropt-before-block-jacobian/target/release/examples/benchmark_ad_selective_se2_isam --before-source /tmp/ropt-before-block-jacobian --input benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o --reference benchmarks/results/csail_real_isam2/primary --output /tmp/ropt-block-primary --rounds 6 --ad-chunk 6 --ad-stride 2 --cpu 0
python3 benchmarks/verify_block_jacobian.py --current-source
```

Repeat the stream command in a fresh directory for confirmation. Omit
`--current-source` to audit archived records after further development. Replay
the MIT functional case with the retained complete input and preceding archive:

```sh
ROPT_BENCH_AD_CHUNK=6 ROPT_BENCH_AD_STRIDE=2 taskset -c 0 target/release/examples/benchmark_ad_selective_se2_isam benchmarks/results/block_jacobian/mit_functional/input.stream /tmp/ropt-block-mit 1 0 editable 1
python3 benchmarks/check_real_replay.py --input benchmarks/results/real_isam2/intel-mit1228-original.g2o --candidate /tmp/ropt-block-mit --reference benchmarks/results/dense_probe/mit_functional/after --output /tmp/ropt-block-mit-checks
```

The archived functional recipe also checks exact complete pose bytes/non-timing
work and archived-native agreement before retaining the replay. Before:
`af6a93f7252853127e972b3128d208ce0cb5798a59c46603565a346b6f96f1ad`.
After: `a4794dd43ee88f82c6155932cf56303cc41ff9048b760edbbd391dec67320080`.
Drivers are hashed separately. Informal maturity remains 75/100.
