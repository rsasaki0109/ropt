# Streaming compensated cost in nonlinear iSAM probes

Current-estimate probes now accumulate the whitened Gaussian cost in factor/row
order without building the concatenated residual vector or an equally sized
zero vector. The same squared-loss reduction and compensated summation are used;
compensation persists across factor boundaries. The cost workspace no longer
scales with the total residual count. Per-factor residuals, gradient/scale vectors,
callback outputs and result maps still allocate normally.

For a **nonempty probe reaching cost reduction**, two allocation requests and
**`16 * residual_count` requested bytes** are removed. Separate measured
successful-probe diagnostics confirm this in every tested analytic/mixed/AD
configuration and both gradient norms: 96 validated samples, two balanced pairs,
exactly repeated allocation totals, identical cost/gradient/estimate bits and
callback counts. With 512 three-residual factors, each query removes **24,576
requested bytes**. These are allocation requests, not measured live/peak memory.
An early callback failure previously allocated only the residual collector, so
it does not imply two saved requests for every failed call.

**Complete CSAIL update time does not improve in these campaigns.** Primary is
1.4% slower (2/6 faster pairs); confirmation is 0.6% slower (4/6). Both miss the
existing criterion: median ratio > 1 AND at least 75% faster pairs in each
campaign. The concrete improvement is allocation traffic and cost workspace,
with the measured latency tradeoff recorded explicitly. No general or native/
GTSAM speed claim is made.

| Campaign | Materialized cost before | Streaming cost after | Time reduction | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Primary | 5.835 s | 5.915 s | -1.4% | 2/6 |
| Confirmation | 5.798 s | 5.835 s | -0.6% | 4/6 |

Negative reduction means more time. Both kernels retain exact complete pose
bytes and all non-timing update fields in every paired round. All **25,056 timed
arrivals** pass unchanged original-information accuracy gates and archived-native
pose/cost agreement. This is the complete CSAIL dataset: 1,045 poses, 1,172 original
edges, 1,044 arrivals, 586 AD / 586 analytic factors, chunk 6, original
observations/information/seeds and the preceding editable-QR/GN policy.
Each kernel/campaign retains 780 requested calls, 618 full recoveries, 7,662
probes and 2,425,338 committed re-eliminated scalar cliques. Failed work is timed
but not fully represented in successful-work counters. These counters are not
used to assert a complete-stream byte-traffic total.

Each process discards one complete warmup stream; before/after order alternates,
CPU 0, one thread, six pairs per campaign. Builds/tests and allocation diagnostics
finish before latency runs. Parsing/factor recipes and serialization are outside
timing; updates, probes, requested passes, failed attempts, full recoveries and
owned reports are included. Independent maximum normalized gradient is unchanged
at `5.8075741199751337e-8`; gate `1e-7`, internal target `8e-8`, exact root and
reported-cost tolerance `1e-10 * max(1,cost)` are unchanged. Archived-native pose/
cost gates remain `2e-6` / `1e-9 * max(1,cost)`; native timing was not measured.

## Allocation diagnostic

The unchanged `benchmark_autodiff_probe` driver measures one successful query
per norm after four warmups, with setup and independent checking outside
measurement. Two balanced pairs cover 32/512 factors, 0/50/100% AD and chunk
1/8, for both Absolute and ColumnNormalized norms. Fixed blocks retain their AD
chunks; callback slot order alternates. Allocation counters are enabled only
in this separate diagnostic; its times are noncomparative.

Examples for ColumnNormalized, chunk 8 (all other tested configurations also
remove exactly two requests and `16 * 3 * blocks` bytes):

| Case | Allocation requests before → after | Requested bytes before → after |
| --- | ---: | ---: |
| blocks32-ad0-chunk8 | 272 → 270 | 13,848 → 12,312 |
| blocks32-ad50-chunk8 | 592 → 590 | 31,768 → 30,232 |
| blocks32-ad100-chunk8 | 912 → 910 | 49,688 → 48,152 |
| blocks512-ad0-chunk8 | 4,269 → 4,267 | 212,424 → 187,848 |
| blocks512-ad50-chunk8 | 9,389 → 9,387 | 499,144 → 474,568 |
| blocks512-ad100-chunk8 | 14,509 → 14,507 | 785,864 → 761,288 |

This remains an allocating probe. The reduction is specific to the two full-
residual vectors; it is not an allocation-free implementation.

## Numerical and error contract

The scalar expression remains `Loss::Squared.reduction(0, abs(residual))`, in the
same row order as the former zero/residual-vector reduction. No FMA, reassociation
or replacement with an ordinary sum of squared norms is introduced. Empty/all-
zero cost preserves its signed-zero bits. Total-row checked addition remains
before callbacks, retaining dimension-overflow validation even without a row
buffer. The existing gradient/scaling reduction and finite checks are unchanged.

A cost arithmetic failure is stored until all callbacks finish. Remaining
callbacks, shape checks, whitening and gradient work still run; a later callback
error retains precedence over the earlier cost overflow, as in the original
materialized path. Cost failure is returned before final gradient checks. Failed
cost accumulation stops further cost arithmetic, matching the old reduction's
first-error behavior. The accumulator is local to each probe.

Two new tests compare blockwise cost to the original full-vector reduction over
empty/signed-zero/subnormal, mixed tiny/large and overflowing/nonfinite inputs,
with multiple block partitions, and verify that later callbacks run and retain
error precedence after cost overflow. Existing direct-versus-CSR bitwise tests
cover mixed manifolds, fixed slots, whitening, AD chunks and output failures.
Rust **325 library/integration/doctests + 2 benchmark tests**, Python **23**, all-
target Clippy, formatting and Rustdoc with warnings denied pass. All **42**
historical source-only rollback fingerprints match with zero-fuzz patches;
older historical kernels are not rebuilt here.

Complete [MIT functional regression](results/probe_cost/mit_functional/) passes
all 1,227 arrivals and archived-native agreement, with 742 AD / 741 analytic
factors and unchanged original information, including AD on ill-conditioned edge
160→161 near `1e12`. A fresh cold candidate is compared to the frozen preceding
`jet_negation/mit_functional/after` replay; complete pose bytes and non-timing
work match exactly. These timers are noncomparative. Maximum archived-native
pose/cost gaps are `1.945092214966316e-7` / `6.423306331271306e-12`.

## Reproduction

Use the prepared toolchain/cache and distinct Cargo target directories for each
source snapshot. The before full-stream executable is the actual preceding
measured binary, checked against its archive before rebuilding the current
binary. The before allocation driver was freshly compiled from frozen source;
before/after driver hashes match.

```sh
python3 benchmarks/make_ldl_baseline.py --before-probe-cost --source-only --output /tmp/ropt-before-probe-cost
cargo build --manifest-path /tmp/ropt-before-probe-cost/Cargo.toml --release --locked --offline -p ropt --example benchmark_autodiff_probe --example benchmark_ad_selective_se2_isam
cargo build --release --locked --offline -p ropt --example benchmark_autodiff_probe --example benchmark_ad_selective_se2_isam
python3 benchmarks/paired_probe_cost_allocations.py --binary target/release/examples/benchmark_autodiff_probe --before-binary /tmp/ropt-before-probe-cost/target/release/examples/benchmark_autodiff_probe --before-source /tmp/ropt-before-probe-cost --output /tmp/ropt-probe-allocations --rounds 2 --queries 1 --allocations --cpu 0
python3 benchmarks/paired_probe_cost.py --binary target/release/examples/benchmark_ad_selective_se2_isam --before-binary /tmp/ropt-before-probe-cost/target/release/examples/benchmark_ad_selective_se2_isam --before-source /tmp/ropt-before-probe-cost --input benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o --reference benchmarks/results/csail_real_isam2/primary --output /tmp/ropt-probe-cost-primary --rounds 6 --ad-chunk 6 --ad-stride 2 --cpu 0
python3 benchmarks/verify_probe_cost.py --current-source
```

Repeat the stream campaign in a fresh directory for confirmation. Verify records
without `--current-source` after further development. Before kernel:
`81722fe58b43addfedeae12d13a94c3062c5731627ff1b62285ffbeee305415d`.
After kernel: `41c27190a3d4eed0dcb9a00c48886cdd60222df5f20e33c0759dc119911d64f4`.
Drivers are hashed separately. The informal maturity estimate remains 75/100.
