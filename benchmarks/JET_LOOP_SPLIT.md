# Width-aware owned DynamicJet loops

Large runtime-AD derivative vectors now choose the broadcast or equal-length
loop once after dimension validation. Equal-length vectors use zipped slices;
constants keep explicit zero products. Widths up to 16 retain the compact indexed
loop. All widths still reuse a consumed operand's derivative buffer. This is an
empirical loop choice for this build, not a dimension restriction or a guaranteed
crossover on other hardware. No public API, callback/chunk policy, algebra, Jet
layout (40 bytes), dependency or allocator policy changes.

**Width 65 evaluation time falls 24.5% / 24.6% in the diagnostic campaigns (6/6 faster
pairs each); width 32 falls 16.3% / 18.6%.** Width 17 also improves in both. Small
widths show mixed results, including a slower width-1 primary. Allocation calls
and requested bytes remain exactly unchanged from the preceding buffer-reuse
kernel. No general AD, BA/SE3, peak-memory or native/GTSAM speed claim is made.

**A repeatable complete CSAIL speedup is not established.** Primary median time
falls 1.7%, but only 4/6 pairs are faster; confirmation is 2.9% slower, with only
1/6 faster pairs. Both miss the existing case-specific speed criterion (median
ratio > 1 AND at least 75% faster pairs in each campaign). These negatives remain
visible; the current improvement is specific to larger derivative widths. All
25,056 timed CSAIL arrivals pass unchanged independent original-information and
archived-native accuracy gates. Complete before/after poses and all non-timing
update fields match exactly in every paired round.

| Campaign | Indexed before | Width-aware after | Time reduction | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Primary | 6.104 s | 6.000 s | 1.7% | 4/6 |
| Confirmation | 5.926 s | 6.099 s | -2.9% | 1/6 |

Negative reduction means more time. Timing includes probes, updates, failed
attempts, explicit recoveries and owned reports; parsing/factor recipes and
serialization are outside timing. Each fresh process discards one complete
warmup stream; order alternates, CPU 0, one thread, six pairs per campaign.
Builds/tests and width diagnostics finish before complete-stream timing.

The complete dataset retains 1,045 poses, 1,172 original edges and 1,044 arrivals,
586 AD / 586 analytic factors, chunk 6, original observations/information/seeds,
fixed root and the preceding editable-QR/GN policy. Each kernel/campaign retains
780 requested calls, 618 full recoveries, 7,662 probes and 2,425,338 committed
re-eliminated scalar cliques. Failed arithmetic is timed but is not fully
represented in successful-work counters. Maximum independent normalized
gradient is unchanged at `5.8075741199751337e-8` (gate `1e-7`, internal `8e-8`).
Reported-cost tolerance remains `1e-10 * max(1,cost)`; archived-native pose/cost
gates remain `2e-6` / `1e-9 * max(1,cost)`. Native timings were not measured.

## Separate width diagnostics

[Primary](results/jet_loop_split/widths/) and
[confirmation](results/jet_loop_split/widths_confirmation/) cover all listed
widths, including both sides of the empirical switch. Each process evaluates
three nonlinear residuals with a full-width chunk, validates independent analytic
formulas and records exact residual/Jacobian bits. Before/after bits match in
all runs. Latency uses `16384 * max(1, 32 // width)` evaluations per process and
four discarded warmups to reduce very short sample noise. Within each width,
both kernels do exactly the same work. Allocation counters run separately, one
evaluation per process, twice per kernel; counts repeat exactly. Instrumented
times are not latency evidence.

| Width | Primary before/after (faster pairs) | Confirmation before/after (faster pairs) | Unchanged allocation calls / requested bytes per evaluation |
| ---: | ---: | ---: | ---: |
| 1 | 0.942x (0/6) | 0.983x (4/6) | 9 / 336 |
| 2 | 1.020x (4/6) | 1.037x (5/6) | 14 / 472 |
| 6 | 1.041x (5/6) | 0.975x (4/6) | 35 / 2,136 |
| 7 | 1.007x (4/6) | 1.012x (3/6) | 40 / 2,672 |
| 12 | 0.972x (2/6) | 0.998x (3/6) | 66 / 7,192 |
| 13 | 0.958x (2/6) | 1.007x (4/6) | 71 / 8,208 |
| 16 | 0.985x (3/6) | 1.017x (3/6) | 86 / 11,736 |
| 17 | 1.076x (6/6) | 1.084x (6/6) | 92 / 14,352 |
| 32 | 1.195x (6/6) | 1.228x (6/6) | 167 / 43,992 |
| 65 | 1.325x (6/6) | 1.326x (6/6) | 334 / 180,240 |

## Correctness and rejected experiment

The original allocating borrowed arithmetic remains the independent bitwise
unit-test oracle. Existing tests now include widths 16/17 for owned forms,
constant broadcasting, dimension failures, signed zero, tiny/large derivatives,
nonfinite arithmetic, failure precedence, unchanged borrowed inputs and buffer
pointer reuse. Explicit `a_i * left + b_i * right` order and missing-dimension
zero products remain intact; no FMA, unsafe code or zero-product omission is used.
Rust 322 library/integration/doctests + 2 benchmark tests, Python 23, all-target
Clippy, formatting and Rustdoc with warnings denied pass. All 40 historical
source-only rollback fingerprints match; older kernels are not rebuilt here.

Complete [MIT functional regression](results/jet_loop_split/mit_functional/)
passes all 1,227 arrivals, with 742 AD / 741 analytic factors, original
information and AD on the ill-conditioned edge 160→161 near `1e12`. A fresh cold
candidate replay is compared to the frozen preceding-kernel replay from
`jet_buffer_reuse/mit_functional/after`; their complete pose bytes and non-timing
work match exactly. These timers are noncomparative. Archived-native agreement
also passes (maximum pose gap `1.945092214966316e-7`, cost gap
`6.423306331271306e-12`).

The [unconditionally split loop trial](results/jet_loop_split/split_trial/)
is retained with its exact source, actual executables, original runners, tests,
raw width records and both complete CSAIL campaigns. It improved wide diagnostics
but slowed complete CSAIL medians 2.5% / 3.3% (3/6 and 1/6 faster pairs), so it
was rejected. Its short width samples used the original constant 16,384 iteration
count; they are not relabeled as the final calibrated diagnostic. Trial source:
`91e09bb217049c2145716355e474703f331b5316c49b5655947d90a20a352945`.

## Reproduction

Use the prepared Rust toolchain/cache and independent Cargo target directories
for each source snapshot. The frozen before executables were checked against
the preceding measured archive before rebuilding current binaries.

```sh
python3 benchmarks/make_ldl_baseline.py --before-jet-loops --source-only --output /tmp/ropt-before-jet-loops
cargo build --manifest-path /tmp/ropt-before-jet-loops/Cargo.toml --release --locked --offline -p ropt --example profile_dynamic_jet --example benchmark_ad_selective_se2_isam
cargo build --release --locked --offline -p ropt --example profile_dynamic_jet --example benchmark_ad_selective_se2_isam
python3 benchmarks/paired_jet_loop_widths.py --binary target/release/examples/profile_dynamic_jet --before-binary /tmp/ropt-before-jet-loops/target/release/examples/profile_dynamic_jet --before-source /tmp/ropt-before-jet-loops --output /tmp/ropt-loop-widths --rounds 6 --iterations 16384 --cpu 0
python3 benchmarks/paired_jet_loop_stream.py --binary target/release/examples/benchmark_ad_selective_se2_isam --before-binary /tmp/ropt-before-jet-loops/target/release/examples/benchmark_ad_selective_se2_isam --before-source /tmp/ropt-before-jet-loops --input benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o --reference benchmarks/results/csail_real_isam2/primary --output /tmp/ropt-loop-primary --rounds 6 --ad-chunk 6 --ad-stride 2 --cpu 0
python3 benchmarks/verify_jet_loop_split.py --current-source
```

Repeat each diagnostic/campaign with a fresh output directory for confirmation.
Verify archived records without `--current-source` after further development.
Before kernel: `0c171c75e64947bb8995b1b77be8d16fb7b9fb1eca3938c5c19f67906dc0c469`.
After kernel: `5e03888fc601316ba2c877e98b0544c6e5daab767dad1b9540d93e29d4760b8c`.
Drivers are pinned separately; earlier native figures retain their measured
kernels. The informal maturity estimate remains 75/100.
