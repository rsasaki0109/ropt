# Validity-preserving DynamicJet negation

Owned and borrowed negation now retain a previously validated Jet's finite
status instead of rescanning the resulting derivatives. A carried failure still
returns the original canonical invalid Jet before any mutation. Finite values
and partials multiplied by `-1` stay finite, including maximum magnitudes,
subnormals and signed zero. This mathematical closure is specific to negation;
general binary and unary operations retain their finite checks. Variable seeding
and the preceding width-aware binary loops keep their previous implementation.
No public API, callback/chunk behavior, layout, allocation policy, dependency,
unsafe code or global cache changes.

In isolated negation diagnostics, width-6 owned negation is **1.51x / 1.59x** as
fast (33.9% / 37.0% less time); borrowed negation is **1.19x / 1.17x** (15.9% /
14.4% less time). Both ownership cases win **6/6 pairs in both campaigns** at
this width. All tested owned widths improve in both campaigns. Borrowed width 1
has mixed paired results and does not meet the speed criterion in primary.
These are operation-specific results, not a whole-AD or native speed claim.

**A repeatable complete CSAIL speedup is not established.** Primary reduces
median time 5.1%, with 6/6 faster pairs; confirmation is 2.0% slower, with 3/6
faster pairs. Confirmation misses the existing case-specific criterion: median
ratio > 1 AND at least 75% faster pairs in each campaign. The negative result
remains recorded. All 25,056 timed arrivals pass the original-information gates
and archived-native pose/cost agreement; complete before/after poses and all
non-timing update fields match exactly in every paired round.

| Campaign | Generic negation before | Validity-preserving negation after | Time reduction | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Primary | 6.095 s | 5.783 s | 5.1% | 6/6 |
| Confirmation | 5.885 s | 6.001 s | -2.0% | 3/6 |

Negative reduction means more time. Timing includes probes, updates, failed
attempts, explicit full recoveries and owned reports; parsing/factor recipes and
serialization are excluded. Each fresh process discards one complete warmup
stream. Before/after order alternates, CPU 0, one thread, six pairs per campaign.
All builds/tests and isolated diagnostics finish before full-stream timing.

The dataset keeps 1,045 poses, 1,172 original edges, 1,044 arrivals, 586 AD / 586
analytic factors, chunk 6, original observations/information/seeds and the
preceding editable-QR/GN policy. Each kernel/campaign keeps 780 requested calls,
618 full recoveries, 7,662 probes and 2,425,338 committed re-eliminated scalar
cliques. Failed arithmetic is timed but not fully captured by successful-work
counters. Independent maximum normalized gradient remains
`5.8075741199751337e-8` (gate `1e-7`, internal target `8e-8`). Fixed root and
reported-cost tolerance `1e-10 * max(1,cost)` are unchanged; archived-native
pose/cost gates remain `2e-6` / `1e-9 * max(1,cost)`. Native timings are frozen.

## Isolated negation diagnostic

[Primary](results/jet_negation/negation/) and
[confirmation](results/jet_negation/negation_confirmation/) use the same new
standalone driver in the before/after builds, with separate Cargo target
directories. Seeding and checks happen outside timing. A maximum-finite primal
and a full-width seeded derivative vector undergo 4,194,304 alternating sign
changes after four warmup operations. The ownership cases are measured
separately. Black-box boundaries preserve each operation. Exact primal/partial
bits match before/after in every run. Allocation runs are separate, one operation
per process, repeated twice; their odd parity is checked separately from the
even timing runs and their times are not latency evidence.

| Width | Ownership | Primary before/after (faster pairs) | Confirmation before/after (faster pairs) |
| ---: | --- | ---: | ---: |
| 1 | owned | 1.288x (6/6) | 1.318x (6/6) |
| 1 | borrowed | 1.003x (4/6) | 1.110x (5/6) |
| 2 | owned | 1.297x (6/6) | 1.336x (6/6) |
| 2 | borrowed | 1.147x (6/6) | 1.111x (6/6) |
| 6 | owned | 1.512x (6/6) | 1.587x (6/6) |
| 6 | borrowed | 1.189x (6/6) | 1.168x (6/6) |
| 16 | owned | 1.524x (6/6) | 1.564x (6/6) |
| 16 | borrowed | 1.404x (6/6) | 1.470x (6/6) |
| 32 | owned | 2.089x (6/6) | 2.168x (6/6) |
| 32 | borrowed | 1.572x (6/6) | 1.469x (6/6) |
| 65 | owned | 2.434x (6/6) | 2.397x (6/6) |
| 65 | borrowed | 2.353x (6/6) | 2.239x (6/6) |

Allocation work is unchanged: owned negation has **zero allocation calls / bytes**;
borrowed negation has **one call / `8 * width` requested bytes** per operation.
Jet remains 40 bytes in this x86_64 build. These are allocation requests, not live
or peak memory. The diagnostic does not imply that arbitrary complete AD
callbacks improve by the same ratio.

## Correctness and validation

A new unit test compares both ownership forms to the original generic unary
arithmetic, covering maximum finite values/partials, smallest subnormals, signed
zero, constants, widths 0/1/6/16/17/32/65, unchanged borrowed inputs and carried
dimension/nonfinite failures. Existing owned binary tests now include finite
primals with overflowing partials (`1e308 * 1`, tiny-denominator division), in
addition to their previous overflow/NaN cases. Original borrowed binary/unary
arithmetic remains the independent bitwise oracle. Pointer tests still verify
consumed-buffer reuse.

Rust **323 library/integration/doctests + 2 benchmark tests**, Python **23**, all-
target Clippy, formatting and Rustdoc with warnings denied pass. All **41**
historical source-only rollback fingerprints match using zero-fuzz patches;
older historical kernels are not rebuilt here. The before negation diagnostic
was freshly built from the frozen preceding source with the identical new
driver; the before full-stream executable is the actual preceding measured
binary, checked against its archive before rebuilding current binaries.

Complete [MIT functional regression](results/jet_negation/mit_functional/)
passes all 1,227 arrivals, 742 AD / 741 analytic factors and original information,
including AD on ill-conditioned edge 160→161 near `1e12`. A fresh cold candidate
is compared to the frozen `jet_loop_split/mit_functional/after` replay, with
exact complete pose bytes and non-timing work. Their timers are noncomparative.
Archived-native agreement also passes (maximum pose gap `1.945092214966316e-7`,
cost gap `6.423306331271306e-12`). [Non-negation AD functionality](results/jet_negation/ad_functional/)
retains exact residual/Jacobian bits and allocation work for all ten preceding
widths. Its instrumented timers are not a new AD latency comparison.

## Rejected proposals

All three rejected proposals retain their exact source, actual binaries, original
runners, test/build logs and complete raw diagnostic records.

- [Fused finite checks](results/jet_negation/rejected_fusion/) combine validation
  into short binary loops. Width 6 does not improve consistently; width 7/16
  regress. This was rejected before full-stream timing.
- [Unbounded seed validation removal](results/jet_negation/unbounded_seeds/)
  includes negation and skips known-finite seed scans at every width. Small
  widths sometimes improve, but width 65 regresses in both campaigns, including
  about 20% in confirmation. Complete streams were not timed for this proposal.
- [Seed removal limited to short vectors](results/jet_negation/guarded_seeds/)
  retains the old wide seed path, but still regresses width 65 (about 12% / 8%).
  Complete CSAIL medians are 0.5% / 0.9% slower (3/6 and 4/6 faster pairs), with
  all 25,056 original-information arrivals accurate and exact before/after
  poses/work. It was rejected. The cause of the surprising wide slowdown was
  not established; fewer source-level scans are not sufficient speed evidence.

Final production change is negation only. No proposal's timing is relabeled as
another kernel or combined with current samples to claim a speed win.

## Reproduction

Use the prepared toolchain/cache and independent Cargo target directories for
source snapshots. Never share one target directory between different sources.

```sh
python3 benchmarks/make_ldl_baseline.py --before-jet-negation --source-only --output /tmp/ropt-before-jet-negation
cargo build --manifest-path /tmp/ropt-before-jet-negation/Cargo.toml --release --locked --offline -p ropt --example profile_dynamic_negation --example benchmark_ad_selective_se2_isam
cargo build --release --locked --offline -p ropt --example profile_dynamic_negation --example benchmark_ad_selective_se2_isam
python3 benchmarks/paired_jet_negation.py --binary target/release/examples/profile_dynamic_negation --before-binary /tmp/ropt-before-jet-negation/target/release/examples/profile_dynamic_negation --before-source /tmp/ropt-before-jet-negation --output /tmp/ropt-negation-primary --rounds 6 --iterations 4194304 --cpu 0
python3 benchmarks/paired_jet_negation_stream.py --binary target/release/examples/benchmark_ad_selective_se2_isam --before-binary /tmp/ropt-before-jet-negation/target/release/examples/benchmark_ad_selective_se2_isam --before-source /tmp/ropt-before-jet-negation --input benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o --reference benchmarks/results/csail_real_isam2/primary --output /tmp/ropt-negation-stream --rounds 6 --ad-chunk 6 --ad-stride 2 --cpu 0
python3 benchmarks/verify_jet_negation.py --current-source
```

Repeat each campaign with a fresh output directory for confirmation. Verify
archived records without `--current-source` after subsequent development.
Before kernel: `5e03888fc601316ba2c877e98b0544c6e5daab767dad1b9540d93e29d4760b8c`.
After kernel: `81722fe58b43addfedeae12d13a94c3062c5731627ff1b62285ffbeee305415d`.
Drivers are hashed separately. The informal maturity estimate stays 75/100.
