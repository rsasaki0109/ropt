# Reusing consumed DynamicJet derivative buffers

Owned `DynamicJet` arithmetic now reuses a consumed operand's `Vec<f64>` instead
of allocating a derivative vector for every operation. This is a Rust ownership
optimization: public dimensions, operators, callback/chunk policies, right
retractions and numerical expressions remain unchanged. The type still contains
a `Vec`; its measured size remains **40 bytes** in this x86_64 build.

Complete mixed-AD CSAIL update time falls **1.2% / 4.6%** in primary and
confirmation. Both campaigns win **5/6 pairs**, meeting the existing case-specific
criterion: median speedup > 1 and at least 75% faster pairs in each campaign.
The improvement is modest and specific to this workload/policy. Slower pairs
remain in the records. No fresh native/GTSAM timing or general superiority claim
is made.

| Campaign | Allocating arithmetic before | Owned-buffer reuse after | Median time reduction | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Primary | 6.130 s | 6.059 s | 1.2% | 5/6 |
| Confirmation | 6.142 s | 5.862 s | 4.6% | 5/6 |

[Primary](results/jet_buffer_reuse/primary/) and
[confirmation](results/jet_buffer_reuse/confirmation/) contain **25,056 timed
arrivals**: two kernels × six complete 1,044-arrival streams per campaign.
Every process discards one complete warmup stream. Before/after order alternates
on CPU 0 with one thread. Builds/tests/profiling finish before timed campaigns;
independent quality checks run after timing. Parsing, factor recipe construction
and serialization are excluded; probes, updates, rejected attempts, requested
relinearization, full recovery and owned reports are included.

All original-information gates and archived-native pose/cost agreement pass.
Complete poses and all non-timing update fields are **byte-identical before/after
in every paired round**. Independent maximum normalized gradient is unchanged
at `5.8075741199751337e-8`; the gate remains `1e-7` and internal target `8e-8`.
The exact fixed root, cost tolerance `1e-10 * max(1,cost)`, native pose norm
`2e-6` and native cost tolerance `1e-9 * max(1,cost)` are unchanged.
Each kernel/campaign retains 780 requested calls, 618 full recoveries, 7,662 probes
and 2,425,338 re-eliminated scalar cliques in committed reports. Rejected work is
timed but successful-work counters do not capture all rejected arithmetic.
The workload preserves all 1,172 edges: 586 runtime AD / 586 analytic factors,
chunk size 6, original odometry seeds/information, Cartesian right SE2 tangents,
ordered sparse QR/GN, topology reuse and four final refinement steps.

## Allocation and width diagnostic

A separate [Euclidean AD diagnostic](results/jet_buffer_reuse/widths/) covers
widths 1/2/6/7/12/13/32/65. Each evaluation has three nonlinear residuals and a
full-width derivative chunk. Setup/output buffers are outside measurement;
results are checked against independent analytic formulas. Residual/Jacobian
bits match the frozen before kernel for every diagnostic. Latency uses 16,384
evaluations per process, four discarded warmups and six balanced pairs. Separate
allocation runs repeat both kernels twice with counters enabled and one evaluation;
counts/bytes repeat exactly. Instrumented times are not used for speed conclusions.

| Width | Allocation calls before → after | Requested bytes before → after | Before / after median latency | Faster pairs |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 13 → 9 | 368 → 336 | 1.041x | 5/6 |
| 2 | 22 → 14 | 600 → 472 | 1.111x | 6/6 |
| 6 | 59 → 35 | 3,288 → 2,136 | 1.148x | 5/6 |
| 7 | 68 → 40 | 4,240 → 2,672 | 1.135x | 4/6 |
| 12 | 114 → 66 | 11,800 → 7,192 | 1.097x | 6/6 |
| 13 | 123 → 71 | 13,616 → 8,208 | 1.015x | 4/6 |
| 32 | 295 → 167 | 76,760 → 43,992 | 0.999x | 3/6 |
| 65 | 594 → 334 | 315,440 → 180,240 | 0.945x | 2/6 |

Allocation calls fall **30.8–43.8%**, requested bytes **8.7–42.9%** across these
widths. Width 6 falls from **59 to 35 calls** and **3,288 to 2,136 bytes**.
These are successful allocation/reallocation requests, **not live or peak memory**.
The optimization retains borrowed-output, seeding, unary and adapter allocations;
it is not allocation-free.

**The width-65 latency diagnostic is about 5.5% slower (0.945x, 2/6 faster
pairs), despite lower allocation traffic.** Width 32 is effectively unchanged
(0.999x, 3/6). Positive medians at other widths are not a universal speed claim;
width 7/13 win only 4/6 pairs. This large-width tradeoff remains visible alongside
the complete mixed-AD stream gain. No complete wide-parameter, BA or SE3 latency
comparison is inferred from this diagnostic.

## Numerical and ownership contract

Both owned operands prefer an already active derivative buffer; one owned
operand reuses its buffer, resizing only when a zero-dimensional constant must
broadcast into a borrowed active derivative space. Two borrowed operands retain
the original allocating path. Owned negation reuses its vector; borrowed unary
functions retain their current signatures and behavior. No memory pool, global
cache, unsafe code or new dependency is introduced.

Validation precedes mutation: carried failures and incompatible nonzero
derivative lengths retain the original precedence. Borrowed inputs remain
untouched. The reduction still computes `a_i * left + b_i * right` in operand
order, including explicit missing-dimension zero products. It does not simplify
zero multiplication, fuse operations or skip nonfinite checks. The primal
expressions, constant broadcasting, signed-zero behavior and error payloads
remain unchanged. Fixed variables still participate in all AD chunks.

Two new unit tests compare every owned form against the frozen allocating
arithmetic, including widths 0/1/2/6/7/12/13/32/65, swapped scalar/active operands,
signed zeros, tiny/large derivatives, primal overflow/NaN, dimension mismatch,
failure precedence and unchanged borrowed inputs. Pointer checks verify consumed
active vectors are reused for either operand and negation. Existing chunk,
central-difference, manifold, AD projection/BA, noise, solver, ownership rollback
and probe tests pass. **Rust 322 library/integration/doctests + 2 benchmark tests;
Python 23; all-target Clippy, formatting and Rustdoc with warnings denied pass.**
All **39 historical source-only rollback fingerprints** match archived kernels
using zero-fuzz patches; older historical kernels are not rebuilt this turn.

Complete [MIT functional regression](results/jet_buffer_reuse/mit_functional/)
retains all 1,227 arrivals with 742 AD / 741 analytic factors and unchanged
information, including AD on the ill-conditioned original edge 160→161 near
`1e12`. Original gates and archived-native agreement pass; complete before/after
pose bytes and non-timing work match exactly. These cold functional processes
overlap, so their timers are not comparative speed evidence.

## Preserved development experiments

The [fixed-storage prototype](results/jet_buffer_reuse/rejected_inline/) eliminates
many small derivative allocations but expands Jet from 40 to 80 bytes and slows
all measured widths. It was rejected. [Unbounded initial reuse](results/jet_buffer_reuse/unbounded_trial/),
[private](results/jet_buffer_reuse/private_guard_trial/) and
[public width guards](results/jet_buffer_reuse/public_guard_trial/), and
[inlining](results/jet_buffer_reuse/inlining_trial/) retain their exact source,
raw width diagnostics and driver hashes. Width guards/inlining did not improve
performance reliably and are not in the final kernel. No source or timing from
these intermediate experiments is relabeled as the final complete-stream result.
Final implementation is the simple general owned-buffer reuse. Final production
profile executable is identical to the initial unbounded trial; later changes to
its source fingerprint strengthen the independent unit-test oracle and coverage.

## Reproduction

Use the prepared toolchain/cache and separate Cargo target directories. Never
share a target directory across different source snapshots.

```sh
python3 benchmarks/make_ldl_baseline.py --before-jet-buffers --source-only --output /tmp/ropt-before-jet-buffers
cargo build --manifest-path /tmp/ropt-before-jet-buffers/Cargo.toml --release --locked --offline -p ropt --example benchmark_ad_selective_se2_isam --example profile_dynamic_jet
cargo build --release --locked --offline -p ropt --example benchmark_ad_selective_se2_isam --example profile_dynamic_jet
python3 benchmarks/paired_jet_buffer_stream.py --binary target/release/examples/benchmark_ad_selective_se2_isam --before-binary /tmp/ropt-before-jet-buffers/target/release/examples/benchmark_ad_selective_se2_isam --before-source /tmp/ropt-before-jet-buffers --input benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o --reference benchmarks/results/csail_real_isam2/primary --output /tmp/ropt-jet-buffer-primary --rounds 6 --ad-chunk 6 --ad-stride 2 --cpu 0
python3 benchmarks/paired_jet_buffer_widths.py --binary target/release/examples/profile_dynamic_jet --before-binary /tmp/ropt-before-jet-buffers/target/release/examples/profile_dynamic_jet --before-source /tmp/ropt-before-jet-buffers --output /tmp/ropt-jet-buffer-widths --rounds 6 --iterations 16384 --cpu 0
```

Repeat the stream command with a fresh output directory for confirmation.
Complete builds/tests before any latency run and run diagnostics separately.
Verify retained records with `python3 benchmarks/verify_jet_buffer_reuse.py --current-source`;
omit `--current-source` after subsequent development.

Before fingerprint:
`5a2f130b59b3b3c8a45f97a825d3339c68f0611fa03c00b63c667246c1eef63e`.
After fingerprint:
`0c171c75e64947bb8995b1b77be8d16fb7b9fb1eca3938c5c19f67906dc0c469`.
Drivers are hashed separately. The before real-stream binary is the frozen actual
executable of the preceding measured kernel, whose SHA matches the previous
[real-AD archive](results/real_ad_probe/primary/metadata.json). Timing and source
hashes are checked again at completion. Earlier native figures remain frozen.
