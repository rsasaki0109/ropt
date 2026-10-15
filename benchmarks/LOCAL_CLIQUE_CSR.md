# Canonical CSR assembly for editable Gaussian cliques

The editable Bayes-tree backend now builds each local Jacobian directly from
canonical rows. It preserves the Gaussian observation and separator-message
order, arithmetic, QR policy, fill limits, and atomic update behavior.

On complete CSAIL, editable stream time falls by **16.0%** in the primary
campaign and **11.2%** in confirmation (**1.19x / 1.13x**, faster in **12/12
pairs**). Every original-information accuracy gate passes, and before/after
complete poses and non-timing work records match exactly. **The improved
editable policy remains slower than current full refresh.**

| Campaign | Current full | Editable before | Editable after | Before / after | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary | 3.908 s | 5.307 s | 4.459 s | 1.190x | 6/6 |
| Confirmation | 3.970 s | 5.062 s | 4.496 s | 1.126x | 6/6 |

Times are medians of six complete warmed streams. All samples are retained,
including the primary full-refresh sample at 5.410 s; no outlier trimming is
used. Against full refresh, after/full speedups are only 0.876x / 0.883x, with
1/6 / 0/6 faster pairs. This negative comparison is retained alongside the
positive before/after result. These measurements support this specific ropt
backend optimization, not a general speed or SOTA claim.

Raw [primary](results/local_clique_csr/primary/) and
[confirmation](results/local_clique_csr/confirmation/) records and
[final verification](results/local_clique_csr/validation_checks.json) are saved.
The campaigns independently check **37,584 timed arrivals**. In each campaign,
each editable policy makes **780 requested calls** and **618 full recoveries**;
successful Gaussian work totals **2,425,338 re-eliminated scalar cliques**.
The maximum independently checked normalized gradient is
`5.636754354038671e-8` for both editable kernels. Work counts describe committed
reports; elapsed time also includes failed attempts.

An instrumented, cold complete CSAIL replay of the preceding kernel spent
0.489 s gathering rows and building local CSR matrices, compared with 0.397 s
in local QR. This diagnostic includes per-clique clocks and is excluded from
comparative speed claims. Its
[patch and stage totals](results/local_clique_csr/diagnostics/) are saved;
instrumentation is absent from the production and measured executables.

## Implementation

Original factor rows have unique columns. Mapping through a bijective
elimination permutation and projecting onto a sorted clique is injective;
therefore only a small per-row sort is needed, with no duplicate reduction.
Cached separator-message rows already use sorted internal columns, so their
projection needs no sorting. The builder emits checked canonical CSR arrays
directly and reuses one small projection buffer across original rows. It also
avoids a per-clique column tree map and a zero-initialized residual vector.

The original factor and child order, signed residual values, empty observation
rows, and carried constants are preserved. The local workspace, factor,
separator-message, conditional, and symbolic entry limits are unchanged. Public
updates still operate on a private candidate and commit only on success.
Metadata cloning/bookkeeping, rank checking and back-substitution still visit
global state; this change does not provide dynamic ordering or wildfire.

A new unit test compares canonical gathering with independent triplet assembly
under a nontrivial permutation, empty observation rows, signed zeros,
coefficients from 1e-150 to 1e150, and informative boundary messages. Existing
tests independently compare the full quadratic with dense SVD, including
replacement, deletion, changed support, column extension, rank failures and
entry-budget rollback.

## Validation and scope

The source suite passes **295 Rust unit/integration tests plus 18 doctests**
(313 total), **23 Python tests**, Clippy with warnings denied, rustdoc with
warnings denied and formatting. All **34 source-only historical fingerprints**
match without patch fuzz; older binaries are not rebuilt in that check.

A complete MIT1228 functional replay passes all 1,227 original-information
gradient, true-cost, fixed-root and archived-native agreement checks. Its
complete pose export and every non-timing update field are identical to the
preceding editable kernel. This cold replay is excluded from speed claims; the
[raw records and independent quality results](results/local_clique_csr/mit_functional/)
are retained.

The current source fingerprint is
`cae5cfa485723513f5799fb7722043b2d39c0299639535c1bd78775d424c9323`.
The preceding measured source fingerprint is
`53671e84cba965edd4c45dc87a63570f86e0acdaf3da06d29111884cb36ebdaa`.

The recorded comparison runs current full refresh and before/after editable
policies on the complete CSAIL1045 stream (1,044 arrivals, 1,172 original
constraints). Both editable policies use the same one-request cap, explicit
full recovery, original information matrices and fixed odometry-only seeds.
Each campaign balances all six permutations of the three executables/policies,
with CPU 0, one thread, one discarded complete warm stream and one timed stream
per process. Timing includes nonlinear probes, failed attempts and full
recoveries. Checking and serialization are outside the timers. Allocator
instrumentation is disabled and memory is not measured.

Every timed arrival must pass the existing independent gates: original
column-normalized gradient <= 1e-7, exact fixed root, reported-cost error
<= 1e-10 * max(1, cost), original-cost disagreement <= 1e-9 * max(1, cost),
and maximum mixed world-pose disagreement <= 2e-6. Before/after complete means
and non-timing update fields must also match exactly in every round.
No fresh GTSAM/native timing is collected; hashed archived native poses/costs
only check agreement. Previous published figures keep their measured kernels.

## Reproduce

```sh
python3 benchmarks/make_ldl_baseline.py \
  --output /tmp/ropt-before-clique-csr --before-local-clique-csr
cargo build -p ropt --example benchmark_selective_se2_isam --release --locked
python3 benchmarks/paired_clique_csr.py \
  --binary target/release/examples/benchmark_selective_se2_isam \
  --before-binary /tmp/ropt-before-clique-csr/target/release/examples/benchmark_selective_se2_isam \
  --before-source /tmp/ropt-before-clique-csr \
  --input benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o \
  --reference benchmarks/results/csail_real_isam2/primary \
  --output /tmp/clique-csr-primary --rounds 6 --cpu 0 --selected-cap 1
```

Repeat into a new output directory for confirmation. The runner verifies
source, executable, driver and reference hashes before/after each campaign;
incomplete streams, changed work, changed means or failed accuracy checks cannot
produce an eligible report.

The informal maturity estimate remains **75%**. General editable-tree latency,
dynamic reordering, wildfire and incremental fixed-lag removal remain open.
