# Cached child and factor incidence in editable Gaussian updates

The editable Bayes-tree backend now retains sorted child and factor-owner lists
instead of rebuilding both lists on every nonempty Gaussian update. This
preserves numerical row order, Gaussian work, atomic failure behavior and the
public API.

On complete CSAIL, editable stream time falls **1.9%** in primary measurement
and **4.5%** in confirmation (**1.019x / 1.048x**, faster in **10/12 pairs**).
Both campaigns meet the existing case-specific speed criterion: median speedup
> 1 and at least 75% faster pairs. The effect is modest and does not establish
general SOTA or a native/GTSAM speedup. All accuracy gates pass and complete
before/after means and non-timing work records are identical.

| Campaign | Current full | Editable before | Editable after | Before / after | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary | 4.009 s | 4.363 s | 4.281 s | 1.019x | 5/6 |
| Confirmation | 4.023 s | 4.501 s | 4.296 s | 1.048x | 5/6 |

Values are medians of six complete warmed streams. All samples are retained.
**The improved editable policy remains slower than current full refresh**:
after/full speedups are only 0.937x / 0.936x, with 1/6 / 0/6 faster pairs.
This negative comparison is preserved alongside the modest before/after gain.

The [primary](results/incidence_cache/primary/) and
[confirmation](results/incidence_cache/confirmation/) campaigns independently
check **37,584 timed arrivals**. Each editable policy makes **780 requested
calls** and **618 full recoveries** per campaign, with exactly the same
**2,425,338 re-eliminated scalar cliques** in successful reports. The maximum
independently checked normalized gradient remains `5.636754354038671e-8`.
Elapsed time also includes failed attempts. [Final verification](results/incidence_cache/validation_checks.json)
records kernel/driver/binary/reference hashes, complete records and checks.

## Implementation

One shared immutable index stores each clique's direct children and original
Gaussian factor IDs. A same-support replacement changes neither list and
reuses the entire index. Adding/deleting factors or changing their support
patches only lists with changed membership. Changes in symbolic parents move
children between the corresponding old/new parent lists.

Changed IDs and child indices are visited in ascending order. Each changed list
is updated by one checked linear merge of sorted removals/additions, preserving
the preceding complete-enumeration order exactly. Unchanged list storage is
shared. Fresh full-model construction enumerates once, avoiding a delta-map
entry for every factor/clique. Column extension appends empty lists while
sharing all existing list storage. Empty/constant-only changes preserve the
index when membership is unchanged.

The candidate computes a new private index before numerical elimination and
commits it only after all finite-value, entry-budget and constant-cost checks
pass. Copies share index storage; mutation detaches the index and changed rows.
No QR downdate, normal-matrix subtraction, reordered observations or approximate
back-substitution is introduced.

Global parent comparison, factor bookkeeping, index-vector copying on structural
change, rank checking and back-substitution still visit global state. This is
not constant-time updating, dynamic ordering or wildfire. The retained index
adds metadata storage: two lists per column, at most one child link per non-root
clique and at most one owner membership per nonconstant Gaussian factor.
Existing dimension/factor budgets bound these counts; their thresholds and all
numerical workspace limits remain unchanged. Peak/live memory is not measured.

## Validation

A new regression test independently reconstructs all child/factor lists by
complete enumeration after permuted construction, repeated additions, an earlier
new parent, same-support replacement, owner migration, deletion, constant-only
factors and column extension. It checks exact sorted membership, actual shared
row identity, unchanged index identity for same-support/constant-only changes,
preserved snapshots, and unchanged public index/solution after a private
candidate's constant-cost overflow. Existing original-quadratic/SVD, symbolic
pruning, mixed-manifold, rank and fill-limit tests pass.

The suite passes **297 Rust unit/integration tests plus 18 doctests** (315 total),
**23 Python tests**, Clippy/rustdoc with warnings denied and formatting.
All **36 source-only historical fingerprints** match without patch fuzz;
those checks do not rebuild older binaries.

A complete MIT1228 functional replay passes all 1,227 original-information
gradient, true-cost, exact-root and archived-native agreement gates. Complete
poses and every non-timing update field match the preceding metadata-sharing
kernel exactly. Its [raw functional records](results/incidence_cache/mit_functional/)
are preserved; the uncontrolled cold timing is excluded from speed claims.

Current source fingerprint:
`c35917f74c8c8d32a434f5ee4f74b54eb9bca0ff2a461bfb37bd5d8afccce41b`.
Preceding measured source fingerprint:
`cdc6d2113705f42d7cf26e2379b4913507b53e4d4b5e6fe62843c84312bdb2f3`.

## Recorded comparison

Before/after editable policies and current full refresh run the same complete
CSAIL1045 stream: 1,044 arrivals, 1,172 unchanged original constraints, identical
original-odometry-only seeds and exact fixed root. Both editable policies use
the same one-request cap and explicit full recovery.

Each campaign balances all six permutations, with CPU 0, one thread, one
discarded complete warm stream and one timed stream per process. Timers include
nonlinear probes, failed attempts, requested relinearization and full recovery.
External checks and serialization are outside the timers; allocator counting is
disabled. Source, executable, driver and archived reference hashes are verified
before/after each campaign. The preceding executable is frozen from its actual
previous build, not recompiled through a shared target cache.

Every timed arrival must pass unchanged independent original-information gates:
normalized gradient <= 1e-7, exact fixed root, reported-cost error
<= 1e-10 * max(1, cost), original-cost disagreement <= 1e-9 * max(1, cost),
and mixed world-pose disagreement <= 2e-6. Before/after complete pose exports
and all non-timing update fields must also match exactly in every round.

No new native/GTSAM timings are collected. Hashed archived precision-corrected
native poses/costs only check correctness agreement; published native figures
retain their measured kernels. All timings are retained without trimming.

## Reproduce

```sh
python3 benchmarks/make_ldl_baseline.py \
  --output /tmp/ropt-before-incidence --before-incidence-cache
cargo build -p ropt --example benchmark_selective_se2_isam --release --locked
python3 benchmarks/paired_incidence_cache.py \
  --binary target/release/examples/benchmark_selective_se2_isam \
  --before-binary /tmp/ropt-before-incidence/target/release/examples/benchmark_selective_se2_isam \
  --before-source /tmp/ropt-before-incidence \
  --input benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o \
  --reference benchmarks/results/csail_real_isam2/primary \
  --output /tmp/incidence-primary --rounds 6 --cpu 0 --selected-cap 1
```

Repeat into a new output directory for confirmation. Partial streams, failed
accuracy checks, changed means/work or changed kernels cannot produce an
eligible report.

The informal maturity estimate remains **75%**. Dynamic reordering, wildfire,
incremental fixed-lag removal and field evidence remain open.
