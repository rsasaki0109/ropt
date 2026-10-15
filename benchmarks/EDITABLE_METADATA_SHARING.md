# Sharing editable Gaussian metadata across snapshots

Editable Bayes-tree candidates and snapshots now share immutable factor and
ownership maps through `Arc`. A changed map is copied before mutation with
`Arc::make_mut`; all public updates still commit only a successful private
candidate. This reduces repeated map copies without changing the Gaussian model,
nonlinear policy, error handling or public API.

A separate 2,048-variable/2,048-factor snapshot diagnostic finds **99.1% fewer
allocation/reallocation requests** and **56.4% fewer requested bytes** across
64 clones. Counts repeat exactly twice per independently built kernel. These
are clone-only allocation traffic, not live/peak memory or solver latency.

**A repeatable complete-stream speed improvement is not established.** On
complete CSAIL, the primary median improves by 2.1%, but confirmation regresses
by 0.9%; only 4/6 and 2/6 pairs improve. Neither campaign meets the existing
speed criterion (median speedup > 1 and at least 75% faster pairs). All accuracy
gates pass and before/after means and non-timing work records match exactly.

| Campaign | Current full | Editable before | Editable after | Before / after | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary | 4.039 s | 4.527 s | 4.433 s | 1.021x | 4/6 |
| Confirmation | 4.051 s | 4.479 s | 4.519 s | 0.991x | 2/6 |

Each value is the median of six complete warmed streams; all samples are
retained without trimming. The editable backend also remains slower than full
refresh (0.911x / 0.896x, 0/6 faster pairs in each campaign). These negative and
mixed results are retained; this change is a metadata-copy improvement with no
claim of better complete-stream latency or general SOTA.

The [primary](results/editable_cow/primary/) and
[confirmation](results/editable_cow/confirmation/) records independently check
**37,584 timed arrivals**. Each editable kernel makes **780 requested calls**
and **618 full recoveries** per campaign, with exactly the same **2,425,338
re-eliminated scalar cliques** in successful reports. The maximum independently
checked normalized gradient remains `5.636754354038671e-8`.
Elapsed time also includes failed attempts.

## Implementation and checks

The nonlinear smoother already makes a private Gaussian-cache copy before an
update. Column extension and the Gaussian update also create private candidates
to preserve state on failure. Previously each of these clones copied both
complete tree maps, even when the operation did not modify them.

With sharing, column extension changes neither map. Same-support factor
replacement changes the factor map but preserves the shared ownership map.
Adding factors detaches both maps; deletion or changed support rebuilds ownership
through the existing complete topology analysis. Individual original Gaussian
factors remain shared immutable blocks. Map iteration order, original row order,
QR arithmetic, factor/message/conditional/workspace limits and all work counts
are unchanged.

This is copy-on-write for **two metadata maps**, not a persistent tree or
constant-time snapshot/update. Detaching a changed map still copies O(n)
entries. Conditional/message/permutation vectors, metadata bookkeeping, rank
checking and back-substitution still visit global state. Memory and allocator
counts are not measured by the speed comparison.

A new regression test verifies actual shared map identity after cloning and
column extension, detachment on replacement/addition/deletion, preserved owners
on same-support replacement, independent means after updating two clones, and
unchanged public map identity/report/solution after a private candidate's
constant-cost overflow. Closed-form means provide an independent numerical
reference. Existing original-quadratic/SVD, permutation, mixed-chart, fill-budget
and rollback tests remain in the full suite.

The complete suite passes **296 Rust unit/integration tests plus 18 doctests**
(314 total), **23 Python tests**, Clippy/rustdoc with warnings denied, and
formatting. All **35 source-only historical fingerprints** match without patch
fuzz; those checks do not rebuild older binaries.

The complete MIT1228 functional replay passes all 1,227 original-information
gradient, true-cost, fixed-root and archived-native agreement checks. Its entire
pose export and non-timing update fields are identical to the preceding
canonical-CSR kernel. This cold functional replay is excluded from speed claims;
its [records](results/editable_cow/mit_functional/) are retained.

Current source fingerprint:
`cdc6d2113705f42d7cf26e2379b4913507b53e4d4b5e6fe62843c84312bdb2f3`.
Preceding measured source fingerprint:
`cae5cfa485723513f5799fb7722043b2d39c0299639535c1bd78775d424c9323`.

## Recorded comparison

The before/after comparison uses the same complete CSAIL1045 stream (1,044
arrivals, 1,172 original constraints), original information matrices, exact
fixed root and original-odometry-only seeds. Both editable policies use the
same one-request cap and explicit full recovery; current full refresh is a
third comparison policy.

Each campaign balances all six permutations, CPU 0, one thread, one discarded
complete warm stream and one timed stream per process. Timers include nonlinear
probes, failed attempts, requested relinearization and full recovery. External
checking and serialization are outside the timers. Sources, executables,
drivers and archived reference hashes are verified before/after each campaign.

Every timed arrival must pass unchanged independent original-information gates:
normalized gradient <= 1e-7, reported-cost error <= 1e-10 * max(1, cost),
original-cost disagreement <= 1e-9 * max(1, cost), mixed world-pose disagreement
<= 2e-6, and exact fixed root. Complete before/after pose exports and all
non-timing update fields must additionally match exactly in every round.

Native/GTSAM timings are not collected. Hashed archived precision-corrected
GTSAM poses/costs only check correctness agreement. Published native figures
retain their original measured kernels.

## Snapshot allocation traffic

The separate [clone diagnostic](results/editable_cow/clone_diagnostic/) uses
2,048 independent unit-information Gaussian priors. Their closed-form means
and zero energy are checked before/after cloning. Eight untimed/uninstrumented
clones warm the allocator; counters cover only construction/destruction of 64
snapshots, excluding model construction, solves, output and warmup. No latency
comparison is made with counters enabled. Both executions of each kernel yield
the following exact totals:

| Kernel | Allocation/reallocation calls | Requested bytes |
| --- | ---: | ---: |
| Before | 43,904 | 19,249,152 |
| Shared metadata | 384 | 8,388,608 |

Requested bytes sum successful allocation/reallocation requests, including full
requested realloc sizes; they are not live or peak bytes. Snapshot cloning still
allocates its global vectors: six calls and 131,072 requested bytes per clone.
Changing a shared map still requires a complete map copy.

Each kernel is compiled in its own Cargo target directory from a checked source
snapshot, with the embedded README and exact profiler/counter sources saved.
Distinct executable hashes and identical repeated counts are required. An
initial shared-target attempt reused one executable for different source trees
and is explicitly [ineligible](results/editable_cow/ineligible_clone_diagnostic/);
a missing-README build is also retained as
[ineligible](results/editable_cow/ineligible_readme_build/). Neither contributes
to the totals or latency campaigns. The latency campaigns use the original
frozen preceding executable and the freshly built current executable.

## Reproduce

```sh
python3 benchmarks/make_ldl_baseline.py \
  --output /tmp/ropt-before-editable-sharing --before-editable-cow
cargo build -p ropt --example benchmark_selective_se2_isam --release --locked
python3 benchmarks/paired_editable_cow.py \
  --binary target/release/examples/benchmark_selective_se2_isam \
  --before-binary /tmp/ropt-before-editable-sharing/target/release/examples/benchmark_selective_se2_isam \
  --before-source /tmp/ropt-before-editable-sharing \
  --input benchmarks/results/csail_real_isam2/csail-se-sync1045-original.g2o \
  --reference benchmarks/results/csail_real_isam2/primary \
  --output /tmp/editable-sharing-primary --rounds 6 --cpu 0 --selected-cap 1
```

Repeat into a new directory for confirmation. The runner cannot mark a partial
stream, failed accuracy gate, changed mean/work record or changed kernel as
eligible. All timings are retained without trimming.

After both latency campaigns finish, run the isolated allocation diagnostic:

```sh
python3 benchmarks/profile_editable_metadata.py \
  --before-source /tmp/ropt-before-editable-sharing \
  --output /tmp/editable-sharing-clones
```

[Final verification](results/editable_cow/validation_checks.json) records
source/driver/binary/reference checks, exact means/work, tests and rollback
fingerprints.

The informal maturity estimate remains **75%**. Dynamic ordering, wildfire,
incremental fixed-lag removal and field evidence remain open.
