# Incremental update overhead and independent stream validation

The initial iSAM-style implementation reevaluated only new factors, but ordinary
updates still cloned the old recipe list and built three temporary graphs,
including one registering the entire factor history. Graph construction also
called `set_fixed(false)` for every newly registered active variable, rebuilding
all tangent ranges even though its fixed flag had not changed.

## Changes and safety contracts

An ordinary update now builds only its new-factor graph, which already contains
the complete variable layout in insertion order. Its IDs/ranges serve both
provisional estimate extraction and the final Local-coordinate cost check.
Old recipes are neither cloned nor registered on this path. New recipes append
to committed history only after all checks succeed. A refresh builds one
additional full-factor graph; evaluating its optimized ambient vector directly
rebuilds QR, avoiding another graph construction at the new anchors.

`FactorGraph::set_fixed` validates owner/index and returns immediately if the
flag is unchanged. Real changes still rebuild the tangent layout. This removes
repeated layout scans for already-active variables during smoother graph builds.
It does not change column order, Jacobians, rotations, weights or solve equations.

The full-refresh dense matrix guard now applies to Dense/DenseQr/DenseSchur
backends. Explicit sparse selections use their own backend limits and the
incremental QR stored-fill limit. Dense QR remains the default. These are entry
limits, not total byte/RSS bounds; input matrices, preconditioners and temporary
containers also allocate. Rebuilding the undamped QR still requires an
observable model even if the nonlinear solve used damping.

New regressions cover mixed SO3/Euclidean dimensions with unsorted keys and a
fixed SE2 block, a sparse 32-dimensional refresh within a 32-entry diagonal-factor
budget while dense refresh remains rejected, and callback failure during the
final QR rebuild. Failure preserves estimates, new-key absence, factor history,
cached information and the update interval; the next ordinary update recovers.
Existing SE3/batch agreement, threshold/periodic refresh, rank errors and old
callback non-replay tests remain passing.

Transactional updates still copy variable/anchor maps and QR. Full estimates are
returned, history grows, and rebuilds still relinearize everything. This is not
a Bayes tree, iSAM2, selective clique update, downdate, reorder or constant-cost
update implementation. Gaussian-only cached models and single-threaded callbacks
remain unchanged. Fixed-lag does not acquire an incremental backend here.

## Fixture, oracle and measurements

`benchmark_isam` generates a deterministic **linear Euclidean2** trajectory with
correlated SPD observation and odometry covariances. Every 25 updates adds a
fixed-root measurement of an already-existing half-history key. This changes
older estimates; because the root is fixed it adds a unary normal-equation
term, retaining a block-tridiagonal problem. This is a well-conditioned synthetic
family, not a general nonlinear SLAM dataset or GTSAM/iSAM2 comparison.

A separate 2x2 block-Thomas implementation with explicit cofactor inversion
computes each prefix's reference solution. It does not use ropt's QR, LDL or
whitening implementation. Every update checks exact root preservation, estimate
agreement <=1e-8, directly computed original gradient infinity norm <=1e-8 and
cost agreement <=`1e-9*max(1,reference_cost)`. Reported costs are checked too.
An additional pure-Python implementation verifies final means, energy and
gradient on original generated measurements. Tests compare that gradient with
central differences and verify oracle stationarity across delayed measurements.

The comparison has three candidates: the original incremental implementation,
the optimized incremental implementation, and current batch optimization after
every update. The batch candidate is `FixedLagSmoother` with **all timestamps
None**, so it retains all history and never marginalizes/expires variables.
It uses sparse LDL LM, explicit initial damping 1e-12 for these constant
Gaussian Jacobians, cost tolerance 1e-15 and gradient/step defaults 1e-10.
This avoids charging it for several default damping trials. It is not a
bounded-window fixed-lag timing comparison.

Incremental thresholds are explicitly 1e9 with interval `usize::MAX`; constant
Jacobians need no refresh in this linear fixture. Ordinary refresh policy and
nonlinear accuracy outside this family remain separate. QR/persistent-window
entry budgets are explicitly 32,000,000; sparse LDL retains its 1,000,000-entry
limit. This budget is not a memory measurement. At 1,000 updates QR stores
6,996 entries; full dimension is 2,000 plus the conditioned root.

Six rotated/reversed rounds put every candidate in each position twice. Each
process discards one full stream and measures one stream. Initialization,
fixture/recipe creation, prefix-oracle precomputation, validation, file output
and returned-report destruction are outside timers. Complete update calls,
including transactional copies, graph registration, factor evaluation,
factorization/back-substitution, state commit and report construction, are
timed. Callback counting is enabled identically; allocator counting is disabled.

Across 100/300/1,000 updates there are **54 timed streams and 25,200 timed
updates**, all passing every prefix gate; no outlier removal. The two incremental
candidates each invoke only 2,040 new-factor callbacks per 1,000-update stream;
batch invokes 2,043,080. The speed improvement does not come from loosening
accuracy, changing refreshes or replacing derivatives. Earlier exploratory
measurements before the unchanged-fixed-flag optimization stay separate in /tmp
and are not pooled with these final samples.

## Recorded results

Median sum of update-call times per complete stream, in milliseconds:

| Updates | Original incremental | Optimized incremental | Batch after every update |
| --- | ---: | ---: | ---: |
| 100 | 8.2080 | **4.3718** | 20.6264 |
| 300 | 87.6685 | **36.3512** | 190.2096 |
| 1,000 | 1,511.4448 | **465.5490** | 2,297.0947 |

At 1,000 updates this is about **3.25x** faster than the original incremental
implementation (69% lower aggregate time), and **4.93x** faster than configured
all-history batch updates. These are aggregate update times, not one update's
latency or end-to-end stream runtime. Across the six 1,000-update runs,
optimized update latencies have median 0.4557 ms, P95 0.9324 ms and maximum
4.2725 ms. Those quantiles combine growing history sizes and delayed-factor
events; they are not a fixed-size or real-time guarantee.

[100](results/isam_updates/frames100/report.json),
[300](results/isam_updates/frames300/report.json) and
[1,000](results/isam_updates/frames1000/report.json) reports retain raw stream
and per-update timings, prefix quality metrics, options, callback counts,
source/template/oracle/executable fingerprints and dispersion. The machine is
the same AMD EPYC 9V74 cloud instance, affinity CPU 0, single-thread environment
settings and default Rust release profile. No other compiler/solver overlaps
the final sweep. Frequency and cloud co-tenancy remain uncontrolled, and
peak/live/RSS memory is not measured.

## Reproduce

```sh
cargo build -p ropt --example benchmark_isam --release --locked
python3 benchmarks/make_ldl_baseline.py --before-isam --output /tmp/ropt-isam-before
benchmark_cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c "$benchmark_cpu" \
  python3 benchmarks/paired_isam.py \
  --before /tmp/ropt-isam-before/target/release/examples/benchmark_isam \
  --before-source /tmp/ropt-isam-before \
  --after target/release/examples/benchmark_isam \
  --frames 1000 --rounds 6 --output /tmp/ropt-isam-comparison
```

The baseline helper reverses `isam_graph_reuse.patch` in an isolated
benchmark-only snapshot, after reversing the later `isam_scheduled_refresh.patch`.
It first reverses the subsequent `isam_qr_merge.patch` as well; the newer
[native iSAM2 comparison](NATIVE_ISAM2.md) measures that QR change separately.
The scheduled-refresh fix changes forced/interval initialization, which this
linear fixture disables. The timings above remain the recorded graph-reuse
checkpoint; they have not been rerun or pooled with later code. Its baseline
kernel fingerprint matches the recorded predecessor
and its stream template matches the current executable. Other historical helper
stages reverse this patch first before the older objective/LDL patches. Output
paths must be new and outside the checkout. Do not overlap timing with compilers,
tests, diagnostics or other solver processes. CI uses an offline small-stream
smoke, not public inputs, external libraries or timing thresholds.
