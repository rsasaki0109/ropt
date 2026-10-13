# Streaming estimation: fixed lag and initial iSAM-style updates

Two distinct implementations now share application keys, variable recipes and
analytic/forward-AD factor recipes:

| Contract | `FixedLagSmoother` | `IsamSmoother` |
| --- | --- | --- |
| Retained state | Timestamp window plus persistent variables | Entire history |
| Ordinary optimization | Batch LM on current window | Insert new Gaussian rows into cached sparse QR, then back-substitute |
| Nonlinear refresh | Every update for live nonlinear factors | Tangent threshold, interval or explicit request triggers full batch LM and QR rebuild |
| Expiry | Dense square-root marginalization of incident factors | No row/factor deletion |
| Loss | Squared or explicit frozen-IRLS marginalization | Gaussian only in this first version |
| Shared limitation | No real-time guarantee; single-threaded callbacks | No Bayes tree, iSAM2, selective clique updates or reordering |

Fixed lag currently does **not** use the incremental QR backend. These are working
initial implementations, not a claim of equivalence to mature GTSAM smoothers.

Two [normal-pattern experiments](../../benchmarks/NORMAL_PATTERN_EXPERIMENTS.md)
show no complete-update improvement at 300 arrivals and are not adopted. Their
raw comparisons are retained, with independent normal-structure/budget tests.

[Aligned QR reductions](../../benchmarks/QR_REDUCTIONS.md) preserve each column's
floating operations while grouping finite validity and exact-zero counts.
Invalid suffixes are rejected before compaction or further rotation/RHS updates;
public append and smoother transactions retain their committed state. Full-refresh
reports and cached traces match the predecessor exactly in the retained comparisons.

[Contiguous QR column matching](../../benchmarks/QR_COLUMNS.md) uses exact proofs
based on sorted ranges. Gapped supports still compare every interior column. No numeric
operation or rank/fill policy changes; cached traces remain byte-identical.

[Shared iSAM local coordinates](../../benchmarks/ANCHOR_COORDINATES.md) evaluate
each active block once for the cached refresh threshold, report norm and model
cost. A refresh recomputes them at its new anchors. Raw-step norms still trigger
refresh when a rotation winds beyond the principal chart. The separate cached
timing policy retains its independent nonlinear checks; approximate cached
prefixes cannot support a native speed comparison at the forced-refresh target.


[Grouped full-model QR rebuilds](../../benchmarks/QR_REBUILD.md) reorder Gaussian
rows by their leading column, preserving stable ties and the same Givens kernel.
Public incremental append remains in input order. Reordered cache coefficients
and costs can round differently, while every full-refresh prefix passes the
unchanged original-gradient/native-pose gates. The synthetic common-accuracy
100/300 benchmarks now beat the verified native configuration; independent
300 confirmation and cached AD/dense-QR validation are retained. This does not
establish default-policy, field-data or Bayes-tree equivalence.

## Fixed-lag data and information contracts

`update(now, SmootherUpdate)` starts from the committed estimates, registers new
variables/factors, optimizes, then removes keys with `timestamp < now-lag`.
Equality at the boundary retains the key. `lag` must be finite and positive;
`now` must be finite and nondecreasing. New timed variables must lie in the
window and cannot be dated in the future. Factors may refer to committed or
newly supplied keys. Once a key is removed, a late factor referencing it is an
error; historical variables cannot be resurrected from the prior.

`timestamp: None` creates a persistent variable, useful for shared calibration.
Timestamp refreshes are explicit and cannot move backwards or beyond `now`.
Keys must be unique among live variables, and retired key values may be reused.
Fixed variables are conditioned constants; their information can transfer to
live neighbors when their timestamps expire. Fixed flags/values are set at
insertion and have no later mutation API in these smoothers. Initial values
must be valid for their manifold, including unit quaternions.

Only factors touching expired variables are linearized into the replacement
prior. Other factors remain nonlinear. All incident factors are then removed,
including a previous marginal prior if it touches an expired key. This prevents
double-counting evidence. The separator is the union of active live neighbors;
cross-variable information is preserved, rather than replacing it with diagonal
priors. Factors that reference fixed neighbors are conditioned on those values.

For whitened tangent residuals `r + Je*de + Js*ds`, Householder QR of `Je`
projects the residual/retained Jacobian onto the orthogonal complement of its
column space. QR of the resulting **augmented** matrix `[Js,r]` compresses the
prior to at most `separator_tangent+1` rows. That extra row can carry irreducible
constant cost. No normal-matrix Schur subtraction or inverse is formed. Scaled
SVD verifies full rank of eliminated columns, with a bounded decomposition;
unobservable elimination returns `RankDeficient`, with no hidden damping or
pseudoinverse. The retained prior may be rank deficient; no gauge constraint is
invented. An isolated expired variable with no incident factor simply disappears.

The prior evaluates `A*Local(anchor,current)+b`. Its matrix and manifold anchors
are frozen at marginalization. Forward AD differentiates Local with respect to
right retractions on Euclidean, SO2, SO3, SE2 and SE3 variables, including the
nonidentity chart Jacobian away from the anchor. Principal rotations have the
usual discontinuity at pi. Original marginalized nonlinear factors are gone,
so this does not recover the exact nonlinear all-history objective after large
state changes. No consistency/coverage guarantee, FEJ policy or gauge-invariant
VIO treatment is claimed.

`MarginalizationLoss::Gaussian` rejects robust factors entering elimination.
`FrozenIrls` explicitly freezes their joint/scalar weights after whitening.
The prior always overrides the global loss with squared loss, so those weights
are not applied twice. The difference between robust cost and weighted squared
cost at the linearization point is retained as a scalar offset. Constants from
completely eliminated components are retained similarly. Report costs include
these offsets and refer to the optimized window **before that update's expiry**;
after nonlinear marginalization they describe a surrogate, not raw all-history
residuals. Iteration costs remain the solver's window costs without those scalar
offsets. Selected covariance is for the live surrogate model, and requires its
own rank/robust-model checks.

Default optimization is dense augmented-QR LM. `max_entries` defaults to
1,000,000 and bounds each input `[J,r]` for marginalization. The smoother also
checks `(residuals+tangent)*(tangent+1)` before window optimization. These are
entry limits, not peak bytes/RSS limits; temporary QR/SVD matrices, factor
callbacks and container overhead also use memory. A fixed time span does not
bound observation rate, persistent variables or separator fill. Updates need a
nonempty optimization problem; after expiring an entire graph, the next update
must supply a new constrained problem.

## iSAM-style square-root contracts

`IncrementalQr` stores sparse upper-triangular rows and the transformed RHS.
Appending a new CSR Jacobian/residual applies Givens rotations against existing
rows. It neither reconstructs `JᵀJ` nor reevaluates/replays old observations.
Appending columns preserves existing tangent indices. Both the triangular
information and irreducible residual cost are retained. Back-substitution
requires positive resolved diagonal magnitudes using an unpivoted relative
cutoff, default 1e-12. This is a diagonal heuristic, not an SVD rank certificate;
poor scaling/conditioning and ordering still matter. No damping, rank truncation,
alternate solver or inverse is used for the cached solve.

[Bounded QR normalization](../../benchmarks/QR_NORM.md) uses `sqrt(a*a+b*b)` when
the largest input magnitude is within `[2^-450,2^450]`, retaining `hypot` outside.
It changes IEEE rounding; reported QR costs and cached traces need not be bit-
identical. Finite checks, rank/fill policies and original-energy/reference accuracy
gates remain unchanged. The norm tests cover random significands across exponents,
boundary-adjacent values, subnormals, overflow, signed zero, NaN and infinity.

`IsamSmoother::update(update, force_relinearize)` linearizes new Gaussian factors
at stored manifold anchors and adds their rows to the cached QR factor. The
estimate is `Retract(anchor,delta)` from the full triangular solve, so a late loop
closure can change existing estimates. A raw/Local tangent block norm above the
default 0.1 threshold triggers batch LM from that provisional state. Every tenth
update or `force_relinearize=true` instead goes directly to batch LM from
committed estimates and new seeds, without evaluating/solving at obsolete
anchors. An undamped QR rebuild at the optimized state still checks rank/fill.
The initialization update counts toward that interval. `linearized_cost`
is the cached Gaussian model's cost at the returned estimate; it is **not** an
independent nonlinear cost/stationarity check between refreshes.

`relinearization_reason` is None for cached updates and Forced/Interval/Threshold
for rebuilds, with explicit requests taking precedence over the interval. A
scheduled rebuild can recover a new factor that is singular at a stale anchor
but observable at the current estimate. A genuinely unobservable final model
still fails and rolls back. Current-estimate initialization changes the nonlinear
starting point relative to the earlier provisional-cache path and does not
guarantee the same local minimum for arbitrary problems. Error diagnostics retain
batch termination, trial count, cost and last trial's gradient.

The current refresh relinearizes the entire graph. It does not use a Bayes tree,
selective clique relinearization, downdates, variable/factor removal or reorder
the triangular factor. Full history is retained for later refreshes. New
variables append tangent columns; fixed variables add no columns. Timestamps on
new variables are ignored, and timestamp refresh requests are rejected. All
noise models remain supported, but robust factors and a robust global fallback
are rejected. The model must be observable even when the refresh used LM
damping. The stored-entry limit bounds R fill; a dense full refresh also checks
`(residuals+tangent)*tangent` against it before invoking the batch solver. These
limits do not measure or bound overall peak memory.

The [subsequent incremental validation](../../benchmarks/ISAM_UPDATES.md)
removes redundant graph/history construction and makes the full-refresh matrix
guard specific to dense backends. Sparse selections use their backend limits
and the QR fill limit; the default dense-QR refresh retains its existing guard.
Neither policy is a total-memory limit. Ordinary updates now register only new
recipes in one graph, reuse its complete variable layout, and commit new recipes
only after all checks succeed. Refresh builds one additional all-factor graph
and consumes the solver's accepted whitened Gaussian model to rebuild QR,
without replaying all callbacks at the optimized state. The internal handoff
accepts squared losses only; public solver behavior is unchanged. Factor inputs
reuse local per-slot storage, including varying arity/ambient dimensions. See
the [model/input reuse measurements](../../benchmarks/MODEL_REUSE.md).

The default restarts each LM refresh at its configured initial damping.
`IsamSmoother::with_batch_damping_reuse()` explicitly carries the adapted damping
between successful full refreshes. It requires LM with finite positive initial
damping, keeps the first configured value, and changes no acceptance or
termination tolerances. Cached updates and failed refreshes preserve the carried
state, including failures in final undamped QR checks after batch convergence.
`next_batch_damping()` exposes the committed next-refresh value (`None` for
default restart); `IsamReport::batch_trials` counts batch trials, while
`batch_initial_damping`/`batch_final_damping` expose LM values or `None` for
cached/Gauss-Newton updates. The final value is the adapted next-trial proposal.
The [policy comparison](../../benchmarks/DAMPING_REUSE.md) retains identical
accuracy gates and separate raw timing records. Reuse still performs full batch
refreshes and QR rebuilds; it is not a selective incremental algorithm.

The subsequent [compact symbolic preparation](../../benchmarks/SYMBOLIC_PATTERN.md)
reduces structural allocation traffic in sparse LDL. It preserves ordering,
numeric arithmetic, damping policies and entry-limit checks; the matched
same-policy stream comparison requires identical estimates and batch reports.

The [QR processing improvement](../../benchmarks/QR_MERGE_BUFFERS.md) reuses input
row storage, simplifies sorted merges and avoids R traversal at exactly zero
delta. It preserves ascending arithmetic, transient entry limits and failure
rollback. Nonzero vectors use the full cost path even when their norms underflow.

[Direct CSR assembly](../../benchmarks/CANONICAL_CSR.md) emits graph rows and
accepted dense Gaussian models without intermediate triplet buffers. Callback
and whitening order stay unchanged; only active assembly slots follow tangent
column order. A checked internal handoff preserves bounds, canonical rows and
finite-value validation, including fixed-layout and exact-zero pattern changes.

[Slice-based graph retraction](../../benchmarks/GRAPH_RETRACTION.md) removes owned
per-variable value/step/result buffers. Input validation follows variable order,
including fixed/disconnected blocks; active increments and updated ambient values
retain finite/unit-quaternion checks. Existing right Lie operations and fixed
values are preserved. Caller inputs remain unchanged on failure.

[In-place QR updates](../../benchmarks/QR_INPLACE.md) rotate/compact existing upper
columns and append trailing fill directly. New interior columns use the original
sorted merge after copying the processed prefix. Ascending arithmetic, exact-zero
removal and transient fill checks remain unchanged. Candidate-based commits
preserve public append/update rollback, including overflow after row writes.

[Aligned QR row reuse](../../benchmarks/QR_ALIGNED.md) compares every live suffix
column before rotating both rows directly. A cursor consumes the lower diagonal;
only exact-zero cancellations compact storage. Different patterns use the sorted
merge on that live suffix. Arithmetic, rank/fill checks and transactional commits
remain unchanged, including overflow after cursor advance.

[Matrix-whitening buffer reuse](../../benchmarks/WHITENING_BUFFERS.md) keeps scratch
local to one graph evaluation and swaps complete GEMM results into owned outputs.
`beta=0` ignores old scratch values; changed shapes are resized. Correlations,
callback/block order, finite checks and robust-loss handling remain unchanged.
No workspace is stored in noise models or retained across evaluations.

[Compact ordering adjacency](../../benchmarks/ORDERING_ADJACENCY.md) shares sorted
hybrid storage with symbolic patterns and reuses a local neighbor list. Greedy
degree/key ties, ascending updates and cumulative fill limits remain unchanged;
eliminating vertices does not refund that budget. Natural ordering is unchanged.

[Normal-graph rebinding](../../benchmarks/NORMAL_PATTERN_REBIND.md) checks every
original-coordinate normal edge after a CSR descriptor change. Equal graphs reuse
ordering/fill and refresh assembly targets/descriptors; different graphs prepare
afresh. Standalone symbolic matching stays exact, `analyses()` still counts all
successful pattern analyses, and numerical refactoring/residual checks always run.

[Sorted symbolic fill and CSR-row reuse](../../benchmarks/SYMBOLIC_MERGE.md) uses
an ascending insertion cursor with unchanged promotion/fill budgets. Consecutive
exactly equal row supports share structural work and copy assembly destinations;
all numeric rows, products and per-row operation budgets remain present in the
same order. Empty boundaries and different column indices are checked explicitly.

[Schur update destination cursors](../../benchmarks/UPDATE_CURSOR.md) start at the
known diagonal and reuse the next stored position for consecutive targets. A gap
searches only the remaining column slice. Cached destinations, arithmetic order,
entry/plan budgets and numeric readiness remain unchanged; gapped structures and
all ordering modes are compared against the previous binary-search plan.

`IsamReport::factors_linearized` counts recipes installed into the final QR
update (new recipes on incremental updates, all recipes on a rebuild). It does
not count batch optimizer callbacks or new rows inserted provisionally before
being replaced by a full rebuild. The tests separately instrument actual
callbacks to prove old factors are not evaluated on ordinary incremental updates.

R now uses sorted nonzero row vectors with reusable merge buffers, retaining
ascending Givens arithmetic, exact-zero removal and stored-entry accounting.
The [native iSAM2 comparison](../../benchmarks/NATIVE_ISAM2.md) checks identical
input bytes, original-energy stationarity and full mean agreement after every
arrival. Its explicit forced-refresh policy differs from default cached updates.

The [nonlinear stream validation](../../benchmarks/NONLINEAR_ISAM.md) measures
actual original-energy gradients between rebuilds and checks analytic/sparse
refreshes against AD/dense-QR references at twelve loop checkpoints through
300 added poses. It uses explicit solver tolerances and forces loop refreshes;
it is not a default-policy, native iSAM2 or performance comparison.

## Failure and validation

Both smoothers prepare candidate state and commit only on success. Invalid
connectivity/values, nonfinite arithmetic, rank/fill/matrix limits and failed
optimization leave owned estimates, factors, cache and time unchanged. An
iteration limit or `NoProgress` from full optimization is an error, not a
successful smoother update. Factor recipes use shared single-threaded callbacks;
they must be deterministic. External callback side effects are not rolled back.

Independent tests verify:

- QR elimination against SVD conditional least-squares objectives for multiple
  retained deltas, preserving constants, rank-deficient retained information,
  complete elimination and very large/small scales.
- Sixteen linear stream updates against independent all-history batch Cholesky
  estimates, covariance and objective cost; the live graph stays at most three
  variables and six factors. A coupled separator checks off-diagonal covariance
  and its response to later observations.
- Expiring a fixed anchor while retaining its information; persistent keys,
  strict cutoff equality, timestamp refresh, unknown/late keys and transactional
  failures. An unanchored graph remains unobservable after elimination.
- SE2/SE3 chain smoothing and tangent covariance, plus Local/AD/central-difference
  agreement across all five variable kinds, including quaternion sign invariance.
- Explicit robust frozen-IRLS handling and rejection of implicit robust
  marginalization, without applying the global robust loss to the prior again.
- Incremental Givens factors against independent batch SVD solution/objective,
  appended columns, fill limits, arithmetic errors and atomic failed insertions.
- Incremental stream estimates versus batch, with actual callback counters
  proving old factors are never reevaluated on ordinary updates; threshold,
  periodic and explicit nonlinear refresh; SE3 refresh versus independent batch;
  and rollback after failed nonlinear optimization.

## Reproduce synthetic examples

```sh
cargo test -p ropt --test marginalization --test local_coordinates --test fixed_lag --test incremental_qr --test isam --locked
cargo run -p ropt --example fixed_lag_odometry --locked
cargo run -p ropt --example isam_pose_graph --locked
```

These examples require no external dataset/dependency and run in offline CI.
The fixed-lag example has 80 one-second SE2 frames, a five-second lag, noisy
odometry, noisy Cartesian position observations and a conditioned first pose.
The observed maximum live state/factor counts are **6/12**, maximum endpoint
position error is 0.100063 m, and the final error is 0.063063 m. Final right-tangent
standard deviations are about `[0.06249,0.07052,0.04874]`; those model scales
do not validate statistical coverage. Data/noise are deterministic synthetic
signals, not real GNSS/SLAM measurements.

The iSAM example adds 40 SE2 poses and four loop closures to old key 0. It records
**36 incremental updates and 4 automatic full refreshes**, then explicitly
refreshes once. The final endpoint error is 0.0000362 m, with 41 live variables
and 45 factors. Its smaller error reflects different observations and constraints;
it is not an accuracy comparison against fixed lag. No execution-time superiority,
real-time suitability, iSAM2 equivalence or general robotics accuracy is claimed.

The [recorded example outputs](streaming_results) include sampled CSV rows,
the printed all-frame diagnostics, source/executable fingerprints and settings.
They are functional validation records, not timing or peak-memory benchmarks.

Next work includes Bayes-tree/selective updates, reordering, robust incremental
weight changes, an incremental fixed-lag backend, sparse marginalization,
complete real streams and matched external smoother benchmarks.
