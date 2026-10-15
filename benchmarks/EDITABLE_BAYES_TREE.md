# Nonlinear selected factors with editable Gaussian Bayes trees

`IsamSmoother::with_selective_bayes_tree(max_passes)` now connects
threshold-selected nonlinear factor evaluation to **affected Gaussian clique
replacement**. It preserves unrelated original factor rows, conditionals and
square-root separator messages. This extends the earlier
[global-QR selected-factor mode](SELECTIVE_RELINEARIZATION.md).

In the complete synthetic nonlinear star example, **2 of 1,025 scalar cliques**
are re-eliminated, **1,023** are retained, and **2,047 of 2,050 nonlinear factors**
are not reevaluated. These are work counts, not latency or peak-memory results.
Symbolic analysis, bookkeeping, rank checks and back-substitution remain global.
No native GTSAM speed claim is made for this new policy.

The subsequent [stationarity-controlled comparison](STATIONARITY_CONTROL.md)
validates all recorded-data arrivals, adds requested-key/probe APIs and improves
symbolic reuse. It records the editable policy's negative latency result against
current full ropt, while selected global QR is faster at the same accuracy.
This document retains the earlier implementation/validation snapshot.
The later [canonical local-clique CSR comparison](LOCAL_CLIQUE_CSR.md) measures
a further 16.0% / 11.2% editable-kernel improvement with identical means/work;
the improved policy still trails current full refresh.

## Original factors and boundary messages

An already triangularized R does not identify how an individual observation
contributed to its ancestors. Removing its conditional row alone would also
remove information supplied by unaffected children. The new
`bayes_tree::EditableBayesTree` retains original Gaussian factor blocks by ID
and caches the marginalized square-root message from each scalar clique.

For a clique, gather its owned original observation rows and direct child
messages, then use the existing sparse Givens QR in frontal/separator order.
The first row is its conditional; remaining triangular rows and marginalized
constant energy form its separator message. No numeric normal matrix, signed
information subtraction, hyperbolic QR downdate or SVD fallback is used.
The global objective is the sum of all conditional quadratics, **current root
message constants**, and original purely constant factors. Adding every child
constant again, or replaying unaffected subtree observations, would double-count.

Ownership uses the earliest internal column in the whole factor block's support.
Whole-block support is conservative when rows are disconnected. Filled symbolic
separators enforce running intersection independently of numeric cancellation
in R. A nonempty update globally recomputes ownership/topology and marks changed
old/new owners, changed separators, and ancestors in **both old and new trees**.
Affected cliques are re-eliminated in increasing fixed order. Unaffected direct
children contribute their cached messages without replaying their observations.
Unchanged conditionals, messages and original factor blocks remain shared Arcs.

An affected root can still gather many child messages: the star example gathers
1,024 separator rows. Two re-eliminated cliques do not imply constant-time update.
There is one scalar frontal per clique; supernodes, dynamic reordering and
wildfire back-substitution are not implemented.

## Public Gaussian API

`GaussianFactor::new(jacobian, residual)` defines
`0.5 * ||J*x + residual||²`, with original parameter columns. `EditableBayesTree`
accepts fixed identity or explicit internal-to-original ordering, and can extend
with unconstrained trailing original columns. Its atomic `update(removed,
replacements)` removes existing IDs and inserts/replaces complete factor blocks.
Lists must have unique, disjoint IDs; replacing a block can change its support,
row count, frontal owner or constant-only status. Replacement matrices match
the current dimension; retained older blocks can precede column extension.

The Gaussian backend can install incomplete models; `solve()` explicitly checks
observability with the existing global relative unpivoted-R diagonal cutoff.
This is not a rank-revealing SVD certificate. Dimensions, finite inputs, overflow
and fill failures leave owned state unchanged. `cost()` preserves the complete
original quadratic, including constants, even for rank-deficient models.

`max_entries` separately bounds original coefficients plus residuals, message
coefficients plus RHS, numeric conditionals, symbolic entries, and each gathered
local coefficient/residual workspace. Transient updates can exhaust a budget.
These are entry counts; memory is not measured. The shared nonlinear factor
cache and editable tree reference the same immutable Gaussian blocks.

## Nonlinear policy and reports

Enable `.with_selective_bayes_tree(4)?` before the first `IsamSmoother` update.
It selects the editable backend instead of the append-only tree. Newly appended
factors are installed by stable history index. On a threshold pass, selected
anchors move and all incident analytic/AD factors are reevaluated in the same
mixed per-variable chart used by the global-QR mode. Their original Gaussian
blocks are replaced through cached boundary messages before the next solve.

`report.selective_update` retains pass/key/factor counts and threshold exhaustion;
`report.editable_bayes_tree_update` records scalar re-eliminated/reused cliques,
gathered original/separator rows and reused boundary messages for the **last
installed Gaussian update**. Counts are not summed across preliminary appends or
multiple selected passes. The append-only `bayes_tree_update` is None in this mode.

The threshold checks both local and raw steps, preserving winding detection.
Passes are bounded and undamped, without a true nonlinear step acceptance test.
**Threshold clearance is not nonlinear stationarity.** Forced and interval
refreshes retain the full batch solver and rebuild the complete editable tree
from its accepted original Gaussian blocks. The cache anchor is preserved through
optional frozen-QR refinement. All updates remain transactional, including rank
failure after Gaussian replacement; callback side effects cannot be rolled back.

The all-history smoother still accepts Gaussian squared-loss factors only and
does not expose factor deletion or variable retirement. Standalone Gaussian
deletion works; incremental fixed-lag removal remains a separate next step.

## Complete runnable example

```sh
cargo run -p ropt --example selective_bayes_tree --locked --offline
```

The fixture has 1,024 leaf variables with quadratic observations, relative
leaf/root observations and a strongly observed free root. A new observation
changes one leaf. Actual output:

```text
selected keys: [0]; nonlinear factors retained: 2047; passes: 1; callbacks: 5
Gaussian cliques: 2 re-eliminated, 1023 reused; 1023 boundary messages reused
cached Gaussian cost: 1.7168021312; original nonlinear cost: 1.7205319284; original gradient inf: 7.466e-2
threshold remaining: false; symbolic analysis/back-substitution remain global
```

The two-variable AD factor uses two one-column chunks; five callbacks represent
one initial new evaluation and three selected factor linearizations. The example
shows the surviving nonlinear gradient explicitly and makes no equal-accuracy
timing comparison. [Captured output](results/editable_bayes_tree/example.log).

## Validation and reproducibility

Six new Gaussian integration tests compare independent dense SVD solutions,
original gradients and **complete quadratic costs at arbitrary parameter values**.
They cover informative star boundaries, multiple separators, support/owner
changes, constants, combined replacement/deletion, permutation, column extension,
rank loss/recovery, invalid IDs/dimensions, NaN, overflow, budgets and 48 successive
varied sparse updates. A unit test checks actual Arc identity of unchanged
conditionals, nonzero separator messages and original factors, and independently
checks the separator's marginalized energy.

Three nonlinear tests compare editable and global-QR selected estimates/costs,
verify callback counts and retained branches, recover after partial callback
failure and interval rebuild, and preserve committed history after selected rank
loss. The earlier central-difference mixed-chart SE2 test now runs both backends
with Lie-exponential and Cartesian charts, checking fixed-root conditioning.

The complete suite passes **289 Rust unit/integration tests plus 18 doctests**
(307 total), with 20 Python checks, formatting, Clippy and rustdoc warnings denied.
Default-policy complete CSAIL (1,044 arrivals) and MIT1228 (1,227 arrivals) replays
retain **byte-identical full poses and identical non-timing reports**. These cold
checks are functional regressions, not new speed samples. Logs and hashes are
saved in [validation checks](results/editable_bayes_tree/validation_checks.json).

The preceding real-data comparisons and figures remain frozen at their measured
kernel. All **32 source-only rollback fingerprints** match without patch fuzz;
older binaries are not rebuilt. Restore the preceding global-QR selected kernel:

```sh
python3 benchmarks/make_ldl_baseline.py --before-editable-bayes-tree \
  --source-only --output /tmp/ropt-before-editable
```

It restores source fingerprint
`7b04bda5bcfc14549c7fa0361da1994aece1c2091087118f2dfc09c927e3d669`.
The next validation target is this nonlinear policy on recorded streams with
unchanged common-accuracy gates and measured update latency. Informal maturity
remains **75%** pending that evidence and field validation.
