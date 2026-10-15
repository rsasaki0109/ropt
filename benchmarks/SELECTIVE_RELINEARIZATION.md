# Threshold-selected nonlinear factor evaluation

`IsamSmoother::with_selective_relinearization(max_passes)` now retains
factor-owned Gaussian rows and reevaluates only factors incident on variables
whose local or raw Gaussian step exceeds the existing threshold. This is an
explicit opt-in before the first update. The default policy is unchanged.

The later [editable Bayes-tree mode](EDITABLE_BAYES_TREE.md),
`.with_selective_bayes_tree(max_passes)?`, replaces selected rows through cached
separator messages and affected Gaussian cliques. This document describes the
earlier global-QR selected mode and its frozen validation.

The implementation is a foundation for nonlinear incremental inference:
**Gaussian QR reconstruction and back-substitution remain global**. It does
not yet remove/replace factor rows through only affected Bayes-tree cliques,
and does not claim equal-accuracy speed gains over GTSAM.

## Per-variable charts and factor rows

Each variable has an anchor; all cached factors use the product of these
per-variable charts. An ordinary append evaluates new factors at those anchors
and solves the cached Gaussian model. On a threshold-selected pass:

1. Move only selected nonfixed anchors to the current cached estimates.
2. Reevaluate every factor incident on at least one selected variable, including
   its unselected neighbors at their unchanged anchors.
3. Replace those complete residual/Jacobian row blocks. Share unrelated blocks
   through immutable `Arc`s; preserve original insertion-order tangent columns
   and row order, including fixed-variable constant rows.
4. Assemble canonical CSR and rebuild undamped sparse QR with the previous
   permutation. Solve and retract through the mixed per-variable anchors.
5. Repeat only while local or raw steps exceed the threshold, up to the supplied
   pass cap (1–200). Report any remaining threshold excess explicitly.

All blocks contain the existing whitened residual and tangent Jacobian; retained
blocks are not whitened again. Both analytic and chunked forward-AD factors work.
As before, this iSAM backend accepts squared Gaussian factors only. It never
forms a numeric normal matrix, invokes implicit numeric differentiation, or adds
damping to repair a rank-deficient selected model.

Checking raw steps as well as principal local coordinates prevents a winding
rotation from hiding a large chart update. A cleared threshold limits anchor
movement; **it is not a nonlinear stationarity certificate**. Selected passes
are bounded undamped Gaussian steps, with no true nonlinear cost acceptance
test. Applications needing a full nonlinear solve can explicitly request one
and inspect the configured stopping criteria and resulting gradient.

Forced and interval refreshes keep their existing full batch solver, ordering,
damping-reuse and optional refinement behavior. The selected cache stores that
accepted Gaussian model at its actual anchor, before optional frozen-QR
refinement changes the estimates. With `with_bayes_tree`, an ordinary row append
can still partially re-eliminate the tree; a selected row replacement rebuilds
the full tree and reports zero reused cliques.

Cache residual and nonzero coefficient counts together are limited by
`qr.max_entries`, separately from the existing numeric/symbolic QR budgets.
This is not a byte or peak-memory measurement. Any evaluation, fill, finite-value,
rank or tree-conversion error leaves owned variables, anchors, factors, Gaussian
rows, damping and update counters uncommitted. Callback side effects remain the
caller's responsibility, as documented for all smoother factor recipes.

## Reports and example

`IsamReport.selective_update` is `Some` only for threshold-selected passes.
It records pass count, distinct moved variable keys, summed selected factor
linearizations, retained factors and `threshold_remaining`. Counts exclude
initial new-factor evaluation and do not count separate AD chunk callbacks.
`factors_linearized` counts distinct new/selected recipes across the update;
`batch_trials` is zero. `kind` remains `Relinearized` and `reason` remains
`Threshold`; the optional report distinguishes this path from a full batch.

```rust
use ropt::isam::{IsamOptions, IsamSmoother};

let smoother = IsamSmoother::new(IsamOptions::default())?
    .with_selective_relinearization(4)?;
```

Run the complete example:

```sh
cargo run -p ropt --example selective_isam --locked --offline
```

It installs 1,024 independent scalar quadratic observations, then adds a new
measurement on one variable. The actual selected update moves key 0, retains
**1,023 of 1,025 factors**, uses two passes and makes five factor callbacks
(one new evaluation plus four selected evaluations). Its nonlinear gradient is
approximately `2.064e-2` even though the threshold is cleared. The following
explicit full refresh reduces it to approximately `1.666e-8`. This synthetic
example demonstrates row retention and the policy distinction; it is not a
native comparison or an equal-accuracy timing benchmark.

## Validation and historical results

Eight new integration tests cover independent scalar mixed-model solutions,
correlated original-information energy, unchanged neighbor anchors, callback
selection, bounded-pass reporting, retry after selected callback failure,
cache-budget failure, late activation, heterogeneous column extension, fixed
constant energy, AD, raw SO2 winding, and full forced/interval recovery.
Both Lie-exponential and Cartesian SE2 selected updates also match independently
formed central-difference Jacobians and dense SVD solutions at a nonidentity
mixed chart, including exact fixed-root conditioning.

The complete suite passes **279 Rust unit/integration tests plus 18 doctests**
(297 total). Clippy and rustdoc deny warnings; formatting passes. Existing
Python benchmark tests pass 20 cases. Complete forced-refresh regression on
CSAIL (1,044 arrivals) and MIT1228 (1,227 arrivals) produces **byte-identical
full poses and identical non-timing update fields** to the archived predecessor.
These regressions are functional checks, not new speed measurements.
Logs, source hashes and
[checks](results/selective_relinearization/validation_checks.json) are archived.

The [MIT1228](PRECISE_REAL_ISAM2.md) and [CSAIL](CSAIL_REAL_ISAM2.md) comparison
results and gallery inputs remain frozen at their measured predecessor kernel.
Source-only rollback verifies all **31** historical fingerprints without patch
fuzz. Older binaries are not rebuilt during these source checks. Reconstruct the
kernel immediately before this feature with:

```sh
python3 benchmarks/make_ldl_baseline.py --before-selective-relinearization \
  --source-only --output /tmp/ropt-before-selective
```

The production source fingerprint restores to
`31c01bc01ed53679b44fccb92bdaf46982010b8a2895b6bf926b84a42b350047`.
Next work is selective factor replacement through cached separator messages,
followed by equal-accuracy nonlinear incremental measurements. The informal
maturity estimate remains **75%**.
