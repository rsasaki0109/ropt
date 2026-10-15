# Numeric factors preserve the first detected output error

Numeric factor evaluation previously called both perturbation callbacks before
validating the positive sample. A wrong-sized or nonfinite positive result could
therefore be hidden by an unrelated error from the negative callback. Similarly,
a derivative column that overflowed could be hidden by a callback error in a
later column, because block validation happened after all columns were assembled.

`FactorGraph` now validates each sample immediately and checks each completed
derivative column before advancing. Invalid sample dimensions return
`InvalidDimensions`; nonfinite samples or derivative columns return `NonFinite`.
Subsequent callbacks are skipped after these errors. Errors returned directly
by callbacks continue to propagate through the existing `Result` path.

The shared factor visitor applies this behavior to both dense and sparse graph
evaluation. Whitening still happens after differentiation, and numeric factors
still use their explicitly requested tangent-coordinate central differences.
The retraction, step rule and derivative formula are unchanged.

Two regression tests fail before the change and pass afterward. Across dense
and sparse evaluation they inject eight failure cases: a wrong-sized, NaN or
Inf positive sample followed by a distinct negative-callback error, and a
derivative overflow followed by a distinct later-column error. Callback counts
verify that evaluation stops at the first detected invalid output.

The tests then disable the injected fault and reuse the same graph. They check
the unchanged initial state, valid residuals and Jacobians, sparse products
against an independent analytic matrix, and successful dense/SparseQR solves
with known solutions. Debug and release builds exercise both tests. This is
an error-reporting and evaluation-order change; no runtime advantage is claimed.

The [evidence manifest](results/numeric_factor_errors/manifest.json) records
source fingerprints, logs and hashes. To reconstruct the previous implementation:

```sh
python3 benchmarks/make_ldl_baseline.py --source-only \
  --before-numeric-factor-errors --output /tmp/ropt-before-numeric-factor-errors
```

All 64 source-only historical reconstruction stages preserve their expected
fingerprints. This does not claim compilation of every historical stage.
