# Preserve tiny robust weights through their square roots

IRLS scales residuals and Jacobian rows by the square root of a loss weight.
Previously, the shared robust model first evaluated the raw weight and then
took its square root. A raw weight around `1e-400` rounded to zero even though
its square root, `1e-200`, was representable. This could erase an informative
row, report gradient convergence before a trial, or make frozen information
rank deficient. Subnormal raw weights could also lose significant precision:
one covariance regression returned `1.0000111329412582` instead of about `1`.

The implementation retains the previous square-root calculation for normal
weights. For subnormal or zero weights, Huber uses `sqrt(a) / sqrt(abs(r))`
outside its quadratic region. Cauchy uses
`(a / abs(r)) / sqrt(1 + (a / abs(r))²)` when `abs(r) > a`. These equivalent
forms avoid materializing a tiny raw weight. Cost and true-loss reduction
formulas are unchanged.

Nonlinear solves, frozen-IRLS covariance and frozen-IRLS fixed-lag
marginalization use the same square-root weights. No public API changes.

Six new public-API regressions cover:

- Dense and sparse-input covariance at ordinary, raw-subnormal and
  raw-underflow scales for Huber and Cauchy. They also check representable
  subnormal square roots. With `r = J = 1e200`, Cauchy scale `1` or Huber
  threshold `1e-200` gives weighted information about `1`, so covariance is
  about `1`.
- LM steps for both losses with those underflowing raw weights. A unit initial
  damping gives a step about `-0.5`; the true reductions are `ln(2)` and `0.5`.
- Rejection of deliberately residual-increasing trials, preserving the
  accepted estimate and cost.
- Fixed-lag covariance and marginalization at an explicitly frozen point.
  A robust anchor and a unit Gaussian between factor give information
  `[[2,-1],[-1,1]]`, covariance `[[1,1],[1,2]]`, and retained separator
  information `1/2` after eliminating the old variable. Both inherited scalar
  loss and explicit joint loss recover these references; the carried Gaussian
  prior retains separator variance `2`.
- Continued zero frozen rank when even the square-root weight is below the
  representable range.

Five tests fail before the fix; the last range-limit control already passes.
All six run in debug and release. LM checks exercise dense, dense QR, sparse
LDL and sparse QR backends. Comparisons use relative tolerances appropriate
to the resulting values; the extreme Huber case whose square root is itself
subnormal allows its binary64 rounding error.

This does not add extended precision or log-weight arithmetic. Square-root
weights, weighted quantities, objective values and subsequent linear algebra
retain their own `f64` limits. Frozen-IRLS covariance and marginalization
remain conditional on the explicitly selected frozen information model.
The fixed-lag regression freezes estimates to inspect information; it does
not assert nonlinear convergence or equivalence to an all-history solve.
No timing or comparative performance claim is made.

The [evidence manifest](results/robust_root_weights/manifest.json) records
source fingerprints, logs and artifact hashes. Reconstruct the previous source:

```sh
python3 benchmarks/make_ldl_baseline.py --source-only \
  --before-root-weights --output /tmp/ropt-before-root-weights
```

All 68 source-only historical reconstruction stages retain their expected
fingerprints. This does not claim compilation of every historical stage.
