# Numeric factor step representability

Numeric factor registration now requires both `step` and `2*step` to be finite
and positive. A denominator that overflows is rejected with
`Error::InvalidOptions` before adding a factor or changing residual counts.
The default and options-bearing registration methods share this validation.

During evaluation, each active tangent-coordinate perturbation is retracted
with the existing absolute step. Both resulting ambient storage vectors must
be distinct from the base vector and from each other. Otherwise evaluation
returns `InvalidOptions` before invoking that coordinate's perturbation
callbacks. The base residual and earlier valid columns may already have been
evaluated. Nonfinite retractions retain their existing `NonFinite` errors.
Fixed variable blocks continue to skip numerical differentiation entirely.

This prevents an unrepresentable perturbation from silently producing a zero
Jacobian and false gradient termination. Comparing both samples also detects
coincident samples after angle canonicalization, even if their representation
differs from the original base. A one-sided perturbation that rounds away is
rejected as well.

These checks compare exact ambient floating-point values. They do not certify
the accuracy of a finite-difference derivative or compare states modulo every
group representation. Callback rounding, truncation error, chart boundaries
and conditioning still matter. The step is not silently adjusted; analytic/AD
comparisons remain useful. The separate Euclidean `CentralDifference` adapter
keeps its documented relative step rule.

Before the change, two of the three new tests fail and the positive control
already passes. Debug and release runs of the fixed implementation cover:

- Rejecting overflow-sized steps through both registration APIs, preserving
  graph dimensions and initial values, then accepting a normal retry whose
  two observations have known least-squares solution `3`.
- Rounded-away active steps for Euclidean, SO(2), SO(3), SE(2), the SE(2)
  Cartesian chart and SE(3), through dense/sparse evaluation and dense/SparseQR
  solving. Cases include asymmetric rounding at `1` and an unwrapped SO(2)
  angle whose two retracted samples coincide after canonicalization.
- Preserving mixed fixed/active blocks and all-fixed evaluation, including
  an identity derivative at the origin with the smallest representable step.

The [evidence manifest](results/numeric_factor_steps/manifest.json) records
the source fingerprints, logs and hashes. To reconstruct the previous source:

```sh
python3 benchmarks/make_ldl_baseline.py --source-only \
  --before-numeric-factor-steps --output /tmp/ropt-before-numeric-factor-steps
```

All 65 source-only historical reconstruction stages preserve their expected
fingerprints. This does not claim compilation of every historical stage.
No timing or comparative performance claim is made.
