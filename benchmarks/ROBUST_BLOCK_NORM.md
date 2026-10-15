# Stable joint residual norms for robust losses

Joint loss blocks previously computed their norm by squaring each residual,
summing, and taking a square root. The squared sum could underflow to zero or
overflow to infinity even when the norm and required robust-model quantities
were representable.

This affected both nonlinear objective evaluation and the shared weight
calculation used by frozen-IRLS covariance. For residual `[3s, 4s]` and Huber
threshold `s`, the joint weight is `1/5`. With an identity Jacobian, inverse
frozen information is `5*I`. At `s = 1e-200`, an incorrectly zero block norm
instead gave unit weight and covariance `I`. Cauchy scale `s` similarly requires
weight `1/26` and inverse frozen information `26*I`.

The implementation retains the usual dot/square-root path when the squared
norm is normal. Otherwise it accumulates with `f64::hypot` over a borrowed
residual-block view. No additional residual vector is allocated. The existing
loss formulas, weighting convention and covariance model remain in use.

Four new regression tests cover:

- The Huber and Cauchy weights above at ordinary, tiny and squared-subnormal
  scales, checked through dense and sparse-input frozen-IRLS covariance.
- Representable Huber/Cauchy costs for a fixed graph with residual
  `[3e200, 4e200]`, checked through dense and SparseQR nonlinear entry points.
  These are cost evaluations with all variables fixed.
- Finite inverse frozen information at that large residual scale, using
  independent scalar multiples of `I` as references.
- Continued rejection of a genuinely unrepresentable joint norm.

Before the fix, three tests fail and the rejection control already passes.
Debug and release builds execute all four. Large-value comparisons normalize
the result before calculating test errors, so the accuracy oracle does not
overflow its own squared norm. Existing covariance, fixed-lag robust-model and
stable-objective regressions are also exercised.

Very small loss values can still underflow while their weights remain
representable. Loss formulas, covariance values and other solver arithmetic
retain their own f64 range limits. The checks evaluate the explicitly selected
FrozenIrls information model at the supplied point. No timing or comparative
performance claim is made.

The [evidence manifest](results/robust_block_norm/manifest.json) records source
fingerprints, logs and hashes. Reconstruct the previous source with:

```sh
python3 benchmarks/make_ldl_baseline.py --source-only \
  --before-robust-block-norm --output /tmp/ropt-before-robust-block-norm
```

All 66 source-only historical reconstruction stages retain their expected
fingerprints. This does not claim compilation of every historical stage.
