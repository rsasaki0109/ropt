# PCG residual norms at extreme scales

PCG previously used the Euclidean norm obtained by squaring every component
and taking the square root of their sum. With residuals around `1e-200`, those
squares underflowed to zero: even a zero initial guess could be reported as
converged with zero residual while the requested relative tolerance was unmet.
At the other extreme, squaring entries around `1e200` overflowed even when the
norm itself and the linear solution were representable.

PCG now keeps the existing dot/square-root path when the squared norm is normal.
Otherwise, it accumulates the norm with `f64::hypot`. The same check applies to
the right-hand side, initial residual, recursive residual and independently
recomputed `b - A*x` residual. A nonfinite norm returns `Error::NonFinite`.
This changes neither the PCG recurrence nor the preconditioner.

Four regressions fail before the fix and pass after it:

- Solving a well-conditioned two-variable SPD system scaled by `1e-200` and
  `1e-160`, with zero absolute tolerance and `1e-12` relative tolerance.
- Solving the same system scaled by `1e200`.
- Reporting the actual residual norm when the initial guess satisfies a
  deliberately loose absolute tolerance.
- Rejecting a tiny identity-system recurrence whose inner products underflow,
  rather than falsely reporting convergence at the zero initial guess.

The tests use the independent solution `[1/11, 7/11]` and a residual measured
after dividing out the common scale. This avoids repeating the failing norm
formula in the accuracy oracle. Buffered operators, preconditioners and the
ordinary sparse solver tests remain covered by the full suite.

This is a convergence correctness fix, not general scale invariance. Dot
products, operator products, preconditioner application and the tolerance
calculation still have finite-range limits. A scaling or conditioning problem
can therefore still return a breakdown error. No runtime improvement or
comparison with another library is claimed.

The [evidence manifest](results/pcg_residual_norm/manifest.json) records source
fingerprints, test logs and hashes. To reconstruct the previous implementation:

```sh
python3 benchmarks/make_ldl_baseline.py --source-only \
  --before-pcg-norm --output /tmp/ropt-before-pcg-norm
```

All 59 source-only historical reconstruction stages retain their expected
fingerprints. This does not claim compilation of every historical stage.
