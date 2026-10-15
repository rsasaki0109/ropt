# Finite GN/LM predictions at extreme scales

The solver evaluates the weighted quadratic model improvement as
`-gᵀ*step - 0.5*||J*step||²`. Each intermediate can overflow while the
difference is representable. Two residuals `1e154` with Jacobian entries
`1e153` give finite initial cost about `1e308` and gradient about `2e307`.
A nearly optimal step around `-10` makes both intermediates about `2e308`,
so the previous calculation rejected the trial with `NonFinite`.

The ordinary expression remains in use whenever it is finite. Otherwise the
solver evaluates the equivalent per-residual expression
`sum(-qᵢ * (rᵢ + 0.5*qᵢ))`, where `q = J*step`, using compensated accumulation.
Here `r` and `J` are the frozen weighted residual and Jacobian. The fallback
does not allocate another residual vector. Actual robust loss reduction remains
separate from this quadratic prediction.

The same regressions exposed a preceding Sparse LDL failure: its RHS and
true-residual checks squared large vectors to compute their norms. Sparse LDL
now shares PCG's checked norm helper, retaining dot/square-root arithmetic for
normal squared norms and using `hypot` otherwise. The helper remains internal;
no public API, tolerance, refinement limit or PCG recurrence changes.

Seven public-API regression tests cover:

- LM acceptance for the Gaussian problem above through Dense, Dense QR,
  Sparse LDL and Sparse QR, checking step, actual improvement and gain ratio.
- Gauss–Newton acceptance through Dense, Dense QR and Sparse QR.
- Huber and Cauchy LM steps with finite costs near `1.5e308` and `1.386e308`.
  Their distinct gain ratios match the true robust improvement divided by
  the frozen quadratic prediction.
- A deliberately incorrect Jacobian proposing a finite, worsening trial.
  The solver computes a negative gain ratio and preserves the accepted state.
- Sparse LDL solutions and reported original-system residual norms for a
  known two-variable SPD problem at ordinary, tiny and large RHS scales,
  through `1e307`.
- Continued rejection of an unrepresentable objective or an unrepresentable
  RHS norm.

Scalar fallback and explicit joint loss are exercised. Norm and cost references
are normalized before comparison so the oracle does not overflow its own
squares. A separate unit test rejects genuinely unrepresentable quadratic
models and nonfinite Jacobian-step products. All eight new tests run in debug
and release.

The seven public tests also run against the isolated pre-change source from
commit `f3540de`, whose source fingerprint is recorded in the manifest, using
a separate build directory. Five fail and two range-limit controls pass before
the fix. Frozen before/after sources, both regression-source versions and logs
are retained with the evidence.

These are model-evaluation and linear-system range fixes. Normal assembly,
gradients, step formation, recurrence products and subsequent calculations
retain their own `f64` limits. The nonlinear regressions inspect a single trial;
they do not establish general convergence or comparative performance at these
scales. No timing claim is made.

The [evidence manifest](results/predicted_reduction_range/manifest.json) records
source fingerprints, logs and artifact hashes. Reconstruct the previous source:

```sh
python3 benchmarks/make_ldl_baseline.py --source-only \
  --before-predicted-range --output /tmp/ropt-before-predicted-range
```

All 72 source-only historical reconstruction stages retain their expected
fingerprints. This does not claim compilation of every historical stage.
