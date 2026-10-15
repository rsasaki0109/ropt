# Preserve finite costs across fixed-lag retirement

Fixed-lag smoothing carries irreducible cost when every connected variable
retires. Frozen-IRLS marginalization also carries the difference between the
robust objective and the weighted quadratic objective. Both calculations
previously evaluated `0.5 * residual.norm_squared()`.

The intermediate squared norm could overflow even when its half was finite.
Two constant residuals `1e154` have squared sum about `2e308`, exceeding
binary64 range, but their objective cost is about `1e308` and is representable.
A valid initial window could therefore fail to retire. Frozen-IRLS could also
produce an invalid offset before creating its prior.

Both sites now reuse the existing ordered Gaussian cost accumulator. It sums
half-squared residual terms directly with compensation, without materializing
another vector. This preserves representable costs while continuing to reject
true objective overflow. Accumulation rounding can differ from the earlier
dot-product calculation; the cost definition is unchanged.

Five public-API regressions cover:

- Complete retirement with two `1e154` constant residuals, followed by repeated
  idle updates. Gaussian and frozen-IRLS paths, active and fixed variables,
  scalar fallback and explicit joint loss all retain the finite cost.
- Genuine Huber and Cauchy frozen offsets whose weighted squared norm exceeds
  the range while the quadratic and robust costs remain finite. The test
  references are about `1.4e308` and `1.3862943611198907e308`.
- A continuous update sequence: retire an anchored predecessor, add an
  observation to its separator, retire the separator, accept an idle update,
  then reuse a retired numeric key for a fresh variable. The prior variance
  `2`, updated estimate `4/3`, updated variance `2/3`, and fresh variance `1`
  match independent linear Gaussian references. The large constant is carried
  throughout. Dense and sparse QR solvers and both marginalization modes are
  exercised.
- Rollback when a new finite objective plus the carried constant exceeds the
  range. Proposed variables and factors are discarded, and a valid retry
  preserves the original constant.
- Continued rejection of an unrepresentable objective before marginalization,
  with estimates, factors and time preserved for retry.

Four tests fail before the fix; the last range-limit control already passes.
All five run in debug and release. Existing covariance, robust-model,
projection, column-scaling and empty-window tests remain covered.

The sequential regression explicitly disables aggregate-relative cost stopping
with `cost_tolerance = 0`, since its deliberately huge constant would dominate
that criterion. It uses linear Gauss–Newton updates and checks accepted steps
with positive reported reduction. It does not establish arbitrary nonlinear
convergence at these scales. QR compression can still change carried costs by
rounding, and contributions far below the aggregate cost's precision may not
appear in the reported total.

All solver and covariance quantities retain their own `f64` limits. Frozen
priors keep their existing local approximation semantics. Input limits remain
matrix-entry limits rather than peak-byte budgets. No timing or comparative
performance claim is made.

The [evidence manifest](results/fixed_lag_cost_range/manifest.json) records
source fingerprints, logs and artifact hashes. Reconstruct the previous source:

```sh
python3 benchmarks/make_ldl_baseline.py --source-only \
  --before-lag-cost-range --output /tmp/ropt-before-lag-cost-range
```

All 71 source-only historical reconstruction stages retain their expected
fingerprints. This does not claim compilation of every historical stage.
