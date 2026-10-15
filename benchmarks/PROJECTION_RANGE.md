# Finite marginal priors after large intermediate projections

Marginalization first applies the eliminated-column QR's orthogonal transform
to the retained columns and residual. A Householder reflection doubles a dot
product before updating a column. This intermediate could overflow for inputs
near `f64::MAX` even when the retained projection and compressed prior were
representable. The final range check then rejected a valid prior.

For example, with eliminated column `[1,0]`, retained column `[MAX,1]` and zero
residual, the eliminated variable can cancel the first equation for any retained
increment. The resulting prior has unit information. Previously, applying the
reflection could overflow before discarding that first equation. A fixed-lag
regression with this structure failed its update despite finite input and
unit retained covariance.

The projection now reserves headroom using a power-of-two divisor equal to
eight times the next power of two at least as large as the row count. Only
columns above `MAX / divisor` are scaled. The ordinary reflection arithmetic
remains in use below that bound. Powers of two preserve normal scaled inputs
exactly, and the headroom covers reflection dot products, doubling and updates.
Column scales are restored only after discarding the eliminated rows.

Scaling down a huge column must also preserve its tiny components. Entries
whose scaled values would be subnormal or zero are separated, independently
normalized and projected with the same QR. Their retained contributions are
then recombined. This preserves a smallest-subnormal component beside an
eliminable `MAX` component in the axis-aligned regression, instead of replacing
an overflow error with silent information loss. Temporary tail buffers are
allocated only when needed on the exceptional path.

Six public-API regression tests check:

- Large eliminable components beside ordinary, tiny and smallest-subnormal
  retained components, including large residuals in the same system.
- Non-axis-aligned elimination with known normalized retained information
  `1/2` and an independent constant residual.
- Multiple eliminated directions and restoration of retained rows only.
- A small eliminated-column coefficient multiplying a large input to produce
  a representable projected coefficient.
- Gaussian and frozen-IRLS fixed-lag updates. An injected problem requiring
  an unrepresentable output returns `NonFinite` and preserves owned estimates,
  factors and time. A valid retry removes the expired variable and recovers
  unit separator covariance through dense and sparse covariance inputs.
- Continued rejection when the compressed retained output exceeds the
  representable range.

Five tests fail before the fix; the output-range control already passes.
All six run in debug and release. Reference checks normalize before forming
squared norms, so the oracle does not overflow. Existing profile-objective,
column-scaling, fixed-lag, manifold and robust-model tests remain covered.

Eliminated-column rank checks and tolerances are unchanged. No damping,
pseudoinverse or extended-precision arithmetic is introduced. Projection
rounding, cancellation, restored coefficients, compression and later solver
operations retain their own `f64` limits. Entry limits continue to describe
input matrix size rather than peak allocation. These tests do not establish
arbitrary ill-conditioned nonlinear convergence or comparative performance.
No timing claim is made.

The [evidence manifest](results/prior_projection_range/manifest.json) records
source fingerprints, logs and artifact hashes. Reconstruct the previous source:

```sh
python3 benchmarks/make_ldl_baseline.py --source-only \
  --before-projection-range --output /tmp/ropt-before-projection-range
```

All 70 source-only historical reconstruction stages retain their expected
fingerprints. This does not claim compilation of every historical stage.
