# Keep prior information across extreme column scales

Square-root marginalization projects out eliminated variables and compresses
the remaining augmented matrix `[J,r]` with QR. Compression previously divided
the entire augmented matrix by its largest entry before QR. A large constant
residual could make a retained Jacobian column so small that QR's squared
column norm underflowed. More extreme ratios could underflow the division
itself. The resulting prior silently lost information.

For example, a Jacobian column around `1e-100` beside a constant residual
`1e150` became roughly `1e-250` after shared scaling. Its squared norm rounded
to zero even though the original column and the required compressed prior
were representable. A fixed-lag regression with a retained column `1e-200`
then reported zero frozen rank after marginalization.

Compression now checks each nonzero column's maximum relative to the shared
scale. If squaring that ratio would be subnormal or zero, it normalizes the
columns separately, runs QR, and restores the individual column scales on the
result. Right column scaling commutes with the orthogonal transformation;
restoring it preserves the original profile objective and constant component.
The original shared-scale arithmetic remains in use otherwise. The extra
column-scale vector is allocated only for the exceptional path.

The eliminated-variable SVD rank policy and its relative cutoff are unchanged.
No damping, pseudoinverse, rank truncation or prior regularization is added.
Within-column precision, projection arithmetic, output coefficients and later
solver operations retain their own `f64` limits. This change does not guarantee
arbitrary ill-conditioned problems are numerically solvable.

Five public-API regression tests cover:

- Independent retained columns at ordinary, extreme and subnormal scales,
  both with and without elimination, beside a large orthogonal constant.
- Non-axis-aligned elimination with dependent retained columns. The known
  normalized Gram matrix `[[2,4],[4,8]]` and independent constant are preserved.
- Fixed-lag retention of a tiny separator column beside a large constant,
  through both Gaussian and explicitly selected frozen-IRLS marginalization.
- A callback failure during marginalization after optimization has finished.
  Owned estimates, factors and time roll back; a valid retry produces the
  expected separator information.
- Continued `NonFinite` rejection when the chosen compressed output itself
  exceeds the representable range.

Information comparisons normalize columns before forming Gram matrices, so
the oracle cannot hide lost information behind the large constant or underflow
its own squares. The fixed-lag case uses an explicit variance multiplier
`1e-200` to report a representable covariance `1e200` from the retained
square-root information `1e-200`. This does not claim an unscaled variance
`1e400` is representable. The estimates start at the linear optimum; this case
checks information and transaction semantics rather than nonlinear convergence.

Before the fix, four tests fail; the output-range control already passes.
All five run in debug and release. Existing marginalization profile-objective,
fixed-lag covariance, robust-weight and manifold tests also remain covered.
No timing or comparative performance claim is made.

The [evidence manifest](results/prior_column_scaling/manifest.json) records
source fingerprints, validation logs and artifact hashes. Reconstruct the
previous source with:

```sh
python3 benchmarks/make_ldl_baseline.py --source-only \
  --before-prior-column-scaling --output /tmp/ropt-before-prior-column-scaling
```

All 69 source-only historical reconstruction stages retain their expected
fingerprints. This does not claim compilation of every historical stage.
