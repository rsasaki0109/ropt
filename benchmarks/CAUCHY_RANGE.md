# Cauchy loss at extreme scales

Cauchy loss evaluates `a²/2 * ln(1 + (r/a)²)` with IRLS weight
`1 / (1 + (r/a)²)`. Previously, squaring a large normalized residual could
produce infinity even when the cost and weight were representable. For
`r = 1e155, a = 1`, the finite cost is about `356.901` and the weight is about
`1e-310`. The earlier formula produced infinite cost and zero weight.

The implementation keeps the original arithmetic when the normalized square
and computed cost are normal finite values. Exceptional inputs use equivalent
forms: the inverse residual ratio for large residuals, a logarithmic ratio
when division overflows, and cancellation of the scale squared for small
normalized residuals. Multiplication order also preserves representable costs
when the squared scale underflows or overflows.

Step acceptance needs the difference of true losses. It retains the ordinary
calculation when the result is normal, then uses bounded scale ratios and
`ln_1p` for extreme inputs. Close residuals are compared before taking a
logarithm; this avoids subtracting two separately rounded objective values.
Increasing trials use the negative of the corresponding decrease.

Six public-API regression tests cover:

- Finite costs when a ratio, its square, or the squared loss scale exceeds
  the floating-point range; subnormal costs and zero residuals are included.
- A subnormal weight checked through dense and sparse-input frozen-IRLS
  covariance. With `r = J = 1e155` and `a = 1`, inverse information is about `1`.
- LM acceptance between two large residuals, with true reduction `ln(2)`.
- A Gauss–Newton update crossing below the loss scale.
- Rejection of a deliberately residual-increasing trial.
- Continued rejection when the true cost exceeds the representable range.

Scalar fallback and explicit joint losses are exercised, using dense, dense
QR, sparse LDL and sparse QR paths where supported. Five of these tests fail
before the fix; the unrepresentable-cost control already passes.

An additional unit test checks eight loss differences against independent
1100-digit Decimal references evaluated from the exact binary64 inputs. It
includes close large residuals, large scales, subnormal results, reverse
directions and unchanged norms. The reference-generation script and output
are archived with the evidence. All seven new tests also run in release mode.

Raw IRLS weights and other solver quantities retain their own `f64` range
limits. A mathematically positive weight below that range can still become
zero; the change does not add a log-weight or extended-precision solver.
The [square-root weight follow-up](ROBUST_ROOT_WEIGHTS.md) now keeps such a
weight usable in the model when its square root is representable.
Frozen-IRLS covariance remains conditional on the selected weight model.
No timing or comparative performance claim is made.

The [evidence manifest](results/cauchy_range/manifest.json) records source
fingerprints, validation logs and artifact hashes. Reconstruct the previous
source with:

```sh
python3 benchmarks/make_ldl_baseline.py --source-only \
  --before-cauchy-range --output /tmp/ropt-before-cauchy-range
```

All 67 source-only historical reconstruction stages retain their expected
fingerprints. This does not claim compilation of every historical stage.
