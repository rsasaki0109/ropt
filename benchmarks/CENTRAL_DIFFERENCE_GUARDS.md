# Central difference perturbation and output guards

The explicit Euclidean `CentralDifference` adapter previously accepted a positive
step even if `x + h` and `x - h` both rounded back to `x`. For the residual
`x - 2`, an initial value of `1` and step `1e-30` produced a zero Jacobian and
gradient termination at cost `0.5`, without moving toward the solution `2`.

The adapter now validates every coordinate before evaluating callbacks or
modifying residual/Jacobian output buffers. Its existing rule remains
`h = step * max(abs(x), 1)`:

- A nonfinite input, perturbation, step product or denominator `2*h` returns
  `Error::NonFinite`.
- A perturbation that rounds back to the original coordinate on either side
  returns `Error::InvalidOptions`.
- Callback vectors must have the requested length and finite entries. Invalid
  output stops evaluation immediately, before subsequent perturbation callbacks.
- A nonfinite derivative column returns `Error::NonFinite`.

This does not silently change the requested step or select another derivative
method. It guards the existing Euclidean adapter; numeric factor-graph callbacks
use a separate tangent-coordinate implementation. Finite distinct input points
alone do not guarantee an accurate finite-difference derivative: callback output
rounding, truncation error and conditioning still matter. Analytic/AD comparisons
remain useful when choosing a numerical step.

Before the change, three of the four new tests fail. The fourth is a positive
control and already passes. The regressions cover preflight errors without
callback/output-buffer changes, the false solver convergence above, nonfinite
base/perturbed callback outputs and derivative overflow. Positive controls cover
the smallest representable step at the origin for the identity function and a
normal solve of `x - 2`. Existing tests compare analytic, AD and central-difference
Jacobians. Debug and release builds exercise all four new tests.

The [evidence manifest](results/central_difference_guards/manifest.json) records
the before/after source fingerprints, logs and file hashes. Reconstruct the
previous source with:

```sh
python3 benchmarks/make_ldl_baseline.py --source-only \
  --before-central-difference --output /tmp/ropt-before-central-difference
```

All 63 source-only historical reconstruction stages preserve their expected
fingerprints. This does not claim compilation of every historical stage.
No timing or comparative performance claim is made.
