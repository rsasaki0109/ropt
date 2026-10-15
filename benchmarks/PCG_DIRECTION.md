# PCG rejects invalid directions before operator callbacks

PCG already checked operator and preconditioner outputs, but the direction
update `p = z + beta*p` could produce Inf or NaN after those checks. The next
iteration passed that direction to the user-supplied operator before discovering
the breakdown. With a custom operator this could surface as an evaluation
error, or expose unexpected nonfinite arguments to callback code.

The solver now checks each updated direction for finite entries before using
it. It also validates `r.dot(z)` before applying the operator to the direction,
so zero/negative preconditioned residual products return `LinearSolve` and
nonfinite products return `NonFinite` without an unnecessary operator call.
The curvature check still follows the operator application because it depends
on that result.

Three regression tests fail before the change and pass afterward:

- A fixed SPD matrix with diagonal `[1e-20, 2e300]`, off-diagonal `1e140` and
  identity preconditioning makes `beta` overflow after the first step. The
  resulting direction must not reach the operator callback.
- A fixed SPD matrix with diagonal `[1e-300, 2e300]`, off-diagonal `1`, and
  preconditioner `M = diag(1e-300, 1)` produces a finite `beta` around `1e300`.
  The updated direction nevertheless overflows when multiplying an old entry
  around `1e150`. Checking the ratio alone would miss this case.
- Zero/negative preconditioned residual products, and a product overflowing
  from finite operands, stop after the initial `A*0` evaluation. The first two
  cases deliberately violate the SPD preconditioner contract; the third uses
  identity preconditioning with a large finite right-hand side.

The first two cases use right-hand side `[1e-150, 0]`, have positive diagonal
entries and strictly positive determinants, and use fixed SPD preconditioners.
They demonstrate floating-point range limits even when the algebraic contracts
are satisfied. The guarded operator records every call and rejects nonfinite
inputs with a distinct error, making accidental callback invocation observable.
Debug and release builds exercise the new cases.

This does not extend the representable range of the PCG recurrence. It detects
breakdown before invalid arguments reach callbacks. Direction validation adds
a finite-entry scan after each direction update; it adds no new vector buffer.
No timing or comparative performance claim is made.

The [evidence manifest](results/pcg_direction/manifest.json) records the source
fingerprints, logs and hashes. To reconstruct the previous source:

```sh
python3 benchmarks/make_ldl_baseline.py --source-only \
  --before-pcg-direction --output /tmp/ropt-before-pcg-direction
```

All 61 source-only historical reconstruction stages retain their expected
fingerprints. This does not claim compilation of every historical stage.
