# PCG checks the true residual at the iteration limit

Previously, PCG only recomputed `b - A*x` when the recursively updated residual
met the stopping tolerance. At the iteration limit, a slightly pessimistic
recursive residual could cause `LinearIterationLimit` even though the actual
candidate met the tolerance. The solver also applied the preconditioner to
prepare another step that would never run.

The final allowed step now always recomputes the true residual. It returns a
successful report if that residual meets the original tolerance, and otherwise
returns `LinearIterationLimit` without preparing another direction. Errors from
the final operator evaluation or residual validation propagate as usual.
`max_iterations` still counts PCG steps; the final residual check adds an
operator application when the recursive residual has not met the tolerance.

Three regressions fail before the change and pass afterward:

- With the identity preconditioner, `0.1*x = 0.1` produces `x = 1` in one step.
  Its true residual is exactly zero, while its recursive residual is
  `-1.3877787807814457e-17`. With zero tolerance and a one-step limit, this
  must succeed after checking the true residual.
- For `[4 1; 1 3] x = [1 2]`, one step with the identity preconditioner gives
  `[1/4, 1/2]` and the nonzero residual `[-1/2, 1/4]`. It must still return the
  iteration-limit error, with no second preconditioner application.
- An injected evaluation error during the final residual probe must be
  returned rather than hidden by an iteration-limit error.

The tests instrument operator and preconditioner calls while retaining a fixed
SPD operator and identity preconditioner; the last case explicitly injects an
evaluation failure. Debug and release builds exercise the new cases. The full
suite covers ordinary buffered PCG and the preceding extreme-scale norm fix.

The [evidence manifest](results/pcg_iteration_limit/manifest.json) records the
source fingerprints, logs and hashes. To reconstruct the previous source:

```sh
python3 benchmarks/make_ldl_baseline.py --source-only \
  --before-pcg-limit --output /tmp/ropt-before-pcg-limit
```

All 60 source-only historical reconstruction stages preserve their expected
fingerprints. This is not a compilation check for every historical stage.
No timing or comparative performance claim is made.
