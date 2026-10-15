# Fixed-lag covariance input validation before callbacks

`FixedLagSmoother::update_with_covariance` now validates covariance options and
requested variable blocks before nonlinear optimization evaluates factors.
Empty, duplicate, missing and fixed-key queries, invalid rank tolerances,
variance scales, SVD iteration limits and fallback losses are rejected without
calling old or newly proposed factor callbacks.

Previously these checks ran after optimization. The new regression fails on
the preceding source because an invalid request executes **15 callbacks**;
both new tests pass with **zero callbacks** after the fix. They also check that
estimates, factors, proposed variables, timestamp refreshes and committed time
remain unchanged. A valid retry without a timestamp refresh expires the original
key and returns independently known covariance in the requested order.

The graph's query-to-tangent mapping is shared with its public covariance API.
Actual covariance still evaluates the optimized full-window Jacobian and checks
its rank. Numerical failures after valid input can therefore invoke callbacks;
callback side effects in that case cannot be rolled back. No change is made to
tolerances, robust-loss semantics, expiry policy or ordinary updates.

```sh
cargo test --locked --offline --test fixed_lag_covariance_preflight
```

The retained [validation evidence](results/covariance_preflight/manifest.json)
contains the failing-before and passing-after logs, complete-suite results,
source delta and rollback checks. This is input-validation work with no new
speed, peak-memory or GTSAM comparison claim.

Restore the preceding source with:

```sh
python3 benchmarks/make_ldl_baseline.py --before-covariance-preflight \
  --source-only --output /tmp/ropt-before-covariance-preflight
```

Its fingerprint is
`12ea8150c0eb9b88c91ef527eb9d77249ed0a45b68bd6f415ac779c619e2e288`;
the changed fingerprint is
`ee9385ac5faef90c00f825ced8a9de02872030a6a892dbdebbe8bcfb108e77a6`.
All 54 source-only rollback stages retain their expected historical fingerprints.
