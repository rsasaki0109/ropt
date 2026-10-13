# Nonlinear SE2 streaming accuracy and scheduled refresh recovery

Forced and interval refreshes previously inserted new rows at cached anchors and
solved the old model before reaching batch LM. This can fail on an obsolete
linearization even when the current nonlinear model is observable. A regression
starts with `x=0`, observes `x=1` without refreshing its anchor, then introduces
`y=1` and `xy-2`. At anchor x=0, the new y column is zero: the old code reports
rank 1 of 2 before even attempting a requested refresh. At current x=1, batch LM
recovers x=1, y=2. Both forced and scheduled interval paths now recover this case.

These scheduled paths start from committed estimates plus new seeds, without
evaluating the new factors at old anchors, copying the discarded QR, or solving
the cached system. Threshold checks still require a successful cached solve and
start batch LM from its provisional estimate. This changes scheduled nonlinear
initialization; arbitrary problems can reach different local minima. No automatic
fallback or tolerance relaxation is added. Batch NoProgress/MaxIterations remain
errors with termination, trial count, cost and last trial-gradient diagnostics.

`IsamReport::relinearization_reason` distinguishes Forced, Interval, Threshold and
None. Explicit refresh takes precedence over the interval. A truly unobservable
final undamped QR still returns rank failure even when damped LM found no reason
to move. Tests verify state/history/new-key/cache/interval rollback, final-rebuild
callback failure, reason selection and continued ordinary updates after failure.
Callback side effects outside the smoother are not rolled back.
The later [accepted-model reuse](MODEL_REUSE.md) removes the final callback pass;
current regressions instead inject a batch-trial failure and check actual final
QR rank/fill failure. The accuracy records here remain the earlier checkpoint.

## Fixture and gates

`validate_isam_se2` adds 300 poses to one exactly fixed, nonzero SE2 root, with
curved motion, deterministic odometry error, perturbed incremental seeds and
full correlated SPD information matrices. Every 25 additions brings two loops:
one between existing non-root keys `(k/3,k-5)`, and another `(k/5,k)`. Thus there
are 300 odometry and 24 loop constraints; these are genuine binary constraints,
not unary observations conditioned on the root. Loops explicitly force refresh.
Other updates retain threshold 0.1 and interval 10 (initialization counts).

Incremental factors have analytic right-tangent Jacobians. Full refresh uses
sparse LDL with block minimum degree, block size 3. At all twelve loop
checkpoints, a separate graph built from uncorrected odometry seeds uses forward
AD and dense QR. It retains the same measurements, objective and fixed root.
Both use LM with explicit gradient tolerance **5e-7**, cost tolerance 1e-15,
step tolerance 1e-10, damping 1e-3 and maximum 200 iterations. Library defaults
are unchanged. Gradient tolerance leaves margin below the independent 1e-6 gate.
Exploratory runs with gradient tolerances 1e-10/1e-8/1e-7 encountered NoProgress
near the floating-point floor; they are not passing measurements. The strict
1e-10 diagnostic stopped at frame 50 with last trial gradient 1.26e-10. A
NoProgress result must not be presented as successful convergence.

An independently coded original measurement-frame energy evaluates
`0.5*rᵀΩr`, without ropt residuals, whitening or linearization. Central differences
of this energy under right retraction check gradient infinity norm after every
update. Step 1e-6 is used; every checkpoint also compares each derivative against
step 5e-7, allowing maximum difference 2e-7. The initially tried 1e-5 step had
visible truncation error on long binary loop lever arms and is not used for
the final oracle. Numerical gradients are validation only, never solver Jacobians.

Passing gates are fixed-root exact preservation at every update; original-energy
gradient <=1e-6 at every refresh and for each batch reference; maximum SE2 Local
pose difference <=2e-6 at checkpoints; cost difference <=`1e-9*max(1,cost)`;
and reported refreshed/batch cost agreement <=`1e-10*max(1,cost)`. All metrics
must be finite. Local norms combine translation and radians, so the pose gate
is not a pure metre error. AD/QR and analytic/LDL share ropt's LM, manifold and
noise infrastructure. This is a cross-path consistency check plus an independent
energy/finite-difference oracle, not a global-optimality certificate.

## Recorded outcome

The 300-update run has 258 cached updates, 30 interval refreshes and 12 forced
loop refreshes, with all twelve batch-reference checkpoints passing. Maximum
Local pose difference is **8.27e-7**, cost difference **1.64e-13**, and refreshed
original-energy gradient **4.90e-7**. Final QR stores 29,511 entries for 900
active tangent coordinates. Entries do not measure peak memory.

Ordinary updates have maximum nonlinear gradient **1.01e-2** and the largest
absolute gap between cached cost and actual nonlinear cost is **5.98e-7**.
These approximations are retained in every-prefix CSV; they are not counted as
exact nonlinear solutions. A small cached cost alone cannot certify stationarity.
Threshold refresh is covered by separate regression tests, not exercised in
this fixture. This is synthetic Gaussian accuracy evidence, with no timings,
GTSAM/iSAM2 comparison, selective updates, real-time or field-reliability claim.

Results are in [report](results/nonlinear_isam/frames300/report.json),
[per-update data](results/nonlinear_isam/frames300/updates.csv) and
[metadata](results/nonlinear_isam/frames300/metadata.json). Metadata retains
source/template/executable fingerprints, exact settings, environment and gates.
The earlier linear timings remain a separate historical checkpoint.
The strict-gradient exploratory [failure diagnostic](results/nonlinear_isam/strict_gradient_exploratory/failure.json)
and its 49 completed prefix rows are retained separately. That exploratory
source/binary was not frozen; it is not a passing or reproducible timing sample.

## Reproduce

```sh
cargo build -p ropt --example validate_isam_se2 --release --locked --offline
python3 benchmarks/nonlinear_isam.py \
  --binary target/release/examples/validate_isam_se2 \
  --frames 300 --output /tmp/ropt-nonlinear-isam-check
```

The output directory must be new and outside the checkout. CI runs a smaller
25-addition integration test and a 100-addition release validation. No public
input, Python third-party package, C++ dependency or network access is needed.

`make_ldl_baseline.py --before-scheduled-refresh --output NEW_DIRECTORY` restores
the immediately preceding graph-reusing kernel in a benchmark-only snapshot.
Older baseline stages reverse `isam_scheduled_refresh.patch` first, preserving
the original graph-reuse/objective/LDL comparison checkpoints.
