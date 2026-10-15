# ropt

**A Rust-native nonlinear optimization library for robotics.**

[![Rust CI](https://github.com/rsasaki0109/ropt/actions/workflows/ci.yml/badge.svg)](https://github.com/rsasaki0109/ropt/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)

Dense and sparse nonlinear least squares, automatic differentiation, Lie groups,
and factor graphs. Pure Rust dependencies, with no C++ runtime or FFI.
**Experimental / pre-alpha.**

![ropt processes the complete recorded Intel943 graph: actual pose estimates, original energy and archived equal-accuracy iSAM2 comparison](docs/assets/ropt-hero.gif)

*Recorded Intel SE2 measurements: 943 poses, 1,837 constraints. Actual selected
arrival checkpoints; pose-ID order is inferred because the source has no timestamps.
Playback is accelerated. At the shared accuracy target, the complete replay is
1.22x faster than verified GTSAM iSAM2 in the primary campaign and 1.09x in the
independent confirmation, with 12/12 faster pairs. These are forced full-refresh
policies. The separately selected MIT1228 artifact now passes a
[Cartesian-chart comparison](benchmarks/PRECISE_REAL_ISAM2.md), with a
precision-corrected native reference; the historical hero data stays frozen.*
[Results and limits](benchmarks/REAL_ISAM2.md) · [Reproduce every visual](docs/visualization/README.md)

On a different recorded graph, [CSAIL](benchmarks/CSAIL_REAL_ISAM2.md), the same
common-accuracy full-refresh policy is **2.29x / 2.28x faster** than the corrected
GTSAM iSAM2 reference in primary/confirmation measurements, winning **12/12 pairs**.

A [fresh comparison of the mixed-AD selective backend](benchmarks/CURRENT_NATIVE_ISAM2.md)
rebuilds GTSAM 4.2.1 and measures both solvers again on CSAIL: ropt stream medians
are **13.739 / 16.865 s**, versus **25.609 / 26.878 s** for the accuracy-targeted
native reference (**1.86x / 1.59x**, **6/6 / 5/6** faster pairs). All 25,056 timed
arrivals pass the common accuracy gates. Cloud timing varies substantially;
the final confirmation pair favors GTSAM. These are selected, different policies,
and the native policy fully relinearizes to the target; this does not measure
default iSAM2 or establish general superiority.
Arrival tails are mixed: confirmation P99 is **209.910 ms versus 179.719 ms**,
and both observed maxima favor native GTSAM.
The subsequent [refinement termination change](benchmarks/REFINEMENT_STEP.md)
skips unused next-step solves. At 128 variables it saves **2 allocation requests /
2 KiB** per editable-tree refresh; complete CSAIL poses and non-timing work remain
exactly equal. A new stream-time or tail-latency improvement is unmeasured.

### Watch the solver work

| Bundle adjustment | Nonlinear least squares |
| --- | --- |
| ![Actual synthetic BA checkpoints: camera poses, landmarks and reprojections converge](docs/assets/ba-convergence.gif) | ![Actual Rosenbrock LM trials follow the curved valley to the minimum, including rejected steps](docs/assets/rosenbrock-convergence.gif) |
| 4 cameras · 12 landmarks · 48 observations. Two poses anchor frame and scale. Actual LM + block Schur checkpoints; visual interpolation. | Rosenbrock · forward automatic differentiation · LM gain-ratio decisions. Actual checkpoints, including rejections; no interpolation. |

These two optimization demos are synthetic and make no comparative speed claim.
Run `cargo run -p ropt --example bundle_adjustment --locked` or
`cargo run -p ropt --example rosenbrock --locked`.

### Geometry and measurements

![Recorded Intel943 original seeds and ropt final geometry with retained loop constraints](docs/assets/intel943-trajectory.png)

*The full recorded graph, using all original constraints and provided vertex seeds.*

![All twelve paired replay measurements show ropt and verified native GTSAM iSAM2 timings under the common accuracy contract](docs/assets/isam2-comparison.png)

*Every dot is an archived complete replay; lines join paired runs and white bars
mark medians. All arrival-level accuracy checks pass on Intel943. Cloud variation
is visible, and the result applies to the stated graph and policies.*

![Recorded CSAIL odometry seeds and actual final geometry alongside all twelve accuracy-matched paired timings](docs/assets/csail-comparison.png)

*CSAIL: 1,045 poses · 1,172 original constraints. Both solvers use the same seeds
integrated from original odometry. All 1,044 arrivals pass the unchanged accuracy
gates. Every timing line is an archived pair; black bars mark campaign medians.
[Data, results and reproduction](benchmarks/CSAIL_REAL_ISAM2.md).*

## Implemented

- Dense Gauss–Newton using SVD or column-pivoted QR with explicit rank failure.
- Levenberg–Marquardt with scaled damping, predicted/actual improvement ratio,
  accepted/rejected trials, adaptive damping, and iteration records.
- Optional dense QR LM solves augmented least squares without forming `JᵀJ`.
- Optional ordered sparse Givens QR for GN and augmented LM, without a numeric
  normal matrix. GN can explicitly permit finite uphill trials; this is disabled
  by default and does not turn iteration limits into convergence.
- Gradient, accepted-step, cost-change, iteration-limit and stagnation outcomes,
  with explicit absolute or weighted-column-normalized gradient stopping.
- Squared, Huber and Cauchy loss using IRLS: a scalar fallback plus optional
  per-factor joint loss on whitened observation vectors.
- Unit, isotropic, diagonal and correlated covariance noise models, consistently
  whitening both residuals and Jacobians.
- Analytic `Residual` trait, fixed-size `Jet`, chunked runtime-sized `DynamicJet`,
  and explicit central differences. AD factors support Euclidean and all four Lie groups
  through right-tangent differentiation.
- Contiguous parameter block ranges; the solver operates on their flattened vector.
- SO(2), SO(3), SE(2), SE(3) Exp/Log, composition, inverse, right retract/local;
  Euclidean retract/local. SE(3) tangent ordering is translation then rotation.
- `LocalProblem` / `solve_local` optimize with tangent Jacobians and right retractions.
- `FactorGraph` with dense and CSR assembly, graph-owned variable IDs, fixed blocks,
  analytic, forward AD or explicitly requested numeric factors, and a small anchored
  SE(3) chain with loop closure.
- AD SE(2) relative pose factors, strict 2D g2o reading/writing and a public
  30-pose sample validated with dense and sparse LM.
- Four-coefficient Brown–Conrady projection and joint BA estimation of shared
  focal lengths, principal point, distortion, camera poses and landmarks.
- Forward AD SE3 relative Lie Log factors and analytic absolute position factors
  with antenna lever arms; a synthetic GNSS/odometry fusion example.
- Selected tangent-space marginal covariance using globally rank-checked SVD,
  with explicit Gaussian or frozen-IRLS models and known noise scaling.
- Batch fixed-lag smoothing with timestamped/persistent variables, square-root
  marginalization, frozen manifold priors and transactional observation updates.
- Initial iSAM-style sparse Givens QR row updates, with cached linearizations
  and threshold/periodic full nonlinear refresh. Optional scalar Gaussian Bayes
  tree updates re-eliminate touched conditional paths. An explicit editable
  mode replaces selected Gaussian factors through cached separator messages,
  retaining unaffected cliques; symbolic analysis/back-substitution stay global.
  Optional threshold-selected nonlinear factor evaluation retains unrelated
  whitened rows and reports bounded-pass exhaustion. Its ordinary QR backend
  rebuilds globally; the editable mode re-eliminates affected cliques.
  Forced/interval refreshes retain the full batch solver.
  Optional batch-order reuse controls QR fill; bounded frozen-QR refinement can
  improve the accepted nonlinear estimate while preserving the cache anchor.
- Canonical CSR matrices, Jacobi-preconditioned CG with true-residual checks, and
  matrix-free sparse LM normal operators. Dense mode remains the default.
- Optional variable-block Jacobi for full sparse LM, with heterogeneous tangent
  blocks, local Cholesky solves and exchangeable PCG preconditioners.
- A dense Schur elimination reference backend for LM.
- Sparse direct LDLᵀ with minimum-degree ordering, fill budgets and checked residuals.
- CSR-based independent-block Schur elimination with a matrix-free reduced PCG
  solve, analytic/AD pinhole reprojection factors, and synthetic Bundle Adjustment.

The initial Cargo workspace has one crate; module boundaries avoid premature
splitting and circular dependencies.

## Develop

Rust 1.99.0 is pinned in `rust-toolchain.toml`. Dependencies are locked.

```sh
cargo test --workspace --all-targets --locked
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo run -p ropt --example rosenbrock --locked
cargo run -p ropt --example ill_conditioned_qr --locked
cargo run -p ropt --example pose_graph_2d --locked
cargo run -p ropt --example pose_graph_g2o --locked -- --input crates/ropt/tests/data/loop_se2.g2o
cargo run -p ropt --example pose_graph_3d --locked
cargo run -p ropt --example gnss_odometry --locked
cargo run -p ropt --example fixed_lag_odometry --locked
cargo run -p ropt --example fixed_lag_covariance --locked
cargo run -p ropt --example isam_pose_graph --locked
cargo run -p ropt --example block_jacobi --locked
cargo run -p ropt --example bundle_adjustment --locked
cargo run -p ropt --example bundle_adjustment --locked -- --autodiff
cargo run -p ropt --example robust_bundle_adjustment --locked
cargo run -p ropt --example calibration_bundle_adjustment --locked
cargo run -p ropt --example dynamic_curve_fitting --locked
cargo doc --workspace --no-deps --locked
```

On this managed cloud machine first export:

```sh
export RUSTUP_HOME=/workspace/.ropt-toolchain/rustup
export CARGO_HOME=/workspace/.ropt-toolchain/cargo
export PATH="$CARGO_HOME/bin:$PATH"
cd /workspace/ropt
```

## Minimal usage

```rust
use ropt::{autodiff::{AutoDiff, Jet}, solve, DVector, Options};

fn main() -> Result<(), ropt::Error> {
    let residual = AutoDiff::<1, 1, _>(|x: [Jet<1>; 1]| [x[0] - Jet::constant(3.0)]);
    let report = solve(&residual, DVector::zeros(1), &Options::default())?;
    assert!((report.x[0] - 3.0).abs() < 1e-7);
    Ok(())
}
```

Inspect `Report::termination`; an iteration limit or `NoProgress` does **not**
prove convergence. Evaluation failures and nonfinite results return errors.
Default GN has no line search and reports `NoProgress` when its full step fails
to improve. Explicit nonmonotonic GN permits uphill steps with positive predicted
improvement; final convergence must still be checked.
LM's robust model is a first-order IRLS approximation; it is not the full robust
Hessian. Default dense LM normal equations square the condition number;
`DenseQr` avoids that extra conditioning penalty by solving augmented least
squares. Scaling and conditioning still matter. Sparse LM offers PCG and
direct LDLᵀ of normal equations; both inherit their conditioning. Performance
comparisons apply to the recorded workloads and solver configurations; real-time
suitability remains unvalidated.

## Scope still missing

Sparse covariance, rank-revealing sparse methods, supernodal factorization,
heterogeneous/coupled eliminated Schur blocks,
parallel evaluation, sparse marginalization, selective nonlinear iSAM2 updates,
IMU preintegration/raw GNSS processing
and large-scale, equal-accuracy comparative benchmarks. BA supports undistorted pinhole and four-coefficient
Brown–Conrady cameras; validation still uses synthetic image observations only.
Sparse LM supports direct LDLᵀ and PCG with
scalar or variable-block Jacobi preconditioning; difficult conditioning can require
stronger preconditioners, scaling or different linear methods.
The SE(2) example solves a graph with AD tangent Jacobians and a fixed pose.
Recorded-data replay includes a complete 943-pose Intel graph. An optional sparse
QR/GN path now completes the separate 1,228-pose version, but its strict reported
cost check remains unmet. This does not establish production SLAM readiness.

Exp/Log use principal rotations, with the usual ambiguity at pi; inputs to the
standalone manifold primitives must be finite. No global convergence guarantees.
This is not yet production-ready or equivalent to GTSAM, Ceres, or g2o.

## Factor graph and manifold solves

`VariableKind` defines ambient storage and right tangent updates. SO3 stores
`[w,x,y,z]`; SE3 stores `[tx,ty,tz,w,x,y,z]` and uses a six-dimensional tangent.
Input quaternions must have unit norm within 1e-8. Variable IDs belong to their graph.
`set_fixed` excludes a block's tangent columns and preserves its ambient values.
An all-fixed graph evaluates its cost without taking any steps.

Analytic factors return a Jacobian block for every connected variable, ordered as
registered and differentiated with respect to **right tangent increments** (even
for fixed variables). Numeric factors explicitly differentiate `retract(x, +/-h)`;
fixed variables are not differentiated. Neither path silently falls back to the
other. Factors must have distinct variable IDs. `Options::loss` is a per-scalar
fallback. `FactorOptions` can add measurement whitening and replace that fallback
with a joint robust loss for the whole factor, as described below.
`solve_local` step tolerance uses a dimensionless scale of 1 for graphs; choose
appropriate units and tolerances when mixing translation and rotation.

Objective values use compensated summation. GN/LM step acceptance and LM gain
ratios use stable true-loss changes per scalar or explicit joint block, before
IRLS weighting. `Iteration::actual_reduction` records that change; it can be
nonzero when the rounded aggregate `cost` and `trial_cost` are identical.
Squared, Huber and Cauchy retain their original objectives. Each model stores
one additional f64 norm per effective loss term for this calculation.
See the [numerical validation](benchmarks/STABLE_OBJECTIVE.md).

```rust
use ropt::{factor::{FactorGraph, VariableKind}, solve_local, DVector, DMatrix, Options};

fn main() -> Result<(), ropt::Error> {
    let mut graph = FactorGraph::new();
    let anchor = graph.add_variable(VariableKind::Euclidean(1), DVector::from_element(1, 3.0))?;
    let estimate = graph.add_variable(VariableKind::Euclidean(1), DVector::zeros(1))?;
    graph.set_fixed(anchor, true)?;
    graph.add_analytic_factor(&[anchor, estimate], 1, |v| {
        Ok((&v[1] - &v[0], vec![-DMatrix::identity(1, 1), DMatrix::identity(1, 1)]))
    })?;
    let result = solve_local(&graph, graph.initial(), &Options::default())?;
    assert!((graph.value(&result.x, estimate)?[0] - 3.0).abs() < 1e-7);
    Ok(())
}
```

## Fixed and runtime automatic differentiation

`autodiff::Jet<N>` keeps derivative storage in fixed-size arrays, with a
compile-time dimension. `DynamicJet` and `DynamicAutoDiff` support parameter and
residual counts chosen at runtime. Both provide arithmetic, sin/cos/tan, atan,
atan2, exp, ln, sqrt and integer powers. Dynamic jets also support borrowed
operands and f64 scalar operators. `y.atan2(x)` uses the same ordering as f64.

```rust
use ropt::{autodiff::{DynamicAutoDiff, DynamicJet}, solve, DVector, Options};

fn main() -> Result<(), ropt::Error> {
    let initial = DVector::from_vec(vec![-1.2, 1.0]);
    let residual = DynamicAutoDiff::new(initial.len(), 2, |x: &[DynamicJet]| {
        Ok(vec![10.0 * (&x[1] - x[0].powi(2)), 1.0 - &x[0]])
    })?.with_chunk_size(1)?;
    let report = solve(&residual, initial, &Options {
        max_iterations: 200, ..Options::default()
    })?;
    assert!(report.cost < 1e-16);
    Ok(())
}
```

Dynamic AD defaults to chunks of 32 derivative directions. For `N` parameters
and width `K`, it calls the callback `ceil(N/K)` times with identical primal
inputs. Only inputs in the current chunk are seeded; the other input jets are
constants with empty derivative vectors, which broadcast into active spaces.
This bounds seed storage by `O(N + K²)` and each jet's derivatives by `K`, but
adds repeated primal evaluations. The final Jacobian is still dense `M×N`;
this is not a claim of faster runtime or globally constant memory. Fixed jets
remain useful for small dimensions and avoiding derivative heap allocations.

Callbacks must be deterministic, return the declared residual count and use
the supplied input jets for differentiation. Do not reseed jets or detach them
with `.value()` during residual arithmetic. The adapter verifies identical
residual values across chunks. Active derivative-space mismatches and nonfinite
values/partials return errors rather than silently truncating derivatives.
User callbacks must still respect their slice indices and mathematical domains.
A positive chunk size is required. Nonsmooth/singular derivatives, such as an
active sqrt at zero or atan2 at the origin, are not made differentiable by AD.

`FactorGraph::add_autodiff_factor` takes slices of jets in the registered variable
order and automatically splits its Jacobian into blocks. Its `_with_options`
variant configures chunk size, whitening and a joint loss. Euclidean, SO2, SO3,
SE2 and SE3 variables are supported, including fixed variables. The adapter
seeds tangent increments at zero, computes jet-valued right retractions, and
passes the resulting ambient blocks to the callback. SO3 therefore produces
three Jacobian columns from four quaternion storage coordinates; SE3 produces
six from seven. SE3 tangent order is body translation then body rotation.
`VariableKind::retract_autodiff` also exposes this jet-valued retraction directly.
Small-angle expansions avoid singular derivatives at zero rotation. Principal
SO2/SE2 angle storage still has a discontinuity at the pi branch cut; write
periodic angular residuals where appropriate. AD does not remove chart cuts or
quaternion sign ambiguity. Graph assembly drops fixed columns, so fixed variables
can be activated later without replacing callbacks.

```rust
use ropt::{factor::{FactorGraph, VariableKind}, solve_local, DVector, Options};

fn main() -> Result<(), ropt::Error> {
    let mut graph = FactorGraph::new();
    let pair = graph.add_variable(VariableKind::Euclidean(2), DVector::zeros(2))?;
    let sum = graph.add_variable(VariableKind::Euclidean(1), DVector::zeros(1))?;
    graph.add_autodiff_factor(&[pair, sum], 3, |b| Ok(vec![
        &b[0][0] - 2.0, &b[0][1] - 3.0, &b[1][0] - (&b[0][0] + &b[0][1])
    ]))?;
    let result = solve_local(&graph, graph.initial(), &Options::default())?;
    assert!((graph.value(&result.x, sum)?[0] - 5.0).abs() < 1e-7);
    Ok(())
}
```

`dynamic_curve_fitting` fits 40 runtime-sized observations of
`a*exp(-rate*t) + bias`. A runtime AD residual and separate Euclidean AD factors
with block Schur both recover the synthetic parameters `(2.0, 0.4, 0.5)` within
1e-6. Numeric differentiation is used only in verification tests, not by either
optimization path. Division and atan/atan2 derivative formulas avoid selected
intermediate square overflows; this is not an assurance of accuracy for all
extreme scales or ill-conditioned expressions.

## Measurement noise and robust observation blocks

Use `add_analytic_factor_with_options` or `add_numeric_factor_with_options` to
configure each factor. Existing registration methods retain unit noise and the
`Options::loss` scalar fallback. Reprojection has a matching
`camera::add_reprojection_factor_with_options` helper.

`NoiseModel` supports:

- `unit()`: keep the residual's original units.
- `isotropic(sigma)`: divide every component and Jacobian row by a positive sigma.
- `diagonal(sigmas)`: divide each row by its sigma, using vector storage.
- `from_covariance(Sigma)`: require a symmetric positive-definite covariance and
  compute `W=L^-1` with a Cholesky triangular solve. It does not form `Sigma^-1`.
- `sqrt_information(W)`: supply a finite square whitening transform directly.
  Rank-deficient transforms are allowed; observability is the caller's responsibility.

For raw residual `r` and Jacobian `J`, graph evaluation returns `W*r` and `W*J`.
An explicit factor loss then minimizes `0.5*rho(||W*r||²)`, treating, for example,
the u/v pixel residual as one observation. Its Huber/Cauchy scale is in normalized
units. One shared `sqrt(rho')` weight multiplies the entire whitened residual and
all Jacobian rows; the model remains IRLS and omits the `rho''` Hessian term.
The true robust cost is computed separately for LM acceptance and `Report::cost`.

`loss: Some(loss)` **replaces** the global scalar loss for that factor, including
`Some(Loss::Squared)` to keep a prior quadratic. `loss: None` uses the global
per-scalar fallback after whitening. Losses are never applied twice. Explicit
joint loss is invariant to orthogonal changes of whitened residual coordinates;
separate scalar robust losses generally are not.

```rust
use ropt::{factor::{FactorGraph, FactorOptions, VariableKind},
           noise::NoiseModel, solve_local, DVector, DMatrix, Loss, Options};

fn main() -> Result<(), ropt::Error> {
    let mut graph = FactorGraph::new();
    let x = graph.add_variable(VariableKind::Euclidean(1), DVector::zeros(1))?;
    graph.add_analytic_factor_with_options(&[x], 2, FactorOptions {
        noise: NoiseModel::diagonal(DVector::from_vec(vec![1.0, 2.0]))?,
        loss: Some(Loss::Huber(1.0)),
    }, |v| Ok((DVector::from_vec(vec![v[0][0] - 3.0, 2.0*v[0][0] - 6.0]),
                vec![DMatrix::from_vec(2, 1, vec![1.0, 2.0])])))?;
    let report = solve_local(&graph, graph.initial(), &Options::default())?;
    assert!((report.x[0] - 3.0).abs() < 1e-7);
    Ok(())
}
```

Custom `LocalProblem` implementations may supply disjoint `LossBlock` row ranges
through `loss_blocks()`. This fixed metadata is validated and captured once per
solve. Uncovered rows use the scalar fallback. Invalid dimensions, overlapping
ranges, invalid scales, nonfinite transforms and invalid covariance matrices are
rejected. Whitening can couple rows within a factor and change its sparsity;
block Schur still requires independent eliminated variables.

`robust_bundle_adjustment` adds heterogeneous diagonal noise models and two gross
pixel outliers to the 4-camera/12-point fixture. For this deterministic synthetic
case, squared loss gives about 5.04 px RMS on the 46 known inlier observations;
joint Huber gives about 0.70 px. Dense, full normal PCG and block Schur agree.
The metric excludes the two deliberately corrupted observations; the squared and
Huber objective costs have different definitions and should not be compared as
if they were the same objective. This is a regression example, not a general
accuracy claim on real data.

## Streaming estimation

`fixed_lag::FixedLagSmoother` optimizes the current window, then removes variables
with timestamps strictly older than `now-lag`. Factors touching those variables
are replaced by a square-root prior on their live neighbors. Other factors stay
nonlinear and are not duplicated into the prior. `timestamp: None` keeps a
persistent variable such as calibration. Existing timestamps may be refreshed.
Time must be finite and nondecreasing; late factors must reference live keys.
Application keys remain stable across window compaction.
`set_lag()` changes the duration for subsequent updates; expiry is applied only
on a successful update, and increasing the lag does not restore retired states.
Reports return live `estimates` and the last optimized values of removed keys in
`marginalized_estimates`, so applications can [retain finalized trajectory poses](docs/fixed-lag-output.md).
These owned snapshots are not revised by later updates.
`update_with_covariance` optionally returns selected covariance before expiry,
including cross-covariance with retiring variables; its dense SVD cost is explicit.
Inherited robust losses require explicit Frozen-IRLS covariance with the same
loss kind and scale as the window optimizer.

The [adaptive SE(3) example](docs/fixed-lag-output.md#adaptive-window-example)
changes the lag from 2 to 4 to 1 second while preserving covariance through
simultaneous retirement. This figure shows a synthetic functional example.

![Adaptive fixed-lag window sizes and local body-tangent standard deviations](docs/assets/adaptive-fixed-lag.png)

```rust
use ropt::{DMatrix, DVector, factor::{VariableKind, FactorOptions}, fixed_lag::*};
fn main() -> Result<(), ropt::Error> {
    let mut smoother = FixedLagSmoother::new(FixedLagOptions {
        lag: 1.0, ..FixedLagOptions::default()
    })?;
    smoother.update(0.0, SmootherUpdate {
        variables: vec![WindowVariable { key: 0, kind: VariableKind::Euclidean(1),
            value: DVector::zeros(1), timestamp: Some(0.0), fixed: false }],
        factors: vec![SmootherFactor::analytic(&[0], 1, FactorOptions::default(), |v|
            Ok((&v[0] - DVector::from_element(1, 3.0), vec![DMatrix::identity(1, 1)])))],
        ..SmootherUpdate::default()
    })?;
    let report = smoother.update(2.0, SmootherUpdate {
        variables: vec![WindowVariable { key: 1, kind: VariableKind::Euclidean(1),
            value: DVector::zeros(1), timestamp: Some(2.0), fixed: false }],
        factors: vec![SmootherFactor::analytic(&[0, 1], 1, FactorOptions::default(), |v|
            Ok((&v[1] - &v[0] - DVector::from_element(1, 1.0),
                vec![-DMatrix::identity(1, 1), DMatrix::identity(1, 1)])))],
        ..SmootherUpdate::default()
    })?;
    assert_eq!(report.marginalized, vec![0]);
    assert!((smoother.estimate(1).unwrap()[0] - 4.0).abs() < 1e-7);
    Ok(())
}
```

Marginalization uses Householder QR projection and augmented QR compression,
without normal equations or an inverse. Eliminated directions undergo a scaled
SVD rank check; unobservable elimination returns an error. The prior preserves
the profiled **linearized** objective, including constant residual components.
For manifold variables it evaluates `A*Local(anchor,x)+b` with forward-AD chart
derivatives. The chart anchor and information stay frozen; nonlinear/robust
all-history inference is approximate. Robust marginalization requires explicit
`MarginalizationLoss::FrozenIrls`; the resulting prior uses squared loss.
`marginal_covariance` reports the live surrogate model. Default optimization uses
dense QR LM; marginalization is dense and entry-limited. A time window alone
does not bound storage at arbitrary observation rates or with persistent variables.

`isam::IsamSmoother` retains all history and inserts new Gaussian rows into a
cached sparse upper-triangular factor using Givens rotations. Ordinary updates
linearize only new factors and solve that cached model. A tangent displacement
threshold, update interval or explicit refresh runs batch LM, relinearizes all
factors and reconstructs the undamped QR model. This initial iSAM-style engine
offers optional Gaussian Bayes-tree updates; selective nonlinear clique
relinearization, dynamic reordering and row deletion remain missing.
It supports analytic/AD manifold factors and requires an observable Gaussian
model. Cached estimates are approximate between full nonlinear refreshes.
Forced and interval refreshes start from current estimates and new seeds,
without solving first at old anchors. Threshold refreshes start from the cached
solution. Reports expose `relinearization_reason` (Forced/Interval/Threshold);
failures preserve owned smoother state and never silently switch backends.

`IsamSmoother::with_bayes_tree()` opts into fixed-order Gaussian partial
re-elimination. The new rows' touched cliques and ancestor paths are rebuilt;
unaffected conditional branches and symbolic separators stay shared. Reports
expose `bayes_tree_update` with scalar clique counts. Nonlinear refreshes still
rebuild the full model, and back-substitution still visits all columns. See
[Gaussian Bayes-tree validation and measurements](benchmarks/BAYES_TREE.md).
Both Gaussian tree types also support [frozen-factor gradient solves](docs/bayes-tree-gradient.md)
without explicitly forming a normal matrix.

Ordinary updates reuse the new-factor graph's full variable layout and defer
new-recipe commit until success, avoiding old-factor copies/registration.
Unchanged `set_fixed` calls keep the existing tangent layout. An explicitly
selected sparse refresh backend uses its own limits and QR fill bounds;
the default dense-QR refresh retains its matrix entry guard. Mixed-manifold
layout, sparse refresh and failure rollback are covered by regressions. See the
[incremental measurements](benchmarks/ISAM_UPDATES.md).

[Nonlinear stream validation](benchmarks/NONLINEAR_ISAM.md) adds 300 SE2 poses
and 24 non-root loop constraints, including delayed old-to-old factors. Twelve
explicit loop refreshes agree with AD/dense-QR batch references within 8.3e-7
in mixed-unit Local norm. Independent original-energy central differences check
stationarity. Ordinary cached updates remain approximate; these accuracy checks
are not an iSAM2/GTSAM performance comparison.

A separate [native iSAM2 comparison](benchmarks/NATIVE_ISAM2.md) now uses
identical SE2 streams and requires nonlinear stationarity after every arrival.
Sorted-vector sparse QR merges reduce ropt's 300-arrival aggregate median from
3.64 s to 1.38 s (2.63x), with identical means; native GTSAM iSAM2 takes 0.42 s
and remains 3.30x faster in this explicit accuracy-conditioned workload.
These are synthetic update/report timings, not default-policy or general SLAM
claims. Allocation diagnostics and all-prefix quality gates are retained.

The subsequent [model/input reuse change](benchmarks/MODEL_REUSE.md) keeps the
accepted Gaussian residual/Jacobian for QR and reuses callback input storage.
The same-condition 300-arrival median falls from 1.39 s to 1.24 s (11.1% lower),
with identical means. Native iSAM2 takes 0.43 s in that sweep and remains 2.87x
faster. Library tolerances and update policies are unchanged.

An explicit [adaptive LM damping reuse policy](benchmarks/DAMPING_REUSE.md)
now carries the damping from one successful full refresh to the next. In a
separate matched sweep, the 300-arrival median falls from 1.28 s with default
restart damping to 0.89 s with reuse (30.3% lower), passing the same accuracy
gates. Native iSAM2 takes 0.43 s and remains 2.06x faster. Reuse changes the
optimization trajectory and is opt-in; default behavior is unchanged. Failed
refreshes preserve the previous damping along with estimates and history.

```rust
use ropt::isam::{IsamOptions, IsamSmoother};
let smoother = IsamSmoother::new(IsamOptions::default())?
    .with_batch_damping_reuse()?;
assert_eq!(smoother.next_batch_damping(), Some(1e-3));
# Ok::<(), ropt::Error>(())
```

The subsequent [compact symbolic-pattern storage](benchmarks/SYMBOLIC_PATTERN.md)
keeps numerical results and LM trials identical. With damping reuse enabled on
both sides, a separate matched 300-arrival sweep measures 0.847 s before versus
0.817 s after (3.5% lower median); native iSAM2 takes 0.428 s and remains 1.91x
faster. One paired round is slower after the change. Allocation requests/traffic
also decrease, with raw records retained; this is a synthetic workload observation.

The [QR buffer/merge and exact-origin cost improvement](benchmarks/QR_MERGE_BUFFERS.md)
preserves estimates, costs and LM trials. In its own matched 300-arrival sweep,
the median is 0.854 s before versus 0.827 s after (3.2% lower), with 4.9% fewer
allocation requests. Native iSAM2 takes 0.444 s and remains 1.86x faster.
Timing differences are small and one paired round is slower after the change;
the first version and all raw records are retained separately.

The [direct CSR assembly change](benchmarks/CANONICAL_CSR.md) emits sorted
graph rows without intermediate triplet buffers. Its matched 300-arrival median
falls from 0.823 s to 0.761 s (7.6% lower), with identical means, costs and LM
trials. Cumulative requested allocation bytes fall by 30.1%; live/peak memory
was not measured. Native iSAM2 takes 0.431 s and remains 1.77x faster. Raw timing,
allocation and default cached-update validation records are retained.

The [slice-based graph retraction](benchmarks/GRAPH_RETRACTION.md) removes
per-variable temporary vectors while preserving all five manifold operations.
Its own matched 300-arrival median is 0.771 s before versus 0.744 s after (3.5%
lower), with identical estimates, costs and LM trials and 8.4% fewer allocation
requests. At 100 arrivals the median is 0.7% slower; timing variability remains
substantial. Native iSAM2 takes 0.424 s at 300 and remains 1.75x faster. These
synthetic measurements do not establish general performance guarantees.

The [in-place sparse QR update](benchmarks/QR_INPLACE.md) rotates existing
upper columns in their own storage, using the sorted merge for new interior
columns. In its matched 300-arrival sweep the median is 0.772 s before versus
0.702 s after (9.1% lower), with identical estimates, costs and LM trials.
Eleven of twelve paired rounds improve; timing remains variable. At 100 arrivals
the observed reduction is only 1.1%. Native iSAM2 takes 0.430 s at 300 and remains
1.63x faster. Raw comparisons and correctness/allocation diagnostics are retained.

The [matrix-whitening buffer reuse](benchmarks/WHITENING_BUFFERS.md) reuses
product storage within each graph evaluation and swaps results into owned outputs.
Its matched 300-arrival median falls from 0.708 s to 0.673 s (5.0% lower), with
identical estimates, costs and LM trials. Allocation requests fall by 14.5%;
this measures cumulative traffic, not live/peak memory. The 100-arrival median is
essentially unchanged. Native iSAM2 takes 0.434 s at 300 and remains 1.55x faster.
Timing remains variable; both versions and separate raw records are retained.

The [compact ordering adjacency](benchmarks/ORDERING_ADJACENCY.md) uses
sorted storage for small neighborhoods and reuses a neighbor buffer, preserving
greedy elimination order and fill budgets. Its matched 300-arrival median is
0.690 s before versus 0.676 s after (1.9% lower), with identical estimates,
costs and LM trials. Only seven of twelve pairs improve; timing variation is
larger than the observed difference. Allocation requests decrease by 2.6%.
Native iSAM2 takes 0.428 s and remains 1.58x faster. Raw diagnostics are retained.

The [normal-graph rebinding](benchmarks/NORMAL_PATTERN_REBIND.md) checks
whether a changed CSR Jacobian has exactly the same normal graph, then reuses
ordering/fill while refreshing assembly destinations. Its matched 300-arrival
median falls from 0.703 s to 0.650 s (7.6% lower); all twelve pairs improve,
with identical estimates, costs and LM trials. Allocation requests decrease by
7.0%, and full preparations fall from 483 to 300 in a separate diagnostic.
Native iSAM2 takes 0.432 s and remains 1.50x faster. Timing remains variable;
these synthetic results do not establish general SLAM superiority.

The [aligned QR row reuse](benchmarks/QR_ALIGNED.md) checks all live suffix
columns before rotating lower storage directly. Matched medians are 0.640 s
before and 0.627 s after at 300 arrivals (2.0% lower), and 46.50 / 45.04 ms at
100 (3.1% lower). Ten of twelve pairs improve at each size, with identical
estimates, costs and LM trials. Timing variation exceeds these small differences;
a dependable general speedup is not established. Allocation traffic is unchanged.
Native iSAM2 takes 0.438 s at 300 and remains 1.43x faster. Raw records are retained.

The [sorted symbolic fill and CSR-row reuse](benchmarks/SYMBOLIC_MERGE.md)
uses an advancing insertion cursor and copies destination plans for exactly equal
consecutive supports. Matched 300-arrival medians are 0.625 s before and 0.592 s
after (5.3% lower); eleven of twelve pairs improve. Estimates, costs and LM trials
match exactly, while allocation traffic is unchanged. Native iSAM2 takes 0.446 s
and remains 1.33x faster. At 100 arrivals the observed reduction is only 2.2%.
Timing remains variable; the synthetic comparison establishes no general SLAM superiority.

The [bounded QR normalization](benchmarks/QR_NORM.md) uses a squared sum
within safe exponent bounds and standard `hypot` at extreme scales. Its matched
300-arrival median falls from 0.585 s to 0.542 s (7.4% lower); all twelve pairs
improve. At 100, ropt takes 40.74 ms versus native's 43.04 ms in this campaign.
Native remains 1.29x faster at 300 (0.421 s). Full-refresh estimates, original costs
and LM trials match; reported QR costs differ by at most 1.1e-15 and pass unchanged
accuracy gates. Floating cached traces remain validated against AD/dense QR.
These synthetic results establish no general SLAM superiority. A slower small-
workload prefix experiment is preserved separately and was not adopted.

The [Schur update destination cursor](benchmarks/UPDATE_CURSOR.md) reuses
consecutive stored positions and searches only the remaining suffix at gaps.
Its matched 300-arrival median falls from 0.586 s to 0.562 s (4.2% lower), with ten
of twelve pairs faster and identical estimates, costs and LM trials. At 100 the
median is essentially unchanged (0.4% higher). Native takes 0.462 s at 300 and
remains 1.22x faster. A separate classifier identifies 92.1% avoidable full-column
lookups, with unchanged allocation traffic. Timing variation remains substantial;
these synthetic results establish no general SLAM superiority.

Two subsequent [normal-pattern experiments](benchmarks/NORMAL_PATTERN_EXPERIMENTS.md)
show no complete-update benefit at 300 arrivals and are not adopted. Their raw
comparisons are retained, with independent structure/budget regression tests.

The [aligned QR reductions](benchmarks/QR_REDUCTIONS.md) collect finite
checks and zero counts across independent column rotations, enabling compiler
vectorization in this build. Matched 300-arrival medians fall from 0.575 s to
0.529 s (8.0% lower), with eleven of twelve pairs faster and identical estimates,
costs and LM trials. At 100 the median is 1.5% higher; a small-workload improvement
is not demonstrated. Native takes 0.467 s at 300 and remains 1.13x faster.
Allocation traffic is unchanged; timing remains variable and synthetic.

The [contiguous QR column checks](benchmarks/QR_COLUMNS.md) prove exact
range equality from endpoints and length, retaining full interior checks for
gaps. A diagnostic identifies 3.4% fewer index comparisons. Matched 300-arrival
medians are 0.543 / 0.533 s (1.8% lower), with ten of twelve pairs faster and
identical estimates, costs and LM trials. At 100 the median is 1.1% higher.
Timing variation exceeds the small difference; a dependable general speedup is
not established. Native takes 0.438 s at 300 and remains 1.22x faster in this campaign.

The preceding [iSAM local-coordinate reuse](benchmarks/ANCHOR_COORDINATES.md) shares
principal coordinates between threshold, report and cached cost. In the matched
cached-policy 300-arrival comparison, cached medians fall from 46.588 to 35.579 ms
(23.6% lower, twelve of twelve pairs faster); complete update medians fall from
171.097 to 157.977 ms (7.7% lower, eleven pairs faster). Full estimates, reports
and independent accuracy traces match exactly. At 100, cached time is 20.1%
lower, but the small total-time difference remains uncertain. Cached prefixes
remain approximate; these times do not support a native GTSAM speed claim.

The preceding [grouped QR rebuild](benchmarks/QR_REBUILD.md) stably groups accepted
Gaussian rows by their leading column while retaining orthogonal Givens QR. In
the common-accuracy forced-refresh comparison, ropt takes 37.367 ms versus native
GTSAM 4.2.1's 44.257 ms at 100 arrivals (1.18x faster), and 405.413 ms versus
435.659 ms at 300 (1.07x faster), winning all twelve pairs at both sizes. An
independent 300-arrival confirmation gives 417.012 / 461.996 ms (1.11x faster,
eleven of twelve pairs). Every prefix passes the unchanged original-energy,
stationarity, fixed-root and native-pose gates. Complete means and LM sequences
match the predecessor exactly; QR cost differs only by rounding. This meets the
measured goal on these synthetic SE2 streams, not a general SLAM superiority or
default-policy claim. The subsequent editable mode is described below.

The latest [recorded Intel replay](benchmarks/REAL_ISAM2.md) preserves all 943
poses and 1,837 measured constraints, using an inferred pose-ID arrival order.
At the explicitly shared normalized-gradient target, ropt takes **15.009 s versus
18.260 s** for verified native GTSAM 4.2.1 iSAM2 (1.22x); independent confirmation
gives **15.122 / 16.553 s** (1.09x). ropt wins all **12/12 pairs** and every arrival
passes independent original-energy, stationarity, fixed-root and pose-agreement
checks. These are forced full-refresh accuracy-target policies, not default cached
updates. The separately selected 1,228-pose Intel version fails in ropt at arrival
166 under the earlier LDL/LM policy. The new [sparse QR/GN path](benchmarks/SPARSE_QR.md)
completes all 1,227 arrivals and matches native poses. Subsequent
[compensated information whitening](benchmarks/INFORMATION_PRECISION.md) passes
every ropt quality gate; the old native reference still fails reported-cost verification.
The latest [Cartesian SE2 comparison](benchmarks/PRECISE_REAL_ISAM2.md) additionally
corrects native measurement precision and exact root conditioning. Both outputs
pass every original gate: ropt takes **7.947 / 11.032 s** (1.39x), with independent
confirmation **8.488 / 11.284 s** (1.33x), winning **12/12 pairs**. This uses explicit
Cartesian right retraction and ambient analytic evaluation; the default Lie
exponential chart remains available. It is a common-accuracy full-refresh result
against the disclosed corrected reference, not default iSAM2 latency. Historical
failures remain archived. Both Intel variants derive from one recorded SE2 route;
these results do not establish general SLAM dominance.
The additional [CSAIL recording](benchmarks/CSAIL_REAL_ISAM2.md) passes all 1,044
arrivals under the same solver policies and quality gates: **3.388 / 7.755 s**
(2.29x) in primary measurement and **3.385 / 7.730 s** (2.28x) in confirmation,
with **12/12 faster pairs**. CSAIL has no supplied vertices; both solvers use
identical original-odometry seeds, with all original constraints retained.

An optional [threshold-selected factor mode](benchmarks/SELECTIVE_RELINEARIZATION.md)
now moves only selected variable anchors and reevaluates their incident factors.
Enable `.with_selective_relinearization(4)?` before the first update.
`report.selective_update` records retained factors and remaining threshold excess;
Gaussian QR rebuilding is global and a cleared threshold does not certify
nonlinear convergence. Run `cargo run -p ropt --example selective_isam --locked`.
The real-data speed results above retain their measured full-refresh policies.
The additional [editable Bayes-tree mode](benchmarks/EDITABLE_BAYES_TREE.md),
`.with_selective_bayes_tree(4)?`, replaces those selected Gaussian factors through
cached boundary messages. A nonlinear 1,025-variable example re-eliminates two
scalar cliques and retains 1,023, while matching the global-QR selected model.
Symbolic topology is reused for same-support replacements and additions;
metadata bookkeeping and back-substitution remain global. Recorded latency and
accuracy comparisons follow below.
Run `cargo run -p ropt --example selective_bayes_tree --locked`.

The later [recorded stationarity-controlled comparison](benchmarks/STATIONARITY_CONTROL.md)
checks actual nonlinear cost/gradient and explicitly requests selected anchors.
On complete CSAIL, the selected **global-QR** policy takes **3.436 / 3.861 s**
against current forced-full ropt, and **3.491 / 3.883 s** in confirmation:
**1.12x / 1.11x faster, 12/12 pairs**. All original-information gates pass;
full recoveries are counted and timed. The editable-tree policy is slower here.
GTSAM poses/costs are an archived correctness reference; no new native timers
are compared. `nonlinear_summary(norm)` and `relinearize_selected(keys)` expose
the explicit probe/request operations without changing default update policy.

The next [local-clique CSR improvement](benchmarks/LOCAL_CLIQUE_CSR.md) reduces
editable-tree time by **16.0% / 11.2%** on complete CSAIL, winning **12/12
before/after pairs**. All accuracy gates pass; complete means and non-timing
work records remain identical. The improved editable policy still takes
**4.459 / 4.496 s**, versus **3.908 / 3.970 s** for current full refresh.
Rust 313/Python 23 checks pass. This compares ropt kernels; the published native
timings and figures retain their original measured kernels.

The subsequent [metadata-sharing change](benchmarks/EDITABLE_METADATA_SHARING.md)
shares editable factor/ownership maps across snapshots and copies changed maps
before mutation. In a separate 2,048-variable, 64-clone diagnostic, allocation
calls fall **99.1%** and requested bytes **56.4%**, with identical repeated
counts. Complete CSAIL latency varies (**1.021x / 0.991x**, 4/6 / 2/6 faster
pairs); a repeatable stream speedup is not established. All accuracy gates and
exact before/after means/work checks pass. Rust 314/Python 23 checks pass.

The next [incidence-cache improvement](benchmarks/INCIDENCE_CACHE.md) reuses
sorted child/factor-owner lists and patches changed membership. Complete CSAIL
editable time falls **1.9% / 4.5%**, winning **10/12 before/after pairs**, with
identical means and non-timing work. Its **4.281 / 4.296 s** streams still trail
current full refresh at **4.009 / 4.023 s**. Rust 315/Python 23 checks pass.
The modest gain is specific to these ropt policies; native figures remain frozen.

The later [direct analytic-probe improvement](benchmarks/DIRECT_NONLINEAR_PROBE.md)
evaluates all-analytic current-estimate queries without temporary graph/CSR
construction. Complete CSAIL editable time falls **8.6% / 8.9%**, winning
**11/12 before/after pairs**, with identical costs, gradients, poses and work.
At that measured stage, mixed/AD recipes keep the original graph path. Editable streams take
**4.096 / 4.128 s**, versus current full refresh at **3.879 / 3.857 s**; both
current policies use direct probes. Rust 318/Python 23 checks pass. No new
native/GTSAM timing is collected.

The latest [mixed/AD probe extension](benchmarks/MIXED_AUTODIFF_PROBE.md) also
avoids temporary graph/CSR assembly for runtime AD recipes, sharing their tangent
AD evaluator with factor graphs. Across 16 synthetic query conditions, time falls
**11.1–25.8% / 9.6–26.3%** in two campaigns, with matching cost/gradient bits,
estimate/variable digests and callback counts. This measures query latency only.
Complete analytic CSAIL/MIT functional replays retain exact poses/work and pass
all existing gates. Rust **320**/Python **23** checks pass. At that measured stage,
complete AD-stream and new native/GTSAM performance were unmeasured.

The subsequent [complete mixed-AD CSAIL comparison](benchmarks/REAL_AD_PROBE.md)
checks 586 AD / 586 analytic factors through every update and recovery. Median
stream time falls **3.0% / 1.6%**, winning **6/6 / 4/6** pairs. Confirmation misses
the speed criterion, so a repeatable complete-stream speedup is not established.
All **25,056 timed arrivals** pass original-information accuracy and native pose/
cost agreement, with identical before/after complete poses and non-timing work.
Complete mixed-AD MIT regression also passes, including its ill-conditioned edge.
Rust 320 + 2 benchmark tests/Python 23 pass; native timings remain unchanged.

The preceding [owned Jet-buffer optimization](benchmarks/JET_BUFFER_REUSE.md)
reuses consumed derivative vectors without changing Jet's representation.
Complete mixed-AD CSAIL streams improve **1.2% / 4.6%**, winning **5/6 pairs in
both campaigns**, with exact before/after poses, gradients and non-timing work.
Separate AD width diagnostics reduce allocation calls **30.8–43.8%** and requested
bytes **8.7–42.9%**; the 65-wide latency case is **5.5% slower** and remains visible.
All 25,056 timed arrivals and complete MIT functional regression pass unchanged
original-information/native agreement gates. Rust 322 + 2 benchmark tests/Python
23 pass. No fresh native/GTSAM, peak-memory or general speed conclusion is drawn.

The preceding [width-aware owned Jet loops](benchmarks/JET_LOOP_SPLIT.md) retain
indexed arithmetic up to width 16 and split broadcast/equal-length loops above
it. Width-65 AD evaluation time falls **24.5% / 24.6%**, with **6/6 faster pairs
in both diagnostic campaigns**; width 32 falls **16.3% / 18.6%**. Allocation
traffic and all result bits remain unchanged. Small widths have mixed results.
Complete mixed-AD CSAIL primary time falls 1.7% (4/6 pairs), but confirmation
is **2.9% slower (1/6)**: a repeatable complete-stream gain is not established.
All 25,056 timed arrivals and complete MIT functional regression retain exact
before/after poses/work and unchanged original-information/native agreement
gates. Rust 322 + 2 benchmark tests/Python 23 and 40 source-only rollback
fingerprints pass. The unconditional split-loop trial and its negative stream
results remain archived. No fresh native timing or general speed claim is made.

The preceding [validity-preserving Jet negation](benchmarks/JET_NEGATION.md) removes
redundant finite scans specifically for validated sign changes. Isolated width-6
owned negation is **1.51x / 1.59x** as fast, borrowed negation **1.19x / 1.17x**,
with **6/6 faster pairs in both campaigns** for either ownership form and unchanged
bits/allocation work. Complete mixed-AD CSAIL primary improves **5.1% (6/6)**, but
confirmation is **2.0% slower (3/6)**: repeatable complete-stream speedup remains
unestablished. All 25,056 timed arrivals and complete MIT functional regression
retain exact poses/work and unchanged quality gates. Rust 323 + 2 benchmark
tests/Python 23 and 41 source-only rollback stages pass. Three broader validation
proposals were rejected and preserved. Constructors and binary loops retain the
preceding implementation; the gain is specific to negation. Native timings and
the informal 75/100 maturity estimate stay unchanged.

The preceding [streaming compensated probe cost](benchmarks/PROBE_COST.md) removes
the full residual and zero vectors from current-estimate probes while preserving
row-order compensation and callback/error precedence. Successful nonempty probes
remove **two allocation requests and 16 bytes per residual**; 512 three-residual
factors remove **24,576 requested bytes per query**. All analytic/mixed/AD allocation
diagnostics retain exact result bits/work. Complete CSAIL update medians are
**1.4% / 0.6% slower (2/6 and 4/6 faster pairs)**; this is an allocation/workspace
improvement, with no complete-stream speed gain established. All 25,056 timed
arrivals and complete MIT functional regression pass unchanged gates, with exact
before/after poses/work. Rust 325 + 2 benchmark tests/Python 23 and 42 source-only
rollback stages pass. Native figures retain their measured kernels.

The preceding [dense probe indexing](benchmarks/DENSE_PROBE.md) uses one offset-indexed
vector for contiguous variable keys, retaining tree lookup for gapped/partial key
spaces. A successful 512-factor query saves **83 allocation requests / 22,944
requested bytes**, with exact result bits/work across analytic/mixed/AD diagnostics.
Complete CSAIL medians decrease **1.5% / 5.3%**, but only **4/6 pairs** improve in
either campaign: repeatable whole-update speedup remains unestablished. All 25,056
timed arrivals and the complete MIT functional regression retain exact poses/work
and pass unchanged quality gates. Rust 327 + 2 benchmark tests/Python 23 and 43
source-only rollback stages pass. Native timing and the informal 75/100 maturity
estimate stay unchanged.

The preceding [direct AD block Jacobians](benchmarks/BLOCK_JACOBIAN.md) remove the
flat tangent Jacobian and zero input vector before block splitting. A successful
512-factor all-AD query saves **1,024 allocation requests / 53,248 requested bytes**;
analytic-only allocation totals are unchanged. Complete CSAIL primary is **2.3%
slower (3/6 faster pairs)** and confirmation **1.8% faster (4/6)**: repeatable
whole-update speedup remains unestablished. All 25,056 timed arrivals and complete
MIT regression retain exact poses/work and pass unchanged gates. Rust 329 + 2
benchmark tests/Python 23 and 44 source-only rollback stages pass. Native figures
and the informal 75/100 maturity estimate remain unchanged.

The latest [stack AD input wrappers](benchmarks/INLINE_AD_INPUTS.md) remove two
temporary vectors for one-/two-block callbacks while retaining the generic path
for larger factors. A 512-factor all-AD, chunk-8 probe saves **1,024 allocation
requests / 65,536 requested bytes**. **12/16 synthetic probe cases** meet the speed
criterion in both campaigns; ColumnNormalized 2,048-factor all-AD/chunk-1 queries
take **16.8% / 16.3% less time (6/6 faster pairs each)**. Complete CSAIL medians are
1.1% faster / 0.04% slower, with **2/6 faster pairs each**: whole-update speedup
remains unestablished. All 25,056 timed arrivals and complete MIT regression retain
exact poses/work and unchanged gates. Rust 331 + 2 benchmark tests/Python 23 and 45
source-only rollback stages pass. A rejected strided-write candidate is preserved.
Native figures and the informal 75/100 maturity estimate remain unchanged.

The [factor-cache append-reservation candidate](benchmarks/CACHE_RESERVE.md)
saves **one reallocation request / 8,184 requested bytes** when adding one factor
to a 512-factor graph. However, [complete CSAIL timing](benchmarks/CACHE_RESERVE_TIMING.md)
shows **2.0% / 2.0% higher observed stream medians (2/6 / 3/6 faster pairs)**.
Its adoption is withdrawn and the exact preceding production implementation is
restored. All 25,056 timed arrivals retain exact poses/work and unchanged gates;
candidate source, allocation data and both timing campaigns remain available.
Allocation savings do not establish runtime or peak-memory improvement.

[Editable-tree recovery](benchmarks/TREE_REFINEMENT.md) now reuses its triangular
factor for frozen-model refinement, avoiding a separate global QR rebuild.
Complete CSAIL stream medians are **4.0% / 4.2% lower** in two six-pair campaigns
(**5/6 and 6/6** faster pairs). All **25,056 timed arrivals** meet the unchanged
original-information and archived native agreement gates. Complete MIT also
passes the same accuracy gates. Arithmetic order changes trajectories slightly;
this compares preceding ropt kernels, with no fresh GTSAM timing claim.
Arrival latency is mixed: the confirmation maximum rises **73.607 → 81.491 ms**.

[Scheduled factor-cache refresh](benchmarks/CACHE_REFRESH.md) now avoids copying
rows that a forced/interval rebuild immediately discards. A 512-factor forced
refresh saves one more allocation request and **4,096 requested bytes**. Failed
refreshes preserve cached rows and the update schedule in both selective
backends. Complete MIT retains exact poses/work and original gates. Rust **334 +
2 benchmark tests**, Python **23** and **47** source-only rollback stages pass.
The [complete CSAIL timing follow-up](benchmarks/CACHE_REFRESH_TIMING.md) finds
**0.9% less / 0.5% more median stream time (2/6 faster pairs each)**: no speedup is
established. All 25,056 timed arrivals retain exact poses/work and unchanged
gates. Arrival-latency changes are mixed; primary maximum rises **76.144 →
100.977 ms**. This remains an allocation-traffic improvement; native claims stay
unchanged.

[Gaussian factor capture](benchmarks/FACTOR_CAPTURE.md) now copies each factor's
consecutive canonical CSR rows at their required lengths. A 512-factor, 3-row
steady Gaussian refresh saves **2,560 allocation requests / 159,744 requested
bytes**; counts measure traffic rather than peak memory. Complete CSAIL at the
same requested cap 0 has **2.5% / 4.6% lower stream medians (4/6 / 5/6 faster
pairs)**, so repeatable whole-update speedup remains unestablished. Pooled arrival
medians rise slightly; P99 decreases, while maximum latency is mixed. All 25,056
timed arrivals and complete MIT regression retain exact poses/work and unchanged
gates. Rust **333 + 2 benchmark tests**, Python **23**, and **46** source-only
rollback stages pass. Native figures and the informal 75/100 estimate stay frozen.

[Complete-stream phase diagnosis](benchmarks/STREAM_PHASES.md) identifies full
recovery as **38–46%** of the update budget and direct probes as **20–22%** in
CSAIL/MIT. A [same-binary recovery-policy comparison](benchmarks/RECOVERY_CAPS.md)
then tests explicit requested cap 1 versus 0, keeping automatic threshold passes
and the nonlinear quality target. The additional isolated CSAIL campaign takes
**7.2% less total update time (5/6 faster pairs)**. Earlier campaigns take 4.2% /
9.4% less time, with concurrent archival work disclosed. All 37,584 timed arrivals
pass common original-information and archived-native gates. A subsequent
[complete MIT comparison](benchmarks/MIT_RECOVERY_CAPS.md) takes **12.4% / 10.7%
less total update time (6/6 faster pairs each)**; all 29,448 timed arrivals pass
the same gates. MIT pooled median arrival latency rises approximately **1%**,
while P99 falls **68.951 → 57.623 ms / 67.323 → 56.710 ms**. Maximum latency
improves in the primary campaign but rises **123.898 → 131.090 ms** in
confirmation. This is a caller-policy result; library defaults, historical native
timings and informal maturity stay unchanged.

In the isolated policy comparison, pooled median arrival latency rises
**2.061 → 2.193 ms**, while P99 falls **58.311 → 53.513 ms**. The total-stream
improvement therefore has a per-arrival tradeoff; see the retained distributions.

```rust
use ropt::{DMatrix, DVector, factor::{VariableKind, FactorOptions}, fixed_lag::*, isam::*};
fn main() -> Result<(), ropt::Error> {
    let mut smoother = IsamSmoother::new(IsamOptions::default())?;
    let report = smoother.update(SmootherUpdate {
        variables: vec![WindowVariable { key: 0, kind: VariableKind::Euclidean(1),
            value: DVector::zeros(1), timestamp: None, fixed: false }],
        factors: vec![SmootherFactor::analytic(&[0], 1, FactorOptions::default(), |v|
            Ok((v[0].clone(), vec![DMatrix::identity(1, 1)])))],
        ..SmootherUpdate::default()
    }, false)?;
    assert_eq!(report.kind, IsamUpdateKind::Incremental);
    smoother.update(SmootherUpdate::default(), true)?; // explicit nonlinear refresh
    Ok(())
}
```

Both smoothers commit owned state only after a successful update. Nonfinite
inputs, malformed connectivity, rank/budget failures and failed full optimization
leave estimates and registered factors unchanged. Callbacks must be deterministic;
external callback side effects cannot be rolled back. The current implementations
are single-threaded. Fixed-lag currently uses batch window optimization and
does not yet use the incremental QR backend. See the [contracts, validation and examples](docs/validation/streaming.md)
for scope and reproducible synthetic runs; no real-time or speed claim is made.

## Linear solver selection

`Options::linear_solver` selects a backend. `Dense` uses SVD for GN and Cholesky
for LM. `DenseQr(QrOptions)` uses column-pivoted Householder QR for both algorithms.
`SparseCg(CgOptions)` uses scalar Jacobi; `SparseBlockCg(CgOptions)` uses one
Jacobi block per active variable. Both use CSR Jacobians and compute `Jᵀ(Jv)`
without storing a global dense Jacobian or normal matrix. `SparseLdl(SparseLdlOptions)`
assembles and factors the sparse damped normal matrix. These three sparse backends
support LM only: there is no rank-revealing sparse GN solve yet. Factor callbacks still create
small dense local Jacobian blocks, and assembly is rebuilt each linearization.

For QR LM, solve the augmented problem
`[J; sqrt(lambda)*D] * step ≈ [-r; 0]`, where
`D[k,k]=max(||J_col_k||,1)`. Its squared regularization matches the other LM
backends' `lambda*max(diag(JᵀJ),1)`. Column norms use stable hypotenuse accumulation;
`lambda*diag` and the normal matrix are not formed. QR operates after the same
whitening and IRLS weights as all other backends. Step prediction, true objective
gain ratio, damping adaptation and rejection state handling stay in the shared
solver. Direct QR trials report zero `linear_iterations`.

`QrOptions::relative_rank_tolerance` (default 1e-14, positive and below one) tests
pivoted `R` diagonal magnitudes relative to their maximum. This is a QR rank
heuristic, not an SVD condition-number test; coordinate units can affect it.
Unresolved rank returns `RankDeficient { rank, parameters }` with no pseudoinverse
or alternate-backend fallback. GN requires full column rank when a linear step
is attempted. LM tests the **augmented** matrix and can solve rank-deficient or
underdetermined measurement systems while damping is numerically sufficient;
that does not establish observability of the original Jacobian. QR supports the
standalone Euclidean API and manifold `LocalProblem` graphs, but is dense and
allocates the `(m+n)*n` augmented matrix in LM.

```rust
use ropt::{autodiff::{AutoDiff,Jet}, solve, DVector, LinearSolver, QrOptions, Options};
fn main() -> Result<(),ropt::Error> {
    let residual=AutoDiff::<1,1,_>(|x:[Jet<1>;1]|[x[0]-Jet::constant(3.0)]);
    let report=solve(&residual,DVector::zeros(1),&Options {
        linear_solver:LinearSolver::DenseQr(QrOptions::default()),
        ..Options::default()
    })?;
    assert!((report.x[0]-3.0).abs()<1e-7);
    Ok(())
}
```

`ill_conditioned_qr` is a deterministic one-trial regression with nearly parallel
columns and Jacobian condition about 2.45e8. On the pinned cloud toolchain, the
normal-equation LM factorization fails at lambda=1e-20, while augmented QR agrees
with an independent augmented SVD solve. GN SVD and GN QR both resolve the weak
direction: observed solution errors were about 1.6e-9 and 1.7e-8 respectively.
The LM QR error to the undamped truth is about 4.2e-4 because damping changes the
minimizer; this is not a comparison of identical GN/LM objectives. All successful
runs stop at the deliberately chosen one-trial limit, not certified convergence.
Rounding details can vary across platforms. Zero stopping tolerances are used
only to inspect this tiny-residual regression. This is not a timing benchmark,
a claim that QR beats SVD, or a general solver-accuracy guarantee.

`FactorGraph` implements sparse evaluation directly. Custom `LocalProblem`
implementations must implement `evaluate_sparse`; selecting sparse mode on an
unsupported problem returns an error. No silent dense fallback is performed.
The original `Residual`/`solve` adapter remains dense-only.

```rust
use ropt::{factor::{FactorGraph, VariableKind}, sparse::CgOptions,
           solve_local, DVector, DMatrix, LinearSolver, Options};

fn main() -> Result<(), ropt::Error> {
    let mut graph = FactorGraph::new();
    let id = graph.add_variable(VariableKind::Euclidean(1), DVector::zeros(1))?;
    graph.add_analytic_factor(&[id], 1, |v| {
        Ok((v[0].clone() - DVector::from_element(1, 3.0), vec![DMatrix::identity(1, 1)]))
    })?;
    let options = Options {
        linear_solver: LinearSolver::SparseCg(CgOptions::default()),
        ..Options::default()
    };
    let result = solve_local(&graph, graph.initial(), &options)?;
    assert!((result.x[0] - 3.0).abs() < 1e-7);
    Ok(())
}
```

PCG uses a zero initial guess, the selected SPD preconditioner, and a stopping threshold
`max(absolute_tolerance, relative_tolerance * ||rhs||)`. It verifies the true
`rhs - A*x` residual before success. Nonpositive curvature, nonfinite values and
iteration exhaustion produce errors; LM propagates them. `Iteration::linear_iterations`
reports PCG work, and is zero for direct solves. These criteria certify the linear
residual only, not the accuracy of the solution for ill-conditioned systems.

### Sparse LDLᵀ

`SparseLdl` assembles the lower triangle directly from a CSR Jacobian, uses a
deterministic greedy scalar minimum-degree ordering, and performs unpivoted
LDLᵀ followed by triangular solves. `SparseOrdering::Natural` is also available.
It requires `evaluate_sparse` but no variable-block metadata. Whitening, IRLS,
damping and nonlinear step acceptance use the shared LM implementation.

```rust
use ropt::{factor::{FactorGraph, VariableKind}, solve_local, DMatrix, DVector,
           LinearSolver, Options, SparseLdlOptions};
fn main() -> Result<(), ropt::Error> {
    let mut graph = FactorGraph::new();
    let id = graph.add_variable(VariableKind::Euclidean(1), DVector::zeros(1))?;
    graph.add_analytic_factor(&[id], 1, |v| Ok((
        &v[0] - DVector::from_element(1, 3.0), vec![DMatrix::identity(1, 1)]
    )))?;
    let report = solve_local(&graph, graph.initial(), &Options {
        linear_solver: LinearSolver::SparseLdl(SparseLdlOptions::default()),
        ..Options::default()
    })?;
    assert!((report.x[0] - 3.0).abs() < 1e-7);
    Ok(())
}
```

The checked solve verifies `rhs - (JᵀJ + lambda*D²)*step` with the original
matrix-free operator. Defaults are relative tolerance 1e-10 and absolute
tolerance 1e-12. Nonpositive pivots, nonfinite results, excessive residuals and
fill limits return errors. No jitter, inverse or alternate-backend fallback is
used. If the original-system residual is too large, up to three iterative
refinement corrections reuse the same factors and prepared buffers, recomputing
the original operator residual after every correction. The tolerance stays
unchanged; failure after refinement remains an error. `SparseLdlReport` and
`SparseLdlStats` expose `refinement_steps`. Unchecked factor solves do not refine.
`max_entries` bounds lower-triangle assembly and factor fill, including
diagonals and structural zeros; it is not a byte or peak-memory limit. The
default limit is 1,000,000 entries. The standalone `SparseLdlFactor::solve`
only solves its computed factors; use `sparse_direct::solve_normal` for the
original-system residual check used by LM.

LM now retains one `SparseLdlWorkspace` per solve. It reuses symbolic ordering,
fill, continuous numeric buffers and bounded assembly/Schur destination plans
while the exact CSR shape, offsets and indices stay unchanged. Values and
damping are always refactored. Changed CSR patterns undergo a checked normal-graph
comparison: an equal graph reuses ordering/fill with refreshed assembly targets,
while different connectivity triggers fresh preparation. Standalone symbolic
matching stays strict. `analyses()` counts successful pattern processing, including
rebindings. A failed refactor invalidates numeric factors. Optional destination plans share
an additional `max_entries` index budget. If they do not fit, the same arithmetic
uses searches instead; no factor entry is pruned. This limit is not total RAM.

```rust
use ropt::{DVector, sparse::CsrMatrix, sparse_direct::SparseLdlWorkspace};
fn main() -> Result<(), ropt::Error> {
    let j = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 2.0), (1, 1, 3.0)])?;
    let mut workspace = SparseLdlWorkspace::new(Default::default())?;
    for (lambda, rhs) in [(0.1, vec![4.4, 9.9]), (0.2, vec![4.8, 10.8])] {
        let report = workspace.solve(&j, &DVector::from_vec(rhs), lambda)?;
        assert!((report.x - DVector::from_element(2, 1.0)).norm() < 1e-12);
    }
    assert_eq!(workspace.analyses(), 1);
    Ok(())
}
```

`SparseLdlSymbolic` supports explicit reusable analysis and owned factor snapshots.
`refactor` and `solve_factored_into` make no allocations after preparing a stable
pattern. The latter only solves computed factors; checked `solve`/`solve_into`
verify the original residual and still allocate the original operator's diagonal.
Entire LM solves and factor callbacks also allocate.
`SparseOrdering::BlockMinimumDegree { block_size }` orders contiguous equal-size
tangent chunks together. The caller chooses their size, which must divide the
active dimension; this does not infer heterogeneous variable layouts. Scalar
minimum degree remains the default. There is no indefinite pivoting or supernodal
execution; a small residual does not certify forward accuracy on ill-conditioned
normal equations.

SE2 has an explicit `add_relative_pose2_factor_analytic` path, checked against AD
and manifold central differences after correlated whitening.
`PoseGraph2::build_with_jacobians(loss, PoseJacobian::Analytic)` selects it;
`build` continues to use AD. For g2o files, the fast explicit configuration is:

```sh
cargo run -p ropt --example pose_graph_g2o --release --locked -- \
  --input input.g2o --ldl --analytic --block-ordering
```

The [current comparison and reproduction instructions](benchmarks/REUSED_LDL.md)
separate buffer/structure reuse from the analytic derivative choice, with the
same objective and accuracy gates. The [initial LDL report](benchmarks/SPARSE_LDL.md)
is preserved as a historical run. Comparative coverage remains limited to the
30-pose sample and derived 100/300-pose between-factor subgraphs.

### Variable-block Jacobi

`SparseBlockCg` is LM-only and requires both `evaluate_sparse` and
`LocalProblem::tangent_blocks()`. The latter must return an ordered, contiguous,
nonempty-block partition of all active tangent columns. `FactorGraph` supplies
its actual variable layout automatically, excluding fixed variables; a pose uses
six tangent coordinates instead of seven quaternion storage coordinates.
Euclidean calibration/landmark blocks may have different sizes. Custom problems
without block metadata return an error, and invalid partitions are rejected
before iterations. Layout is captured once for the solve.

```rust
use ropt::{factor::{FactorGraph,VariableKind},sparse::CgOptions,
           solve_local,DMatrix,DVector,LinearSolver,Options};
fn main() -> Result<(),ropt::Error> {
    let mut graph=FactorGraph::new();
    let id=graph.add_variable(VariableKind::Euclidean(2),DVector::zeros(2))?;
    graph.add_analytic_factor(&[id],2,|v|Ok((
        &v[0]-DVector::from_vec(vec![1.0,2.0]),vec![DMatrix::identity(2,2)]
    )))?;
    let report=solve_local(&graph,graph.initial(),&Options {
        linear_solver:LinearSolver::SparseBlockCg(CgOptions::default()),
        ..Options::default()
    })?;
    assert!((report.x-DVector::from_vec(vec![1.0,2.0])).norm()<1e-7);
    Ok(())
}
```

`preconditioner::BlockJacobi::from_normal` accumulates only each principal block
of the **whitened, IRLS-weighted** damped normal operator. It factors those blocks
by Cholesky and applies their triangular solves; no inverse or additional jitter
is computed. Within-variable cross terms enter the preconditioner. Between-variable
couplings remain in the full matrix-free operator. With block sizes `b_i`, stored
factor entries total `sum(b_i²)`, plus layout/scratch storage. A user-specified large
Euclidean block can still require a large dense factor; this is not linear storage
independently of block size.

`sparse::conjugate_gradient_preconditioned` accepts any implementation of the
`Preconditioner` trait. Its `apply` returns `M^-1*r`; the implementation must
provide a fixed symmetric positive-definite M for each PCG solve. Invalid output
sizes, nonfinite values, nonpositive preconditioned inner products and apply errors
are propagated. The original `conjugate_gradient` still selects scalar Jacobi.
Both paths retain the checked true linear residual and restart when a recursive
residual was too optimistic. Neither path certifies global SPD for arbitrary
caller-supplied operators. Block Cholesky failure or PCG failure aborts the solve;
no scalar/dense fallback is selected. Block Schur's reduced solver currently retains
scalar Jacobi and does not automatically use the full-variable preconditioner.

`block_jacobi` compares the same GNSS fixture with the same initial state, LM
options, CG tolerances and loss. The pinned cloud run observed:

| GNSS loss | Scalar Jacobi PCG iterations | Block Jacobi PCG iterations | Nonlinear trials, both |
| --- | ---: | ---: | ---: |
| Squared | 7,438 | 7,116 | 39 |
| Joint Huber | 1,367 | 1,306 | 7 |

Final costs, body-position RMSE and trajectories agree. The 35 active SE3 blocks
store 1,260 factor entries versus 44,100 entries in a hypothetical full 210x210
normal matrix; this is not a total-memory measurement. Counts depend on the
problem and tolerances and may change with floating-point behavior. Factoring and
applying blocks adds work, so fewer PCG iterations **do not establish a runtime
improvement**. This is a deterministic regression and iteration comparison,
not a timing benchmark or a performance claim against other OSS.

`DenseSchur { retained: k }` retains the first `k` active tangent columns and
eliminates the trailing columns. It is LM-only, with `0 < k < tangent_dimension`.
It factors the eliminated block, solves the reduced system, then back-substitutes;
no explicit inverse is computed. It still allocates dense matrices and is a
reference for later sparse block elimination. `sparse::schur_solve` exposes the
standalone SPD solve. The SE3 example runs both dense and sparse LM.

## 2D pose graphs and g2o files

`pose_graph::RelativePose2` uses the translation and principal angle of
`Z^-1 * (Xi^-1 * Xj)`, matching g2o SE2 edge coordinates. This differs from a
Lie Log residual; supply information matrices in the documented coordinates.
`add_relative_pose2_factor` uses forward AD of right SE2 retractions and accepts
measurement whitening plus an optional joint robust loss through `FactorOptions`.
`NoiseModel::from_information` takes finite symmetric positive-definite information
and uses the **transpose** of its Cholesky factor to preserve `r^T*Omega*r`.

`g2o::PoseGraph2::from_reader` accepts `VERTEX_SE2`, `EDGE_SE2`, `FIX`, blank lines
and `#` comments. `VERTEX2`/`EDGE2` are aliases using the same g2o information
order (`Ixx Ixy Ixt Iyy Iyt Itt`); TORO matrix order is not guessed. It supports
forward references and noncontiguous nonnegative IDs. Duplicate vertices,
unknown references, nonfinite values, non-SPD information, malformed fields and
unsupported records return errors with input line numbers. 3D/landmark records
are currently unsupported and are never silently ignored.

`build` honors all `FIX` records and anchors the smallest ID in each component
that has no explicit fixed vertex. Isolated vertices are rejected. Poses use
three-dimensional right tangents; the parser's world coordinates are not Exp
coordinates. `write` preserves original IDs, edges, information and explicit
`FIX` records, using modern tags. Automatic gauge anchors are solver policy and
are not added to exported data. Graphs with all vertices fixed can be evaluated,
but `solve_local` requires active tangent variables.

```rust
use ropt::{g2o::PoseGraph2, solve_local, Options};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = b"VERTEX_SE2 10 0 0 0\nVERTEX_SE2 40 1.2 0.2 0.1\nEDGE_SE2 10 40 1 0 0 10 0 0 10 0 10\n";
    let data = PoseGraph2::from_reader(input.as_slice())?;
    let loaded = data.build(None)?;
    let report = solve_local(&loaded.graph, loaded.graph.initial(), &Options::default())?;
    assert!(report.cost < 1e-12);
    Ok(())
}
```

The `pose_graph_g2o` example defaults to sparse LM; `--dense` selects the dense
reference, `--huber scale` applies joint loss to each whitened edge, and
`--max-iterations N` limits nonlinear trials. `--cost-tolerance C` explicitly
sets the relative cost-change stopping threshold (default 1e-12).
`--output new.g2o` writes optimized
poses only when the destination does not exist. It prints the actual termination
reason and exits with an error for `NoProgress` or an iteration limit.

A public GTSAM sample with 30 poses and 59 constraints reduced cost from
10.720255 to 0.082908 in seven LM trials. Dense/sparse results agree; an independent
Python checker verifies the original objective and fixed anchor. The exact input
revision, checksum, options, observed results and reproduction commands are in
[the validation note](docs/validation/pose_graph_2d.md). This is one public sample
and does not establish large-scale performance or real-robot readiness.

### Release timings and independent quality gates

[The benchmark workflow](benchmarks/README.md) measures four ropt backends and
an optional native GTSAM 4.2.1 baseline on the same pinned sample. It saves every
timing sample, solver options, independent objective/gradient checks and machine,
version and source fingerprints. Parsing and graph construction are untimed;
forward AD, whitening, assembly, preconditioning and nonlinear trials are part
of the measured solve call. CI uses an offline synthetic-loop smoke check and
does not assert timing thresholds or require GTSAM.

In the recorded release run (five warmups and 30 samples per backend), median
solve times were 3.4422 ms for dense Cholesky, 11.5485 ms for QR, 4.3528 ms for
scalar CG and 5.0173 ms for block CG. All four pass the common quality gate.
Block CG takes 406 iterations versus scalar CG's 443, but has a higher median
runtime here. These shared-cloud, small-problem measurements do not establish
scalability or a general backend ranking; memory was not measured.

The pinned GTSAM native baseline matches the residual coordinates and objective,
but its final independent gradient infinity norm is 3.5769e-3, above the 1e-5
quality threshold. Its factor linearization also differs from central differences
in off-solution probes. The benchmark preserves this result and marks it
ineligible for an equal-accuracy cross-library speed comparison; the full runner
exits with an error after saving the report. No superiority over other OSS is
claimed. Details, limitations and raw data are in the benchmark workflow.

The later [exact-Jacobian native GTSAM comparison](benchmarks/gtsam_exact/README.md)
enables GTSAM's full BetweenFactor chain rule and passes the unchanged common
accuracy gates at 30, 100 and 300 poses. The larger cases are explicitly derived
between-factor subgraphs with documented equivalence-constraint exclusions and
TORO information conversion. This sweep records 120 solve timings, with stricter
CG residual tolerances explicitly set in the comparison runner. At 300 poses,
median times are 2033.51 ms for dense, 708.76 ms for scalar CG, 795.29 ms for
block CG and 98.60 ms for native GTSAM. These configured runs expose a performance
gap; they do not establish a general ranking. Input/quality/build details and
all raw samples are saved. Peak memory and larger complete datasets remain work.

[PCG workspace reuse](benchmarks/BUFFERED_PCG.md) adds compatible `apply_into`
methods, reuses linear-solve buffers, fuses the normal operator row by row and
solves Jacobi blocks in place. An alternating before/after comparison on the
300-pose derived problem reduces scalar-CG median time from 731.79 to 413.99 ms
and block-CG time from 822.27 to 414.91 ms, with unchanged tolerances, iteration
counts and accuracy gates. Block-CG allocation requests drop from 6,857,251 to
1,361,691; this measures allocation traffic, not peak memory. Prepared built-in
operator/preconditioner applications allocate nothing, while whole solves and
legacy custom implementations still can. An updated native-reference sweep and
all raw data are saved; GTSAM remains faster in those configurations.

## Block Schur and Bundle Adjustment

`BlockSchur { retained, block_size, cg }` selects CSR-based Schur elimination for
LM. The active tangent vector must contain a retained prefix, followed by equally
sized independent eliminated blocks. `FactorGraph::schur_layout(&landmark_ids)`
checks that ordering, ignores fixed IDs, checks equal sizes and rejects factors
that connect multiple active eliminated variables. Register cameras before points.
A residual row coupling different eliminated blocks is also rejected by the
linear backend; unsupported coupling is never discarded.

Each eliminated block is factored with a small dense Cholesky decomposition.
Retained Jacobian entries and cross-block rows are stored sparsely. The reduced
operator applies `A*v - B*C^-1*Bᵀ*v` through block solves; neither the full normal
matrix nor the reduced Schur matrix is stored densely. Back-substitution recovers
landmark increments. The final linear residual is checked against the **full**
damped normal system; `LinearResidualTooLarge` reports a failed accuracy check.
Jacobi preconditioning and cancellation in Schur reduction can limit difficult
problems; no large-scale performance advantage has been demonstrated.

```rust
use ropt::{factor::{FactorGraph, VariableId}, sparse::CgOptions,
           solve_local, LinearSolver, Options, Report, Error};
fn optimize_ba(graph: &FactorGraph, points: &[VariableId]) -> Result<Report, Error> {
    let (retained, block_size) = graph.schur_layout(points)?;
    let options = Options {
        linear_solver: LinearSolver::BlockSchur {
            retained, block_size, cg: CgOptions::default()
        },
        ..Options::default()
    };
    solve_local(graph, graph.initial(), &options)
}
```

`camera::Pinhole` uses fixed `fx, fy, cx, cy`. Poses are camera-to-world SE3;
landmarks are Euclidean world coordinates. `camera::add_reprojection_factor`
registers analytic derivatives with respect to right pose increments and world
point coordinates. `camera::add_autodiff_reprojection_factor` generates the same
local Jacobians by forward AD; its `_with_options` variant adds whitening and a
joint pixel loss. `Pinhole::reprojection_autodiff` accepts ambient jet-valued pose
and landmark slices for custom factors. The residual is prediction minus measured pixel coordinates.
It does not select numerical differentiation or optimize intrinsics/distortion.
Points behind the camera or within 1e-8 of its projection plane cause evaluation
errors. Such invalid trial evaluations abort the solve; cheirality-aware trial
rejection and global initialization are not implemented.

`bundle_adjustment` builds 4 cameras, 12 points and 48 observations with all points
visible. Two known camera poses are fixed to anchor the world frame and metric
scale. It jointly refines the other two poses and all points. Run with `--autodiff`
to generate Jacobians through SE3 retractions instead of the analytic factor.
Both dense and block
Schur LM reduce initial 2D reprojection RMS of about 5.85 pixels below 1e-7 and
recover the deterministic noise-free state within 1e-6 in regression tests.
A separate deterministic 0.2-pixel-amplitude noise test with Huber loss checks
error reduction and dense/Schur agreement; it does not require exact ground-truth
recovery in the presence of noise. These are small synthetic checks, not validation
on a public BA dataset or a production SLAM pipeline.

## Shared calibration and image distortion

`camera::BrownConrady` adds two radial coefficients `k1,k2` and two tangential
coefficients `p1,p2` to `Pinhole`. For normalized camera coordinates `(x,y)`,
`r2=x*x+y*y` and `s=1+k1*r2+k2*r2*r2`:

```text
xd = x*s + 2*p1*x*y + p2*(r2 + 2*x*x)
yd = y*s + p1*(r2 + 2*y*y) + 2*p2*x*y
pixel = (fx*xd + cx, fy*yd + cy)
```

`project` and `reprojection` evaluate this forward model in f64.
`parameters()` returns `[ln(fx),ln(fy),cx,cy,k1,k2,p1,p2]`; `from_parameters`
decodes this eight-dimensional Euclidean storage. Focal lengths remain positive
when their exponentials are representable; overflow or underflow is an error.
There is no skew, third radial coefficient, fisheye model or inverse undistortion.
Finite coefficients are accepted without guaranteeing that the distortion map
is globally invertible.

`add_calibration_reprojection_factor` connects one SE3 camera-to-world pose,
one Euclidean(3) world point and one Euclidean(8) calibration block. It generates
all 17 local Jacobian columns by forward AD, and accepts whitening/joint loss
through `FactorOptions`. Multiple factors can share a calibration ID; multiple
cameras can use distinct IDs. Freeze the calibration block with `set_fixed`
to use the same factor with a fixed distorted camera. Calibration domains are
checked during evaluation. Invalid focal values or camera depth abort the solve,
as with the fixed pinhole path. Only the whole block can be fixed through this
helper; subset parameter constraints require a custom factor/block layout.

Register calibration and poses **before** landmarks when using block Schur.
Calibration belongs to the retained prefix, so sharing it across observations
does not violate the independent landmark-block assumption.
`brown_conrady_reprojection_autodiff` exposes the parameterized jet residual for
custom factors; `BrownConrady::reprojection_autodiff` uses fixed calibration jets.

```rust
use nalgebra::Vector3;
use ropt::{camera::{BrownConrady, Pinhole, add_calibration_reprojection_factor},
           factor::{FactorGraph, FactorOptions, VariableKind, se3_values},
           manifold::{SE3, Tangent6}, DVector, LocalProblem};
fn main() -> Result<(), ropt::Error> {
    let model = BrownConrady::new(Pinhole::new(420.0,415.0,320.0,240.0)?,
                                 [-0.12,0.035,0.004,-0.003])?;
    let point = Vector3::new(0.8,-0.6,4.0);
    let observation = model.project(point)?;
    let mut graph = FactorGraph::new();
    let calibration = graph.add_variable(VariableKind::Euclidean(8), model.parameters()?)?;
    let pose = graph.add_variable(VariableKind::SE3, se3_values(SE3::exp(Tangent6::zeros())))?;
    let landmark = graph.add_variable(VariableKind::Euclidean(3),
                                      DVector::from_column_slice(point.as_slice()))?;
    add_calibration_reprojection_factor(&mut graph, pose, landmark, calibration,
                                        observation, FactorOptions::default())?;
    assert_eq!(graph.dimensions(), (18,17,2));
    Ok(())
}
```

This illustrates factor registration; optimization needs sufficient constraints
and anchors. Shared intrinsics and scene geometry can be ambiguous in degenerate
configurations. `calibration_bundle_adjustment` supplies six diverse camera poses
(three known metric poses are fixed), 25 noncoplanar points and 150 noise-free
observations. It jointly estimates the shared eight calibration parameters,
three remaining poses and all landmarks, using 101 active tangent coordinates.
Dense Cholesky, dense QR and block Schur LM reduce reprojection RMS from 14.393076 pixels to about
3.8e-10 in eight trials and recover the selected synthetic truth. Tests also
compare full PCG and a deterministic 0.2-pixel-amplitude noise case; noisy data
are checked for fit improvement/backend agreement rather than exact truth recovery.
No noisy physical calibration accuracy or real-camera validity is claimed.

## Cartesian GNSS and odometry refinement

`navigation::Position3` predicts `t_world_body + R_world_body*lever_body` for a
known antenna lever arm in body coordinates. Observations and their covariance
must already be in a common metric Cartesian world frame (for example local ENU).
The residual is predicted antenna position minus measured position, in world
coordinates. Its analytic right/body tangent Jacobian is `[R, -R*[lever]_x]`;
rotation columns vanish when the lever arm is zero. `residual_autodiff` exposes
the same model for custom AD factors. `add_position3_factor` applies the usual
`FactorOptions` whitening and optional joint robust loss to the full position fix.

```rust
use nalgebra::Vector3;
use ropt::{navigation::{Position3, add_position3_factor},
           factor::{FactorGraph, FactorOptions, VariableKind, se3_values},
           manifold::{SE3, Tangent6}, noise::NoiseModel, Loss};
fn main() -> Result<(), ropt::Error> {
    let lever = Vector3::new(0.3,-0.12,0.8);
    let observed = Vector3::new(1.0,2.0,3.0);
    let model = Position3::new(observed,lever)?;
    let pose = SE3::exp(Tangent6::zeros());
    assert_eq!(model.evaluate(pose)?.residual, lever-observed);
    let mut graph = FactorGraph::new();
    let id = graph.add_variable(VariableKind::SE3,se3_values(pose))?;
    add_position3_factor(&mut graph,id,observed,lever,FactorOptions {
        noise: NoiseModel::diagonal(ropt::DVector::from_vec(vec![0.12,0.12,0.2]))?,
        loss: Some(Loss::Huber(2.5)),
    })?;
    Ok(())
}
```

One position observation supplies three constraints and does not fully constrain
an SE3 pose. This snippet registers a factor; trajectory estimation also needs
sufficient relative/orientation constraints and an appropriate gauge policy.
If the world frame changes, rotate the position covariance along with the
observations and poses. The lever arm remains in body coordinates. Rotating just
the observations changes the statistical problem.

`pose_graph::RelativePose3` uses the full
`Log(Z^-1 * (Xi^-1 * Xj))` residual, ordered translation then rotation, and
`add_relative_pose3_factor` generates its twelve local columns by forward AD.
Its covariance is in that measurement Lie tangent, unlike the world-frame
position covariance. Translation includes the inverse SO3 left Jacobian; this
is distinct from merely subtracting positions or using g2o quaternion-vector
coordinates. Small-angle branches keep derivatives finite at identity, but the
principal rotation Log retains its pi cut. `pose_graph_3d` now uses this AD factor
instead of its previous explicit numerical differentiation.

`gnss_odometry` refines 36 synthetic SE3 poses using 35 biased relative
measurements and nine position fixes, including two gross outliers. It fixes a
known initial metric pose, uses a known `(0.3,-0.12,0.8)` metre antenna offset,
and applies correlated Cartesian covariance to fixes. Odometry factors retain
squared loss; the optional Huber loss applies jointly to each whitened GNSS
three-vector. Its 2.5 threshold is in normalized units, not metres.

| Model | Body-position RMSE, all 36 times | Endpoint error |
| --- | ---: | ---: |
| Odometry only | 0.341882 m | 0.635503 m |
| GNSS with squared loss | 0.738878 m | 0.654683 m |
| GNSS with joint Huber | 0.091894 m | 0.105521 m |

These errors use the synthetic ground truth, including corrupted observation
times. Costs from different loss functions are not used as an accuracy comparison.
Dense and sparse LM agree on this fixture. This is one deterministic offline
position-fusion example, not a GNSS/INS estimator or field validation. It does
not convert latitude/longitude, process satellite pseudoranges, model clock
biases, synchronize sensors, estimate lever arms or preintegrate IMU data.

## Local marginal covariance

`FactorGraph::marginal_covariance(x, ids, options)` returns covariance blocks for
selected **active right-tangent variables**, in the order of `ids` and each
variable's tangent coordinates. Other active variables remain uncertain: the
result is a marginal block of the full information inverse, not the inverse of
just the selected Jacobian columns. Fixed variables are conditioned constants;
querying them returns an error. `tangent_range(id)` returns their current active
column range, or `None` for fixed variables. Ranges change after `set_fixed`.

`covariance::marginal_covariance` accepts any `LocalProblem` and a list of active
column indices. `marginal_covariance_euclidean` adapts the standalone `Residual`
API used by `solve`. Duplicate or out-of-range queries are errors. These APIs
evaluate the supplied point; they do not certify that optimization converged.

```rust
use ropt::{covariance::CovarianceOptions, factor::{FactorGraph,FactorOptions,VariableKind},
           noise::NoiseModel, solve_local, DMatrix, DVector, Options};
fn main() -> Result<(), ropt::Error> {
    let mut graph=FactorGraph::new();
    let id=graph.add_variable(VariableKind::Euclidean(2),DVector::from_vec(vec![1.0,-2.0]))?;
    graph.add_analytic_factor_with_options(&[id],2,FactorOptions {
        noise:NoiseModel::diagonal(DVector::from_vec(vec![0.2,0.3]))?,
        ..FactorOptions::default()
    },|v|Ok((v[0].clone(),vec![DMatrix::identity(2,2)])))?;
    let report=solve_local(&graph,graph.initial(),&Options::default())?;
    let covariance=graph.marginal_covariance(&report.x,&[id],&CovarianceOptions::default())?;
    assert!((covariance.matrix[(0,0)]-0.04).abs()<1e-12);
    assert!((covariance.matrix[(1,1)]-0.09).abs()<1e-12);
    Ok(())
}
```

Default `CovarianceModel::Gaussian` uses `C=variance_scale*(JᵀJ)^-1` from the
already whitened local Jacobian. It rejects robust factor losses. With robust
factors, select `FrozenIrls { fallback_loss }` explicitly: it uses the same joint
factor weights and scalar fallback as the solver, held fixed at `x`. Match the
fallback loss to `Options::loss` from the solve; a problem does not store its
solver options. Frozen-IRLS information is a **local approximation**, not a
robust sandwich covariance, exact robust Hessian or calibrated confidence interval.
Neither model captures unmodeled biases or guarantees nonlinear/global accuracy.

The SVD uses all active columns and refuses a globally rank-deficient problem,
even if the selected coordinates alone are observable. Values at or below
`relative_rank_tolerance*largest_singular_value` (default 1e-12) produce
`Error::RankDeficient { rank, parameters }`. Coordinate units and conditioning
affect this threshold; inspect the reported Jacobian `condition_number`.
No LM damping, pseudoinverse or implicit gauge prior is added. `variance_scale`
is an explicit positive multiplier, defaulting to 1 for known measurement noise;
residual sums of squares and degrees of freedom are not used automatically.
SVD iterations are bounded by `svd_max_iterations` (default 10,000); decomposition
failure returns `LinearSolve`, while nonfinite arithmetic returns `NonFinite`.

Only requested covariance entries are formed from the SVD basis. The implementation
still stores a dense `m*n` Jacobian and `n*n` right-singular basis, so requesting a
small block does **not** make this a large-scale sparse algorithm.
`CovarianceInput::SparseToDense` explicitly converts CSR input to the same dense
SVD; it has no implicit dense evaluation fallback. Sparse selected-inverse methods
and sparse covariance factorization remain future work.

For SE3, each covariance block is 6x6 in body translation/rotation increments,
not 7x7 quaternion storage. A world body-position covariance is
`R*C_translation*Rᵀ`; an antenna-position covariance needs the full propagation
`G*C_pose*Gᵀ`, with `G=[R,-R*[lever]_x]`, including translation/rotation cross terms.
`gnss_odometry` now reports local-model endpoint world-position standard deviations
in addition to ground-truth errors. For its squared GNSS case they are about
`[0.0894,0.1050,0.1472]` m; for Huber the frozen-IRLS values are about
`[0.0915,0.1065,0.1499]` m. These model scales do not validate coverage or substitute
for the separately reported trajectory error.

## Validation and maturity

Tests cover Rosenbrock convergence and rejected-state preservation, linear GN,
rank deficiency, Huber/Cauchy outliers, derivative agreement, invalid inputs,
termination reasons and Lie group roundtrips near zero and pi. Graph tests cover
fixed blocks, analytic SO3 tangent Jacobians versus manifold central differences,
SE3 chain/loop convergence, quaternion normalization and invalid graph inputs.
Sparse tests check CSR products and duplicate handling, PCG against Cholesky,
true residuals, non-SPD failures, iteration limits, robust LM and rejection state,
Schur against full solves, dense/sparse agreement and a 300-variable anchored chain
with 597 Jacobian nonzeros (versus 299*299 dense entries). Storage counts are not
runtime or overall-memory benchmarks. Block Schur tests independently compare
the reduced operator, diagonal, right-hand side, back-substitution and full-system
residual with dense elimination; projection derivatives are checked against known
signs and manifold central differences. g2o tests validate the parser, measurement
coordinates, information energy and gradients, component anchoring, synthetic
loop recovery and serialization. BA tests check noiseless recovery, fixed
anchors, noisy robust optimization and explicit linear-solver failure propagation.
Measurement tests check correlated covariance against Mahalanobis energy and
normal equations, analytic/numeric whitened derivatives, joint-loss gradients and
LM gain ratios against the true objective, scalar fallback coverage, rotational
invariance, mixed prior/measurement losses and outlier BA across all backends.
Runtime AD tests compare fixed jets, analytic formulas and central differences;
check complete column coverage across chunks, constants, dimension/domain errors,
non-deterministic callback rejection and selected extreme-scale derivative cases;
and solve Rosenbrock and whitened Euclidean AD factors across backends.
Manifold AD tests compare retraction values and derivatives with f64 retractions
and central differences at zero, near zero, and nonzero increments, including
nonzero translation with zero rotation. SE2 Exp uses a half-angle identity to
avoid cancellation at the small-angle branch boundary. They verify
mixed-block tangent dimensions, quaternion sign invariance of reprojection,
analytic/AD/numeric agreement with correlated whitening, fixed-pose activation,
and noisy/noiseless BA agreement across dense, full PCG and block Schur LM.
Calibration tests check known distorted pixels, zero-distortion agreement with
analytic pinhole derivatives, analytic intrinsic derivatives in log-focal storage,
all 17 local columns against whitened central differences, domain errors, fixed
calibration activation and shared-calibration BA across Cholesky, QR, scalar-PCG,
block-PCG, sparse LDLᵀ and block Schur. Sparse-direct tests compare both orders
with dense Cholesky, repeated right-hand sides and sparse-only custom problems;
check singular/nonfinite systems, true residuals and assembly/fill budgets;
and verify fixed variables and correlated robust LM. Block preconditioner tests independently compare local
principal solves with dense information, check cross-block operator preservation,
true residuals, one-step exact block solves, malformed metadata, failed Cholesky,
custom preconditioner errors, rejected states and iteration exhaustion.
QR tests compare rectangular solves with SVD, undo multiple column pivots, check
rank-deficient/underdetermined GN and damped LM, stable large/small scales,
near-parallel columns, regularization floors, true robust gain ratios, Rosenbrock
rejections and preserved rejected states.
Navigation tests check lever-arm Jacobian signs, analytic/AD/central derivative
agreement with correlated covariance, invariance of whitened energy and body
normal equations under world-frame transforms, SE3 Log derivatives at identity,
near zero and near pi, quaternion sign invariance, anchored loop recovery and
GNSS outlier trajectory agreement across dense and sparse LM.
Covariance tests recover known correlated noise, selected cross-covariance and
nuisance-variable marginals; check fixed-variable conditioning, changed layouts,
rank failure in unqueried directions and unanchored SE3 graphs, six-dimensional
tangent output, explicit joint/scalar IRLS weights, variance scaling, large/small
Jacobian scales, invalid metadata and bounded decomposition failure.

Our informal maturity estimate measures engineering progress, not a percentage
of another project's speed, feature count or reliability. Comparison target:
GTSAM/Ceres/g2o capability and operational maturity. Suggested rubric: solver
core 25%, differentiation 15%, manifolds/robotics 20%, sparse/scalability 15%,
validation/performance 15%, documentation/maintenance/field evidence 10%.
Current estimate: core 17 + differentiation 12 + manifolds/robotics 16 +
sparse/scalability 14 + validation 14 + documentation 2 = 75/100. The latest
increase from 74 reflects incremental layout reuse, sparse refresh safety and
independent validation of 1,000-update Gaussian streams. The estimate stays at
75 after scheduled-refresh recovery, nonlinear SE2 accuracy validation, sorted
QR merges, native iSAM2 comparison, model/input reuse and explicit adaptive LM
damping reuse, compact symbolic preparation, QR processing, direct CSR assembly,
slice-based graph retraction, in-place QR row updates, whitening buffer reuse,
compact ordering adjacency, normal-graph rebinding, aligned QR row reuse and
sorted symbolic fill/CSR-row planning, bounded QR normalization and Schur-update
destination cursors.
These improve existing contracts and performance. Recorded selected-update
latency now has a common-accuracy comparison; field evidence remains missing.
The complete CSAIL recording strengthens independent real-data validation;
the informal maturity estimate remains **75/100**.
Threshold-selected nonlinear factor evaluation now has an optional editable
Gaussian Bayes tree with cached separator messages and affected clique
replacement. Dynamic reordering, wildfire back-substitution and incremental
fixed-lag removal remain open.
Earlier improvements include stable true-loss reductions, 3,000-pose accuracy
validation, reusable sparse LDL, bounded
plans, block ordering and checked analytic SE2 derivatives. These measurements
do not establish general superiority over other OSS. This subjective estimate is not a measured comparison. Mature
robotics applications need substantially more work.

The [large-graph validation](benchmarks/LARGE_POSE_GRAPH.md) now covers a
1,000-pose derived graph with all common accuracy gates passing, and fixes a
3,000-pose original-system residual failure through iterative refinement.
The subsequent [stable-objective change](benchmarks/STABLE_OBJECTIVE.md) resolves
that 3,000-pose accuracy failure with explicit cost tolerance 1e-15. All 16
comparison samples pass the unchanged gates: median complete solves are 3.10 s
for ropt versus 12.50 s for native GTSAM Cholesky on this derived graph. The prior
failed case is preserved. Complete original datasets, other workloads and field
reliability remain outside this validation.

## License

Dual licensed under [MIT](./LICENSE-MIT) or [Apache-2.0](./LICENSE-APACHE).
