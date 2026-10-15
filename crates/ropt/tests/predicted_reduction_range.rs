use ropt::{factor::*, *};
use ropt::{sparse::CsrMatrix, sparse_direct::*};

fn graph(
    rows: usize,
    base: f64,
    actual_slope: f64,
    model_slope: f64,
    loss: Loss,
    joint: bool,
) -> FactorGraph {
    let mut graph = FactorGraph::new();
    let id = graph
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    graph
        .add_analytic_factor_with_options(
            &[id],
            rows,
            FactorOptions {
                loss: joint.then_some(loss),
                ..Default::default()
            },
            move |values| {
                Ok((
                    DVector::from_element(rows, base + actual_slope * values[0][0]),
                    vec![DMatrix::from_element(rows, 1, model_slope)],
                ))
            },
        )
        .unwrap();
    graph
}

fn lm_backends() -> [LinearSolver; 4] {
    [
        LinearSolver::Dense,
        LinearSolver::DenseQr(Default::default()),
        LinearSolver::SparseLdl(Default::default()),
        LinearSolver::SparseQr(Default::default()),
    ]
}

#[test]
fn lm_accepts_finite_gaussian_improvement_after_intermediate_overflow() {
    let damping = 1e-3;
    let remaining = damping / (1. + damping);
    for joint in [false, true] {
        let graph = graph(2, 1e154, 1e153, 1e153, Loss::Squared, joint);
        for linear_solver in lm_backends() {
            let report = solve_local(
                &graph,
                graph.initial(),
                &Options {
                    linear_solver,
                    initial_damping: damping,
                    max_iterations: 1,
                    ..Default::default()
                },
            )
            .unwrap_or_else(|error| panic!("{linear_solver:?}: {error:?}"));
            let iteration = &report.iterations[0];
            assert!(iteration.accepted && iteration.gain_ratio.is_finite());
            assert!((report.x[0] / (-10. / (1. + damping)) - 1.).abs() < 1e-12);
            assert!((report.initial_cost / 1e308 - 1.).abs() < 1e-12);
            assert!(
                (iteration.actual_reduction / 1e308 - (1. - remaining * remaining)).abs() < 1e-12
            );
            assert!((iteration.gain_ratio - 1.).abs() < 1e-12);
            assert!((report.cost / 1e308 - remaining * remaining).abs() < 1e-12);
        }
    }
}

#[test]
fn gauss_newton_uses_a_finite_model_prediction_at_large_scales() {
    for joint in [false, true] {
        let graph = graph(2, 1e154, 1e153, 1e153, Loss::Squared, joint);
        for linear_solver in [
            LinearSolver::Dense,
            LinearSolver::DenseQr(Default::default()),
            LinearSolver::SparseQr(Default::default()),
        ] {
            let report = solve_local(
                &graph,
                graph.initial(),
                &Options {
                    algorithm: Algorithm::GaussNewton,
                    linear_solver,
                    max_iterations: 1,
                    ..Default::default()
                },
            )
            .unwrap();
            let iteration = &report.iterations[0];
            assert!(iteration.accepted);
            assert!((report.x[0] + 10.).abs() < 1e-12);
            assert!(report.cost / 1e308 < 1e-25);
            assert!((iteration.actual_reduction / 1e308 - 1.).abs() < 1e-12);
            assert!((iteration.gain_ratio - 1.).abs() < 1e-12);
        }
    }
}

#[test]
fn robust_gain_ratios_use_the_weighted_quadratic_prediction() {
    let damping = 1e-3;
    let remaining: f64 = damping / (1. + damping);
    let prediction = 1. - remaining * remaining;
    for (loss, initial, actual) in [
        (Loss::Huber(1e154), 1.5, 1.5 - 2. * remaining * remaining),
        (
            Loss::Cauchy(2e154),
            2. * std::f64::consts::LN_2,
            2. * (std::f64::consts::LN_2 - (remaining * remaining).ln_1p()),
        ),
    ] {
        for joint in [false, true] {
            let graph = graph(1, 2e154, 1e153, 1e153, loss, joint);
            for linear_solver in lm_backends() {
                let report = solve_local(
                    &graph,
                    graph.initial(),
                    &Options {
                        loss,
                        linear_solver,
                        initial_damping: damping,
                        max_iterations: 1,
                        ..Default::default()
                    },
                )
                .unwrap();
                let iteration = &report.iterations[0];
                assert!(iteration.accepted);
                assert!((report.x[0] / (-20. / (1. + damping)) - 1.).abs() < 1e-12);
                assert!((report.initial_cost / 1e308 - initial).abs() < 1e-12);
                assert!((iteration.actual_reduction / 1e308 - actual).abs() < 1e-12);
                assert!((iteration.gain_ratio - actual / prediction).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn a_finite_worsening_trial_is_rejected_after_prediction_overflow() {
    let damping = 1e-3;
    let fraction: f64 = 1. / (1. + damping);
    let actual = 1. - (1. + 0.1 * fraction).powi(2);
    for joint in [false, true] {
        // Deliberately reversed and overestimated model Jacobian proposes an
        // increasing trial whose true objective remains representable.
        let graph = graph(2, 1e154, 1e152, -1e153, Loss::Squared, joint);
        for linear_solver in lm_backends() {
            let report = solve_local(
                &graph,
                graph.initial(),
                &Options {
                    linear_solver,
                    initial_damping: damping,
                    max_iterations: 1,
                    ..Default::default()
                },
            )
            .unwrap();
            let iteration = &report.iterations[0];
            assert!(!iteration.accepted && iteration.gain_ratio < 0.);
            assert!((iteration.actual_reduction / 1e308 - actual).abs() < 1e-12);
            assert_eq!(report.x[0], 0.);
            assert_eq!(report.cost, report.initial_cost);
            assert!(iteration.trial_cost > report.initial_cost);
        }
    }
}

#[test]
fn a_genuinely_unrepresentable_objective_still_returns_nonfinite() {
    let graph = graph(1, f64::MAX, 1., 1., Loss::Squared, false);
    for linear_solver in lm_backends() {
        assert_eq!(
            solve_local(
                &graph,
                graph.initial(),
                &Options {
                    linear_solver,
                    ..Default::default()
                }
            )
            .unwrap_err(),
            Error::NonFinite
        );
    }
}

#[test]
fn sparse_ldl_checks_large_and_tiny_rhs_and_true_residual_norms() {
    // J=[[2,1],[0,3]], so J'J=[[4,2],[2,10]]. For b=[s,2s],
    // the exact solution is [s/6,s/6], with unchanged conditioning as s varies.
    let j = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 2.), (0, 1, 1.), (1, 1, 3.)]).unwrap();
    for scale in [1., 1e-200, 1e200, 1e307] {
        let rhs = DVector::from_vec(vec![scale, 2. * scale]);
        let report = SparseLdlWorkspace::new(SparseLdlOptions {
            relative_tolerance: 1e-12,
            absolute_tolerance: 0.,
            ..Default::default()
        })
        .unwrap()
        .solve(&j, &rhs, 0.)
        .unwrap();
        assert!((report.x[0] / scale - 1. / 6.).abs() < 1e-12);
        assert!((report.x[1] / scale - 1. / 6.).abs() < 1e-12);
        let a = rhs[0] - 4. * report.x[0] - 2. * report.x[1];
        let b = rhs[1] - 2. * report.x[0] - 10. * report.x[1];
        assert!((a / scale).hypot(b / scale) < 1e-12);
        assert!(report.residual_norm.is_finite());
        // Match the original CSR operator's ordering for the reported norm.
        let applied = j
            .transpose_multiply(&j.multiply(&report.x).unwrap())
            .unwrap();
        let residual = &rhs - applied;
        let expected = residual[0].hypot(residual[1]);
        assert!((report.residual_norm / scale - expected / scale).abs() < 1e-25);
    }
}

#[test]
fn an_unrepresentable_ldl_rhs_norm_still_returns_nonfinite() {
    let j = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 1.), (1, 1, 1.)]).unwrap();
    assert_eq!(
        SparseLdlWorkspace::new(Default::default())
            .unwrap()
            .solve(&j, &DVector::from_element(2, f64::MAX), 0.)
            .unwrap_err(),
        Error::NonFinite
    );
}
