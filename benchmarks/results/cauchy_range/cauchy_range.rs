use ropt::{covariance::*, factor::*, *};

fn graph(
    residual: f64,
    scale: f64,
    joint: bool,
    fixed: bool,
    sign: f64,
) -> (FactorGraph, VariableId) {
    let mut graph = FactorGraph::new();
    let id = graph
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    graph.set_fixed(id, fixed).unwrap();
    graph
        .add_analytic_factor_with_options(
            &[id],
            1,
            FactorOptions {
                loss: joint.then_some(Loss::Cauchy(scale)),
                ..Default::default()
            },
            move |values| {
                Ok((
                    DVector::from_element(1, residual * (1. + values[0][0])),
                    vec![DMatrix::from_element(1, 1, sign * residual)],
                ))
            },
        )
        .unwrap();
    (graph, id)
}

fn backends() -> [LinearSolver; 4] {
    [
        LinearSolver::Dense,
        LinearSolver::DenseQr(Default::default()),
        LinearSolver::SparseLdl(Default::default()),
        LinearSolver::SparseQr(Default::default()),
    ]
}

#[test]
fn representable_cauchy_costs_survive_extreme_intermediates() {
    let ln10 = std::f64::consts::LN_10;
    for (residual, scale, expected) in [
        (1e155, 1., 155. * ln10),
        // The normalized residual itself overflows; the cost is finite.
        (1e300, 1e-100, 1e-200 * (400. * ln10)),
        // The squared scale overflows, or the normalized square underflows.
        (1., 1e200, 0.5),
        (2e154, 2e154, 1.3862943611198907e308),
        // The squared scale underflows but its product with the log survives.
        (1., 1e-162, 3.7e-322),
        (0., 1e200, 0.),
        (0., 1e-200, 0.),
    ] {
        for joint in [false, true] {
            let (graph, _) = graph(residual, scale, joint, true, 1.);
            for linear_solver in backends() {
                let report = solve_local(
                    &graph,
                    graph.initial(),
                    &Options {
                        linear_solver,
                        loss: Loss::Cauchy(scale),
                        ..Default::default()
                    },
                )
                .unwrap();
                let tolerance = (expected * 5e-14).max(f64::from_bits(2));
                assert!(
                    (report.cost - expected).abs() <= tolerance,
                    "r={residual}, a={scale}, joint={joint}, {linear_solver:?}: {report:?}"
                );
                assert_eq!(report.initial_cost, report.cost);
                assert!(report.iterations.is_empty());
            }
        }
    }
}

#[test]
fn subnormal_cauchy_weight_preserves_frozen_information() {
    for joint in [false, true] {
        let (graph, id) = graph(1e155, 1., joint, false, 1.);
        for input in [CovarianceInput::Dense, CovarianceInput::SparseToDense] {
            let covariance = graph
                .marginal_covariance(
                    &graph.initial(),
                    &[id],
                    &CovarianceOptions {
                        model: CovarianceModel::FrozenIrls {
                            fallback_loss: Loss::Cauchy(1.),
                        },
                        input,
                        ..Default::default()
                    },
                )
                .unwrap();
            // w ~= 1e-310, J=1e155, so J'WJ ~= 1 and its inverse ~= 1.
            assert!((covariance.matrix[(0, 0)] - 1.).abs() < 1e-12);
            assert_eq!(covariance.rank, 1);
        }
    }
}

#[test]
fn lm_accepts_finite_cauchy_reduction_between_large_residuals() {
    for joint in [false, true] {
        let (graph, _) = graph(1e155, 1., joint, false, 1.);
        for linear_solver in backends() {
            let report = solve_local(
                &graph,
                graph.initial(),
                &Options {
                    initial_damping: 1.,
                    max_iterations: 1,
                    linear_solver,
                    loss: Loss::Cauchy(1.),
                    ..Default::default()
                },
            )
            .unwrap();
            let iteration = &report.iterations[0];
            assert!(iteration.accepted && iteration.gain_ratio > 0.);
            assert!((report.x[0] + 0.5).abs() < 1e-12);
            assert!((iteration.actual_reduction - std::f64::consts::LN_2).abs() < 1e-12);
        }
    }
}

#[test]
fn gauss_newton_accepts_cauchy_reduction_across_scale_boundary() {
    for joint in [false, true] {
        let (graph, _) = graph(1e155, 1., joint, false, 1.);
        for linear_solver in [
            LinearSolver::Dense,
            LinearSolver::SparseQr(Default::default()),
        ] {
            let report = solve_local(
                &graph,
                graph.initial(),
                &Options {
                    algorithm: Algorithm::GaussNewton,
                    linear_solver,
                    loss: Loss::Cauchy(1.),
                    max_iterations: 1,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(report.x[0], -1.);
            assert_eq!(report.cost, 0.);
            let iteration = &report.iterations[0];
            assert!(iteration.accepted && iteration.gain_ratio > 0.);
            assert!((iteration.actual_reduction - 155. * std::f64::consts::LN_10).abs() < 1e-12);
        }
    }
}

#[test]
fn increasing_extreme_cauchy_trial_is_rejected() {
    for joint in [false, true] {
        // Deliberately reversed Jacobian proposes a residual-increasing step.
        let (graph, _) = graph(1e155, 1., joint, false, -1.);
        for linear_solver in backends() {
            let report = solve_local(
                &graph,
                graph.initial(),
                &Options {
                    linear_solver,
                    initial_damping: 1.,
                    loss: Loss::Cauchy(1.),
                    max_iterations: 1,
                    ..Default::default()
                },
            )
            .unwrap();
            let iteration = &report.iterations[0];
            assert!(!iteration.accepted && iteration.gain_ratio < 0.);
            assert!((iteration.actual_reduction + 1.5_f64.ln()).abs() < 1e-12);
            assert_eq!(report.x[0], 0.);
            assert_eq!(report.cost, report.initial_cost);
        }
    }
}

#[test]
fn unrepresentable_cauchy_cost_is_still_rejected() {
    let (graph, _) = graph(f64::MAX, f64::MAX, true, true, 1.);
    for linear_solver in backends() {
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
