use ropt::{covariance::*, factor::*, fixed_lag::*, *};

fn graph(residual: f64, jacobian: f64, loss: Loss, joint: bool) -> (FactorGraph, VariableId) {
    let mut graph = FactorGraph::new();
    let id = graph
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    graph
        .add_analytic_factor_with_options(
            &[id],
            1,
            FactorOptions {
                loss: joint.then_some(loss),
                ..Default::default()
            },
            move |values| {
                Ok((
                    DVector::from_element(1, residual + jacobian * values[0][0]),
                    vec![DMatrix::from_element(1, 1, jacobian)],
                ))
            },
        )
        .unwrap();
    (graph, id)
}

fn covariance_options(loss: Loss, input: CovarianceInput) -> CovarianceOptions {
    CovarianceOptions {
        model: CovarianceModel::FrozenIrls {
            fallback_loss: loss,
        },
        input,
        ..Default::default()
    }
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
fn tiny_robust_weights_keep_representable_frozen_information() {
    for (residual, jacobian, loss, expected, tolerance) in [
        // Ordinary-scale controls: covariance is 1 / (J^2 * w).
        (10., 10., Loss::Cauchy(1.), 1.01, 1e-12),
        (10., 10., Loss::Huber(0.1), 1., 1e-12),
        // Subnormal weights: sqrt(round(w)) loses significant digits.
        (1e160, 1e160, Loss::Cauchy(1.), 1., 1e-12),
        (1e160, 1e160, Loss::Huber(1e-160), 1., 1e-12),
        // The raw weights underflow to zero; their square roots survive.
        (1e200, 1e200, Loss::Cauchy(1.), 1., 1e-12),
        (1e200, 1e200, Loss::Huber(1e-200), 1., 1e-12),
        // A subnormal square root remains usable with a large Jacobian.
        (
            f64::MAX,
            f64::MAX,
            Loss::Huber(f64::from_bits(1)),
            1. / (f64::from_bits(1) * f64::MAX),
            1e-7,
        ),
        (
            1.,
            f64::MAX,
            Loss::Cauchy(f64::from_bits(1)),
            (1. / (f64::from_bits(1) * f64::MAX)).powi(2),
            1e-12,
        ),
    ] {
        for joint in [false, true] {
            let (graph, id) = graph(residual, jacobian, loss, joint);
            for input in [CovarianceInput::Dense, CovarianceInput::SparseToDense] {
                let covariance = graph
                    .marginal_covariance(&graph.initial(), &[id], &covariance_options(loss, input))
                    .unwrap();
                assert!(
                    (covariance.matrix[(0, 0)] / expected - 1.).abs() < tolerance,
                    "r={residual}, J={jacobian}, loss={loss:?}, joint={joint}: {covariance:?}"
                );
                assert_eq!(covariance.rank, 1);
            }
        }
    }
}

fn check_lm(loss: Loss, expected_reduction: f64) {
    for joint in [false, true] {
        let (graph, _) = graph(1e200, 1e200, loss, joint);
        for linear_solver in backends() {
            let report = solve_local(
                &graph,
                graph.initial(),
                &Options {
                    loss,
                    linear_solver,
                    initial_damping: 1.,
                    max_iterations: 1,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!((report.x[0] + 0.5).abs() < 1e-12, "{report:?}");
            let iteration = &report.iterations[0];
            assert!(iteration.accepted && iteration.gain_ratio > 0.);
            assert!((iteration.actual_reduction - expected_reduction).abs() < 1e-12);
        }
    }
}

#[test]
fn cauchy_lm_moves_when_only_the_square_root_weight_is_representable() {
    check_lm(Loss::Cauchy(1.), std::f64::consts::LN_2);
}

#[test]
fn huber_lm_moves_when_only_the_square_root_weight_is_representable() {
    check_lm(Loss::Huber(1e-200), 0.5);
}

#[test]
fn increasing_trials_with_tiny_weights_leave_the_accepted_state_intact() {
    for (loss, expected_reduction) in [
        (Loss::Cauchy(1.), -1.5_f64.ln()),
        (Loss::Huber(1e-200), -0.5),
    ] {
        for joint in [false, true] {
            // Deliberately reversed Jacobian proposes a worsening trial.
            let mut graph = FactorGraph::new();
            let id = graph
                .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
                .unwrap();
            graph
                .add_analytic_factor_with_options(
                    &[id],
                    1,
                    FactorOptions {
                        loss: joint.then_some(loss),
                        ..Default::default()
                    },
                    |values| {
                        Ok((
                            DVector::from_element(1, 1e200 * (1. + values[0][0])),
                            vec![DMatrix::from_element(1, 1, -1e200)],
                        ))
                    },
                )
                .unwrap();
            for linear_solver in backends() {
                let report = solve_local(
                    &graph,
                    graph.initial(),
                    &Options {
                        loss,
                        linear_solver,
                        initial_damping: 1.,
                        max_iterations: 1,
                        ..Default::default()
                    },
                )
                .unwrap();
                let iteration = &report.iterations[0];
                assert!(!iteration.accepted && iteration.gain_ratio < 0.);
                assert!((iteration.actual_reduction - expected_reduction).abs() < 1e-12);
                assert_eq!(report.x[0], 0.);
                assert_eq!(report.initial_cost, report.cost);
            }
        }
    }
}

fn variable(key: u64, time: f64) -> WindowVariable {
    WindowVariable {
        key,
        kind: VariableKind::Euclidean(1),
        value: DVector::zeros(1),
        timestamp: Some(time),
        fixed: false,
    }
}

#[test]
fn fixed_lag_freezes_tiny_weights_without_losing_separator_information() {
    for loss in [Loss::Cauchy(1.), Loss::Huber(1e-200)] {
        for joint in [false, true] {
            let mut smoother = FixedLagSmoother::new(FixedLagOptions {
                lag: 1.,
                marginalization_loss: MarginalizationLoss::FrozenIrls,
                solver: Options {
                    loss,
                    // Deliberately freeze at the initial point. This checks
                    // the information model, rather than its optimum.
                    gradient_tolerance: 10.,
                    ..Default::default()
                },
                ..Default::default()
            })
            .unwrap();
            smoother
                .update(
                    0.,
                    SmootherUpdate {
                        variables: vec![variable(0, 0.)],
                        factors: vec![SmootherFactor::analytic(
                            &[0],
                            1,
                            FactorOptions {
                                loss: joint.then_some(loss),
                                ..Default::default()
                            },
                            |values| {
                                Ok((
                                    DVector::from_element(1, 1e200 * (1. + values[0][0])),
                                    vec![DMatrix::from_element(1, 1, 1e200)],
                                ))
                            },
                        )],
                        ..Default::default()
                    },
                )
                .unwrap();
            let (report, covariance) = smoother
                .update_with_covariance(
                    2.,
                    SmootherUpdate {
                        variables: vec![variable(1, 2.)],
                        factors: vec![SmootherFactor::analytic(
                            &[0, 1],
                            1,
                            FactorOptions {
                                loss: Some(Loss::Squared),
                                ..Default::default()
                            },
                            |values| {
                                Ok((
                                    &values[1] - &values[0],
                                    vec![-DMatrix::identity(1, 1), DMatrix::identity(1, 1)],
                                ))
                            },
                        )],
                        ..Default::default()
                    },
                    &[0, 1],
                    &covariance_options(loss, CovarianceInput::Dense),
                )
                .unwrap();
            // Frozen J'WJ = [[2,-1],[-1,1]]; its inverse is [[1,1],[1,2]].
            let expected = DMatrix::from_row_slice(2, 2, &[1., 1., 1., 2.]);
            assert!((covariance.matrix - expected).norm() < 1e-12);
            assert_eq!(report.marginalized, vec![0]);
            assert_eq!(report.marginalization.separator_tangent, 1);
            let retained = smoother
                .marginal_covariance(&[1], &CovarianceOptions::default())
                .unwrap();
            // Schur complement leaves information 1/2 on the separator.
            assert!((retained.matrix[(0, 0)] - 2.).abs() < 1e-12);
        }
    }
}

#[test]
fn unrepresentable_square_root_weight_still_has_zero_frozen_rank() {
    let loss = Loss::Cauchy(1e-200);
    for joint in [false, true] {
        let (graph, id) = graph(1e200, 1e200, loss, joint);
        for input in [CovarianceInput::Dense, CovarianceInput::SparseToDense] {
            assert_eq!(
                graph
                    .marginal_covariance(&graph.initial(), &[id], &covariance_options(loss, input))
                    .unwrap_err(),
                Error::RankDeficient {
                    rank: 0,
                    parameters: 1
                }
            );
        }
    }
}
