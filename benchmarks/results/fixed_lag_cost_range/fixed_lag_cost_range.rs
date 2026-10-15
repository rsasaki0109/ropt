use ropt::{factor::*, fixed_lag::*, *};

fn variable(key: u64, timestamp: Option<f64>) -> WindowVariable {
    WindowVariable {
        key,
        kind: VariableKind::Euclidean(1),
        value: DVector::zeros(1),
        timestamp,
        fixed: false,
    }
}

fn initial_window(
    model: MarginalizationLoss,
    loss: Loss,
    constants: &[f64],
    fixed: bool,
    joint: bool,
) -> FixedLagSmoother {
    let mut smoother = FixedLagSmoother::new(FixedLagOptions {
        lag: 1.,
        marginalization_loss: model,
        solver: Options {
            loss,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    let mut state = variable(0, Some(0.));
    state.fixed = fixed;
    let constants = constants.to_vec();
    smoother
        .update(
            0.,
            SmootherUpdate {
                variables: vec![state],
                factors: vec![SmootherFactor::analytic(
                    &[0],
                    constants.len() + 1,
                    FactorOptions {
                        loss: joint.then_some(loss),
                        ..Default::default()
                    },
                    move |values| {
                        let mut residual = DVector::zeros(constants.len() + 1);
                        residual[0] = values[0][0];
                        residual
                            .rows_mut(1, constants.len())
                            .copy_from_slice(&constants);
                        let mut jacobian = DMatrix::zeros(residual.len(), 1);
                        jacobian[(0, 0)] = 1.;
                        Ok((residual, vec![jacobian]))
                    },
                )],
                ..Default::default()
            },
        )
        .unwrap();
    smoother
}

fn check_empty_cost(smoother: &mut FixedLagSmoother, expected: f64) {
    assert_eq!(smoother.variable_count(), 0);
    assert_eq!(smoother.factor_count(), 0);
    let first = smoother.update(3., Default::default()).unwrap();
    assert!((first.final_cost / expected - 1.).abs() < 1e-12);
    assert_eq!(first.initial_cost, first.final_cost);
    for time in [4., 10.] {
        let idle = smoother.update(time, Default::default()).unwrap();
        assert_eq!(idle.final_cost, first.final_cost);
        assert!(idle.iterations.is_empty());
        assert!(idle.estimates.is_empty());
    }
}

#[test]
fn finite_large_gaussian_cost_survives_full_retirement_and_idle_updates() {
    for model in [
        MarginalizationLoss::Gaussian,
        MarginalizationLoss::FrozenIrls,
    ] {
        for fixed in [false, true] {
            for joint in [false, true] {
                let mut smoother =
                    initial_window(model, Loss::Squared, &[1e154, 1e154], fixed, joint);
                let report = smoother.update(2., Default::default()).unwrap();
                assert_eq!(report.marginalized, vec![0]);
                assert!((report.final_cost / 1e308 - 1.).abs() < 1e-12);
                check_empty_cost(&mut smoother, 1e308);
            }
        }
    }
}

#[test]
fn frozen_robust_offsets_preserve_finite_costs_beyond_squared_norm_range() {
    for (loss, constant, expected) in [
        (Loss::Huber(1e154), 1.9e154, 1.4e308),
        (Loss::Cauchy(2e154), 2e154, 1.3862943611198907e308),
    ] {
        for fixed in [false, true] {
            for joint in [false, true] {
                let mut smoother = initial_window(
                    MarginalizationLoss::FrozenIrls,
                    loss,
                    &[constant],
                    fixed,
                    joint,
                );
                let report = smoother.update(2., Default::default()).unwrap();
                assert!((report.final_cost / expected - 1.).abs() < 1e-12);
                check_empty_cost(&mut smoother, expected);
            }
        }
    }
}

#[test]
fn carried_prior_keeps_updates_covariance_and_cost_through_repeated_retirement() {
    for model in [
        MarginalizationLoss::Gaussian,
        MarginalizationLoss::FrozenIrls,
    ] {
        for linear_solver in [
            LinearSolver::DenseQr(Default::default()),
            LinearSolver::SparseQr(Default::default()),
        ] {
            let mut smoother = FixedLagSmoother::new(FixedLagOptions {
                lag: 1.,
                marginalization_loss: model,
                solver: Options {
                    algorithm: Algorithm::GaussNewton,
                    linear_solver,
                    // The independent constant dominates rounded aggregate cost.
                    cost_tolerance: 0.,
                    ..Default::default()
                },
                ..Default::default()
            })
            .unwrap();
            smoother
                .update(
                    0.,
                    SmootherUpdate {
                        variables: vec![variable(0, Some(0.)), variable(1, None)],
                        factors: vec![SmootherFactor::analytic(
                            &[0, 1],
                            4,
                            Default::default(),
                            |values| {
                                Ok((
                                    DVector::from_vec(vec![
                                        values[0][0],
                                        values[1][0] - values[0][0],
                                        1e154,
                                        1e154,
                                    ]),
                                    vec![
                                        DMatrix::from_column_slice(4, 1, &[1., -1., 0., 0.]),
                                        DMatrix::from_column_slice(4, 1, &[0., 1., 0., 0.]),
                                    ],
                                ))
                            },
                        )],
                        ..Default::default()
                    },
                )
                .unwrap();
            let first_retirement = smoother.update(2., Default::default()).unwrap();
            assert_eq!(first_retirement.marginalized, vec![0]);
            let prior_covariance = smoother
                .marginal_covariance(&[1], &Default::default())
                .unwrap();
            assert!((prior_covariance.matrix[(0, 0)] - 2.).abs() < 1e-12);
            let (updated, covariance) = smoother
                .update_with_covariance(
                    3.,
                    SmootherUpdate {
                        timestamps: vec![(1, 3.)],
                        factors: vec![SmootherFactor::analytic(
                            &[1],
                            1,
                            Default::default(),
                            |values| {
                                Ok((
                                    &values[0] - DVector::from_element(1, 2.),
                                    vec![DMatrix::identity(1, 1)],
                                ))
                            },
                        )],
                        ..Default::default()
                    },
                    &[1],
                    &Default::default(),
                )
                .unwrap();
            // Unit-variance measurement 2 combined with prior variance 2:
            // estimate = 4/3, posterior variance = 2/3.
            assert!((updated.estimates[&1][0] - 4. / 3.).abs() < 1e-10);
            assert!((covariance.matrix[(0, 0)] - 2. / 3.).abs() < 1e-12);
            assert!(
                updated
                    .iterations
                    .iter()
                    .any(|iteration| iteration.accepted && iteration.actual_reduction > 0.)
            );
            assert!((updated.final_cost / 1e308 - 1.).abs() < 1e-12);
            let flushed = smoother.update(5., Default::default()).unwrap();
            assert_eq!(flushed.marginalized, vec![1]);
            assert!((flushed.marginalized_estimates[&1][0] - 4. / 3.).abs() < 1e-10);
            assert_eq!(first_retirement.marginalized_estimates[&0][0], 0.);
            let idle = smoother.update(6., Default::default()).unwrap();
            assert!((idle.final_cost / 1e308 - 1.).abs() < 1e-12);
            let restarted = smoother
                .update(
                    7.,
                    SmootherUpdate {
                        variables: vec![variable(1, Some(7.))],
                        factors: vec![SmootherFactor::analytic(
                            &[1],
                            1,
                            Default::default(),
                            |values| {
                                Ok((
                                    &values[0] + DVector::from_element(1, 3.),
                                    vec![DMatrix::identity(1, 1)],
                                ))
                            },
                        )],
                        ..Default::default()
                    },
                )
                .unwrap();
            assert!((restarted.estimates[&1][0] + 3.).abs() < 1e-10);
            let covariance = smoother
                .marginal_covariance(&[1], &Default::default())
                .unwrap();
            assert!((covariance.matrix[(0, 0)] - 1.).abs() < 1e-12);
            assert_eq!(restarted.final_cost, idle.final_cost);
        }
    }
}

#[test]
fn carried_cost_overflow_rolls_back_new_variables_and_allows_a_valid_retry() {
    let mut smoother = initial_window(
        MarginalizationLoss::Gaussian,
        Loss::Squared,
        &[1e154, 1e154],
        false,
        false,
    );
    smoother.update(2., Default::default()).unwrap();
    let error = smoother
        .update(
            3.,
            SmootherUpdate {
                variables: vec![variable(1, Some(3.))],
                factors: vec![SmootherFactor::analytic(
                    &[1],
                    2,
                    Default::default(),
                    |_| {
                        Ok((
                            DVector::from_vec(vec![1e154, 1e154]),
                            vec![DMatrix::zeros(2, 1)],
                        ))
                    },
                )],
                ..Default::default()
            },
        )
        .unwrap_err();
    assert_eq!(error, Error::NonFinite);
    assert_eq!(smoother.last_time(), Some(2.));
    assert_eq!(smoother.variable_count(), 0);
    assert_eq!(smoother.factor_count(), 0);
    let idle = smoother.update(3., Default::default()).unwrap();
    assert!((idle.final_cost / 1e308 - 1.).abs() < 1e-12);
    let retry = smoother
        .update(
            3.,
            SmootherUpdate {
                variables: vec![variable(1, Some(3.))],
                factors: vec![SmootherFactor::analytic(
                    &[1],
                    1,
                    Default::default(),
                    |values| Ok((values[0].clone(), vec![DMatrix::identity(1, 1)])),
                )],
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(retry.final_cost, idle.final_cost);
    assert_eq!(smoother.variable_count(), 1);
    assert_eq!(smoother.factor_count(), 1);
}

#[test]
fn unrepresentable_objective_is_rejected_before_marginalization() {
    let mut smoother = initial_window(
        MarginalizationLoss::Gaussian,
        Loss::Squared,
        &[1.],
        false,
        false,
    );
    let error = smoother
        .update(
            1.,
            SmootherUpdate {
                factors: vec![SmootherFactor::analytic(
                    &[0],
                    1,
                    Default::default(),
                    |_| {
                        Ok((
                            DVector::from_element(1, f64::MAX),
                            vec![DMatrix::zeros(1, 1)],
                        ))
                    },
                )],
                ..Default::default()
            },
        )
        .unwrap_err();
    assert_eq!(error, Error::NonFinite);
    assert_eq!(smoother.last_time(), Some(0.));
    assert_eq!(smoother.variable_count(), 1);
    assert_eq!(smoother.factor_count(), 1);
    assert_eq!(smoother.estimate(0).unwrap()[0], 0.);
    let retry = smoother.update(1., Default::default()).unwrap();
    assert_eq!(retry.final_cost, 0.5);
}
