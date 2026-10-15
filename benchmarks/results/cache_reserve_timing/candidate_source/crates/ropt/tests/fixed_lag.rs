use ropt::{
    DMatrix, DVector, Error, LinearSolver, Loss, Options, covariance::CovarianceOptions, factor::*,
    fixed_lag::*, solve_local,
};

fn variable(key: u64, time: Option<f64>, value: f64) -> WindowVariable {
    WindowVariable {
        key,
        kind: VariableKind::Euclidean(1),
        value: DVector::from_element(1, value),
        timestamp: time,
        fixed: false,
    }
}
fn observation(key: u64, measurement: f64, weight: f64) -> SmootherFactor {
    SmootherFactor::analytic(&[key], 1, FactorOptions::default(), move |v| {
        Ok((
            DVector::from_element(1, weight * (v[0][0] - measurement)),
            vec![DMatrix::from_element(1, 1, weight)],
        ))
    })
}
fn odometry(a: u64, b: u64, measurement: f64) -> SmootherFactor {
    SmootherFactor::analytic(&[a, b], 1, FactorOptions::default(), move |v| {
        Ok((
            DVector::from_element(1, v[1][0] - v[0][0] - measurement),
            vec![DMatrix::from_element(1, 1, -1.), DMatrix::identity(1, 1)],
        ))
    })
}
#[test]
fn linear_stream_matches_full_history_estimates_and_covariance_without_double_counting() {
    let mut smoother = FixedLagSmoother::new(FixedLagOptions {
        lag: 2.,
        ..FixedLagOptions::default()
    })
    .unwrap();
    let mut batch = FactorGraph::new();
    let mut ids = Vec::new();
    for k in 0..16_u64 {
        let measurement = k as f64 + 0.2 * (k as f64).sin();
        let id = batch
            .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
            .unwrap();
        ids.push(id);
        let obs = observation(k, measurement, 2.);
        batch
            .add_analytic_factor(&[id], 1, move |v| {
                Ok((
                    DVector::from_element(1, 2. * (v[0][0] - measurement)),
                    vec![DMatrix::from_element(1, 1, 2.)],
                ))
            })
            .unwrap();
        let mut update = SmootherUpdate {
            variables: vec![variable(k, Some(k as f64), k as f64)],
            factors: vec![obs],
            ..SmootherUpdate::default()
        };
        if k > 0 {
            update.factors.push(odometry(k - 1, k, 1.));
            batch
                .add_analytic_factor(&[ids[k as usize - 1], id], 1, |v| {
                    Ok((
                        &v[1] - &v[0] - DVector::from_element(1, 1.),
                        vec![-DMatrix::identity(1, 1), DMatrix::identity(1, 1)],
                    ))
                })
                .unwrap();
        }
        let report = smoother.update(k as f64, update).unwrap();
        let full = solve_local(&batch, batch.initial(), &Options::default()).unwrap();
        for (&key, estimate) in &report.estimates {
            assert!(
                (estimate[0] - batch.value(&full.x, ids[key as usize]).unwrap()[0]).abs() < 1e-8
            );
        }
        let live = report.estimates.keys().copied().collect::<Vec<_>>();
        let a = smoother
            .marginal_covariance(&live, &CovarianceOptions::default())
            .unwrap();
        let b = batch
            .marginal_covariance(
                &full.x,
                &live.iter().map(|k| ids[*k as usize]).collect::<Vec<_>>(),
                &CovarianceOptions::default(),
            )
            .unwrap();
        assert!((a.matrix - b.matrix).norm() < 1e-10);
        assert!((report.final_cost - full.cost).abs() < 1e-9);
        assert!(smoother.variable_count() <= 3);
        assert!(smoother.factor_count() <= 6);
        if k >= 3 {
            assert_eq!(report.marginalized, vec![k - 3]);
            assert_eq!(
                report.marginalization.removed_factors,
                if k == 3 { 2 } else { 3 }
            );
            assert_eq!(report.marginalization.separator_variables, 1);
            assert!(report.marginalization.prior_rows <= 2);
        }
    }
}
#[test]
fn fixed_anchor_information_survives_expiry_and_persistent_variables_stay_live() {
    let mut s = FixedLagSmoother::new(FixedLagOptions {
        lag: 1.,
        ..FixedLagOptions::default()
    })
    .unwrap();
    let mut root = variable(0, Some(0.), 0.);
    root.fixed = true;
    s.update(
        0.,
        SmootherUpdate {
            variables: vec![root],
            factors: vec![observation(0, 0., 1.)],
            ..SmootherUpdate::default()
        },
    )
    .unwrap();
    for k in 1..=20 {
        let report = s
            .update(
                k as f64,
                SmootherUpdate {
                    variables: vec![variable(k, Some(k as f64), 0.)],
                    factors: vec![odometry(k - 1, k, 1.)],
                    ..SmootherUpdate::default()
                },
            )
            .unwrap();
        assert!((report.estimates[&k][0] - k as f64).abs() < 1e-7);
        if k >= 2 {
            assert!(s.estimate(0).is_none());
        }
        let c = s
            .marginal_covariance(&[k], &CovarianceOptions::default())
            .unwrap();
        assert!((c.matrix[(0, 0)] - k as f64).abs() < 1e-8);
        assert!(s.variable_count() <= 2 && s.factor_count() <= 2);
    }
    let mut s = FixedLagSmoother::new(FixedLagOptions {
        lag: 1.,
        ..FixedLagOptions::default()
    })
    .unwrap();
    s.update(
        0.,
        SmootherUpdate {
            variables: vec![variable(100, None, 0.)],
            factors: vec![observation(100, 3., 1.)],
            ..SmootherUpdate::default()
        },
    )
    .unwrap();
    let r = s.update(10., SmootherUpdate::default()).unwrap();
    assert!(r.marginalized.is_empty());
    assert!((s.estimate(100).unwrap()[0] - 3.).abs() < 1e-8);
}
#[test]
fn timestamps_are_strict_at_cutoff_and_refresh_does_not_change_estimates() {
    let mut s = FixedLagSmoother::new(FixedLagOptions {
        lag: 2.,
        ..FixedLagOptions::default()
    })
    .unwrap();
    s.update(
        0.,
        SmootherUpdate {
            variables: vec![variable(0, Some(0.), 0.)],
            factors: vec![observation(0, 1., 1.)],
            ..SmootherUpdate::default()
        },
    )
    .unwrap();
    assert!(
        s.update(2., SmootherUpdate::default())
            .unwrap()
            .marginalized
            .is_empty()
    );
    s.update(
        2.,
        SmootherUpdate {
            timestamps: vec![(0, 2.)],
            ..SmootherUpdate::default()
        },
    )
    .unwrap();
    assert!(
        s.update(4., SmootherUpdate::default())
            .unwrap()
            .marginalized
            .is_empty()
    );
    let expired = s.update(4.1, SmootherUpdate::default()).unwrap();
    assert_eq!(expired.marginalized, vec![0]);
    assert_eq!(s.variable_count(), 0);
    assert_eq!(s.factor_count(), 0);
}
#[test]
fn failed_updates_leave_state_time_and_factors_unchanged() {
    let mut s = FixedLagSmoother::new(FixedLagOptions::default()).unwrap();
    s.update(
        0.,
        SmootherUpdate {
            variables: vec![variable(0, Some(0.), 0.)],
            factors: vec![observation(0, 2., 1.)],
            ..SmootherUpdate::default()
        },
    )
    .unwrap();
    let before = s.estimate(0).unwrap().clone();
    for (now, update) in [
        (-1., SmootherUpdate::default()),
        (
            1.,
            SmootherUpdate {
                variables: vec![variable(0, Some(1.), 0.)],
                ..SmootherUpdate::default()
            },
        ),
        (
            4.,
            SmootherUpdate {
                variables: vec![variable(1, Some(0.), 0.)],
                ..SmootherUpdate::default()
            },
        ),
        (
            1.,
            SmootherUpdate {
                factors: vec![odometry(0, 999, 1.)],
                ..SmootherUpdate::default()
            },
        ),
        (
            1.,
            SmootherUpdate {
                timestamps: vec![(0, f64::NAN)],
                ..SmootherUpdate::default()
            },
        ),
        (
            1.,
            SmootherUpdate {
                factors: vec![observation(0, f64::NAN, 1.)],
                ..SmootherUpdate::default()
            },
        ),
    ] {
        assert!(s.update(now, update).is_err());
        assert_eq!(s.estimate(0).unwrap(), &before);
        assert_eq!(s.last_time(), Some(0.));
        assert_eq!(s.factor_count(), 1);
    }
    let mut options = FixedLagOptions::default();
    options.marginalization.max_entries = 3;
    let mut limited = FixedLagSmoother::new(options).unwrap();
    assert!(
        limited
            .update(
                0.,
                SmootherUpdate {
                    variables: vec![variable(0, Some(0.), 0.)],
                    factors: vec![observation(0, 2., 1.)],
                    ..SmootherUpdate::default()
                }
            )
            .is_err()
    );
    assert_eq!(limited.variable_count(), 0);
    assert_eq!(limited.last_time(), None);
}
#[test]
fn rank_failure_during_marginalization_rolls_back_and_gauge_is_not_invented() {
    let mut s = FixedLagSmoother::new(FixedLagOptions {
        lag: 1.,
        ..FixedLagOptions::default()
    })
    .unwrap();
    s.update(
        0.,
        SmootherUpdate {
            variables: vec![variable(0, Some(0.), 0.)],
            factors: vec![observation(0, 1., 0.)],
            ..SmootherUpdate::default()
        },
    )
    .unwrap();
    assert_eq!(
        s.update(2., SmootherUpdate::default()).unwrap_err(),
        Error::RankDeficient {
            rank: 0,
            parameters: 1
        }
    );
    assert_eq!(s.last_time(), Some(0.));
    assert!(s.estimate(0).is_some());
    let mut s = FixedLagSmoother::new(FixedLagOptions {
        lag: 1.,
        ..FixedLagOptions::default()
    })
    .unwrap();
    s.update(
        1.,
        SmootherUpdate {
            variables: vec![variable(0, Some(0.), 0.), variable(1, Some(1.), 1.)],
            factors: vec![odometry(0, 1, 1.)],
            ..SmootherUpdate::default()
        },
    )
    .unwrap();
    s.update(2., SmootherUpdate::default()).unwrap();
    assert!(matches!(
        s.marginal_covariance(&[1], &CovarianceOptions::default()),
        Err(Error::RankDeficient { .. })
    ));
}
#[test]
fn robust_marginalization_requires_explicit_frozen_irls() {
    for model in [
        MarginalizationLoss::Gaussian,
        MarginalizationLoss::FrozenIrls,
    ] {
        let mut s = FixedLagSmoother::new(FixedLagOptions {
            lag: 1.,
            marginalization_loss: model,
            solver: Options {
                linear_solver: LinearSolver::DenseQr(Default::default()),
                loss: Loss::Huber(1.),
                ..Options::default()
            },
            ..FixedLagOptions::default()
        })
        .unwrap();
        s.update(
            0.,
            SmootherUpdate {
                variables: vec![variable(0, Some(0.), 0.)],
                factors: vec![
                    observation(0, 2., 1.),
                    observation(0, 2.1, 1.),
                    observation(0, 20., 1.),
                ],
                ..SmootherUpdate::default()
            },
        )
        .unwrap();
        let before = s.estimate(0).unwrap().clone();
        let result = s.update(
            2.,
            SmootherUpdate {
                variables: vec![variable(1, Some(2.), 0.)],
                factors: vec![odometry(0, 1, 1.)],
                ..SmootherUpdate::default()
            },
        );
        match model {
            MarginalizationLoss::Gaussian => {
                assert!(result.is_err());
                assert_eq!(s.estimate(0).unwrap(), &before);
                assert_eq!(s.last_time(), Some(0.));
            }
            MarginalizationLoss::FrozenIrls => {
                let r = result.unwrap();
                assert_eq!(r.marginalized, vec![0]);
                assert!((s.estimate(1).unwrap()[0] - before[0] - 1.).abs() < 1e-6);
                assert_eq!(s.factor_count(), 1);
                // The prior explicitly overrides the global Huber fallback with squared loss.
                s.marginal_covariance(&[1], &CovarianceOptions::default())
                    .unwrap();
            }
        }
    }
}

#[test]
fn coupled_separator_prior_preserves_cross_covariance_after_new_observations() {
    let mut s = FixedLagSmoother::new(FixedLagOptions {
        lag: 1.,
        ..FixedLagOptions::default()
    })
    .unwrap();
    let mut batch = FactorGraph::new();
    let ids = (0..3)
        .map(|_| {
            batch
                .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
                .unwrap()
        })
        .collect::<Vec<_>>();
    batch
        .add_analytic_factor(&[ids[0]], 1, |v| {
            Ok((
                &v[0] - DVector::from_element(1, 3.),
                vec![DMatrix::identity(1, 1)],
            ))
        })
        .unwrap();
    for (b, m) in [(1, 1.), (2, -1.)] {
        batch
            .add_analytic_factor(&[ids[0], ids[b]], 1, move |v| {
                Ok((
                    &v[1] - &v[0] - DVector::from_element(1, m),
                    vec![-DMatrix::identity(1, 1), DMatrix::identity(1, 1)],
                ))
            })
            .unwrap();
    }
    s.update(
        1.,
        SmootherUpdate {
            variables: vec![
                variable(0, Some(0.), 0.),
                variable(1, Some(1.), 0.),
                variable(2, Some(1.), 0.),
            ],
            factors: vec![
                observation(0, 3., 1.),
                odometry(0, 1, 1.),
                odometry(0, 2, -1.),
            ],
            ..SmootherUpdate::default()
        },
    )
    .unwrap();
    let r = s.update(2., SmootherUpdate::default()).unwrap();
    assert_eq!(r.marginalization.separator_variables, 2);
    assert_eq!(s.factor_count(), 1);
    let cov = s
        .marginal_covariance(&[1, 2], &CovarianceOptions::default())
        .unwrap();
    assert!((cov.matrix - DMatrix::from_row_slice(2, 2, &[2., 1., 1., 2.])).norm() < 1e-9);
    s.update(
        2.,
        SmootherUpdate {
            factors: vec![observation(1, 10., 2.)],
            ..SmootherUpdate::default()
        },
    )
    .unwrap();
    batch
        .add_analytic_factor(&[ids[1]], 1, |v| {
            Ok((
                &v[0] * 2. - DVector::from_element(1, 20.),
                vec![DMatrix::from_element(1, 1, 2.)],
            ))
        })
        .unwrap();
    let full = solve_local(&batch, batch.initial(), &Options::default()).unwrap();
    for k in [1, 2] {
        assert!(
            (s.estimate(k).unwrap() - batch.value(&full.x, ids[k as usize]).unwrap()).norm() < 1e-8
        );
    }
    let a = s
        .marginal_covariance(&[1, 2], &CovarianceOptions::default())
        .unwrap();
    let b = batch
        .marginal_covariance(&full.x, &[ids[1], ids[2]], &CovarianceOptions::default())
        .unwrap();
    assert!((a.matrix - b.matrix).norm() < 1e-10);
}

#[test]
fn se2_and_se3_streams_keep_manifold_priors_and_anchor_information() {
    use ropt::{manifold::*, pose_graph::*};
    for (kind, anchor, increment) in [
        (
            VariableKind::SE2,
            se2_values(SE2::exp(nalgebra::Vector3::new(2., -1., 0.4))),
            DVector::from_vec(vec![0.6, 0.1, 0.15]),
        ),
        (
            VariableKind::SE3,
            se3_values(SE3::exp(Tangent6::new(2., -1., 0.5, 0.2, 0.1, -0.3))),
            DVector::from_vec(vec![0.6, 0.1, 0.05, 0.02, -0.01, 0.15]),
        ),
    ] {
        let mut s = FixedLagSmoother::new(FixedLagOptions {
            lag: 2.,
            ..FixedLagOptions::default()
        })
        .unwrap();
        let mut root = WindowVariable {
            key: 0,
            kind,
            value: anchor.clone(),
            timestamp: Some(0.),
            fixed: true,
        };
        let n = kind.dimensions().1;
        let target = anchor.clone();
        let prior = SmootherFactor::autodiff(&[0], n, 32, FactorOptions::default(), move |v| {
            kind.local_autodiff(&target, v[0])
        });
        s.update(
            0.,
            SmootherUpdate {
                variables: vec![root.clone()],
                factors: vec![prior],
                ..SmootherUpdate::default()
            },
        )
        .unwrap();
        let mut previous = anchor;
        for k in 1..=12 {
            let truth = kind.retract(&previous, &increment).unwrap();
            let perturbed = kind.retract(&truth, &(&increment * 0.1)).unwrap();
            let factor = if matches!(kind, VariableKind::SE2) {
                let measurement = read_se2(&previous)
                    .unwrap()
                    .inverse()
                    .compose(read_se2(&truth).unwrap());
                let residual = RelativePose2::new(measurement).unwrap();
                SmootherFactor::autodiff(&[k - 1, k], 3, 6, FactorOptions::default(), move |v| {
                    residual.residual_autodiff(v[0], v[1])
                })
            } else {
                let measurement = read_se3(&previous)
                    .unwrap()
                    .inverse()
                    .compose(read_se3(&truth).unwrap());
                let residual = RelativePose3::new(measurement).unwrap();
                SmootherFactor::autodiff(&[k - 1, k], 6, 12, FactorOptions::default(), move |v| {
                    residual.residual_autodiff(v[0], v[1])
                })
            };
            root.key = k;
            root.value = perturbed;
            root.timestamp = Some(k as f64);
            root.fixed = false;
            let report = s
                .update(
                    k as f64,
                    SmootherUpdate {
                        variables: vec![root.clone()],
                        factors: vec![factor],
                        ..SmootherUpdate::default()
                    },
                )
                .unwrap();
            assert!(kind.local(&truth, &report.estimates[&k]).unwrap().norm() < 2e-7);
            assert!(s.variable_count() <= 3 && s.factor_count() <= 3);
            if k >= 3 {
                assert!(s.estimate(0).is_none());
            }
            let c = s
                .marginal_covariance(&[k], &CovarianceOptions::default())
                .unwrap();
            assert_eq!(c.matrix.shape(), (n, n));
            assert!(c.matrix.diagonal().iter().all(|v| *v > 0.));
            previous = truth;
        }
    }
}
