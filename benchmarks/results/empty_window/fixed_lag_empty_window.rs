use ropt::{factor::*, fixed_lag::*, *};

fn variable(time: f64) -> WindowVariable {
    WindowVariable {
        key: 0,
        kind: VariableKind::Euclidean(1),
        value: DVector::zeros(1),
        timestamp: Some(time),
        fixed: false,
    }
}
fn observation(target: f64) -> SmootherFactor {
    SmootherFactor::analytic(&[0], 1, Default::default(), move |v| {
        Ok((
            DVector::from_element(1, v[0][0] - target),
            vec![DMatrix::identity(1, 1)],
        ))
    })
}
fn assert_idle(report: &FixedLagReport, cost: f64) {
    assert!(report.estimates.is_empty());
    assert!(report.marginalized.is_empty());
    assert!(report.marginalized_estimates.is_empty());
    assert!(report.marginalized_timestamps.is_empty());
    assert!(report.iterations.is_empty());
    assert_eq!(report.termination, Termination::Gradient);
    assert_eq!(report.initial_cost, cost);
    assert_eq!(report.final_cost, cost);
    assert_eq!(report.marginalization.removed_factors, 0);
}

#[test]
fn empty_window_ticks_preserve_gaussian_and_robust_cost_then_allow_key_reuse() {
    for robust in [false, true] {
        let mut smoother = FixedLagSmoother::new(FixedLagOptions {
            lag: 1.,
            marginalization_loss: if robust {
                MarginalizationLoss::FrozenIrls
            } else {
                MarginalizationLoss::Gaussian
            },
            solver: Options {
                loss: if robust {
                    Loss::Huber(1.)
                } else {
                    Loss::Squared
                },
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap();
        smoother
            .update(
                0.,
                SmootherUpdate {
                    variables: vec![variable(0.)],
                    factors: vec![observation(2.), observation(2.1), observation(20.)],
                    ..Default::default()
                },
            )
            .unwrap();
        let retired = smoother.update(2., Default::default()).unwrap();
        assert_eq!(retired.marginalized, vec![0]);
        assert!(retired.final_cost > 1.);
        let first_idle = smoother.update(2., Default::default()).unwrap();
        // QR compression can round the pre-expiry cost by a few ulps.
        assert!((first_idle.final_cost - retired.final_cost).abs() < 1e-10);
        let carried_cost = first_idle.final_cost;
        assert_idle(&first_idle, carried_cost);
        for now in [3., 10.] {
            assert_idle(
                &smoother.update(now, Default::default()).unwrap(),
                carried_cost,
            );
            assert_eq!(smoother.last_time(), Some(now));
            assert_eq!(smoother.variable_count(), 0);
            assert_eq!(smoother.factor_count(), 0);
        }
        let restarted = smoother
            .update(
                11.,
                SmootherUpdate {
                    variables: vec![variable(11.)],
                    factors: vec![observation(-5.)],
                    ..Default::default()
                },
            )
            .unwrap();
        assert!((restarted.estimates[&0][0] + 5.).abs() < 1e-7);
        assert!((restarted.final_cost - retired.final_cost).abs() < 1e-10);
        assert_eq!(retired.marginalized_timestamps[&0], 0.);
    }
}

#[test]
fn fresh_empty_clock_validates_time_and_rejects_covariance_or_dangling_factors_transactionally() {
    let mut s = FixedLagSmoother::new(Default::default()).unwrap();
    assert_idle(&s.update(1., Default::default()).unwrap(), 0.);
    for time in [0., f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(s.update(time, Default::default()).is_err());
        assert_eq!(s.last_time(), Some(1.));
    }
    for keys in [vec![], vec![0]] {
        assert!(
            s.update_with_covariance(2., Default::default(), &keys, &Default::default())
                .is_err()
        );
        assert_eq!(s.last_time(), Some(1.));
    }
    assert!(
        s.update(
            2.,
            SmootherUpdate {
                factors: vec![observation(2.)],
                ..Default::default()
            }
        )
        .is_err()
    );
    assert_eq!(s.last_time(), Some(1.));
    assert_idle(&s.update(2., Default::default()).unwrap(), 0.);
    let mut overflow = FixedLagSmoother::new(FixedLagOptions {
        lag: f64::MAX,
        ..Default::default()
    })
    .unwrap();
    assert!(overflow.update(-f64::MAX, Default::default()).is_err());
    assert_eq!(overflow.last_time(), None);
    let mut deferred = FixedLagSmoother::new(FixedLagOptions {
        solver: Options {
            max_iterations: 0,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    assert_idle(&deferred.update(1., Default::default()).unwrap(), 0.);
    assert!(matches!(
        deferred.update(
            2.,
            SmootherUpdate {
                variables: vec![variable(2.)],
                factors: vec![observation(2.)],
                ..Default::default()
            }
        ),
        Err(Error::InvalidOptions)
    ));
    assert_eq!(deferred.last_time(), Some(1.));
    assert_eq!(deferred.variable_count(), 0);
}
