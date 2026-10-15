use ropt::{
    Algorithm, DMatrix, DVector, Error, LinearSolver, Options,
    covariance::{CovarianceInput, CovarianceOptions},
    factor::*,
    fixed_lag::*,
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
fn observation(key: u64, target: f64, weight: f64) -> SmootherFactor {
    SmootherFactor::analytic(&[key], 1, Default::default(), move |v| {
        Ok((
            DVector::from_element(1, weight * (v[0][0] - target)),
            vec![DMatrix::from_element(1, 1, weight)],
        ))
    })
}
fn smoother() -> FixedLagSmoother {
    FixedLagSmoother::new(FixedLagOptions {
        lag: 1.,
        solver: Options {
            algorithm: Algorithm::GaussNewton,
            linear_solver: LinearSolver::DenseQr(Default::default()),
            gradient_tolerance: 1e-12,
            step_tolerance: 1e-14,
            cost_tolerance: 1e-15,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap()
}
fn coupled() -> FixedLagSmoother {
    let mut s = smoother();
    s.update(
        1.,
        SmootherUpdate {
            variables: vec![variable(0, Some(0.), 0.), variable(1, Some(1.), 0.)],
            factors: vec![
                observation(0, 0., 2.),
                observation(1, 3., 1.),
                SmootherFactor::analytic(&[0, 1], 1, Default::default(), |v| {
                    Ok((
                        DVector::from_element(1, v[1][0] - v[0][0] - 1.),
                        vec![-DMatrix::identity(1, 1), DMatrix::identity(1, 1)],
                    ))
                }),
            ],
            ..Default::default()
        },
    )
    .unwrap();
    s
}
fn added_observation() -> SmootherUpdate {
    SmootherUpdate {
        factors: vec![observation(0, 10., 1.)],
        ..Default::default()
    }
}

#[test]
fn pre_expiry_covariance_keeps_cross_blocks_query_order_and_variance_scale() {
    for options in [
        CovarianceOptions::default(),
        CovarianceOptions {
            variance_scale: 2.,
            input: CovarianceInput::SparseToDense,
            ..Default::default()
        },
    ] {
        let mut s = coupled();
        let (report, covariance) = s
            .update_with_covariance(2., added_observation(), &[1, 0], &options)
            .unwrap();
        // In key order [0,1], J^T J = [[6,-1],[-1,2]].
        let expected =
            DMatrix::from_row_slice(2, 2, &[6., 1., 1., 2.]) * (options.variance_scale / 11.);
        assert!((covariance.matrix - expected).norm() < 1e-12);
        assert_eq!(covariance.indices, vec![1, 0]);
        assert_eq!(covariance.rank, 2);
        assert!((report.marginalized_estimates[&0][0] - 2.).abs() < 1e-10);
        assert!((report.estimates[&1][0] - 3.).abs() < 1e-10);
        assert!(s.marginal_covariance(&[0], &options).is_err());
        assert!(
            (s.marginal_covariance(&[1], &options).unwrap().matrix[(0, 0)]
                - 6. * options.variance_scale / 11.)
                .abs()
                < 1e-12
        );
    }
}

#[test]
fn covariance_failures_roll_back_optimization_expiry_and_new_factors() {
    let mut s = coupled();
    let old = s.estimate(0).unwrap().clone();
    for (keys, options) in [
        (vec![], CovarianceOptions::default()),
        (vec![999], CovarianceOptions::default()),
        (vec![0, 0], CovarianceOptions::default()),
        (
            vec![0],
            CovarianceOptions {
                svd_max_iterations: 0,
                ..Default::default()
            },
        ),
    ] {
        assert!(
            s.update_with_covariance(2., added_observation(), &keys, &options)
                .is_err()
        );
        assert_eq!(s.last_time(), Some(1.));
        assert_eq!(s.variable_count(), 2);
        assert_eq!(s.factor_count(), 3);
        assert_eq!(s.estimate(0).unwrap(), &old);
    }
    let (retried, _) = s
        .update_with_covariance(2., added_observation(), &[0], &Default::default())
        .unwrap();
    let expected = coupled().update(2., added_observation()).unwrap();
    assert_eq!(retried.estimates, expected.estimates);
    assert_eq!(
        retried.marginalized_estimates,
        expected.marginalized_estimates
    );
}

#[test]
fn covariance_uses_optimized_nonlinear_point_before_all_variables_expire() {
    let factor = |target| {
        SmootherFactor::analytic(&[0], 1, Default::default(), move |v| {
            Ok((
                DVector::from_element(1, v[0][0].powi(2) - target),
                vec![DMatrix::from_element(1, 1, 2. * v[0][0])],
            ))
        })
    };
    let mut s = smoother();
    s.update(
        0.,
        SmootherUpdate {
            variables: vec![variable(0, Some(0.), 1.)],
            factors: vec![factor(4.)],
            ..Default::default()
        },
    )
    .unwrap();
    let (report, covariance) = s
        .update_with_covariance(
            2.,
            SmootherUpdate {
                factors: vec![factor(16.)],
                ..Default::default()
            },
            &[0],
            &Default::default(),
        )
        .unwrap();
    let value = report.marginalized_estimates[&0][0];
    assert!((value.powi(2) - 10.).abs() < 1e-7);
    assert!((covariance.matrix[(0, 0)] - 1. / (8. * value.powi(2))).abs() < 1e-12);
    assert!(report.estimates.is_empty());
    assert_eq!(s.variable_count(), 0);
}

#[test]
fn covariance_rejects_unobservable_unqueried_variables() {
    let mut s = FixedLagSmoother::new(Default::default()).unwrap();
    s.update(
        0.,
        SmootherUpdate {
            variables: vec![variable(0, None, 0.), variable(99, None, 0.)],
            factors: vec![observation(0, 0., 1.)],
            ..Default::default()
        },
    )
    .unwrap();
    assert!(matches!(
        s.update_with_covariance(1., Default::default(), &[0], &Default::default()),
        Err(Error::RankDeficient {
            rank: 1,
            parameters: 2
        })
    ));
    assert_eq!(s.last_time(), Some(0.));
    assert_eq!(s.variable_count(), 2);
}

#[test]
fn invalid_queries_do_not_register_new_variables_or_refresh_existing_timestamps() {
    let mut s = coupled();
    let old = [
        s.estimate(0).unwrap().clone(),
        s.estimate(1).unwrap().clone(),
    ];
    let update = |refresh: bool| {
        let mut fixed = variable(9, Some(2.), 7.);
        fixed.fixed = true;
        SmootherUpdate {
            variables: vec![variable(2, Some(2.), 0.), fixed],
            factors: vec![
                observation(0, 10., 1.),
                observation(2, 5., 2.),
                observation(9, 7., 1.),
            ],
            timestamps: if refresh { vec![(0, 2.)] } else { vec![] },
        }
    };
    // Query a newly supplied fixed key, a duplicate new active key, or a
    // missing key. All failures happen after preparing the proposed update.
    for keys in [vec![9], vec![2, 2], vec![2, 999]] {
        assert!(
            s.update_with_covariance(2., update(true), &keys, &Default::default())
                .is_err()
        );
        assert_eq!(s.last_time(), Some(1.));
        assert_eq!(s.variable_count(), 2);
        assert_eq!(s.factor_count(), 3);
        assert_eq!(s.estimate(0).unwrap(), &old[0]);
        assert_eq!(s.estimate(1).unwrap(), &old[1]);
        assert!(s.estimate(2).is_none());
        assert!(s.estimate(9).is_none());
    }
    // Omitting the refresh on retry must still expire key 0. A leaked timestamp
    // from a failed proposal would incorrectly retain it.
    let (report, covariance) = s
        .update_with_covariance(2., update(false), &[2, 0], &Default::default())
        .unwrap();
    assert_eq!(report.marginalized, vec![0]);
    assert_eq!(covariance.indices, vec![2, 0]);
    let expected = DMatrix::from_diagonal(&DVector::from_vec(vec![0.25, 2. / 11.]));
    assert!((covariance.matrix - expected).norm() < 1e-12);
    assert!((report.marginalized_estimates[&0][0] - 2.).abs() < 1e-10);
    assert!((report.estimates[&2][0] - 5.).abs() < 1e-10);
    assert_eq!(report.estimates[&9][0], 7.);
    assert_eq!(s.last_time(), Some(2.));
    assert_eq!(s.variable_count(), 3);
    assert_eq!(s.factor_count(), 4);
}
