use ropt::{Algorithm, DMatrix, DVector, LinearSolver, Options, factor::*, fixed_lag::*};

fn variable(key: u64, time: Option<f64>, value: f64, fixed: bool) -> WindowVariable {
    WindowVariable {
        key,
        kind: VariableKind::Euclidean(1),
        value: DVector::from_element(1, value),
        timestamp: time,
        fixed,
    }
}
fn observation(key: u64, target: f64) -> SmootherFactor {
    SmootherFactor::analytic(&[key], 1, Default::default(), move |v| {
        Ok((
            DVector::from_element(1, v[0][0] - target),
            vec![DMatrix::identity(1, 1)],
        ))
    })
}
fn between(a: u64, b: u64) -> SmootherFactor {
    SmootherFactor::analytic(&[a, b], 1, Default::default(), |v| {
        Ok((
            DVector::from_element(1, v[1][0] - v[0][0] - 1.),
            vec![-DMatrix::identity(1, 1), DMatrix::identity(1, 1)],
        ))
    })
}
fn initial() -> FixedLagSmoother {
    let mut s = FixedLagSmoother::new(FixedLagOptions {
        lag: 5.,
        solver: Options {
            algorithm: Algorithm::GaussNewton,
            linear_solver: LinearSolver::DenseQr(Default::default()),
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    s.update(
        2.,
        SmootherUpdate {
            variables: vec![
                variable(0, Some(0.), 0., true),
                variable(1, Some(1.), 1., false),
                variable(2, Some(2.), 2., false),
                variable(100, None, 10., false),
            ],
            factors: vec![
                observation(0, 0.),
                between(0, 1),
                between(1, 2),
                observation(100, 10.),
            ],
            ..Default::default()
        },
    )
    .unwrap();
    s
}

#[test]
fn shortening_defers_expiry_until_success_and_extension_preserves_prior_information() {
    let mut s = initial();
    s.set_lag(0.5).unwrap();
    assert_eq!(s.lag(), 0.5);
    assert_eq!(s.variable_count(), 4);
    assert_eq!(s.factor_count(), 4);
    assert_eq!(s.last_time(), Some(2.));
    assert_eq!(s.estimate(1).unwrap()[0], 1.);
    assert!(
        s.update_with_covariance(2., Default::default(), &[999], &Default::default())
            .is_err()
    );
    assert_eq!(s.lag(), 0.5);
    assert_eq!(s.variable_count(), 4);
    assert_eq!(s.factor_count(), 4);
    assert_eq!(s.last_time(), Some(2.));
    let (trimmed, covariance) = s
        .update_with_covariance(2., Default::default(), &[1, 2], &Default::default())
        .unwrap();
    assert_eq!(trimmed.marginalized, vec![0, 1]);
    assert_eq!(trimmed.marginalized_estimates[&0][0], 0.);
    assert_eq!(trimmed.marginalized_estimates[&1][0], 1.);
    assert_eq!(
        trimmed.estimates.keys().copied().collect::<Vec<_>>(),
        vec![2, 100]
    );
    assert!((covariance.matrix - DMatrix::from_row_slice(2, 2, &[1., 1., 1., 2.])).norm() < 1e-12);
    assert!(
        (s.marginal_covariance(&[2], &Default::default())
            .unwrap()
            .matrix[(0, 0)]
            - 2.)
            .abs()
            < 1e-12
    );

    s.set_lag(5.).unwrap();
    assert!(s.estimate(0).is_none() && s.estimate(1).is_none());
    let extended = s
        .update(
            3.,
            SmootherUpdate {
                variables: vec![variable(3, Some(3.), 0., false)],
                factors: vec![between(2, 3)],
                ..Default::default()
            },
        )
        .unwrap();
    assert!(extended.marginalized.is_empty());
    assert!((extended.estimates[&3][0] - 3.).abs() < 1e-10);
    assert_eq!(extended.estimates[&100][0], 10.);
    assert!(
        (s.marginal_covariance(&[3], &Default::default())
            .unwrap()
            .matrix[(0, 0)]
            - 3.)
            .abs()
            < 1e-12
    );
    assert_eq!(s.lag(), 5.);
}

#[test]
fn invalid_duration_does_not_change_configuration_or_committed_state() {
    let mut s = initial();
    for lag in [0., -0., -1., f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(s.set_lag(lag).is_err());
        assert_eq!(s.lag(), 5.);
        assert_eq!(s.last_time(), Some(2.));
        assert_eq!(s.variable_count(), 4);
        assert_eq!(s.factor_count(), 4);
        assert_eq!(s.estimate(1).unwrap()[0], 1.);
    }
    let unchanged = s.update(2., Default::default()).unwrap();
    assert!(unchanged.marginalized.is_empty());
    assert_eq!(unchanged.estimates.len(), 4);
}
