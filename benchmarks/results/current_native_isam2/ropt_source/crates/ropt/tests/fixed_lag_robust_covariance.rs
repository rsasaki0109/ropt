use ropt::{
    DMatrix, DVector, Error, LinearSolver, Loss, Options,
    covariance::{CovarianceModel, CovarianceOptions},
    factor::*,
    fixed_lag::*,
};
fn variable(key: u64, time: f64, value: f64) -> WindowVariable {
    WindowVariable {
        key,
        kind: VariableKind::Euclidean(1),
        value: DVector::from_element(1, value),
        timestamp: Some(time),
        fixed: false,
    }
}
fn robust_window(model: MarginalizationLoss) -> FixedLagSmoother {
    let mut s = FixedLagSmoother::new(FixedLagOptions {
        lag: 1.,
        marginalization_loss: model,
        solver: Options {
            loss: Loss::Huber(1.),
            linear_solver: LinearSolver::DenseQr(Default::default()),
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    let factors = [0., 0.2, 10.].map(|target| {
        SmootherFactor::analytic(&[0], 1, Default::default(), move |v| {
            Ok((
                DVector::from_element(1, v[0][0] - target),
                vec![DMatrix::identity(1, 1)],
            ))
        })
    });
    s.update(
        0.,
        SmootherUpdate {
            variables: vec![variable(0, 0., 0.6)],
            factors: factors.into(),
            ..Default::default()
        },
    )
    .unwrap();
    s
}
fn new_pose() -> SmootherUpdate {
    SmootherUpdate {
        variables: vec![variable(1, 2., 1.6)],
        factors: vec![SmootherFactor::analytic(
            &[0, 1],
            1,
            FactorOptions {
                loss: Some(Loss::Squared),
                ..Default::default()
            },
            |v| {
                Ok((
                    DVector::from_element(1, v[1][0] - v[0][0] - 1.),
                    vec![-DMatrix::identity(1, 1), DMatrix::identity(1, 1)],
                ))
            },
        )],
        ..Default::default()
    }
}
fn frozen(loss: Loss) -> CovarianceOptions {
    CovarianceOptions {
        model: CovarianceModel::FrozenIrls {
            fallback_loss: loss,
        },
        ..Default::default()
    }
}

#[test]
fn inherited_robust_loss_requires_matching_explicit_covariance_model() {
    let mut s = robust_window(MarginalizationLoss::FrozenIrls);
    let before = s.estimate(0).unwrap().clone();
    for options in [
        CovarianceOptions::default(),
        frozen(Loss::Squared),
        frozen(Loss::Huber(2.)),
        frozen(Loss::Cauchy(1.)),
    ] {
        assert!(s.marginal_covariance(&[0], &options).is_err());
        assert!(
            s.update_with_covariance(2., new_pose(), &[0, 1], &options)
                .is_err()
        );
        assert_eq!(s.estimate(0).unwrap(), &before);
        assert!(s.estimate(1).is_none());
        assert_eq!(s.last_time(), Some(0.));
        assert_eq!(s.factor_count(), 3);
    }
    let (report, covariance) = s
        .update_with_covariance(2., new_pose(), &[0, 1], &frozen(Loss::Huber(1.)))
        .unwrap();
    // Huber weights are [1,1,1/9.4], so Var(x)=1/(2+1/9.4)=47/99.
    let c = 47. / 99.;
    let expected = DMatrix::from_row_slice(2, 2, &[c, c, c, c + 1.]);
    assert!((covariance.matrix - expected).norm() < 1e-10);
    assert!((report.marginalized_estimates[&0][0] - 0.6).abs() < 1e-10);
    assert!((report.estimates[&1][0] - 1.6).abs() < 1e-10);
    // The carried prior explicitly overrides the solver's robust default with
    // Squared. It must not be rejected or downweighted a second time.
    for options in [CovarianceOptions::default(), frozen(Loss::Squared)] {
        let live = s.marginal_covariance(&[1], &options).unwrap();
        assert!((live.matrix[(0, 0)] - (c + 1.)).abs() < 1e-10);
    }
}

#[test]
fn successful_covariance_then_rejected_marginalization_rolls_back_everything() {
    let mut s = robust_window(MarginalizationLoss::Gaussian);
    let options = frozen(Loss::Huber(1.));
    let before = s.estimate(0).unwrap().clone();
    assert!(s.marginal_covariance(&[0], &options).is_ok());
    for _ in 0..2 {
        let error = s
            .update_with_covariance(2., new_pose(), &[0, 1], &options)
            .unwrap_err();
        assert!(
            matches!(error, Error::Evaluation(ref message) if message == "robust marginalization requires explicit FrozenIrls")
        );
        assert_eq!(s.estimate(0).unwrap(), &before);
        assert!(s.estimate(1).is_none());
        assert_eq!(s.variable_count(), 1);
        assert_eq!(s.factor_count(), 3);
        assert_eq!(s.last_time(), Some(0.));
    }
    // A valid retry can refresh the timestamp and defer marginalization.
    let mut retry = new_pose();
    retry.timestamps.push((0, 2.));
    let (report, covariance) = s
        .update_with_covariance(2., retry, &[0, 1], &options)
        .unwrap();
    assert!(report.marginalized.is_empty());
    assert_eq!(report.estimates.len(), 2);
    assert_eq!(covariance.rank, 2);
    assert_eq!(s.factor_count(), 4);
    assert_eq!(s.last_time(), Some(2.));
}
