use ropt::{covariance::*, factor::*, fixed_lag::*, marginalization::*, *};
use std::{cell::Cell, rc::Rc};

#[test]
fn projection_preserves_tiny_tails_beside_large_eliminable_components() {
    for (large, tiny) in [
        (1., 1.),
        (1e308, 1.),
        (f64::MAX, 1e-200),
        (f64::MAX, f64::from_bits(1)),
    ] {
        let prior = marginalize_linearized(
            &DMatrix::from_row_slice(2, 2, &[1., large, 0., tiny]),
            &DVector::from_vec(vec![large, tiny]),
            1,
            &MarginalizationOptions::default(),
        )
        .unwrap();
        // x can cancel the first equation for any retained y. The profile
        // residual is tiny*(y+1), including when tiny is the smallest subnormal.
        assert!((prior.jacobian[(0, 0)].abs() / tiny - 1.).abs() < 1e-12);
        assert!((prior.residual[0] / prior.jacobian[(0, 0)] - 1.).abs() < 1e-12);
    }
}

#[test]
fn rotated_projection_keeps_finite_information_after_intermediate_overflow() {
    for large in [1., 1e308, f64::MAX] {
        let prior = marginalize_linearized(
            &DMatrix::from_row_slice(3, 2, &[1., large, 1., 0., 0., 0.]),
            &DVector::from_vec(vec![0., 0., 1.]),
            1,
            &MarginalizationOptions::default(),
        )
        .unwrap();
        // Projecting [large,0,0] orthogonal to [1,1,0] leaves squared norm
        // large^2/2. Normalize before squaring to keep the oracle finite.
        let normalized = prior.jacobian.column(0) / large;
        assert!((normalized.norm_squared() - 0.5).abs() < 1e-12);
        assert!((prior.residual.norm_squared() - 1.).abs() < 1e-12);
        assert!(normalized.dot(&prior.residual).abs() < 1e-12);
    }
}

#[test]
fn multiple_eliminated_directions_restore_only_the_retained_rows() {
    for tail in [1., 1e-200, f64::from_bits(1)] {
        let prior = marginalize_linearized(
            &DMatrix::from_row_slice(3, 3, &[1., 0., f64::MAX, 0., 1., f64::MAX, 0., 0., tail]),
            &DVector::from_vec(vec![f64::MAX, -f64::MAX, tail]),
            2,
            &MarginalizationOptions::default(),
        )
        .unwrap();
        assert!((prior.jacobian[(0, 0)].abs() / tail - 1.).abs() < 1e-12);
        assert!((prior.residual[0] / prior.jacobian[(0, 0)] - 1.).abs() < 1e-12);
    }
}

#[test]
fn small_elimination_coefficients_can_project_large_columns_to_finite_values() {
    let large = f64::MAX;
    let small = 1e-200;
    let prior = marginalize_linearized(
        &DMatrix::from_row_slice(2, 2, &[1., large, small, 0.]),
        &DVector::zeros(2),
        1,
        &MarginalizationOptions::default(),
    )
    .unwrap();
    assert!((prior.jacobian[(0, 0)].abs() / (large * small) - 1.).abs() < 1e-12);
    assert_eq!(prior.residual[0], 0.);
}

#[test]
fn fixed_lag_rolls_back_unrepresentable_output_then_retries_large_projection() {
    for model in [
        MarginalizationLoss::Gaussian,
        MarginalizationLoss::FrozenIrls,
    ] {
        let fault = Rc::new(Cell::new(false));
        let evaluate_fault = fault.clone();
        let mut smoother = FixedLagSmoother::new(FixedLagOptions {
            lag: 1.,
            marginalization_loss: model,
            ..Default::default()
        })
        .unwrap();
        smoother
            .update(
                0.,
                SmootherUpdate {
                    variables: vec![
                        WindowVariable {
                            key: 0,
                            kind: VariableKind::Euclidean(1),
                            value: DVector::zeros(1),
                            timestamp: Some(0.),
                            fixed: false,
                        },
                        WindowVariable {
                            key: 1,
                            kind: VariableKind::Euclidean(1),
                            value: DVector::zeros(1),
                            timestamp: None,
                            fixed: false,
                        },
                    ],
                    factors: vec![SmootherFactor::analytic(
                        &[0, 1],
                        3,
                        Default::default(),
                        move |values| {
                            let (residual, kept) = if evaluate_fault.get() {
                                // Fault injection: QR of the retained [MAX,MAX]
                                // requires an unrepresentable output coefficient.
                                (
                                    vec![
                                        values[0][0],
                                        f64::MAX * values[1][0],
                                        f64::MAX * values[1][0],
                                    ],
                                    vec![0., f64::MAX, f64::MAX],
                                )
                            } else {
                                (
                                    vec![values[0][0] + f64::MAX * values[1][0], values[1][0], 0.],
                                    vec![f64::MAX, 1., 0.],
                                )
                            };
                            Ok((
                                DVector::from_vec(residual),
                                vec![
                                    DMatrix::from_column_slice(3, 1, &[1., 0., 0.]),
                                    DMatrix::from_column_slice(3, 1, &kept),
                                ],
                            ))
                        },
                    )],
                    ..Default::default()
                },
            )
            .unwrap();
        fault.set(true);
        assert_eq!(
            smoother.update(2., Default::default()).unwrap_err(),
            Error::NonFinite
        );
        assert_eq!(smoother.variable_count(), 2);
        assert_eq!(smoother.factor_count(), 1);
        assert_eq!(smoother.last_time(), Some(0.));
        assert_eq!(smoother.estimate(0).unwrap()[0], 0.);
        assert_eq!(smoother.estimate(1).unwrap()[0], 0.);
        fault.set(false);
        let report = smoother.update(2., Default::default()).unwrap();
        assert_eq!(report.marginalized, vec![0]);
        assert_eq!(report.marginalization.eliminated_tangent, 1);
        assert_eq!(report.marginalization.separator_tangent, 1);
        assert!(report.iterations.is_empty());
        assert_eq!(smoother.last_time(), Some(2.));
        assert!(smoother.estimate(0).is_none());
        assert_eq!(smoother.variable_count(), 1);
        for input in [CovarianceInput::Dense, CovarianceInput::SparseToDense] {
            let covariance = smoother
                .marginal_covariance(
                    &[1],
                    &CovarianceOptions {
                        input,
                        ..Default::default()
                    },
                )
                .unwrap();
            assert_eq!(covariance.rank, 1);
            assert!((covariance.matrix[(0, 0)] - 1.).abs() < 1e-12);
        }
    }
}

#[test]
fn unrepresentable_retained_projection_still_returns_nonfinite() {
    assert_eq!(
        marginalize_linearized(
            &DMatrix::from_row_slice(3, 2, &[1., 0., 0., f64::MAX, 0., f64::MAX]),
            &DVector::zeros(3),
            1,
            &MarginalizationOptions::default()
        )
        .unwrap_err(),
        Error::NonFinite
    );
}
