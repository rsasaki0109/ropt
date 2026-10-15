use ropt::{covariance::*, factor::*, fixed_lag::*, marginalization::*, *};
use std::{cell::Cell, rc::Rc};

#[test]
fn compression_keeps_independently_scaled_columns_and_constant_residual() {
    for small in [1., 1e-100, 1e-170, 1e-200, 1e-310, f64::from_bits(1)] {
        let j = DMatrix::from_row_slice(
            4,
            3,
            &[1., 0., 0., 0., 3. * small, 0., 0., 0., 4., 0., 0., 0.],
        );
        let residual = DVector::from_vec(vec![0., 0., 0., 1e150]);
        for eliminated in [0, 1] {
            let prior = marginalize_linearized(
                &j,
                &residual,
                eliminated,
                &MarginalizationOptions::default(),
            )
            .unwrap();
            // Compare normalized Gram entries, since raw information small^2
            // can underflow and the huge constant would mask a lost column.
            let scales = [1., small, 1.];
            let normalized = DMatrix::from_fn(
                prior.jacobian.nrows(),
                prior.jacobian.ncols(),
                |row, column| prior.jacobian[(row, column)] / scales[column + eliminated],
            );
            let expected =
                DMatrix::from_diagonal(&DVector::from_column_slice(&[1., 9., 16.][eliminated..]));
            assert!(
                (normalized.transpose() * &normalized - expected).norm() < 1e-12,
                "small={small}, eliminated={eliminated}: {prior:?}"
            );
            let normalized_residual = &prior.residual / 1e150;
            assert!((normalized_residual.norm_squared() - 1.).abs() < 1e-12);
            assert!((normalized.transpose() * normalized_residual).norm() < 1e-12);
        }
    }
}

#[test]
fn rotated_elimination_preserves_dependent_retained_information() {
    for small in [1., 1e-100, 1e-170, 1e-200] {
        // The eliminated column [1,1,0] is orthogonal to [s,-s,0].
        // Retained columns are dependent; their normalized Gram matrix is
        // [[2,4],[4,8]], and the constant residual is orthogonal to both.
        let j = DMatrix::from_row_slice(
            3,
            3,
            &[1., small, 2. * small, 1., -small, -2. * small, 0., 0., 0.],
        );
        let prior = marginalize_linearized(
            &j,
            &DVector::from_vec(vec![0., 0., 1e150]),
            1,
            &MarginalizationOptions::default(),
        )
        .unwrap();
        let normalized = &prior.jacobian / small;
        let gram = normalized.transpose() * &normalized;
        assert!((gram - DMatrix::from_row_slice(2, 2, &[2., 4., 4., 8.])).norm() < 1e-12);
        let normalized_residual = &prior.residual / 1e150;
        assert!((normalized_residual.norm_squared() - 1.).abs() < 1e-12);
        assert!((normalized.transpose() * normalized_residual).norm() < 1e-12);
    }
}

fn smoother(
    loss: MarginalizationLoss,
    calls: Rc<Cell<usize>>,
    fail_during_marginalization: Rc<Cell<bool>>,
) -> FixedLagSmoother {
    let mut smoother = FixedLagSmoother::new(FixedLagOptions {
        lag: 1.,
        marginalization_loss: loss,
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
                        let call = calls.get() + 1;
                        calls.set(call);
                        if fail_during_marginalization.get() && call == 2 {
                            return Err(Error::Evaluation(
                                "injected marginalization failure".into(),
                            ));
                        }
                        Ok((
                            DVector::from_vec(vec![
                                values[0][0],
                                1e-200 * (values[1][0] - values[0][0]),
                                1e150,
                            ]),
                            vec![
                                DMatrix::from_column_slice(3, 1, &[1., -1e-200, 0.]),
                                DMatrix::from_column_slice(3, 1, &[0., 1e-200, 0.]),
                            ],
                        ))
                    },
                )],
                ..Default::default()
            },
        )
        .unwrap();
    smoother
}

fn check_retained_information(smoother: &FixedLagSmoother) {
    assert_eq!(smoother.variable_count(), 1);
    assert_eq!(smoother.factor_count(), 1);
    assert!(smoother.estimate(0).is_none());
    assert_eq!(smoother.estimate(1).unwrap()[0], 0.);
    assert_eq!(smoother.last_time(), Some(2.));
    for input in [CovarianceInput::Dense, CovarianceInput::SparseToDense] {
        let covariance = smoother
            .marginal_covariance(
                &[1],
                &CovarianceOptions {
                    input,
                    // Scale the reported covariance into the representable
                    // range; the undamped square-root prior is still 1e-200.
                    variance_scale: 1e-200,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(covariance.rank, 1);
        assert!((covariance.matrix[(0, 0)] / 1e200 - 1.).abs() < 1e-12);
    }
}

#[test]
fn fixed_lag_keeps_tiny_separator_information_beside_a_large_constant() {
    for loss in [
        MarginalizationLoss::Gaussian,
        MarginalizationLoss::FrozenIrls,
    ] {
        let mut smoother = smoother(loss, Rc::new(Cell::new(0)), Rc::new(Cell::new(false)));
        let report = smoother.update(2., Default::default()).unwrap();
        assert_eq!(report.marginalized, vec![0]);
        assert_eq!(report.marginalization.eliminated_tangent, 1);
        assert_eq!(report.marginalization.separator_tangent, 1);
        assert!(report.iterations.is_empty());
        assert!((report.initial_cost / 5e299 - 1.).abs() < 1e-12);
        assert_eq!(report.initial_cost, report.final_cost);
        check_retained_information(&smoother);
    }
}

#[test]
fn marginalization_failure_rolls_back_and_a_retry_preserves_information() {
    for loss in [
        MarginalizationLoss::Gaussian,
        MarginalizationLoss::FrozenIrls,
    ] {
        let calls = Rc::new(Cell::new(0));
        let fail = Rc::new(Cell::new(false));
        let mut smoother = smoother(loss, calls.clone(), fail.clone());
        calls.set(0);
        fail.set(true);
        assert_eq!(
            smoother.update(2., Default::default()).unwrap_err(),
            Error::Evaluation("injected marginalization failure".into())
        );
        assert_eq!(calls.get(), 2);
        assert_eq!(smoother.variable_count(), 2);
        assert_eq!(smoother.factor_count(), 1);
        assert_eq!(smoother.last_time(), Some(0.));
        assert_eq!(smoother.estimate(0).unwrap()[0], 0.);
        assert_eq!(smoother.estimate(1).unwrap()[0], 0.);
        fail.set(false);
        smoother.update(2., Default::default()).unwrap();
        check_retained_information(&smoother);
    }
}

#[test]
fn compression_still_rejects_an_unrepresentable_output() {
    // A one-column QR requires norm([MAX,MAX]), which exceeds f64 range.
    assert_eq!(
        marginalize_linearized(
            &DMatrix::from_column_slice(2, 1, &[f64::MAX, f64::MAX]),
            &DVector::zeros(2),
            0,
            &MarginalizationOptions::default()
        )
        .unwrap_err(),
        Error::NonFinite
    );
}
