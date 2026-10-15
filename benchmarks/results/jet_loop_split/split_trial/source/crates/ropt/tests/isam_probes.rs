use ropt::{factor::*, fixed_lag::*, isam::*, *};
use std::{cell::Cell, rc::Rc};
fn variable(key: u64, value: f64, fixed: bool) -> WindowVariable {
    WindowVariable {
        key,
        kind: VariableKind::Euclidean(1),
        value: DVector::from_element(1, value),
        timestamp: None,
        fixed,
    }
}
fn square(target: f64, fail: Rc<Cell<bool>>) -> SmootherFactor {
    SmootherFactor::analytic(&[4], 1, Default::default(), move |v| {
        if fail.get() {
            return Err(Error::Evaluation(
                "probe or requested callback rejected".into(),
            ));
        }
        let x = v[0][0];
        Ok((
            DVector::from_element(1, x * x - target),
            vec![DMatrix::from_element(1, 1, 2. * x)],
        ))
    })
}
#[test]
fn nonlinear_probe_matches_independent_original_polynomial_and_leaves_schedule_unchanged() {
    let fail = Rc::new(Cell::new(false));
    let mut smoother = IsamSmoother::new(IsamOptions {
        relinearization_interval: 2,
        ..Default::default()
    })
    .unwrap()
    .with_selective_bayes_tree(1)
    .unwrap();
    let first = smoother
        .update(
            SmootherUpdate {
                variables: vec![variable(4, 1., false), variable(1, 2., true)],
                factors: vec![
                    square(1., fail),
                    SmootherFactor::analytic(&[1], 1, Default::default(), |v| {
                        Ok((
                            DVector::from_element(1, v[0][0]),
                            vec![DMatrix::identity(1, 1)],
                        ))
                    }),
                ],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    for norm in [GradientNorm::Absolute, GradientNorm::ColumnNormalized] {
        let summary = smoother.nonlinear_summary(norm).unwrap();
        assert_eq!(summary.cost, 2.);
        assert_eq!(summary.gradient_norm, 0.);
        assert_eq!(summary.variable_gradients.len(), 1);
        assert_eq!(summary.variable_gradients[&4], 0.);
    }
    assert_eq!(smoother.estimate(4), first.estimates.get(&4));
    let request = smoother.relinearize_selected(&[4]).unwrap();
    assert_eq!(
        request.relinearization_reason,
        Some(IsamRelinearizationReason::Requested)
    );
    let next = smoother.update(Default::default(), false).unwrap();
    assert_eq!(
        next.relinearization_reason,
        Some(IsamRelinearizationReason::Interval)
    );
}
#[test]
fn requested_keys_can_remove_gradient_left_below_anchor_threshold() {
    let mut smoother = IsamSmoother::new(IsamOptions {
        relinearization_interval: 100,
        ..Default::default()
    })
    .unwrap()
    .with_selective_bayes_tree(4)
    .unwrap();
    smoother
        .update(
            SmootherUpdate {
                variables: vec![variable(4, 1., false)],
                factors: vec![square(1., Rc::new(Cell::new(false)))],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    smoother
        .update(
            SmootherUpdate {
                factors: vec![SmootherFactor::analytic(&[4], 1, Default::default(), |v| {
                    Ok((
                        DVector::from_element(1, v[0][0] - 3.),
                        vec![DMatrix::identity(1, 1)],
                    ))
                })],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    let initial = smoother
        .nonlinear_summary(GradientNorm::ColumnNormalized)
        .unwrap();
    assert!(initial.gradient_norm > 1e-4);
    for _ in 0..8 {
        let probe = smoother
            .nonlinear_summary(GradientNorm::ColumnNormalized)
            .unwrap();
        if probe.gradient_norm <= 1e-8 {
            break;
        }
        smoother.relinearize_selected(&[4]).unwrap();
    }
    let x = smoother.estimate(4).unwrap()[0];
    let g = 2. * x * (x * x - 1.) + x - 3.;
    let summary = smoother
        .nonlinear_summary(GradientNorm::ColumnNormalized)
        .unwrap();
    assert!((summary.cost - 0.5 * ((x * x - 1.).powi(2) + (x - 3.).powi(2))).abs() < 1e-14);
    assert!((summary.gradient_norm - g.abs() / (4. * x * x + 1.).sqrt()).abs() < 1e-14);
    assert!(summary.gradient_norm < 1e-8);
}
#[test]
fn probe_and_request_errors_preserve_owned_history_and_cached_model() {
    let fail = Rc::new(Cell::new(false));
    let mut smoother = IsamSmoother::new(Default::default())
        .unwrap()
        .with_selective_bayes_tree(1)
        .unwrap();
    smoother
        .update(
            SmootherUpdate {
                variables: vec![variable(4, 1., false), variable(8, 2., true)],
                factors: vec![square(1., fail.clone())],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    for keys in [vec![], vec![4, 4], vec![99], vec![8]] {
        assert!(smoother.relinearize_selected(&keys).is_err());
    }
    fail.set(true);
    assert!(smoother.nonlinear_summary(GradientNorm::Absolute).is_err());
    assert!(smoother.relinearize_selected(&[4]).is_err());
    assert_eq!(smoother.factor_count(), 1);
    assert_eq!(smoother.estimate(4).unwrap()[0], 1.);
    fail.set(false);
    assert_eq!(
        smoother
            .update(Default::default(), false)
            .unwrap()
            .linearized_cost,
        0.
    );
    assert!(
        IsamSmoother::new(Default::default())
            .unwrap()
            .relinearize_selected(&[4])
            .is_err()
    );
}
#[test]
fn fixed_only_probe_and_normalized_scale_overflow_have_explicit_results() {
    let mut fixed = IsamSmoother::new(Default::default()).unwrap();
    fixed
        .update(
            SmootherUpdate {
                variables: vec![variable(4, 3., true)],
                factors: vec![square(1., Rc::new(Cell::new(false)))],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    let probe = fixed
        .nonlinear_summary(GradientNorm::ColumnNormalized)
        .unwrap();
    assert_eq!(probe.cost, 32.);
    assert_eq!(probe.gradient_norm, 0.);
    assert!(probe.variable_gradients.is_empty());
    // Probe callbacks can vary their scale for this explicit failure test; the
    // installed observable Gaussian model itself remains finite and unchanged.
    let huge = Rc::new(Cell::new(false));
    let flag = huge.clone();
    let mut smoother = IsamSmoother::new(Default::default()).unwrap();
    smoother
        .update(
            SmootherUpdate {
                variables: vec![variable(4, 0., false)],
                factors: vec![SmootherFactor::analytic(
                    &[4],
                    2,
                    Default::default(),
                    move |v| {
                        let scale = if flag.get() { 1.5e308 } else { 1. };
                        Ok((
                            DVector::from_element(2, v[0][0] * scale),
                            vec![DMatrix::from_element(2, 1, scale)],
                        ))
                    },
                )],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    huge.set(true);
    assert!(matches!(
        smoother.nonlinear_summary(GradientNorm::ColumnNormalized),
        Err(Error::NonFinite)
    ));
    assert_eq!(smoother.estimate(4).unwrap()[0], 0.);
}
