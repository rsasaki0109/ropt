use ropt::{covariance::*, factor::*, fixed_lag::*, *};
use std::{cell::Cell, rc::Rc};

fn variable(key: u64, time: Option<f64>, fixed: bool) -> WindowVariable {
    WindowVariable {
        key,
        kind: VariableKind::Euclidean(1),
        value: DVector::zeros(1),
        timestamp: time,
        fixed,
    }
}
fn observation(key: u64, target: f64, calls: &Rc<Cell<usize>>) -> SmootherFactor {
    let calls = calls.clone();
    SmootherFactor::analytic(&[key], 1, Default::default(), move |v| {
        calls.set(calls.get() + 1);
        Ok((
            DVector::from_element(1, v[0][0] - target),
            vec![DMatrix::identity(1, 1)],
        ))
    })
}
fn initial(calls: &Rc<Cell<usize>>) -> FixedLagSmoother {
    let mut smoother = FixedLagSmoother::new(FixedLagOptions {
        lag: 1.,
        ..Default::default()
    })
    .unwrap();
    smoother
        .update(
            0.,
            SmootherUpdate {
                variables: vec![variable(0, Some(0.), false), variable(99, None, true)],
                factors: vec![observation(0, 2., calls)],
                ..Default::default()
            },
        )
        .unwrap();
    calls.set(0);
    smoother
}
fn proposal(calls: &Rc<Cell<usize>>) -> SmootherUpdate {
    SmootherUpdate {
        variables: vec![variable(2, Some(2.), false), variable(3, Some(2.), true)],
        factors: vec![observation(0, 4., calls), observation(2, 6., calls)],
        timestamps: vec![(0, 2.)],
    }
}
fn unchanged(s: &FixedLagSmoother, calls: &Rc<Cell<usize>>) {
    assert_eq!(
        calls.get(),
        0,
        "invalid covariance request evaluated a factor"
    );
    assert_eq!(s.last_time(), Some(0.));
    assert_eq!(s.variable_count(), 2);
    assert_eq!(s.factor_count(), 1);
    assert!((s.estimate(0).unwrap()[0] - 2.).abs() < 1e-7);
    assert_eq!(s.estimate(99).unwrap()[0], 0.);
    assert!(s.estimate(2).is_none() && s.estimate(3).is_none());
}
fn retry_without_refresh(s: &mut FixedLagSmoother, calls: &Rc<Cell<usize>>) {
    let mut update = proposal(calls);
    update.timestamps.clear();
    let (report, covariance) = s
        .update_with_covariance(2., update, &[2, 0], &Default::default())
        .unwrap();
    assert!(calls.get() > 0);
    assert_eq!(report.marginalized, vec![0]);
    assert!((report.marginalized_estimates[&0][0] - 3.).abs() < 1e-7);
    assert!((report.estimates[&2][0] - 6.).abs() < 1e-7);
    assert!(
        (covariance.matrix - DMatrix::from_diagonal(&DVector::from_vec(vec![1., 0.5]))).norm()
            < 1e-12
    );
}

#[test]
fn invalid_covariance_keys_never_evaluate_old_or_proposed_factors() {
    let calls = Rc::new(Cell::new(0));
    let mut s = initial(&calls);
    for keys in [
        vec![],
        vec![999],
        vec![2, 999],
        vec![2, 2],
        vec![99],
        vec![3],
    ] {
        assert!(
            s.update_with_covariance(2., proposal(&calls), &keys, &Default::default())
                .is_err()
        );
        unchanged(&s, &calls);
    }
    retry_without_refresh(&mut s, &calls);
}

#[test]
fn invalid_covariance_options_never_evaluate_and_retry_keeps_expiry_schedule() {
    let calls = Rc::new(Cell::new(0));
    let mut s = initial(&calls);
    let options = [
        CovarianceOptions {
            relative_rank_tolerance: 0.,
            ..Default::default()
        },
        CovarianceOptions {
            relative_rank_tolerance: 1.,
            ..Default::default()
        },
        CovarianceOptions {
            relative_rank_tolerance: f64::NAN,
            ..Default::default()
        },
        CovarianceOptions {
            variance_scale: 0.,
            ..Default::default()
        },
        CovarianceOptions {
            variance_scale: f64::INFINITY,
            ..Default::default()
        },
        CovarianceOptions {
            svd_max_iterations: 0,
            ..Default::default()
        },
        CovarianceOptions {
            model: CovarianceModel::FrozenIrls {
                fallback_loss: Loss::Huber(0.),
            },
            ..Default::default()
        },
    ];
    for options in options {
        assert!(matches!(
            s.update_with_covariance(2., proposal(&calls), &[2, 0], &options),
            Err(Error::InvalidOptions)
        ));
        unchanged(&s, &calls);
    }
    retry_without_refresh(&mut s, &calls);
}
