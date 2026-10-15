use ropt::{factor::*, fixed_lag::*, isam::*, *};
use std::{cell::Cell, rc::Rc};
fn variable(key: u64, value: f64) -> WindowVariable {
    WindowVariable {
        key,
        kind: VariableKind::Euclidean(1),
        value: DVector::from_element(1, value),
        timestamp: None,
        fixed: false,
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
#[test]
fn reuse_is_explicit_reduces_trials_and_ordinary_updates_leave_it_unchanged() {
    let options = IsamOptions {
        relinearization_threshold: 100.,
        relinearization_interval: usize::MAX,
        ..Default::default()
    };
    let initial = options.batch_solver.initial_damping;
    let mut cold = IsamSmoother::new(options.clone()).unwrap();
    let mut warm = IsamSmoother::new(options)
        .unwrap()
        .with_batch_damping_reuse()
        .unwrap();
    let (mut cold_trials, mut warm_trials) = (0, 0);
    for key in 0..12 {
        let make = || SmootherUpdate {
            variables: vec![variable(key, 0.)],
            factors: vec![observation(key, key as f64 + 1.)],
            ..Default::default()
        };
        let prior = warm.next_batch_damping().unwrap();
        let a = cold.update(make(), true).unwrap();
        let b = warm.update(make(), true).unwrap();
        assert_eq!(cold.next_batch_damping(), None);
        assert_eq!(a.batch_initial_damping, Some(initial));
        assert_eq!(b.batch_initial_damping, Some(prior));
        assert_eq!(warm.next_batch_damping(), b.batch_final_damping);
        assert!(b.batch_final_damping.unwrap().is_finite() && b.batch_final_damping.unwrap() > 0.);
        for k in 0..=key {
            assert!((a.estimates[&k][0] - b.estimates[&k][0]).abs() < 1e-8);
        }
        cold_trials += a.batch_trials;
        warm_trials += b.batch_trials;
    }
    assert!(warm_trials < cold_trials);
    let prior = warm.next_batch_damping();
    let report = warm.update(SmootherUpdate::default(), false).unwrap();
    assert_eq!(report.kind, IsamUpdateKind::Incremental);
    assert_eq!(report.batch_trials, 0);
    assert_eq!(report.batch_initial_damping, None);
    assert_eq!(report.batch_final_damping, None);
    assert_eq!(warm.next_batch_damping(), prior);
    assert_eq!(
        warm.with_batch_damping_reuse()
            .unwrap()
            .next_batch_damping(),
        prior
    );
}
#[test]
fn failed_final_fill_does_not_commit_the_adapted_damping_or_history() {
    let mut s = IsamSmoother::new(IsamOptions {
        qr: ropt::incremental_qr::IncrementalQrOptions {
            max_entries: 2,
            ..Default::default()
        },
        batch_solver: Options {
            linear_solver: LinearSolver::SparseLdl(Default::default()),
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap()
    .with_batch_damping_reuse()
    .unwrap();
    s.update(
        SmootherUpdate {
            variables: vec![variable(0, 0.)],
            factors: vec![observation(0, 1.)],
            ..Default::default()
        },
        true,
    )
    .unwrap();
    let damping = s.next_batch_damping();
    let mean = s.estimate(0).unwrap().clone();
    let result = s.update(
        SmootherUpdate {
            variables: vec![variable(1, 0.)],
            factors: vec![SmootherFactor::analytic(
                &[0, 1],
                1,
                Default::default(),
                |v| {
                    Ok((
                        DVector::from_element(1, v[0][0] + v[1][0] - 3.),
                        vec![DMatrix::identity(1, 1), DMatrix::identity(1, 1)],
                    ))
                },
            )],
            ..Default::default()
        },
        true,
    );
    assert!(matches!(result, Err(Error::SparseFillLimit { limit: 2 })));
    assert_eq!(s.next_batch_damping(), damping);
    assert_eq!(s.estimate(0), Some(&mean));
    assert!(s.estimate(1).is_none());
    assert_eq!(s.variable_count(), 1);
    assert_eq!(s.factor_count(), 1);
    let r = s
        .update(
            SmootherUpdate {
                factors: vec![observation(0, 2.)],
                ..Default::default()
            },
            true,
        )
        .unwrap();
    assert_eq!(r.batch_initial_damping, damping);
    assert!((r.estimates[&0][0] - 1.5).abs() < 1e-8);
}
#[test]
fn abrupt_nonlinear_change_still_rejects_overshoots_and_converges() {
    let mut s = IsamSmoother::new(Default::default())
        .unwrap()
        .with_batch_damping_reuse()
        .unwrap();
    let weak = SmootherFactor::analytic(
        &[0],
        1,
        FactorOptions {
            noise: ropt::noise::NoiseModel::isotropic(100.).unwrap(),
            ..Default::default()
        },
        |v| {
            Ok((
                DVector::from_element(1, v[0][0] - 0.1),
                vec![DMatrix::identity(1, 1)],
            ))
        },
    );
    s.update(
        SmootherUpdate {
            variables: vec![variable(0, 0.)],
            factors: vec![weak],
            ..Default::default()
        },
        true,
    )
    .unwrap();
    let carried = s.next_batch_damping().unwrap();
    assert!(carried < 1e-3);
    let over = Rc::new(Cell::new(false));
    let probe = over.clone();
    let r = s
        .update(
            SmootherUpdate {
                factors: vec![SmootherFactor::analytic(
                    &[0],
                    1,
                    Default::default(),
                    move |v| {
                        if v[0][0] > 10. {
                            probe.set(true);
                        }
                        Ok((
                            DVector::from_element(1, v[0][0].powi(2) - 4.),
                            vec![DMatrix::from_element(1, 1, 2. * v[0][0])],
                        ))
                    },
                )],
                ..Default::default()
            },
            true,
        )
        .unwrap();
    assert!(over.get());
    assert_eq!(r.batch_initial_damping, Some(carried));
    assert!(r.batch_trials > 1);
    assert!((r.estimates[&0][0] - 2.).abs() < 1e-3);
    assert!(r.linearized_cost < 1e-3);
}
#[test]
fn invalid_reuse_configuration_is_rejected_and_gn_reports_no_lm_damping() {
    let options = IsamOptions {
        batch_solver: Options {
            algorithm: Algorithm::GaussNewton,
            linear_solver: LinearSolver::DenseQr(Default::default()),
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(matches!(
        IsamSmoother::new(options.clone())
            .unwrap()
            .with_batch_damping_reuse(),
        Err(Error::InvalidOptions)
    ));
    let mut s = IsamSmoother::new(options).unwrap();
    let r = s
        .update(
            SmootherUpdate {
                variables: vec![variable(0, 0.)],
                factors: vec![observation(0, 1.)],
                ..Default::default()
            },
            true,
        )
        .unwrap();
    assert_eq!(r.batch_initial_damping, None);
    assert_eq!(r.batch_final_damping, None);
    for value in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(matches!(
            IsamSmoother::new(IsamOptions {
                batch_solver: Options {
                    initial_damping: value,
                    ..Default::default()
                },
                ..Default::default()
            })
            .unwrap()
            .with_batch_damping_reuse(),
            Err(Error::InvalidOptions)
        ));
    }
}
