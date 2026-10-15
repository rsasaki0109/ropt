use ropt::{DMatrix, DVector, Error, Options, factor::*, fixed_lag::*, isam::*, solve_local};
use std::cell::Cell;
use std::rc::Rc;
fn variable(key: u64, value: f64) -> WindowVariable {
    WindowVariable {
        key,
        kind: VariableKind::Euclidean(1),
        value: DVector::from_element(1, value),
        timestamp: None,
        fixed: false,
    }
}
fn observation(key: u64, measurement: f64, count: Rc<Cell<usize>>) -> SmootherFactor {
    SmootherFactor::analytic(&[key], 1, FactorOptions::default(), move |v| {
        count.set(count.get() + 1);
        Ok((
            &v[0] - DVector::from_element(1, measurement),
            vec![DMatrix::identity(1, 1)],
        ))
    })
}
#[test]
fn incremental_stream_matches_batch_and_never_reevaluates_old_factors() {
    let mut isam = IsamSmoother::new(IsamOptions {
        relinearization_threshold: 100.,
        relinearization_interval: 100,
        ..Default::default()
    })
    .unwrap();
    let mut batch = FactorGraph::new();
    let mut ids = Vec::new();
    let mut counts = Vec::new();
    for key in 0..12_u64 {
        let count = Rc::new(Cell::new(0));
        counts.push(count.clone());
        let measurement = key as f64 + 0.1 * (key as f64).sin();
        let id = batch
            .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
            .unwrap();
        ids.push(id);
        batch
            .add_analytic_factor(&[id], 1, move |v| {
                Ok((
                    &v[0] - DVector::from_element(1, measurement),
                    vec![DMatrix::identity(1, 1)],
                ))
            })
            .unwrap();
        let mut update = SmootherUpdate {
            variables: vec![variable(key, 0.)],
            factors: vec![observation(key, measurement, count)],
            ..Default::default()
        };
        if key > 0 {
            update.factors.push(SmootherFactor::analytic(
                &[key - 1, key],
                1,
                Default::default(),
                |v| {
                    Ok((
                        &v[1] - &v[0] - DVector::from_element(1, 1.),
                        vec![-DMatrix::identity(1, 1), DMatrix::identity(1, 1)],
                    ))
                },
            ));
            batch
                .add_analytic_factor(&[ids[key as usize - 1], id], 1, |v| {
                    Ok((
                        &v[1] - &v[0] - DVector::from_element(1, 1.),
                        vec![-DMatrix::identity(1, 1), DMatrix::identity(1, 1)],
                    ))
                })
                .unwrap();
        }
        let report = isam.update(update, false).unwrap();
        assert_eq!(report.kind, IsamUpdateKind::Incremental);
        assert_eq!(report.relinearization_reason, None);
        assert_eq!(report.factors_linearized, if key == 0 { 1 } else { 2 });
        let full = solve_local(&batch, batch.initial(), &Options::default()).unwrap();
        for (&k, x) in &report.estimates {
            assert!((x - batch.value(&full.x, ids[k as usize]).unwrap()).norm() < 1e-8);
        }
        assert!((report.linearized_cost - full.cost).abs() < 1e-10);
        assert!(counts.iter().all(|c| c.get() == 1));
    }
    let refreshed = isam.update(SmootherUpdate::default(), true).unwrap();
    assert_eq!(refreshed.kind, IsamUpdateKind::Relinearized);
    assert_eq!(refreshed.factors_linearized, 23);
    assert!(counts.iter().all(|c| c.get() > 1));
}
#[test]
fn threshold_and_periodic_refresh_converge_on_nonlinear_factors() {
    for (threshold, interval) in [(0.1, 100), (100., 1)] {
        let mut s = IsamSmoother::new(IsamOptions {
            relinearization_threshold: threshold,
            relinearization_interval: interval,
            ..Default::default()
        })
        .unwrap();
        let f = SmootherFactor::autodiff(&[0], 1, 1, Default::default(), |v| {
            Ok(vec![&v[0][0] * &v[0][0] - 4.])
        });
        let r = s
            .update(
                SmootherUpdate {
                    variables: vec![variable(0, 1.)],
                    factors: vec![f],
                    ..Default::default()
                },
                false,
            )
            .unwrap();
        assert_eq!(r.kind, IsamUpdateKind::Relinearized);
        assert_eq!(
            r.relinearization_reason,
            Some(if interval == 1 {
                IsamRelinearizationReason::Interval
            } else {
                IsamRelinearizationReason::Threshold
            })
        );
        assert!((r.estimates[&0][0] - 2.).abs() < 1e-8);
        assert_eq!(r.max_anchor_delta, 0.);
        assert!(r.linearized_cost < 1e-14);
    }
}

#[test]
fn scheduled_refresh_still_rejects_an_unobservable_final_model_and_rolls_back() {
    let mut s = IsamSmoother::new(IsamOptions {
        relinearization_threshold: 100.,
        ..Default::default()
    })
    .unwrap();
    s.update(
        SmootherUpdate {
            variables: vec![variable(0, 0.)],
            factors: vec![observation(0, 1., Rc::new(Cell::new(0)))],
            ..Default::default()
        },
        false,
    )
    .unwrap();
    let before = s.estimate(0).unwrap().clone();
    let result = s.update(
        SmootherUpdate {
            variables: vec![variable(1, 0.)],
            ..Default::default()
        },
        true,
    );
    assert!(matches!(
        result,
        Err(Error::RankDeficient {
            rank: 1,
            parameters: 2
        })
    ));
    assert_eq!(s.estimate(0), Some(&before));
    assert!(s.estimate(1).is_none());
    assert_eq!(s.variable_count(), 1);
    assert_eq!(s.factor_count(), 1);
    assert_eq!(
        s.update(SmootherUpdate::default(), false).unwrap().kind,
        IsamUpdateKind::Incremental
    );
}
#[test]
fn unobservable_invalid_and_failing_updates_preserve_cached_information() {
    let mut s = IsamSmoother::new(IsamOptions {
        relinearization_threshold: 100.,
        ..Default::default()
    })
    .unwrap();
    s.update(
        SmootherUpdate {
            variables: vec![variable(0, 0.)],
            factors: vec![observation(0, 1., Rc::new(Cell::new(0)))],
            ..Default::default()
        },
        false,
    )
    .unwrap();
    let before = s.estimate(0).unwrap().clone();
    for update in [
        SmootherUpdate {
            variables: vec![variable(1, 0.)],
            ..Default::default()
        },
        SmootherUpdate {
            variables: vec![variable(0, 0.)],
            ..Default::default()
        },
        SmootherUpdate {
            timestamps: vec![(0, 1.)],
            ..Default::default()
        },
        SmootherUpdate {
            factors: vec![observation(0, f64::NAN, Rc::new(Cell::new(0)))],
            ..Default::default()
        },
        SmootherUpdate {
            factors: vec![observation(99, 1., Rc::new(Cell::new(0)))],
            ..Default::default()
        },
    ] {
        assert!(s.update(update, false).is_err());
        assert_eq!(s.estimate(0).unwrap(), &before);
        assert_eq!(s.variable_count(), 1);
        assert_eq!(s.factor_count(), 1);
    }
    let refreshed = s.update(SmootherUpdate::default(), true).unwrap();
    assert!((refreshed.estimates[&0][0] - 1.).abs() < 1e-10);
    assert!(matches!(
        IsamSmoother::new(IsamOptions {
            relinearization_interval: 0,
            ..Default::default()
        }),
        Err(Error::InvalidOptions)
    ));
}
#[test]
fn failed_nonlinear_refresh_keeps_the_committed_estimate_and_factor_count() {
    let mut s = IsamSmoother::new(IsamOptions {
        relinearization_threshold: 100.,
        batch_solver: Options {
            max_iterations: 1,
            linear_solver: ropt::LinearSolver::DenseQr(Default::default()),
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    s.update(
        SmootherUpdate {
            variables: vec![variable(0, 0.)],
            factors: vec![observation(0, 1., Rc::new(Cell::new(0)))],
            ..Default::default()
        },
        false,
    )
    .unwrap();
    let f = SmootherFactor::autodiff(&[0], 1, 1, Default::default(), |v| {
        Ok(vec![&v[0][0] * &v[0][0] - 4.])
    });
    assert!(
        s.update(
            SmootherUpdate {
                factors: vec![f],
                ..Default::default()
            },
            true
        )
        .is_err()
    );
    assert!((s.estimate(0).unwrap()[0] - 1.).abs() < 1e-12);
    assert_eq!(s.factor_count(), 1);
}
#[test]
fn se3_updates_use_six_right_tangent_columns_and_refresh_matches_batch() {
    use ropt::{manifold::*, pose_graph::*};
    let kind = VariableKind::SE3;
    let root = SE3::exp(Tangent6::new(2., -1., 0.3, 0.1, 0.2, -0.1));
    let truth = root.retract(Tangent6::new(1., 0.2, -0.1, 0.1, -0.05, 0.1));
    let seed = truth.retract(Tangent6::repeat(0.005));
    let mut s = IsamSmoother::new(Default::default()).unwrap();
    let root_values = se3_values(root);
    let anchor = root_values.clone();
    s.update(
        SmootherUpdate {
            variables: vec![WindowVariable {
                key: 0,
                kind,
                value: root_values,
                timestamp: None,
                fixed: true,
            }],
            factors: vec![SmootherFactor::autodiff(
                &[0],
                6,
                6,
                Default::default(),
                move |v| kind.local_autodiff(&anchor, v[0]),
            )],
            ..Default::default()
        },
        false,
    )
    .unwrap();
    let measured = root.inverse().compose(truth);
    let relative = RelativePose3::new(measured).unwrap();
    let report = s
        .update(
            SmootherUpdate {
                variables: vec![WindowVariable {
                    key: 1,
                    kind,
                    value: se3_values(seed),
                    timestamp: None,
                    fixed: false,
                }],
                factors: vec![SmootherFactor::autodiff(
                    &[0, 1],
                    6,
                    12,
                    Default::default(),
                    move |v| relative.residual_autodiff(v[0], v[1]),
                )],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    assert_eq!(report.kind, IsamUpdateKind::Incremental);
    assert!(
        kind.local(&se3_values(truth), &report.estimates[&1])
            .unwrap()
            .norm()
            < 1e-9
    );
    let target = truth.retract(Tangent6::new(0.3, -0.1, 0.2, 0.1, 0.05, -0.1));
    let anchor = se3_values(target);
    let report = s
        .update(
            SmootherUpdate {
                factors: vec![SmootherFactor::autodiff(
                    &[1],
                    6,
                    6,
                    Default::default(),
                    move |v| kind.local_autodiff(&anchor, v[0]),
                )],
                ..Default::default()
            },
            true,
        )
        .unwrap();
    assert_eq!(report.kind, IsamUpdateKind::Relinearized);
    let mut batch = FactorGraph::new();
    let a = batch.add_variable(kind, se3_values(root)).unwrap();
    batch.set_fixed(a, true).unwrap();
    let b = batch.add_variable(kind, se3_values(seed)).unwrap();
    add_relative_pose3_factor(&mut batch, a, b, measured, Default::default()).unwrap();
    let anchor = se3_values(target);
    batch
        .add_autodiff_factor(&[b], 6, move |v| kind.local_autodiff(&anchor, v[0]))
        .unwrap();
    let full = solve_local(
        &batch,
        batch.initial(),
        &Options {
            linear_solver: ropt::LinearSolver::DenseQr(Default::default()),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        kind.local(&batch.value(&full.x, b).unwrap(), &report.estimates[&1])
            .unwrap()
            .norm()
            < 2e-7
    );
}

#[test]
fn sparse_refresh_uses_backend_and_factor_limits_instead_of_a_dense_matrix_limit() {
    use ropt::{LinearSolver, sparse_direct::SparseLdlOptions};
    let make_update = || SmootherUpdate {
        variables: vec![WindowVariable {
            key: 7,
            kind: VariableKind::Euclidean(32),
            value: DVector::from_element(32, 1.),
            timestamp: None,
            fixed: false,
        }],
        factors: vec![SmootherFactor::autodiff(
            &[7],
            32,
            32,
            Default::default(),
            |v| Ok(v[0].iter().map(|x| x * x - 4.).collect()),
        )],
        ..Default::default()
    };
    let mut options = IsamOptions::default();
    options.qr.max_entries = 32;
    let mut dense = IsamSmoother::new(options.clone()).unwrap();
    assert!(dense.update(make_update(), true).is_err());
    assert_eq!(dense.variable_count(), 0);
    assert_eq!(dense.factor_count(), 0);
    options.batch_solver.linear_solver = LinearSolver::SparseLdl(SparseLdlOptions {
        max_entries: 32,
        ..Default::default()
    });
    let mut sparse = IsamSmoother::new(options).unwrap();
    let result = sparse.update(make_update(), true).unwrap();
    assert_eq!(result.kind, IsamUpdateKind::Relinearized);
    assert_eq!(result.stored_entries, 32);
    assert!((&result.estimates[&7] - DVector::from_element(32, 2.)).norm() < 1e-8);
    assert_eq!(result.max_anchor_delta, 0.);
}

#[test]
fn failing_batch_trial_preserves_new_keys_history_cache_and_update_interval() {
    let mut s = IsamSmoother::new(IsamOptions {
        relinearization_threshold: 100.,
        relinearization_interval: 3,
        ..Default::default()
    })
    .unwrap();
    s.update(
        SmootherUpdate {
            variables: vec![variable(0, 0.)],
            factors: vec![observation(0, 1., Rc::new(Cell::new(0)))],
            ..Default::default()
        },
        false,
    )
    .unwrap();
    let count = Rc::new(Cell::new(0));
    let probe = count.clone();
    let failing = SmootherFactor::analytic(&[9], 1, Default::default(), move |v| {
        probe.set(probe.get() + 1);
        if probe.get() == 2 {
            return Err(Error::Evaluation("injected batch trial failure".into()));
        }
        Ok((
            &v[0] - DVector::from_element(1, 2.),
            vec![DMatrix::identity(1, 1)],
        ))
    });
    assert!(
        s.update(
            SmootherUpdate {
                // Initial evaluation succeeds; evaluation of the first trial fails.
                variables: vec![variable(9, 0.)],
                factors: vec![failing],
                ..Default::default()
            },
            true
        )
        .is_err()
    );
    assert_eq!(count.get(), 2);
    assert_eq!(s.variable_count(), 1);
    assert_eq!(s.factor_count(), 1);
    assert!(s.estimate(9).is_none());
    assert!((s.estimate(0).unwrap()[0] - 1.).abs() < 1e-12);
    let recovered = s
        .update(
            SmootherUpdate {
                variables: vec![variable(9, 0.)],
                factors: vec![observation(9, 3., Rc::new(Cell::new(0)))],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    assert_eq!(recovered.kind, IsamUpdateKind::Incremental);
    assert_eq!(recovered.factors_linearized, 1);
    assert!((recovered.estimates[&0][0] - 1.).abs() < 1e-12);
    assert!((recovered.estimates[&9][0] - 3.).abs() < 1e-12);
    assert_eq!(s.factor_count(), 2);
}

#[test]
fn reused_layout_handles_unsorted_keys_fixed_blocks_and_mixed_manifold_dimensions() {
    use ropt::manifold::*;
    let mut s = IsamSmoother::new(IsamOptions {
        relinearization_threshold: 100.,
        relinearization_interval: 2,
        ..Default::default()
    })
    .unwrap();
    let root = se2_values(SE2::exp(nalgebra::Vector3::new(2., -1., 0.4)));
    let anchor = root.clone();
    s.update(
        SmootherUpdate {
            variables: vec![WindowVariable {
                key: 90,
                kind: VariableKind::SE2,
                value: root.clone(),
                timestamp: None,
                fixed: true,
            }],
            factors: vec![SmootherFactor::autodiff(
                &[90],
                3,
                3,
                Default::default(),
                move |v| VariableKind::SE2.local_autodiff(&anchor, v[0]),
            )],
            ..Default::default()
        },
        false,
    )
    .unwrap();
    let target = so3_values(SO3::exp(nalgebra::Vector3::new(0.3, -0.2, 0.1)));
    let seed = so3_values(
        read_so3(&target)
            .unwrap()
            .retract(nalgebra::Vector3::new(0.01, 0.02, -0.01)),
    );
    let prior = target.clone();
    let update = SmootherUpdate {
        variables: vec![
            WindowVariable {
                key: 8,
                kind: VariableKind::SO3,
                value: seed,
                timestamp: None,
                fixed: false,
            },
            WindowVariable {
                key: 3,
                kind: VariableKind::Euclidean(2),
                value: DVector::zeros(2),
                timestamp: None,
                fixed: false,
            },
        ],
        factors: vec![
            SmootherFactor::autodiff(&[8], 3, 3, Default::default(), move |v| {
                VariableKind::SO3.local_autodiff(&prior, v[0])
            }),
            SmootherFactor::analytic(
                &[3],
                2,
                FactorOptions {
                    noise: ropt::noise::NoiseModel::from_covariance(DMatrix::from_row_slice(
                        2,
                        2,
                        &[2., 0.3, 0.3, 1.],
                    ))
                    .unwrap(),
                    ..Default::default()
                },
                |v| {
                    Ok((
                        &v[0] - DVector::from_vec(vec![1., -2.]),
                        vec![DMatrix::identity(2, 2)],
                    ))
                },
            ),
        ],
        ..Default::default()
    };
    let r = s.update(update, false).unwrap();
    assert_eq!(r.kind, IsamUpdateKind::Relinearized);
    assert_eq!(r.estimates[&90], root);
    assert!(
        VariableKind::SO3
            .local(&target, &r.estimates[&8])
            .unwrap()
            .norm()
            < 1e-9
    );
    assert!((&r.estimates[&3] - DVector::from_vec(vec![1., -2.])).norm() < 1e-9);
    let r = s
        .update(
            SmootherUpdate {
                variables: vec![variable(7, 0.)],
                factors: vec![observation(7, 2., Rc::new(Cell::new(0)))],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    assert_eq!(r.kind, IsamUpdateKind::Incremental);
    assert_eq!(r.estimates[&90], root);
    assert!((r.estimates[&7][0] - 2.).abs() < 1e-12);
    let r = s.update(SmootherUpdate::default(), true).unwrap();
    assert!(r.max_anchor_delta < 1e-12);
    assert!(r.linearized_cost < 1e-16);
}

#[test]
fn scheduled_refresh_uses_current_estimates_when_new_rows_are_singular_at_old_anchors() {
    for force in [true, false] {
        let mut s = IsamSmoother::new(IsamOptions {
            relinearization_threshold: 100.,
            relinearization_interval: if force { 100 } else { 2 },
            ..Default::default()
        })
        .unwrap();
        s.update(
            SmootherUpdate {
                variables: vec![variable(0, 0.)],
                factors: vec![observation(0, 1., Rc::new(Cell::new(0)))],
                ..Default::default()
            },
            false,
        )
        .unwrap();
        // Committed x=1, cached anchor x=0. At (0,1), xy-2 has no y
        // derivative, so cached row insertion cannot observe the new y.
        // At the committed x=1 the full nonlinear model is observable.
        let make_update = || SmootherUpdate {
            variables: vec![variable(1, 1.)],
            factors: vec![SmootherFactor::autodiff(
                &[0, 1],
                1,
                2,
                Default::default(),
                |v| Ok(vec![&v[0][0] * &v[1][0] - 2.]),
            )],
            ..Default::default()
        };
        if force {
            assert!(s.update(make_update(), false).is_err());
            assert_eq!(s.variable_count(), 1);
            assert_eq!(s.factor_count(), 1);
            assert!(s.estimate(1).is_none());
        }
        let report = s.update(make_update(), force).unwrap();
        assert_eq!(report.kind, IsamUpdateKind::Relinearized);
        assert_eq!(
            report.relinearization_reason,
            Some(if force {
                IsamRelinearizationReason::Forced
            } else {
                IsamRelinearizationReason::Interval
            })
        );
        assert!((report.estimates[&0][0] - 1.).abs() < 1e-8);
        assert!((report.estimates[&1][0] - 2.).abs() < 1e-8);
        assert!(report.linearized_cost < 1e-16);
    }
}

#[test]
fn refresh_reuses_the_accepted_whitened_model_without_a_final_callback_replay() {
    use ropt::{LinearSolver, noise::NoiseModel, sparse_direct::SparseLdlOptions};
    for backend in [
        LinearSolver::DenseQr(Default::default()),
        LinearSolver::SparseLdl(SparseLdlOptions::default()),
    ] {
        let calls = Rc::new(Cell::new(0));
        let probe = calls.clone();
        let mut s = IsamSmoother::new(IsamOptions {
            relinearization_threshold: 100.,
            relinearization_interval: 100,
            batch_solver: Options {
                linear_solver: backend,
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap();
        let r = s
            .update(
                SmootherUpdate {
                    variables: vec![variable(0, 2.)],
                    factors: vec![SmootherFactor::analytic(
                        &[0],
                        1,
                        FactorOptions {
                            noise: NoiseModel::isotropic(0.5).unwrap(),
                            loss: Some(ropt::Loss::Squared),
                        },
                        move |v| {
                            probe.set(probe.get() + 1);
                            Ok((
                                DVector::from_element(1, v[0][0] * v[0][0] - 4.),
                                vec![DMatrix::from_element(1, 1, 2. * v[0][0])],
                            ))
                        },
                    )],
                    ..Default::default()
                },
                true,
            )
            .unwrap();
        assert_eq!(calls.get(), 1); // Already stationary: one solve evaluation only.
        assert_eq!(r.stored_entries, 1);
        assert_eq!(r.linearized_cost, 0.);
        let r = s
            .update(
                SmootherUpdate {
                    factors: vec![observation(0, 3., Rc::new(Cell::new(0)))],
                    ..Default::default()
                },
                false,
            )
            .unwrap();
        assert_eq!(calls.get(), 1); // The old recipe was not evaluated again.
        // Accepted J=4, whitened by 2: cached normal contribution 64, not 16.
        assert!((r.estimates[&0][0] - (2. + 1. / 65.)).abs() < 1e-12);
        assert!((r.linearized_cost - 0.5 * 64. / 65.).abs() < 1e-12);
    }
}

#[test]
fn subsequent_cached_update_uses_the_converged_model_after_rejected_nonlinear_trials() {
    use ropt::{LinearSolver, noise::NoiseModel, sparse_direct::SparseLdlOptions};
    for backend in [
        LinearSolver::DenseQr(Default::default()),
        LinearSolver::SparseLdl(SparseLdlOptions::default()),
    ] {
        let calls = Rc::new(Cell::new(0));
        let probe = calls.clone();
        let overshot = Rc::new(Cell::new(false));
        let over = overshot.clone();
        let mut s = IsamSmoother::new(IsamOptions {
            relinearization_threshold: 100.,
            relinearization_interval: 100,
            batch_solver: Options {
                linear_solver: backend,
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap();
        let r = s
            .update(
                SmootherUpdate {
                    variables: vec![variable(0, 0.1)],
                    factors: vec![SmootherFactor::analytic(
                        &[0],
                        1,
                        FactorOptions {
                            noise: NoiseModel::isotropic(0.5).unwrap(),
                            ..Default::default()
                        },
                        move |v| {
                            probe.set(probe.get() + 1);
                            if v[0][0] > 10. {
                                over.set(true);
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
        assert!(overshot.get()); // This trial's objective exceeds the initial cost.
        let anchor = r.estimates[&0][0];
        assert!((anchor - 2.).abs() < 1e-8);
        let count = calls.get();
        let jac = 4. * anchor;
        let residual = 2. * (anchor.powi(2) - 4.);
        let r = s
            .update(
                SmootherUpdate {
                    factors: vec![observation(0, 3., Rc::new(Cell::new(0)))],
                    ..Default::default()
                },
                false,
            )
            .unwrap();
        assert_eq!(calls.get(), count);
        let exact = anchor + (3. - anchor - jac * residual) / (jac * jac + 1.);
        assert!((r.estimates[&0][0] - exact).abs() < 1e-12);
    }
}

#[test]
fn reused_stationary_model_still_checks_final_qr_fill_and_rolls_back() {
    let mut s = IsamSmoother::new(IsamOptions {
        relinearization_threshold: 100.,
        relinearization_interval: 2,
        qr: ropt::incremental_qr::IncrementalQrOptions {
            max_entries: 2,
            ..Default::default()
        },
        batch_solver: Options {
            linear_solver: ropt::LinearSolver::SparseLdl(Default::default()),
            ..Default::default()
        },
    })
    .unwrap();
    let failure = s.update(
        SmootherUpdate {
            variables: vec![variable(1, 1.), variable(2, 2.)],
            factors: vec![SmootherFactor::analytic(
                &[1, 2],
                2,
                Default::default(),
                |v| {
                    Ok((
                        DVector::from_vec(vec![v[0][0] + v[1][0] - 3., v[1][0] - 2.]),
                        vec![
                            DMatrix::from_column_slice(2, 1, &[1., 0.]),
                            DMatrix::from_column_slice(2, 1, &[1., 1.]),
                        ],
                    ))
                },
            )],
            ..Default::default()
        },
        true,
    );
    assert!(matches!(failure, Err(Error::SparseFillLimit { limit: 2 })));
    assert_eq!(s.variable_count(), 0);
    assert_eq!(s.factor_count(), 0);
    let recovered = s
        .update(
            SmootherUpdate {
                variables: vec![variable(3, 0.)],
                factors: vec![observation(3, 2., Rc::new(Cell::new(0)))],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    assert_eq!(recovered.kind, IsamUpdateKind::Incremental);
    assert!((recovered.estimates[&3][0] - 2.).abs() < 1e-12);
}
