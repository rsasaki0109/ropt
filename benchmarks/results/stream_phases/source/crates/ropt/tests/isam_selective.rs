use ropt::{factor::*, fixed_lag::*, isam::*, noise::NoiseModel, *};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

fn variable(key: u64, value: f64) -> WindowVariable {
    WindowVariable {
        key,
        kind: VariableKind::Euclidean(1),
        value: DVector::from_element(1, value),
        timestamp: None,
        fixed: false,
    }
}
fn options() -> IsamOptions {
    IsamOptions {
        relinearization_threshold: 0.1,
        relinearization_interval: 100,
        ..Default::default()
    }
}
fn linear(key: u64, target: f64, scale: f64, calls: Rc<Cell<usize>>) -> SmootherFactor {
    SmootherFactor::analytic(&[key], 1, Default::default(), move |v| {
        calls.set(calls.get() + 1);
        Ok((
            DVector::from_element(1, scale * (v[0][0] - target)),
            vec![DMatrix::from_element(1, 1, scale)],
        ))
    })
}
fn square(key: u64, target: f64, calls: Rc<Cell<usize>>, fail: Rc<Cell<bool>>) -> SmootherFactor {
    SmootherFactor::analytic(&[key], 1, Default::default(), move |v| {
        calls.set(calls.get() + 1);
        if fail.get() && v[0][0] > 1.2 {
            return Err(Error::Evaluation(
                "selected callback rejected anchor".into(),
            ));
        }
        Ok((
            DVector::from_element(1, v[0][0] * v[0][0] - target),
            vec![DMatrix::from_element(1, 1, 2. * v[0][0])],
        ))
    })
}

#[test]
fn only_selected_factors_are_evaluated_and_cap_is_reported() {
    let counts = (0..3).map(|_| Rc::new(Cell::new(0))).collect::<Vec<_>>();
    let mut smoother = IsamSmoother::new(options())
        .unwrap()
        .with_selective_relinearization(1)
        .unwrap();
    smoother
        .update(
            SmootherUpdate {
                variables: vec![variable(20, 1.), variable(3, 2.), variable(99, 3.)],
                factors: counts
                    .iter()
                    .zip([(20, 1.), (3, 4.), (99, 9.)])
                    .map(|(calls, (key, target))| {
                        square(key, target, calls.clone(), Rc::new(Cell::new(false)))
                    })
                    .collect(),
                ..Default::default()
            },
            false,
        )
        .unwrap();
    for count in &counts {
        count.set(0);
    }
    let added = Rc::new(Cell::new(0));
    let report = smoother
        .update(
            SmootherUpdate {
                factors: vec![linear(20, 3., 1., added.clone())],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    let work = report.selective_update.as_ref().unwrap();
    assert_eq!(work.passes, 1);
    assert_eq!(work.variables_relinearized, vec![20]);
    assert_eq!(work.factor_relinearizations, 2);
    assert_eq!(work.retained_factors, 2);
    assert!(work.threshold_remaining);
    assert_eq!(
        counts.iter().map(|c| c.get()).collect::<Vec<_>>(),
        vec![1, 0, 0]
    );
    assert_eq!(added.get(), 2);
    assert_eq!(report.factors_linearized, 2);
    assert_eq!(report.batch_trials, 0);
    assert_eq!(
        report.relinearization_reason,
        Some(IsamRelinearizationReason::Threshold)
    );
    // Independently solve the two scalar rows at the new anchor x=1.4.
    let anchor = 1.4_f64;
    let step =
        -(2. * anchor * (anchor * anchor - 1.) + (anchor - 3.)) / (4. * anchor * anchor + 1.);
    assert!((report.estimates[&20][0] - (anchor + step)).abs() < 1e-13);
    let cost =
        0.5 * ((anchor * anchor - 1. + 2. * anchor * step).powi(2) + (anchor - 3. + step).powi(2));
    assert!((report.linearized_cost - cost).abs() < 1e-13);
    assert_eq!(report.estimates[&3][0], 2.);
    assert_eq!(report.estimates[&99][0], 3.);
    // An empty update can continue from the committed mixed chart.
    let continued = smoother.update(Default::default(), false).unwrap();
    assert!(!continued.selective_update.unwrap().threshold_remaining);
    assert_eq!(counts[1].get(), 0);
    assert_eq!(counts[2].get(), 0);
}

#[test]
fn correlated_incident_factor_uses_mixed_anchors_and_independent_original_energy() {
    let calls = Rc::new(RefCell::new(Vec::new()));
    let saved = calls.clone();
    let omega = DMatrix::from_row_slice(2, 2, &[4., 0.5, 0.5, 3.]);
    let nonlinear = SmootherFactor::analytic(
        &[20, 3],
        2,
        FactorOptions {
            noise: NoiseModel::from_information(omega.clone()).unwrap(),
            ..Default::default()
        },
        move |v| {
            let (x, y) = (v[0][0], v[1][0]);
            saved.borrow_mut().push((x, y));
            Ok((
                DVector::from_vec(vec![x * x - y + 1., x + y - 3.]),
                vec![
                    DMatrix::from_column_slice(2, 1, &[2. * x, 1.]),
                    DMatrix::from_column_slice(2, 1, &[-1., 1.]),
                ],
            ))
        },
    );
    let x_calls = Rc::new(Cell::new(0));
    let y_calls = Rc::new(Cell::new(0));
    let mut smoother = IsamSmoother::new(options())
        .unwrap()
        .with_bayes_tree()
        .unwrap()
        .with_selective_relinearization(1)
        .unwrap();
    smoother
        .update(
            SmootherUpdate {
                variables: vec![variable(20, 1.), variable(3, 2.)],
                factors: vec![
                    nonlinear,
                    linear(20, 1., 1., x_calls.clone()),
                    linear(3, 2., 10., y_calls.clone()),
                ],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    calls.borrow_mut().clear();
    x_calls.set(0);
    y_calls.set(0);
    let report = smoother
        .update(
            SmootherUpdate {
                factors: vec![linear(20, 5., 1., Rc::new(Cell::new(0)))],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    let j0 = DMatrix::from_row_slice(2, 2, &[2., -1., 1., 1.]);
    let diagonal = DMatrix::from_diagonal(&DVector::from_vec(vec![2., 100.]));
    let h0 = j0.transpose() * &omega * &j0 + &diagonal;
    let d0 = h0.lu().solve(&DVector::from_vec(vec![4., 0.])).unwrap();
    let anchor = 1. + d0[0];
    assert!(d0[0] > 0.1 && d0[1].abs() < 0.1);
    assert_eq!(calls.borrow().len(), 1);
    assert!((calls.borrow()[0].0 - anchor).abs() < 1e-13);
    assert_eq!(calls.borrow()[0].1, 2.); // unselected neighbor keeps its old anchor
    assert_eq!(x_calls.get(), 1);
    assert_eq!(y_calls.get(), 0);
    let j = DMatrix::from_row_slice(2, 2, &[2. * anchor, -1., 1., 1.]);
    let r = DVector::from_vec(vec![anchor * anchor - 1., anchor - 1.]);
    let h = j.transpose() * &omega * &j + diagonal;
    let g =
        j.transpose() * &omega * &r + DVector::from_vec(vec![(anchor - 1.) + (anchor - 5.), 0.]);
    let delta = h.lu().solve(&(-g)).unwrap();
    let expected = DVector::from_vec(vec![anchor + delta[0], 2. + delta[1]]);
    assert!((report.estimates[&20][0] - expected[0]).abs() < 1e-12);
    assert!((report.estimates[&3][0] - expected[1]).abs() < 1e-12);
    let affine = &r + j * &delta;
    let original = 0.5
        * (affine.dot(&(&omega * &affine))
            + (expected[0] - 1.).powi(2)
            + (10. * delta[1]).powi(2)
            + (expected[0] - 5.).powi(2));
    assert!((report.linearized_cost - original).abs() < 1e-12);
    assert_eq!(
        report.selective_update.as_ref().unwrap().retained_factors,
        1
    );
    assert_eq!(
        report.selective_update.unwrap().variables_relinearized,
        vec![20]
    );
    assert_eq!(report.bayes_tree_update.unwrap().reused_cliques, 0); // global rebuild disclosed
}

#[test]
fn rejected_selected_callback_leaves_owned_history_and_cache_unchanged() {
    let fail = Rc::new(Cell::new(false));
    let mut smoother = IsamSmoother::new(options())
        .unwrap()
        .with_selective_relinearization(2)
        .unwrap();
    smoother
        .update(
            SmootherUpdate {
                variables: vec![variable(20, 1.)],
                factors: vec![square(20, 1., Rc::new(Cell::new(0)), fail.clone())],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    fail.set(true);
    assert!(
        smoother
            .update(
                SmootherUpdate {
                    factors: vec![linear(20, 3., 1., Rc::new(Cell::new(0)))],
                    ..Default::default()
                },
                false
            )
            .is_err()
    );
    assert_eq!(smoother.factor_count(), 1);
    assert_eq!(smoother.estimate(20).unwrap()[0], 1.);
    fail.set(false);
    let unchanged = smoother.update(Default::default(), false).unwrap();
    assert_eq!(unchanged.linearized_cost, 0.);
    assert!(unchanged.selective_update.is_none());
    let retried = smoother
        .update(
            SmootherUpdate {
                factors: vec![linear(20, 3., 1., Rc::new(Cell::new(0)))],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    assert_eq!(retried.selective_update.unwrap().passes, 2);
    assert_eq!(smoother.factor_count(), 2);
}

#[test]
fn extending_heterogeneous_columns_retains_old_rows_and_fixed_constant_energy() {
    let old_calls = Rc::new(Cell::new(0));
    let mut fixed = variable(8, 2.);
    fixed.fixed = true;
    let mut smoother = IsamSmoother::new(IsamOptions {
        relinearization_threshold: 0.5,
        ..options()
    })
    .unwrap()
    .with_selective_relinearization(1)
    .unwrap();
    smoother
        .update(
            SmootherUpdate {
                variables: vec![variable(90, 1.), fixed],
                factors: vec![
                    linear(90, 1., 1., old_calls.clone()),
                    linear(8, 0., 1., Rc::new(Cell::new(0))),
                ],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    old_calls.set(0);
    let report = smoother
        .update(
            SmootherUpdate {
                variables: vec![WindowVariable {
                    key: 1,
                    kind: VariableKind::Euclidean(2),
                    value: DVector::from_vec(vec![2., 4.]),
                    timestamp: None,
                    fixed: false,
                }],
                factors: vec![SmootherFactor::autodiff(
                    &[1],
                    2,
                    1,
                    Default::default(),
                    |v| Ok(vec![v[0][0].powi(2) - 9., &v[0][1] - 6.]),
                )],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    assert_eq!(old_calls.get(), 0);
    assert_eq!(report.estimates[&90][0], 1.);
    assert_eq!(report.estimates[&8][0], 2.);
    assert!((report.estimates[&1][0] - (3.25 - (3.25_f64.powi(2) - 9.) / 6.5)).abs() < 1e-13);
    assert_eq!(report.estimates[&1][1], 6.);
    assert!((report.linearized_cost - 2.).abs() < 1e-12);
    assert_eq!(report.selective_update.unwrap().retained_factors, 2);
}

#[test]
fn raw_winding_is_not_hidden_by_principal_local_coordinates() {
    let mut v = variable(20, 0.);
    v.kind = VariableKind::SO2;
    let mut smoother = IsamSmoother::new(options())
        .unwrap()
        .with_selective_relinearization(1)
        .unwrap();
    smoother
        .update(
            SmootherUpdate {
                variables: vec![v],
                factors: vec![linear(20, 0., 1., Rc::new(Cell::new(0)))],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    let report = smoother
        .update(
            SmootherUpdate {
                factors: vec![linear(
                    20,
                    4. * std::f64::consts::PI + 0.05,
                    1.,
                    Rc::new(Cell::new(0)),
                )],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    assert!(report.max_anchor_delta < 0.1);
    let selected = report.selective_update.unwrap();
    assert_eq!(selected.variables_relinearized, vec![20]);
    assert!(selected.threshold_remaining);
}

#[test]
fn cache_budget_and_late_activation_are_transactional_errors() {
    for cap in [0, 201] {
        assert!(
            IsamSmoother::new(options())
                .unwrap()
                .with_selective_relinearization(cap)
                .is_err()
        );
    }
    let mut configured = options();
    configured.qr.max_entries = 6;
    let mut smoother = IsamSmoother::new(configured)
        .unwrap()
        .with_selective_relinearization(1)
        .unwrap();
    smoother
        .update(
            SmootherUpdate {
                variables: vec![variable(20, 1.)],
                factors: vec![linear(20, 1., 1., Rc::new(Cell::new(0)))],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    for _ in 0..2 {
        smoother
            .update(
                SmootherUpdate {
                    factors: vec![linear(20, 1., 1., Rc::new(Cell::new(0)))],
                    ..Default::default()
                },
                false,
            )
            .unwrap();
    }
    assert!(matches!(
        smoother.update(
            SmootherUpdate {
                factors: vec![linear(20, 1., 1., Rc::new(Cell::new(0)))],
                ..Default::default()
            },
            false
        ),
        Err(Error::SparseFillLimit { limit: 6 })
    ));
    assert_eq!(smoother.factor_count(), 3);
    assert_eq!(smoother.estimate(20).unwrap()[0], 1.);
    let report = smoother.update(Default::default(), false).unwrap();
    assert!(report.selective_update.is_none());
    assert!(smoother.with_selective_relinearization(1).is_err());
}

#[test]
fn forced_and_interval_refresh_still_use_full_batch_policy() {
    let mut configured = options();
    configured.relinearization_interval = 2;
    let mut smoother = IsamSmoother::new(configured)
        .unwrap()
        .with_selective_relinearization(2)
        .unwrap();
    let first = smoother
        .update(
            SmootherUpdate {
                variables: vec![variable(20, 1.)],
                factors: vec![square(
                    20,
                    1.,
                    Rc::new(Cell::new(0)),
                    Rc::new(Cell::new(false)),
                )],
                ..Default::default()
            },
            true,
        )
        .unwrap();
    assert!(first.selective_update.is_none());
    assert_eq!(
        first.relinearization_reason,
        Some(IsamRelinearizationReason::Forced)
    );
    let second = smoother
        .update(
            SmootherUpdate {
                factors: vec![linear(20, 3., 1., Rc::new(Cell::new(0)))],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    assert!(second.selective_update.is_none());
    assert_eq!(
        second.relinearization_reason,
        Some(IsamRelinearizationReason::Interval)
    );
    assert!(second.batch_trials > 0);
    assert!(
        (2. * second.estimates[&20][0] * (second.estimates[&20][0].powi(2) - 1.)
            + second.estimates[&20][0]
            - 3.)
            .abs()
            < 1e-6
    );
    // The accepted batch model also populates the factor-owned cache.
    let third = smoother
        .update(
            SmootherUpdate {
                factors: vec![linear(20, 6., 1., Rc::new(Cell::new(0)))],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    assert!(third.selective_update.is_some());
    assert_eq!(third.batch_trials, 0);
}

#[test]
fn both_se2_charts_match_independent_central_difference_mixed_model() {
    use ropt::pose_graph::RelativePose2;
    fn residual(poses: &[[f64; 3]], a: usize, b: usize, z: [f64; 3]) -> [f64; 3] {
        let (ca, sa) = (poses[a][2].cos(), poses[a][2].sin());
        let (cz, sz) = (z[2].cos(), z[2].sin());
        let (dx, dy) = (poses[b][0] - poses[a][0], poses[b][1] - poses[a][1]);
        let (x, y) = (ca * dx + sa * dy - z[0], -sa * dx + ca * dy - z[1]);
        let angle = poses[b][2] - poses[a][2] - z[2];
        [
            cz * x + sz * y,
            -sz * x + cz * y,
            angle.sin().atan2(angle.cos()),
        ]
    }
    fn retract(p: [f64; 3], d: &[f64], cartesian: bool) -> [f64; 3] {
        let (a, b) = if cartesian || d[2] == 0. {
            (1., 0.)
        } else {
            (d[2].sin() / d[2], 2. * (d[2] / 2.).sin().powi(2) / d[2])
        };
        let (x, y) = (a * d[0] - b * d[1], b * d[0] + a * d[1]);
        let angle = p[2] + d[2];
        [
            p[0] + p[2].cos() * x - p[2].sin() * y,
            p[1] + p[2].sin() * x + p[2].cos() * y,
            angle.sin().atan2(angle.cos()),
        ]
    }
    type Edge = (usize, usize, [f64; 3], f64);
    fn model(poses: &[[f64; 3]], edges: &[Edge], cartesian: bool) -> (DVector<f64>, DMatrix<f64>) {
        let evaluate = |p: &[[f64; 3]]| {
            DVector::from_vec(
                edges
                    .iter()
                    .flat_map(|&(a, b, z, scale)| residual(p, a, b, z).map(|v| scale * v))
                    .collect(),
            )
        };
        let r = evaluate(poses);
        let mut j = DMatrix::zeros(r.len(), 6);
        let h = 1e-6;
        for col in 0..6 {
            let key = 1 + col / 3;
            let mut d = [0.; 3];
            d[col % 3] = h;
            let mut plus = poses.to_vec();
            let mut minus = poses.to_vec();
            plus[key] = retract(poses[key], &d, cartesian);
            d[col % 3] = -h;
            minus[key] = retract(poses[key], &d, cartesian);
            j.set_column(col, &((evaluate(&plus) - evaluate(&minus)) / (2. * h)));
        }
        (r, j)
    }
    let original = vec![[2., -1., 0.4], [3., 0.2, 0.8], [4.8, 1.5, 1.1]];
    let z01 = residual(&original, 0, 1, [0., 0., 0.]);
    let z12 = residual(&original, 1, 2, [0., 0., 0.]);
    let z02 = residual(&original, 0, 2, [0., 0., 0.]);
    let target = retract(z01, &[0.8, 0.2, 0.2], true);
    let edges = vec![
        (0, 1, z01, 1.),
        (1, 2, z12, 1.),
        (0, 2, z02, 100.),
        (0, 1, target, 1.),
    ];
    for (cartesian, editable) in [(false, false), (true, false), (false, true), (true, true)] {
        let kind = if cartesian {
            VariableKind::SE2Cartesian
        } else {
            VariableKind::SE2
        };
        let make = |(a, b, z, scale): Edge| {
            let relative =
                RelativePose2::new(read_se2(&DVector::from_column_slice(&z)).unwrap()).unwrap();
            let keys = [0, 20, 3];
            SmootherFactor::analytic(&[keys[a], keys[b]], 3, Default::default(), move |v| {
                let (r, ja, jb) = relative.linearize_ambient(v[0].as_slice(), v[1].as_slice())?;
                Ok((
                    DVector::from_column_slice(r.as_slice()) * scale,
                    vec![
                        DMatrix::from_column_slice(3, 3, ja.as_slice()) * scale,
                        DMatrix::from_column_slice(3, 3, jb.as_slice()) * scale,
                    ],
                ))
            })
        };
        let smoother = IsamSmoother::new(options()).unwrap();
        let mut smoother = if editable {
            smoother.with_selective_bayes_tree(1).unwrap()
        } else {
            smoother.with_selective_relinearization(1).unwrap()
        };
        smoother
            .update(
                SmootherUpdate {
                    variables: original
                        .iter()
                        .zip([0, 20, 3])
                        .map(|(p, key)| WindowVariable {
                            key,
                            kind,
                            value: DVector::from_column_slice(p),
                            timestamp: None,
                            fixed: key == 0,
                        })
                        .collect(),
                    factors: edges[..3].iter().copied().map(make).collect(),
                    ..Default::default()
                },
                false,
            )
            .unwrap();
        let report = smoother
            .update(
                SmootherUpdate {
                    factors: vec![make(edges[3])],
                    ..Default::default()
                },
                false,
            )
            .unwrap();
        let (r0, j0) = model(&original, &edges, cartesian);
        let d0 = j0.svd(true, true).solve(&(-r0), 1e-12).unwrap();
        assert!(d0.rows(0, 3).norm() > 0.1 && d0.rows(3, 3).norm() < 0.1);
        let mut mixed = original.clone();
        mixed[1] = retract(original[1], &d0.as_slice()[..3], cartesian);
        let (r, j) = model(&mixed, &edges, cartesian);
        let delta = j.clone().svd(true, true).solve(&(-&r), 1e-12).unwrap();
        for (i, key) in [(1, 20), (2, 3)] {
            let expected = retract(mixed[i], &delta.as_slice()[(i - 1) * 3..i * 3], cartesian);
            assert!(
                (&report.estimates[&key] - DVector::from_column_slice(&expected)).norm() < 2e-8
            );
        }
        assert!((report.linearized_cost - 0.5 * (r + j * delta).norm_squared()).abs() < 2e-8);
        let work = report.selective_update.unwrap();
        assert_eq!(work.variables_relinearized, vec![20]);
        assert_eq!(work.retained_factors, 1);
        assert_eq!(
            report.estimates[&0],
            DVector::from_column_slice(&original[0])
        );
    }
}
