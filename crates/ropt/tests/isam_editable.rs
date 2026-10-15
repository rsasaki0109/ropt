use ropt::{factor::*, fixed_lag::*, isam::*, *};
use std::{cell::Cell, rc::Rc};

fn configured(editable: bool, interval: usize) -> IsamSmoother {
    let smoother = IsamSmoother::new(IsamOptions {
        relinearization_interval: interval,
        batch_solver: Options {
            algorithm: Algorithm::GaussNewton,
            linear_solver: LinearSolver::SparseQr(Default::default()),
            cost_tolerance: 1e-15,
            step_tolerance: 1e-12,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    if editable {
        smoother.with_selective_bayes_tree(4).unwrap()
    } else {
        smoother.with_selective_relinearization(4).unwrap()
    }
}
fn star(n: usize, calls: Rc<Cell<usize>>) -> SmootherUpdate {
    let mut update = SmootherUpdate::default();
    for key in 0..n as u64 {
        update.variables.push(WindowVariable {
            key,
            kind: VariableKind::Euclidean(1),
            value: DVector::from_element(1, 1.),
            timestamp: None,
            fixed: false,
        });
        let count = calls.clone();
        update.factors.push(SmootherFactor::autodiff(
            &[key],
            1,
            1,
            Default::default(),
            move |v| {
                count.set(count.get() + 1);
                Ok(vec![v[0][0].powi(2) - 1.])
            },
        ));
        let count = calls.clone();
        update.factors.push(SmootherFactor::autodiff(
            &[key, 100000],
            1,
            1,
            Default::default(),
            move |v| {
                count.set(count.get() + 1);
                Ok(vec![&v[0][0] - &v[1][0] - 1.])
            },
        ));
    }
    update.variables.push(WindowVariable {
        key: 100000,
        kind: VariableKind::Euclidean(1),
        value: DVector::zeros(1),
        timestamp: None,
        fixed: false,
    });
    update.factors.push(SmootherFactor::autodiff(
        &[100000],
        1,
        1,
        Default::default(),
        move |v| {
            calls.set(calls.get() + 1);
            Ok(vec![1000. * &v[0][0]])
        },
    ));
    update
}
fn observation(
    key: u64,
    target: f64,
    calls: Rc<Cell<usize>>,
    fail: Rc<Cell<bool>>,
) -> SmootherFactor {
    SmootherFactor::analytic(&[key], 1, Default::default(), move |v| {
        calls.set(calls.get() + 1);
        if fail.get() && v[0][0] > 1.2 {
            return Err(Error::Evaluation("selected anchor rejected".into()));
        }
        Ok((
            DVector::from_element(1, v[0][0] - target),
            vec![DMatrix::identity(1, 1)],
        ))
    })
}
#[test]
fn nonlinear_selected_replacement_reuses_star_branches_and_matches_full_gaussian_rebuild() {
    let n = 128;
    let mut reports = Vec::new();
    for editable in [false, true] {
        let calls = Rc::new(Cell::new(0));
        let mut smoother = configured(editable, 100);
        smoother.update(star(n, calls.clone()), false).unwrap();
        calls.set(0);
        let report = smoother
            .update(
                SmootherUpdate {
                    factors: vec![observation(0, 3., calls.clone(), Rc::new(Cell::new(false)))],
                    ..Default::default()
                },
                false,
            )
            .unwrap();
        // The two-variable AD factor uses two one-column chunks.
        assert_eq!(calls.get(), 5);
        assert_eq!(
            report
                .selective_update
                .as_ref()
                .unwrap()
                .variables_relinearized,
            vec![0]
        );
        assert_eq!(
            report.selective_update.as_ref().unwrap().retained_factors,
            2 * n - 1
        );
        if editable {
            let work = report.editable_bayes_tree_update.unwrap();
            assert_eq!(work.reeliminated_cliques, 2);
            assert_eq!(work.reused_cliques, n - 1);
            assert_eq!(work.reused_boundary_messages, n - 1);
            assert_eq!(work.input_factor_rows, 4);
            assert!(report.bayes_tree_update.is_none());
        } else {
            assert!(report.editable_bayes_tree_update.is_none());
        }
        reports.push(report);
    }
    for (key, x) in &reports[0].estimates {
        assert!((x - &reports[1].estimates[key]).norm() < 1e-12);
    }
    assert!((reports[0].linearized_cost - reports[1].linearized_cost).abs() < 1e-12);
}
#[test]
fn rejected_partial_replacement_rolls_back_and_interval_refresh_repopulates_messages() {
    let calls = Rc::new(Cell::new(0));
    let fail = Rc::new(Cell::new(true));
    let mut smoother = configured(true, 3);
    smoother.update(star(8, calls.clone()), false).unwrap();
    let count = smoother.factor_count();
    let added = || SmootherUpdate {
        factors: vec![observation(0, 3., calls.clone(), fail.clone())],
        ..Default::default()
    };
    assert!(smoother.update(added(), false).is_err());
    assert_eq!(smoother.factor_count(), count);
    assert_eq!(smoother.estimate(0).unwrap()[0], 1.);
    assert_eq!(smoother.estimate(100000).unwrap()[0], 0.);
    fail.set(false);
    let retry = smoother.update(added(), false).unwrap();
    assert!(retry.selective_update.is_some());
    let refresh = smoother.update(Default::default(), false).unwrap();
    assert_eq!(
        refresh.relinearization_reason,
        Some(IsamRelinearizationReason::Interval)
    );
    assert!(refresh.selective_update.is_none());
    assert!(refresh.batch_trials > 0);
    assert_eq!(
        refresh
            .editable_bayes_tree_update
            .unwrap()
            .reeliminated_cliques,
        9
    );
    let next = smoother
        .update(
            SmootherUpdate {
                factors: vec![observation(2, 3., calls, Rc::new(Cell::new(false)))],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    assert_eq!(
        next.selective_update.unwrap().variables_relinearized,
        vec![2]
    );
    assert_eq!(
        next.editable_bayes_tree_update
            .unwrap()
            .reeliminated_cliques,
        2
    );
}
#[test]
fn rank_loss_in_selected_model_preserves_the_committed_original_factor() {
    let factor = |target| {
        SmootherFactor::autodiff(&[4], 1, 1, Default::default(), move |v| {
            Ok(vec![v[0][0].powi(2) - target])
        })
    };
    let mut smoother = configured(true, 100);
    smoother
        .update(
            SmootherUpdate {
                variables: vec![
                    WindowVariable {
                        key: 4,
                        kind: VariableKind::Euclidean(1),
                        value: DVector::from_element(1, 1.),
                        timestamp: None,
                        fixed: false,
                    },
                    WindowVariable {
                        key: 99,
                        kind: VariableKind::Euclidean(1),
                        value: DVector::from_element(1, 2.),
                        timestamp: None,
                        fixed: false,
                    },
                ],
                factors: vec![
                    factor(1.),
                    SmootherFactor::autodiff(&[99], 1, 1, Default::default(), |v| {
                        Ok(vec![&v[0][0] - 2.])
                    }),
                ],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    assert!(matches!(
        smoother.update(
            SmootherUpdate {
                factors: vec![factor(-3.)],
                ..Default::default()
            },
            false
        ),
        Err(Error::RankDeficient {
            rank: 1,
            parameters: 2
        })
    ));
    assert_eq!(smoother.factor_count(), 2);
    assert_eq!(smoother.estimate(4).unwrap()[0], 1.);
    let recovered = smoother
        .update(
            SmootherUpdate {
                factors: vec![factor(1.)],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    assert_eq!(recovered.linearized_cost, 0.);
    assert_eq!(recovered.estimates[&4][0], 1.);
}

#[test]
fn editable_frozen_factor_refinement_matches_global_qr_and_survives_refresh() {
    let mut solvers = [false, true].map(|editable| {
        let smoother = IsamSmoother::new(IsamOptions {
            relinearization_interval: 2,
            batch_solver: Options {
                algorithm: Algorithm::GaussNewton,
                linear_solver: LinearSolver::SparseQr(Default::default()),
                cost_tolerance: 1e-5,
                gradient_tolerance: 1e-12,
                step_tolerance: 1e-14,
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap()
        .with_final_refinement(4)
        .unwrap();
        if editable {
            smoother.with_selective_bayes_tree(4).unwrap()
        } else {
            smoother.with_selective_relinearization(4).unwrap()
        }
    });
    let mut refinements = [0, 0];
    for round in 0..3 {
        let mut reports = Vec::new();
        for (index, solver) in solvers.iter_mut().enumerate() {
            let update = if round == 0 {
                let mut initial = star(6, Rc::new(Cell::new(0)));
                for variable in &mut initial.variables {
                    if variable.key != 100000 {
                        variable.value[0] = 1.7;
                    }
                }
                initial
            } else {
                SmootherUpdate {
                    factors: vec![observation(
                        0,
                        1.3 + round as f64 * 0.1,
                        Rc::new(Cell::new(0)),
                        Rc::new(Cell::new(false)),
                    )],
                    ..Default::default()
                }
            };
            let report = solver.update(update, true).unwrap();
            refinements[index] += report.refinement_steps;
            reports.push(report);
        }
        for (a, b) in reports[0].estimates.iter().zip(&reports[1].estimates) {
            assert_eq!(a.0, b.0);
            assert!((a.1 - b.1).norm() < 1e-9);
        }
    }
    assert!(refinements.iter().all(|&steps| steps > 0));
}
