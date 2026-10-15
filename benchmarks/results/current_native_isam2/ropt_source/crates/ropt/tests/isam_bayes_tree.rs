use ropt::{
    bayes_tree::BayesTreeUpdate, factor::*, fixed_lag::*, incremental_qr::IncrementalQrOptions,
    isam::*, *,
};
use std::{cell::Cell, rc::Rc};
#[path = "../examples/support/isam_se2.rs"]
mod fixture;

fn variable(key: u64, value: f64) -> WindowVariable {
    WindowVariable {
        key,
        kind: VariableKind::Euclidean(1),
        value: DVector::from_element(1, value),
        timestamp: None,
        fixed: false,
    }
}
fn observation(key: u64, value: f64, calls: Rc<Cell<usize>>) -> SmootherFactor {
    SmootherFactor::analytic(&[key], 1, Default::default(), move |v| {
        calls.set(calls.get() + 1);
        Ok((
            &v[0] - DVector::from_element(1, value),
            vec![DMatrix::identity(1, 1)],
        ))
    })
}
#[test]
fn cached_tree_stream_preserves_old_callbacks_and_matches_default_qr() {
    let options = IsamOptions {
        relinearization_threshold: 100.,
        relinearization_interval: 100,
        ..Default::default()
    };
    let mut tree = IsamSmoother::new(options.clone())
        .unwrap()
        .with_bayes_tree()
        .unwrap();
    let mut qr = IsamSmoother::new(options).unwrap();
    let calls = Rc::new(Cell::new(0));
    for k in 0..8 {
        // Nonsequential keys ensure original graph insertion coordinates survive.
        let key = 100 - k * 7;
        let update = SmootherUpdate {
            variables: vec![variable(key, 0.)],
            factors: vec![observation(key, k as f64 * 0.3, calls.clone())],
            ..Default::default()
        };
        let a = tree
            .update(
                SmootherUpdate {
                    variables: update.variables.clone(),
                    factors: update.factors.clone(),
                    ..Default::default()
                },
                false,
            )
            .unwrap();
        let b = qr.update(update, false).unwrap();
        assert_eq!(
            a.bayes_tree_update,
            Some(BayesTreeUpdate {
                reeliminated_cliques: 1,
                reused_cliques: k as usize,
                input_rows: 2
            })
        );
        assert_eq!(a.kind, IsamUpdateKind::Incremental);
        assert_eq!(a.factors_linearized, 1);
        assert!(b.bayes_tree_update.is_none());
        assert_eq!(a.estimates, b.estimates);
        assert!((a.linearized_cost - b.linearized_cost).abs() < 1e-12);
        assert_eq!(calls.get(), 2 * (k as usize + 1));
    }
    let before = calls.get();
    let a = tree.update(SmootherUpdate::default(), true).unwrap();
    let b = qr.update(SmootherUpdate::default(), true).unwrap();
    assert_eq!(a.bayes_tree_update.unwrap().reeliminated_cliques, 8);
    assert_eq!(
        a.relinearization_reason,
        Some(IsamRelinearizationReason::Forced)
    );
    assert_eq!(a.estimates, b.estimates);
    assert!(calls.get() > before);
    let a = tree.update(SmootherUpdate::default(), false).unwrap();
    assert_eq!(a.bayes_tree_update.unwrap().reeliminated_cliques, 0);
    assert_eq!(a.bayes_tree_update.unwrap().reused_cliques, 8);
}

#[test]
fn tree_refresh_conversion_fill_failure_cannot_commit_variables_or_factors() {
    let mut smoother = IsamSmoother::new(IsamOptions {
        qr: IncrementalQrOptions {
            max_entries: 5,
            ..Default::default()
        },
        batch_solver: Options {
            linear_solver: LinearSolver::SparseQr(SparseQrOptions {
                ordering: SparseOrdering::Natural,
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap()
    .with_bayes_tree()
    .unwrap();
    // Numeric R stores five entries. Symbolic separator fill requires a sixth.
    let update = SmootherUpdate {
        variables: vec![variable(0, 0.), variable(1, 0.), variable(2, 0.)],
        factors: vec![SmootherFactor::analytic(
            &[0, 1, 2],
            3,
            Default::default(),
            |v| {
                let j = DMatrix::from_row_slice(3, 3, &[1., 1., 1., 0., 1., 0., 0., 0., 1.]);
                Ok((
                    DVector::from_vec(vec![v[0][0] + v[1][0] + v[2][0], v[1][0], v[2][0]]),
                    vec![
                        j.columns(0, 1).into_owned(),
                        j.columns(1, 1).into_owned(),
                        j.columns(2, 1).into_owned(),
                    ],
                ))
            },
        )],
        ..Default::default()
    };
    assert!(matches!(
        smoother.update(update, true),
        Err(Error::SparseFillLimit { limit: 5 })
    ));
    assert_eq!(smoother.variable_count(), 0);
    assert_eq!(smoother.factor_count(), 0);
    let report = smoother
        .update(
            SmootherUpdate {
                variables: vec![variable(10, 0.)],
                factors: vec![observation(10, 1., Rc::new(Cell::new(0)))],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    assert_eq!(report.estimates.len(), 1);
    assert!((report.estimates[&10][0] - 1.).abs() < 1e-12);
}

#[test]
fn rank_failure_preserves_committed_tree_and_allows_later_valid_extension() {
    let mut smoother = IsamSmoother::new(IsamOptions {
        relinearization_threshold: 100.,
        ..Default::default()
    })
    .unwrap()
    .with_bayes_tree()
    .unwrap();
    smoother
        .update(
            SmootherUpdate {
                variables: vec![variable(0, 0.)],
                factors: vec![observation(0, 2., Rc::new(Cell::new(0)))],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    assert!(matches!(
        smoother.update(
            SmootherUpdate {
                variables: vec![variable(1, 0.)],
                ..Default::default()
            },
            false
        ),
        Err(Error::RankDeficient {
            rank: 1,
            parameters: 2
        })
    ));
    assert_eq!(smoother.variable_count(), 1);
    assert_eq!(smoother.factor_count(), 1);
    assert_eq!(smoother.estimate(0).unwrap()[0], 2.);
    let report = smoother
        .update(
            SmootherUpdate {
                variables: vec![variable(1, 0.)],
                factors: vec![observation(1, 3., Rc::new(Cell::new(0)))],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    assert_eq!(report.bayes_tree_update.unwrap().reeliminated_cliques, 1);
    assert_eq!(report.estimates[&1][0], 3.);
}

#[test]
fn nonlinear_pose_stream_keeps_existing_independent_dense_and_stationarity_gates() {
    let baseline = fixture::validate(50, &mut std::io::sink()).unwrap();
    let result = fixture::validate_observed_with_smoother(50, &mut std::io::sink(), &mut (), |s| {
        Ok(s.with_bayes_tree()?.with_batch_ordering_reuse()?)
    })
    .unwrap();
    assert_eq!(result.checkpoints, 2);
    assert_eq!(result.checkpoints, baseline.checkpoints);
    assert!(result.incremental > 0);
    assert!(result.interval > 0);
    assert!(result.max_pose_difference < 2e-6);
    assert!(result.max_refreshed_gradient < 1e-6);
}
