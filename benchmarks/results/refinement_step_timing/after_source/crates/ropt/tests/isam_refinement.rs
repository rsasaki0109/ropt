use ropt::{
    DMatrix, DVector, Error, LinearSolver, LocalProblem, Options, SparseLdlOptions, SparseOrdering,
    factor::*, fixed_lag::*, incremental_qr::IncrementalQr, isam::*, solve_local,
};

fn variable(key: u64, value: f64, fixed: bool) -> WindowVariable {
    WindowVariable {
        key,
        value: DVector::from_element(1, value),
        kind: VariableKind::Euclidean(1),
        timestamp: None,
        fixed,
    }
}

fn batch_options() -> Options {
    Options {
        linear_solver: LinearSolver::SparseLdl(SparseLdlOptions {
            ordering: SparseOrdering::MinimumDegree,
            ..Default::default()
        }),
        gradient_tolerance: 1e-3,
        cost_tolerance: 1e-5,
        initial_damping: 1e-2,
        ..Default::default()
    }
}

#[test]
fn refined_estimates_keep_the_batch_anchor_for_cached_updates_and_rollback() {
    for cap in [1, 4] {
        for backend in 0..3 {
            check_refined_anchor_and_rollback(cap, backend);
        }
    }
}

fn check_refined_anchor_and_rollback(cap: usize, backend: usize) {
    let mut graph = FactorGraph::new();
    let mut update = SmootherUpdate::default();
    let mut ids = vec![];
    // Insertion layout differs from key order; a fixed block lies in the middle.
    for (key, value, fixed) in [
        (90, 1.5, false),
        (1, 0., true),
        (40, 2.5, false),
        (3, 3.5, false),
    ] {
        let id = graph
            .add_variable(VariableKind::Euclidean(1), DVector::from_element(1, value))
            .unwrap();
        graph.set_fixed(id, fixed).unwrap();
        ids.push(id);
        update.variables.push(variable(key, value, fixed));
    }
    graph
        .add_autodiff_factor(&[ids[0]], 1, |v| Ok(vec![&v[0][0] * &v[0][0] - 4.]))
        .unwrap();
    update.factors.push(SmootherFactor::autodiff(
        &[90],
        1,
        1,
        Default::default(),
        |v| Ok(vec![&v[0][0] * &v[0][0] - 4.]),
    ));
    for (index, key, target) in [(2, 40, 1.), (3, 3, 2.)] {
        graph
            .add_analytic_factor(&[ids[0], ids[index]], 1, move |v| {
                Ok((
                    &v[1] - &v[0] - DVector::from_element(1, target),
                    vec![-DMatrix::identity(1, 1), DMatrix::identity(1, 1)],
                ))
            })
            .unwrap();
        update.factors.push(SmootherFactor::analytic(
            &[90, key],
            1,
            Default::default(),
            move |v| {
                Ok((
                    &v[1] - &v[0] - DVector::from_element(1, target),
                    vec![-DMatrix::identity(1, 1), DMatrix::identity(1, 1)],
                ))
            },
        ));
    }
    let options = batch_options();
    let batch = solve_local(&graph, graph.initial(), &options).unwrap();
    let (r, j) = graph.evaluate_sparse(&batch.x).unwrap();
    let mut reference = IncrementalQr::new(3, Default::default()).unwrap();
    reference.append(&j, &r).unwrap();
    let step = reference.solve().unwrap();
    let expected_cached = graph.retract(&batch.x, &step).unwrap();
    let mut smoother = IsamSmoother::new(IsamOptions {
        batch_solver: options,
        relinearization_threshold: 100.,
        relinearization_interval: 100,
        ..Default::default()
    })
    .unwrap()
    .with_batch_ordering_reuse()
    .unwrap()
    .with_final_refinement(cap)
    .unwrap();
    smoother = match backend {
        1 => smoother.with_bayes_tree().unwrap(),
        2 => smoother.with_selective_bayes_tree(1).unwrap(),
        _ => smoother,
    };
    let refreshed = smoother.update(update, true).unwrap();
    assert!(refreshed.refinement_steps > 0);
    assert!(refreshed.refinement_steps <= cap);
    assert!((refreshed.estimates[&90][0] - 2.).abs() < (batch.x[0] - 2.).abs());
    assert_eq!(refreshed.estimates[&1][0], 0.);
    let before = refreshed.estimates.clone();
    let failed = smoother.update(
        SmootherUpdate {
            variables: vec![variable(100, 0., false)],
            factors: vec![SmootherFactor::analytic(
                &[100],
                1,
                Default::default(),
                |_| Err(Error::Evaluation("deliberate failure".into())),
            )],
            ..Default::default()
        },
        true,
    );
    assert!(matches!(failed, Err(Error::Evaluation(_))));
    assert_eq!(smoother.variable_count(), 4);
    assert_eq!(smoother.factor_count(), 3);
    for (key, value) in before {
        assert_eq!(smoother.estimate(key), Some(&value));
    }
    let cached = smoother.update(SmootherUpdate::default(), false).unwrap();
    assert_eq!(cached.kind, IsamUpdateKind::Incremental);
    assert_eq!(cached.refinement_steps, 0);
    for (index, key) in [(0, 90), (2, 40), (3, 3)] {
        assert!(
            (&cached.estimates[&key] - graph.value(&expected_cached, ids[index]).unwrap()).norm()
                < 1e-12
        );
    }
    assert!((cached.linearized_cost - reference.cost(&step).unwrap()).abs() < 1e-20);
}

fn prior(key: u64, target: f64) -> SmootherFactor {
    SmootherFactor::analytic(&[key], 1, Default::default(), move |v| {
        Ok((
            &v[0] - DVector::from_element(1, target),
            vec![DMatrix::identity(1, 1)],
        ))
    })
}

#[test]
fn initially_converged_refresh_extends_an_ordered_cache_then_appends_in_external_coordinates() {
    let mut smoother = IsamSmoother::new(IsamOptions {
        batch_solver: batch_options(),
        relinearization_threshold: 100.,
        relinearization_interval: 100,
        ..Default::default()
    })
    .unwrap()
    .with_batch_ordering_reuse()
    .unwrap();
    smoother
        .update(
            SmootherUpdate {
                variables: vec![variable(90, 0., false)],
                factors: vec![prior(90, 2.)],
                ..Default::default()
            },
            true,
        )
        .unwrap();
    let refreshed = smoother
        .update(
            SmootherUpdate {
                variables: vec![variable(1, 3., false)],
                factors: vec![prior(1, 3.)],
                ..Default::default()
            },
            true,
        )
        .unwrap();
    assert_eq!(refreshed.batch_trials, 0);
    let appended = smoother
        .update(
            SmootherUpdate {
                variables: vec![variable(40, -1., false)],
                factors: vec![prior(40, 4.), prior(1, 5.)],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    assert_eq!(appended.kind, IsamUpdateKind::Incremental);
    for (key, target) in [(90, 2.), (1, 4.), (40, 4.)] {
        assert!((appended.estimates[&key][0] - target).abs() < 1e-12);
    }
    assert!((appended.linearized_cost - 1.).abs() < 1e-12);
}

#[test]
fn refinement_rejects_invalid_limits_and_incompatible_ordering_backend() {
    for count in [0, 201] {
        assert!(matches!(
            IsamSmoother::new(Default::default())
                .unwrap()
                .with_final_refinement(count),
            Err(Error::InvalidOptions)
        ));
    }
    assert!(matches!(
        IsamSmoother::new(Default::default())
            .unwrap()
            .with_batch_ordering_reuse(),
        Err(Error::InvalidOptions)
    ));
}
