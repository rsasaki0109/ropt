use ropt::{
    DVector, LocalProblem, factor::*, fixed_lag::*, incremental_qr::IncrementalQr, isam::*,
    manifold::*,
};

#[test]
fn mixed_charts_preserve_insertion_layout_cost_and_refreshed_coordinates() {
    mixed_charts(false);
}

#[test]
fn batch_ordering_preserves_mixed_charts_and_fixed_insertion_layout() {
    mixed_charts(true);
}

fn mixed_charts(reuse_ordering: bool) {
    let data = [
        (
            20,
            VariableKind::Euclidean(2),
            DVector::from_vec(vec![2., -1.]),
            false,
        ),
        (
            5,
            VariableKind::SO2,
            DVector::from_element(1, std::f64::consts::PI - 0.001),
            false,
        ),
        (
            90,
            VariableKind::SE2,
            se2_values(SE2::exp(nalgebra::Vector3::new(2., -1., 0.4))),
            true,
        ),
        (
            3,
            VariableKind::SO3,
            so3_values(SO3::exp(nalgebra::Vector3::new(0.3, -0.2, 0.1))),
            false,
        ),
        (
            45,
            VariableKind::SE2,
            se2_values(SE2::exp(nalgebra::Vector3::new(2., -1., 0.4))),
            false,
        ),
        (
            1,
            VariableKind::SE3,
            se3_values(SE3::exp(Tangent6::new(2., -1., 0.3, 0.1, 0.2, -0.1))),
            false,
        ),
    ];
    let mut graph = FactorGraph::new();
    let mut update = SmootherUpdate::default();
    let mut layout = Vec::new();
    for (key, kind, target, fixed) in data {
        let anchor = if fixed {
            target.clone()
        } else {
            kind.retract(
                &target,
                &DVector::from_fn(kind.dimensions().1, |i, _| {
                    0.005 * (i + 1) as f64 * (key % 7 + 1) as f64
                }),
            )
            .unwrap()
        };
        let id = graph.add_variable(kind, anchor.clone()).unwrap();
        graph.set_fixed(id, fixed).unwrap();
        let prior = target.clone();
        graph
            .add_autodiff_factor(&[id], kind.dimensions().1, move |v| {
                kind.local_autodiff(&prior, v[0])
            })
            .unwrap();
        update.variables.push(WindowVariable {
            key,
            kind,
            value: anchor.clone(),
            timestamp: None,
            fixed,
        });
        update.factors.push(SmootherFactor::autodiff(
            &[key],
            kind.dimensions().1,
            kind.dimensions().1,
            Default::default(),
            move |v| kind.local_autodiff(&target, v[0]),
        ));
        layout.push((key, kind, id, anchor));
    }
    let (r, j) = graph.evaluate_sparse(&graph.initial()).unwrap();
    let mut qr = IncrementalQr::new(graph.dimensions().1, Default::default()).unwrap();
    qr.append(&j, &r).unwrap();
    let state = graph
        .retract(&graph.initial(), &qr.solve().unwrap())
        .unwrap();
    let mut local = DVector::zeros(graph.dimensions().1);
    let mut expected_max = 0.0_f64;
    let mut smoother = if reuse_ordering {
        IsamSmoother::new(IsamOptions {
            batch_solver: ropt::Options {
                linear_solver: ropt::LinearSolver::SparseLdl(Default::default()),
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap()
        .with_batch_ordering_reuse()
        .unwrap()
    } else {
        IsamSmoother::new(Default::default()).unwrap()
    };
    let report = smoother.update(update, false).unwrap();
    assert_eq!(report.kind, IsamUpdateKind::Incremental);
    for (key, kind, id, anchor) in &layout {
        assert_eq!(report.estimates[key], graph.value(&state, *id).unwrap());
        if let Some(range) = graph.tangent_range(*id).unwrap() {
            let coordinates = kind.local(anchor, &report.estimates[key]).unwrap();
            expected_max = expected_max.max(coordinates.norm());
            local
                .rows_mut(range.start, range.len())
                .copy_from(&coordinates);
        }
    }
    assert_eq!(report.max_anchor_delta.to_bits(), expected_max.to_bits());
    assert_eq!(
        report.linearized_cost.to_bits(),
        qr.cost(&local).unwrap().to_bits()
    );
    let refreshed = smoother.update(Default::default(), true).unwrap();
    let mut expected_max = 0.0_f64;
    for (key, kind, id, _) in &layout {
        if graph.tangent_range(*id).unwrap().is_some() {
            let estimate = &refreshed.estimates[key];
            expected_max = expected_max.max(kind.local(estimate, estimate).unwrap().norm());
        } else {
            assert_eq!(refreshed.estimates[key], report.estimates[key]);
        }
    }
    assert_eq!(refreshed.max_anchor_delta.to_bits(), expected_max.to_bits());
    assert!(refreshed.linearized_cost < 1e-14);
}

#[test]
fn a_full_turn_raw_step_still_triggers_refresh_in_a_small_principal_chart() {
    let seed = std::f64::consts::TAU.atan();
    let raw_step = -seed.sin() / seed.cos();
    assert!(raw_step.abs() > 6.);
    let kind = VariableKind::SO2;
    let anchor = DVector::from_element(1, seed);
    let wrapped = kind
        .retract(&anchor, &DVector::from_element(1, raw_step))
        .unwrap();
    assert!(kind.local(&anchor, &wrapped).unwrap().norm() < 1e-14);
    let mut smoother = IsamSmoother::new(Default::default()).unwrap();
    let report = smoother
        .update(
            SmootherUpdate {
                variables: vec![WindowVariable {
                    key: 8,
                    kind,
                    value: anchor,
                    timestamp: None,
                    fixed: false,
                }],
                factors: vec![SmootherFactor::analytic(&[8], 1, Default::default(), |v| {
                    Ok((
                        DVector::from_element(1, v[0][0].sin()),
                        vec![ropt::DMatrix::from_element(1, 1, v[0][0].cos())],
                    ))
                })],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    assert_eq!(
        report.relinearization_reason,
        Some(IsamRelinearizationReason::Threshold)
    );
    assert!(report.estimates[&8][0].sin().abs() < 1e-8);
    assert_eq!(report.max_anchor_delta, 0.);
}
