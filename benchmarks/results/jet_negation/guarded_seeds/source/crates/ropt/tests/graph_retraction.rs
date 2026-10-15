use ropt::{factor::*, manifold::*, *};

fn mixed_graph() -> (FactorGraph, Vec<VariableId>) {
    let mut graph = FactorGraph::new();
    let rotation = SO3::exp(nalgebra::Vector3::new(0.2, -0.4, 0.1));
    let mut q = so3_values(rotation);
    q *= -(1.0 + 4e-9); // Accepted, slightly nonunit quaternion with negative sign.
    let cases = [
        (
            VariableKind::Euclidean(3),
            DVector::from_vec(vec![-0., 2., -3.]),
        ),
        (VariableKind::SO2, DVector::from_element(1, 7.)),
        (VariableKind::SO3, q),
        (VariableKind::SE2, DVector::from_vec(vec![2., -1., -7.])),
        (VariableKind::Euclidean(1), DVector::from_element(1, 4.)),
        (
            VariableKind::SE3,
            se3_values(SE3 {
                translation: nalgebra::Vector3::new(1., -2., 3.),
                rotation,
            }),
        ),
    ];
    let ids = cases
        .into_iter()
        .map(|(kind, x)| graph.add_variable(kind, x).unwrap())
        .collect();
    (graph, ids)
}

#[test]
fn mixed_graph_retraction_matches_owned_api_bits_across_fixed_layouts() {
    let (mut graph, ids) = mixed_graph();
    for fixed in [vec![], vec![0, 2], vec![1, 3, 5], (0..ids.len()).collect()] {
        for (i, &id) in ids.iter().enumerate() {
            graph.set_fixed(id, fixed.contains(&i)).unwrap();
        }
        for scale in [0., 1e-12, 1e-5, 0.3, 4.] {
            let x = graph.initial();
            let original = x.clone();
            let d = DVector::from_fn(graph.dimensions().1, |i, _| scale * (i as f64 - 4.) / 5.);
            let updated = graph.retract(&x, &d).unwrap();
            for &id in &ids {
                let value = graph.value(&x, id).unwrap();
                let expected = match graph.tangent_range(id).unwrap() {
                    Some(t) => graph
                        .variable_kind(id)
                        .unwrap()
                        .retract(&value, &d.rows(t.start, t.len()).into_owned())
                        .unwrap(),
                    None => value,
                };
                let actual = graph.value(&updated, id).unwrap();
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
            assert_eq!(x, original);
        }
    }
}

#[test]
fn invalid_fixed_and_disconnected_values_and_steps_are_rejected() {
    let (mut graph, ids) = mixed_graph(); // No factors: every variable is disconnected.
    graph.set_fixed(ids[2], true).unwrap();
    let x = graph.initial();
    let d = DVector::zeros(graph.dimensions().1);
    let quaternion_offset = 4;
    for invalid in [0., f64::NAN, f64::INFINITY] {
        let mut bad = x.clone();
        bad[quaternion_offset] = invalid;
        assert!(graph.retract(&bad, &d).is_err());
        assert_eq!(bad[quaternion_offset].to_bits(), invalid.to_bits());
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut bad = d.clone();
        let last = bad.len() - 1;
        bad[last] = invalid;
        assert!(matches!(graph.retract(&x, &bad), Err(Error::NonFinite)));
    }
    assert!(matches!(
        graph.retract(&DVector::zeros(x.len() - 1), &d),
        Err(Error::InvalidDimensions)
    ));
    assert!(matches!(
        graph.retract(&x, &DVector::zeros(d.len() + 1)),
        Err(Error::InvalidDimensions)
    ));
    // Preserve variable-order failure precedence: earlier step validation wins
    // over a later invalid fixed quaternion.
    let mut bad_x = x.clone();
    bad_x[quaternion_offset] = 0.;
    let mut bad_d = d.clone();
    bad_d[0] = f64::NAN;
    assert!(matches!(
        graph.retract(&bad_x, &bad_d),
        Err(Error::NonFinite)
    ));
    let good = graph.retract(&x, &d).unwrap();
    assert!(good.iter().all(|v| v.is_finite()));
}

#[test]
fn overflow_errors_preserve_inputs_and_agree_with_owned_retraction() {
    for kind in [
        VariableKind::Euclidean(1),
        VariableKind::SE2,
        VariableKind::SE3,
    ] {
        let mut graph = FactorGraph::new();
        let mut x = DVector::zeros(kind.dimensions().0);
        x[0] = f64::MAX;
        if let VariableKind::SE3 = kind {
            x[3] = 1.;
        }
        graph.add_variable(kind, x.clone()).unwrap();
        let mut d = DVector::zeros(kind.dimensions().1);
        d[0] = f64::MAX;
        assert!(matches!(kind.retract(&x, &d), Err(Error::NonFinite)));
        assert!(matches!(graph.retract(&x, &d), Err(Error::NonFinite)));
        assert_eq!(graph.initial(), x);
        assert_eq!(x[0], f64::MAX);
    }
}
