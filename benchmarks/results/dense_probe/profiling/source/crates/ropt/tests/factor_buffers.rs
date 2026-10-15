use ropt::{factor::*, *};

#[test]
fn alternating_callback_arity_and_dimensions_do_not_expose_stale_values() {
    let mut graph = FactorGraph::new();
    let a = graph
        .add_variable(VariableKind::Euclidean(2), DVector::from_vec(vec![1., 2.]))
        .unwrap();
    let b = graph
        .add_variable(VariableKind::Euclidean(1), DVector::from_element(1, 3.))
        .unwrap();
    let c = graph
        .add_variable(
            VariableKind::Euclidean(3),
            DVector::from_vec(vec![4., 5., 6.]),
        )
        .unwrap();
    graph
        .add_analytic_factor(&[a, b, c], 1, |v| {
            assert_eq!(
                v.iter().map(DVector::len).collect::<Vec<_>>(),
                vec![2, 1, 3]
            );
            Ok((
                DVector::from_element(1, v[0][0] + v[1][0] + v[2][0]),
                vec![
                    DMatrix::from_row_slice(1, 2, &[1., 0.]),
                    DMatrix::identity(1, 1),
                    DMatrix::from_row_slice(1, 3, &[1., 0., 0.]),
                ],
            ))
        })
        .unwrap();
    graph
        .add_numeric_factor(&[b], 1, 1e-6, |v| {
            assert_eq!(v.len(), 1);
            assert_eq!(v[0].len(), 1);
            Ok(DVector::from_element(1, v[0][0].powi(2)))
        })
        .unwrap();
    graph
        .add_analytic_factor(&[c, a], 1, |v| {
            assert_eq!(v.len(), 2);
            assert_eq!(v[0].len(), 3);
            assert_eq!(v[1].len(), 2);
            Ok((
                DVector::from_element(1, v[0][2] - v[1][1]),
                vec![
                    DMatrix::from_row_slice(1, 3, &[0., 0., 1.]),
                    DMatrix::from_row_slice(1, 2, &[0., -1.]),
                ],
            ))
        })
        .unwrap();
    for shift in [0., 10.] {
        let x = graph.initial().add_scalar(shift);
        let mut r = DVector::zeros(3);
        let mut j = DMatrix::zeros(3, 6);
        graph.evaluate_local(&x, &mut r, &mut j).unwrap();
        let (sr, sj) = graph.evaluate_sparse(&x).unwrap();
        assert_eq!(r, sr);
        assert_eq!(
            r,
            DVector::from_vec(vec![8. + 3. * shift, (3. + shift).powi(2), 4.])
        );
        let expected = DMatrix::from_row_slice(
            3,
            6,
            &[
                1.,
                0.,
                1.,
                1.,
                0.,
                0.,
                0.,
                0.,
                2. * (3. + shift),
                0.,
                0.,
                0.,
                0.,
                -1.,
                0.,
                0.,
                0.,
                1.,
            ],
        );
        assert!((&j - &expected).norm() < 1e-7);
        for k in 0..6 {
            let mut basis = DVector::zeros(6);
            basis[k] = 1.;
            assert!((sj.multiply(&basis).unwrap() - j.column(k)).norm() < 1e-12);
        }
    }
}

#[test]
fn disconnected_and_fixed_manifold_blocks_are_validated_before_any_callback() {
    use std::{cell::Cell, rc::Rc};
    let mut graph = FactorGraph::new();
    let a = graph
        .add_variable(VariableKind::Euclidean(1), DVector::from_element(1, 2.))
        .unwrap();
    let b = graph
        .add_variable(VariableKind::SO3, DVector::from_vec(vec![1., 0., 0., 0.]))
        .unwrap();
    graph.set_fixed(b, true).unwrap();
    let calls = Rc::new(Cell::new(0));
    let probe = calls.clone();
    graph
        .add_analytic_factor(&[a], 1, move |v| {
            probe.set(probe.get() + 1);
            Ok((v[0].clone(), vec![DMatrix::identity(1, 1)]))
        })
        .unwrap();
    for invalid in [0.5, f64::NAN] {
        let mut x = graph.initial();
        x[1] = invalid;
        let mut r = DVector::zeros(1);
        let mut j = DMatrix::zeros(1, 1);
        assert!(graph.evaluate_local(&x, &mut r, &mut j).is_err());
        assert!(graph.evaluate_sparse(&x).is_err());
        assert_eq!(calls.get(), 0);
    }
    graph.evaluate_sparse(&graph.initial()).unwrap();
    assert_eq!(calls.get(), 1);
}
