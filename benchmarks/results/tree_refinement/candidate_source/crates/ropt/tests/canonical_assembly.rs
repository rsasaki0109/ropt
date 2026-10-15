use ropt::{factor::*, noise::NoiseModel, sparse::CsrMatrix, *};

fn check_dense_reference(graph: &FactorGraph, x: &DVector<f64>) -> CsrMatrix {
    let (_, n, m) = graph.dimensions();
    let mut r = DVector::zeros(m);
    let mut j = DMatrix::zeros(m, n);
    graph.evaluate_local(x, &mut r, &mut j).unwrap();
    let (sr, sparse) = graph.evaluate_sparse(x).unwrap();
    assert_eq!(sr, r);
    let expected = CsrMatrix::from_triplets(
        m,
        n,
        (0..m)
            .flat_map(|row| {
                let j = &j;
                (0..n).filter_map(move |col| {
                    (j[(row, col)] != 0.).then_some((row, col, j[(row, col)]))
                })
            })
            .collect(),
    )
    .unwrap();
    assert_eq!(
        sparse.entries().collect::<Vec<_>>(),
        expected.entries().collect::<Vec<_>>()
    );
    for row in 0..m {
        let entries = sparse.row(row).unwrap().collect::<Vec<_>>();
        assert!(entries.windows(2).all(|pair| pair[0].0 < pair[1].0));
        assert!(entries.iter().all(|&(_, value)| value != 0.));
    }
    sparse
}

#[test]
fn reversed_callback_blocks_whitening_and_changed_fixed_layout_match_dense_assembly() {
    let mut graph = FactorGraph::new();
    let a = graph
        .add_variable(VariableKind::Euclidean(2), DVector::from_vec(vec![1., 2.]))
        .unwrap();
    let b = graph
        .add_variable(VariableKind::SO3, DVector::from_vec(vec![1., 0., 0., 0.]))
        .unwrap();
    let c = graph
        .add_variable(
            VariableKind::Euclidean(3),
            DVector::from_vec(vec![4., 5., 6.]),
        )
        .unwrap();
    let d = graph
        .add_variable(VariableKind::Euclidean(1), DVector::from_element(1, 7.))
        .unwrap();
    graph
        .add_analytic_factor_with_options(
            &[c, a, b],
            3,
            FactorOptions {
                noise: NoiseModel::from_covariance(DMatrix::from_row_slice(
                    3,
                    3,
                    &[4., 1., 0.2, 1., 3., 0.5, 0.2, 0.5, 2.],
                ))
                .unwrap(),
                ..Default::default()
            },
            |v| {
                assert_eq!(v[0], DVector::from_vec(vec![4., 5., 6.]));
                assert_eq!(v[1], DVector::from_vec(vec![1., 2.]));
                assert_eq!(v[2], DVector::from_vec(vec![1., 0., 0., 0.]));
                Ok((
                    DVector::from_vec(vec![1., 2., 3.]),
                    vec![
                        DMatrix::from_row_slice(3, 3, &[1., -0., -2., 0., 0., 0., 0., 3., 0.]),
                        DMatrix::from_row_slice(3, 2, &[0., 4., 0., 0., -5., 0.]),
                        DMatrix::from_row_slice(3, 3, &[1., 0., 0., 0., 0., 0., 0., 1., -1.]),
                    ],
                ))
            },
        )
        .unwrap();
    graph
        .add_analytic_factor(&[b], 2, |v| {
            assert_eq!(v.len(), 1);
            assert_eq!(v[0].len(), 4);
            Ok((
                DVector::zeros(2),
                vec![DMatrix::from_row_slice(2, 3, &[0., 1., 0., 1., 0., 0.])],
            ))
        })
        .unwrap();
    for (af, bf, cf, df) in [
        (false, true, false, false),
        (true, false, false, false),
        (false, false, true, false),
        (true, true, true, false),
        (true, true, true, true),
    ] {
        for (id, fixed) in [(a, af), (b, bf), (c, cf), (d, df)] {
            graph.set_fixed(id, fixed).unwrap();
        }
        let sparse = check_dense_reference(&graph, &graph.initial());
        if bf {
            assert_eq!(sparse.row(3).unwrap().count(), 0);
            assert_eq!(sparse.row(4).unwrap().count(), 0);
        }
    }
}

#[test]
fn exact_zero_derivative_changes_and_tiny_nonzero_entries_keep_the_correct_pattern() {
    let mut graph = FactorGraph::new();
    let x = graph
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    let unused = graph
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    graph
        .add_analytic_factor(&[unused, x], 2, |v| {
            Ok((
                DVector::from_vec(vec![v[1][0].powi(2), 1e-200 * v[1][0]]),
                vec![
                    DMatrix::zeros(2, 1),
                    DMatrix::from_vec(2, 1, vec![2. * v[1][0], 1e-200]),
                ],
            ))
        })
        .unwrap();
    for value in [0., 1., -0., -2.] {
        let state = DVector::from_vec(vec![value, 0.]);
        let sparse = check_dense_reference(&graph, &state);
        assert_eq!(sparse.nnz(), if value == 0. { 1 } else { 2 });
        assert_eq!(
            sparse.row(1).unwrap().collect::<Vec<_>>(),
            vec![(0, 1e-200)]
        );
    }
}
