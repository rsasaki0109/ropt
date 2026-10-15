use ropt::{
    DMatrix, DVector, Error,
    sparse::CsrMatrix,
    sparse_direct::{SparseLdlOptions, SparseLdlSymbolic, SparseOrdering, solve_normal},
};

// Independent dense support oracle, including mandatory normal diagonals.
fn normal_entries(j: &CsrMatrix) -> usize {
    let (m, n) = j.shape();
    let mut graph = vec![vec![false; n]; n];
    for (k, column) in graph.iter_mut().enumerate() {
        column[k] = true;
    }
    for r in 0..m {
        let support = j.row(r).unwrap().map(|(col, _)| col).collect::<Vec<_>>();
        for (k, &a) in support.iter().enumerate() {
            for &b in &support[k..] {
                graph[a][b] = true;
            }
        }
    }
    graph.iter().flatten().filter(|&&present| present).count()
}

#[test]
fn normal_budget_boundaries_cover_duplicates_gaps_and_tree_promotion() {
    for n in [1, 9, 63, 64, 65, 72] {
        let mut supports = vec![vec![], vec![], (0..n).step_by(2).collect::<Vec<_>>()];
        supports.push((0..n).filter(|k| k % 3 != 1).collect());
        supports.push(supports[2].clone());
        supports.push((0..n).collect());
        supports.push(supports.last().unwrap().clone());
        for reverse in [false, true] {
            let m = supports.len();
            let triplets = supports
                .iter()
                .enumerate()
                .flat_map(|(r, support)| {
                    support
                        .iter()
                        .map(move |&col| (if reverse { m - 1 - r } else { r }, col, 1.))
                })
                .collect();
            let j = CsrMatrix::from_triplets(m, n, triplets).unwrap();
            let required = normal_entries(&j);
            assert_eq!(required, n * (n + 1) / 2);
            for limit in 0..=required + 1 {
                let options = SparseLdlOptions {
                    ordering: SparseOrdering::Natural,
                    max_entries: limit,
                    ..Default::default()
                };
                match SparseLdlSymbolic::analyze(&j, &options) {
                    Err(error) => assert_eq!(
                        error,
                        if limit == 0 {
                            Error::InvalidOptions
                        } else {
                            assert!(limit < required);
                            Error::SparseFillLimit { limit }
                        }
                    ),
                    Ok(symbolic) => {
                        assert!(limit >= required);
                        assert_eq!(symbolic.normal_entries(), required);
                        assert_eq!(symbolic.factor_entries(), required);
                        assert_eq!(symbolic.permutation(), &(0..n).collect::<Vec<_>>());
                    }
                }
            }
        }
    }
}

#[test]
fn gapped_normal_graphs_match_dense_cholesky_with_all_orderings() {
    for n in [6, 9, 72] {
        let mut triplets = (0..n).map(|k| (k, k, 2.)).collect::<Vec<_>>();
        // A high-degree hub exercises tree promotion; non-adjacent repeated
        // supports still contribute every row numerically, with changing signs.
        for repeat in 0..2 {
            for k in 1..n {
                let row = n + repeat * n + k;
                triplets.push((row, 0, if repeat == 0 { 0.3 } else { -0.7 }));
                triplets.push((row, k, 0.1 + k as f64 * 0.01));
                if k + 2 < n {
                    triplets.push((row, k + 2, -0.2));
                }
            }
        }
        let j = CsrMatrix::from_triplets(3 * n + 1, n, triplets).unwrap();
        let mut dense = DMatrix::zeros(j.shape().0, n);
        for (r, c, v) in j.entries() {
            dense[(r, c)] = v;
        }
        let rhs = DVector::from_fn(n, |k, _| (k as f64 * 0.31).cos());
        for ordering in [
            SparseOrdering::Natural,
            SparseOrdering::MinimumDegree,
            SparseOrdering::BlockMinimumDegree { block_size: 3 },
        ] {
            let options = SparseLdlOptions {
                ordering,
                ..Default::default()
            };
            assert_eq!(
                SparseLdlSymbolic::analyze(&j, &options)
                    .unwrap()
                    .normal_entries(),
                normal_entries(&j)
            );
            for damping in [0., 1e-6, 0.1] {
                let mut normal = dense.transpose() * &dense;
                for k in 0..n {
                    normal[(k, k)] += damping * normal[(k, k)].max(1.);
                }
                let expected = normal.clone().cholesky().unwrap().solve(&rhs);
                let solved = solve_normal(&j, &rhs, damping, &options).unwrap();
                assert!((&solved.x - expected).norm() < 1e-11);
                assert!((normal * solved.x - &rhs).norm() < 1e-10);
                assert_eq!(solved.normal_entries, normal_entries(&j));
            }
        }
    }
    let empty = CsrMatrix::from_triplets(0, 9, vec![]).unwrap();
    let options = SparseLdlOptions::default();
    assert_eq!(
        SparseLdlSymbolic::analyze(&empty, &options)
            .unwrap()
            .normal_entries(),
        9
    );
    let zero_columns = CsrMatrix::from_triplets(1, 0, vec![]).unwrap();
    assert!(matches!(
        SparseLdlSymbolic::analyze(&zero_columns, &options),
        Err(Error::InvalidDimensions)
    ));
}
