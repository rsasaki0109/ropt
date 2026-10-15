use ropt::{sparse::CsrMatrix, sparse_direct::*, *};

fn dense_normal(j: &DMatrix<f64>, damping: f64) -> DMatrix<f64> {
    let mut h = j.transpose() * j;
    for k in 0..h.ncols() {
        h[(k, k)] += damping * h[(k, k)].max(1.);
    }
    h
}

#[test]
fn descending_high_degree_star_preserves_fill_order_limits_and_dense_solution() {
    let n = 96;
    let mut dense = DMatrix::zeros(n, n);
    let mut entries = vec![(n - 1, 0, 1.)];
    dense[(n - 1, 0)] = 1.;
    for row in 0..n - 1 {
        let column = n - 1 - row;
        entries.extend([(row, 0, -1.), (row, column, 1.)]);
        dense[(row, 0)] = -1.;
        dense[(row, column)] = 1.;
    }
    let j = CsrMatrix::from_triplets(n, n, entries).unwrap();
    let options = SparseLdlOptions {
        max_entries: 2 * n - 1,
        ..Default::default()
    };
    let symbolic = SparseLdlSymbolic::analyze(&j, &options).unwrap();
    // Every leaf is degree one. At the final tie, root 0 precedes leaf n-1.
    let expected = (1..n - 1).chain([0, n - 1]).collect::<Vec<_>>();
    assert_eq!(symbolic.permutation(), expected);
    assert_eq!(symbolic.normal_entries(), 2 * n - 1);
    assert_eq!(symbolic.factor_entries(), 2 * n - 1);
    assert_eq!(
        SparseLdlSymbolic::analyze(
            &j,
            &SparseLdlOptions {
                max_entries: 2 * n - 2,
                ..options
            }
        )
        .unwrap_err(),
        Error::SparseFillLimit { limit: 2 * n - 2 }
    );
    assert_eq!(
        SparseLdlSymbolic::analyze(
            &j,
            &SparseLdlOptions {
                ordering: SparseOrdering::Natural,
                ..options
            }
        )
        .unwrap_err(),
        Error::SparseFillLimit { limit: 2 * n - 1 }
    );
    let b = DVector::from_fn(n, |k, _| (k as f64 * 0.17).sin());
    let solved = solve_normal(&j, &b, 0.01, &options).unwrap();
    let h = dense_normal(&dense, 0.01);
    assert!((&solved.x - h.clone().cholesky().unwrap().solve(&b)).norm() < 1e-10);
    assert!((&h * &solved.x - &b).norm() < 1e-10);
}

#[test]
fn repeated_dense_rows_empty_rows_and_cancelled_triplets_keep_exact_pattern_budget() {
    let n = 96;
    let m = n + 4;
    let mut dense = DMatrix::zeros(m, n);
    let mut entries = vec![(0, 0, 2.), (0, 0, -2.)];
    for k in 0..n {
        let value = 1. + (k % 7) as f64 * 0.1;
        dense[(1, k)] = value;
        dense[(k + 2, k)] = 1.;
        dense[(m - 1, k)] = -value;
        entries.extend([(1, k, value), (k + 2, k, 1.), (m - 1, k, -value)]);
    }
    let j = CsrMatrix::from_triplets(m, n, entries).unwrap();
    let count = n * (n + 1) / 2;
    let b = DVector::from_fn(n, |k, _| (k as f64 * 0.37).cos());
    let h = dense_normal(&dense, 0.001);
    let expected = h.clone().cholesky().unwrap().solve(&b);
    for ordering in [
        SparseOrdering::Natural,
        SparseOrdering::MinimumDegree,
        SparseOrdering::BlockMinimumDegree { block_size: 3 },
    ] {
        let options = SparseLdlOptions {
            ordering,
            max_entries: count,
            ..Default::default()
        };
        let symbolic = SparseLdlSymbolic::analyze(&j, &options).unwrap();
        assert_eq!(symbolic.normal_entries(), count);
        assert_eq!(symbolic.factor_entries(), count);
        assert_eq!(symbolic.permutation(), (0..n).collect::<Vec<_>>());
        assert_eq!(
            SparseLdlSymbolic::analyze(
                &j,
                &SparseLdlOptions {
                    max_entries: count - 1,
                    ..options
                }
            )
            .unwrap_err(),
            Error::SparseFillLimit { limit: count - 1 }
        );
        let solved = solve_normal(&j, &b, 0.001, &options).unwrap();
        assert!((&solved.x - &expected).norm() < 1e-10);
        assert!((&h * &solved.x - &b).norm() < 1e-10);
    }
}
