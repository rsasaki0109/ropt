use ropt::{DMatrix, DVector, Error, sparse::CsrMatrix, sparse_direct::*};

fn csr(j: &DMatrix<f64>) -> CsrMatrix {
    CsrMatrix::from_triplets(
        j.nrows(),
        j.ncols(),
        (0..j.nrows())
            .flat_map(|r| {
                (0..j.ncols()).filter_map(move |c| (j[(r, c)] != 0.).then_some((r, c, j[(r, c)])))
            })
            .collect(),
    )
    .unwrap()
}

fn normal(j: &DMatrix<f64>, damping: f64) -> DMatrix<f64> {
    let mut h = j.transpose() * j;
    for k in 0..h.ncols() {
        h[(k, k)] += damping * h[(k, k)].max(1.);
    }
    h
}

#[test]
fn repeated_support_plans_keep_all_numeric_rows_empty_boundaries_and_column_identities() {
    let n = 6;
    let mut dense = DMatrix::zeros(31, n);
    let supports = [[0, 2, 4], [1, 3, 5], [0, 1, 5], [2, 3, 4]];
    for (group, columns) in supports.iter().enumerate() {
        for repeat in 0..5 {
            for (k, &column) in columns.iter().enumerate() {
                dense[(1 + group * 6 + repeat, column)] =
                    (1. + (repeat + k) as f64 * 0.17) * if repeat % 2 == 0 { 1. } else { -1. };
            }
        }
    }
    // Empty rows separate repeated runs; equal row widths have different indices.
    dense
        .view_mut((25, 0), (n, n))
        .copy_from(&DMatrix::identity(n, n));
    let j = csr(&dense);
    let rhs = DVector::from_fn(n, |k, _| (k as f64 * 0.31).cos());
    let h = normal(&dense, 0.01);
    let expected = h.clone().cholesky().unwrap().solve(&rhs);
    for ordering in [
        SparseOrdering::Natural,
        SparseOrdering::MinimumDegree,
        SparseOrdering::BlockMinimumDegree { block_size: 3 },
    ] {
        let planned = SparseLdlOptions {
            ordering,
            max_entries: 1000,
            ..Default::default()
        };
        let bounded = SparseLdlOptions {
            max_entries: n * (n + 1) / 2,
            ..planned
        };
        let a = SparseLdlSymbolic::analyze(&j, &planned).unwrap();
        let b = SparseLdlSymbolic::analyze(&j, &bounded).unwrap();
        assert_eq!(a.assembly_entries(), 126); // 20 width-three rows, six diagonals.
        assert_eq!(b.assembly_entries(), 0);
        assert_eq!(a.permutation(), b.permutation());
        assert_eq!(a.normal_entries(), b.normal_entries());
        assert_eq!(a.factor_entries(), b.factor_entries());
        let a = solve_normal(&j, &rhs, 0.01, &planned).unwrap();
        let b = solve_normal(&j, &rhs, 0.01, &bounded).unwrap();
        assert_eq!(a.x, b.x);
        assert_eq!(a.residual_norm, b.residual_norm);
        assert_eq!(a.refinement_steps, b.refinement_steps);
        assert!((&a.x - &expected).norm() < 1e-12);
        assert!((&h * &a.x - &rhs).norm() < 1e-12);
    }
}

#[test]
fn ascending_clique_fill_promotes_long_columns_and_preserves_the_exact_budget() {
    let n = 72;
    let mut dense = DMatrix::zeros(n, n);
    dense[(0, 0)] = 1.;
    for k in 1..n {
        dense[(k, 0)] = -1.;
        dense[(k, k)] = 1.;
    }
    let j = csr(&dense);
    let count = n * (n + 1) / 2;
    let options = SparseLdlOptions {
        ordering: SparseOrdering::Natural,
        max_entries: count,
        ..Default::default()
    };
    let symbolic = SparseLdlSymbolic::analyze(&j, &options).unwrap();
    assert_eq!(symbolic.permutation(), (0..n).collect::<Vec<_>>());
    assert_eq!(symbolic.normal_entries(), 2 * n - 1);
    assert_eq!(symbolic.factor_entries(), count);
    assert!(!symbolic.has_update_plan());
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
    let rhs = DVector::from_fn(n, |k, _| (k as f64 * 0.21).sin());
    let h = normal(&dense, 0.25);
    let expected = h.clone().cholesky().unwrap().solve(&rhs);
    let result = solve_normal(&j, &rhs, 0.25, &options).unwrap();
    assert!((&result.x - &expected).norm() < 1e-11);
    assert!((&h * &result.x - &rhs).norm() < 1e-11);
}
