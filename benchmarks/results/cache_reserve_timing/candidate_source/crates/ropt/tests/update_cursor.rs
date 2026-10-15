use ropt::{DMatrix, DVector, sparse::CsrMatrix, sparse_direct::*};

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

fn check(j: &DMatrix<f64>, options: SparseLdlOptions, limit: usize) {
    let a = SparseLdlSymbolic::analyze(&csr(j), &options).unwrap();
    let bounded = SparseLdlOptions {
        max_entries: limit,
        ..options
    };
    let b = SparseLdlSymbolic::analyze(&csr(j), &bounded).unwrap();
    assert!(a.has_update_plan());
    assert!(!b.has_update_plan());
    assert_eq!(a.permutation(), b.permutation());
    assert_eq!(a.factor_entries(), b.factor_entries());
    let rhs = DVector::from_fn(j.ncols(), |k, _| (k as f64 * 0.17).cos());
    for damping in [0., 1e-6, 0.1] {
        let a = solve_normal(&csr(j), &rhs, damping, &options).unwrap();
        let b = solve_normal(&csr(j), &rhs, damping, &bounded).unwrap();
        assert_eq!(a.x, b.x);
        assert_eq!(a.residual_norm, b.residual_norm);
        assert_eq!(a.refinement_steps, b.refinement_steps);
        let mut h = j.transpose() * j;
        for k in 0..h.ncols() {
            h[(k, k)] += damping * h[(k, k)].max(1.);
        }
        let expected = h.clone().cholesky().unwrap().solve(&rhs);
        assert!((&a.x - expected).norm() < 1e-11);
        assert!((&h * &a.x - &rhs).norm() < 1e-11);
    }
}

#[test]
fn planned_and_search_updates_match_dense_cholesky_for_all_orders() {
    let n = 24;
    let j = DMatrix::from_fn(48, n, |r, c| {
        let value = ((r + 3 * c) as f64 * 0.13).sin() * 0.2;
        if r == c { value + 5. } else { value }
    });
    for ordering in [
        SparseOrdering::Natural,
        SparseOrdering::MinimumDegree,
        SparseOrdering::BlockMinimumDegree { block_size: 3 },
    ] {
        check(
            &j,
            SparseLdlOptions {
                ordering,
                ..Default::default()
            },
            n * (n + 1) / 2,
        );
    }
}

#[test]
fn exact_update_plan_budget_keeps_the_same_destinations_and_search_fallback() {
    let n = 6;
    let j = DMatrix::from_fn(8, n, |r, c| {
        if r == c {
            4.
        } else {
            0.2 + ((r + 2 * c) as f64 * 0.23).cos() * 0.1
        }
    });
    let options = SparseLdlOptions {
        ordering: SparseOrdering::Natural,
        max_entries: 35,
        ..Default::default()
    };
    let s = SparseLdlSymbolic::analyze(&csr(&j), &options).unwrap();
    assert_eq!(s.update_entries(), 35);
    assert_eq!(s.assembly_entries(), 0);
    check(&j, options, 34);
}
