use ropt::{
    DMatrix, DVector, Error,
    bayes_tree::{EditableBayesTree, GaussianBayesTree, GaussianFactor},
    incremental_qr::IncrementalQr,
    sparse::CsrMatrix,
};
use std::collections::BTreeMap;

fn factor(n: usize, rows: usize, entries: Vec<(usize, usize, f64)>) -> GaussianFactor {
    GaussianFactor::new(
        CsrMatrix::from_triplets(rows, n, entries).unwrap(),
        DVector::from_fn(rows, |i, _| 0.3 * (i + 1) as f64),
    )
    .unwrap()
}

fn check(tree: &EditableBayesTree, factors: &BTreeMap<u64, GaussianFactor>) {
    let n = tree.dimension();
    let rows = factors.values().map(|f| f.residual().len()).sum();
    let mut j = DMatrix::zeros(rows, n);
    let mut residual = DVector::zeros(rows);
    let mut offset = 0;
    let mut entries = Vec::new();
    for f in factors.values() {
        for (r, c, value) in f.jacobian().entries() {
            j[(offset + r, c)] = value;
            entries.push((offset + r, c, value));
        }
        residual
            .rows_mut(offset, f.residual().len())
            .copy_from(f.residual());
        offset += f.residual().len();
    }
    let gradient = DVector::from_fn(n, |k, _| 0.7 - k as f64 * 0.4);
    // Independent dense reference is appropriate for this deliberately small,
    // well-conditioned test; production uses triangular solves, not J^T J.
    let expected = (j.transpose() * &j).lu().solve(&(-&gradient)).unwrap();
    let mut qr = IncrementalQr::new(n, Default::default()).unwrap();
    qr.append(
        &CsrMatrix::from_triplets(rows, n, entries).unwrap(),
        &residual,
    )
    .unwrap();
    let fresh = GaussianBayesTree::from_qr(qr).unwrap();
    let old_solution = tree.solve().unwrap();
    let old_cost = tree.cost(&DVector::zeros(n)).unwrap();
    for actual in [
        tree.solve_gradient(&gradient).unwrap(),
        fresh.solve_gradient(&gradient).unwrap(),
    ] {
        assert!((&actual - &expected).norm() < 1e-12);
    }
    assert_eq!(tree.solve().unwrap(), old_solution);
    assert_eq!(tree.cost(&DVector::zeros(n)).unwrap(), old_cost);
    assert!(tree.solve_gradient(&DVector::zeros(n + 1)).is_err());
    assert!((tree.solve_gradient(&gradient).unwrap() - &expected).norm() < 1e-12);
}

#[test]
fn gradient_solve_tracks_added_removed_replaced_and_new_variable_factors() {
    let mut tree = EditableBayesTree::new_ordered(vec![2, 0, 1], Default::default()).unwrap();
    let mut factors = BTreeMap::from([
        (0, factor(3, 3, vec![(0, 0, 2.), (1, 1, 3.), (2, 2, 4.)])),
        (
            1,
            factor(3, 2, vec![(0, 0, 1.), (0, 2, -1.), (1, 1, 2.), (1, 2, 1.)]),
        ),
    ]);
    tree.update(
        &[],
        factors.iter().map(|(&id, f)| (id, f.clone())).collect(),
    )
    .unwrap();
    check(&tree, &factors);

    factors.insert(2, factor(3, 1, vec![(0, 0, -2.), (0, 1, 1.)]));
    tree.update(&[], vec![(2, factors[&2].clone())]).unwrap();
    check(&tree, &factors);

    factors.remove(&1);
    tree.update(&[1], vec![]).unwrap();
    check(&tree, &factors);

    factors.insert(0, factor(3, 3, vec![(0, 0, 5.), (1, 1, 2.), (2, 2, 1.)]));
    tree.update(&[], vec![(0, factors[&0].clone())]).unwrap();
    check(&tree, &factors);
    assert!(tree.update(&[999], vec![]).is_err());
    check(&tree, &factors);

    tree.extend_columns(4).unwrap();
    assert!(matches!(
        tree.solve_gradient(&DVector::zeros(4)),
        Err(Error::RankDeficient {
            rank: 3,
            parameters: 4
        })
    ));
    factors.insert(3, factor(4, 1, vec![(0, 0, -1.), (0, 3, 2.)]));
    tree.update(&[], vec![(3, factors[&3].clone())]).unwrap();
    check(&tree, &factors);
}
