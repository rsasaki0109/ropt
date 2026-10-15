use ropt::{DMatrix, DVector, Error, bayes_tree::*, sparse::CsrMatrix};
use std::collections::BTreeMap;

fn block(
    n: usize,
    rows: usize,
    entries: Vec<(usize, usize, f64)>,
    residual: Vec<f64>,
) -> GaussianFactor {
    GaussianFactor::new(
        CsrMatrix::from_triplets(rows, n, entries).unwrap(),
        DVector::from_vec(residual),
    )
    .unwrap()
}
fn prior(n: usize, column: usize, target: f64) -> GaussianFactor {
    block(n, 1, vec![(0, column, 1.)], vec![-target])
}
fn star_leaf(n: usize, leaf: usize, target: f64, offset: f64) -> GaussianFactor {
    block(
        n,
        2,
        vec![(0, leaf, 1.), (1, leaf, 1.), (1, n - 1, -1.)],
        vec![-target, -offset],
    )
}
fn dense(n: usize, factors: &BTreeMap<u64, GaussianFactor>) -> (DVector<f64>, DMatrix<f64>) {
    let rows = factors.values().map(|f| f.residual().len()).sum();
    let mut j = DMatrix::zeros(rows, n);
    let mut r = DVector::zeros(rows);
    let mut start = 0;
    for f in factors.values() {
        for (row, col, value) in f.jacobian().entries() {
            j[(start + row, col)] = value;
        }
        r.rows_mut(start, f.residual().len())
            .copy_from(f.residual());
        start += f.residual().len();
    }
    (r, j)
}
fn original_cost(x: &DVector<f64>, factors: &BTreeMap<u64, GaussianFactor>) -> f64 {
    factors
        .values()
        .map(|f| {
            let old = x.rows(0, f.jacobian().shape().1).into_owned();
            0.5 * (f.jacobian().multiply(&old).unwrap() + f.residual()).norm_squared()
        })
        .sum()
}
fn assert_model(tree: &EditableBayesTree, factors: &BTreeMap<u64, GaussianFactor>) {
    let n = tree.dimension();
    let (r, j) = dense(n, factors);
    let expected = j.clone().svd(true, true).solve(&(-&r), 1e-12).unwrap();
    let actual = tree.solve().unwrap();
    assert!((&actual - &expected).norm() < 1e-10 * (1. + expected.norm()));
    assert!((j.transpose() * (&j * &actual + r)).norm() < 1e-10);
    for x in [
        actual,
        DVector::zeros(n),
        DVector::from_iterator(n, (0..n).map(|k| k as f64 - 1.2)),
    ] {
        let original = original_cost(&x, factors);
        assert!(
            (tree.cost(&x).unwrap() - original).abs() < 2e-11 * (1. + original),
            "original={original}, tree={:?}",
            tree.cost(&x)
        );
    }
}

#[test]
fn replacing_one_star_leaf_keeps_all_other_branches_and_boundary_information() {
    let n = 33;
    let mut tree = EditableBayesTree::new(n, Default::default()).unwrap();
    let mut factors = BTreeMap::new();
    for leaf in 0..n - 1 {
        factors.insert(
            leaf as u64,
            star_leaf(n, leaf, leaf as f64 * 0.2, leaf as f64 * 0.1),
        );
    }
    factors.insert(100, prior(n, n - 1, 0.));
    tree.update(
        &[],
        factors.iter().map(|(&id, f)| (id, f.clone())).collect(),
    )
    .unwrap();
    assert_model(&tree, &factors);
    factors.insert(7, star_leaf(n, 7, 7. * 0.2 + 3., 7. * 0.1));
    let work = tree.update(&[], vec![(7, factors[&7].clone())]).unwrap();
    assert_eq!(work.reeliminated_cliques, 2);
    assert_eq!(work.reused_cliques, 31);
    assert_eq!(work.reused_boundary_messages, 31);
    assert_eq!(work.input_factor_rows, 3);
    assert_eq!(work.input_separator_rows, 32);
    let x = tree.solve().unwrap();
    let root = ((0..n - 1).map(|k| k as f64 * 0.1).sum::<f64>() + 3.) / (n as f64 + 1.);
    assert!((x[n - 1] - root).abs() < 1e-13);
    for leaf in 0..n - 1 {
        let target = leaf as f64 * 0.2 + if leaf == 7 { 3. } else { 0. };
        assert!((x[leaf] - 0.5 * (target + leaf as f64 * 0.1 + root)).abs() < 1e-13);
    }
    assert_model(&tree, &factors);
    // Removing the root prior changes one clique, retaining all informative child messages.
    factors.remove(&100);
    let work = tree.update(&[100], vec![]).unwrap();
    assert_eq!(work.reeliminated_cliques, 1);
    assert_eq!(work.reused_boundary_messages, 32);
    assert_model(&tree, &factors);
}

#[test]
fn changed_support_owner_constants_and_multi_column_separators_preserve_full_quadratic() {
    let n = 6;
    let mut tree = EditableBayesTree::new(n, Default::default()).unwrap();
    let mut factors = (0..n)
        .map(|k| (k as u64, prior(n, k, k as f64 * 0.3)))
        .collect::<BTreeMap<_, _>>();
    factors.insert(
        100,
        block(
            n,
            3,
            vec![
                (0, 0, 1.),
                (0, 2, 2.),
                (0, 5, 1.),
                (1, 0, 1.),
                (1, 2, -2.),
                (1, 5, 1.),
                (2, 2, 1.),
                (2, 5, -1.),
            ],
            vec![1., -2., 0.5],
        ),
    );
    factors.insert(
        200,
        block(
            n,
            2,
            vec![(0, 1, 1.), (0, 3, 2.), (1, 3, 1.), (1, 4, -1.)],
            vec![-0.3, 0.8],
        ),
    );
    tree.update(
        &[],
        factors.iter().map(|(&id, f)| (id, f.clone())).collect(),
    )
    .unwrap();
    assert_eq!(tree.separator(0).unwrap(), &[2, 5]);
    assert_model(&tree, &factors);
    // The same ID moves from frontal 0 to frontal 3 and joins a different subtree.
    factors.insert(
        100,
        block(
            n,
            2,
            vec![(0, 3, 1.), (0, 4, -3.), (1, 3, 2.)],
            vec![2., -1.],
        ),
    );
    tree.update(&[], vec![(100, factors[&100].clone())])
        .unwrap();
    assert_model(&tree, &factors);
    assert!(tree.separator(0).unwrap().is_empty());
    // Replace it by pure constant observations, and remove another block in the same transaction.
    factors.insert(100, block(n, 2, vec![], vec![3., -4.]));
    factors.remove(&200);
    tree.update(&[200], vec![(100, factors[&100].clone())])
        .unwrap();
    assert_model(&tree, &factors);
    assert!((tree.cost(&tree.solve().unwrap()).unwrap() - 12.5).abs() < 1e-12);
    factors.remove(&100);
    let work = tree.update(&[100], vec![]).unwrap();
    assert_eq!(work.reeliminated_cliques, 0);
    assert_model(&tree, &factors);
}

#[test]
fn permutation_extension_and_repeated_removal_use_original_coordinates() {
    let mut tree = EditableBayesTree::new_ordered(vec![2, 0, 1], Default::default()).unwrap();
    let mut factors = (0..3)
        .map(|k| (k as u64, prior(3, k, k as f64 + 1.)))
        .collect::<BTreeMap<_, _>>();
    tree.update(
        &[],
        factors.iter().map(|(&id, f)| (id, f.clone())).collect(),
    )
    .unwrap();
    assert_model(&tree, &factors);
    tree.extend_columns(4).unwrap();
    assert_eq!(tree.column_permutation(), &[2, 0, 1, 3]);
    factors.insert(
        10,
        block(
            4,
            2,
            vec![(0, 0, -1.), (0, 3, 1.), (1, 3, 1.)],
            vec![-2., -5.],
        ),
    );
    tree.update(&[], vec![(10, factors[&10].clone())]).unwrap();
    assert_model(&tree, &factors);
    for target in [-1., 2., 7.] {
        factors.insert(0, prior(4, 0, target));
        tree.update(&[], vec![(0, factors[&0].clone())]).unwrap();
        assert_model(&tree, &factors);
    }
    factors.remove(&10);
    tree.update(&[10], vec![]).unwrap();
    assert!(matches!(
        tree.solve(),
        Err(Error::RankDeficient { parameters: 4, .. })
    ));
    let x = DVector::from_vec(vec![1., 2., 3., 4.]);
    assert!((tree.cost(&x).unwrap() - original_cost(&x, &factors)).abs() < 1e-12);
    factors.insert(10, prior(4, 3, -2.));
    tree.update(&[], vec![(10, factors[&10].clone())]).unwrap();
    assert_model(&tree, &factors);
}

#[test]
fn invalid_ids_dimensions_overflow_and_budgets_are_transactional() {
    let mut tree = EditableBayesTree::new(1, Default::default()).unwrap();
    tree.update(&[], vec![(1, prior(1, 0, 1.))]).unwrap();
    let snapshot = tree.last_update();
    for (removed, replacements) in [
        (vec![99], vec![]),
        (vec![1, 1], vec![]),
        (vec![1], vec![(1, prior(1, 0, 2.))]),
        (vec![], vec![(2, prior(1, 0, 2.)), (2, prior(1, 0, 3.))]),
        (vec![], vec![(2, prior(2, 0, 2.))]),
    ] {
        assert!(tree.update(&removed, replacements).is_err());
        assert_eq!(tree.factor_count(), 1);
        assert_eq!(tree.last_update(), snapshot);
        assert_eq!(tree.solve().unwrap()[0], 1.);
    }
    let huge = block(1, 1, vec![(0, 0, 1.5e308)], vec![0.]);
    assert!(matches!(
        tree.update(&[], vec![(2, huge.clone()), (3, huge)]),
        Err(Error::NonFinite)
    ));
    assert_eq!(tree.factor_count(), 1);
    assert_eq!(tree.solve().unwrap()[0], 1.);
    assert!(
        GaussianFactor::new(
            CsrMatrix::from_triplets(1, 1, vec![]).unwrap(),
            DVector::from_element(1, f64::NAN)
        )
        .is_err()
    );
    assert!(
        GaussianFactor::new(
            CsrMatrix::from_triplets(0, 1, vec![]).unwrap(),
            DVector::zeros(0)
        )
        .is_err()
    );
    let mut bounded = EditableBayesTree::new(
        1,
        ropt::incremental_qr::IncrementalQrOptions {
            max_entries: 3,
            ..Default::default()
        },
    )
    .unwrap();
    bounded.update(&[], vec![(0, prior(1, 0, 1.))]).unwrap();
    assert!(matches!(
        bounded.update(&[], vec![(1, prior(1, 0, 2.))]),
        Err(Error::SparseFillLimit { limit: 3 })
    ));
    assert_eq!(bounded.factor_count(), 1);
    assert_eq!(bounded.solve().unwrap()[0], 1.);
    assert!(bounded.extend_columns(4).is_err());
    assert_eq!(bounded.dimension(), 1);
    assert!(
        EditableBayesTree::new(
            usize::MAX,
            ropt::incremental_qr::IncrementalQrOptions {
                max_entries: 3,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(EditableBayesTree::new_ordered(vec![0, 0], Default::default()).is_err());
}

#[test]
fn incomplete_models_and_zero_dimension_constants_have_explicit_rank_behavior() {
    let mut tree = EditableBayesTree::new(3, Default::default()).unwrap();
    let edge = |a, b| block(3, 1, vec![(0, a, -1.), (0, b, 1.)], vec![-1.]);
    tree.update(
        &[],
        vec![(0, prior(3, 0, 0.)), (1, edge(0, 1)), (2, edge(1, 2))],
    )
    .unwrap();
    tree.update(&[0], vec![]).unwrap();
    assert!(matches!(
        tree.solve(),
        Err(Error::RankDeficient {
            rank: 2,
            parameters: 3
        })
    ));
    assert!(tree.cost(&DVector::from_vec(vec![3., 4., 5.])).unwrap() < 1e-24);
    let mut constant = EditableBayesTree::new(0, Default::default()).unwrap();
    constant
        .update(&[], vec![(0, block(0, 2, vec![], vec![3., 4.]))])
        .unwrap();
    assert_eq!(constant.solve().unwrap().len(), 0);
    assert_eq!(constant.cost(&DVector::zeros(0)).unwrap(), 12.5);
    constant.update(&[0], vec![]).unwrap();
    assert_eq!(constant.cost(&DVector::zeros(0)).unwrap(), 0.);
}

#[test]
fn varied_sparse_support_changes_match_independent_dense_svd_and_original_cost() {
    let n = 8;
    let mut tree =
        EditableBayesTree::new_ordered(vec![7, 2, 5, 0, 4, 1, 6, 3], Default::default()).unwrap();
    let mut factors = (0..n)
        .map(|k| (k as u64, prior(n, k, 0.2 * k as f64)))
        .collect::<BTreeMap<_, _>>();
    tree.update(
        &[],
        factors.iter().map(|(&id, f)| (id, f.clone())).collect(),
    )
    .unwrap();
    for step in 0..48 {
        let id = 100 + (step % 5) as u64;
        if step % 7 == 6 && factors.contains_key(&id) {
            factors.remove(&id);
            tree.update(&[id], vec![]).unwrap();
        } else {
            let mut entries = Vec::new();
            for row in 0..3 {
                for col in 0..n {
                    if (row * 11 + col * 7 + step * 3) % 5 == 0 {
                        entries.push((row, col, ((row * 13 + col * 3 + step) % 7) as f64 - 3.));
                    }
                }
            }
            let f = block(n, 3, entries, vec![(step as f64 * 0.3).sin(), -0.4, 1.]);
            factors.insert(id, f.clone());
            tree.update(&[], vec![(id, f)]).unwrap();
        }
        assert_model(&tree, &factors);
    }
    assert_eq!(tree.update(&[], vec![]).unwrap().reused_cliques, n);
}
