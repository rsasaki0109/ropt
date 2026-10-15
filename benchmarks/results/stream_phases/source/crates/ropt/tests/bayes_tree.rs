use ropt::{bayes_tree::*, incremental_qr::*, sparse::CsrMatrix, *};

fn csr(matrix: &DMatrix<f64>) -> CsrMatrix {
    CsrMatrix::from_triplets(
        matrix.nrows(),
        matrix.ncols(),
        (0..matrix.nrows())
            .flat_map(|r| {
                (0..matrix.ncols()).filter_map(move |c| {
                    let value = matrix[(r, c)];
                    (value != 0.).then_some((r, c, value))
                })
            })
            .collect(),
    )
    .unwrap()
}
fn check(tree: &GaussianBayesTree, j: &DMatrix<f64>, r: &DVector<f64>) {
    let x = tree.solve().unwrap();
    let reference = j
        .clone()
        .svd(true, true)
        .solve(&(-r), 1e-12 * j.amax())
        .unwrap();
    assert!((&x - reference).norm() < 2e-10);
    for point in [
        x,
        DVector::zeros(j.ncols()),
        DVector::from_element(j.ncols(), 0.37),
    ] {
        let expected = 0.5 * (j * &point + r).norm_squared();
        assert!(
            (tree.cost(&point).unwrap() - expected).abs() < 5e-12 * expected.max(f64::MIN_POSITIVE)
        );
    }
}

#[test]
fn forest_updates_reeliminate_only_touched_paths_and_keep_full_objective() {
    // Three independent branches, each with three leaves and a root. Root
    // columns follow leaves, so a new leaf observation touches only two cliques.
    for scale in [1e-80, 1., 1e80] {
        let mut j = DMatrix::zeros(12, 12);
        for k in 0..9 {
            j[(k, k)] = 2. * scale;
            j[(k, 9 + k / 3)] = -scale;
        }
        for k in 9..12 {
            j[(k, k)] = 3. * scale;
        }
        let mut r = DVector::from_fn(12, |k, _| (k as f64 + 1.).sin() * scale);
        let mut qr = IncrementalQr::new(12, Default::default()).unwrap();
        qr.append(&csr(&j), &r).unwrap();
        let mut tree = GaussianBayesTree::from_qr(qr).unwrap();
        for k in [0, 4, 8, 1, 5] {
            let mut added = DMatrix::zeros(2, 12);
            added[(0, k)] = scale;
            let residual = DVector::from_vec(vec![-0.3 * scale, 0.4 * scale]); // constant-only row
            let work = tree.append(&csr(&added), &residual).unwrap();
            assert_eq!(work.reeliminated_cliques, 2);
            assert_eq!(work.reused_cliques, 10);
            assert_eq!(work.input_rows, 4);
            let old = j.nrows();
            j = j.resize(old + 2, 12, 0.);
            j.rows_mut(old, 2).copy_from(&added);
            r = r.resize_vertically(old + 2, 0.);
            r.rows_mut(old, 2).copy_from(&residual);
            check(&tree, &j, &r);
        }
        // Cross-branch loop: both leaf-to-root paths, never the untouched branch.
        let mut added = DMatrix::zeros(1, 12);
        added[(0, 1)] = scale;
        added[(0, 7)] = -scale;
        let residual = DVector::from_element(1, 0.2 * scale);
        let work = tree.append(&csr(&added), &residual).unwrap();
        assert_eq!(work.reeliminated_cliques, 4);
        assert_eq!(work.reused_cliques, 8);
        let old = j.nrows();
        j = j.resize(old + 1, 12, 0.);
        j.rows_mut(old, 1).copy_from(&added);
        r = r.resize_vertically(old + 1, 0.);
        r.rows_mut(old, 1).copy_from(&residual);
        check(&tree, &j, &r);
        for k in 0..tree.dimension() {
            if let Some(parent) = tree.parent(k).unwrap() {
                for &separator in tree.separator(k).unwrap().iter().skip(1) {
                    assert!(tree.separator(parent).unwrap().contains(&separator));
                }
            }
        }
    }
}

#[test]
fn column_extension_missing_rank_and_constant_rows_keep_original_coordinates() {
    let mut tree = GaussianBayesTree::new(1, Default::default()).unwrap();
    tree.append(
        &CsrMatrix::from_triplets(1, 1, vec![(0, 0, 2.)]).unwrap(),
        &DVector::from_element(1, -4.),
    )
    .unwrap();
    tree.extend_columns(2).unwrap();
    assert!(matches!(
        tree.solve(),
        Err(Error::RankDeficient {
            rank: 1,
            parameters: 2
        })
    ));
    let work = tree
        .append(
            &CsrMatrix::from_triplets(1, 2, vec![(0, 0, -1.), (0, 1, 1.)]).unwrap(),
            &DVector::from_element(1, -1.),
        )
        .unwrap();
    assert_eq!(work.reeliminated_cliques, 2);
    assert!((tree.solve().unwrap() - DVector::from_vec(vec![2., 3.])).norm() < 1e-12);
    let before = tree.solve().unwrap();
    let work = tree
        .append(
            &CsrMatrix::from_triplets(2, 2, vec![]).unwrap(),
            &DVector::from_vec(vec![3., 4.]),
        )
        .unwrap();
    assert_eq!(work.reeliminated_cliques, 0);
    assert_eq!(work.reused_cliques, 2);
    assert_eq!(tree.solve().unwrap(), before);
    assert!((tree.cost(&before).unwrap() - 12.5).abs() < 1e-12);
    assert_eq!(tree.extend_columns(1), Err(Error::InvalidDimensions));
    assert_eq!(tree.parent(2), Err(Error::InvalidDimensions));
    assert_eq!(tree.clique_for_column(2), Err(Error::InvalidDimensions));
    assert_eq!(tree.cost(&DVector::zeros(3)), Err(Error::InvalidDimensions));
}

#[test]
fn failed_fill_nonfinite_and_shape_updates_leave_all_tree_state_unchanged() {
    let options = IncrementalQrOptions {
        max_entries: 2,
        ..Default::default()
    };
    let mut tree = GaussianBayesTree::new(2, options).unwrap();
    tree.append(
        &csr(&DMatrix::identity(2, 2)),
        &DVector::from_vec(vec![-1., -2.]),
    )
    .unwrap();
    let before = tree.solve().unwrap();
    let cost = tree.cost(&DVector::zeros(2)).unwrap();
    let work = tree.last_update();
    let coupling = CsrMatrix::from_triplets(1, 2, vec![(0, 0, 1.), (0, 1, 1.)]).unwrap();
    assert_eq!(
        tree.append(&coupling, &DVector::zeros(1)),
        Err(Error::SparseFillLimit { limit: 2 })
    );
    let constant = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 1.)]).unwrap();
    assert_eq!(
        tree.append(&constant, &DVector::from_vec(vec![-1., f64::MAX])),
        Err(Error::NonFinite)
    );
    assert_eq!(
        tree.append(&coupling, &DVector::from_element(1, f64::INFINITY)),
        Err(Error::NonFinite)
    );
    assert_eq!(
        tree.append(&coupling, &DVector::zeros(2)),
        Err(Error::InvalidDimensions)
    );
    assert_eq!(tree.solve().unwrap(), before);
    assert_eq!(tree.cost(&DVector::zeros(2)).unwrap(), cost);
    assert_eq!(tree.last_update(), work);
    assert_eq!(tree.stored_entries(), 2);
    assert_eq!(tree.parent(0).unwrap(), None);
    assert_eq!(
        tree.extend_columns(3),
        Err(Error::SparseFillLimit { limit: 2 })
    );
    assert_eq!(tree.dimension(), 2);
}
