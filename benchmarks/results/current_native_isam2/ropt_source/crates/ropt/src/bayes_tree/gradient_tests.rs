use crate::{
    DMatrix, DVector, Error,
    bayes_tree::{EditableBayesTree, GaussianBayesTree, GaussianFactor},
    incremental_qr::IncrementalQr,
    sparse::CsrMatrix,
};

fn csr(j: &DMatrix<f64>) -> CsrMatrix {
    CsrMatrix::from_triplets(
        j.nrows(),
        j.ncols(),
        (0..j.nrows())
            .flat_map(|r| (0..j.ncols()).map(move |c| (r, c, j[(r, c)])))
            .collect(),
    )
    .unwrap()
}

#[test]
fn frozen_gradient_solve_uses_original_coordinates_at_extreme_scales() {
    // Manufacture a step independently of either QR implementation. A nonzero
    // residual ensures this is not accidentally a solve() of the cached RHS.
    let expected = DVector::from_vec(vec![0.7, -0.4, 1.3]);
    for scale in [1e-80, 1., 1e80] {
        let j = DMatrix::from_row_slice(
            5,
            3,
            &[2., 1., 0., 0., 3., -1., 1., 0., 4., -1., 2., 1., 0., 1., 2.],
        ) * scale;
        let r = DVector::from_element(5, 0.3 * scale);
        let gradient = -(j.transpose() * (&j * &expected));
        for order in [vec![0, 1, 2], vec![2, 0, 1], vec![1, 2, 0]] {
            let mut qr = IncrementalQr::new(3, Default::default()).unwrap();
            qr.set_column_order(order.clone()).unwrap();
            qr.append(&csr(&j), &r).unwrap();
            let tree = GaussianBayesTree::from_qr(qr).unwrap();
            let mut editable = EditableBayesTree::new_ordered(order, Default::default()).unwrap();
            editable
                .update(
                    &[],
                    vec![(0, GaussianFactor::new(csr(&j), r.clone()).unwrap())],
                )
                .unwrap();
            for actual in [
                tree.solve_gradient(&gradient).unwrap(),
                editable.solve_gradient(&gradient).unwrap(),
            ] {
                assert!((&actual - &expected).norm() < 2e-14);
            }
            // Replacing a factor must invalidate the old numeric triangular factor.
            let replacement = &j * 2.;
            editable
                .update(
                    &[],
                    vec![(
                        0,
                        GaussianFactor::new(csr(&replacement), r.clone()).unwrap(),
                    )],
                )
                .unwrap();
            assert!(
                (editable.solve_gradient(&gradient).unwrap() - &expected * 0.25).norm() < 2e-14
            );
        }
    }
}

#[test]
fn frozen_gradient_errors_do_not_change_the_tree() {
    let mut tree = EditableBayesTree::new(2, Default::default()).unwrap();
    let j = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 1.), (1, 1, 2.)]).unwrap();
    tree.update(
        &[],
        vec![(0, GaussianFactor::new(j, DVector::zeros(2)).unwrap())],
    )
    .unwrap();
    assert!(matches!(
        tree.solve_gradient(&DVector::zeros(1)),
        Err(Error::InvalidDimensions)
    ));
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            tree.solve_gradient(&DVector::from_element(2, value)),
            Err(Error::NonFinite)
        ));
    }
    assert_eq!(
        tree.solve_gradient(&DVector::from_vec(vec![1., 4.]))
            .unwrap(),
        DVector::from_vec(vec![-1., -1.])
    );
    tree.update(&[0], vec![]).unwrap();
    assert!(matches!(
        tree.solve_gradient(&DVector::zeros(2)),
        Err(Error::RankDeficient {
            rank: 0,
            parameters: 2
        })
    ));
}

#[test]
fn frozen_gradient_rejects_overflow_and_relative_rank_loss() {
    for (diagonal, overflow) in [(1e-200, true), (1e-20, false)] {
        let mut qr = IncrementalQr::new(2, Default::default()).unwrap();
        // Equal small diagonals are observable by a relative rank test, but
        // solving twice for a unit gradient overflows. Unequal diagonals are
        // rank deficient under the default relative cutoff.
        let j = CsrMatrix::from_triplets(
            2,
            2,
            vec![
                (0, 0, diagonal),
                (1, 1, if overflow { diagonal } else { 1. }),
            ],
        )
        .unwrap();
        qr.append(&j, &DVector::zeros(2)).unwrap();
        let tree = GaussianBayesTree::from_qr(qr).unwrap();
        let result = tree.solve_gradient(&DVector::from_element(2, 1.));
        if overflow {
            assert!(matches!(result, Err(Error::NonFinite)));
        } else {
            assert!(matches!(
                result,
                Err(Error::RankDeficient {
                    rank: 1,
                    parameters: 2
                })
            ));
        }
    }
    let empty = GaussianBayesTree::new(0, Default::default()).unwrap();
    assert_eq!(empty.solve_gradient(&DVector::zeros(0)).unwrap().len(), 0);
}
