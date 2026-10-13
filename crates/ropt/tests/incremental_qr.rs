use ropt::{DMatrix, DVector, Error, incremental_qr::*, sparse::CsrMatrix};
fn csr(j: &DMatrix<f64>) -> CsrMatrix {
    CsrMatrix::from_triplets(
        j.nrows(),
        j.ncols(),
        (0..j.nrows())
            .flat_map(|r| {
                (0..j.ncols()).filter_map(move |c| {
                    if j[(r, c)] == 0. {
                        None
                    } else {
                        Some((r, c, j[(r, c)]))
                    }
                })
            })
            .collect(),
    )
    .unwrap()
}
#[test]
fn row_updates_match_independent_svd_solution_and_full_objective() {
    let j = DMatrix::from_fn(15, 5, |r, c| {
        if r == c {
            3.
        } else if (r + c) % 3 == 0 {
            ((r + c + 1) as f64).sin()
        } else {
            0.
        }
    });
    let r = DVector::from_fn(15, |i, _| ((i + 1) as f64).cos());
    let mut qr = IncrementalQr::new(5, Default::default()).unwrap();
    for k in 0..15 {
        qr.append(&csr(&j.rows(k, 1).into_owned()), &r.rows(k, 1).into_owned())
            .unwrap();
    }
    let x = qr.solve().unwrap();
    let exact = j.clone().svd(true, true).solve(&(-&r), 1e-12).unwrap();
    assert!((&x - exact).norm() < 1e-12);
    for v in [x, DVector::zeros(5), DVector::from_element(5, 2.)] {
        assert!((qr.cost(&v).unwrap() - 0.5 * (&j * &v + &r).norm_squared()).abs() < 1e-10);
    }
    assert!(qr.stored_entries() <= 15);
}
#[test]
fn appended_columns_preserve_old_information_and_require_observability() {
    let mut qr = IncrementalQr::new(1, Default::default()).unwrap();
    qr.append(
        &CsrMatrix::from_triplets(1, 1, vec![(0, 0, 2.)]).unwrap(),
        &DVector::from_element(1, -4.),
    )
    .unwrap();
    assert!((qr.solve().unwrap()[0] - 2.).abs() < 1e-12);
    qr.extend_columns(2).unwrap();
    assert!(matches!(
        qr.solve(),
        Err(Error::RankDeficient {
            rank: 1,
            parameters: 2
        })
    ));
    qr.append(
        &CsrMatrix::from_triplets(1, 2, vec![(0, 0, -1.), (0, 1, 1.)]).unwrap(),
        &DVector::from_element(1, -1.),
    )
    .unwrap();
    assert!((qr.solve().unwrap() - DVector::from_vec(vec![2., 3.])).norm() < 1e-12);
    assert!(qr.extend_columns(1).is_err());
}
#[test]
fn failed_row_insertion_is_atomic_and_limits_and_nonfinite_values_are_checked() {
    let mut qr = IncrementalQr::new(
        2,
        IncrementalQrOptions {
            max_entries: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let identity = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 1.), (1, 1, 1.)]).unwrap();
    qr.append(&identity, &DVector::from_vec(vec![-1., -2.]))
        .unwrap();
    let before = qr.solve().unwrap();
    let coupled = CsrMatrix::from_triplets(1, 2, vec![(0, 0, 1.), (0, 1, 1.)]).unwrap();
    assert_eq!(
        qr.append(&coupled, &DVector::zeros(1)).unwrap_err(),
        Error::SparseFillLimit { limit: 2 }
    );
    assert_eq!(qr.solve().unwrap(), before);
    assert_eq!(
        qr.append(&identity, &DVector::from_element(2, f64::INFINITY))
            .unwrap_err(),
        Error::NonFinite
    );
    assert_eq!(qr.solve().unwrap(), before);
    assert_eq!(qr.stored_entries(), 2);
    assert!(
        IncrementalQr::new(
            2,
            IncrementalQrOptions {
                max_entries: 1,
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn sparse_merge_handles_disjoint_columns_cancellations_and_scaled_batches() {
    for n in [1, 3, 7, 16] {
        let j = DMatrix::from_fn(3 * n, n, |r, c| {
            let scale = if c % 2 == 0 { 100. } else { 0.1 };
            if r == c {
                2. * scale
            } else if r >= n && (r + 2 * c) % 5 == 0 {
                ((r + c + 1) as f64).cos() * scale
            } else {
                0.
            }
        });
        let r = DVector::from_fn(3 * n, |k, _| ((k + 1) as f64).sin());
        let mut qr = IncrementalQr::new(n, Default::default()).unwrap();
        // Reverse chunks so initially missing diagonals and disjoint fill merge.
        for block in (0..3).rev() {
            qr.append(
                &csr(&j.rows(block * n, n).into_owned()),
                &r.rows(block * n, n).into_owned(),
            )
            .unwrap();
        }
        let x = qr.solve().unwrap();
        let reference = j.clone().svd(true, true).solve(&(-&r), 1e-12).unwrap();
        assert!((&x - &reference).norm() < 1e-10);
        for candidate in [x, DVector::zeros(n), DVector::from_element(n, 0.1)] {
            let exact = 0.5 * (&j * &candidate + &r).norm_squared();
            assert!((qr.cost(&candidate).unwrap() - exact).abs() < 1e-10 * exact.max(1.));
        }
    }
    let j = DMatrix::from_row_slice(3, 2, &[1., 1., 1., 1., 0., 2.]);
    let mut qr = IncrementalQr::new(2, Default::default()).unwrap();
    qr.append(&csr(&j), &DVector::from_vec(vec![-1., -1., -2.]))
        .unwrap();
    assert_eq!(qr.stored_entries(), 3);
    assert!((qr.solve().unwrap() - DVector::from_vec(vec![0., 1.])).norm() < 1e-12);
}

#[test]
fn overflow_after_a_successful_rotation_restores_the_original_factor() {
    let j = DMatrix::identity(2, 2);
    let mut qr = IncrementalQr::new(2, Default::default()).unwrap();
    qr.append(&csr(&j), &DVector::from_vec(vec![-1., -2.]))
        .unwrap();
    let before = qr.solve().unwrap();
    let cost = qr.cost(&DVector::zeros(2)).unwrap();
    // First row modifies R/RHS; the second is a constant residual whose cost overflows.
    let huge = DMatrix::from_row_slice(2, 2, &[1., 0., 0., 0.]);
    assert_eq!(
        qr.append(&csr(&huge), &DVector::from_vec(vec![-3., f64::MAX]))
            .unwrap_err(),
        Error::NonFinite
    );
    assert_eq!(qr.solve().unwrap(), before);
    assert_eq!(qr.cost(&DVector::zeros(2)).unwrap(), cost);
    assert_eq!(qr.stored_entries(), 2);
}
