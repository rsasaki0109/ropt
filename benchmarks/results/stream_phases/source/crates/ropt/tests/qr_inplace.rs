use ropt::{DMatrix, DVector, Error, incremental_qr::*, sparse::CsrMatrix};

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

#[test]
fn existing_prefix_cancellation_interior_fill_and_trailing_growth_match_svd() {
    let n = 8;
    let mut j = DMatrix::zeros(32, n);
    j.view_mut((0, 0), (n, n))
        .copy_from(&DMatrix::identity(n, n));
    j[(0, 1)] = 1.;
    j[(0, 3)] = 1.;
    j[(0, 5)] = 1.;
    // Cancel an existing prefix before the first new interior column.
    j.row_mut(8)
        .copy_from(&nalgebra::RowDVector::from_row_slice(&[
            1., -1., 1., 1., -1., -1., 0., 0.,
        ]));
    // Existing prefix followed by new trailing fill, plus upper-only columns.
    j.row_mut(9)
        .copy_from(&nalgebra::RowDVector::from_row_slice(&[
            1., 0., 0., 0., 0., 0., 0., 2.,
        ]));
    for r in 10..30 {
        for c in 0..n {
            if (r + c) % 3 != 0 {
                j[(r, c)] = ((r + 2 * c) as f64 * 0.17).sin();
            }
        }
    }
    let residual = DVector::from_fn(j.nrows(), |r, _| (r as f64 * 0.31).cos());
    let mut qr = IncrementalQr::new(n, Default::default()).unwrap();
    qr.append(&csr(&j), &residual).unwrap();
    let x = qr.solve().unwrap();
    let reference = j
        .clone()
        .svd(true, true)
        .solve(&(-&residual), 1e-12)
        .unwrap();
    assert!((&x - reference).norm() < 1e-12);
    let mut split = IncrementalQr::new(n, Default::default()).unwrap();
    for r in 0..j.nrows() {
        split
            .append(
                &csr(&j.rows(r, 1).into_owned()),
                &residual.rows(r, 1).into_owned(),
            )
            .unwrap();
    }
    assert_eq!(split.solve().unwrap(), x);
    assert_eq!(split.stored_entries(), qr.stored_entries());
    for value in [x, DVector::zeros(n), DVector::from_element(n, -0.2)] {
        let expected = 0.5 * (&j * &value + &residual).norm_squared();
        let cost = qr.cost(&value).unwrap();
        assert!((cost - expected).abs() < 1e-12 * expected.max(1.));
        assert_eq!(split.cost(&value).unwrap(), cost);
    }
}

#[test]
fn prefix_compaction_does_not_hide_transient_fill_and_failure_rolls_back() {
    let n = 6;
    let mut j = DMatrix::identity(n, n);
    j[(0, 1)] = 1.;
    j[(0, 3)] = 1.;
    j[(0, 5)] = 1.;
    let residual = DVector::from_vec(vec![-1., 2., -3., 4., -5., 6.]);
    let added = DMatrix::from_row_slice(1, n, &[1., -1., 1., 1., -1., -1.]);
    let mut qr = IncrementalQr::new(
        n,
        IncrementalQrOptions {
            max_entries: 9,
            ..Default::default()
        },
    )
    .unwrap();
    qr.append(&csr(&j), &residual).unwrap();
    let before = qr.solve().unwrap();
    let cost = qr.cost(&DVector::zeros(n)).unwrap();
    assert_eq!(qr.stored_entries(), 9);
    // Column 1 cancels in place (freeing a slot); column 2 reuses that slot.
    // Column 4 exceeds the budget before column 5 can cancel in the fallback.
    assert_eq!(
        qr.append(&csr(&added), &DVector::from_element(1, 0.25)),
        Err(Error::SparseFillLimit { limit: 9 })
    );
    assert_eq!(qr.solve().unwrap(), before);
    assert_eq!(qr.cost(&DVector::zeros(n)).unwrap(), cost);
    assert_eq!(qr.stored_entries(), 9);
    // A subsequent append must still work after the failed candidate.
    qr.append(&csr(&DMatrix::zeros(1, n)), &DVector::from_element(1, 0.5))
        .unwrap();
    assert_eq!(qr.solve().unwrap(), before);
    assert_eq!(qr.cost(&DVector::zeros(n)).unwrap(), cost + 0.125);
}

#[test]
fn coefficient_and_rhs_overflow_after_in_place_writes_restore_the_factor() {
    let initial = DMatrix::from_row_slice(2, 2, &[1., 1.7e308, 0., 1.]);
    let mut qr = IncrementalQr::new(2, Default::default()).unwrap();
    qr.append(&csr(&initial), &DVector::zeros(2)).unwrap();
    let before = qr.solve().unwrap();
    let added = DMatrix::from_row_slice(1, 2, &[1., 1.7e308]);
    assert_eq!(
        qr.append(&csr(&added), &DVector::zeros(1)),
        Err(Error::NonFinite)
    );
    assert_eq!(qr.solve().unwrap(), before);
    assert_eq!(qr.stored_entries(), 3);
    assert_eq!(qr.cost(&DVector::zeros(2)).unwrap(), 0.);

    let mut qr = IncrementalQr::new(2, Default::default()).unwrap();
    qr.append(
        &csr(&DMatrix::identity(2, 2)),
        &DVector::from_vec(vec![f64::MAX, 0.]),
    )
    .unwrap();
    let before = qr.solve().unwrap();
    assert_eq!(
        qr.append(
            &csr(&DMatrix::from_row_slice(1, 2, &[1., 0.])),
            &DVector::from_element(1, f64::MAX)
        ),
        Err(Error::NonFinite)
    );
    assert_eq!(qr.solve().unwrap(), before);
    assert_eq!(qr.stored_entries(), 2);
}
