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

fn check_objective(j: &DMatrix<f64>, r: &DVector<f64>, qr: &IncrementalQr) {
    let x = qr.solve().unwrap();
    let reference = j.clone().svd(true, true).solve(&(-r), 1e-12).unwrap();
    assert!((&x - reference).norm() < 1e-12);
    for delta in [
        x,
        DVector::zeros(j.ncols()),
        DVector::from_element(j.ncols(), -0.2),
    ] {
        let expected = 0.5 * (j * &delta + r).norm_squared();
        assert!((qr.cost(&delta).unwrap() - expected).abs() < 1e-12 * expected.max(1.));
    }
}

#[test]
fn aligned_dense_suffixes_preserve_svd_objective_and_batch_boundaries() {
    let n = 12;
    let j = DMatrix::from_fn(47, n, |r, c| {
        if r < n {
            if c < r {
                0.
            } else if c == r {
                2.
            } else {
                0.3 + ((r + 3 * c) as f64 * 0.17).sin()
            }
        } else {
            ((r + 2 * c) as f64 * 0.23).cos()
        }
    });
    let residual = DVector::from_fn(j.nrows(), |r, _| (r as f64 * 0.31).sin());
    let mut qr = IncrementalQr::new(n, Default::default()).unwrap();
    qr.append(&csr(&j), &residual).unwrap();
    check_objective(&j, &residual, &qr);
    let mut split = IncrementalQr::new(n, Default::default()).unwrap();
    for r in 0..j.nrows() {
        split
            .append(
                &csr(&j.rows(r, 1).into_owned()),
                &residual.rows(r, 1).into_owned(),
            )
            .unwrap();
    }
    assert_eq!(split.solve().unwrap(), qr.solve().unwrap());
    assert_eq!(split.stored_entries(), qr.stored_entries());
    assert_eq!(
        split.cost(&DVector::zeros(n)).unwrap(),
        qr.cost(&DVector::zeros(n)).unwrap()
    );
}

#[test]
fn cancellations_then_equal_width_different_columns_use_the_live_suffix() {
    let j = DMatrix::from_row_slice(
        7,
        5,
        &[
            1., 2., 3., 4., 5., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.7, 0., 0., 0., 1., 0., 0.,
            0., 0., 0., 1., 1., 2., -3., -4., 5., 1., -2., 3., 4., -5.,
        ],
    );
    let r = DVector::from_vec(vec![1., -2., 3., -4., 5., -0.25, 0.5]);
    let mut qr = IncrementalQr::new(5, Default::default()).unwrap();
    qr.append(&csr(&j.rows(0, 5).into_owned()), &r.rows(0, 5).into_owned())
        .unwrap();
    // The first aligned rotation removes alternating upper/lower columns.
    // Its next live row and the old upper row have equal widths but different
    // column indices, requiring the ordinary sorted merge after cursor advance.
    qr.append(&csr(&j.rows(5, 1).into_owned()), &r.rows(5, 1).into_owned())
        .unwrap();
    check_objective(&j.rows(0, 6).into_owned(), &r.rows(0, 6).into_owned(), &qr);
    qr.append(&csr(&j.rows(6, 1).into_owned()), &r.rows(6, 1).into_owned())
        .unwrap();
    check_objective(&j, &r, &qr);
    let mut batch = IncrementalQr::new(5, Default::default()).unwrap();
    batch.append(&csr(&j), &r).unwrap();
    assert_eq!(batch.solve().unwrap(), qr.solve().unwrap());
    assert_eq!(batch.stored_entries(), qr.stored_entries());
    assert_eq!(
        batch.cost(&DVector::zeros(5)).unwrap(),
        qr.cost(&DVector::zeros(5)).unwrap()
    );
}

#[test]
fn overflow_after_cursor_advance_rolls_back_coefficients_and_rhs() {
    let j = DMatrix::from_row_slice(3, 3, &[1., 1., 1., 0., 1., 1.7e308, 0., 0., 1.]);
    let mut qr = IncrementalQr::new(3, Default::default()).unwrap();
    qr.append(&csr(&j), &DVector::zeros(3)).unwrap();
    let x = qr.solve().unwrap();
    let entries = qr.stored_entries();
    assert_eq!(
        qr.append(
            &csr(&DMatrix::from_row_slice(1, 3, &[1., -1., 1.7e308])),
            &DVector::zeros(1)
        ),
        Err(Error::NonFinite)
    );
    assert_eq!(qr.solve().unwrap(), x);
    assert_eq!(qr.stored_entries(), entries);
    assert_eq!(qr.cost(&DVector::zeros(3)).unwrap(), 0.);
    qr.append(&csr(&DMatrix::zeros(1, 3)), &DVector::from_element(1, 0.5))
        .unwrap();
    assert_eq!(qr.cost(&DVector::zeros(3)).unwrap(), 0.125);

    let j = DMatrix::from_row_slice(2, 2, &[1., 1., 0., 1.]);
    let mut qr = IncrementalQr::new(2, Default::default()).unwrap();
    qr.append(&csr(&j), &DVector::from_vec(vec![0., -f64::MAX]))
        .unwrap();
    let x = qr.solve().unwrap();
    assert_eq!(
        qr.append(
            &csr(&DMatrix::from_row_slice(1, 2, &[1., -1.])),
            &DVector::from_element(1, -f64::MAX)
        ),
        Err(Error::NonFinite)
    );
    assert_eq!(qr.solve().unwrap(), x);
    assert_eq!(qr.stored_entries(), 3);
}
