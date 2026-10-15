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
fn constant_rows_and_changing_widths_match_svd_and_preserve_the_entire_objective() {
    let n = 7;
    let j = DMatrix::from_fn(29, n, |r, c| {
        if r < 3 || r >= 10 && r % 4 == 0 {
            0.
        } else if r < 10 {
            if c == r - 3 {
                if c % 2 == 0 { -2. } else { 3. }
            } else {
                0.
            }
        } else if c <= (r * 3) % 8 && (r + c) % 3 != 0 {
            ((r + 2 * c) as f64 * 0.37).cos()
        } else {
            0.
        }
    });
    let r = DVector::from_fn(j.nrows(), |k, _| (k as f64 * 0.41).sin() + 0.7);
    let mut qr = IncrementalQr::new(n, Default::default()).unwrap();
    qr.append(&csr(&j), &r).unwrap();
    let x = qr.solve().unwrap();
    let reference = j.clone().svd(true, true).solve(&(-&r), 1e-12).unwrap();
    assert!((&x - reference).norm() < 1e-12);
    for v in [x.clone(), DVector::zeros(n), DVector::from_element(n, 0.25)] {
        let expected = 0.5 * (&j * &v + &r).norm_squared();
        assert!((qr.cost(&v).unwrap() - expected).abs() < 1e-12 * expected.max(1.));
    }
    // Splitting a batch changes buffer lifetimes, but must keep the same row order.
    let mut split = IncrementalQr::new(n, Default::default()).unwrap();
    for (start, count) in [(0, 11), (11, 18)] {
        split
            .append(
                &csr(&j.rows(start, count).into_owned()),
                &r.rows(start, count).into_owned(),
            )
            .unwrap();
    }
    assert_eq!(split.solve().unwrap(), x);
    assert_eq!(split.stored_entries(), qr.stored_entries());
    let cost = split.cost(&DVector::zeros(n)).unwrap();
    assert_eq!(cost, qr.cost(&DVector::zeros(n)).unwrap());
    split
        .append(
            &CsrMatrix::from_triplets(0, n, vec![]).unwrap(),
            &DVector::zeros(0),
        )
        .unwrap();
    assert_eq!(split.solve().unwrap(), x);
    assert_eq!(split.cost(&DVector::zeros(n)).unwrap(), cost);
}

#[test]
fn transient_fill_is_checked_before_later_cancellation_even_when_final_fill_fits() {
    let initial = DMatrix::from_row_slice(3, 3, &[1., 0., 1., 0., 1., 2., 0., 0., 1.]);
    let residual = DVector::from_vec(vec![-1., -2., -3.]);
    let added = DMatrix::from_row_slice(1, 3, &[1., 1., -1.]);
    let added_residual = DVector::from_element(1, -0.25);
    let make = |limit| {
        let mut qr = IncrementalQr::new(
            3,
            IncrementalQrOptions {
                max_entries: limit,
                ..Default::default()
            },
        )
        .unwrap();
        qr.append(&csr(&initial), &residual).unwrap();
        qr
    };
    let mut bounded = make(5);
    let before = bounded.solve().unwrap();
    let before_cost = bounded.cost(&DVector::zeros(3)).unwrap();
    assert_eq!(bounded.stored_entries(), 5);
    assert_eq!(
        bounded.append(&csr(&added), &added_residual).unwrap_err(),
        Error::SparseFillLimit { limit: 5 }
    );
    assert_eq!(bounded.solve().unwrap(), before);
    assert_eq!(bounded.cost(&DVector::zeros(3)).unwrap(), before_cost);
    assert_eq!(bounded.stored_entries(), 5);

    // With one transient slot, column 1 is inserted before column 2 cancels.
    let mut permitted = make(6);
    permitted.append(&csr(&added), &added_residual).unwrap();
    assert_eq!(permitted.stored_entries(), 5);
    let j = DMatrix::from_row_slice(4, 3, &[1., 0., 1., 0., 1., 2., 0., 0., 1., 1., 1., -1.]);
    let r = DVector::from_vec(vec![-1., -2., -3., -0.25]);
    let reference = j.clone().svd(true, true).solve(&(-&r), 1e-12).unwrap();
    let x = permitted.solve().unwrap();
    assert!((&x - reference).norm() < 1e-12);
    assert!((permitted.cost(&x).unwrap() - 0.5 * (&j * x + r).norm_squared()).abs() < 1e-12);

    bounded
        .append(
            &CsrMatrix::from_triplets(1, 3, vec![]).unwrap(),
            &DVector::from_element(1, 0.5),
        )
        .unwrap();
    assert_eq!(bounded.solve().unwrap(), before);
    assert_eq!(
        bounded.cost(&DVector::zeros(3)).unwrap(),
        before_cost + 0.125
    );
}

#[test]
fn zero_delta_cost_checks_overflow_signed_zero_and_nonzero_underflowed_norms() {
    let identity = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 1.)]).unwrap();
    for residual in [-2., 1e-200] {
        let mut qr = IncrementalQr::new(1, Default::default()).unwrap();
        qr.append(&identity, &DVector::from_element(1, residual))
            .unwrap();
        for zero in [0., -0.] {
            assert_eq!(
                qr.cost(&DVector::from_element(1, zero)).unwrap(),
                0.5 * residual * residual
            );
        }
    }
    let mut overflow = IncrementalQr::new(1, Default::default()).unwrap();
    overflow
        .append(&identity, &DVector::from_element(1, f64::MAX))
        .unwrap();
    assert_eq!(overflow.cost(&DVector::zeros(1)), Err(Error::NonFinite));
    assert_eq!(
        overflow.cost(&DVector::from_element(1, f64::NAN)),
        Err(Error::NonFinite)
    );

    let mut large = IncrementalQr::new(1, Default::default()).unwrap();
    large
        .append(
            &CsrMatrix::from_triplets(1, 1, vec![(0, 0, 1e200)]).unwrap(),
            &DVector::zeros(1),
        )
        .unwrap();
    let tiny = DVector::from_element(1, 1e-200);
    assert_eq!(tiny.norm(), 0.);
    assert!((large.cost(&tiny).unwrap() - 0.5).abs() < 1e-14);
}
