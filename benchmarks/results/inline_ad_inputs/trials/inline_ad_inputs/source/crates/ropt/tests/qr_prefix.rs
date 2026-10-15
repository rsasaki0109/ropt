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

fn check(j: &DMatrix<f64>, r: &DVector<f64>, qr: &IncrementalQr) {
    let x = qr.solve().unwrap();
    let reference = j.clone().svd(true, true).solve(&(-r), 1e-12).unwrap();
    assert!((&x - reference).norm() < 1e-12);
    for delta in [
        x,
        DVector::zeros(j.ncols()),
        DVector::from_element(j.ncols(), 0.25),
    ] {
        let expected = 0.5 * (j * &delta + r).norm_squared();
        assert!((qr.cost(&delta).unwrap() - expected).abs() < 1e-12 * expected.max(1.));
    }
}

#[test]
fn unequal_suffixes_and_divergence_after_cancellation_preserve_the_complete_objective() {
    let n = 7;
    for (initial, added) in [
        (
            [1., 2., -3., 4., 5., -6., 7.],
            [1., 2., 3., -4., 0., 0., 0.],
        ),
        ([1., 2., -3., 0., 0., 0., 0.], [1., 2., 3., 4., 5., -6., 7.]),
        (
            [1., 2., -3., 0., 5., -6., 7.],
            [1., 2., 3., 4., 0., -6., 7.],
        ),
        (
            [1., 2., -3., 4., 0., -6., 7.],
            [1., 2., 3., 0., 5., -6., 7.],
        ),
    ] {
        let mut j = DMatrix::zeros(n + 2, n);
        j.view_mut((0, 0), (n, n))
            .copy_from(&DMatrix::identity(n, n));
        j.row_mut(0)
            .copy_from(&nalgebra::RowDVector::from_row_slice(&initial));
        j.row_mut(n)
            .copy_from(&nalgebra::RowDVector::from_row_slice(&added));
        j.row_mut(n + 1)
            .copy_from(&nalgebra::RowDVector::from_row_slice(&[
                0., 1., -2., 3., -4., 5., -6.,
            ]));
        let residual = DVector::from_fn(j.nrows(), |r, _| (r as f64 * 0.31).cos());
        let mut batch = IncrementalQr::new(n, Default::default()).unwrap();
        batch.append(&csr(&j), &residual).unwrap();
        check(&j, &residual, &batch);
        let mut split = IncrementalQr::new(n, Default::default()).unwrap();
        split
            .append(
                &csr(&j.rows(0, n).into_owned()),
                &residual.rows(0, n).into_owned(),
            )
            .unwrap();
        for row in n..j.nrows() {
            split
                .append(
                    &csr(&j.rows(row, 1).into_owned()),
                    &residual.rows(row, 1).into_owned(),
                )
                .unwrap();
            check(
                &j.rows(0, row + 1).into_owned(),
                &residual.rows(0, row + 1).into_owned(),
                &split,
            );
        }
        assert_eq!(split.solve().unwrap(), batch.solve().unwrap());
        assert_eq!(split.stored_entries(), batch.stored_entries());
        assert_eq!(
            split.cost(&DVector::zeros(n)).unwrap(),
            batch.cost(&DVector::zeros(n)).unwrap()
        );
    }
}

#[test]
fn prefix_cancellation_frees_a_slot_before_trailing_fill_and_later_failure_rolls_back() {
    let initial = DMatrix::from_row_slice(3, 3, &[1., 1., 0., 0., 1., 0., 0., 0., 1.]);
    let r = DVector::from_vec(vec![1., -2., 3.]);
    let added = DMatrix::from_row_slice(1, 3, &[1., -1., 1.]);
    let mut bounded = IncrementalQr::new(
        3,
        IncrementalQrOptions {
            max_entries: 4,
            ..Default::default()
        },
    )
    .unwrap();
    bounded.append(&csr(&initial), &r).unwrap();
    let x = bounded.solve().unwrap();
    let cost = bounded.cost(&DVector::zeros(3)).unwrap();
    assert_eq!(
        bounded.append(&csr(&added), &DVector::from_element(1, -0.25)),
        Err(Error::SparseFillLimit { limit: 4 })
    );
    assert_eq!(bounded.solve().unwrap(), x);
    assert_eq!(bounded.stored_entries(), 4);
    assert_eq!(bounded.cost(&DVector::zeros(3)).unwrap(), cost);
    bounded
        .append(&csr(&DMatrix::zeros(1, 3)), &DVector::from_element(1, 0.5))
        .unwrap();
    assert_eq!(bounded.cost(&DVector::zeros(3)).unwrap(), cost + 0.125);

    let mut permitted = IncrementalQr::new(
        3,
        IncrementalQrOptions {
            max_entries: 5,
            ..Default::default()
        },
    )
    .unwrap();
    permitted.append(&csr(&initial), &r).unwrap();
    permitted
        .append(&csr(&added), &DVector::from_element(1, -0.25))
        .unwrap();
    assert_eq!(permitted.stored_entries(), 5);
    let j = DMatrix::from_row_slice(4, 3, &[1., 1., 0., 0., 1., 0., 0., 0., 1., 1., -1., 1.]);
    check(&j, &DVector::from_vec(vec![1., -2., 3., -0.25]), &permitted);
}

#[test]
fn norm_overflow_after_reused_upper_and_lower_tails_restores_the_factor() {
    let j = DMatrix::from_row_slice(
        3,
        3,
        &[1e308, 1e308, 1.7e308, 0., 1e308, 0., 0., 0., 1.7e308],
    );
    let mut qr = IncrementalQr::new(3, Default::default()).unwrap();
    qr.append(&csr(&j), &DVector::zeros(3)).unwrap();
    let x = qr.solve().unwrap();
    assert_eq!(
        qr.append(
            &csr(&DMatrix::from_row_slice(1, 3, &[1e308, -1e308, 0.])),
            &DVector::zeros(1)
        ),
        Err(Error::NonFinite)
    );
    assert_eq!(qr.solve().unwrap(), x);
    assert_eq!(qr.stored_entries(), 5);
    assert_eq!(qr.cost(&DVector::zeros(3)).unwrap(), 0.);
    qr.append(&csr(&DMatrix::zeros(1, 3)), &DVector::from_element(1, 0.5))
        .unwrap();
    assert_eq!(qr.solve().unwrap(), x);
    assert_eq!(qr.cost(&DVector::zeros(3)).unwrap(), 0.125);
}
