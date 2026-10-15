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
fn mixed_wide_cancellations_preserve_svd_objective_and_exact_split_reports() {
    for n in [2, 3, 7, 16, 65] {
        let mut j = DMatrix::from_fn(n + 19, n, |r, c| {
            if r < n {
                if c < r {
                    0.
                } else if c == r {
                    4.
                } else {
                    ((r + 3 * c) as f64 * 0.17).sin() * 0.3
                }
            } else {
                ((r + 2 * c) as f64 * 0.23).cos()
            }
        });
        j.row_mut(0).fill(1.);
        // Equal c and s cancel alternating upper/lower entries across lanes.
        for c in 0..n {
            j[(n, c)] = if c == 0 || c % 2 == 1 { 1. } else { -1. };
        }
        let residual = DVector::from_fn(j.nrows(), |r, _| (r as f64 * 0.31).sin());
        let mut qr = IncrementalQr::new(n, Default::default()).unwrap();
        qr.append(&csr(&j), &residual).unwrap();
        let expected = j
            .clone()
            .svd(true, true)
            .solve(&(-&residual), 1e-12)
            .unwrap();
        let x = qr.solve().unwrap();
        assert!((&x - expected).norm() < 1e-11);
        let mut split = IncrementalQr::new(n, Default::default()).unwrap();
        for r in 0..j.nrows() {
            split
                .append(
                    &csr(&j.rows(r, 1).into_owned()),
                    &residual.rows(r, 1).into_owned(),
                )
                .unwrap();
        }
        assert_eq!(split.stored_entries(), qr.stored_entries());
        assert_eq!(
            split.solve().unwrap().map(f64::to_bits),
            x.map(f64::to_bits)
        );
        for probe in [x, DVector::zeros(n), DVector::from_element(n, -0.2)] {
            let expected = 0.5 * (&j * &probe + &residual).norm_squared();
            let cost = qr.cost(&probe).unwrap();
            assert!((cost - expected).abs() < 1e-11 * expected.max(1.));
            assert_eq!(split.cost(&probe).unwrap().to_bits(), cost.to_bits());
        }
    }
}

#[test]
fn first_middle_and_last_suffix_overflow_roll_back_at_exact_fill_limit() {
    for n in [3, 8, 65] {
        for fault in [1, n / 2, n - 1] {
            let mut j = DMatrix::identity(n, n);
            j.row_mut(0).fill(1.);
            j[(0, fault)] = 1.7e308;
            let options = IncrementalQrOptions {
                max_entries: 2 * n - 1,
                ..Default::default()
            };
            let mut qr = IncrementalQr::new(n, options).unwrap();
            qr.append(&csr(&j), &DVector::zeros(n)).unwrap();
            let x = qr.solve().unwrap().map(f64::to_bits);
            let zero = DVector::zeros(n);
            let cost = qr.cost(&zero).unwrap().to_bits();
            assert_eq!(qr.stored_entries(), options.max_entries);
            let mut added = DMatrix::from_element(1, n, 1.);
            added[(0, fault)] = 1.7e308;
            assert_eq!(
                qr.append(&csr(&added), &DVector::zeros(1)),
                Err(Error::NonFinite)
            );
            assert_eq!(qr.solve().unwrap().map(f64::to_bits), x);
            assert_eq!(qr.stored_entries(), options.max_entries);
            assert_eq!(qr.cost(&zero).unwrap().to_bits(), cost);
            qr.append(&csr(&DMatrix::zeros(1, n)), &DVector::from_element(1, 0.5))
                .unwrap();
            assert_eq!(qr.cost(&zero).unwrap(), 0.125);
            assert_eq!(qr.solve().unwrap().map(f64::to_bits), x);
        }
    }
}

#[test]
fn extreme_scales_preserve_rank_solution_and_scaled_objective() {
    for scale in [1e-290, 1., 1e290] {
        let n = 7;
        let unscaled = DMatrix::from_fn(23, n, |r, c| {
            if r < n {
                if c < r {
                    0.
                } else if c == r {
                    2.
                } else {
                    0.25
                }
            } else {
                ((r + 3 * c) as f64 * 0.13).cos()
            }
        });
        let j = &unscaled * scale;
        let mut qr = IncrementalQr::new(n, Default::default()).unwrap();
        qr.append(&csr(&j), &DVector::zeros(j.nrows())).unwrap();
        assert_eq!(qr.solve().unwrap(), DVector::zeros(n));
        let probe = DVector::from_fn(n, |k, _| (k as f64 * 0.17).sin() / scale);
        let expected = 0.5 * (&j * &probe).norm_squared();
        assert!((qr.cost(&probe).unwrap() - expected).abs() < 1e-11 * expected.max(1.));
    }
}
