use ropt::{DMatrix, DVector, Error, marginalization::*};

fn check_profile(j: &DMatrix<f64>, r: &DVector<f64>, ne: usize) {
    let prior = marginalize_linearized(j, r, ne, &MarginalizationOptions::default()).unwrap();
    let nr = j.ncols() - ne;
    assert!(prior.residual.len() <= nr + 1);
    assert_eq!(prior.jacobian.ncols(), nr);
    for trial in 0..5 {
        let d = DVector::from_fn(nr, |i, _| ((trial + i + 1) as f64).cos());
        let adjusted = r + j.columns(ne, nr) * &d;
        let minimized = if ne == 0 {
            adjusted
        } else {
            let e = j.columns(0, ne).into_owned();
            let delta = e
                .clone()
                .svd(true, true)
                .solve(&(-&adjusted), 1e-12)
                .unwrap();
            adjusted + e * delta
        };
        let actual = &prior.residual + &prior.jacobian * &d;
        assert!(
            (actual.norm_squared() - minimized.norm_squared()).abs() < 1e-10,
            "{} != {}",
            actual.norm_squared(),
            minimized.norm_squared()
        );
    }
}
#[test]
fn qr_prior_preserves_profile_objective_and_constant_without_normal_equations() {
    let j = DMatrix::from_fn(9, 5, |r, c| {
        if r == c {
            2.0
        } else {
            ((r + 3 * c + 1) as f64).sin()
        }
    });
    let r = DVector::from_fn(9, |i, _| ((i + 2) as f64).cos());
    for ne in 0..=5 {
        check_profile(&j, &r, ne);
    }
    // Retained columns are deliberately dependent; the prior stays rank deficient.
    let j = DMatrix::from_row_slice(
        5,
        3,
        &[1., 2., 4., 0., 1., 2., 1., 3., 6., 2., 1., 2., 0., 0., 0.],
    );
    check_profile(&j, &DVector::from_vec(vec![1., -2., 0.5, 0.7, 3.]), 1);
}
#[test]
fn exact_and_underdetermined_retained_systems_work() {
    check_profile(
        &DMatrix::identity(3, 3),
        &DVector::from_vec(vec![1., 2., 3.]),
        3,
    );
    check_profile(
        &DMatrix::from_row_slice(2, 4, &[1., 0., 2., 3., 0., 1., 4., 5.]),
        &DVector::from_vec(vec![1., 2.]),
        1,
    );
    check_profile(
        &DMatrix::zeros(3, 0),
        &DVector::from_vec(vec![1., 2., 3.]),
        0,
    );
}
#[test]
fn compression_preserves_information_at_large_and_small_scales() {
    let j = DMatrix::from_row_slice(4, 2, &[1., 2., 0., 1., 2., 1., 1., 0.]);
    let r = DVector::from_vec(vec![1., 2., 3., 4.]);
    let baseline = marginalize_linearized(&j, &r, 1, &MarginalizationOptions::default()).unwrap();
    for scale in [1e-150, 1e150] {
        let actual = marginalize_linearized(
            &(&j * scale),
            &(&r * scale),
            1,
            &MarginalizationOptions::default(),
        )
        .unwrap();
        for delta in [-2., 0., 3.] {
            let a = (&actual.residual + &actual.jacobian * DVector::from_element(1, delta)) / scale;
            let b = &baseline.residual + &baseline.jacobian * DVector::from_element(1, delta);
            assert!((a.norm_squared() - b.norm_squared()).abs() < 1e-10);
        }
    }
}
#[test]
fn unobservable_eliminated_directions_and_bad_inputs_fail() {
    let opts = MarginalizationOptions::default();
    assert_eq!(
        marginalize_linearized(
            &DMatrix::from_row_slice(2, 3, &[1., 1., 2., 2., 2., 1.]),
            &DVector::zeros(2),
            2,
            &opts
        )
        .unwrap_err(),
        Error::RankDeficient {
            rank: 1,
            parameters: 2
        }
    );
    assert_eq!(
        marginalize_linearized(&DMatrix::zeros(1, 2), &DVector::zeros(1), 2, &opts).unwrap_err(),
        Error::RankDeficient {
            rank: 0,
            parameters: 2
        }
    );
    assert_eq!(
        marginalize_linearized(&DMatrix::zeros(2, 2), &DVector::zeros(1), 0, &opts).unwrap_err(),
        Error::InvalidDimensions
    );
    assert_eq!(
        marginalize_linearized(
            &DMatrix::from_element(2, 2, f64::NAN),
            &DVector::zeros(2),
            0,
            &opts
        )
        .unwrap_err(),
        Error::NonFinite
    );
    for bad in [
        MarginalizationOptions {
            max_entries: 0,
            ..opts
        },
        MarginalizationOptions {
            relative_rank_tolerance: f64::NAN,
            ..opts
        },
        MarginalizationOptions {
            svd_max_iterations: 0,
            ..opts
        },
    ] {
        assert_eq!(
            marginalize_linearized(&DMatrix::identity(2, 2), &DVector::zeros(2), 1, &bad)
                .unwrap_err(),
            Error::InvalidOptions
        );
    }
    assert!(
        marginalize_linearized(
            &DMatrix::identity(2, 2),
            &DVector::zeros(2),
            1,
            &MarginalizationOptions {
                max_entries: 5,
                ..opts
            }
        )
        .is_err()
    );
}
