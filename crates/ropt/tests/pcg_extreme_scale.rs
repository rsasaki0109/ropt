use ropt::{DVector, Error, sparse::*};

fn options() -> CgOptions {
    CgOptions {
        max_iterations: 4,
        relative_tolerance: 1e-12,
        absolute_tolerance: 0.,
    }
}

fn check_scaled_spd(scale: f64) {
    // The solution of [4 1; 1 3] x = [1 2] is [1/11, 7/11].
    // A common scale leaves the solution and condition number unchanged.
    let matrix = CsrMatrix::from_triplets(
        2,
        2,
        vec![
            (0, 0, 4. * scale),
            (0, 1, scale),
            (1, 0, scale),
            (1, 1, 3. * scale),
        ],
    )
    .unwrap();
    let b = DVector::from_vec(vec![scale, 2. * scale]);
    let report = conjugate_gradient(&matrix, &b, &options()).unwrap();
    assert!((report.x[0] - 1. / 11.).abs() < 1e-12);
    assert!((report.x[1] - 7. / 11.).abs() < 1e-12);
    assert!(report.iterations > 0 && report.iterations <= 2);
    // Independent scaled residual oracle: neither squaring huge entries nor
    // squaring tiny ones is needed to check the accuracy of the returned x.
    let residual = &b - matrix.multiply(&report.x).unwrap();
    let relative = (residual[0] / scale).hypot(residual[1] / scale) / 5_f64.sqrt();
    assert!(relative <= options().relative_tolerance, "{relative}");
    let true_norm = residual[0].hypot(residual[1]);
    assert!(report.residual_norm.is_finite());
    assert!((report.residual_norm / scale - true_norm / scale).abs() < 1e-25);
}

#[test]
fn tiny_well_conditioned_system_does_not_converge_at_zero() {
    for scale in [1e-200, 1e-160] {
        check_scaled_spd(scale);
    }
}

#[test]
fn large_well_conditioned_system_does_not_overflow_its_norm() {
    check_scaled_spd(1e200);
}

#[test]
fn accepted_initial_residual_is_reported_at_its_actual_scale() {
    let identity = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 1.), (1, 1, 1.)]).unwrap();
    for scale in [1e-200, 1e-160, 1e200] {
        let b = DVector::from_vec(vec![3. * scale, 4. * scale]);
        let report = conjugate_gradient(
            &identity,
            &b,
            &CgOptions {
                absolute_tolerance: 6. * scale,
                ..options()
            },
        )
        .unwrap();
        assert_eq!(report.iterations, 0);
        assert_eq!(report.x, DVector::zeros(2));
        assert!((report.residual_norm / scale - 5.).abs() < 1e-14);
    }
}

#[test]
fn inner_product_underflow_is_breakdown_instead_of_false_convergence() {
    let identity = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 1.)]).unwrap();
    // Robust norms do not make the PCG recurrence scale invariant: r.dot(z)
    // still underflows here. It must fail rather than certify a zero solution.
    let b = DVector::from_element(1, 1e-200);
    assert_eq!(
        conjugate_gradient(&identity, &b, &options()).unwrap_err(),
        Error::LinearSolve
    );
}
