use ropt::*;
use std::cell::Cell;

#[test]
fn invalid_perturbations_are_rejected_before_callbacks_or_output_changes() {
    for (value, step, expected) in [
        (1., 1e-30, Error::InvalidOptions),
        (f64::MAX, 1e-6, Error::NonFinite),
        (2., f64::MAX, Error::NonFinite),
        (0., 1e308, Error::NonFinite),
        (f64::NAN, 1e-6, Error::NonFinite),
        (f64::INFINITY, 1e-6, Error::NonFinite),
    ] {
        let calls = Cell::new(0);
        let problem = CentralDifference {
            parameters: 2,
            residuals: 1,
            step,
            function: |x: &DVector<f64>| {
                calls.set(calls.get() + 1);
                Ok(DVector::from_element(1, x[0] + x[1]))
            },
        };
        // Include a valid first coordinate when only the second is invalid.
        let x = DVector::from_vec(vec![0., value]);
        let mut residual = DVector::from_element(1, 7.);
        let mut jacobian = DMatrix::from_element(1, 2, 11.);
        assert_eq!(
            problem.evaluate(&x, &mut residual, &mut jacobian),
            Err(expected)
        );
        assert_eq!(calls.get(), 0);
        assert_eq!(residual[0], 7.);
        assert_eq!(jacobian, DMatrix::from_element(1, 2, 11.));
    }
}

#[test]
fn too_small_a_step_cannot_falsely_establish_solver_convergence() {
    let calls = Cell::new(0);
    let problem = CentralDifference {
        parameters: 1,
        residuals: 1,
        step: 1e-30,
        function: |x: &DVector<f64>| {
            calls.set(calls.get() + 1);
            Ok(DVector::from_element(1, x[0] - 2.))
        },
    };
    // x +/- 1e-30 both round to 1, so a naive difference spuriously gives J=0
    // and a zero gradient, despite the nonzero residual and exact minimizer 2.
    assert_eq!(
        solve(&problem, DVector::from_element(1, 1.), &Options::default()).unwrap_err(),
        Error::InvalidOptions
    );
    assert_eq!(calls.get(), 0);
}

#[test]
fn nonfinite_callback_values_and_derivative_overflow_are_errors() {
    for (behavior, expected_calls) in [(0, 1), (1, 2), (2, 3)] {
        let calls = Cell::new(0);
        let problem = CentralDifference {
            parameters: 1,
            residuals: 1,
            step: 1e-6,
            function: |x: &DVector<f64>| {
                calls.set(calls.get() + 1);
                let value = match behavior {
                    0 => f64::NAN,
                    1 if x[0] > 0. => f64::INFINITY,
                    1 => 0.,
                    _ if x[0] == 0. => 0.,
                    _ => f64::MAX * x[0].signum(),
                };
                Ok(DVector::from_element(1, value))
            },
        };
        let mut residual = DVector::zeros(1);
        let mut jacobian = DMatrix::zeros(1, 1);
        assert_eq!(
            problem.evaluate(&DVector::zeros(1), &mut residual, &mut jacobian),
            Err(Error::NonFinite)
        );
        assert_eq!(calls.get(), expected_calls);
    }
}

#[test]
fn representable_subnormal_step_and_normal_solve_remain_valid() {
    let identity = CentralDifference {
        parameters: 1,
        residuals: 1,
        step: f64::from_bits(1),
        function: |x: &DVector<f64>| Ok(x.clone()),
    };
    let mut residual = DVector::zeros(1);
    let mut jacobian = DMatrix::zeros(1, 1);
    identity
        .evaluate(&DVector::zeros(1), &mut residual, &mut jacobian)
        .unwrap();
    assert_eq!(residual[0], 0.);
    assert_eq!(jacobian[(0, 0)], 1.);

    let problem = CentralDifference {
        parameters: 1,
        residuals: 1,
        step: 1e-6,
        function: |x: &DVector<f64>| Ok(DVector::from_element(1, x[0] - 2.)),
    };
    let report = solve(&problem, DVector::from_element(1, 1.), &Options::default()).unwrap();
    assert!((report.x[0] - 2.).abs() < 1e-8);
    assert!(report.cost < 1e-16);
}
