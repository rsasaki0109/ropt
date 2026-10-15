use ropt::{
    autodiff::{AutoDiff, Jet},
    *,
};
fn rosenbrock() -> impl Residual {
    AutoDiff::<2, 2, _>(|x: [Jet<2>; 2]| {
        [
            Jet::constant(10.0) * (x[1] - x[0] * x[0]),
            Jet::constant(1.0) - x[0],
        ]
    })
}
#[test]
fn lm_rosenbrock_converges_with_rejections() {
    let options = Options {
        max_iterations: 200,
        ..Options::default()
    };
    let report = solve(&rosenbrock(), DVector::from_vec(vec![-1.2, 1.0]), &options).unwrap();
    assert!(report.cost < 1e-16, "{report:?}");
    assert!((&report.x - DVector::from_element(2, 1.0)).norm() < 1e-7);
    assert!(report.iterations.iter().any(|i| !i.accepted));
    for i in &report.iterations {
        if i.accepted {
            assert!(i.trial_cost < i.cost);
            assert!(i.gain_ratio > 0.0);
        }
    }
    assert_ne!(report.termination, Termination::MaxIterations);
}
#[test]
fn gn_solves_linear_problem() {
    let p = AutoDiff::<2, 3, _>(|x: [Jet<2>; 2]| {
        [
            x[0] + x[1] - Jet::constant(3.0),
            x[0] - x[1] - Jet::constant(1.0),
            x[0] * Jet::constant(2.0) - Jet::constant(4.0),
        ]
    });
    let report = solve(
        &p,
        DVector::zeros(2),
        &Options {
            algorithm: Algorithm::GaussNewton,
            ..Options::default()
        },
    )
    .unwrap();
    assert!(report.cost < 1e-25);
    assert!((report.x[0] - 2.0).abs() < 1e-12);
    assert!((report.x[1] - 1.0).abs() < 1e-12);
}
#[test]
fn derivatives_match_analytic_and_central_difference() {
    let x = DVector::from_vec(vec![0.7, 1.2]);
    let mut r = DVector::zeros(2);
    let mut j = DMatrix::zeros(2, 2);
    rosenbrock().evaluate(&x, &mut r, &mut j).unwrap();
    let analytic = DMatrix::from_row_slice(2, 2, &[-14.0, 10.0, -1.0, 0.0]);
    assert!((&j - analytic).norm() < 1e-12);
    let numeric = CentralDifference {
        parameters: 2,
        residuals: 2,
        step: 1e-6,
        function: |x: &DVector<f64>| {
            Ok(DVector::from_vec(vec![
                10.0 * (x[1] - x[0] * x[0]),
                1.0 - x[0],
            ]))
        },
    };
    let mut nr = DVector::zeros(2);
    let mut nj = DMatrix::zeros(2, 2);
    numeric.evaluate(&x, &mut nr, &mut nj).unwrap();
    assert!((nj - j).norm() < 1e-7);
    assert!((nr - r).norm() < 1e-12);
}
#[test]
fn elementary_jet_functions_match_difference() {
    fn f(x: Jet<1>) -> Jet<1> {
        ((x.sin() + x.cos()) * x.exp() / x.sqrt()).ln()
    }
    let x = 0.7;
    let a = f(Jet::variable(x, 0));
    let h = 1e-6;
    let expected = (f(Jet::constant(x + h)).value - f(Jet::constant(x - h)).value) / (2.0 * h);
    assert!((a.derivative[0] - expected).abs() < 1e-8);
}
#[test]
fn robust_loss_reduces_outlier_bias() {
    let p = AutoDiff::<1, 5, _>(|x: [Jet<1>; 1]| {
        [
            x[0] - Jet::constant(1.0),
            x[0] - Jet::constant(1.0),
            x[0] - Jet::constant(1.0),
            x[0] - Jet::constant(1.0),
            x[0] - Jet::constant(100.0),
        ]
    });
    let plain = solve(&p, DVector::zeros(1), &Options::default()).unwrap();
    let robust = solve(
        &p,
        DVector::zeros(1),
        &Options {
            loss: Loss::Huber(1.0),
            ..Options::default()
        },
    )
    .unwrap();
    assert!((plain.x[0] - 20.8).abs() < 1e-5);
    assert!((robust.x[0] - 1.25).abs() < 1e-5);
}
#[test]
fn invalid_inputs_and_rank_deficiency_are_errors() {
    let p = AutoDiff::<1, 1, _>(|x: [Jet<1>; 1]| [x[0]]);
    assert_eq!(
        solve(&p, DVector::zeros(2), &Options::default()).unwrap_err(),
        Error::InvalidDimensions
    );
    assert_eq!(
        solve(&p, DVector::from_element(1, f64::NAN), &Options::default()).unwrap_err(),
        Error::NonFinite
    );
    assert_eq!(
        solve(
            &p,
            DVector::zeros(1),
            &Options {
                loss: Loss::Huber(-1.0),
                ..Options::default()
            }
        )
        .unwrap_err(),
        Error::InvalidOptions
    );
    let singular = AutoDiff::<2, 2, _>(|x: [Jet<2>; 2]| {
        [
            x[0] + x[1] - Jet::constant(1.0),
            x[0] + x[1] - Jet::constant(1.0),
        ]
    });
    assert_eq!(
        solve(
            &singular,
            DVector::zeros(2),
            &Options {
                algorithm: Algorithm::GaussNewton,
                ..Options::default()
            }
        )
        .unwrap_err(),
        Error::LinearSolve
    );
    assert!(
        solve(&singular, DVector::zeros(2), &Options::default())
            .unwrap()
            .cost
            < 1e-15
    );
    let nan = AutoDiff::<1, 1, _>(|_x: [Jet<1>; 1]| [Jet::constant(f64::INFINITY)]);
    assert_eq!(
        solve(&nan, DVector::zeros(1), &Options::default()).unwrap_err(),
        Error::NonFinite
    );
}
#[test]
fn iteration_limit_is_reported() {
    let result = solve(
        &rosenbrock(),
        DVector::from_vec(vec![-1.2, 1.0]),
        &Options {
            max_iterations: 1,
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(result.termination, Termination::MaxIterations);
    assert_eq!(result.iterations.len(), 1);
}
#[test]
fn parameter_blocks_are_stable_ranges() {
    let mut p = Parameters::default();
    let a = p.add_block(&[1.0, 2.0]).unwrap();
    let b = p.add_block(&[3.0]).unwrap();
    assert_eq!(p.block_range(a), Some(0..2));
    assert_eq!(p.block_range(b), Some(2..3));
    assert_eq!(p.vector().as_slice(), &[1.0, 2.0, 3.0]);
    assert_eq!(p.add_block(&[]), Err(Error::InvalidDimensions));
}
#[test]
fn evaluation_errors_propagate() {
    let p = CentralDifference {
        parameters: 1,
        residuals: 1,
        step: 1e-6,
        function: |_: &DVector<f64>| Err(Error::Evaluation("domain".into())),
    };
    assert_eq!(
        solve(&p, DVector::zeros(1), &Options::default()).unwrap_err(),
        Error::Evaluation("domain".into())
    );
}

#[test]
fn analytic_residual_and_termination_reasons() {
    struct Linear;
    impl Residual for Linear {
        fn dimensions(&self) -> (usize, usize) {
            (1, 1)
        }
        fn evaluate(
            &self,
            x: &DVector<f64>,
            r: &mut DVector<f64>,
            j: &mut DMatrix<f64>,
        ) -> Result<(), Error> {
            r[0] = x[0] - 3.0;
            j[(0, 0)] = 1.0;
            Ok(())
        }
    }
    let p = Linear;
    assert_eq!(
        solve(&p, DVector::from_element(1, 3.0), &Options::default())
            .unwrap()
            .termination,
        Termination::Gradient
    );
    let base = Options {
        gradient_tolerance: 0.0,
        cost_tolerance: 0.0,
        ..Options::default()
    };
    assert_eq!(
        solve(
            &p,
            DVector::zeros(1),
            &Options {
                step_tolerance: 10.0,
                ..base.clone()
            }
        )
        .unwrap()
        .termination,
        Termination::Step
    );
    assert_eq!(
        solve(
            &p,
            DVector::zeros(1),
            &Options {
                cost_tolerance: 10.0,
                step_tolerance: 0.0,
                ..base
            }
        )
        .unwrap()
        .termination,
        Termination::Cost
    );
}
#[test]
fn rejected_trials_preserve_the_previous_state() {
    let p = AutoDiff::<1, 1, _>(|x: [Jet<1>; 1]| [x[0] * x[0] - Jet::constant(1.0)]);
    let report = solve(
        &p,
        DVector::from_element(1, 0.01),
        &Options {
            max_iterations: 1,
            initial_damping: 1e-12,
            ..Options::default()
        },
    )
    .unwrap();
    assert!(!report.iterations[0].accepted);
    assert_eq!(report.x[0], 0.01);
    assert_eq!(report.cost, report.initial_cost);
}
#[test]
fn cauchy_loss_and_underconstrained_gn() {
    let p = AutoDiff::<1, 3, _>(|x: [Jet<1>; 1]| {
        [
            x[0] - Jet::constant(1.0),
            x[0] - Jet::constant(1.0),
            x[0] - Jet::constant(100.0),
        ]
    });
    let r = solve(
        &p,
        DVector::zeros(1),
        &Options {
            loss: Loss::Cauchy(1.0),
            ..Options::default()
        },
    )
    .unwrap();
    assert!((r.x[0] - 1.0).abs() < 0.02);
    let under = AutoDiff::<2, 1, _>(|x: [Jet<2>; 2]| [x[0] + x[1] - Jet::constant(1.0)]);
    assert_eq!(
        solve(
            &under,
            DVector::zeros(2),
            &Options {
                algorithm: Algorithm::GaussNewton,
                ..Options::default()
            }
        )
        .unwrap_err(),
        Error::LinearSolve
    );
}
