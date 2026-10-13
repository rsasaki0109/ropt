use ropt::{
    autodiff::{AutoDiff, DynamicAutoDiff, DynamicJet, Jet},
    factor::*,
    manifold::*,
    noise::NoiseModel,
    sparse::CgOptions,
    *,
};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

#[test]
fn runtime_chunks_match_fixed_jets_analytic_formula_and_central_differences() {
    let fixed = AutoDiff::<2, 3, _>(|x: [Jet<2>; 2]| {
        [
            x[0].sin() * x[1].exp(),
            x[0].atan2(x[1]) + x[0].tan() / x[1],
            x[0].sqrt() + x[1].ln() + x[0].powi(3) + x[1].atan(),
        ]
    });
    let dynamic = |x: &[DynamicJet]| {
        Ok(vec![
            x[0].sin() * x[1].exp(),
            x[0].atan2(&x[1]) + x[0].tan() / &x[1],
            x[0].sqrt() + x[1].ln() + x[0].powi(3) + x[1].atan(),
        ])
    };
    let numeric = CentralDifference {
        parameters: 2,
        residuals: 3,
        step: 1e-6,
        function: |x: &DVector<f64>| {
            Ok(DVector::from_vec(vec![
                x[0].sin() * x[1].exp(),
                x[0].atan2(x[1]) + x[0].tan() / x[1],
                x[0].sqrt() + x[1].ln() + x[0].powi(3) + x[1].atan(),
            ]))
        },
    };
    let x = DVector::from_vec(vec![0.7, 1.2]);
    let mut r = DVector::zeros(3);
    let mut expected = DMatrix::zeros(3, 2);
    fixed.evaluate(&x, &mut r, &mut expected).unwrap();
    let analytic = DMatrix::from_row_slice(
        3,
        2,
        &[
            x[0].cos() * x[1].exp(),
            x[0].sin() * x[1].exp(),
            x[1] / (x[0] * x[0] + x[1] * x[1]) + (1.0 + x[0].tan().powi(2)) / x[1],
            -x[0] / (x[0] * x[0] + x[1] * x[1]) - x[0].tan() / (x[1] * x[1]),
            0.5 / x[0].sqrt() + 3.0 * x[0] * x[0],
            1.0 / x[1] + 1.0 / (1.0 + x[1] * x[1]),
        ],
    );
    assert!((&expected - analytic).norm() < 1e-12);
    let mut nr = DVector::zeros(3);
    let mut nj = DMatrix::zeros(3, 2);
    numeric.evaluate(&x, &mut nr, &mut nj).unwrap();
    assert!((&nj - &expected).norm() < 1e-7);
    for chunk in [1, 2, 32] {
        let residual = DynamicAutoDiff::new(2, 3, dynamic)
            .unwrap()
            .with_chunk_size(chunk)
            .unwrap();
        let mut dr = DVector::zeros(3);
        let mut dj = DMatrix::zeros(3, 2);
        residual.evaluate(&x, &mut dr, &mut dj).unwrap();
        assert!((&dr - &r).norm() < 1e-12);
        assert!((&dj - &expected).norm() < 1e-12);
    }
}
#[test]
fn bounded_seed_chunks_cover_a_runtime_parameter_count() {
    let count = 41;
    let calls = Rc::new(Cell::new(0));
    let widths = Rc::new(RefCell::new(Vec::new()));
    let c = calls.clone();
    let w = widths.clone();
    let residual = DynamicAutoDiff::new(count, 3, move |x: &[DynamicJet]| {
        c.set(c.get() + 1);
        w.borrow_mut()
            .push(x.iter().map(|v| v.partials().unwrap().len()).max().unwrap());
        let linear = x
            .iter()
            .enumerate()
            .fold(DynamicJet::constant(0.0), |sum, (i, v)| {
                sum + v * (i + 1) as f64
            });
        let quadratic = x
            .iter()
            .fold(DynamicJet::constant(0.0), |sum, v| sum + v.powi(2));
        Ok(vec![linear, quadratic, DynamicJet::constant(7.0)])
    })
    .unwrap()
    .with_chunk_size(7)
    .unwrap();
    let x = DVector::from_iterator(count, (0..count).map(|i| (i as f64).sin()));
    let mut r = DVector::zeros(3);
    let mut j = DMatrix::from_element(3, count, 99.0);
    residual.evaluate(&x, &mut r, &mut j).unwrap();
    assert_eq!(calls.get(), 6);
    assert_eq!(*widths.borrow(), vec![7, 7, 7, 7, 7, 6]);
    for i in 0..count {
        assert_eq!(j[(0, i)], (i + 1) as f64);
        assert_eq!(j[(1, i)], 2.0 * x[i]);
        assert_eq!(j[(2, i)], 0.0);
    }
    assert!(
        (r[0]
            - x.iter()
                .enumerate()
                .map(|(i, v)| v * (i + 1) as f64)
                .sum::<f64>())
        .abs()
            < 1e-10
    );
    assert!((r[1] - x.norm_squared()).abs() < 1e-12);
    assert_eq!(r[2], 7.0);
}
#[test]
fn constant_broadcast_scalar_operators_and_error_carrying_are_explicit() {
    let x = DynamicJet::variable(2.0, 1, 3).unwrap();
    let expression = 3.0 * &x + (&x - 1.0) / 2.0 - 1.0 / &x;
    assert!((expression.value() - 6.0).abs() < 1e-12);
    assert_eq!(expression.partials().unwrap(), &[0.0, 3.75, 0.0]);
    assert_eq!(DynamicJet::constant(3.0).partials().unwrap(), &[]);
    assert_eq!(
        (x + DynamicJet::variable(1.0, 0, 2).unwrap())
            .partials()
            .unwrap_err(),
        Error::InvalidDimensions
    );
    assert_eq!(
        DynamicJet::variable(1.0, 3, 3).unwrap_err(),
        Error::InvalidDimensions
    );
    assert_eq!(
        DynamicJet::variable(f64::NAN, 0, 1).unwrap_err(),
        Error::NonFinite
    );
}
#[test]
fn changing_values_dimensions_domains_and_callback_errors_are_rejected() {
    let counter = Cell::new(0);
    let stateful = DynamicAutoDiff::new(2, 1, |_: &[DynamicJet]| {
        counter.set(counter.get() + 1);
        Ok(vec![DynamicJet::constant(counter.get() as f64)])
    })
    .unwrap()
    .with_chunk_size(1)
    .unwrap();
    let mut r = DVector::zeros(1);
    let mut j = DMatrix::zeros(1, 2);
    assert!(matches!(
        stateful.evaluate(&DVector::zeros(2), &mut r, &mut j),
        Err(Error::Evaluation(_))
    ));
    let wrong = DynamicAutoDiff::new(2, 1, |_: &[DynamicJet]| {
        Ok(vec![DynamicJet::variable(1.0, 0, 3).unwrap()])
    })
    .unwrap();
    assert_eq!(
        wrong.evaluate(&DVector::zeros(2), &mut r, &mut j),
        Err(Error::InvalidDimensions)
    );
    let wrong_count = DynamicAutoDiff::new(2, 1, |_: &[DynamicJet]| Ok(vec![])).unwrap();
    assert_eq!(
        wrong_count.evaluate(&DVector::zeros(2), &mut r, &mut j),
        Err(Error::InvalidDimensions)
    );
    let domain = DynamicAutoDiff::new(2, 1, |x: &[DynamicJet]| Ok(vec![x[0].ln()])).unwrap();
    assert_eq!(
        domain.evaluate(&DVector::from_vec(vec![-1.0, 1.0]), &mut r, &mut j),
        Err(Error::NonFinite)
    );
    let failed = DynamicAutoDiff::new(2, 1, |_: &[DynamicJet]| {
        Err(Error::Evaluation("domain".into()))
    })
    .unwrap();
    assert_eq!(
        failed.evaluate(&DVector::zeros(2), &mut r, &mut j),
        Err(Error::Evaluation("domain".into()))
    );
    assert!(DynamicAutoDiff::new(0, 1, |_: &[DynamicJet]| Ok(vec![])).is_err());
    assert!(
        DynamicAutoDiff::new(2, 1, |_: &[DynamicJet]| Ok(vec![]))
            .unwrap()
            .with_chunk_size(0)
            .is_err()
    );
}
#[test]
fn fixed_and_dynamic_edge_formulas_are_finite_or_report_domain_errors() {
    let fixed = Jet::<2>::variable(1e200, 0) / Jet::<2>::variable(1e200, 1);
    assert_eq!(fixed.value, 1.0);
    assert!((fixed.derivative[0] / 1e-200 - 1.0).abs() < 1e-12);
    assert!((fixed.derivative[1] / 1e-200 + 1.0).abs() < 1e-12);
    let dynamic =
        DynamicJet::variable(1e200, 0, 2).unwrap() / DynamicJet::variable(1e200, 1, 2).unwrap();
    assert!((dynamic.partials().unwrap()[0] / 1e-200 - 1.0).abs() < 1e-12);
    let angle = DynamicJet::variable(1e200, 0, 2)
        .unwrap()
        .atan2(&DynamicJet::variable(1e200, 1, 2).unwrap());
    assert!((angle.partials().unwrap()[0] / 5e-201 - 1.0).abs() < 1e-12);
    let atan = DynamicJet::variable(1e155, 0, 1).unwrap().atan();
    assert!(atan.partials().unwrap()[0] > 0.0);
    assert!((atan.partials().unwrap()[0] / 1e-310 - 1.0).abs() < 1e-12);
    assert_eq!(
        DynamicJet::variable(1.0, 0, 1)
            .unwrap()
            .powi(i32::MIN)
            .partials()
            .unwrap()[0],
        i32::MIN as f64
    );
    assert_eq!(
        Jet::<1>::variable(1.0, 0).powi(i32::MIN).derivative[0],
        i32::MIN as f64
    );
    let origin = DynamicJet::variable(0.0, 0, 1).unwrap();
    assert_eq!(
        origin
            .atan2(&DynamicJet::constant(0.0))
            .partials()
            .unwrap_err(),
        Error::NonFinite
    );
    let fixed_bad = AutoDiff::<1, 1, _>(|x: [Jet<1>; 1]| [x[0].sqrt()]);
    assert_eq!(
        fixed_bad.evaluate(
            &DVector::from_element(1, -1.0),
            &mut DVector::zeros(1),
            &mut DMatrix::zeros(1, 1)
        ),
        Err(Error::NonFinite)
    );
}
#[test]
fn runtime_rosenbrock_solves_with_each_chunk_size() {
    for chunk in [1, 2, 32] {
        let p = DynamicAutoDiff::new(2, 2, |x: &[DynamicJet]| {
            Ok(vec![10.0 * (&x[1] - x[0].powi(2)), 1.0 - &x[0]])
        })
        .unwrap()
        .with_chunk_size(chunk)
        .unwrap();
        let result = solve(
            &p,
            DVector::from_vec(vec![-1.2, 1.0]),
            &Options {
                max_iterations: 200,
                ..Options::default()
            },
        )
        .unwrap();
        assert!(result.cost < 1e-16);
        assert!((&result.x - DVector::from_element(2, 1.0)).norm() < 1e-7);
    }
}
#[test]
fn euclidean_autodiff_factors_support_whitening_joint_losses_and_backends() {
    let mut graph = FactorGraph::new();
    let anchor = graph
        .add_variable(VariableKind::Euclidean(1), DVector::from_element(1, 2.0))
        .unwrap();
    let x = graph
        .add_variable(VariableKind::Euclidean(2), DVector::zeros(2))
        .unwrap();
    let y = graph
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    graph.set_fixed(anchor, true).unwrap();
    graph
        .add_autodiff_factor_with_options(
            &[anchor, x, y],
            3,
            1,
            FactorOptions {
                noise: NoiseModel::from_covariance(DMatrix::from_row_slice(
                    3,
                    3,
                    &[2.0, 0.2, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0, 1.5],
                ))
                .unwrap(),
                loss: Some(Loss::Huber(1.0)),
            },
            |b| {
                Ok(vec![
                    &b[1][0] - &b[0][0],
                    &b[1][1] - 3.0,
                    &b[2][0] - (&b[1][0] + &b[1][1]),
                ])
            },
        )
        .unwrap();
    let (retained, block_size) = graph.schur_layout(&[y]).unwrap();
    for linear_solver in [
        LinearSolver::Dense,
        LinearSolver::SparseCg(CgOptions::default()),
        LinearSolver::BlockSchur {
            retained,
            block_size,
            cg: CgOptions::default(),
        },
    ] {
        let report = solve_local(
            &graph,
            graph.initial(),
            &Options {
                linear_solver,
                ..Options::default()
            },
        )
        .unwrap();
        assert!(report.cost < 1e-16);
        assert_eq!(graph.value(&report.x, anchor).unwrap()[0], 2.0);
        assert!((graph.value(&report.x, y).unwrap()[0] - 5.0).abs() < 1e-7);
    }
    graph.set_fixed(anchor, false).unwrap();
    let mut r = DVector::zeros(3);
    let mut j = DMatrix::zeros(3, 4);
    graph
        .evaluate_local(&graph.initial(), &mut r, &mut j)
        .unwrap();
    assert!(j.column(0).norm() > 0.0);
}
#[test]
fn autodiff_factor_rejects_invalid_chunks_and_preserves_registration_state() {
    let mut graph = FactorGraph::new();
    let pose = graph
        .add_variable(
            VariableKind::SE3,
            ropt::factor::se3_values(SE3::exp(Tangent6::zeros())),
        )
        .unwrap();
    assert!(matches!(
        graph.add_autodiff_factor_with_options(&[pose], 1, 0, FactorOptions::default(), |_| Ok(
            vec![DynamicJet::constant(1.0)]
        )),
        Err(Error::InvalidOptions)
    ));
    assert_eq!(graph.dimensions(), (7, 6, 0));
}
