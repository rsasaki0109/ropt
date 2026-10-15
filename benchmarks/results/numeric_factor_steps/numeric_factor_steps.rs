use nalgebra::Vector3;
use ropt::{factor::*, manifold::*, *};
use std::{cell::Cell, rc::Rc};

#[test]
fn overflowing_denominator_is_rejected_without_registering_a_factor() {
    for with_options in [false, true] {
        let mut graph = FactorGraph::new();
        let variable = graph
            .add_variable(VariableKind::Euclidean(1), DVector::from_element(1, 1.))
            .unwrap();
        graph
            .add_analytic_factor(&[variable], 1, |values| {
                Ok((
                    DVector::from_element(1, values[0][0] - 2.),
                    vec![DMatrix::from_element(1, 1, 1.)],
                ))
            })
            .unwrap();
        let calls = Rc::new(Cell::new(0));
        for step in [1e308, f64::MAX] {
            let count = calls.clone();
            let callback = move |_: &[DVector<f64>]| {
                count.set(count.get() + 1);
                Ok(DVector::zeros(1))
            };
            let result = if with_options {
                graph.add_numeric_factor_with_options(
                    &[variable],
                    1,
                    step,
                    FactorOptions::default(),
                    callback,
                )
            } else {
                graph.add_numeric_factor(&[variable], 1, step, callback)
            };
            assert_eq!(result, Err(Error::InvalidOptions));
            assert_eq!(graph.dimensions(), (1, 1, 1));
            assert_eq!(graph.initial()[0], 1.);
        }
        // A valid retry averages the two observations 2 and 4, giving x = 3.
        graph
            .add_numeric_factor(&[variable], 1, 1e-6, |values| {
                Ok(DVector::from_element(1, values[0][0] - 4.))
            })
            .unwrap();
        let report = solve_local(&graph, graph.initial(), &Options::default()).unwrap();
        assert!((report.x[0] - 3.).abs() < 1e-8);
        assert_eq!(calls.get(), 0);
    }
}

#[test]
fn rounded_away_active_perturbations_cannot_establish_convergence() {
    let cases = vec![
        (
            VariableKind::Euclidean(1),
            DVector::from_element(1, 1.),
            1e-30,
        ),
        // Only the positive perturbation rounds away at this power-of-two boundary.
        (
            VariableKind::Euclidean(1),
            DVector::from_element(1, 1.),
            3. * f64::EPSILON / 8.,
        ),
        (VariableKind::SO2, DVector::from_element(1, 1.), 1e-30),
        // Retraction canonicalizes this angle; both samples coincide even though
        // their storage differs from the original unwrapped angle.
        (VariableKind::SO2, DVector::from_element(1, 10.), 1e-30),
        (
            VariableKind::SO3,
            so3_values(SO3::exp(Vector3::new(0.2, -0.1, 0.3))),
            1e-30,
        ),
        (
            VariableKind::SE2,
            DVector::from_vec(vec![1., 2., 0.4]),
            1e-30,
        ),
        (
            VariableKind::SE2Cartesian,
            DVector::from_vec(vec![1., 2., 0.4]),
            1e-30,
        ),
        (
            VariableKind::SE3,
            se3_values(SE3::exp(Tangent6::from_row_slice(&[
                1., 2., 3., 0.2, -0.1, 0.3,
            ]))),
            1e-30,
        ),
    ];
    for (kind, initial, step) in cases {
        for sparse in [false, true] {
            let mut graph = FactorGraph::new();
            let variable = graph.add_variable(kind, initial.clone()).unwrap();
            let calls = Rc::new(Cell::new(0));
            let count = calls.clone();
            graph
                .add_numeric_factor(&[variable], 1, step, move |values| {
                    count.set(count.get() + 1);
                    Ok(DVector::from_element(1, values[0][0] - 2.))
                })
                .unwrap();
            let result = if sparse {
                graph.evaluate_sparse(&initial).map(|_| ())
            } else {
                let (_, n, m) = graph.dimensions();
                graph.evaluate_local(&initial, &mut DVector::zeros(m), &mut DMatrix::zeros(m, n))
            };
            assert_eq!(result, Err(Error::InvalidOptions), "{kind:?}, step {step}");
            assert_eq!(calls.get(), 1); // Base residual only; no perturbation callbacks.
            assert_eq!(graph.initial(), initial);
            calls.set(0);
            let linear_solver = if sparse {
                LinearSolver::SparseQr(SparseQrOptions::default())
            } else {
                LinearSolver::Dense
            };
            assert_eq!(
                solve_local(
                    &graph,
                    graph.initial(),
                    &Options {
                        linear_solver,
                        ..Options::default()
                    }
                )
                .unwrap_err(),
                Error::InvalidOptions
            );
            assert_eq!(calls.get(), 1);
        }
    }
}

#[test]
fn fixed_blocks_and_representable_tiny_steps_remain_valid() {
    for step in [1e-30, f64::from_bits(1)] {
        let mut graph = FactorGraph::new();
        let fixed = graph
            .add_variable(VariableKind::Euclidean(1), DVector::from_element(1, 1.))
            .unwrap();
        graph.set_fixed(fixed, true).unwrap();
        let active = graph
            .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
            .unwrap();
        let calls = Rc::new(Cell::new(0));
        let count = calls.clone();
        graph
            .add_numeric_factor(&[fixed, active], 1, step, move |values| {
                count.set(count.get() + 1);
                Ok(values[1].clone())
            })
            .unwrap();
        let mut residual = DVector::zeros(1);
        let mut jacobian = DMatrix::zeros(1, 1);
        graph
            .evaluate_local(&graph.initial(), &mut residual, &mut jacobian)
            .unwrap();
        assert_eq!(residual[0], 0.);
        assert_eq!(jacobian[(0, 0)], 1.);
        assert_eq!(calls.get(), 3);
        calls.set(0);
        let (r, j) = graph.evaluate_sparse(&graph.initial()).unwrap();
        assert_eq!(r[0], 0.);
        assert_eq!(j.entries().collect::<Vec<_>>(), vec![(0, 0, 1.)]);
        assert_eq!(calls.get(), 3);
        graph.set_fixed(active, true).unwrap();
        calls.set(0);
        let (r, j) = graph.evaluate_sparse(&graph.initial()).unwrap();
        assert_eq!(r[0], 0.);
        assert_eq!(j.shape(), (1, 0));
        assert_eq!(j.nnz(), 0);
        assert_eq!(calls.get(), 1);
    }
}
