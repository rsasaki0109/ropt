use ropt::{factor::*, *};
use std::{cell::Cell, rc::Rc};

fn evaluate(graph: &FactorGraph, sparse: bool) -> Result<(), Error> {
    let initial = graph.initial();
    if sparse {
        graph.evaluate_sparse(&initial).map(|_| ())
    } else {
        let (_, n, m) = graph.dimensions();
        graph.evaluate_local(&initial, &mut DVector::zeros(m), &mut DMatrix::zeros(m, n))
    }
}

#[test]
fn invalid_positive_sample_is_reported_before_a_later_callback_error() {
    for sparse in [false, true] {
        for behavior in 0..3 {
            let fault = Rc::new(Cell::new(true));
            let calls = Rc::new(Cell::new(0));
            let mut graph = FactorGraph::new();
            let variable = graph
                .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
                .unwrap();
            let (active, count) = (fault.clone(), calls.clone());
            graph
                .add_numeric_factor(&[variable], 1, 1e-6, move |values| {
                    count.set(count.get() + 1);
                    let x = values[0][0];
                    if active.get() && x > 0. {
                        return Ok(match behavior {
                            0 => DVector::zeros(2),
                            1 => DVector::from_element(1, f64::NAN),
                            _ => DVector::from_element(1, f64::INFINITY),
                        });
                    }
                    if active.get() && x < 0. {
                        return Err(Error::Evaluation("later negative sample failed".into()));
                    }
                    Ok(DVector::from_element(1, x - 2.))
                })
                .unwrap();
            let expected = if behavior == 0 {
                Error::InvalidDimensions
            } else {
                Error::NonFinite
            };
            assert_eq!(evaluate(&graph, sparse), Err(expected));
            assert_eq!(calls.get(), 2); // Base and positive sample only.
            assert_eq!(graph.initial(), DVector::zeros(1));
            assert_eq!(graph.dimensions(), (1, 1, 1));

            // Failure does not poison a subsequent valid graph evaluation.
            fault.set(false);
            evaluate(&graph, sparse).unwrap();
            let report = solve_local(&graph, graph.initial(), &Options::default()).unwrap();
            assert!((report.x[0] - 2.).abs() < 1e-8);
            assert!(report.cost < 1e-16);
        }
    }
}

#[test]
fn overflowing_derivative_stops_before_evaluating_the_next_column() {
    for sparse in [false, true] {
        let fault = Rc::new(Cell::new(true));
        let calls = Rc::new(Cell::new(0));
        let mut graph = FactorGraph::new();
        let variable = graph
            .add_variable(VariableKind::Euclidean(2), DVector::zeros(2))
            .unwrap();
        let (active, count) = (fault.clone(), calls.clone());
        graph
            .add_numeric_factor(&[variable], 2, 1e-6, move |values| {
                count.set(count.get() + 1);
                let (x, y) = (values[0][0], values[0][1]);
                if active.get() {
                    if y != 0. {
                        return Err(Error::Evaluation("later derivative column failed".into()));
                    }
                    return Ok(DVector::from_vec(vec![
                        if x == 0. { 0. } else { f64::MAX * x.signum() },
                        0.,
                    ]));
                }
                Ok(DVector::from_vec(vec![x + 2. * y - 5., 2. * x - y]))
            })
            .unwrap();
        assert_eq!(evaluate(&graph, sparse), Err(Error::NonFinite));
        assert_eq!(calls.get(), 3); // Base, then both samples of the first column.
        assert_eq!(graph.initial(), DVector::zeros(2));
        assert_eq!(graph.dimensions(), (2, 2, 2));

        fault.set(false);
        let expected = DMatrix::from_row_slice(2, 2, &[1., 2., 2., -1.]);
        let mut residual = DVector::zeros(2);
        let mut jacobian = DMatrix::zeros(2, 2);
        graph
            .evaluate_local(&graph.initial(), &mut residual, &mut jacobian)
            .unwrap();
        assert_eq!(residual, DVector::from_vec(vec![-5., 0.]));
        assert!((&jacobian - &expected).norm() < 1e-8);
        let (sparse_residual, sparse_jacobian) = graph.evaluate_sparse(&graph.initial()).unwrap();
        assert_eq!(sparse_residual, residual);
        for column in 0..2 {
            let basis = DVector::from_fn(2, |row, _| if row == column { 1. } else { 0. });
            assert!(
                (sparse_jacobian.multiply(&basis).unwrap() - expected.column(column)).norm() < 1e-8
            );
        }
        for linear_solver in [
            LinearSolver::Dense,
            LinearSolver::SparseQr(SparseQrOptions::default()),
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
            assert!((report.x - DVector::from_vec(vec![1., 2.])).norm() < 1e-8);
            assert!(report.cost < 1e-16);
        }
    }
}
