use ropt::{factor::*, fixed_lag::*, isam::*, *};
use std::{cell::Cell, rc::Rc};

fn observation(target: f64) -> SmootherFactor {
    SmootherFactor::analytic(&[0], 1, Default::default(), move |v| {
        Ok((
            DVector::from_element(1, v[0][0] - target),
            vec![DMatrix::identity(1, 1)],
        ))
    })
}
fn initial_variable() -> WindowVariable {
    WindowVariable {
        key: 0,
        kind: VariableKind::Euclidean(1),
        value: DVector::zeros(1),
        timestamp: Some(0.),
        fixed: false,
    }
}
fn register_bad(
    graph: &mut FactorGraph,
    id: VariableId,
    mode: usize,
    calls: Rc<Cell<usize>>,
) -> Result<(), Error> {
    match mode {
        0 => graph.add_analytic_factor(&[id], usize::MAX, move |_| {
            calls.set(calls.get() + 1);
            Err(Error::Evaluation("must not evaluate".into()))
        }),
        1 => graph.add_autodiff_factor(&[id], usize::MAX, move |_| {
            calls.set(calls.get() + 1);
            Err(Error::Evaluation("must not evaluate".into()))
        }),
        _ => graph.add_numeric_factor(&[id], usize::MAX, 1e-6, move |_| {
            calls.set(calls.get() + 1);
            Err(Error::Evaluation("must not evaluate".into()))
        }),
    }
}
fn bad_update(automatic: bool, calls: Rc<Cell<usize>>) -> SmootherUpdate {
    let bad = if automatic {
        SmootherFactor::autodiff(&[0], usize::MAX, 1, Default::default(), move |_| {
            calls.set(calls.get() + 1);
            Err(Error::Evaluation("must not evaluate".into()))
        })
    } else {
        SmootherFactor::analytic(&[0], usize::MAX, Default::default(), move |_| {
            calls.set(calls.get() + 1);
            Err(Error::Evaluation("must not evaluate".into()))
        })
    };
    SmootherUpdate {
        factors: vec![observation(4.), bad],
        ..Default::default()
    }
}

#[test]
fn oversized_factor_counts_are_rejected_without_mutating_graph_or_calling_callbacks() {
    for mode in 0..3 {
        for existing in [false, true] {
            let mut graph = FactorGraph::new();
            let id = graph
                .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
                .unwrap();
            if existing {
                graph
                    .add_analytic_factor(&[id], 1, |v| {
                        Ok((
                            DVector::from_element(1, v[0][0] - 2.),
                            vec![DMatrix::identity(1, 1)],
                        ))
                    })
                    .unwrap();
            }
            let dimensions = graph.dimensions();
            let calls = Rc::new(Cell::new(0));
            assert!(matches!(
                register_bad(&mut graph, id, mode, calls.clone()),
                Err(Error::InvalidDimensions)
            ));
            assert_eq!(graph.dimensions(), dimensions);
            assert_eq!(calls.get(), 0);
            graph
                .add_analytic_factor(&[id], 1, |v| {
                    Ok((
                        DVector::from_element(1, v[0][0] - 4.),
                        vec![DMatrix::identity(1, 1)],
                    ))
                })
                .unwrap();
            let solved = solve_local(&graph, graph.initial(), &Default::default()).unwrap();
            assert!((solved.x[0] - if existing { 3. } else { 4. }).abs() < 1e-7);
            assert_eq!(calls.get(), 0);
        }
    }
}

#[test]
fn residual_sum_and_csr_offset_overflow_leave_existing_metadata_unchanged() {
    let mut graph = FactorGraph::new();
    let id = graph
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    // Registration is metadata-only: never evaluate or allocate these residuals.
    graph
        .add_analytic_factor(&[id], usize::MAX - 2, |_| unreachable!())
        .unwrap();
    let dimensions = graph.dimensions();
    for rows in [2, 3] {
        assert!(matches!(
            graph.add_analytic_factor(&[id], rows, |_| unreachable!()),
            Err(Error::InvalidDimensions)
        ));
        assert_eq!(graph.dimensions(), dimensions);
    }
}

#[test]
fn overflow_rejection_preserves_fixed_lag_and_isam_state_then_retries_cleanly() {
    for automatic in [false, true] {
        let calls = Rc::new(Cell::new(0));
        let mut lag = FixedLagSmoother::new(FixedLagOptions {
            lag: 1.,
            ..Default::default()
        })
        .unwrap();
        lag.update(
            0.,
            SmootherUpdate {
                variables: vec![initial_variable()],
                factors: vec![observation(2.)],
                ..Default::default()
            },
        )
        .unwrap();
        let old = lag.estimate(0).unwrap().clone();
        let mut bad = bad_update(automatic, calls.clone());
        bad.timestamps.push((0, 2.));
        assert!(matches!(lag.update(2., bad), Err(Error::InvalidDimensions)));
        assert_eq!(lag.last_time(), Some(0.));
        assert_eq!(lag.estimate(0), Some(&old));
        assert_eq!(lag.factor_count(), 1);
        assert_eq!(calls.get(), 0);
        let retired = lag
            .update(
                2.,
                SmootherUpdate {
                    factors: vec![observation(4.)],
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(retired.marginalized, vec![0]);
        assert_eq!(retired.marginalized_timestamps[&0], 0.);
        assert!((retired.marginalized_estimates[&0][0] - 3.).abs() < 1e-7);
        let mut isam = IsamSmoother::new(Default::default()).unwrap();
        isam.update(
            SmootherUpdate {
                variables: vec![initial_variable()],
                factors: vec![observation(2.)],
                ..Default::default()
            },
            false,
        )
        .unwrap();
        let old = isam.estimate(0).unwrap().clone();
        assert!(matches!(
            isam.update(bad_update(automatic, calls.clone()), false),
            Err(Error::InvalidDimensions)
        ));
        assert_eq!(isam.estimate(0), Some(&old));
        assert_eq!(isam.factor_count(), 1);
        assert_eq!(calls.get(), 0);
        let retried = isam
            .update(
                SmootherUpdate {
                    factors: vec![observation(4.)],
                    ..Default::default()
                },
                true,
            )
            .unwrap();
        assert!((retried.estimates[&0][0] - 3.).abs() < 1e-7);
        let next = isam.update(Default::default(), false).unwrap();
        assert!((next.estimates[&0][0] - 3.).abs() < 1e-7);
    }
}
