use ropt::{covariance::*, factor::*, fixed_lag::*, *};
use std::{cell::Cell, rc::Rc};

fn variable(key: u64) -> WindowVariable {
    WindowVariable {
        key,
        kind: VariableKind::Euclidean(1),
        value: DVector::zeros(1),
        timestamp: Some(key as f64),
        fixed: key == 0,
    }
}
fn observation(key: u64, target: f64) -> SmootherFactor {
    SmootherFactor::analytic(&[key], 1, Default::default(), move |v| {
        Ok((
            DVector::from_element(1, v[0][0] - target),
            vec![DMatrix::identity(1, 1)],
        ))
    })
}
fn odometry(a: u64, b: u64) -> SmootherFactor {
    SmootherFactor::analytic(&[a, b], 1, Default::default(), |v| {
        Ok((
            DVector::from_element(1, v[1][0] - v[0][0] - 1.),
            vec![-DMatrix::identity(1, 1), DMatrix::identity(1, 1)],
        ))
    })
}
fn full_history() -> (FactorGraph, Vec<VariableId>, Report) {
    let mut graph = FactorGraph::new();
    let ids = (0..5)
        .map(|_| {
            graph
                .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
                .unwrap()
        })
        .collect::<Vec<_>>();
    graph.set_fixed(ids[0], true).unwrap();
    for k in 1..5 {
        graph
            .add_analytic_factor(&[ids[k - 1], ids[k]], 1, |v| {
                Ok((
                    DVector::from_element(1, v[1][0] - v[0][0] - 1.),
                    vec![-DMatrix::identity(1, 1), DMatrix::identity(1, 1)],
                ))
            })
            .unwrap();
    }
    for target in [3., 5.] {
        graph
            .add_analytic_factor(&[ids[1]], 1, move |v| {
                Ok((
                    DVector::from_element(1, v[0][0] - target),
                    vec![DMatrix::identity(1, 1)],
                ))
            })
            .unwrap();
    }
    let solved = solve_local(
        &graph,
        graph.initial(),
        &Options {
            algorithm: Algorithm::GaussNewton,
            linear_solver: LinearSolver::DenseQr(Default::default()),
            ..Default::default()
        },
    )
    .unwrap();
    (graph, ids, solved)
}

#[test]
fn delayed_live_observations_match_full_history_and_retired_references_cannot_leak_updates() {
    let (batch, ids, solved) = full_history();
    for sparse_solver in [false, true] {
        for input in [CovarianceInput::Dense, CovarianceInput::SparseToDense] {
            let covariance_options = CovarianceOptions {
                input,
                ..Default::default()
            };
            let mut smoother = FixedLagSmoother::new(FixedLagOptions {
                lag: 1.5,
                solver: Options {
                    algorithm: Algorithm::GaussNewton,
                    linear_solver: if sparse_solver {
                        LinearSolver::SparseQr(Default::default())
                    } else {
                        LinearSolver::DenseQr(Default::default())
                    },
                    ..Default::default()
                },
                ..Default::default()
            })
            .unwrap();
            smoother
                .update(
                    1.,
                    SmootherUpdate {
                        variables: vec![variable(0), variable(1)],
                        factors: vec![odometry(0, 1)],
                        ..Default::default()
                    },
                )
                .unwrap();
            let first = smoother
                .update(
                    2.,
                    SmootherUpdate {
                        variables: vec![variable(2)],
                        factors: vec![odometry(1, 2), observation(1, 3.)],
                        ..Default::default()
                    },
                )
                .unwrap();
            assert_eq!(first.marginalized, vec![0]);
            assert!((first.estimates[&1][0] - 2.).abs() < 1e-9);
            // Key 1 is still live when the request arrives, but expires in this update.
            let (retired, covariance) = smoother
                .update_with_covariance(
                    3.,
                    SmootherUpdate {
                        variables: vec![variable(3)],
                        factors: vec![odometry(2, 3), observation(1, 5.)],
                        ..Default::default()
                    },
                    &[3, 1, 2],
                    &covariance_options,
                )
                .unwrap();
            assert_eq!(retired.marginalized, vec![1]);
            assert_eq!(retired.marginalized_timestamps[&1], 1.);
            assert!((retired.marginalized_estimates[&1][0] - 3.).abs() < 1e-9);
            let oracle = batch
                .marginal_covariance(&solved.x, &[ids[3], ids[1], ids[2]], &covariance_options)
                .unwrap();
            assert!((&covariance.matrix - oracle.matrix).norm() < 1e-11);
            let expected = DMatrix::from_row_slice(
                3,
                3,
                &[
                    7. / 3.,
                    1. / 3.,
                    4. / 3.,
                    1. / 3.,
                    1. / 3.,
                    1. / 3.,
                    4. / 3.,
                    1. / 3.,
                    4. / 3.,
                ],
            );
            assert!((&covariance.matrix - expected).norm() < 1e-11);
            for key in [2, 3] {
                assert!(
                    (retired.estimates[&key][0]
                        - batch.value(&solved.x, ids[key as usize]).unwrap()[0])
                        .abs()
                        < 1e-9
                );
            }
            assert!((retired.final_cost - solved.cost).abs() < 1e-9);
            let calls = Rc::new(Cell::new(0));
            let counter = calls.clone();
            let invalid = SmootherFactor::analytic(&[1], 1, Default::default(), move |v| {
                counter.set(counter.get() + 1);
                Ok((
                    DVector::from_element(1, v[0][0] - 100.),
                    vec![DMatrix::identity(1, 1)],
                ))
            });
            assert!(matches!(
                smoother.update(
                    4.,
                    SmootherUpdate {
                        variables: vec![variable(4)],
                        factors: vec![odometry(3, 4), invalid],
                        timestamps: vec![(2, 4.)]
                    }
                ),
                Err(Error::InvalidDimensions)
            ));
            assert_eq!(calls.get(), 0);
            assert_eq!(smoother.last_time(), Some(3.));
            assert_eq!(smoother.variable_count(), 2);
            assert!(smoother.estimate(4).is_none());
            for key in [2, 3] {
                assert_eq!(smoother.estimate(key), Some(&retired.estimates[&key]));
            }
            // Retry without the failed timestamp refresh: key 2 must expire normally.
            let (resumed, covariance) = smoother
                .update_with_covariance(
                    4.,
                    SmootherUpdate {
                        variables: vec![variable(4)],
                        factors: vec![odometry(3, 4)],
                        ..Default::default()
                    },
                    &[4, 2, 3],
                    &covariance_options,
                )
                .unwrap();
            assert_eq!(resumed.marginalized, vec![2]);
            assert_eq!(resumed.marginalized_timestamps[&2], 2.);
            let oracle = batch
                .marginal_covariance(&solved.x, &[ids[4], ids[2], ids[3]], &covariance_options)
                .unwrap();
            assert!((covariance.matrix - oracle.matrix).norm() < 1e-11);
            for key in [3, 4] {
                assert!(
                    (resumed.estimates[&key][0]
                        - batch.value(&solved.x, ids[key as usize]).unwrap()[0])
                        .abs()
                        < 1e-9
                );
            }
            assert!((resumed.final_cost - solved.cost).abs() < 1e-9);
            assert!((retired.marginalized_estimates[&1][0] - 3.).abs() < 1e-9);
        }
    }
}
