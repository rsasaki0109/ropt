use ropt::{factor::*, sparse::*, *};

#[test]
fn csr_combines_duplicates_and_matches_dense_products() {
    let j = CsrMatrix::from_triplets(
        3,
        2,
        vec![
            (2, 1, 4.0),
            (0, 0, 2.0),
            (0, 0, 1.0),
            (1, 0, -2.0),
            (1, 1, 0.0),
            (2, 0, 1.0),
        ],
    )
    .unwrap();
    assert_eq!(j.nnz(), 4);
    assert_eq!(j.shape(), (3, 2));
    let dense = DMatrix::from_row_slice(3, 2, &[3.0, 0.0, -2.0, 0.0, 1.0, 4.0]);
    let x = DVector::from_vec(vec![2.0, -1.0]);
    let y = DVector::from_vec(vec![1.0, -3.0, 0.5]);
    assert_eq!(j.multiply(&x).unwrap(), &dense * &x);
    assert_eq!(j.transpose_multiply(&y).unwrap(), dense.transpose() * y);
    assert_eq!(
        j.normal_diagonal().unwrap(),
        DVector::from_vec(vec![14.0, 16.0])
    );
    assert!(CsrMatrix::from_triplets(2, 2, vec![(2, 0, 1.0)]).is_err());
    assert_eq!(
        CsrMatrix::from_triplets(1, 1, vec![(0, 0, f64::NAN)]).unwrap_err(),
        Error::NonFinite
    );
    assert_eq!(
        j.multiply(&DVector::zeros(1)).unwrap_err(),
        Error::InvalidDimensions
    );
    let zero = CsrMatrix::from_triplets(1, 1, vec![]).unwrap();
    assert_eq!(
        zero.multiply(&DVector::from_element(1, f64::NAN))
            .unwrap_err(),
        Error::NonFinite
    );
}
struct DenseOperator(DMatrix<f64>);
impl LinearOperator for DenseOperator {
    fn dimension(&self) -> usize {
        self.0.nrows()
    }
    fn apply(&self, x: &DVector<f64>) -> Result<DVector<f64>, Error> {
        Ok(&self.0 * x)
    }
    fn diagonal(&self) -> Result<DVector<f64>, Error> {
        Ok(self.0.diagonal())
    }
}
#[test]
fn pcg_matches_cholesky_and_verifies_true_residual() {
    let a = DMatrix::from_row_slice(3, 3, &[4.0, 1.0, 0.2, 1.0, 3.0, 0.5, 0.2, 0.5, 2.0]);
    let b = DVector::from_vec(vec![1.0, 2.0, -0.4]);
    let expected = a.clone().cholesky().unwrap().solve(&b);
    let report = conjugate_gradient(&DenseOperator(a.clone()), &b, &CgOptions::default()).unwrap();
    assert!((&report.x - expected).norm() < 1e-10);
    assert!((&b - &a * &report.x).norm() < 1e-10);
    assert!(report.residual_norm < 1e-10);
    assert!(report.iterations > 0);
    let zero =
        conjugate_gradient(&DenseOperator(a), &DVector::zeros(3), &CgOptions::default()).unwrap();
    assert_eq!(zero.iterations, 0);
}
#[test]
fn pcg_failure_modes_are_explicit() {
    let a = DenseOperator(DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 3.0]));
    let b = DVector::from_vec(vec![1.0, 0.0]);
    assert_eq!(
        conjugate_gradient(
            &a,
            &b,
            &CgOptions {
                max_iterations: 1,
                relative_tolerance: 0.0,
                absolute_tolerance: 1e-15
            }
        )
        .unwrap_err(),
        Error::LinearIterationLimit { iterations: 1 }
    );
    assert_eq!(
        conjugate_gradient(
            &a,
            &b,
            &CgOptions {
                max_iterations: 0,
                ..CgOptions::default()
            }
        )
        .unwrap_err(),
        Error::InvalidOptions
    );
    let indefinite = DenseOperator(DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]));
    assert_eq!(
        conjugate_gradient(
            &indefinite,
            &DVector::from_vec(vec![1.0, -1.0]),
            &CgOptions::default()
        )
        .unwrap_err(),
        Error::LinearSolve
    );
    let singular = DenseOperator(DMatrix::zeros(2, 2));
    assert_eq!(
        conjugate_gradient(&singular, &b, &CgOptions::default()).unwrap_err(),
        Error::LinearSolve
    );
}
#[test]
fn matrix_free_damped_normal_operator_matches_dense() {
    let j = CsrMatrix::from_triplets(
        3,
        2,
        vec![(0, 0, 2.0), (0, 1, 1.0), (1, 1, -1.0), (2, 0, 3.0)],
    )
    .unwrap();
    let dense = DMatrix::<f64>::from_row_slice(3, 2, &[2.0, 1.0, 0.0, -1.0, 3.0, 0.0]);
    let mut h = dense.transpose() * dense;
    for i in 0..2 {
        h[(i, i)] += 0.2 * h[(i, i)].max(1.0);
    }
    let op = NormalOperator::new(&j, 0.2).unwrap();
    let x = DVector::from_vec(vec![0.3, -0.8]);
    assert!((op.apply(&x).unwrap() - &h * x).norm() < 1e-12);
    assert_eq!(op.diagonal().unwrap(), h.diagonal());
}
fn chain(count: usize) -> FactorGraph {
    let mut g = FactorGraph::new();
    let mut ids = Vec::new();
    for _ in 0..count {
        ids.push(
            g.add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
                .unwrap(),
        );
    }
    g.set_fixed(ids[0], true).unwrap();
    for i in 1..count {
        g.add_analytic_factor(&[ids[i - 1], ids[i]], 1, |v| {
            Ok((
                &v[1] - &v[0] - DVector::from_element(1, 1.0),
                vec![-DMatrix::identity(1, 1), DMatrix::identity(1, 1)],
            ))
        })
        .unwrap();
    }
    g
}
#[test]
fn graph_sparse_assembly_and_lm_match_dense() {
    let graph = chain(15);
    let initial = graph.initial();
    let (r, j) = graph.evaluate_sparse(&initial).unwrap();
    let mut rd = DVector::zeros(r.len());
    let mut jd = DMatrix::zeros(r.len(), 14);
    graph.evaluate_local(&initial, &mut rd, &mut jd).unwrap();
    assert_eq!(r, rd);
    for (row, col, v) in j.entries() {
        assert_eq!(v, jd[(row, col)]);
    }
    assert_eq!(j.nnz(), 27);
    let dense = solve_local(&graph, initial.clone(), &Options::default()).unwrap();
    let sparse = solve_local(
        &graph,
        initial,
        &Options {
            linear_solver: LinearSolver::SparseCg(CgOptions::default()),
            ..Options::default()
        },
    )
    .unwrap();
    assert!((&dense.x - &sparse.x).norm() < 1e-6);
    assert!(sparse.cost < 1e-16);
    assert!(sparse.iterations.iter().any(|i| i.linear_iterations > 0));
}
#[test]
fn larger_sparse_chain_converges_with_linear_storage() {
    let graph = chain(300);
    let (_, j) = graph.evaluate_sparse(&graph.initial()).unwrap();
    assert_eq!(j.nnz(), 597);
    let report = solve_local(
        &graph,
        graph.initial(),
        &Options {
            linear_solver: LinearSolver::SparseCg(CgOptions {
                max_iterations: 1000,
                ..CgOptions::default()
            }),
            max_iterations: 100,
            ..Options::default()
        },
    )
    .unwrap();
    assert!(report.cost < 1e-12, "cost {}", report.cost);
    assert!((report.x[299] - 299.0).abs() < 1e-5);
    assert_eq!(report.x[0], 0.0);
}
#[test]
fn sparse_backend_does_not_silently_use_dense_or_gn() {
    let residual = CentralDifference {
        parameters: 1,
        residuals: 1,
        step: 1e-6,
        function: |v: &DVector<f64>| Ok(v.clone()),
    };
    let options = Options {
        linear_solver: LinearSolver::SparseCg(CgOptions::default()),
        ..Options::default()
    };
    assert!(matches!(
        solve(&residual, DVector::zeros(1), &options),
        Err(Error::Evaluation(_))
    ));
    assert_eq!(
        solve_local(
            &chain(3),
            DVector::zeros(3),
            &Options {
                algorithm: Algorithm::GaussNewton,
                ..options
            }
        )
        .unwrap_err(),
        Error::InvalidOptions
    );
}
#[test]
fn dense_schur_matches_full_system_and_rejects_invalid_input() {
    let j = DMatrix::from_row_slice(
        4,
        4,
        &[
            2.0, 1.0, 0.2, 0.0, 0.1, 3.0, 0.0, 0.3, 0.0, 0.1, 2.0, 0.5, 0.2, 0.0, 0.4, 1.5,
        ],
    );
    let a = j.transpose() * j + DMatrix::identity(4, 4) * 0.1;
    let b = DVector::from_vec(vec![1.0, -0.3, 0.8, 2.0]);
    let expected = a.clone().cholesky().unwrap().solve(&b);
    for k in 1..4 {
        let x = schur_solve(&a, &b, k).unwrap();
        assert!((&x - &expected).norm() < 1e-12);
        assert!((&a * x - &b).norm() < 1e-12);
    }
    assert_eq!(
        schur_solve(&a, &b, 0).unwrap_err(),
        Error::InvalidDimensions
    );
    assert_eq!(
        schur_solve(&DMatrix::zeros(4, 4), &b, 2).unwrap_err(),
        Error::LinearSolve
    );
    let mut asymmetric = a;
    asymmetric[(0, 1)] += 1.0;
    assert_eq!(
        schur_solve(&asymmetric, &b, 2).unwrap_err(),
        Error::LinearSolve
    );
}

#[test]
fn robust_losses_and_schur_backend_match_dense_lm() {
    let mut graph = FactorGraph::new();
    let id = graph
        .add_variable(VariableKind::Euclidean(2), DVector::zeros(2))
        .unwrap();
    graph
        .add_analytic_factor(&[id], 6, |v| {
            let r = DVector::from_vec(vec![
                v[0][0] - 1.0,
                v[0][0] - 1.0,
                v[0][0] - 100.0,
                v[0][1] + 2.0,
                v[0][1] + 2.0,
                v[0][1] + 2.0,
            ]);
            let j = DMatrix::from_row_slice(
                6,
                2,
                &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            );
            Ok((r, vec![j]))
        })
        .unwrap();
    for loss in [Loss::Squared, Loss::Huber(1.0), Loss::Cauchy(1.0)] {
        let options = Options {
            loss,
            ..Options::default()
        };
        let dense = solve_local(&graph, graph.initial(), &options).unwrap();
        for solver in [
            LinearSolver::SparseCg(CgOptions::default()),
            LinearSolver::DenseSchur { retained: 1 },
        ] {
            let report = solve_local(
                &graph,
                graph.initial(),
                &Options {
                    linear_solver: solver,
                    ..options.clone()
                },
            )
            .unwrap();
            assert!((&dense.x - &report.x).norm() < 1e-6);
            assert!((dense.cost - report.cost).abs() < 1e-8);
        }
    }
}
#[test]
fn sparse_rejections_and_iteration_limit_propagation() {
    let mut graph = FactorGraph::new();
    let id = graph
        .add_variable(VariableKind::Euclidean(1), DVector::from_element(1, 0.01))
        .unwrap();
    graph
        .add_analytic_factor(&[id], 1, |v| {
            Ok((
                DVector::from_element(1, v[0][0].powi(2) - 1.0),
                vec![DMatrix::from_element(1, 1, 2.0 * v[0][0])],
            ))
        })
        .unwrap();
    let options = Options {
        linear_solver: LinearSolver::SparseCg(CgOptions::default()),
        max_iterations: 1,
        initial_damping: 1e-12,
        ..Options::default()
    };
    let r = solve_local(&graph, graph.initial(), &options).unwrap();
    assert!(!r.iterations[0].accepted);
    assert_eq!(r.x, graph.initial());
    let graph = chain(10);
    assert_eq!(
        solve_local(
            &graph,
            graph.initial(),
            &Options {
                linear_solver: LinearSolver::SparseCg(CgOptions {
                    max_iterations: 1,
                    ..CgOptions::default()
                }),
                ..Options::default()
            }
        )
        .unwrap_err(),
        Error::LinearIterationLimit { iterations: 1 }
    );
    assert_eq!(
        solve_local(
            &graph,
            graph.initial(),
            &Options {
                linear_solver: LinearSolver::DenseSchur { retained: 0 },
                ..Options::default()
            }
        )
        .unwrap_err(),
        Error::InvalidOptions
    );
}
#[test]
fn csr_spd_operator_and_duplicate_cancellation() {
    let matrix = CsrMatrix::from_triplets(
        2,
        2,
        vec![
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
            (1, 1, 5.0),
            (1, 1, -5.0),
        ],
    )
    .unwrap();
    let b = DVector::from_vec(vec![1.0, 2.0]);
    let result = conjugate_gradient(&matrix, &b, &CgOptions::default()).unwrap();
    assert!((matrix.multiply(&result.x).unwrap() - b).norm() < 1e-12);
    let canceled = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 2.0), (0, 0, -2.0)]).unwrap();
    assert_eq!(canceled.nnz(), 0);
}

#[test]
fn damping_overflow_is_an_error_for_dense_and_sparse() {
    let mut graph = FactorGraph::new();
    let id = graph
        .add_variable(VariableKind::Euclidean(1), DVector::from_element(1, 10.0))
        .unwrap();
    graph
        .add_analytic_factor(&[id], 1, |v| {
            Ok((v[0].clone() * 2.0, vec![DMatrix::from_element(1, 1, 2.0)]))
        })
        .unwrap();
    for linear_solver in [
        LinearSolver::Dense,
        LinearSolver::SparseCg(CgOptions::default()),
    ] {
        assert_eq!(
            solve_local(
                &graph,
                graph.initial(),
                &Options {
                    linear_solver,
                    initial_damping: 1e308,
                    ..Options::default()
                }
            )
            .unwrap_err(),
            Error::NonFinite
        );
    }
}
