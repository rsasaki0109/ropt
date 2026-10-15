use ropt::{factor::*, noise::NoiseModel, *};

#[test]
fn correlated_covariance_whitens_residuals_and_jacobians_consistently() {
    let covariance = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 2.0]);
    let precision = covariance.clone().cholesky().unwrap();
    let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, -1.0]);
    let b = DVector::from_vec(vec![1.0, 2.0]);
    let initial = DVector::from_vec(vec![0.3, -0.4]);
    let mut graph = FactorGraph::new();
    let id = graph
        .add_variable(VariableKind::Euclidean(2), initial.clone())
        .unwrap();
    let ac = a.clone();
    let bc = b.clone();
    graph
        .add_analytic_factor_with_options(
            &[id],
            2,
            FactorOptions {
                noise: NoiseModel::from_covariance(covariance).unwrap(),
                loss: Some(Loss::Squared),
            },
            move |v| Ok((&ac * &v[0] - &bc, vec![ac.clone()])),
        )
        .unwrap();
    let mut r = DVector::zeros(2);
    let mut j = DMatrix::zeros(2, 2);
    graph.evaluate_local(&initial, &mut r, &mut j).unwrap();
    let raw = &a * &initial - &b;
    assert!((r.norm_squared() - raw.dot(&precision.solve(&raw))).abs() < 1e-12);
    assert!((j.transpose() * &j - a.transpose() * precision.solve(&a)).norm() < 1e-12);
    assert!((j.transpose() * r - a.transpose() * precision.solve(&raw)).norm() < 1e-12);
    let gn = solve_local(
        &graph,
        initial.clone(),
        &Options {
            algorithm: Algorithm::GaussNewton,
            ..Options::default()
        },
    )
    .unwrap();
    assert!((&a * &gn.x - &b).norm() < 1e-12);
    let report = solve_local(
        &graph,
        initial,
        &Options {
            loss: Loss::Cauchy(0.1),
            max_iterations: 1,
            ..Options::default()
        },
    )
    .unwrap();
    assert!((report.initial_cost - 0.5 * raw.dot(&precision.solve(&raw))).abs() < 1e-12);
}
#[test]
fn heterogeneous_sigmas_produce_expected_weighted_estimate() {
    let mut graph = FactorGraph::new();
    let id = graph
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    for (observation, sigma) in [(0.0, 1.0), (10.0, 10.0)] {
        graph
            .add_analytic_factor_with_options(
                &[id],
                1,
                FactorOptions {
                    noise: NoiseModel::isotropic(sigma).unwrap(),
                    loss: Some(Loss::Squared),
                },
                move |v| {
                    Ok((
                        v[0].clone() - DVector::from_element(1, observation),
                        vec![DMatrix::identity(1, 1)],
                    ))
                },
            )
            .unwrap();
    }
    for linear_solver in [
        LinearSolver::Dense,
        LinearSolver::SparseCg(ropt::sparse::CgOptions::default()),
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
        assert!((report.x[0] - 10.0 / 101.0).abs() < 1e-8);
    }
}
#[test]
fn joint_huber_cost_is_not_a_sum_of_component_costs_or_applied_twice() {
    let mut graph = FactorGraph::new();
    let id = graph
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    graph.set_fixed(id, true).unwrap();
    graph
        .add_analytic_factor_with_options(
            &[id],
            2,
            FactorOptions {
                loss: Some(Loss::Huber(1.0)),
                ..FactorOptions::default()
            },
            |_| {
                Ok((
                    DVector::from_vec(vec![3.0, 4.0]),
                    vec![DMatrix::zeros(2, 1)],
                ))
            },
        )
        .unwrap();
    let report = solve_local(
        &graph,
        graph.initial(),
        &Options {
            loss: Loss::Cauchy(0.1),
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(report.cost, 4.5);
    let mut scalar = FactorGraph::new();
    let id = scalar
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    scalar.set_fixed(id, true).unwrap();
    scalar
        .add_analytic_factor(&[id], 2, |_| {
            Ok((
                DVector::from_vec(vec![3.0, 4.0]),
                vec![DMatrix::zeros(2, 1)],
            ))
        })
        .unwrap();
    assert_eq!(
        solve_local(
            &scalar,
            scalar.initial(),
            &Options {
                loss: Loss::Huber(1.0),
                ..Options::default()
            }
        )
        .unwrap()
        .cost,
        6.0
    );
}
fn huber_cost(x: f64) -> f64 {
    let norm = ((x - 1.0).powi(2) + (x + 1.5).powi(2)).sqrt();
    0.5 * (norm - 0.25)
}
#[test]
fn whitened_joint_loss_gradient_and_lm_gain_match_true_objective() {
    let mut graph = FactorGraph::new();
    let id = graph
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    graph
        .add_analytic_factor_with_options(
            &[id],
            2,
            FactorOptions {
                noise: NoiseModel::diagonal(DVector::from_vec(vec![1.0, 2.0])).unwrap(),
                loss: Some(Loss::Huber(0.5)),
            },
            |v| {
                Ok((
                    DVector::from_vec(vec![v[0][0] - 1.0, 2.0 * v[0][0] + 3.0]),
                    vec![DMatrix::from_vec(2, 1, vec![1.0, 2.0])],
                ))
            },
        )
        .unwrap();
    let report = solve_local(
        &graph,
        graph.initial(),
        &Options {
            max_iterations: 1,
            loss: Loss::Cauchy(0.01),
            ..Options::default()
        },
    )
    .unwrap();
    let iteration = &report.iterations[0];
    let weight = 0.5 / 3.25_f64.sqrt();
    let gradient = weight * 0.5;
    let hessian = 2.0 * weight;
    let expected_step = -gradient / (hessian + 1e-3 * hessian.max(1.0));
    assert!((report.x[0] - expected_step).abs() < 1e-12);
    let h = 1e-6;
    assert!((gradient - (huber_cost(h) - huber_cost(-h)) / (2.0 * h)).abs() < 1e-9);
    assert!((iteration.cost - huber_cost(0.0)).abs() < 1e-12);
    assert!((report.cost - huber_cost(report.x[0])).abs() < 1e-12);
    let predicted = -gradient * expected_step - 0.5 * hessian * expected_step.powi(2);
    assert!(
        (iteration.gain_ratio - (huber_cost(0.0) - huber_cost(report.x[0])) / predicted).abs()
            < 1e-12
    );
}
#[test]
fn analytic_and_numeric_whitening_share_sparse_and_dense_results() {
    let noise =
        NoiseModel::from_covariance(DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 2.0])).unwrap();
    let options = FactorOptions {
        noise,
        loss: Some(Loss::Cauchy(2.0)),
    };
    let mut a = FactorGraph::new();
    let id = a
        .add_variable(VariableKind::Euclidean(1), DVector::from_element(1, 0.3))
        .unwrap();
    a.add_analytic_factor_with_options(&[id], 2, options.clone(), |v| {
        Ok((
            DVector::from_vec(vec![v[0][0].sin(), v[0][0].exp()]),
            vec![DMatrix::from_vec(2, 1, vec![v[0][0].cos(), v[0][0].exp()])],
        ))
    })
    .unwrap();
    let mut b = FactorGraph::new();
    let id = b
        .add_variable(VariableKind::Euclidean(1), a.initial())
        .unwrap();
    b.add_numeric_factor_with_options(&[id], 2, 1e-6, options, |v| {
        Ok(DVector::from_vec(vec![v[0][0].sin(), v[0][0].exp()]))
    })
    .unwrap();
    let mut ar = DVector::zeros(2);
    let mut aj = DMatrix::zeros(2, 1);
    let mut br = DVector::zeros(2);
    let mut bj = DMatrix::zeros(2, 1);
    a.evaluate_local(&a.initial(), &mut ar, &mut aj).unwrap();
    b.evaluate_local(&b.initial(), &mut br, &mut bj).unwrap();
    assert!((&aj - &bj).norm() < 1e-8);
    assert!((&ar - &br).norm() < 1e-12);
    let (sr, sj) = a.evaluate_sparse(&a.initial()).unwrap();
    assert_eq!(sr, ar);
    assert!((sj.multiply(&DVector::from_element(1, 1.0)).unwrap() - aj.column(0)).norm() < 1e-12);
}
#[test]
fn invalid_noise_and_loss_registration_do_not_modify_the_graph() {
    assert!(NoiseModel::isotropic(0.0).is_err());
    assert!(NoiseModel::isotropic(f64::NAN).is_err());
    assert!(NoiseModel::diagonal(DVector::from_vec(vec![1.0, -2.0])).is_err());
    assert!(
        NoiseModel::from_covariance(DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0])).is_err()
    );
    assert!(
        NoiseModel::from_covariance(DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.0, 1.0])).is_err()
    );
    assert!(
        NoiseModel::from_covariance(DMatrix::from_row_slice(2, 2, &[1e-20, 0.5e-20, 0.0, 1e-20]))
            .is_err()
    );
    let mut graph = FactorGraph::new();
    let id = graph
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    let options = FactorOptions {
        noise: NoiseModel::diagonal(DVector::from_vec(vec![1.0, 1.0])).unwrap(),
        loss: None,
    };
    assert_eq!(
        graph.add_analytic_factor_with_options(&[id], 1, options, |_| unreachable!()),
        Err(Error::InvalidDimensions)
    );
    assert_eq!(
        graph.add_analytic_factor_with_options(
            &[id],
            1,
            FactorOptions {
                loss: Some(Loss::Huber(0.0)),
                ..FactorOptions::default()
            },
            |_| unreachable!()
        ),
        Err(Error::InvalidOptions)
    );
    assert_eq!(graph.dimensions(), (1, 1, 0));
}
#[test]
fn invalid_custom_loss_block_metadata_is_rejected() {
    struct Invalid(Vec<LossBlock>);
    impl LocalProblem for Invalid {
        fn dimensions(&self) -> (usize, usize, usize) {
            (1, 1, 2)
        }
        fn evaluate_local(
            &self,
            _: &DVector<f64>,
            r: &mut DVector<f64>,
            j: &mut DMatrix<f64>,
        ) -> Result<(), Error> {
            r.fill(1.0);
            j.fill(1.0);
            Ok(())
        }
        fn retract(&self, x: &DVector<f64>, d: &DVector<f64>) -> Result<DVector<f64>, Error> {
            Ok(x + d)
        }
        fn loss_blocks(&self) -> Vec<LossBlock> {
            self.0.clone()
        }
    }
    for blocks in [
        vec![LossBlock {
            rows: 0..0,
            loss: Loss::Squared,
        }],
        vec![
            LossBlock {
                rows: 0..2,
                loss: Loss::Squared,
            },
            LossBlock {
                rows: 1..2,
                loss: Loss::Squared,
            },
        ],
        vec![LossBlock {
            rows: 0..3,
            loss: Loss::Squared,
        }],
    ] {
        assert_eq!(
            solve_local(&Invalid(blocks), DVector::zeros(1), &Options::default()).unwrap_err(),
            Error::InvalidDimensions
        );
    }
}

#[path = "../examples/support/robust_ba.rs"]
mod robust_ba;
#[test]
fn heterogeneous_outlier_ba_improves_inlier_fit_and_all_backends_agree() {
    let baseline = robust_ba::scene(Loss::Squared).unwrap();
    let plain = solve_local(
        &baseline.graph,
        baseline.graph.initial(),
        &Options {
            max_iterations: 200,
            ..Options::default()
        },
    )
    .unwrap();
    let baseline_rms = baseline.inlier_rms(&plain.x).unwrap();
    assert!(baseline_rms > 3.0);
    let scene = robust_ba::scene(Loss::Huber(2.0)).unwrap();
    let (retained, block_size) = scene.graph.schur_layout(&scene.points).unwrap();
    let mut reports = Vec::new();
    for linear_solver in [
        LinearSolver::Dense,
        LinearSolver::SparseCg(ropt::sparse::CgOptions::default()),
        LinearSolver::BlockSchur {
            retained,
            block_size,
            cg: ropt::sparse::CgOptions::default(),
        },
    ] {
        let report = solve_local(
            &scene.graph,
            scene.graph.initial(),
            &Options {
                linear_solver,
                max_iterations: 200,
                ..Options::default()
            },
        )
        .unwrap();
        let rms = scene.inlier_rms(&report.x).unwrap();
        assert!(rms < 1.0);
        assert!(rms < baseline_rms * 0.25);
        assert_ne!(report.termination, Termination::MaxIterations);
        reports.push(report);
    }
    for report in &reports[1..] {
        assert!((&report.x - &reports[0].x).norm() < 1e-5);
        assert!((report.cost - reports[0].cost).abs() < 1e-8);
    }
}
#[test]
fn mixed_squared_prior_and_huber_measurements_agree_across_backends() {
    let mut graph = FactorGraph::new();
    let x = graph
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    let y = graph
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    graph
        .add_analytic_factor_with_options(
            &[x],
            1,
            FactorOptions {
                noise: NoiseModel::isotropic(0.1).unwrap(),
                loss: Some(Loss::Squared),
            },
            |v| Ok((v[0].clone(), vec![DMatrix::identity(1, 1)])),
        )
        .unwrap();
    for measured in [1.0, 1.0, 100.0] {
        graph
            .add_analytic_factor_with_options(
                &[x, y],
                1,
                FactorOptions {
                    loss: Some(Loss::Huber(1.0)),
                    ..FactorOptions::default()
                },
                move |v| {
                    Ok((
                        &v[1] - &v[0] - DVector::from_element(1, measured),
                        vec![-DMatrix::identity(1, 1), DMatrix::identity(1, 1)],
                    ))
                },
            )
            .unwrap();
    }
    let (retained, block_size) = graph.schur_layout(&[y]).unwrap();
    for linear_solver in [
        LinearSolver::Dense,
        LinearSolver::SparseCg(ropt::sparse::CgOptions::default()),
        LinearSolver::BlockSchur {
            retained,
            block_size,
            cg: ropt::sparse::CgOptions::default(),
        },
    ] {
        let report = solve_local(
            &graph,
            graph.initial(),
            &Options {
                linear_solver,
                loss: Loss::Cauchy(0.001),
                ..Options::default()
            },
        )
        .unwrap();
        assert!(report.x[0].abs() < 1e-10);
        assert!((report.x[1] - 1.5).abs() < 1e-6);
    }
}

#[test]
fn explicit_blocks_and_scalar_fallback_cover_middle_and_trailing_rows() {
    let mut graph = FactorGraph::new();
    let id = graph
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    graph.set_fixed(id, true).unwrap();
    graph
        .add_analytic_factor_with_options(
            &[id],
            1,
            FactorOptions {
                loss: Some(Loss::Squared),
                ..FactorOptions::default()
            },
            |_| Ok((DVector::from_element(1, 2.0), vec![DMatrix::zeros(1, 1)])),
        )
        .unwrap();
    graph
        .add_analytic_factor(&[id], 2, |_| {
            Ok((
                DVector::from_vec(vec![3.0, 4.0]),
                vec![DMatrix::zeros(2, 1)],
            ))
        })
        .unwrap();
    graph
        .add_analytic_factor_with_options(
            &[id],
            2,
            FactorOptions {
                loss: Some(Loss::Cauchy(1.0)),
                ..FactorOptions::default()
            },
            |_| Ok((DVector::zeros(2), vec![DMatrix::zeros(2, 1)])),
        )
        .unwrap();
    graph
        .add_analytic_factor(&[id], 1, |_| {
            Ok((DVector::from_element(1, -5.0), vec![DMatrix::zeros(1, 1)]))
        })
        .unwrap();
    assert_eq!(
        solve_local(
            &graph,
            graph.initial(),
            &Options {
                loss: Loss::Huber(1.0),
                ..Options::default()
            }
        )
        .unwrap()
        .cost,
        12.5
    );
}
#[test]
fn joint_robust_solution_is_invariant_under_orthogonal_whitening() {
    let rotation = DMatrix::from_row_slice(2, 2, &[0.8, -0.6, 0.6, 0.8]);
    let mut results = Vec::new();
    for noise in [
        NoiseModel::unit(),
        NoiseModel::sqrt_information(rotation).unwrap(),
    ] {
        let mut graph = FactorGraph::new();
        let id = graph
            .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
            .unwrap();
        graph
            .add_analytic_factor_with_options(
                &[id],
                2,
                FactorOptions {
                    noise,
                    loss: Some(Loss::Huber(0.5)),
                },
                |v| {
                    Ok((
                        DVector::from_vec(vec![v[0][0] - 1.0, 2.0 * v[0][0] + 3.0]),
                        vec![DMatrix::from_vec(2, 1, vec![1.0, 2.0])],
                    ))
                },
            )
            .unwrap();
        results.push(solve_local(&graph, graph.initial(), &Options::default()).unwrap());
    }
    assert!((results[0].x[0] + 1.0).abs() < 1e-8);
    assert!((&results[0].x - &results[1].x).norm() < 1e-8);
    assert!((results[0].cost - results[1].cost).abs() < 1e-12);
}
