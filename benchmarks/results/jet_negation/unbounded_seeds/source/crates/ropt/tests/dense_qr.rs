use ropt::{
    autodiff::{AutoDiff, Jet},
    dense::qr_step,
    factor::*,
    noise::NoiseModel,
    *,
};
struct Linear {
    j: DMatrix<f64>,
    target: DVector<f64>,
}
impl Residual for Linear {
    fn dimensions(&self) -> (usize, usize) {
        (self.j.ncols(), self.j.nrows())
    }
    fn evaluate(
        &self,
        x: &DVector<f64>,
        r: &mut DVector<f64>,
        j: &mut DMatrix<f64>,
    ) -> Result<(), Error> {
        *r = &self.j * (x - &self.target);
        *j = self.j.clone();
        Ok(())
    }
}
#[test]
fn rectangular_pivoted_qr_restores_column_order_and_matches_svd() {
    let j = DMatrix::<f64>::from_row_slice(
        5,
        3,
        &[
            20.0, 0.2, 0.0, 0.0, 2.0, 0.5, 0.1, 0.0, 200.0, 5.0, 0.3, 20.0, -1.0, 0.5, 2.0,
        ],
    );
    let truth = DVector::from_vec(vec![-0.4, 3.0, 1.2]);
    let r = -&j * &truth;
    let step = qr_step(&j, &r, 0.0, &QrOptions::default()).unwrap();
    assert!((&step - &truth).norm() < 1e-12);
    let noise = DVector::from_vec(vec![0.03, -0.01, 0.02, 0.04, -0.025]);
    let r = r + noise;
    let reference = j.clone().svd(true, true).solve(&(-&r), 1e-13).unwrap();
    let step = qr_step(&j, &r, 0.0, &QrOptions::default()).unwrap();
    assert!((&step - reference).norm() < 1e-12);
    assert!((j.transpose() * (&j * step + r)).norm() < 1e-9);
}
#[test]
fn lm_qr_matches_damped_normal_solution_and_column_floor() {
    let j = DMatrix::<f64>::from_row_slice(
        4,
        3,
        &[3.0, 0.1, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
    );
    let r = DVector::from_vec(vec![1.0, -2.0, 0.5, -0.2]);
    for damping in [1e-9, 1e-3, 0.5, 10.0] {
        let mut normal = j.transpose() * &j;
        for k in 0..3 {
            normal[(k, k)] += damping * normal[(k, k)].max(1.0);
        }
        let expected = normal.cholesky().unwrap().solve(&(-j.transpose() * &r));
        let actual = qr_step(&j, &r, damping, &QrOptions::default()).unwrap();
        assert!((actual - expected).norm() < 1e-10);
    }
}
#[test]
fn singular_and_underdetermined_gn_fail_but_lm_regularization_is_solved() {
    for j in [
        DMatrix::<f64>::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]),
        DMatrix::<f64>::from_row_slice(1, 2, &[1.0, 1.0]),
    ] {
        let r = DVector::from_element(j.nrows(), -1.0);
        assert_eq!(
            qr_step(&j, &r, 0.0, &QrOptions::default()).unwrap_err(),
            Error::RankDeficient {
                rank: 1,
                parameters: 2
            }
        );
        let step = qr_step(&j, &r, 1e-3, &QrOptions::default()).unwrap();
        assert!((step[0] - step[1]).abs() < 1e-12);
        let p = Linear {
            j,
            target: DVector::from_element(2, 0.5),
        };
        let report = solve(
            &p,
            DVector::zeros(2),
            &Options {
                linear_solver: LinearSolver::DenseQr(QrOptions::default()),
                ..Options::default()
            },
        )
        .unwrap();
        assert!(report.cost < 1e-15);
        assert_eq!(
            solve(
                &p,
                DVector::zeros(2),
                &Options {
                    linear_solver: LinearSolver::DenseQr(QrOptions::default()),
                    algorithm: Algorithm::GaussNewton,
                    ..Options::default()
                }
            )
            .unwrap_err(),
            Error::RankDeficient {
                rank: 1,
                parameters: 2
            }
        );
    }
    assert_eq!(
        qr_step(
            &DMatrix::zeros(2, 2),
            &DVector::zeros(2),
            0.0,
            &QrOptions::default()
        )
        .unwrap_err(),
        Error::RankDeficient {
            rank: 0,
            parameters: 2
        }
    );
}
#[test]
fn nearly_dependent_columns_are_solved_without_a_collapsed_normal_matrix() {
    let eps = 1e-8;
    let j = DMatrix::<f64>::from_row_slice(3, 2, &[1.0, 1.0, 1.0, 1.0 + eps, 1.0, 1.0 - eps]);
    let truth = DVector::from_vec(vec![1.0, -1.0]);
    let r = -&j * &truth;
    let step = qr_step(&j, &r, 0.0, &QrOptions::default()).unwrap();
    assert!((&step - &truth).norm() < 1e-7);
    let lambda = 1e-20;
    let mut normal = j.transpose() * &j;
    for k in 0..2 {
        normal[(k, k)] += lambda * normal[(k, k)].max(1.0);
    }
    let normal_factor = normal.cholesky();
    let mut augmented = DMatrix::zeros(5, 2);
    augmented.rows_mut(0, 3).copy_from(&j);
    for k in 0..2 {
        augmented[(3 + k, k)] = (lambda * j.column(k).norm_squared()).sqrt();
    }
    let mut rhs = DVector::zeros(5);
    rhs.rows_mut(0, 3).copy_from(&(-&r));
    let reference = augmented.svd(true, true).solve(&rhs, 1e-14).unwrap();
    let actual = qr_step(&j, &r, lambda, &QrOptions::default()).unwrap();
    assert!((actual - &reference).norm() < 1e-7);
    // Rounding can produce a failed factorization or an inaccurate weak direction.
    if let Some(factor) = normal_factor {
        let normal_step = factor.solve(&(-j.transpose() * &r));
        assert!((normal_step - &reference).norm() > 1e-2);
    }
    let p = Linear { j, target: truth };
    let opts = Options {
        linear_solver: LinearSolver::DenseQr(QrOptions::default()),
        initial_damping: lambda,
        max_iterations: 1,
        gradient_tolerance: 0.0,
        step_tolerance: 0.0,
        cost_tolerance: 0.0,
        ..Options::default()
    };
    let qr = solve(&p, DVector::zeros(2), &opts).unwrap();
    assert!(qr.iterations[0].accepted);
    assert!(qr.cost < qr.initial_cost * 1e-5);
    match solve(
        &p,
        DVector::zeros(2),
        &Options {
            linear_solver: LinearSolver::Dense,
            ..opts
        },
    ) {
        Err(Error::LinearSolve) => {}
        Ok(report) => assert!((report.x - &reference).norm() > 1e-2),
        Err(e) => panic!("unexpected normal-equation failure: {e}"),
    }
}
#[test]
fn qr_scaling_and_damping_rows_avoid_unnecessary_overflow() {
    let base = DMatrix::<f64>::from_row_slice(3, 2, &[1.0, 0.2, -0.3, 2.0, 0.5, -0.7]);
    let truth = DVector::from_vec(vec![0.4, -0.8]);
    for scale in [1e-100, 1e100] {
        let j = &base * scale;
        let r = -&j * &truth;
        assert!((qr_step(&j, &r, 0.0, &QrOptions::default()).unwrap() - &truth).norm() < 1e-12);
    }
    let j = DMatrix::from_element(1, 1, 1e200);
    let r = DVector::from_element(1, 1.0);
    let step = qr_step(&j, &r, 1e-3, &QrOptions::default()).unwrap();
    assert!((step[0] * 1e200 + 1.0 / 1.001).abs() < 1e-14);
    assert_eq!(
        qr_step(
            &DMatrix::from_element(1, 1, f64::MAX),
            &r,
            f64::MAX,
            &QrOptions::default()
        )
        .unwrap_err(),
        Error::NonFinite
    );
}
#[test]
fn robust_whitened_lm_qr_uses_the_true_objective_for_gain_ratio() {
    let mut g = FactorGraph::new();
    let id = g
        .add_variable(VariableKind::Euclidean(2), DVector::zeros(2))
        .unwrap();
    let w = DMatrix::<f64>::from_row_slice(2, 2, &[1.0, 0.3, 0.0, 0.7]);
    let target = DVector::from_vec(vec![6.0, 8.0]);
    let target_copy = target.clone();
    g.add_analytic_factor_with_options(
        &[id],
        2,
        FactorOptions {
            noise: NoiseModel::sqrt_information(w.clone()).unwrap(),
            loss: Some(Loss::Huber(2.0)),
        },
        move |v| Ok((&v[0] - &target_copy, vec![DMatrix::<f64>::identity(2, 2)])),
    )
    .unwrap();
    g.add_analytic_factor(&[id], 2, |v| {
        Ok((&v[0] * 0.2, vec![DMatrix::<f64>::identity(2, 2) * 0.2]))
    })
    .unwrap();
    let options = Options {
        max_iterations: 1,
        loss: Loss::Cauchy(1.0),
        ..Options::default()
    };
    let dense = solve_local(&g, g.initial(), &options).unwrap();
    let qr = solve_local(
        &g,
        g.initial(),
        &Options {
            linear_solver: LinearSolver::DenseQr(QrOptions::default()),
            ..options
        },
    )
    .unwrap();
    assert!(qr.iterations[0].accepted);
    assert!((&qr.x - &dense.x).norm() < 1e-10);
    let norm = (&w * &target).norm();
    let weight = 2.0 / norm;
    let gradient = -w.transpose() * &w * &target * weight;
    let prediction = -gradient.dot(&qr.x)
        - 0.5 * (weight * (&w * &qr.x).norm_squared() + 0.04 * qr.x.norm_squared());
    let huber = |n: f64| {
        if n > 2.0 {
            2.0 * (n - 1.0)
        } else {
            0.5 * n * n
        }
    };
    let actual = huber(norm)
        - huber((&w * (&qr.x - target)).norm())
        - qr.x
            .iter()
            .map(|x| 0.5 * (1.0 + (0.2 * x).powi(2)).ln())
            .sum::<f64>();
    assert!((qr.iterations[0].gain_ratio - actual / prediction).abs() < 1e-12);
    assert!((qr.cost - dense.cost).abs() < 1e-12);
}
#[test]
fn qr_rosenbrock_has_rejections_and_rejected_trials_preserve_state() {
    let p = AutoDiff::<2, 2, _>(|x: [Jet<2>; 2]| {
        [
            Jet::constant(10.0) * (x[1] - x[0] * x[0]),
            Jet::constant(1.0) - x[0],
        ]
    });
    let options = Options {
        linear_solver: LinearSolver::DenseQr(QrOptions::default()),
        max_iterations: 200,
        ..Options::default()
    };
    let report = solve(&p, DVector::from_vec(vec![-1.2, 1.0]), &options).unwrap();
    assert!(report.cost < 1e-16);
    assert!(report.iterations.iter().any(|i| !i.accepted));
    assert!(report.iterations.iter().all(|i| i.linear_iterations == 0));
    assert!((report.x - DVector::from_element(2, 1.0)).norm() < 1e-7);
    let p = AutoDiff::<1, 1, _>(|x: [Jet<1>; 1]| [x[0] * x[0] - Jet::constant(1.0)]);
    let report = solve(
        &p,
        DVector::from_element(1, 0.01),
        &Options {
            max_iterations: 1,
            initial_damping: 1e-12,
            ..options
        },
    )
    .unwrap();
    assert!(!report.iterations[0].accepted);
    assert_eq!(report.x[0], 0.01);
    assert_eq!(report.cost, report.initial_cost);
}
#[test]
fn invalid_qr_inputs_and_rank_tolerance_are_errors() {
    let j = DMatrix::<f64>::identity(2, 2);
    let r = DVector::from_element(2, 1.0);
    for options in [
        QrOptions {
            relative_rank_tolerance: 0.0,
        },
        QrOptions {
            relative_rank_tolerance: 1.0,
        },
        QrOptions {
            relative_rank_tolerance: f64::NAN,
        },
    ] {
        assert_eq!(
            qr_step(&j, &r, 0.0, &options).unwrap_err(),
            Error::InvalidOptions
        );
    }
    for damping in [-1.0, f64::INFINITY, f64::NAN] {
        assert_eq!(
            qr_step(&j, &r, damping, &QrOptions::default()).unwrap_err(),
            Error::InvalidOptions
        );
    }
    assert_eq!(
        qr_step(&j, &DVector::zeros(1), 0.0, &QrOptions::default()).unwrap_err(),
        Error::InvalidDimensions
    );
    assert_eq!(
        qr_step(
            &DMatrix::zeros(0, 2),
            &DVector::zeros(0),
            0.0,
            &QrOptions::default()
        )
        .unwrap_err(),
        Error::InvalidDimensions
    );
    let mut invalid = j.clone();
    invalid[(0, 0)] = f64::NAN;
    assert_eq!(
        qr_step(&invalid, &r, 0.0, &QrOptions::default()).unwrap_err(),
        Error::NonFinite
    );
    let weak = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1e-10]));
    assert_eq!(
        qr_step(
            &weak,
            &r,
            0.0,
            &QrOptions {
                relative_rank_tolerance: 1e-8
            }
        )
        .unwrap_err(),
        Error::RankDeficient {
            rank: 1,
            parameters: 2
        }
    );
    let p = Linear {
        j,
        target: DVector::zeros(2),
    };
    assert_eq!(
        solve(
            &p,
            DVector::zeros(2),
            &Options {
                linear_solver: LinearSolver::DenseQr(QrOptions {
                    relative_rank_tolerance: 0.0
                }),
                ..Options::default()
            }
        )
        .unwrap_err(),
        Error::InvalidOptions
    );
}
