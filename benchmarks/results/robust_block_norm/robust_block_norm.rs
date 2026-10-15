use ropt::{covariance::*, factor::*, *};

fn graph(scale: f64, loss: Loss, fixed: bool) -> (FactorGraph, VariableId) {
    let mut graph = FactorGraph::new();
    let id = graph
        .add_variable(VariableKind::Euclidean(2), DVector::zeros(2))
        .unwrap();
    graph.set_fixed(id, fixed).unwrap();
    graph
        .add_analytic_factor_with_options(
            &[id],
            2,
            FactorOptions {
                loss: Some(loss),
                ..Default::default()
            },
            move |values| {
                Ok((
                    &values[0] + DVector::from_vec(vec![3. * scale, 4. * scale]),
                    vec![DMatrix::identity(2, 2)],
                ))
            },
        )
        .unwrap();
    (graph, id)
}

fn check_covariance(scale: f64, loss: Loss, covariance_scale: f64, expected_diagonal: f64) {
    let (graph, id) = graph(scale, loss, false);
    for input in [CovarianceInput::Dense, CovarianceInput::SparseToDense] {
        let covariance = graph
            .marginal_covariance(
                &graph.initial(),
                &[id],
                &CovarianceOptions {
                    model: CovarianceModel::FrozenIrls {
                        fallback_loss: Loss::Squared,
                    },
                    input,
                    ..Default::default()
                },
            )
            .unwrap();
        // The independent 3-4-5 norm determines one joint weight. With J=I,
        // inverse frozen-IRLS information is I / weight. Normalize before
        // comparison so the test oracle cannot overflow its own squared norm.
        let normalized = covariance.matrix.map(|value| value / covariance_scale);
        assert!(
            (normalized - DMatrix::identity(2, 2) * expected_diagonal).norm() < 1e-12,
            "scale={scale}, loss={loss:?}, input={input:?}"
        );
        assert_eq!(covariance.rank, 2);
        assert!((covariance.condition_number - 1.).abs() < 1e-12);
    }
}

#[test]
fn tiny_joint_norms_preserve_huber_and_cauchy_frozen_weights() {
    for scale in [1., 1e-200, 1e-160] {
        // Huber: w = scale/(5*scale) = 1/5.
        check_covariance(scale, Loss::Huber(scale), 1., 5.);
        // Cauchy: w = 1/(1 + (5*scale/scale)^2) = 1/26.
        check_covariance(scale, Loss::Cauchy(scale), 1., 26.);
    }
}

#[test]
fn large_joint_norms_have_representable_robust_costs() {
    let scale = 1e200;
    for (loss, expected) in [
        (Loss::Huber(1.), 5.),
        // 0.5*a^2*ln(1+(5e200/a)^2), a=1e100. The correction to
        // a^2*ln(5e100) is below f64 precision at this scale.
        (
            Loss::Cauchy(1e100),
            5_f64.ln() + 100. * std::f64::consts::LN_10,
        ),
    ] {
        let (graph, _) = graph(scale, loss, true);
        for linear_solver in [
            LinearSolver::Dense,
            LinearSolver::SparseQr(Default::default()),
        ] {
            let report = solve_local(
                &graph,
                graph.initial(),
                &Options {
                    linear_solver,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!((report.initial_cost / scale - expected).abs() < 1e-12);
            assert!((report.cost / scale - expected).abs() < 1e-12);
            assert!(report.iterations.is_empty());
            assert_eq!(report.x, DVector::zeros(2));
        }
    }
}

#[test]
fn large_joint_norms_preserve_finite_frozen_covariance() {
    // Huber w = 1/(5e200); inverse information is 5e200*I.
    check_covariance(1e200, Loss::Huber(1.), 1e200, 5.);
    // Cauchy w ~= 4e-202; inverse information is 2.5e201*I.
    check_covariance(1e200, Loss::Cauchy(1e100), 1e200, 25.);
}

#[test]
fn genuinely_unrepresentable_joint_norm_still_returns_nonfinite() {
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
                loss: Some(Loss::Squared),
                ..Default::default()
            },
            |_| {
                Ok((
                    DVector::from_element(2, f64::MAX),
                    vec![DMatrix::zeros(2, 1)],
                ))
            },
        )
        .unwrap();
    for linear_solver in [
        LinearSolver::Dense,
        LinearSolver::SparseQr(Default::default()),
    ] {
        assert_eq!(
            solve_local(
                &graph,
                graph.initial(),
                &Options {
                    linear_solver,
                    ..Default::default()
                }
            )
            .unwrap_err(),
            Error::NonFinite
        );
    }
}
