use nalgebra::Vector2;
use ropt::{
    DMatrix, DVector, Error, LinearSolver, LocalProblem, Loss, Options, factor::*, g2o::*,
    manifold::*, noise::NoiseModel, pose_graph::*, solve_local,
};
fn pose(x: f64, y: f64, a: f64) -> SE2 {
    SE2 {
        translation: Vector2::new(x, y),
        rotation: SO2::exp(a),
    }
}
fn graph(a: SE2, b: SE2, z: SE2, analytic: bool) -> FactorGraph {
    let mut g = FactorGraph::new();
    let a = g.add_variable(VariableKind::SE2, se2_values(a)).unwrap();
    let b = g.add_variable(VariableKind::SE2, se2_values(b)).unwrap();
    let options = FactorOptions {
        noise: NoiseModel::from_information(DMatrix::from_row_slice(
            3,
            3,
            &[4., 0.3, -0.2, 0.3, 3., 0.4, -0.2, 0.4, 2.],
        ))
        .unwrap(),
        loss: Some(Loss::Huber(1.)),
    };
    if analytic {
        add_relative_pose2_factor_analytic(&mut g, a, b, z, options).unwrap();
    } else {
        add_relative_pose2_factor(&mut g, a, b, z, options).unwrap();
    }
    g
}
fn evaluate(g: &FactorGraph, x: &DVector<f64>) -> (DVector<f64>, DMatrix<f64>) {
    let (_, n, m) = g.dimensions();
    let mut r = DVector::zeros(m);
    let mut j = DMatrix::zeros(m, n);
    g.evaluate_local(x, &mut r, &mut j).unwrap();
    (r, j)
}
#[test]
fn exact_analytic_jacobians_match_ad_and_central_differences_after_correlated_whitening() {
    for (a, b, z) in [
        (pose(0., 0., 0.), pose(1., 2., 0.7), pose(0.5, -0.2, 0.3)),
        (
            pose(-4., 3., 1.2),
            pose(2., -1., -0.9),
            pose(0.8, 0.3, -0.2),
        ),
        (pose(2., 1., 3.1), pose(-1., 4., -3.1), pose(0.2, 0.1, 0.3)),
        (
            pose(1., -2., 0.),
            pose(3., 4., std::f64::consts::PI - 1e-4),
            pose(0., 0., 0.),
        ),
        (
            pose(1., -2., 0.),
            pose(3., 4., -std::f64::consts::PI + 1e-4),
            pose(0., 0., 0.),
        ),
    ] {
        let ad = graph(a, b, z, false);
        let analytic = graph(a, b, z, true);
        let x = analytic.initial();
        let (r, j) = evaluate(&analytic, &x);
        let (ra, ja) = evaluate(&ad, &ad.initial());
        assert!((&r - ra).norm() < 1e-12);
        assert!((&j - ja).norm() < 1e-11);
        for col in 0..6 {
            let mut d = DVector::zeros(6);
            d[col] = 1e-6;
            let plus = evaluate(&analytic, &analytic.retract(&x, &d).unwrap()).0;
            let minus = evaluate(&analytic, &analytic.retract(&x, &(-&d)).unwrap()).0;
            assert!(((plus - minus) / 2e-6 - j.column(col)).norm() < 1e-8);
        }
    }
}
#[test]
fn analytic_g2o_graphs_match_ad_solutions_with_squared_and_joint_huber_losses() {
    let data =
        PoseGraph2::from_reader(std::io::Cursor::new(include_str!("data/loop_se2.g2o"))).unwrap();
    for loss in [None, Some(Loss::Huber(0.2))] {
        let ad = data.build(loss).unwrap();
        let reference = solve_local(&ad.graph, ad.graph.initial(), &Options::default()).unwrap();
        let target = ad.poses(&reference.x).unwrap();
        for solver in [
            LinearSolver::Dense,
            LinearSolver::DenseQr(Default::default()),
            LinearSolver::SparseLdl(Default::default()),
        ] {
            let g = data
                .build_with_jacobians(loss, PoseJacobian::Analytic)
                .unwrap();
            let r = solve_local(
                &g.graph,
                g.graph.initial(),
                &Options {
                    linear_solver: solver,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!((r.cost - reference.cost).abs() < 1e-10);
            for (key, value) in g.poses(&r.x).unwrap() {
                assert!(target[&key].local(value).norm() < 1e-7);
            }
            for key in &g.anchors {
                assert_eq!(
                    g.graph.value(&r.x, g.variables[key]).unwrap(),
                    g.graph.value(&g.graph.initial(), g.variables[key]).unwrap()
                );
            }
        }
    }
}
#[test]
fn nonfinite_analytic_results_and_wrong_variable_kinds_are_rejected() {
    let r = RelativePose2::new(pose(0., 0., 0.)).unwrap();
    assert_eq!(
        r.linearize(pose(f64::NAN, 0., 0.), pose(1., 0., 0.))
            .unwrap_err(),
        Error::NonFinite
    );
    let mut g = FactorGraph::new();
    let a = g
        .add_variable(VariableKind::Euclidean(3), DVector::zeros(3))
        .unwrap();
    let b = g
        .add_variable(VariableKind::SE2, DVector::zeros(3))
        .unwrap();
    assert_eq!(
        add_relative_pose2_factor_analytic(&mut g, a, b, pose(0., 0., 0.), Default::default())
            .unwrap_err(),
        Error::InvalidDimensions
    );
}
