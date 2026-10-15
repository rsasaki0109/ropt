use nalgebra::{Vector2, Vector3};
use ropt::{autodiff::DynamicJet as J, factor::*, manifold::*, pose_graph::*, *};
fn pose(x: f64, y: f64, angle: f64) -> SE2 {
    SE2 {
        translation: Vector2::new(x, y),
        rotation: SO2::exp(angle),
    }
}
#[test]
fn cartesian_chart_inverts_retraction_and_its_ad_chain_has_identity_derivative() {
    let kind = VariableKind::SE2Cartesian;
    for (base, d) in [
        (pose(2., -1., 0.4), Vector3::new(1., 2., 0.3)),
        (pose(-4., 3., -2.8), Vector3::new(-0.2, 0.7, -0.8)),
        (pose(0., 0., 0.), Vector3::new(0., 0., 0.)),
    ] {
        let x = se2_values(base);
        let delta = DVector::from_column_slice(d.as_slice());
        let value = kind.retract(&x, &delta).unwrap();
        assert!((kind.local(&x, &value).unwrap() - &delta).norm() < 2e-14);
        let jets: Vec<_> = (0..3).map(|k| J::variable(d[k], k, 3).unwrap()).collect();
        let value = kind.retract_autodiff(&x, &jets).unwrap();
        let recovered = kind.local_autodiff(&x, &value).unwrap();
        for (r, jet) in recovered.iter().enumerate() {
            assert!((jet.value() - d[r]).abs() < 2e-14);
            for (c, derivative) in jet.partials().unwrap().iter().enumerate() {
                assert!((derivative - f64::from(r == c)).abs() < 2e-13);
            }
        }
    }
    // The group exponential remains a different, available chart.
    let base = pose(1., 2., 0.3);
    let d = Vector3::new(1., 2., 0.7);
    assert!((base.retract(d).translation - base.retract_cartesian(d).translation).norm() > 0.1);
}
#[test]
fn cartesian_right_chart_solves_a_fixed_root_relative_pose_in_one_gauss_newton_trial() {
    let mut graph = FactorGraph::new();
    let root = pose(2., -1., 0.4);
    let measurement = pose(1., 2., 0.8);
    let a = graph
        .add_variable(VariableKind::SE2Cartesian, se2_values(root))
        .unwrap();
    graph.set_fixed(a, true).unwrap();
    let b = graph
        .add_variable(VariableKind::SE2Cartesian, se2_values(pose(-2., 3., -1.2)))
        .unwrap();
    add_relative_pose2_factor_analytic(&mut graph, a, b, measurement, Default::default()).unwrap();
    let result = solve_local(
        &graph,
        graph.initial(),
        &Options {
            algorithm: Algorithm::GaussNewton,
            linear_solver: LinearSolver::DenseQr(QrOptions::default()),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.iterations.len(), 1);
    assert_eq!(result.termination, Termination::Gradient);
    assert!(result.cost < 1e-25);
    let value = read_se2(&graph.value(&result.x, b).unwrap()).unwrap();
    assert!(root.compose(measurement).local_cartesian(value).norm() < 1e-12);
}
#[test]
fn graph_retraction_and_error_handling_follow_the_selected_cartesian_chart() {
    let x = se2_values(pose(2., 1., 0.4));
    let mut graph = FactorGraph::new();
    graph
        .add_variable(VariableKind::SE2Cartesian, x.clone())
        .unwrap();
    let delta = DVector::from_vec(vec![0.3, -0.2, 0.7]);
    assert_eq!(
        graph.retract(&x, &delta).unwrap(),
        VariableKind::SE2Cartesian.retract(&x, &delta).unwrap()
    );
    assert_eq!(
        VariableKind::SE2Cartesian.retract(&x, &DVector::zeros(2)),
        Err(Error::InvalidDimensions)
    );
    assert_eq!(
        VariableKind::SE2Cartesian.retract(&x, &DVector::from_element(3, f64::NAN)),
        Err(Error::NonFinite)
    );
    assert_eq!(
        VariableKind::SE2Cartesian.local(&x, &DVector::from_element(3, f64::INFINITY)),
        Err(Error::NonFinite)
    );
}
