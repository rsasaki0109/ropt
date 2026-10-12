use nalgebra::{Vector2, Vector3};
use ropt::{camera::*, factor::*, manifold::*, *};
#[test]
fn pinhole_analytic_jacobians_have_expected_signs() {
    let model = Pinhole::new(300.0, 250.0, 0.0, 0.0).unwrap();
    let result = model
        .reprojection(
            SE3::exp(Tangent6::zeros()),
            Vector3::new(1.0, 2.0, 5.0),
            Vector2::new(60.0, 100.0),
        )
        .unwrap();
    assert_eq!(result.residual, Vector2::zeros());
    let expected = DMatrix::from_row_slice(
        2,
        6,
        &[
            -60.0, 0.0, 12.0, 24.0, -312.0, 120.0, 0.0, -50.0, 20.0, 290.0, -20.0, -50.0,
        ],
    );
    assert!(
        (DMatrix::from_column_slice(2, 6, result.pose_jacobian.as_slice()) - expected).norm()
            < 1e-12
    );
}
#[test]
fn reprojection_pose_and_point_derivatives_match_central_differences() {
    let model = Pinhole::new(420.0, 415.0, 320.0, 240.0).unwrap();
    let pose = SE3::exp(Tangent6::new(0.5, -0.2, 0.1, 0.2, -0.15, 0.3));
    let point = Vector3::new(0.8, 0.1, 5.0);
    let observed = Vector2::new(370.0, 250.0);
    let result = model.reprojection(pose, point, observed).unwrap();
    let h = 1e-6;
    for k in 0..6 {
        let mut d = Tangent6::zeros();
        d[k] = h;
        let plus = model
            .reprojection(pose.retract(d), point, observed)
            .unwrap()
            .residual;
        let minus = model
            .reprojection(pose.retract(-d), point, observed)
            .unwrap()
            .residual;
        assert!(((plus - minus) / (2.0 * h) - result.pose_jacobian.column(k)).norm() < 2e-6);
    }
    for k in 0..3 {
        let mut d = Vector3::zeros();
        d[k] = h;
        let plus = model
            .reprojection(pose, point + d, observed)
            .unwrap()
            .residual;
        let minus = model
            .reprojection(pose, point - d, observed)
            .unwrap()
            .residual;
        assert!(((plus - minus) / (2.0 * h) - result.point_jacobian.column(k)).norm() < 2e-6);
    }
}
#[test]
fn camera_invalid_domains_and_factor_types_are_rejected() {
    assert_eq!(
        Pinhole::new(-1.0, 1.0, 0.0, 0.0).unwrap_err(),
        Error::InvalidOptions
    );
    assert_eq!(
        Pinhole::new(f64::NAN, 1.0, 0.0, 0.0).unwrap_err(),
        Error::NonFinite
    );
    let model = Pinhole::new(400.0, 400.0, 320.0, 240.0).unwrap();
    for depth in [-1.0, 0.0, 1e-9] {
        assert!(matches!(
            model.project(Vector3::new(1.0, 1.0, depth)),
            Err(Error::Evaluation(_))
        ));
    }
    assert_eq!(
        model
            .project(Vector3::new(f64::INFINITY, 0.0, 1.0))
            .unwrap_err(),
        Error::NonFinite
    );
    let mut graph = FactorGraph::new();
    let wrong = graph
        .add_variable(VariableKind::Euclidean(7), DVector::zeros(7))
        .unwrap();
    let point = graph
        .add_variable(VariableKind::Euclidean(3), DVector::zeros(3))
        .unwrap();
    assert_eq!(
        add_reprojection_factor(&mut graph, wrong, point, model, Vector2::zeros()),
        Err(Error::InvalidDimensions)
    );
}
