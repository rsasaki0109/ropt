use nalgebra::{Vector2, Vector3};
use ropt::{factor::*, manifold::*, pose_graph::*, *};
fn pose(v: &[f64]) -> SE2 {
    SE2 {
        translation: Vector2::new(v[0], v[1]),
        rotation: SO2::exp(v[2]),
    }
}
#[test]
fn ambient_analytic_jacobians_match_forward_ad_and_right_retraction_differences() {
    for (a, b, z) in [
        ([2., -1., 0.4], [3., 2., -0.2], [1., 0.1, 0.3]),
        ([-4., 3., 2.1], [2., -1., -2.5], [0.8, 0.3, -0.4]),
        ([1., 2., 3.1], [-2., 4., -3.1], [0.2, 0.1, 0.3]),
    ] {
        let cost = RelativePose2::new(pose(&z)).unwrap();
        let (r, ja, jb) = cost.linearize_ambient(&a, &b).unwrap();
        assert!((r - cost.residual(pose(&a), pose(&b)).unwrap()).norm() < 3e-14);
        let mut graph = FactorGraph::new();
        let ia = graph
            .add_variable(VariableKind::SE2, DVector::from_column_slice(&a))
            .unwrap();
        let ib = graph
            .add_variable(VariableKind::SE2, DVector::from_column_slice(&b))
            .unwrap();
        add_relative_pose2_factor(&mut graph, ia, ib, pose(&z), Default::default()).unwrap();
        let mut ad_r = DVector::zeros(3);
        let mut ad_j = DMatrix::zeros(3, 6);
        graph
            .evaluate_local(&graph.initial(), &mut ad_r, &mut ad_j)
            .unwrap();
        assert!((ad_r - r).norm() < 1e-13);
        assert!((ad_j.columns(0, 3) - ja).norm() < 1e-12);
        assert!((ad_j.columns(3, 3) - jb).norm() < 1e-12);
        for c in 0..6 {
            let mut d = Vector3::zeros();
            d[c % 3] = 1e-6;
            let plus = if c < 3 {
                cost.linearize_ambient(se2_values(pose(&a).retract(d)).as_slice(), &b)
                    .unwrap()
                    .0
            } else {
                cost.linearize_ambient(&a, se2_values(pose(&b).retract(d)).as_slice())
                    .unwrap()
                    .0
            };
            let minus = if c < 3 {
                cost.linearize_ambient(se2_values(pose(&a).retract(-d)).as_slice(), &b)
                    .unwrap()
                    .0
            } else {
                cost.linearize_ambient(&a, se2_values(pose(&b).retract(-d)).as_slice())
                    .unwrap()
                    .0
            };
            let expected = if c < 3 {
                ja.column(c)
            } else {
                jb.column(c - 3)
            };
            assert!(((plus - minus) / 2e-6 - expected).norm() < 2e-8);
        }
    }
}
#[test]
fn nonprincipal_and_extreme_angles_keep_group_chart_range_reduction_and_error_checks() {
    let cost = RelativePose2::new(pose(&[0.3, -0.1, 0.2])).unwrap();
    for angle in [12., -20., 1e250, -1e250] {
        let a = [1., 2., angle];
        let b = [-3., 4., -angle];
        let actual = cost.linearize_ambient(&a, &b).unwrap();
        let expected = cost.linearize(pose(&a), pose(&b)).unwrap();
        assert_eq!(actual, expected);
    }
    assert_eq!(
        cost.linearize_ambient(&[0., 0.], &[0., 0., 0.]),
        Err(Error::InvalidDimensions)
    );
    assert_eq!(
        cost.linearize_ambient(&[0., 0., f64::NAN], &[0., 0., 0.]),
        Err(Error::NonFinite)
    );
    assert_eq!(
        cost.linearize_ambient(&[f64::MAX, 0., 0.], &[-f64::MAX, 0., 0.]),
        Err(Error::NonFinite)
    );
}
