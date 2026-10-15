use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use ropt::{
    autodiff::*, factor::*, manifold::*, navigation::*, noise::NoiseModel, pose_graph::*,
    sparse::CgOptions, *,
};
#[path = "../examples/support/gnss.rs"]
mod gnss;
fn evaluate(graph: &FactorGraph) -> (DVector<f64>, DMatrix<f64>) {
    let (_, n, m) = graph.dimensions();
    let mut r = DVector::zeros(m);
    let mut j = DMatrix::zeros(m, n);
    graph
        .evaluate_local(&graph.initial(), &mut r, &mut j)
        .unwrap();
    (r, j)
}
#[test]
fn antenna_translation_and_right_body_jacobian_have_expected_signs() {
    let pose = SE3 {
        translation: Vector3::new(1.0, 2.0, 3.0),
        rotation: SO3::exp(Vector3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2)),
    };
    let cost = Position3::new(Vector3::new(1.0, 2.0, 3.0), Vector3::new(0.3, -0.1, 0.8)).unwrap();
    let result = cost.evaluate(pose).unwrap();
    assert!((result.residual - Vector3::new(0.1, 0.3, 0.8)).norm() < 1e-14);
    assert!((result.pose_jacobian.column(0) - Vector3::new(0.0, 1.0, 0.0)).norm() < 1e-14);
    let identity = SE3::exp(Tangent6::zeros());
    let result = Position3::new(Vector3::zeros(), Vector3::new(1.0, 2.0, 3.0))
        .unwrap()
        .evaluate(identity)
        .unwrap();
    let expected = DMatrix::from_row_slice(3, 3, &[0.0, 3.0, -2.0, -3.0, 0.0, 1.0, 2.0, -1.0, 0.0]);
    assert!((result.pose_jacobian.columns(3, 3) - expected).norm() < 1e-14);
    let result = Position3::new(Vector3::zeros(), Vector3::zeros())
        .unwrap()
        .evaluate(pose)
        .unwrap();
    assert_eq!(result.pose_jacobian.columns(3, 3).norm(), 0.0);
}
#[test]
fn position_analytic_ad_and_numeric_match_with_correlated_covariance() {
    for sign in [1.0, -1.0] {
        let pose = SE3::exp(Tangent6::new(1.0, -0.5, 0.4, 0.2, -0.3, 0.5));
        let observed = Vector3::new(1.2, -0.3, 0.8);
        let lever = Vector3::new(0.3, -0.12, 0.8);
        let noise = NoiseModel::from_covariance(DMatrix::from_row_slice(
            3,
            3,
            &[0.04, 0.01, 0.005, 0.01, 0.09, -0.008, 0.005, -0.008, 0.16],
        ))
        .unwrap();
        let cost = Position3::new(observed, lever).unwrap();
        let mut results = Vec::new();
        for mode in 0..3 {
            let mut graph = FactorGraph::new();
            let mut x = se3_values(pose);
            for k in 3..7 {
                x[k] *= sign;
            }
            let id = graph.add_variable(VariableKind::SE3, x).unwrap();
            let options = FactorOptions {
                noise: noise.clone(),
                loss: Some(Loss::Huber(2.0)),
            };
            match mode {
                0 => add_position3_factor(&mut graph, id, observed, lever, options).unwrap(),
                1 => graph
                    .add_autodiff_factor_with_options(&[id], 3, 2, options, move |b| {
                        cost.residual_autodiff(b[0])
                    })
                    .unwrap(),
                _ => graph
                    .add_numeric_factor_with_options(&[id], 3, 1e-6, options, move |b| {
                        Ok(DVector::from_column_slice(
                            cost.evaluate(read_se3(&b[0])?)?.residual.as_slice(),
                        ))
                    })
                    .unwrap(),
            }
            results.push(evaluate(&graph));
            graph.set_fixed(id, true).unwrap();
            let (r, j) = evaluate(&graph);
            assert_eq!(j.shape(), (3, 0));
            assert!((r - &results.last().unwrap().0).norm() < 1e-12);
        }
        for mode in 1..3 {
            assert!((&results[0].0 - &results[mode].0).norm() < 1e-12);
            assert!((&results[0].1 - &results[mode].1).norm() < 2e-8);
        }
    }
}
#[test]
fn world_frame_transform_preserves_whitened_energy_and_body_normal_equations() {
    let pose = SE3::exp(Tangent6::new(1.0, -0.2, 0.3, 0.1, -0.3, 0.4));
    let world = SE3::exp(Tangent6::new(-3.0, 2.0, 4.0, -0.2, 0.4, -1.0));
    let observed = Vector3::new(0.8, -0.1, 0.5);
    let lever = Vector3::new(0.3, -0.12, 0.8);
    let sigma = DMatrix::from_row_slice(
        3,
        3,
        &[0.04, 0.01, 0.005, 0.01, 0.09, -0.008, 0.005, -0.008, 0.16],
    );
    let q = DMatrix::from_column_slice(
        3,
        3,
        world.rotation.0.to_rotation_matrix().matrix().as_slice(),
    );
    let mut results = Vec::new();
    for (p, z, s) in [
        (pose, observed, sigma.clone()),
        (
            world.compose(pose),
            world.translation + world.rotation.0 * observed,
            &q * sigma * q.transpose(),
        ),
    ] {
        let mut graph = FactorGraph::new();
        let id = graph
            .add_variable(VariableKind::SE3, se3_values(p))
            .unwrap();
        add_position3_factor(
            &mut graph,
            id,
            z,
            lever,
            FactorOptions {
                noise: NoiseModel::from_covariance(s).unwrap(),
                loss: Some(Loss::Squared),
            },
        )
        .unwrap();
        results.push(evaluate(&graph));
    }
    let (r, j) = &results[0];
    let (s, k) = &results[1];
    assert!((r.norm_squared() - s.norm_squared()).abs() < 1e-12);
    assert!((j.transpose() * j - k.transpose() * k).norm() < 1e-11);
    assert!((j.transpose() * r - k.transpose() * s).norm() < 1e-12);
}
#[test]
fn relative_se3_log_ad_matches_f64_and_central_tangent_derivatives() {
    let a = SE3::exp(Tangent6::new(0.4, -0.2, 0.3, 0.2, -0.3, 0.4));
    let z = SE3::exp(Tangent6::new(1.0, 0.1, -0.2, -0.1, 0.2, 0.3));
    let f = RelativePose3::new(z).unwrap();
    for angle in [0.0, 1e-9, 1e-6, 0.7, std::f64::consts::PI - 1e-3] {
        let delta = Tangent6::new(0.3, -0.2, 0.15, 0.0, 0.0, angle);
        let b = a.compose(z).retract(delta);
        let va = se3_values(a);
        let vb = se3_values(b);
        let ad = DynamicAutoDiff::new(12, 6, |d| {
            f.residual_autodiff(
                &VariableKind::SE3.retract_autodiff(&va, &d[..6])?,
                &VariableKind::SE3.retract_autodiff(&vb, &d[6..])?,
            )
        })
        .unwrap()
        .with_chunk_size(3)
        .unwrap();
        let mut r = DVector::zeros(6);
        let mut j = DMatrix::zeros(6, 12);
        ad.evaluate(&DVector::zeros(12), &mut r, &mut j).unwrap();
        assert!(
            (&r - f.residual(a, b).unwrap()).norm() < 1e-11,
            "angle={angle}"
        );
        assert!((&r - delta).norm() < 1e-11);
        for k in 0..12 {
            let mut d = Tangent6::zeros();
            d[k % 6] = 1e-6;
            let plus = if k < 6 {
                f.residual(a.retract(d), b)
            } else {
                f.residual(a, b.retract(d))
            }
            .unwrap();
            let minus = if k < 6 {
                f.residual(a.retract(-d), b)
            } else {
                f.residual(a, b.retract(-d))
            }
            .unwrap();
            assert!(
                (j.column(k) - (plus - minus) / 2e-6).norm() < 5e-7,
                "angle={angle},col={k}"
            );
        }
    }
}
#[test]
fn identity_relative_log_jacobian_is_finite_and_quaternion_signs_are_invariant() {
    let mut reference: Option<DMatrix<f64>> = None;
    for sign_a in [1.0, -1.0] {
        for sign_b in [1.0, -1.0] {
            let mut graph = FactorGraph::new();
            let mut value = se3_values(SE3::exp(Tangent6::zeros()));
            value[3] *= sign_a;
            let a = graph.add_variable(VariableKind::SE3, value).unwrap();
            let mut value = se3_values(SE3::exp(Tangent6::zeros()));
            value[3] *= sign_b;
            let b = graph.add_variable(VariableKind::SE3, value).unwrap();
            add_relative_pose3_factor(
                &mut graph,
                a,
                b,
                SE3::exp(Tangent6::zeros()),
                FactorOptions::default(),
            )
            .unwrap();
            let (r, j) = evaluate(&graph);
            assert_eq!(r.norm(), 0.0);
            assert!((j.columns(0, 6) + DMatrix::identity(6, 6)).norm() < 1e-14);
            assert!((j.columns(6, 6) - DMatrix::identity(6, 6)).norm() < 1e-14);
            if let Some(ref_j) = &reference {
                assert!((&j - ref_j).norm() < 1e-14);
            }
            reference = Some(j);
        }
    }
}
#[test]
fn sensor_factor_bad_inputs_are_errors_and_registration_remains_unchanged() {
    assert_eq!(
        Position3::new(Vector3::new(f64::NAN, 0.0, 0.0), Vector3::zeros()).unwrap_err(),
        Error::NonFinite
    );
    assert_eq!(
        Position3::new(Vector3::zeros(), Vector3::new(0.0, f64::INFINITY, 0.0)).unwrap_err(),
        Error::NonFinite
    );
    let cost = Position3::new(Vector3::zeros(), Vector3::zeros()).unwrap();
    assert_eq!(
        cost.residual_autodiff(&[]).unwrap_err(),
        Error::InvalidDimensions
    );
    let bad = SE3 {
        translation: Vector3::zeros(),
        rotation: SO3(UnitQuaternion::new_unchecked(Quaternion::new(
            2.0, 0.0, 0.0, 0.0,
        ))),
    };
    assert!(cost.evaluate(bad).is_err());
    assert!(RelativePose3::new(bad).is_err());
    let bad = se3_values(bad)
        .iter()
        .copied()
        .map(DynamicJet::constant)
        .collect::<Vec<_>>();
    assert!(cost.residual_autodiff(&bad).is_err());
    let mut graph = FactorGraph::new();
    let wrong = graph
        .add_variable(VariableKind::Euclidean(7), DVector::zeros(7))
        .unwrap();
    let pose = graph
        .add_variable(VariableKind::SE3, se3_values(SE3::exp(Tangent6::zeros())))
        .unwrap();
    assert_eq!(
        add_position3_factor(
            &mut graph,
            wrong,
            Vector3::zeros(),
            Vector3::zeros(),
            FactorOptions::default()
        ),
        Err(Error::InvalidDimensions)
    );
    assert_eq!(
        add_relative_pose3_factor(
            &mut graph,
            wrong,
            pose,
            SE3::exp(Tangent6::zeros()),
            FactorOptions::default()
        ),
        Err(Error::InvalidDimensions)
    );
    assert_eq!(graph.dimensions().2, 0);
}
#[test]
fn relative_log_factors_solve_the_anchored_se3_loop_with_both_backends() {
    let truth = [
        SE3::exp(Tangent6::zeros()),
        SE3::exp(Tangent6::new(1.0, 0.1, 0.0, 0.1, 0.0, 0.2)),
        SE3::exp(Tangent6::new(2.0, 0.2, 0.1, 0.1, 0.1, 0.4)),
    ];
    let mut graph = FactorGraph::new();
    let ids = truth
        .iter()
        .enumerate()
        .map(|(i, p)| {
            graph.add_variable(
                VariableKind::SE3,
                se3_values(if i == 0 {
                    *p
                } else {
                    p.retract(Tangent6::new(0.2, -0.1, 0.1, 0.05, 0.1, -0.1))
                }),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    graph.set_fixed(ids[0], true).unwrap();
    for (a, b) in [(0, 1), (1, 2), (0, 2)] {
        add_relative_pose3_factor(
            &mut graph,
            ids[a],
            ids[b],
            truth[a].inverse().compose(truth[b]),
            FactorOptions::default(),
        )
        .unwrap();
    }
    for linear_solver in [
        LinearSolver::Dense,
        LinearSolver::SparseCg(CgOptions::default()),
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
        assert!(report.cost < 1e-16);
        for (&id, p) in ids.iter().zip(truth) {
            assert!(
                p.local(read_se3(&graph.value(&report.x, id).unwrap()).unwrap())
                    .norm()
                    < 1e-8
            );
        }
    }
}
#[test]
fn robust_gnss_reduces_all_timestamp_position_error_and_dense_sparse_agree() {
    let mut errors = Vec::new();
    for loss in [None, Some(Loss::Squared), Some(Loss::Huber(2.5))] {
        let scene = gnss::scene(loss).unwrap();
        assert_eq!(scene.fixes.iter().filter(|f| f.2).count(), 2);
        assert_eq!(scene.lever_arm, Vector3::new(0.3, -0.12, 0.8));
        let initial = scene.graph.initial();
        let mut reference: Option<Report> = None;
        for linear_solver in [
            LinearSolver::Dense,
            LinearSolver::SparseCg(CgOptions::default()),
        ] {
            let report = solve_local(
                &scene.graph,
                initial.clone(),
                &Options {
                    linear_solver,
                    ..Options::default()
                },
            )
            .unwrap();
            assert!(!matches!(
                report.termination,
                Termination::MaxIterations | Termination::NoProgress
            ));
            assert_eq!(
                scene.graph.value(&report.x, scene.poses[0]).unwrap(),
                scene.graph.value(&initial, scene.poses[0]).unwrap()
            );
            assert!(scene.endpoint_error(&report.x).unwrap().is_finite());
            if let Some(r) = &reference {
                assert!((r.cost - report.cost).abs() < 1e-8);
                assert!((&r.x - &report.x).norm() < 1e-6);
            }
            reference = Some(report);
        }
        errors.push(scene.position_rmse(&reference.unwrap().x).unwrap());
    }
    assert!(errors[2] < errors[0] * 0.5, "{errors:?}");
    assert!(errors[2] < errors[1] * 0.5, "{errors:?}");
}
