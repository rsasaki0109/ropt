use nalgebra::{Vector2, Vector3};
use ropt::{
    autodiff::*, camera::*, factor::*, manifold::*, noise::NoiseModel, sparse::CgOptions, *,
};
#[path = "../examples/support/calibration_ba.rs"]
mod ba;
fn camera() -> BrownConrady {
    BrownConrady::new(
        Pinhole::new(420.0, 415.0, 320.0, 240.0).unwrap(),
        [-0.12, 0.035, 0.004, -0.003],
    )
    .unwrap()
}
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
fn log_focal_storage_roundtrips_and_invalid_domains_are_reported() {
    let c = camera();
    let p = c.parameters().unwrap();
    let d = BrownConrady::from_parameters(p.as_slice()).unwrap();
    assert!((c.pinhole.fx - d.pinhole.fx).abs() < 1e-12);
    assert_eq!(c.distortion, d.distortion);
    assert_eq!(
        BrownConrady::from_parameters(&[0.0; 7]).unwrap_err(),
        Error::InvalidDimensions
    );
    for (v, expected) in [
        (f64::NAN, Error::NonFinite),
        (1000.0, Error::NonFinite),
        (-1000.0, Error::InvalidOptions),
    ] {
        let mut p = p.clone();
        p[0] = v;
        assert_eq!(
            BrownConrady::from_parameters(p.as_slice()).unwrap_err(),
            expected
        );
    }
    assert_eq!(
        BrownConrady::new(c.pinhole, [0.0, 0.0, f64::INFINITY, 0.0]).unwrap_err(),
        Error::NonFinite
    );
    let pose = se3_values(SE3::exp(Tangent6::zeros()))
        .iter()
        .copied()
        .map(DynamicJet::constant)
        .collect::<Vec<_>>();
    let point = [0.2, 0.3, 3.0].map(DynamicJet::constant);
    for (v, expected) in [
        (f64::NAN, Error::NonFinite),
        (1000.0, Error::NonFinite),
        (-1000.0, Error::InvalidOptions),
    ] {
        let mut jets = p
            .iter()
            .copied()
            .map(DynamicJet::constant)
            .collect::<Vec<_>>();
        jets[0] = DynamicJet::constant(v);
        assert_eq!(
            brown_conrady_reprojection_autodiff(&jets, &pose, &point, Vector2::zeros())
                .unwrap_err(),
            expected
        );
    }
    assert_eq!(
        brown_conrady_reprojection_autodiff(&[], &pose, &point, Vector2::zeros()).unwrap_err(),
        Error::InvalidDimensions
    );
    for z in [0.0, -1.0, 1e-9] {
        assert!(matches!(
            c.project(Vector3::new(0.0, 0.0, z)),
            Err(Error::Evaluation(_))
        ));
        let point = [0.0, 0.0, z].map(DynamicJet::constant);
        assert!(matches!(
            c.reprojection_autodiff(&pose, &point, Vector2::zeros()),
            Err(Error::Evaluation(_))
        ));
    }
}
#[test]
fn known_radial_and_tangential_pixel_matches_independent_formula() {
    let c = BrownConrady::new(
        Pinhole::new(100.0, 120.0, 10.0, 20.0).unwrap(),
        [0.1, 0.02, 0.01, -0.02],
    )
    .unwrap();
    // x=.2, y=-.3, r²=.13, radial=1.013338.
    // xd=.1972676, yd=-.2985014, with both tangential terms nonzero.
    let expected = Vector2::new(29.72676, -15.820168);
    assert!((c.project(Vector3::new(2.0, -3.0, 10.0)).unwrap() - expected).norm() < 1e-13);
    assert_eq!(
        c.project(Vector3::new(0.0, 0.0, 2.0)).unwrap(),
        Vector2::new(10.0, 20.0)
    );
}
#[test]
fn zero_distortion_matches_pinhole_residuals_and_tangent_derivatives() {
    let pin = Pinhole::new(420.0, 415.0, 320.0, 240.0).unwrap();
    let c = BrownConrady::new(pin, [0.0; 4]).unwrap();
    let pose = SE3::exp(Tangent6::new(0.3, -0.2, 0.1, 0.2, -0.1, 0.3));
    let point = Vector3::new(0.8, 0.6, 4.0);
    let observed = Vector2::new(360.0, 250.0);
    let analytic = pin.reprojection(pose, point, observed).unwrap();
    assert!((c.reprojection(pose, point, observed).unwrap() - analytic.residual).norm() < 1e-12);
    let base = se3_values(pose);
    let ad = DynamicAutoDiff::new(9, 2, |d| {
        let p = VariableKind::SE3.retract_autodiff(&base, &d[..6])?;
        let q = (0..3).map(|i| point[i] + &d[6 + i]).collect::<Vec<_>>();
        c.reprojection_autodiff(&p, &q, observed)
    })
    .unwrap()
    .with_chunk_size(2)
    .unwrap();
    let mut r = DVector::zeros(2);
    let mut j = DMatrix::zeros(2, 9);
    ad.evaluate(&DVector::zeros(9), &mut r, &mut j).unwrap();
    assert!((r - analytic.residual).norm() < 1e-12);
    assert!((j.columns(0, 6) - analytic.pose_jacobian).norm() < 1e-10);
    assert!((j.columns(6, 3) - analytic.point_jacobian).norm() < 1e-10);
}
#[test]
fn intrinsic_jacobian_matches_analytic_formulas_in_log_focal_coordinates() {
    let c = camera();
    let point = Vector3::new(0.8, -0.6, 3.0);
    let observed = Vector2::new(360.0, 210.0);
    let pose = se3_values(SE3::exp(Tangent6::zeros()))
        .iter()
        .copied()
        .map(DynamicJet::constant)
        .collect::<Vec<_>>();
    let point_jets = point
        .iter()
        .copied()
        .map(DynamicJet::constant)
        .collect::<Vec<_>>();
    let f = DynamicAutoDiff::new(8, 2, |p| {
        brown_conrady_reprojection_autodiff(p, &pose, &point_jets, observed)
    })
    .unwrap()
    .with_chunk_size(3)
    .unwrap();
    let mut r = DVector::zeros(2);
    let mut j = DMatrix::zeros(2, 8);
    f.evaluate(&c.parameters().unwrap(), &mut r, &mut j)
        .unwrap();
    let x = point.x / point.z;
    let y = point.y / point.z;
    let s = x * x + y * y;
    let pred = c.project(point).unwrap();
    let fx = c.pinhole.fx;
    let fy = c.pinhole.fy;
    let expected = DMatrix::from_row_slice(
        2,
        8,
        &[
            pred.x - c.pinhole.cx,
            0.0,
            1.0,
            0.0,
            fx * x * s,
            fx * x * s * s,
            2.0 * fx * x * y,
            fx * (s + 2.0 * x * x),
            0.0,
            pred.y - c.pinhole.cy,
            0.0,
            1.0,
            fy * y * s,
            fy * y * s * s,
            fy * (s + 2.0 * y * y),
            2.0 * fy * x * y,
        ],
    );
    assert!((j - expected).norm() < 1e-12);
    assert!((r - (pred - observed)).norm() < 1e-12);
}
#[test]
fn all_seventeen_tangent_columns_match_whitened_central_differences() {
    let c = camera();
    let pose = SE3::exp(Tangent6::new(0.4, -0.2, 0.1, 0.1, -0.15, 0.2));
    let point = Vector3::new(1.1, -0.6, 4.0);
    let observed = Vector2::new(380.0, 220.0);
    let noise =
        NoiseModel::sqrt_information(DMatrix::from_row_slice(2, 2, &[2.0, 0.4, 0.0, 0.7])).unwrap();
    let mut results = Vec::new();
    for numeric in [false, true] {
        let mut graph = FactorGraph::new();
        let pose_id = graph
            .add_variable(VariableKind::SE3, se3_values(pose))
            .unwrap();
        let point_id = graph
            .add_variable(
                VariableKind::Euclidean(3),
                DVector::from_column_slice(point.as_slice()),
            )
            .unwrap();
        let calibration_id = graph
            .add_variable(VariableKind::Euclidean(8), c.parameters().unwrap())
            .unwrap();
        let options = FactorOptions {
            noise: noise.clone(),
            loss: Some(Loss::Huber(2.0)),
        };
        if numeric {
            graph
                .add_numeric_factor_with_options(
                    &[pose_id, point_id, calibration_id],
                    2,
                    1e-6,
                    options,
                    move |b| {
                        let model = BrownConrady::from_parameters(b[2].as_slice())?;
                        Ok(DVector::from_column_slice(
                            model
                                .reprojection(
                                    read_se3(&b[0])?,
                                    Vector3::from_column_slice(b[1].as_slice()),
                                    observed,
                                )?
                                .as_slice(),
                        ))
                    },
                )
                .unwrap();
        } else {
            add_calibration_reprojection_factor(
                &mut graph,
                pose_id,
                point_id,
                calibration_id,
                observed,
                options,
            )
            .unwrap();
        }
        let (r, j) = evaluate(&graph);
        assert_eq!(j.ncols(), 17);
        results.push((r, j));
        graph.set_fixed(calibration_id, true).unwrap();
        assert_eq!(evaluate(&graph).1.ncols(), 9);
        graph.set_fixed(calibration_id, false).unwrap();
        assert_eq!(evaluate(&graph).1.ncols(), 17);
    }
    assert!((&results[0].0 - &results[1].0).norm() < 1e-10);
    assert!((&results[0].1 - &results[1].1).norm() < 5e-6);
}
#[test]
fn invalid_calibration_factor_types_do_not_register_a_factor() {
    let mut graph = FactorGraph::new();
    let pose = graph
        .add_variable(VariableKind::SE3, se3_values(SE3::exp(Tangent6::zeros())))
        .unwrap();
    let point = graph
        .add_variable(VariableKind::Euclidean(3), DVector::zeros(3))
        .unwrap();
    let wrong = graph
        .add_variable(VariableKind::Euclidean(4), DVector::zeros(4))
        .unwrap();
    assert_eq!(
        add_calibration_reprojection_factor(
            &mut graph,
            pose,
            point,
            wrong,
            Vector2::zeros(),
            FactorOptions::default()
        ),
        Err(Error::InvalidDimensions)
    );
    assert_eq!(graph.dimensions().2, 0);
}
#[test]
fn shared_calibration_ba_recovers_intrinsics_distortion_and_geometry() {
    for noise in [0.0, 0.2] {
        let scene = ba::scene(noise).unwrap();
        let initial = scene.graph.initial();
        let (retained, block_size) = scene.graph.schur_layout(&scene.points).unwrap();
        assert_eq!((retained, block_size), (26, 3));
        let mut reference: Option<Report> = None;
        for linear_solver in [
            LinearSolver::Dense,
            LinearSolver::DenseQr(QrOptions::default()),
            LinearSolver::SparseCg(CgOptions::default()),
            LinearSolver::SparseBlockCg(CgOptions::default()),
            LinearSolver::SparseLdl(SparseLdlOptions::default()),
            LinearSolver::BlockSchur {
                retained,
                block_size,
                cg: CgOptions::default(),
            },
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
            assert!(scene.rms(&report.x).unwrap() < scene.rms(&initial).unwrap() * 0.02);
            for &id in &scene.cameras[..3] {
                assert_eq!(
                    scene.graph.value(&initial, id).unwrap(),
                    scene.graph.value(&report.x, id).unwrap()
                );
            }
            if noise == 0.0 {
                assert!(scene.rms(&report.x).unwrap() < 1e-6);
                assert!(scene.max_state_error(&report.x).unwrap() < 1e-5);
                assert!(scene.max_calibration_error(&report.x).unwrap() < 1e-4);
            } else {
                assert!(report.cost > 0.0);
            }
            if let Some(r) = &reference {
                assert!((report.cost - r.cost).abs() < 1e-8);
                assert!((&report.x - &r.x).norm() < 1e-5);
            }
            reference = Some(report);
        }
    }
}
