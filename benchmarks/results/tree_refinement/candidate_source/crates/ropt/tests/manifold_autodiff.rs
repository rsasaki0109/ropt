use nalgebra::{Vector2, Vector3};
use ropt::{
    autodiff::*, camera::*, factor::*, manifold::*, noise::NoiseModel, sparse::CgOptions, *,
};
#[path = "../examples/support/ba.rs"]
mod ba;

fn differentiate(
    kind: VariableKind,
    x: &DVector<f64>,
    d: &DVector<f64>,
    chunk: usize,
) -> (DVector<f64>, DMatrix<f64>) {
    let (m, n) = kind.dimensions();
    let f = DynamicAutoDiff::new(n, m, |jets| kind.retract_autodiff(x, jets))
        .unwrap()
        .with_chunk_size(chunk)
        .unwrap();
    let mut r = DVector::zeros(m);
    let mut j = DMatrix::zeros(m, n);
    f.evaluate(d, &mut r, &mut j).unwrap();
    (r, j)
}
#[test]
fn every_retraction_matches_f64_values_and_central_derivatives() {
    let cases = [
        (
            VariableKind::Euclidean(3),
            DVector::from_vec(vec![1.0, -2.0, 0.4]),
        ),
        (VariableKind::SO2, DVector::from_element(1, 2.9)),
        (
            VariableKind::SE2,
            se2_values(SE2::exp(Vector3::new(0.5, -0.2, 2.9))),
        ),
        (VariableKind::SO3, so3_values(SO3::exp(Vector3::zeros()))),
        (
            VariableKind::SO3,
            so3_values(SO3::exp(Vector3::new(3.0, 0.1, -0.2))),
        ),
        (VariableKind::SE3, se3_values(SE3::exp(Tangent6::zeros()))),
        (
            VariableKind::SE3,
            se3_values(SE3::exp(Tangent6::new(0.5, -0.2, 0.3, 0.2, -0.4, 2.9))),
        ),
    ];
    for (kind, x) in cases {
        let n = kind.dimensions().1;
        for (scale, translation_only) in [(0.0, false), (1e-8, false), (0.07, false), (0.07, true)]
        {
            let mut d =
                DVector::from_iterator(n, (0..n).map(|k| scale * (k + 1) as f64 / n as f64));
            if translation_only {
                match kind {
                    VariableKind::SE2 => d[2] = 0.0,
                    VariableKind::SE3 => d.rows_mut(3, 3).fill(0.0),
                    _ => {}
                }
            }
            let (r, j) = differentiate(kind, &x, &d, 2);
            assert!(
                (&r - kind.retract(&x, &d).unwrap()).norm() < 2e-12,
                "{kind:?} scale={scale}"
            );
            let (r_full, j_full) = differentiate(kind, &x, &d, n);
            assert_eq!(r, r_full);
            assert!((&j - j_full).norm() < 1e-13);
            let h = 1e-6;
            for k in 0..n {
                let mut step = DVector::zeros(n);
                step[k] = h;
                let numeric = (kind.retract(&x, &(&d + &step)).unwrap()
                    - kind.retract(&x, &(&d - &step)).unwrap())
                    / (2.0 * h);
                assert!(
                    (j.column(k) - numeric).norm() < 3e-8,
                    "{kind:?} scale={scale} col={k}"
                );
            }
        }
    }
}
#[test]
fn manifold_retractions_reject_invalid_inputs_without_panics() {
    let kind = VariableKind::SO3;
    let x = so3_values(SO3::exp(Vector3::zeros()));
    assert_eq!(
        kind.retract_autodiff(&x, &[]).unwrap_err(),
        Error::InvalidDimensions
    );
    let d = [
        DynamicJet::variable(0.0, 0, 1).unwrap(),
        DynamicJet::variable(0.0, 0, 2).unwrap(),
        DynamicJet::constant(0.0),
    ];
    assert_eq!(
        kind.retract_autodiff(&x, &d).unwrap_err(),
        Error::InvalidDimensions
    );
    assert_eq!(
        kind.retract_autodiff(&x, &vec![DynamicJet::constant(f64::NAN); 3])
            .unwrap_err(),
        Error::NonFinite
    );
    let d = vec![DynamicJet::constant(0.0); 3];
    assert!(kind.retract_autodiff(&DVector::zeros(4), &d).is_err());
    let d = vec![DynamicJet::constant(f64::INFINITY); 3];
    assert_eq!(kind.retract_autodiff(&x, &d).unwrap_err(), Error::NonFinite);
}
fn graph_evaluation(graph: &FactorGraph) -> (DVector<f64>, DMatrix<f64>) {
    let (_, n, m) = graph.dimensions();
    let mut r = DVector::zeros(m);
    let mut j = DMatrix::zeros(m, n);
    graph
        .evaluate_local(&graph.initial(), &mut r, &mut j)
        .unwrap();
    (r, j)
}
#[test]
fn graph_reprojection_ad_matches_analytic_and_numeric_with_noise_and_fixed_pose() {
    for sign in [1.0, -1.0] {
        for fixed in [false, true] {
            let mut results = Vec::new();
            for mode in 0..3 {
                let mut graph = FactorGraph::new();
                let mut value =
                    se3_values(SE3::exp(Tangent6::new(0.5, -0.2, 0.1, 0.2, -0.15, 0.3)));
                for k in 3..7 {
                    value[k] *= sign;
                }
                let pose = graph.add_variable(VariableKind::SE3, value).unwrap();
                let point = graph
                    .add_variable(
                        VariableKind::Euclidean(3),
                        DVector::from_vec(vec![0.8, 0.1, 5.0]),
                    )
                    .unwrap();
                let camera = Pinhole::new(420.0, 415.0, 320.0, 240.0).unwrap();
                let observed = Vector2::new(370.0, 250.0);
                let options = FactorOptions {
                    noise: NoiseModel::sqrt_information(DMatrix::from_row_slice(
                        2,
                        2,
                        &[2.0, 0.3, 0.0, 0.7],
                    ))
                    .unwrap(),
                    loss: Some(Loss::Huber(2.0)),
                };
                match mode {
                    0 => add_reprojection_factor_with_options(
                        &mut graph, pose, point, camera, observed, options,
                    )
                    .unwrap(),
                    1 => graph
                        .add_autodiff_factor_with_options(&[pose, point], 2, 2, options, move |b| {
                            camera.reprojection_autodiff(b[0], b[1], observed)
                        })
                        .unwrap(),
                    _ => graph
                        .add_numeric_factor_with_options(
                            &[pose, point],
                            2,
                            1e-6,
                            options,
                            move |b| {
                                Ok(DVector::from_column_slice(
                                    camera
                                        .reprojection(
                                            read_se3(&b[0])?,
                                            Vector3::from_column_slice(b[1].as_slice()),
                                            observed,
                                        )?
                                        .residual
                                        .as_slice(),
                                ))
                            },
                        )
                        .unwrap(),
                }
                graph.set_fixed(pose, fixed).unwrap();
                let result = graph_evaluation(&graph);
                if fixed {
                    assert_eq!(result.1.ncols(), 3);
                }
                results.push(result);
                // AD factors remain valid when a previously fixed pose is activated.
                graph.set_fixed(pose, false).unwrap();
                assert_eq!(graph_evaluation(&graph).1.ncols(), 9);
            }
            for mode in 1..3 {
                assert!((&results[0].0 - &results[mode].0).norm() < 1e-10);
                assert!((&results[0].1 - &results[mode].1).norm() < 5e-6);
            }
        }
    }
}
#[test]
fn all_manifold_graph_blocks_use_tangent_not_ambient_columns() {
    let mut graph = FactorGraph::new();
    let kinds = [
        VariableKind::SO2,
        VariableKind::SO3,
        VariableKind::SE2,
        VariableKind::SE3,
    ];
    let values = [
        DVector::from_element(1, 0.4),
        so3_values(SO3::exp(Vector3::new(0.2, -0.1, 0.3))),
        se2_values(SE2::exp(Vector3::new(1.0, 0.2, 0.3))),
        se3_values(SE3::exp(Tangent6::new(1.0, 0.2, 0.3, 0.1, -0.2, 0.4))),
    ];
    let ids = kinds
        .iter()
        .zip(&values)
        .map(|(k, x)| graph.add_variable(*k, x.clone()).unwrap())
        .collect::<Vec<_>>();
    graph
        .add_autodiff_factor_with_options(&ids, 15, 2, FactorOptions::default(), |b| {
            Ok(b.iter().flat_map(|v| v.iter().cloned()).collect())
        })
        .unwrap();
    let (r, j) = graph_evaluation(&graph);
    assert_eq!(j.shape(), (15, 13));
    let mut row = 0;
    let mut col = 0;
    for (kind, x) in kinds.iter().zip(values) {
        let (m, n) = kind.dimensions();
        let (expected_r, expected_j) = differentiate(*kind, &x, &DVector::zeros(n), n);
        assert!((r.rows(row, m) - expected_r).norm() < 1e-12);
        assert!((j.view((row, col), (m, n)) - expected_j).norm() < 1e-12);
        row += m;
        col += n;
    }
}
#[test]
fn autodiff_ba_matches_analytic_backends_with_and_without_noise() {
    for noise in [0.0, 0.35] {
        let analytic = if noise == 0.0 {
            ba::scene().unwrap()
        } else {
            ba::scene_with_noise(noise).unwrap()
        };
        let ad = ba::scene_with_autodiff(noise, true).unwrap();
        let (r, j) = graph_evaluation(&analytic.graph);
        let (ad_r, ad_j) = graph_evaluation(&ad.graph);
        assert!((&r - ad_r).norm() < 1e-10);
        assert!((&j - ad_j).norm() < 1e-9);
        let (retained, block_size) = ad.graph.schur_layout(&ad.points).unwrap();
        let reference = solve_local(
            &analytic.graph,
            analytic.graph.initial(),
            &Options::default(),
        )
        .unwrap();
        for linear_solver in [
            LinearSolver::Dense,
            LinearSolver::SparseCg(CgOptions::default()),
            LinearSolver::BlockSchur {
                retained,
                block_size,
                cg: CgOptions::default(),
            },
        ] {
            let report = solve_local(
                &ad.graph,
                ad.graph.initial(),
                &Options {
                    linear_solver,
                    ..Options::default()
                },
            )
            .unwrap();
            assert!((report.cost - reference.cost).abs() < 1e-8);
            assert!((&report.x - &reference.x).norm() < 1e-6);
            for &id in &ad.cameras[..2] {
                assert_eq!(
                    ad.graph.value(&report.x, id).unwrap(),
                    ad.graph.value(&ad.graph.initial(), id).unwrap()
                );
            }
            if noise == 0.0 {
                assert!(ad.rms(&report.x).unwrap() < 1e-6);
                assert!(ad.max_state_error(&report.x).unwrap() < 1e-5);
            }
        }
    }
}
#[test]
fn ad_projection_reports_shape_cheirality_and_nonfinite_errors() {
    let c = Pinhole::new(400.0, 400.0, 320.0, 240.0).unwrap();
    let pose = se3_values(SE3::exp(Tangent6::zeros()))
        .iter()
        .copied()
        .map(DynamicJet::constant)
        .collect::<Vec<_>>();
    assert_eq!(
        c.reprojection_autodiff(&pose, &[], Vector2::zeros())
            .unwrap_err(),
        Error::InvalidDimensions
    );
    for z in [-1.0, 0.0, 1e-9] {
        let point = [0.0, 0.0, z].map(DynamicJet::constant);
        assert!(matches!(
            c.reprojection_autodiff(&pose, &point, Vector2::zeros()),
            Err(Error::Evaluation(_))
        ));
    }
    let point = [0.0, f64::INFINITY, 5.0].map(DynamicJet::constant);
    assert_eq!(
        c.reprojection_autodiff(&pose, &point, Vector2::zeros())
            .unwrap_err(),
        Error::NonFinite
    );
}
