use nalgebra::{Vector2, Vector3};
use ropt::{
    autodiff::*, factor::*, g2o::*, manifold::*, noise::NoiseModel, pose_graph::*,
    sparse::CgOptions, *,
};
use std::collections::BTreeSet;
const LOOP: &[u8] = include_bytes!("data/loop_se2.g2o");
fn p(x: f64, y: f64, a: f64) -> SE2 {
    SE2 {
        translation: Vector2::new(x, y),
        rotation: SO2::exp(a),
    }
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
fn relative_measurement_coordinates_match_g2o_not_lie_log() {
    let z = p(1.0, 0.0, std::f64::consts::FRAC_PI_2);
    let a = p(0.0, 0.0, 0.0);
    let b = p(3.0, 1.0, std::f64::consts::FRAC_PI_2 + 0.4);
    let f = RelativePose2::new(z).unwrap();
    let expected = Vector3::new(1.0, -2.0, 0.4);
    let r = f.residual(a, b).unwrap();
    assert!((r - expected).norm() < 1e-14);
    assert!((r - z.local(a.inverse().compose(b))).norm() > 0.4);
    let world = p(-3.0, 2.0, -2.4);
    assert!((f.residual(world.compose(a), world.compose(b)).unwrap() - r).norm() < 1e-14);
    let input = se2_values(a)
        .iter()
        .chain(se2_values(b).iter())
        .copied()
        .map(DynamicJet::constant)
        .collect::<Vec<_>>();
    let ad = f.residual_autodiff(&input[..3], &input[3..]).unwrap();
    assert!((DVector::from_iterator(3, ad.iter().map(|j| j.value())) - r).norm() < 1e-14);
    // Crossing the stored angle branch does not create a spurious 2*pi residual.
    assert!(
        RelativePose2::new(p(0.0, 0.0, 0.04))
            .unwrap()
            .residual(
                p(0.0, 0.0, std::f64::consts::PI - 0.02),
                p(0.0, 0.0, -std::f64::consts::PI + 0.02)
            )
            .unwrap()
            .norm()
            < 1e-14
    );
}
#[test]
fn relative_ad_tangent_jacobian_matches_central_differences() {
    let a = p(0.4, -0.1, 2.6);
    let b = p(1.2, 0.9, -2.8);
    let z = p(0.6, -0.2, 0.7);
    let f = RelativePose2::new(z).unwrap();
    let va = se2_values(a);
    let vb = se2_values(b);
    let ad = DynamicAutoDiff::new(6, 3, |d| {
        f.residual_autodiff(
            &VariableKind::SE2.retract_autodiff(&va, &d[..3])?,
            &VariableKind::SE2.retract_autodiff(&vb, &d[3..])?,
        )
    })
    .unwrap()
    .with_chunk_size(2)
    .unwrap();
    let mut r = DVector::zeros(3);
    let mut j = DMatrix::zeros(3, 6);
    ad.evaluate(&DVector::zeros(6), &mut r, &mut j).unwrap();
    assert!((r - f.residual(a, b).unwrap()).norm() < 1e-14);
    for k in 0..6 {
        let mut d = Vector3::zeros();
        d[k % 3] = 1e-6;
        let plus = if k < 3 {
            f.residual(a.retract(d), b)
        } else {
            f.residual(a, b.retract(d))
        }
        .unwrap();
        let minus = if k < 3 {
            f.residual(a.retract(-d), b)
        } else {
            f.residual(a, b.retract(-d))
        }
        .unwrap();
        assert!((j.column(k) - (plus - minus) / 2e-6).norm() < 1e-8);
    }
}
#[test]
fn information_whitening_matches_correlated_energy_and_gradient() {
    let data = PoseGraph2::from_reader(LOOP).unwrap();
    let loaded = data.build(None).unwrap();
    let (r, j) = evaluate(&loaded.graph);
    let initial = loaded.poses(&loaded.graph.initial()).unwrap();
    let expected = data
        .edges()
        .iter()
        .map(|e| {
            let raw = RelativePose2::new(e.measurement)
                .unwrap()
                .residual(initial[&e.from], initial[&e.to])
                .unwrap();
            0.5 * (raw.transpose() * &e.information * raw)[0]
        })
        .sum::<f64>();
    assert!((0.5 * r.norm_squared() - expected).abs() < 1e-13);
    let gradient = j.transpose() * r;
    let energy = |x: &DVector<f64>| {
        let poses = loaded.poses(x).unwrap();
        data.edges()
            .iter()
            .map(|e| {
                let raw = RelativePose2::new(e.measurement)
                    .unwrap()
                    .residual(poses[&e.from], poses[&e.to])
                    .unwrap();
                0.5 * (raw.transpose() * &e.information * raw)[0]
            })
            .sum::<f64>()
    };
    for k in 0..gradient.len() {
        let mut d = DVector::zeros(gradient.len());
        d[k] = 1e-6;
        let plus = loaded.graph.retract(&loaded.graph.initial(), &d).unwrap();
        let minus = loaded
            .graph
            .retract(&loaded.graph.initial(), &(-d))
            .unwrap();
        assert!((gradient[k] - (energy(&plus) - energy(&minus)) / 2e-6).abs() < 1e-8);
    }
}
#[test]
fn information_requires_finite_symmetric_positive_definite_matrix() {
    for information in [
        DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.0, 1.0]),
        DMatrix::zeros(2, 2),
        DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, -1.0])),
    ] {
        assert_eq!(
            NoiseModel::from_information(information).unwrap_err(),
            Error::InvalidOptions
        );
    }
    assert_eq!(
        NoiseModel::from_information(DMatrix::zeros(2, 3)).unwrap_err(),
        Error::InvalidDimensions
    );
    assert_eq!(
        NoiseModel::from_information(DMatrix::from_element(1, 1, f64::NAN)).unwrap_err(),
        Error::NonFinite
    );
    assert!(RelativePose2::new(p(f64::INFINITY, 0.0, 0.0)).is_err());
    let f = RelativePose2::new(p(0.0, 0.0, 0.0)).unwrap();
    assert_eq!(
        f.residual_autodiff(&[], &[]).unwrap_err(),
        Error::InvalidDimensions
    );
}
#[test]
fn parser_handles_comments_forward_references_aliases_and_large_ids() {
    let id = u64::MAX;
    let text = format!(
        "# comment\n\nEDGE2 9 {id} 1 0 0 4 1 0.5 5 0.25 6\nFIX {id}\nVERTEX2 {id} 1 0 0 # end\nVERTEX_SE2 9 0 0 0\n"
    );
    let data = PoseGraph2::from_reader(text.as_bytes()).unwrap();
    assert_eq!(data.vertices().len(), 2);
    assert_eq!(
        data.edges()[0].information,
        DMatrix::from_row_slice(3, 3, &[4.0, 1.0, 0.5, 1.0, 5.0, 0.25, 0.5, 0.25, 6.0])
    );
    assert_eq!(data.fixed(), &BTreeSet::from([id]));
    let loaded = data.build(None).unwrap();
    assert_eq!(loaded.anchors, BTreeSet::from([id]));
    assert_eq!(loaded.graph.dimensions(), (6, 3, 3));
}
#[test]
fn invalid_records_report_lines_instead_of_discarding_data() {
    let prefix = "VERTEX_SE2 0 0 0 0\n";
    for line in [
        "VERTEX_SE2 0 1 0 0",
        "VERTEX_SE2 -1 0 0 0",
        "VERTEX_SE2 1 NaN 0 0",
        "VERTEX_SE2 1 0 0 0 extra",
        "VERTEX_SE3:QUAT 1 0 0 0 0 0 0 1",
        "EDGE_SE2 0 1 1 0 0 1 0 0 1 0 1",
        "EDGE_SE2 0 0 1 0 0 1 0 0 1 0 1",
        "EDGE_SE2 0 1 1 0 0 1 2 0 1 0 1",
        "FIX 99",
    ] {
        let text = format!("{prefix}{line}\n");
        assert!(
            matches!(
                PoseGraph2::from_reader(text.as_bytes()),
                Err(G2oError::Invalid { line: 2, .. })
            ),
            "{line}"
        );
    }
    assert!(PoseGraph2::from_reader("\n# blank\n".as_bytes()).is_err());
    let isolated = PoseGraph2::from_reader(prefix.as_bytes()).unwrap();
    assert!(isolated.build(None).is_err());
    let data = PoseGraph2::from_reader(LOOP).unwrap();
    assert!(data.build(Some(Loss::Huber(-1.0))).is_err());
}
#[test]
fn connected_components_get_anchors_and_explicit_fixes_are_respected() {
    let text = "VERTEX_SE2 90 0 0 0\nVERTEX_SE2 91 1 0 0\nVERTEX_SE2 5 0 0 0\nVERTEX_SE2 6 1 0 0\nEDGE_SE2 90 91 1 0 0 1 0 0 1 0 1\nEDGE_SE2 5 6 1 0 0 1 0 0 1 0 1\nFIX 91\n";
    let loaded = PoseGraph2::from_reader(text.as_bytes())
        .unwrap()
        .build(None)
        .unwrap();
    assert_eq!(loaded.anchors, BTreeSet::from([5, 91]));
    assert_eq!(loaded.graph.dimensions(), (12, 6, 6));
}
#[test]
fn loop_closure_recovers_truth_across_backends_and_roundtrips_output() {
    let data = PoseGraph2::from_reader(LOOP).unwrap();
    let loaded = data.build(None).unwrap();
    let initial = loaded.graph.initial();
    let mut reference: Option<DVector<f64>> = None;
    for linear_solver in [
        LinearSolver::Dense,
        LinearSolver::SparseCg(CgOptions::default()),
    ] {
        let report = solve_local(
            &loaded.graph,
            initial.clone(),
            &Options {
                linear_solver,
                ..Options::default()
            },
        )
        .unwrap();
        assert!(report.cost < 1e-18);
        assert!(!matches!(
            report.termination,
            Termination::MaxIterations | Termination::NoProgress
        ));
        let poses = loaded.poses(&report.x).unwrap();
        for (key, x, y, angle) in [
            (10, 0.0, 0.0, 0.3),
            (20, 1.0, 0.0, 0.3 + std::f64::consts::FRAC_PI_2),
            (40, 1.0, 1.0, 0.3 + std::f64::consts::PI),
            (70, 0.0, 1.0, 0.3 - std::f64::consts::FRAC_PI_2),
        ] {
            assert!(p(x, y, angle).local(poses[&key]).norm() < 1e-8);
        }
        assert_eq!(
            loaded
                .graph
                .value(&report.x, loaded.variables[&10])
                .unwrap(),
            loaded.graph.value(&initial, loaded.variables[&10]).unwrap()
        );
        if let Some(x) = &reference {
            assert!((&report.x - x).norm() < 1e-8);
        }
        reference = Some(report.x);
        let mut output = Vec::new();
        data.write(&mut output, &poses).unwrap();
        let reloaded = PoseGraph2::from_reader(output.as_slice()).unwrap();
        assert_eq!(data.fixed(), reloaded.fixed());
        assert_eq!(reloaded.edges().len(), 5);
        for (before, after) in data.edges().iter().zip(reloaded.edges()) {
            assert_eq!(before.information, after.information);
            assert!(before.measurement.local(after.measurement).norm() < 1e-14);
        }
        let second = reloaded.build(None).unwrap();
        assert!(evaluate(&second.graph).0.norm() < 2e-9);
    }
}
