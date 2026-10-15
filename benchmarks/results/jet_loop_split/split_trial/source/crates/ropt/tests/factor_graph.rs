use nalgebra::{Matrix3, Vector3};
use ropt::{factor::*, manifold::*, *};

#[test]
fn analytic_blocks_assemble_and_fixed_variables_are_excluded() {
    let mut g = FactorGraph::new();
    let a = g
        .add_variable(
            VariableKind::Euclidean(2),
            DVector::from_vec(vec![2.0, 3.0]),
        )
        .unwrap();
    let b = g
        .add_variable(VariableKind::Euclidean(2), DVector::zeros(2))
        .unwrap();
    g.set_fixed(a, true).unwrap();
    g.add_analytic_factor(&[a, b], 2, |v| {
        Ok((
            &v[1] - &v[0],
            vec![-DMatrix::identity(2, 2), DMatrix::identity(2, 2)],
        ))
    })
    .unwrap();
    assert_eq!(g.dimensions(), (4, 2, 2));
    let mut r = DVector::zeros(2);
    let mut j = DMatrix::zeros(2, 2);
    g.evaluate_local(&g.initial(), &mut r, &mut j).unwrap();
    assert_eq!(j, DMatrix::identity(2, 2));
    let report = solve_local(
        &g,
        g.initial(),
        &Options {
            algorithm: Algorithm::GaussNewton,
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(
        g.value(&report.x, a).unwrap(),
        DVector::from_vec(vec![2.0, 3.0])
    );
    assert!((g.value(&report.x, b).unwrap() - g.value(&report.x, a).unwrap()).norm() < 1e-12);
    g.set_fixed(a, false).unwrap();
    assert_eq!(g.dimensions(), (4, 4, 2));
    g.set_fixed(b, true).unwrap();
    assert_eq!(g.dimensions(), (4, 2, 2));
}
#[test]
fn all_fixed_graph_is_evaluated_without_a_solve() {
    let mut g = FactorGraph::new();
    let a = g
        .add_variable(VariableKind::Euclidean(1), DVector::from_element(1, 2.0))
        .unwrap();
    g.set_fixed(a, true).unwrap();
    g.add_numeric_factor(&[a], 1, 1e-6, |v| Ok(v[0].clone()))
        .unwrap();
    let report = solve_local(&g, g.initial(), &Options::default()).unwrap();
    assert_eq!(report.cost, 2.0);
    assert!(report.iterations.is_empty());
    assert_eq!(report.x, g.initial());
}
fn rotation_factor(v: &DVector<f64>, target: SO3) -> (DVector<f64>, DMatrix<f64>) {
    let error = target.inverse().compose(read_so3(v).unwrap());
    let q = error.0.quaternion();
    let w = Vector3::new(q.i, q.j, q.k);
    let skew = Matrix3::new(0.0, -w.z, w.y, w.z, 0.0, -w.x, -w.y, w.x, 0.0);
    let sign = if q.w >= 0.0 { 1.0 } else { -1.0 };
    (
        DVector::from_column_slice((w * 2.0 * sign).as_slice()),
        DMatrix::from_column_slice(3, 3, ((Matrix3::identity() * q.w + skew) * sign).as_slice()),
    )
}
#[test]
fn analytic_so3_tangent_jacobian_matches_numeric_and_converges() {
    let target = SO3::exp(Vector3::new(0.2, -0.4, 1.0));
    let initial = so3_values(SO3::exp(Vector3::new(-0.4, 0.3, -0.8)));
    let mut analytic = FactorGraph::new();
    let a = analytic
        .add_variable(VariableKind::SO3, initial.clone())
        .unwrap();
    analytic
        .add_analytic_factor(&[a], 3, move |v| {
            let (r, j) = rotation_factor(&v[0], target);
            Ok((r, vec![j]))
        })
        .unwrap();
    let mut numeric = FactorGraph::new();
    let b = numeric.add_variable(VariableKind::SO3, initial).unwrap();
    numeric
        .add_numeric_factor(&[b], 3, 1e-6, move |v| Ok(rotation_factor(&v[0], target).0))
        .unwrap();
    let mut r = DVector::zeros(3);
    let mut ja = DMatrix::zeros(3, 3);
    let mut jn = DMatrix::zeros(3, 3);
    analytic
        .evaluate_local(&analytic.initial(), &mut r, &mut ja)
        .unwrap();
    numeric
        .evaluate_local(&numeric.initial(), &mut r, &mut jn)
        .unwrap();
    assert!((ja - jn).norm() < 1e-8);
    let report = solve_local(&analytic, analytic.initial(), &Options::default()).unwrap();
    assert!(report.cost < 1e-18);
    assert!(
        target
            .local(read_so3(&analytic.value(&report.x, a).unwrap()).unwrap())
            .norm()
            < 1e-8
    );
    assert!((analytic.value(&report.x, a).unwrap().norm() - 1.0).abs() < 1e-14);
}
#[test]
fn se3_graph_solves_chain_and_loop_with_fixed_anchor() {
    let truth = [
        SE3::exp(Tangent6::new(1.0, 0.2, -0.4, 0.1, -0.2, 0.3)),
        SE3::exp(Tangent6::new(2.0, 0.6, -0.1, 0.2, 0.1, 0.5)),
        SE3::exp(Tangent6::new(3.0, 0.1, 0.3, 0.3, -0.2, 0.6)),
    ];
    let mut g = FactorGraph::new();
    let mut ids = Vec::new();
    for (i, p) in truth.iter().enumerate() {
        let value = if i == 0 {
            *p
        } else {
            p.retract(Tangent6::new(0.3, -0.2, 0.1, 0.1, -0.1, 0.2))
        };
        ids.push(
            g.add_variable(VariableKind::SE3, se3_values(value))
                .unwrap(),
        );
    }
    g.set_fixed(ids[0], true).unwrap();
    for (a, b) in [(0, 1), (1, 2), (0, 2)] {
        let measured = truth[a].inverse().compose(truth[b]);
        g.add_numeric_factor(&[ids[a], ids[b]], 6, 1e-6, move |v| {
            let relative = read_se3(&v[0])?.inverse().compose(read_se3(&v[1])?);
            Ok(DVector::from_column_slice(
                measured.local(relative).as_slice(),
            ))
        })
        .unwrap();
    }
    assert_eq!(g.dimensions(), (21, 12, 18));
    let initial = g.initial();
    let report = solve_local(&g, initial.clone(), &Options::default()).unwrap();
    assert!(report.cost < 1e-18, "{report:?}");
    assert_eq!(
        g.value(&report.x, ids[0]).unwrap(),
        g.value(&initial, ids[0]).unwrap()
    );
    let sparse_report = solve_local(
        &g,
        initial.clone(),
        &Options {
            linear_solver: LinearSolver::SparseCg(ropt::sparse::CgOptions::default()),
            ..Options::default()
        },
    )
    .unwrap();
    assert!(sparse_report.cost < 1e-18);
    assert!((&sparse_report.x - &report.x).norm() < 1e-8);
    assert_eq!(
        g.value(&sparse_report.x, ids[0]).unwrap(),
        g.value(&initial, ids[0]).unwrap()
    );
    for i in 0..3 {
        assert!(
            truth[i]
                .local(read_se3(&g.value(&report.x, ids[i]).unwrap()).unwrap())
                .norm()
                < 1e-8
        );
    }
}
#[test]
fn graph_rejects_bad_shapes_and_nonfinite_callbacks() {
    let mut g = FactorGraph::new();
    assert_eq!(
        g.add_variable(VariableKind::Euclidean(0), DVector::zeros(0))
            .unwrap_err(),
        Error::InvalidDimensions
    );
    assert!(
        g.add_variable(VariableKind::SO3, DVector::zeros(4))
            .is_err()
    );
    let a = g
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    assert!(
        g.add_numeric_factor(&[a, a], 1, 1e-6, |_| Ok(DVector::zeros(1)))
            .is_err()
    );
    assert_eq!(
        g.add_numeric_factor(&[a], 1, 0.0, |_| Ok(DVector::zeros(1))),
        Err(Error::InvalidOptions)
    );
    g.add_analytic_factor(&[a], 1, |_| {
        Ok((DVector::zeros(1), vec![DMatrix::zeros(1, 2)]))
    })
    .unwrap();
    assert_eq!(
        solve_local(&g, g.initial(), &Options::default()).unwrap_err(),
        Error::InvalidDimensions
    );
    let mut bad = FactorGraph::new();
    let b = bad
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    bad.add_numeric_factor(&[b], 1, 1e-6, |_| Ok(DVector::from_element(1, f64::NAN)))
        .unwrap();
    assert_eq!(
        solve_local(&bad, bad.initial(), &Options::default()).unwrap_err(),
        Error::NonFinite
    );
}
#[test]
fn quaternion_retraction_preserves_unit_norm_over_many_updates() {
    let mut q = so3_values(SO3::exp(Vector3::zeros()));
    for i in 0..1000 {
        q = VariableKind::SO3
            .retract(
                &q,
                &DVector::from_vec(vec![0.01, (i as f64).sin() * 0.01, -0.02]),
            )
            .unwrap();
        assert!((q.norm() - 1.0).abs() < 1e-12);
    }
}

#[test]
fn variable_ids_cannot_cross_graphs() {
    let mut a = FactorGraph::new();
    let foreign = a
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    let mut b = FactorGraph::new();
    b.add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    assert_eq!(b.set_fixed(foreign, true), Err(Error::InvalidDimensions));
    assert_eq!(
        b.value(&b.initial(), foreign).unwrap_err(),
        Error::InvalidDimensions
    );
    assert_eq!(
        b.add_numeric_factor(&[foreign], 1, 1e-6, |v| Ok(v[0].clone())),
        Err(Error::InvalidDimensions)
    );
}
#[test]
fn quaternion_sign_representation_and_se2_wrap_are_consistent() {
    let p = SE3::exp(Tangent6::new(1.0, -0.5, 0.3, 0.2, -0.3, 1.0));
    let mut negative = se3_values(p);
    for i in 3..7 {
        negative[i] = -negative[i];
    }
    let decoded = read_se3(&negative).unwrap();
    assert!(p.local(decoded).norm() < 1e-12);
    let x = se2_values(SE2::exp(nalgebra::Vector3::new(
        0.1,
        0.2,
        std::f64::consts::PI - 0.01,
    )));
    let d = DVector::from_vec(vec![0.03, -0.04, 0.02]);
    let updated = VariableKind::SE2.retract(&x, &d).unwrap();
    assert!(
        (read_se2(&x).unwrap().local(read_se2(&updated).unwrap())
            - nalgebra::Vector3::from_column_slice(d.as_slice()))
        .norm()
            < 1e-12
    );
}
