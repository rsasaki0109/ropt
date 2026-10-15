use ropt::{factor::*, schur::*, sparse::*, *};
#[path = "../examples/support/ba.rs"]
mod ba;
fn fixture() -> (CsrMatrix, DMatrix<f64>, DVector<f64>) {
    let mut entries = Vec::new();
    for r in 0..12 {
        for c in 0..3 {
            entries.push((r, c, 0.1 + ((r + 1) * (c + 1)) as f64 * 0.03));
        }
        for k in 0..2 {
            entries.push((r, 3 + (r % 2) * 2 + k, 0.2 + ((r + 1 + k) as f64).cos()));
        }
    }
    let jacobian = CsrMatrix::from_triplets(12, 7, entries).unwrap();
    let mut j = DMatrix::zeros(12, 7);
    for (r, c, v) in jacobian.entries() {
        j[(r, c)] = v;
    }
    let mut h = j.transpose() * j;
    for i in 0..7 {
        h[(i, i)] += 0.05 * h[(i, i)].max(1.0);
    }
    (
        jacobian,
        h,
        DVector::from_vec(vec![0.1, 0.3, -0.2, 1.0, -0.5, 0.2, 0.7]),
    )
}
#[test]
fn reduced_operator_rhs_diagonal_and_backsub_match_dense_elimination() {
    let (j, h, rhs) = fixture();
    let system = BlockSchurSystem::new(&j, 0.05, 3, 2).unwrap();
    let a = h.view((0, 0), (3, 3)).into_owned();
    let b = h.view((0, 3), (3, 4)).into_owned();
    let c = h.view((3, 3), (4, 4)).into_owned().cholesky().unwrap();
    let reduced = a - &b * c.solve(&b.transpose());
    let reduced_rhs = rhs.rows(0, 3) - &b * c.solve(&rhs.rows(3, 4).into_owned());
    let x = DVector::from_vec(vec![0.2, -0.4, 0.3]);
    assert!((system.apply(&x).unwrap() - &reduced * &x).norm() < 1e-11);
    assert!((system.diagonal().unwrap() - reduced.diagonal()).norm() < 1e-11);
    assert!((system.reduced_rhs(&rhs).unwrap() - reduced_rhs).norm() < 1e-11);
    let full = system.back_substitute(&rhs, &x).unwrap();
    let expected = c.solve(&(rhs.rows(3, 4) - b.transpose() * x));
    assert!((full.rows(3, 4) - expected).norm() < 1e-11);
    assert_eq!(system.eliminated_blocks(), 2);
    assert_eq!(system.cross_rows(), 6);
}
#[test]
fn block_schur_solution_matches_full_normal_solve_and_residual() {
    let (j, h, rhs) = fixture();
    let result = block_schur_solve(&j, &rhs, 0.05, 3, 2, &CgOptions::default()).unwrap();
    let expected = h.clone().cholesky().unwrap().solve(&rhs);
    assert!((&result.x - expected).norm() < 1e-9);
    assert!((&h * &result.x - &rhs).norm() < 1e-10);
    assert!(result.residual_norm < 1e-10);
}
#[test]
fn coupled_blocks_and_invalid_partitions_do_not_silently_pass() {
    let j = CsrMatrix::from_triplets(1, 3, vec![(0, 0, 1.0), (0, 1, 1.0), (0, 2, 1.0)]).unwrap();
    assert!(matches!(
        BlockSchurSystem::new(&j, 0.1, 1, 1),
        Err(Error::Evaluation(_))
    ));
    assert!(matches!(
        BlockSchurSystem::new(&j, 0.1, 1, 3),
        Err(Error::InvalidDimensions)
    ));
    assert!(matches!(
        BlockSchurSystem::new(&j, 0.1, 0, 1),
        Err(Error::InvalidDimensions)
    ));
    let disconnected = CsrMatrix::from_triplets(1, 3, vec![(0, 0, 1.0)]).unwrap();
    assert!(matches!(
        BlockSchurSystem::new(&disconnected, 0.0, 1, 2),
        Err(Error::LinearSolve)
    ));
    let solve = block_schur_solve(
        &disconnected,
        &DVector::from_vec(vec![1.0, 0.2, -0.1]),
        0.1,
        1,
        2,
        &CgOptions::default(),
    )
    .unwrap();
    assert!(solve.residual_norm < 1e-10);
}
#[test]
fn graph_layout_checks_fixed_blocks_and_connectivity() {
    let mut graph = FactorGraph::new();
    let camera = graph
        .add_variable(VariableKind::Euclidean(2), DVector::zeros(2))
        .unwrap();
    let p = graph
        .add_variable(VariableKind::Euclidean(3), DVector::zeros(3))
        .unwrap();
    let q = graph
        .add_variable(VariableKind::Euclidean(3), DVector::zeros(3))
        .unwrap();
    assert_eq!(graph.schur_layout(&[p, q]).unwrap(), (2, 3));
    assert!(graph.schur_layout(&[p]).is_err());
    assert!(graph.schur_layout(&[p, p, q]).is_err());
    graph.set_fixed(p, true).unwrap();
    assert_eq!(graph.schur_layout(&[p, q]).unwrap(), (2, 3));
    graph.set_fixed(p, false).unwrap();
    graph
        .add_analytic_factor(&[camera, p, q], 1, |_| {
            Ok((
                DVector::zeros(1),
                vec![
                    DMatrix::zeros(1, 2),
                    DMatrix::zeros(1, 3),
                    DMatrix::zeros(1, 3),
                ],
            ))
        })
        .unwrap();
    assert!(matches!(
        graph.schur_layout(&[p, q]),
        Err(Error::Evaluation(_))
    ));
}
#[test]
fn analytic_bundle_adjustment_recovers_pose_and_points_with_fixed_anchors() {
    let scene = ba::scene().unwrap();
    let initial = scene.graph.initial();
    let initial_rms = scene.rms(&initial).unwrap();
    assert!(initial_rms > 1.0);
    let (retained, block_size) = scene.graph.schur_layout(&scene.points).unwrap();
    assert_eq!((retained, block_size), (12, 3));
    let dense = solve_local(&scene.graph, initial.clone(), &Options::default()).unwrap();
    let schur = solve_local(
        &scene.graph,
        initial.clone(),
        &Options {
            linear_solver: LinearSolver::BlockSchur {
                retained,
                block_size,
                cg: CgOptions::default(),
            },
            ..Options::default()
        },
    )
    .unwrap();
    assert!(scene.rms(&schur.x).unwrap() < 1e-7);
    assert!(scene.max_state_error(&schur.x).unwrap() < 1e-6);
    assert!((&dense.x - &schur.x).norm() < 1e-6);
    assert_ne!(schur.termination, Termination::MaxIterations);
    for id in &scene.cameras[..2] {
        assert_eq!(
            scene.graph.value(&schur.x, *id).unwrap(),
            scene.graph.value(&initial, *id).unwrap()
        );
    }
}
#[test]
fn block_schur_linear_iteration_failure_is_propagated() {
    let scene = ba::scene().unwrap();
    let (retained, block_size) = scene.graph.schur_layout(&scene.points).unwrap();
    let result = solve_local(
        &scene.graph,
        scene.graph.initial(),
        &Options {
            linear_solver: LinearSolver::BlockSchur {
                retained,
                block_size,
                cg: CgOptions {
                    max_iterations: 1,
                    ..CgOptions::default()
                },
            },
            ..Options::default()
        },
    );
    assert_eq!(
        result.unwrap_err(),
        Error::LinearIterationLimit { iterations: 1 }
    );
}

#[test]
fn noisy_robust_ba_reduces_error_and_matches_dense_solution() {
    let scene = ba::scene_with_noise(0.2).unwrap();
    let initial = scene.graph.initial();
    let (retained, block_size) = scene.graph.schur_layout(&scene.points).unwrap();
    let options = Options {
        loss: Loss::Huber(2.0),
        max_iterations: 100,
        ..Options::default()
    };
    let dense = solve_local(&scene.graph, initial.clone(), &options).unwrap();
    let schur = solve_local(
        &scene.graph,
        initial.clone(),
        &Options {
            linear_solver: LinearSolver::BlockSchur {
                retained,
                block_size,
                cg: CgOptions::default(),
            },
            ..options
        },
    )
    .unwrap();
    assert!(scene.rms(&schur.x).unwrap() < 0.3);
    assert!(scene.rms(&schur.x).unwrap() < scene.rms(&initial).unwrap() * 0.1);
    assert!((&dense.x - &schur.x).norm() < 1e-5);
    assert!((dense.cost - schur.cost).abs() < 1e-9);
    assert_ne!(schur.termination, Termination::MaxIterations);
}
