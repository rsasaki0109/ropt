use ropt::{factor::*, noise::NoiseModel, sparse::*, sparse_direct::*, *};
fn csr(j: &DMatrix<f64>) -> CsrMatrix {
    let mut entries = Vec::new();
    for r in 0..j.nrows() {
        for c in 0..j.ncols() {
            if j[(r, c)] != 0.0 {
                entries.push((r, c, j[(r, c)]));
            }
        }
    }
    CsrMatrix::from_triplets(j.nrows(), j.ncols(), entries).unwrap()
}
fn hessian(j: &DMatrix<f64>, lambda: f64) -> DMatrix<f64> {
    let mut h = j.transpose() * j;
    for k in 0..h.nrows() {
        h[(k, k)] += lambda * h[(k, k)].max(1.0);
    }
    h
}
#[test]
fn rectangular_systems_match_dense_cholesky_with_both_orderings() {
    let j = DMatrix::from_fn(11, 7, |r, c| {
        if r == c {
            1.0 + c as f64 * 0.2
        } else if (r + 2 * c) % 4 == 0 {
            ((r + c + 1) as f64).sin() * 0.4
        } else {
            0.0
        }
    });
    let sparse = csr(&j);
    let b = DVector::from_fn(7, |k, _| (k as f64 * 0.7).cos());
    for lambda in [0.0, 1e-6, 0.1] {
        for ordering in [SparseOrdering::Natural, SparseOrdering::MinimumDegree] {
            let h = hessian(&j, lambda);
            let options = SparseLdlOptions {
                ordering,
                ..SparseLdlOptions::default()
            };
            let factor = SparseLdlFactor::factor_normal(&sparse, lambda, &options).unwrap();
            let report = solve_normal(&sparse, &b, lambda, &options).unwrap();
            assert!((&report.x - h.clone().cholesky().unwrap().solve(&b)).norm() < 1e-11);
            assert!((&b - &h * &report.x).norm() < 1e-11);
            let actual = (&b
                - NormalOperator::new(&sparse, lambda)
                    .unwrap()
                    .apply(&report.x)
                    .unwrap())
            .norm();
            assert!((actual - report.residual_norm).abs() < 1e-14);
            assert_eq!(factor.dimension(), 7);
            let mut permutation = factor.permutation().to_vec();
            permutation.sort_unstable();
            assert_eq!(permutation, (0..7).collect::<Vec<_>>());
            let other = &b * 1.7;
            assert!(
                (factor.solve(&other).unwrap() - h.cholesky().unwrap().solve(&other)).norm()
                    < 1e-11
            );
        }
    }
}
fn star(n: usize) -> CsrMatrix {
    let mut entries = vec![(0, 0, 1.0)];
    for k in 1..n {
        entries.push((k, 0, -1.0));
        entries.push((k, k, 1.0));
    }
    CsrMatrix::from_triplets(n, n, entries).unwrap()
}
#[test]
fn minimum_degree_reduces_star_fill_and_checks_budgets() {
    let j = star(12);
    let natural = SparseLdlOptions {
        ordering: SparseOrdering::Natural,
        ..SparseLdlOptions::default()
    };
    let a = SparseLdlFactor::factor_normal(&j, 0.1, &natural).unwrap();
    let b = SparseLdlFactor::factor_normal(&j, 0.1, &SparseLdlOptions::default()).unwrap();
    assert_eq!(a.normal_entries(), 23);
    assert_eq!(a.factor_entries(), 78);
    assert_eq!(b.normal_entries(), 23);
    assert_eq!(b.factor_entries(), 23);
    assert_ne!(b.permutation(), (0..12).collect::<Vec<_>>());
    let bounded = SparseLdlOptions {
        max_entries: 23,
        ..SparseLdlOptions::default()
    };
    SparseLdlFactor::factor_normal(&j, 0.1, &bounded).unwrap();
    assert_eq!(
        SparseLdlFactor::factor_normal(
            &j,
            0.1,
            &SparseLdlOptions {
                ordering: SparseOrdering::Natural,
                ..bounded
            }
        )
        .unwrap_err(),
        Error::SparseFillLimit { limit: 23 }
    );
    assert_eq!(
        SparseLdlFactor::factor_normal(
            &j,
            0.1,
            &SparseLdlOptions {
                max_entries: 22,
                ..bounded
            }
        )
        .unwrap_err(),
        Error::SparseFillLimit { limit: 22 }
    );
}
#[test]
fn singular_and_uncovered_columns_fail_without_damping_and_solve_with_damping() {
    let j = DMatrix::from_row_slice(2, 3, &[1.0, 1.0, 0.0, 2.0, 2.0, 0.0]);
    let sparse = csr(&j);
    assert_eq!(
        SparseLdlFactor::factor_normal(&sparse, 0.0, &SparseLdlOptions::default()).unwrap_err(),
        Error::LinearSolve
    );
    let b = DVector::from_vec(vec![0.2, 0.7, -0.3]);
    let report = solve_normal(&sparse, &b, 0.01, &SparseLdlOptions::default()).unwrap();
    assert!((report.x - hessian(&j, 0.01).cholesky().unwrap().solve(&b)).norm() < 1e-10);
    let zero = CsrMatrix::from_triplets(0, 3, vec![]).unwrap();
    let report = solve_normal(&zero, &b, 0.1, &SparseLdlOptions::default()).unwrap();
    assert!((report.x - &b / 0.1).norm() < 1e-12);
}
#[test]
fn invalid_input_options_and_arithmetic_fail_explicitly() {
    let j = star(3);
    let options = SparseLdlOptions::default();
    for lambda in [-1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            SparseLdlFactor::factor_normal(&j, lambda, &options).unwrap_err(),
            Error::InvalidOptions
        );
    }
    for bad in [
        SparseLdlOptions {
            max_entries: 0,
            ..options
        },
        SparseLdlOptions {
            relative_tolerance: -1.0,
            ..options
        },
        SparseLdlOptions {
            absolute_tolerance: f64::NAN,
            ..options
        },
    ] {
        assert_eq!(
            SparseLdlFactor::factor_normal(&j, 0.1, &bad).unwrap_err(),
            Error::InvalidOptions
        );
    }
    assert_eq!(
        SparseLdlFactor::factor_normal(
            &CsrMatrix::from_triplets(2, 0, vec![]).unwrap(),
            0.1,
            &options
        )
        .unwrap_err(),
        Error::InvalidDimensions
    );
    assert_eq!(
        solve_normal(&j, &DVector::zeros(2), 0.1, &options).unwrap_err(),
        Error::InvalidDimensions
    );
    assert_eq!(
        solve_normal(&j, &DVector::from_element(3, f64::INFINITY), 0.1, &options).unwrap_err(),
        Error::NonFinite
    );
    let factor = SparseLdlFactor::factor_normal(&j, 0.1, &options).unwrap();
    assert_eq!(
        factor.solve(&DVector::zeros(4)).unwrap_err(),
        Error::InvalidDimensions
    );
    assert_eq!(
        factor
            .solve(&DVector::from_element(3, f64::NAN))
            .unwrap_err(),
        Error::NonFinite
    );
    let huge = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 1e200)]).unwrap();
    assert_eq!(
        SparseLdlFactor::factor_normal(&huge, 0.1, &options).unwrap_err(),
        Error::NonFinite
    );
    let tiny = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 1e-154)]).unwrap();
    assert_eq!(
        SparseLdlFactor::factor_normal(&tiny, 0.0, &options)
            .unwrap()
            .solve(&DVector::from_element(1, 1e200))
            .unwrap_err(),
        Error::NonFinite
    );
}
#[test]
fn checked_solve_rejects_nonzero_true_residual_at_zero_tolerance() {
    let j = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 2.0), (0, 1, 1.0), (1, 1, 1.0)]).unwrap();
    let options = SparseLdlOptions {
        relative_tolerance: 0.0,
        absolute_tolerance: 0.0,
        ..SparseLdlOptions::default()
    };
    let error = solve_normal(&j, &DVector::from_vec(vec![0.1, 0.3]), 0.0, &options).unwrap_err();
    assert!(
        matches!(error,Error::LinearResidualTooLarge {residual_norm,tolerance} if residual_norm>0.0 && tolerance==0.0)
    );
}
#[test]
fn correlated_joint_robust_graph_matches_dense_and_preserves_fixed_state() {
    let mut graph = FactorGraph::new();
    let fixed = graph
        .add_variable(VariableKind::Euclidean(2), DVector::zeros(2))
        .unwrap();
    graph.set_fixed(fixed, true).unwrap();
    let x = graph
        .add_variable(VariableKind::Euclidean(2), DVector::zeros(2))
        .unwrap();
    let noise =
        NoiseModel::from_covariance(DMatrix::from_row_slice(2, 2, &[0.4, 0.1, 0.1, 0.3])).unwrap();
    for k in 0..20 {
        let obs = if k == 19 {
            DVector::from_vec(vec![20.0, -30.0])
        } else {
            DVector::from_vec(vec![
                1.0 + (k as f64).sin() * 0.02,
                2.0 + (k as f64).cos() * 0.02,
            ])
        };
        graph
            .add_analytic_factor_with_options(
                &[fixed, x],
                2,
                FactorOptions {
                    noise: noise.clone(),
                    loss: Some(Loss::Huber(1.0)),
                },
                move |v| {
                    Ok((
                        &v[1] - &v[0] - &obs,
                        vec![-DMatrix::identity(2, 2), DMatrix::identity(2, 2)],
                    ))
                },
            )
            .unwrap();
    }
    let dense = solve_local(&graph, graph.initial(), &Options::default()).unwrap();
    let direct = solve_local(
        &graph,
        graph.initial(),
        &Options {
            linear_solver: LinearSolver::SparseLdl(SparseLdlOptions::default()),
            ..Options::default()
        },
    )
    .unwrap();
    assert!((dense.cost - direct.cost).abs() < 1e-10);
    assert!((&dense.x - &direct.x).norm() < 1e-8);
    assert_eq!(graph.value(&direct.x, fixed).unwrap(), DVector::zeros(2));
    assert!(!matches!(
        direct.termination,
        Termination::NoProgress | Termination::MaxIterations
    ));
    assert!(direct.iterations.iter().all(|i| i.linear_iterations == 0));
    graph.set_fixed(x, true).unwrap();
    let unchanged = solve_local(
        &graph,
        graph.initial(),
        &Options {
            linear_solver: LinearSolver::SparseLdl(SparseLdlOptions::default()),
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(unchanged.x, graph.initial());
}
struct SparseOnly;
impl LocalProblem for SparseOnly {
    fn dimensions(&self) -> (usize, usize, usize) {
        (2, 2, 2)
    }
    fn evaluate_local(
        &self,
        _: &DVector<f64>,
        _: &mut DVector<f64>,
        _: &mut DMatrix<f64>,
    ) -> Result<(), Error> {
        Err(Error::Evaluation("dense provider called".into()))
    }
    fn evaluate_sparse(&self, x: &DVector<f64>) -> Result<(DVector<f64>, CsrMatrix), Error> {
        Ok((
            x - DVector::from_vec(vec![1.0, -2.0]),
            CsrMatrix::from_triplets(2, 2, vec![(0, 0, 1.0), (1, 1, 1.0)])?,
        ))
    }
    fn retract(&self, x: &DVector<f64>, d: &DVector<f64>) -> Result<DVector<f64>, Error> {
        Ok(x + d)
    }
}
#[test]
fn sparse_lm_requires_no_dense_provider_and_propagates_options_and_fill_errors() {
    let options = Options {
        linear_solver: LinearSolver::SparseLdl(SparseLdlOptions::default()),
        ..Options::default()
    };
    let report = solve_local(&SparseOnly, DVector::zeros(2), &options).unwrap();
    assert!((report.x - DVector::from_vec(vec![1.0, -2.0])).norm() < 1e-8);
    assert_eq!(
        solve_local(
            &SparseOnly,
            DVector::zeros(2),
            &Options {
                algorithm: Algorithm::GaussNewton,
                ..options.clone()
            }
        )
        .unwrap_err(),
        Error::InvalidOptions
    );
    assert_eq!(
        solve_local(
            &SparseOnly,
            DVector::zeros(2),
            &Options {
                linear_solver: LinearSolver::SparseLdl(SparseLdlOptions {
                    max_entries: 1,
                    ..SparseLdlOptions::default()
                }),
                ..options
            }
        )
        .unwrap_err(),
        Error::SparseFillLimit { limit: 1 }
    );
}

#[test]
fn workspace_reuses_symbolic_analysis_for_changed_values_damping_and_rhs() {
    let base = DMatrix::from_row_slice(4, 3, &[2., 1., 0., 0., 3., 1., 1., 0., 2., 0., 1., 1.]);
    let b = DVector::from_vec(vec![0.3, -0.2, 0.7]);
    for ordering in [SparseOrdering::Natural, SparseOrdering::MinimumDegree] {
        let mut workspace = SparseLdlWorkspace::new(SparseLdlOptions {
            ordering,
            ..Default::default()
        })
        .unwrap();
        for (scale, lambda) in [(1., 0.01), (0.7, 0.1), (2., 0.), (1.2, 1e-5)] {
            let j = &base * scale;
            let sparse = csr(&j);
            let h = hessian(&j, lambda);
            let mut x = DVector::from_element(3, f64::NAN);
            let stats = workspace.solve_into(&sparse, &b, lambda, &mut x).unwrap();
            assert_eq!(workspace.analyses(), 1);
            assert!((&x - h.clone().cholesky().unwrap().solve(&b)).norm() < 1e-11);
            assert!((&b - &h * &x).norm() < 1e-11);
            assert!(stats.residual_norm < 1e-11);
            let other = &b * 1.7;
            x.fill(f64::NAN);
            workspace.solve_factored_into(&other, &mut x).unwrap();
            assert!((x - h.cholesky().unwrap().solve(&other)).norm() < 1e-11);
            assert_eq!(workspace.analyses(), 1);
        }
    }
}
#[test]
fn symbolic_pattern_checks_shape_offsets_and_indices_not_only_nonzero_count() {
    let a = star(3);
    let b = CsrMatrix::from_triplets(
        3,
        3,
        vec![(0, 0, 1.), (1, 0, -1.), (1, 2, 1.), (2, 0, -1.), (2, 1, 1.)],
    )
    .unwrap();
    assert_eq!(a.nnz(), b.nnz());
    let symbolic = SparseLdlSymbolic::analyze(&a, &Default::default()).unwrap();
    assert!(symbolic.matches_pattern(&a));
    assert!(!symbolic.matches_pattern(&b));
    assert!(symbolic.factor_normal(&b, 0.1).is_err());
    let mut workspace = SparseLdlWorkspace::new(Default::default()).unwrap();
    for (index, j) in [
        a,
        b,
        CsrMatrix::from_triplets(
            4,
            3,
            vec![(0, 0, 1.), (1, 0, -1.), (1, 2, 1.), (2, 0, -1.), (2, 1, 1.)],
        )
        .unwrap(),
        CsrMatrix::from_triplets(2, 2, vec![(0, 0, 1.), (1, 1, 1.)]).unwrap(),
    ]
    .into_iter()
    .enumerate()
    {
        let b = DVector::from_element(j.shape().1, 0.2);
        let actual = workspace.solve(&j, &b, 0.1).unwrap();
        let mut dense = DMatrix::zeros(j.shape().0, j.shape().1);
        for (r, c, v) in j.entries() {
            dense[(r, c)] = v;
        }
        assert!((actual.x - hessian(&dense, 0.1).cholesky().unwrap().solve(&b)).norm() < 1e-11);
        assert_eq!(workspace.analyses(), index + 1);
    }
}
#[test]
fn failed_refactor_cannot_expose_previous_numeric_factors_and_can_recover() {
    let mut workspace = SparseLdlWorkspace::new(SparseLdlOptions {
        max_entries: 5,
        ..Default::default()
    })
    .unwrap();
    let j = star(3);
    let b = DVector::from_vec(vec![1., 2., 3.]);
    workspace.solve(&j, &b, 0.1).unwrap();
    assert_eq!(
        workspace.refactor(&j, f64::NAN).unwrap_err(),
        Error::InvalidOptions
    );
    assert_eq!(
        workspace
            .solve_factored_into(&b, &mut DVector::zeros(3))
            .unwrap_err(),
        Error::LinearSolve
    );
    workspace.solve(&j, &b, 0.1).unwrap();
    let singular = CsrMatrix::from_triplets(3, 3, vec![(0, 0, 1.), (1, 1, 1.)]).unwrap();
    assert_eq!(
        workspace.refactor(&singular, 0.).unwrap_err(),
        Error::LinearSolve
    );
    assert_eq!(
        workspace
            .solve_factored_into(&b, &mut DVector::zeros(3))
            .unwrap_err(),
        Error::LinearSolve
    );
    let dense = CsrMatrix::from_triplets(1, 3, vec![(0, 0, 1.), (0, 1, 1.), (0, 2, 1.)]).unwrap();
    assert_eq!(
        workspace.refactor(&dense, 0.1).unwrap_err(),
        Error::SparseFillLimit { limit: 5 }
    );
    assert_eq!(
        workspace
            .solve_factored_into(&b, &mut DVector::zeros(3))
            .unwrap_err(),
        Error::LinearSolve
    );
    let restored = workspace.solve(&j, &b, 0.1).unwrap();
    let fresh = solve_normal(
        &j,
        &b,
        0.1,
        &SparseLdlOptions {
            max_entries: 5,
            ..Default::default()
        },
    )
    .unwrap();
    assert!((restored.x - fresh.x).norm() < 1e-12);
}
struct ChangingPattern(std::cell::Cell<u8>);
impl LocalProblem for ChangingPattern {
    fn dimensions(&self) -> (usize, usize, usize) {
        (2, 2, 2)
    }
    fn evaluate_local(
        &self,
        x: &DVector<f64>,
        r: &mut DVector<f64>,
        j: &mut DMatrix<f64>,
    ) -> Result<(), Error> {
        r[0] = 10. * (x[1] - x[0] * x[0]);
        r[1] = 1. - x[0];
        j[(0, 0)] = -20. * x[0];
        j[(0, 1)] = 10.;
        j[(1, 0)] = -1.;
        j[(1, 1)] = 0.;
        Ok(())
    }
    fn evaluate_sparse(&self, x: &DVector<f64>) -> Result<(DVector<f64>, CsrMatrix), Error> {
        let mut r = DVector::zeros(2);
        let mut j = DMatrix::zeros(2, 2);
        self.evaluate_local(x, &mut r, &mut j)?;
        let j = csr(&j);
        self.0.set(self.0.get() | if j.nnz() == 2 { 1 } else { 2 });
        Ok((r, j))
    }
    fn retract(&self, x: &DVector<f64>, d: &DVector<f64>) -> Result<DVector<f64>, Error> {
        Ok(x + d)
    }
}
#[test]
fn lm_reanalyzes_when_exact_zero_derivatives_appear_or_disappear() {
    let problem = ChangingPattern(std::cell::Cell::new(0));
    let sparse = solve_local(
        &problem,
        DVector::zeros(2),
        &Options {
            linear_solver: LinearSolver::SparseLdl(Default::default()),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(problem.0.get(), 3);
    let dense = solve_local(&problem, DVector::zeros(2), &Options::default()).unwrap();
    assert!((&sparse.x - &dense.x).norm() < 1e-7);
    assert!((sparse.x - DVector::from_element(2, 1.)).norm() < 1e-7);
    assert!(sparse.cost < 1e-14);
}
#[test]
fn optional_plan_budget_uses_searches_without_changing_numeric_results() {
    let j = DMatrix::from_fn(7, 6, |r, c| {
        if r == c {
            2.
        } else {
            0.1 * ((r + c + 1) as f64).cos()
        }
    });
    let j = csr(&j);
    let planned = SparseLdlSymbolic::analyze(&j, &Default::default()).unwrap();
    assert!(planned.has_update_plan());
    assert!(planned.update_entries() > 0);
    let bounded = SparseLdlSymbolic::analyze(
        &j,
        &SparseLdlOptions {
            max_entries: 21,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!bounded.has_update_plan());
    assert_eq!(bounded.assembly_entries(), 0);
    assert_eq!(planned.factor_entries(), bounded.factor_entries());
    let b = DVector::from_vec(vec![0.1, 0.3, -0.1, 0.2, 0.4, -0.3]);
    let a = planned.factor_normal(&j, 0.1).unwrap().solve(&b).unwrap();
    let z = bounded.factor_normal(&j, 0.1).unwrap().solve(&b).unwrap();
    assert_eq!(a, z); // identical arithmetic, not an approximate low-fill solve
}
#[test]
fn block_ordering_keeps_coordinates_together_reduces_fill_and_matches_dense() {
    let mut j = DMatrix::zeros(12, 12);
    for c in 0..3 {
        j[(c, c)] = 1.;
    }
    let w = DMatrix::from_row_slice(3, 3, &[2., 0.3, 0.2, 0., 2., 0.2, 0., 0., 1.]);
    for block in 1..4 {
        j.view_mut((3 * block, 0), (3, 3)).copy_from(&(-&w));
        j.view_mut((3 * block, 3 * block), (3, 3)).copy_from(&w);
    }
    let sparse = csr(&j);
    let options = SparseLdlOptions {
        ordering: SparseOrdering::BlockMinimumDegree { block_size: 3 },
        ..Default::default()
    };
    let block = SparseLdlSymbolic::analyze(&sparse, &options).unwrap();
    let natural = SparseLdlSymbolic::analyze(
        &sparse,
        &SparseLdlOptions {
            ordering: SparseOrdering::Natural,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(block.factor_entries() < natural.factor_entries());
    for chunk in block.permutation().chunks(3) {
        assert_eq!(chunk[0] % 3, 0);
        assert_eq!(chunk[1], chunk[0] + 1);
        assert_eq!(chunk[2], chunk[0] + 2);
    }
    let b = DVector::from_fn(12, |k, _| (k as f64).cos());
    let actual = solve_normal(&sparse, &b, 0.1, &options).unwrap();
    assert!((actual.x - hessian(&j, 0.1).cholesky().unwrap().solve(&b)).norm() < 1e-11);
    let scalar = SparseLdlSymbolic::analyze(&sparse, &Default::default()).unwrap();
    let one = SparseLdlSymbolic::analyze(
        &sparse,
        &SparseLdlOptions {
            ordering: SparseOrdering::BlockMinimumDegree { block_size: 1 },
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(scalar.permutation(), one.permutation());
    assert_eq!(
        SparseLdlSymbolic::analyze(&star(4), &options).unwrap_err(),
        Error::InvalidDimensions
    );
    assert_eq!(
        SparseLdlSymbolic::analyze(
            &sparse,
            &SparseLdlOptions {
                ordering: SparseOrdering::BlockMinimumDegree { block_size: 0 },
                ..Default::default()
            }
        )
        .unwrap_err(),
        Error::InvalidOptions
    );
}

#[test]
fn refinement_corrects_original_normal_residual_without_relaxing_tolerance() {
    // Weakly anchored chain: elimination is accurate enough to refine, but its
    // initial original-operator residual exceeds the requested absolute bound.
    let n = 96;
    let mut entries = vec![(0, 0, 0.01)];
    for k in 1..n {
        entries.extend([(k, k - 1, -1.), (k, k, 1.)]);
    }
    let j = CsrMatrix::from_triplets(n, n, entries).unwrap();
    let b = DVector::from_fn(n, |k, _| (k as f64 * 0.13).cos());
    let operator = NormalOperator::new(&j, 1e-8).unwrap();
    let mut workspace = SparseLdlWorkspace::new(Default::default()).unwrap();
    workspace.refactor(&j, 1e-8).unwrap();
    let mut raw = DVector::zeros(n);
    workspace.solve_factored_into(&b, &mut raw).unwrap();
    let raw_norm = (&b - operator.apply(&raw).unwrap()).norm();
    let tolerance = raw_norm * 0.75;
    assert!(tolerance > 0.);
    let mut checked = SparseLdlWorkspace::new(SparseLdlOptions {
        relative_tolerance: 0.,
        absolute_tolerance: tolerance,
        ..Default::default()
    })
    .unwrap();
    let report = checked.solve(&j, &b, 1e-8).unwrap();
    assert!((1..=3).contains(&report.refinement_steps));
    let actual = (&b - operator.apply(&report.x).unwrap()).norm();
    assert!(actual <= tolerance);
    assert_eq!(actual, report.residual_norm);
    assert_eq!(checked.analyses(), 1);
    // Ordinary solves retain the initial factor result when it already meets
    // the bound; refinement does not silently change all existing results.
    let ordinary = workspace.solve(&j, &b, 1e-8).unwrap();
    assert_eq!(ordinary.refinement_steps, 0);
    assert_eq!(ordinary.x, raw);
}
