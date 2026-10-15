use ropt::{sparse::CsrMatrix, sparse_qr::solve_least_squares, *};
fn csr(j: &DMatrix<f64>) -> Result<CsrMatrix, Error> {
    CsrMatrix::from_triplets(
        j.nrows(),
        j.ncols(),
        (0..j.nrows())
            .flat_map(|r| {
                (0..j.ncols()).filter_map(move |c| (j[(r, c)] != 0.).then_some((r, c, j[(r, c)])))
            })
            .collect(),
    )
}

#[test]
fn ordered_sparse_steps_match_independent_dense_qr_for_gn_and_scaled_lm() {
    for scale in [1e-80, 1., 1e80] {
        let j = DMatrix::from_row_slice(
            5,
            3,
            &[3., 1., 0., 0., 2., 1., 1., 0., 4., 2., -1., 1., 0., 1., -2.],
        ) * scale;
        let r = DVector::from_vec(vec![1., -2., 3., 0.5, -1.]) * scale;
        let csr = csr(&j).unwrap();
        for damping in [0., 1e-3, 1.] {
            let expected = dense::qr_step(&j, &r, damping, &Default::default()).unwrap();
            for ordering in [
                SparseOrdering::Natural,
                SparseOrdering::MinimumDegree,
                SparseOrdering::BlockMinimumDegree { block_size: 3 },
            ] {
                let actual = solve_least_squares(
                    &csr,
                    &r,
                    damping,
                    &SparseQrOptions {
                        ordering,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert!(
                    (&actual - &expected).norm() < 1e-10 * expected.norm().max(1.),
                    "{scale} {damping} {ordering:?}"
                );
            }
        }
    }
}

#[test]
fn sparse_qr_keeps_rank_budget_dimension_and_finite_checks() {
    let rank_one = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 1.), (0, 1, 1.)]).unwrap();
    let r = DVector::from_vec(vec![1., 2.]);
    assert!(matches!(
        solve_least_squares(&rank_one, &r, 0., &Default::default()),
        Err(Error::RankDeficient {
            rank: 1,
            parameters: 2
        })
    ));
    let x = solve_least_squares(&rank_one, &r, 0.1, &Default::default()).unwrap();
    assert!((x[0] + 1. / 2.1).abs() < 1e-12 && (x[1] + 1. / 2.1).abs() < 1e-12);
    assert!(matches!(
        solve_least_squares(
            &rank_one,
            &r,
            0.,
            &SparseQrOptions {
                max_entries: 1,
                ..Default::default()
            }
        ),
        Err(Error::SparseFillLimit { limit: 1 })
    ));
    assert_eq!(
        solve_least_squares(&rank_one, &DVector::zeros(1), 0., &Default::default()),
        Err(Error::InvalidDimensions)
    );
    assert_eq!(
        solve_least_squares(
            &rank_one,
            &DVector::from_vec(vec![f64::NAN, 0.]),
            0.,
            &Default::default()
        ),
        Err(Error::NonFinite)
    );
    for bad in [-1., f64::NAN] {
        assert_eq!(
            solve_least_squares(&rank_one, &r, bad, &Default::default()),
            Err(Error::InvalidOptions)
        );
    }
    for bad in [0., 1., f64::NAN] {
        assert_eq!(
            solve_least_squares(
                &rank_one,
                &r,
                0.,
                &SparseQrOptions {
                    relative_rank_tolerance: bad,
                    ..Default::default()
                }
            ),
            Err(Error::InvalidOptions)
        );
    }
}

struct CurvedConstraint;
impl LocalProblem for CurvedConstraint {
    fn dimensions(&self) -> (usize, usize, usize) {
        (2, 2, 2)
    }
    fn evaluate_local(
        &self,
        x: &DVector<f64>,
        r: &mut DVector<f64>,
        j: &mut DMatrix<f64>,
    ) -> Result<(), Error> {
        r[0] = 1e6 * (x[1] - x[0].sin());
        r[1] = x[0] - 1.;
        j[(0, 0)] = -1e6 * x[0].cos();
        j[(0, 1)] = 1e6;
        j[(1, 0)] = 1.;
        j[(1, 1)] = 0.;
        Ok(())
    }
    fn evaluate_sparse(&self, x: &DVector<f64>) -> Result<(DVector<f64>, CsrMatrix), Error> {
        let mut r = DVector::zeros(2);
        let mut j = DMatrix::zeros(2, 2);
        self.evaluate_local(x, &mut r, &mut j)?;
        Ok((r, csr(&j)?))
    }
    fn retract(&self, x: &DVector<f64>, d: &DVector<f64>) -> Result<DVector<f64>, Error> {
        Ok(x + d)
    }
}

#[test]
fn explicit_nonmonotonic_gn_crosses_a_stiff_curve_without_claiming_uphill_convergence() {
    for backend in [
        LinearSolver::DenseQr(Default::default()),
        LinearSolver::SparseQr(Default::default()),
    ] {
        let options = Options {
            algorithm: Algorithm::GaussNewton,
            linear_solver: backend,
            gradient_norm: GradientNorm::ColumnNormalized,
            gradient_tolerance: 1e-8,
            ..Default::default()
        };
        let guarded = solve_local(&CurvedConstraint, DVector::zeros(2), &options).unwrap();
        assert_eq!(guarded.termination, Termination::NoProgress);
        assert_eq!(guarded.x, DVector::zeros(2));
        let one = solve_local(
            &CurvedConstraint,
            DVector::zeros(2),
            &Options {
                nonmonotonic_gauss_newton: true,
                max_iterations: 1,
                ..options.clone()
            },
        )
        .unwrap();
        assert_eq!(one.termination, Termination::MaxIterations);
        assert!(one.cost > one.initial_cost);
        assert!(one.iterations[0].accepted && one.iterations[0].actual_reduction < 0.);
        let solved = solve_local(
            &CurvedConstraint,
            DVector::zeros(2),
            &Options {
                nonmonotonic_gauss_newton: true,
                ..options
            },
        )
        .unwrap();
        assert_eq!(solved.termination, Termination::Gradient);
        assert!((solved.x[0] - 1.).abs() < 1e-9 && (solved.x[1] - 1_f64.sin()).abs() < 1e-9);
        assert!(solved.cost < 1e-14);
    }
    assert!(!Options::default().nonmonotonic_gauss_newton);
    assert!(matches!(
        solve_local(
            &CurvedConstraint,
            DVector::zeros(2),
            &Options {
                nonmonotonic_gauss_newton: true,
                ..Default::default()
            }
        ),
        Err(Error::InvalidOptions)
    ));
}

fn curved_factor() -> ropt::fixed_lag::SmootherFactor {
    ropt::fixed_lag::SmootherFactor::analytic(&[7], 2, Default::default(), |v| {
        let mut r = DVector::zeros(2);
        let mut j = DMatrix::zeros(2, 2);
        CurvedConstraint.evaluate_local(&v[0], &mut r, &mut j)?;
        Ok((r, vec![j]))
    })
}
fn curved_variable() -> ropt::fixed_lag::WindowVariable {
    ropt::fixed_lag::WindowVariable {
        key: 7,
        kind: ropt::factor::VariableKind::Euclidean(2),
        value: DVector::zeros(2),
        timestamp: None,
        fixed: false,
    }
}
fn sparse_gn_options(max_iterations: usize) -> ropt::isam::IsamOptions {
    ropt::isam::IsamOptions {
        relinearization_threshold: 100.,
        relinearization_interval: 100,
        batch_solver: Options {
            algorithm: Algorithm::GaussNewton,
            nonmonotonic_gauss_newton: true,
            linear_solver: LinearSolver::SparseQr(Default::default()),
            gradient_norm: GradientNorm::ColumnNormalized,
            gradient_tolerance: 1e-8,
            max_iterations,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn sparse_qr_smoother_extends_order_and_preserves_committed_state_on_rank_failure() {
    use ropt::{factor::VariableKind, fixed_lag::*, isam::*};
    let mut s = IsamSmoother::new(sparse_gn_options(100))
        .unwrap()
        .with_batch_ordering_reuse()
        .unwrap();
    let first = s
        .update(
            SmootherUpdate {
                variables: vec![curved_variable()],
                factors: vec![curved_factor()],
                ..Default::default()
            },
            true,
        )
        .unwrap();
    assert_eq!(first.kind, IsamUpdateKind::Relinearized);
    assert!((first.estimates[&7][0] - 1.).abs() < 1e-9);
    let cached = s
        .update(
            SmootherUpdate {
                variables: vec![WindowVariable {
                    key: 1,
                    kind: VariableKind::Euclidean(1),
                    value: DVector::from_element(1, 3.),
                    timestamp: None,
                    fixed: false,
                }],
                factors: vec![SmootherFactor::analytic(&[1], 1, Default::default(), |v| {
                    Ok((
                        &v[0] - DVector::from_element(1, 5.),
                        vec![DMatrix::identity(1, 1)],
                    ))
                })],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    assert_eq!(cached.kind, IsamUpdateKind::Incremental);
    assert!((cached.estimates[&1][0] - 5.).abs() < 1e-12);
    let failed = s.update(
        SmootherUpdate {
            variables: vec![WindowVariable {
                key: 90,
                kind: VariableKind::Euclidean(1),
                value: DVector::zeros(1),
                timestamp: None,
                fixed: false,
            }],
            ..Default::default()
        },
        true,
    );
    assert!(matches!(
        failed,
        Err(Error::RankDeficient {
            rank: 3,
            parameters: 4
        })
    ));
    assert_eq!(s.variable_count(), 2);
    assert_eq!(s.factor_count(), 2);
    assert!(s.estimate(90).is_none());
    for (key, value) in cached.estimates {
        assert_eq!(s.estimate(key), Some(&value));
    }
}

#[test]
fn one_uphill_batch_trial_cannot_commit_an_unconverged_smoother() {
    use ropt::{fixed_lag::*, isam::*};
    let mut s = IsamSmoother::new(sparse_gn_options(1))
        .unwrap()
        .with_batch_ordering_reuse()
        .unwrap();
    assert!(matches!(
        s.update(
            SmootherUpdate {
                variables: vec![curved_variable()],
                factors: vec![curved_factor()],
                ..Default::default()
            },
            true
        ),
        Err(Error::Evaluation(_))
    ));
    assert_eq!(s.variable_count(), 0);
    assert_eq!(s.factor_count(), 0);
}
