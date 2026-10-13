use ropt::{factor::*, manifold::*, preconditioner::*, sparse::*, *};
use std::ops::Range;
fn one_block(end: usize) -> Vec<Range<usize>> {
    std::iter::once(0..end).collect()
}
fn csr(matrix: &DMatrix<f64>) -> CsrMatrix {
    let mut entries = Vec::new();
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            if matrix[(row, col)] != 0.0 {
                entries.push((row, col, matrix[(row, col)]));
            }
        }
    }
    CsrMatrix::from_triplets(matrix.nrows(), matrix.ncols(), entries).unwrap()
}
fn information(j: &DMatrix<f64>, lambda: f64) -> DMatrix<f64> {
    let mut h = j.transpose() * j;
    for k in 0..h.nrows() {
        h[(k, k)] += lambda * h[(k, k)].max(1.0);
    }
    h
}
#[test]
fn block_factors_match_dense_principal_solves_without_discarding_operator_couplings() {
    let j = DMatrix::from_row_slice(
        4,
        6,
        &[
            2.0, 0.3, 1.0, -0.2, 0.0, 0.4, 0.1, 1.5, 0.0, 1.0, 0.6, -0.1, 0.0, 0.2, 0.3, 0.0, 1.2,
            0.7, -0.5, 0.0, 0.1, 0.7, -0.2, 1.3,
        ],
    );
    let sparse = csr(&j);
    let lambda = 0.1;
    let h = information(&j, lambda);
    let ranges = [0..2, 2..5, 5..6];
    let pre = BlockJacobi::from_normal(&sparse, lambda, &ranges).unwrap();
    assert_eq!(pre.dimension(), 6);
    assert_eq!(pre.stored_entries(), 14);
    let r = DVector::from_vec(vec![1.0, -0.3, 0.5, 2.0, -1.0, 0.4]);
    let z = pre.apply(&r).unwrap();
    for range in &ranges {
        let block = h
            .view((range.start, range.start), (range.len(), range.len()))
            .into_owned();
        let expected = block
            .cholesky()
            .unwrap()
            .solve(&r.rows(range.start, range.len()).into_owned());
        assert!((z.rows(range.start, range.len()) - expected).norm() < 1e-12);
    }
    assert!(h[(0, 2)].abs() > 0.1); // Coupling exists in A; M is only its block diagonal.
    let singleton = BlockJacobi::from_normal(
        &sparse,
        lambda,
        &(0..6).map(|k| k..k + 1).collect::<Vec<_>>(),
    )
    .unwrap();
    assert!(
        (singleton.apply(&r).unwrap() - Jacobi::new(h.diagonal()).unwrap().apply(&r).unwrap())
            .norm()
            < 1e-12
    );
    let full = BlockJacobi::from_normal(&sparse, lambda, &one_block(6)).unwrap();
    let report = conjugate_gradient_preconditioned(
        &NormalOperator::new(&sparse, lambda).unwrap(),
        &full,
        &r,
        &CgOptions::default(),
    )
    .unwrap();
    assert_eq!(report.iterations, 1);
    assert!((&report.x - h.clone().cholesky().unwrap().solve(&r)).norm() < 1e-12);
    assert!((&r - h * report.x).norm() < 1e-11);
}
#[test]
fn correlated_independent_blocks_reduce_cg_iterations_and_check_the_true_residual() {
    let n = 16;
    let mut j = DMatrix::zeros(n, n);
    for block in 0..n / 2 {
        j[(2 * block, 2 * block)] = 1.0;
        j[(2 * block, 2 * block + 1)] = 1.0;
        j[(2 * block + 1, 2 * block + 1)] = 0.01 * (block + 1) as f64;
    }
    let sparse = csr(&j);
    let lambda = 1e-3;
    let op = NormalOperator::new(&sparse, lambda).unwrap();
    let pre = BlockJacobi::from_normal(
        &sparse,
        lambda,
        &(0..n / 2).map(|k| 2 * k..2 * k + 2).collect::<Vec<_>>(),
    )
    .unwrap();
    let b = DVector::from_iterator(n, (0..n).map(|k| 0.1 + k as f64 * 0.13));
    let options = CgOptions::default();
    let diagonal = conjugate_gradient(&op, &b, &options).unwrap();
    let blocked = conjugate_gradient_preconditioned(&op, &pre, &b, &options).unwrap();
    assert_eq!(blocked.iterations, 1);
    assert!(blocked.iterations < diagonal.iterations);
    let expected = information(&j, lambda).cholesky().unwrap().solve(&b);
    assert!((&blocked.x - expected).norm() < 1e-8);
    let true_r = (&b - op.apply(&blocked.x).unwrap()).norm();
    assert_eq!(true_r, blocked.residual_norm);
    assert!(
        true_r
            <= options
                .absolute_tolerance
                .max(options.relative_tolerance * b.norm())
    );
    let zero = conjugate_gradient_preconditioned(&op, &pre, &DVector::zeros(n), &options).unwrap();
    assert_eq!(zero.iterations, 0);
}
struct InvalidPre {
    dimension: usize,
    behavior: usize,
}
impl Preconditioner for InvalidPre {
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn apply(&self, r: &DVector<f64>) -> Result<DVector<f64>, Error> {
        match self.behavior {
            0 => Ok(DVector::zeros(r.len() + 1)),
            1 => Ok(DVector::from_element(r.len(), f64::INFINITY)),
            2 => Ok(-r),
            _ => Err(Error::Evaluation("preconditioner failure".into())),
        }
    }
}
#[test]
fn preconditioner_contract_failures_are_propagated_without_fallback() {
    let op = csr(&DMatrix::<f64>::identity(2, 2));
    let b = DVector::from_element(2, 1.0);
    let options = CgOptions::default();
    for (behavior, error) in [
        (0, Error::InvalidDimensions),
        (1, Error::NonFinite),
        (2, Error::LinearSolve),
        (3, Error::Evaluation("preconditioner failure".into())),
    ] {
        assert_eq!(
            conjugate_gradient_preconditioned(
                &op,
                &InvalidPre {
                    dimension: 2,
                    behavior
                },
                &b,
                &options
            )
            .unwrap_err(),
            error
        );
    }
    assert_eq!(
        conjugate_gradient_preconditioned(
            &op,
            &InvalidPre {
                dimension: 1,
                behavior: 3
            },
            &b,
            &options
        )
        .unwrap_err(),
        Error::InvalidDimensions
    );
}
#[test]
fn invalid_partitions_non_spd_blocks_and_overflow_are_errors() {
    let j = csr(&DMatrix::<f64>::identity(3, 3));
    for ranges in [
        vec![],
        one_block(2),
        vec![0..1, 2..3],
        vec![0..2, 1..3],
        vec![1..3, 0..1],
        one_block(4),
        vec![Range { start: 0, end: 0 }, 0..3],
    ] {
        assert!(matches!(
            BlockJacobi::from_normal(&j, 0.1, &ranges),
            Err(Error::InvalidDimensions)
        ));
    }
    for damping in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            BlockJacobi::from_normal(&j, damping, &one_block(3)),
            Err(Error::InvalidOptions)
        ));
    }
    let zero = CsrMatrix::from_triplets(2, 2, vec![]).unwrap();
    assert!(matches!(
        BlockJacobi::from_normal(&zero, 0.0, &one_block(2)),
        Err(Error::LinearSolve)
    ));
    let valid = BlockJacobi::from_normal(&zero, 0.5, &one_block(2)).unwrap();
    assert!(
        (valid.apply(&DVector::from_vec(vec![1.0, 2.0])).unwrap()
            - DVector::from_vec(vec![2.0, 4.0]))
        .norm()
            < 1e-13
    );
    assert_eq!(
        valid.apply(&DVector::zeros(3)).unwrap_err(),
        Error::InvalidDimensions
    );
    assert_eq!(
        valid
            .apply(&DVector::from_element(2, f64::NAN))
            .unwrap_err(),
        Error::NonFinite
    );
    let huge = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 1e308)]).unwrap();
    assert!(matches!(
        BlockJacobi::from_normal(&huge, 0.1, &one_block(1)),
        Err(Error::NonFinite)
    ));
    assert!(matches!(
        Jacobi::new(DVector::zeros(2)),
        Err(Error::LinearSolve)
    ));
}
#[test]
fn heterogeneous_graph_blocks_follow_fixed_layout_and_use_tangents() {
    let mut g = FactorGraph::new();
    let calibration = g
        .add_variable(VariableKind::Euclidean(8), DVector::zeros(8))
        .unwrap();
    let pose = g
        .add_variable(VariableKind::SE3, se3_values(SE3::exp(Tangent6::zeros())))
        .unwrap();
    let point = g
        .add_variable(VariableKind::Euclidean(3), DVector::zeros(3))
        .unwrap();
    assert_eq!(g.tangent_blocks().unwrap(), vec![0..8, 8..14, 14..17]);
    g.set_fixed(pose, true).unwrap();
    assert_eq!(g.tangent_blocks().unwrap(), vec![0..8, 8..11]);
    g.set_fixed(calibration, true).unwrap();
    assert_eq!(g.tangent_blocks().unwrap(), one_block(3));
    g.set_fixed(point, true).unwrap();
    assert!(g.tangent_blocks().unwrap().is_empty());
    g.set_fixed(pose, false).unwrap();
    assert_eq!(g.tangent_blocks().unwrap(), one_block(6));
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
        Err(Error::Evaluation(
            "dense evaluation must not be used".into(),
        ))
    }
    fn evaluate_sparse(&self, x: &DVector<f64>) -> Result<(DVector<f64>, CsrMatrix), Error> {
        let j = csr(&DMatrix::from_row_slice(2, 2, &[1.0, 0.3, 0.2, 1.2]));
        Ok((j.multiply(x)? - DVector::from_vec(vec![1.0, 2.0]), j))
    }
    fn retract(&self, x: &DVector<f64>, d: &DVector<f64>) -> Result<DVector<f64>, Error> {
        Ok(x + d)
    }
}
struct WithLayout {
    blocks: Vec<Range<usize>>,
}
impl LocalProblem for WithLayout {
    fn dimensions(&self) -> (usize, usize, usize) {
        SparseOnly.dimensions()
    }
    fn evaluate_local(
        &self,
        x: &DVector<f64>,
        r: &mut DVector<f64>,
        j: &mut DMatrix<f64>,
    ) -> Result<(), Error> {
        SparseOnly.evaluate_local(x, r, j)
    }
    fn evaluate_sparse(&self, x: &DVector<f64>) -> Result<(DVector<f64>, CsrMatrix), Error> {
        SparseOnly.evaluate_sparse(x)
    }
    fn retract(&self, x: &DVector<f64>, d: &DVector<f64>) -> Result<DVector<f64>, Error> {
        SparseOnly.retract(x, d)
    }
    fn tangent_blocks(&self) -> Result<Vec<Range<usize>>, Error> {
        Ok(self.blocks.clone())
    }
}
#[test]
fn sparse_block_backend_requires_explicit_valid_layout_and_supports_no_dense_provider() {
    let options = Options {
        linear_solver: LinearSolver::SparseBlockCg(CgOptions::default()),
        ..Options::default()
    };
    assert!(matches!(
        solve_local(&SparseOnly, DVector::zeros(2), &options),
        Err(Error::Evaluation(_))
    ));
    for blocks in [vec![], one_block(1), vec![0..1, 0..2]] {
        assert_eq!(
            solve_local(&WithLayout { blocks }, DVector::zeros(2), &options).unwrap_err(),
            Error::InvalidDimensions
        );
    }
    let report = solve_local(
        &WithLayout {
            blocks: vec![0..1, 1..2],
        },
        DVector::zeros(2),
        &options,
    )
    .unwrap();
    assert!(report.cost < 1e-15);
    assert_eq!(
        solve_local(
            &WithLayout {
                blocks: vec![0..1, 1..2]
            },
            DVector::zeros(2),
            &Options {
                algorithm: Algorithm::GaussNewton,
                ..options
            }
        )
        .unwrap_err(),
        Error::InvalidOptions
    );
}
#[test]
fn block_lm_preserves_rejected_states_and_reports_iteration_exhaustion() {
    let mut g = FactorGraph::new();
    let id = g
        .add_variable(VariableKind::Euclidean(1), DVector::from_element(1, 0.01))
        .unwrap();
    g.add_analytic_factor(&[id], 1, |v| {
        Ok((
            DVector::from_element(1, v[0][0].powi(2) - 1.0),
            vec![DMatrix::from_element(1, 1, 2.0 * v[0][0])],
        ))
    })
    .unwrap();
    let options = Options {
        linear_solver: LinearSolver::SparseBlockCg(CgOptions::default()),
        max_iterations: 1,
        initial_damping: 1e-12,
        ..Options::default()
    };
    let report = solve_local(&g, g.initial(), &options).unwrap();
    assert!(!report.iterations[0].accepted);
    assert_eq!(report.x, g.initial());
    g.set_fixed(id, true).unwrap();
    let fixed = solve_local(&g, g.initial(), &options).unwrap();
    assert!(fixed.iterations.is_empty());
    assert_eq!(fixed.x, g.initial());
    let mut g = FactorGraph::new();
    let x = g
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    let y = g
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    g.add_analytic_factor(&[x, y], 2, |v| {
        Ok((
            DVector::from_vec(vec![
                v[0][0] + 0.3 * v[1][0] - 1.0,
                0.2 * v[0][0] + 1.2 * v[1][0] - 2.0,
            ]),
            vec![
                DMatrix::from_column_slice(2, 1, &[1.0, 0.2]),
                DMatrix::from_column_slice(2, 1, &[0.3, 1.2]),
            ],
        ))
    })
    .unwrap();
    assert_eq!(
        solve_local(
            &g,
            g.initial(),
            &Options {
                linear_solver: LinearSolver::SparseBlockCg(CgOptions {
                    max_iterations: 1,
                    relative_tolerance: 0.0,
                    absolute_tolerance: 1e-15
                }),
                ..Options::default()
            }
        )
        .unwrap_err(),
        Error::LinearIterationLimit { iterations: 1 }
    );
}
