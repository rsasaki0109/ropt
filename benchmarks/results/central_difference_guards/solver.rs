pub use crate::dense::QrOptions;
use crate::schur::block_schur_solve;
use crate::sparse::{CgOptions, CsrMatrix, NormalOperator, conjugate_gradient, schur_solve};
pub use crate::sparse_direct::{SparseLdlOptions, SparseOrdering};
pub use crate::sparse_qr::SparseQrOptions;
use crate::{DMatrix, DVector};
use std::ops::Range;

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    InvalidDimensions,
    InvalidOptions,
    NonFinite,
    LinearSolve,
    SparseFillLimit { limit: usize },
    RankDeficient { rank: usize, parameters: usize },
    LinearIterationLimit { iterations: usize },
    LinearResidualTooLarge { residual_norm: f64, tolerance: f64 },
    Evaluation(String),
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}
/// Implementations must fill every entry; dimensions are fixed for a solve.
pub trait Residual {
    fn dimensions(&self) -> (usize, usize);
    fn evaluate(
        &self,
        x: &DVector<f64>,
        r: &mut DVector<f64>,
        j: &mut DMatrix<f64>,
    ) -> Result<(), Error>;
}
/// Jacobian columns are derivatives of right tangent increments. Ambient storage
/// (e.g. four quaternion coefficients) may differ from tangent dimension.
pub trait LocalProblem {
    fn dimensions(&self) -> (usize, usize, usize);
    fn evaluate_local(
        &self,
        x: &DVector<f64>,
        r: &mut DVector<f64>,
        j: &mut DMatrix<f64>,
    ) -> Result<(), Error>;
    fn retract(&self, x: &DVector<f64>, delta: &DVector<f64>) -> Result<DVector<f64>, Error>;
    /// Sparse evaluation is opt-in; no dense fallback is allocated.
    fn evaluate_sparse(&self, _x: &DVector<f64>) -> Result<(DVector<f64>, CsrMatrix), Error> {
        Err(Error::Evaluation(
            "sparse Jacobian unavailable for this problem".into(),
        ))
    }
    /// Optional complete contiguous partition of active tangent columns.
    /// Required for SparseBlockCg; fixed variable storage must not appear here.
    fn tangent_blocks(&self) -> Result<Vec<Range<usize>>, Error> {
        Err(Error::Evaluation(
            "active tangent block layout unavailable".into(),
        ))
    }
    /// Fixed joint losses for disjoint residual ranges, captured once per solve.
    /// Other rows use Options::loss per scalar.
    fn loss_blocks(&self) -> Vec<LossBlock> {
        Vec::new()
    }
    fn step_scale(&self, _x: &DVector<f64>) -> f64 {
        1.0
    }
}
pub(crate) struct EuclideanProblem<'a, P>(pub(crate) &'a P);
impl<P: Residual> LocalProblem for EuclideanProblem<'_, P> {
    fn dimensions(&self) -> (usize, usize, usize) {
        let (n, m) = self.0.dimensions();
        (n, n, m)
    }
    fn evaluate_local(
        &self,
        x: &DVector<f64>,
        r: &mut DVector<f64>,
        j: &mut DMatrix<f64>,
    ) -> Result<(), Error> {
        self.0.evaluate(x, r, j)
    }
    fn retract(&self, x: &DVector<f64>, d: &DVector<f64>) -> Result<DVector<f64>, Error> {
        Ok(x + d)
    }
    fn step_scale(&self, x: &DVector<f64>) -> f64 {
        x.norm()
    }
}
/// Explicit debug/verification adapter; never selected implicitly.
/// Every coordinate must admit distinct finite `x +/- h`, with finite `2*h`,
/// where `h = step * max(abs(x), 1)`. A step rounded away at either side returns
/// `InvalidOptions`; nonfinite inputs, perturbations or outputs return `NonFinite`.
/// Perturbations are validated before callbacks or output buffers are modified.
pub struct CentralDifference<F> {
    pub parameters: usize,
    pub residuals: usize,
    pub step: f64,
    pub function: F,
}
impl<F: Fn(&DVector<f64>) -> Result<DVector<f64>, Error>> Residual for CentralDifference<F> {
    fn dimensions(&self) -> (usize, usize) {
        (self.parameters, self.residuals)
    }
    fn evaluate(
        &self,
        x: &DVector<f64>,
        r: &mut DVector<f64>,
        j: &mut DMatrix<f64>,
    ) -> Result<(), Error> {
        if !self.step.is_finite() || self.step <= 0.0 {
            return Err(Error::InvalidOptions);
        }
        if x.len() != self.parameters || j.shape() != (self.residuals, self.parameters) {
            return Err(Error::InvalidDimensions);
        }
        crate::sparse::finite_vector(x)?;
        for &value in x.iter() {
            let h = self.step * value.abs().max(1.0);
            let (plus, minus) = (value + h, value - h);
            if !h.is_finite() || !(2.0 * h).is_finite() || !plus.is_finite() || !minus.is_finite() {
                return Err(Error::NonFinite);
            }
            if plus == value || minus == value {
                return Err(Error::InvalidOptions);
            }
        }
        *r = (self.function)(x)?;
        if r.len() != self.residuals {
            return Err(Error::InvalidDimensions);
        }
        crate::sparse::finite_vector(r)?;
        for k in 0..x.len() {
            let h = self.step * x[k].abs().max(1.0);
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[k] += h;
            minus[k] -= h;
            let a = (self.function)(&plus)?;
            if a.len() != self.residuals {
                return Err(Error::InvalidDimensions);
            }
            crate::sparse::finite_vector(&a)?;
            let b = (self.function)(&minus)?;
            if b.len() != self.residuals {
                return Err(Error::InvalidDimensions);
            }
            crate::sparse::finite_vector(&b)?;
            let column = (a - b) / (2.0 * h);
            crate::sparse::finite_vector(&column)?;
            j.set_column(k, &column);
        }
        Ok(())
    }
}
/// Indexed parameter blocks share a contiguous optimization vector.
#[derive(Debug, Clone, Default)]
pub struct Parameters {
    values: Vec<f64>,
    blocks: Vec<Range<usize>>,
}
impl Parameters {
    pub fn add_block(&mut self, values: &[f64]) -> Result<usize, Error> {
        if values.is_empty() {
            return Err(Error::InvalidDimensions);
        }
        if !values.iter().all(|v| v.is_finite()) {
            return Err(Error::NonFinite);
        }
        let start = self.values.len();
        self.values.extend_from_slice(values);
        self.blocks.push(start..self.values.len());
        Ok(self.blocks.len() - 1)
    }
    pub fn vector(&self) -> DVector<f64> {
        DVector::from_vec(self.values.clone())
    }
    pub fn block_range(&self, index: usize) -> Option<Range<usize>> {
        self.blocks.get(index).cloned()
    }
}
#[derive(Debug, Clone, Copy)]
pub enum Algorithm {
    GaussNewton,
    LevenbergMarquardt,
}
/// Options uses this loss per scalar; explicit LossBlock overrides use a joint norm.
#[derive(Debug, Clone, Copy)]
pub enum Loss {
    Squared,
    Huber(f64),
    Cauchy(f64),
}
impl Loss {
    pub(crate) fn valid(self) -> bool {
        match self {
            Self::Squared => true,
            Self::Huber(a) | Self::Cauchy(a) => a.is_finite() && a > 0.0,
        }
    }
    fn value_weight(self, r: f64) -> (f64, f64) {
        match self {
            Self::Squared => (0.5 * r * r, 1.0),
            Self::Huber(a) if r.abs() > a => (a * (r.abs() - 0.5 * a), a / r.abs()),
            Self::Huber(_) => (0.5 * r * r, 1.0),
            Self::Cauchy(a) => {
                let s = (r / a).powi(2);
                (0.5 * a * a * s.ln_1p(), 1.0 / (1.0 + s))
            }
        }
    }
    // Difference of true losses for nonnegative scalar/block norms. Orient the
    // Cauchy ratio positively so ln_1p cannot round its argument down to -1.
    fn reduction(self, before: f64, after: f64) -> f64 {
        if before < after {
            return -self.reduction(after, before);
        }
        match self {
            Self::Squared => 0.5 * (before - after) * (before + after),
            Self::Huber(a) if after >= a => a * (before - after),
            Self::Huber(a) if before > a => a * (before - a) + 0.5 * (a - after) * (a + after),
            Self::Huber(_) => 0.5 * (before - after) * (before + after),
            Self::Cauchy(a) => {
                let (u, v) = (before / a, after / a);
                let ratio = ((before - after) / a) * (u + v) / (1.0 + v * v);
                if before <= a {
                    // Cancel a^2 algebraically when normalized squared changes
                    // can underflow. ln(1+z)/z has continuous value 1 at z=0.
                    let log_ratio = if ratio == 0.0 {
                        1.0
                    } else {
                        ratio.ln_1p() / ratio
                    };
                    0.5 * (before - after) * (before + after) / (1.0 + v * v) * log_ratio
                } else {
                    0.5 * a * a * ratio.ln_1p()
                }
            }
        }
    }
}
#[derive(Default)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}
impl CompensatedSum {
    fn add(&mut self, value: f64) -> Result<(), Error> {
        let next = self.sum + value;
        if !value.is_finite() || !next.is_finite() {
            return Err(Error::NonFinite);
        }
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
        Ok(())
    }
    fn total(self) -> Result<f64, Error> {
        let value = self.sum + self.correction;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(Error::NonFinite)
        }
    }
}
/// Explicit robust loss on the joint squared norm of a disjoint, whitened residual block.
#[derive(Debug, Clone)]
pub struct LossBlock {
    pub rows: Range<usize>,
    pub loss: Loss,
}
#[derive(Debug, Clone, Copy)]
pub enum LinearSolver {
    Dense,
    /// Column-pivoted QR on J for GN or augmented least squares for LM.
    DenseQr(QrOptions),
    /// Reference dense elimination of trailing tangent variables; LM only.
    DenseSchur {
        retained: usize,
    },
    SparseCg(CgOptions),
    /// Full sparse LM normal operator with one Jacobi block per active variable.
    SparseBlockCg(CgOptions),
    /// Sparse LDL^T of damped normal equations with bounded fill; LM only.
    SparseLdl(SparseLdlOptions),
    /// Ordered sparse Givens QR of J (GN) or the augmented model (LM).
    SparseQr(SparseQrOptions),
    /// CSR-based independent-block elimination and reduced PCG; LM only.
    BlockSchur {
        retained: usize,
        block_size: usize,
        cg: CgOptions,
    },
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GradientNorm {
    #[default]
    Absolute,
    /// Infinity norm of g_i / ||weighted J[:, i]||_2. Uses stable column
    /// norms, without constructing a normal matrix. A zero column has zero
    /// gradient and contributes zero; observability checks remain separate.
    ColumnNormalized,
}

#[derive(Debug, Clone)]
pub struct Options {
    pub linear_solver: LinearSolver,
    pub algorithm: Algorithm,
    /// Explicit GN policy allowing finite uphill trials, like unrestricted
    /// Gauss-Newton iterations. Default GN rejects them. Invalid for LM.
    /// Final convergence must still be checked; negative reduction is not convergence.
    pub nonmonotonic_gauss_newton: bool,
    pub loss: Loss,
    pub max_iterations: usize,
    pub gradient_tolerance: f64,
    /// Absolute tangent gradient by default. ColumnNormalized measures each
    /// component in units of its weighted Jacobian column norm.
    pub gradient_norm: GradientNorm,
    pub step_tolerance: f64,
    pub cost_tolerance: f64,
    pub initial_damping: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            linear_solver: LinearSolver::Dense,
            algorithm: Algorithm::LevenbergMarquardt,
            nonmonotonic_gauss_newton: false,
            loss: Loss::Squared,
            max_iterations: 100,
            gradient_tolerance: 1e-10,
            gradient_norm: GradientNorm::Absolute,
            step_tolerance: 1e-10,
            cost_tolerance: 1e-12,
            initial_damping: 1e-3,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    Gradient,
    Step,
    Cost,
    NoProgress,
    MaxIterations,
}
#[derive(Debug, Clone)]
pub struct Iteration {
    pub iteration: usize,
    pub cost: f64,
    pub trial_cost: f64,
    /// True-loss reduction accumulated per scalar/joint block. It can be
    /// nonzero even when rounded aggregate cost and trial_cost are identical.
    pub actual_reduction: f64,
    pub gradient_norm: f64,
    pub step_norm: f64,
    pub damping: f64,
    pub gain_ratio: f64,
    pub accepted: bool,
    /// PCG iterations for this trial; zero for direct solves.
    pub linear_iterations: usize,
}
#[derive(Debug, Clone)]
pub struct Report {
    pub x: DVector<f64>,
    pub initial_cost: f64,
    pub cost: f64,
    pub termination: Termination,
    pub iterations: Vec<Iteration>,
}

enum Jacobian {
    Dense(DMatrix<f64>),
    Sparse(CsrMatrix),
}
struct Linearization {
    cost: f64,
    objective_norms: Vec<f64>,
    residual: DVector<f64>,
    jacobian: Jacobian,
}
// Keep the LDL workspace inline to avoid an extra heap allocation per solve.
#[allow(clippy::large_enum_variant)]
enum LinearSolveWorkspace {
    None,
    Ldl(crate::sparse_direct::SparseLdlWorkspace),
    Qr(crate::sparse_qr::SparseQrWorkspace),
}
impl LinearSolveWorkspace {
    fn permutation(&self) -> Option<&[usize]> {
        match self {
            Self::None => None,
            Self::Ldl(w) => w.symbolic().map(|s| s.permutation()),
            Self::Qr(w) => w.permutation(),
        }
    }
}
impl Linearization {
    /// Consume the already evaluated accepted model. iSAM uses this only for
    /// squared loss with unit IRLS weights; no callback or differentiation replay.
    fn into_sparse(self) -> Result<(DVector<f64>, CsrMatrix), Error> {
        let j = match self.jacobian {
            Jacobian::Sparse(j) => j,
            Jacobian::Dense(j) => {
                let mut offsets = Vec::with_capacity(j.nrows() + 1);
                offsets.push(0);
                let mut indices = Vec::new();
                let mut values = Vec::new();
                for r in 0..j.nrows() {
                    for c in 0..j.ncols() {
                        let value = j[(r, c)];
                        if value != 0. {
                            indices.push(c);
                            values.push(value);
                        }
                    }
                    offsets.push(indices.len());
                }
                CsrMatrix::from_canonical(j.nrows(), j.ncols(), offsets, indices, values)?
            }
        };
        Ok((self.residual, j))
    }
    fn actual_reduction(
        &self,
        trial: &Self,
        fallback: Loss,
        blocks: &[LossBlock],
    ) -> Result<f64, Error> {
        if self.objective_norms.len() != trial.objective_norms.len() {
            return Err(Error::InvalidDimensions);
        }
        let mut sum = CompensatedSum::default();
        let mut index = 0;
        let mut add = |loss: Loss| -> Result<(), Error> {
            sum.add(loss.reduction(self.objective_norms[index], trial.objective_norms[index]))?;
            index += 1;
            Ok(())
        };
        let mut cursor = 0;
        for block in blocks {
            for _ in cursor..block.rows.start {
                add(fallback)?;
            }
            add(block.loss)?;
            cursor = block.rows.end;
        }
        for _ in cursor..self.residual.len() {
            add(fallback)?;
        }
        sum.total()
    }
    fn gradient(&self) -> Result<DVector<f64>, Error> {
        let g = match &self.jacobian {
            Jacobian::Dense(j) => j.transpose() * &self.residual,
            Jacobian::Sparse(j) => j.transpose_multiply(&self.residual)?,
        };
        if !g.iter().all(|v| v.is_finite()) {
            return Err(Error::NonFinite);
        }
        Ok(g)
    }
    fn gradient_norm(&self, gradient: &DVector<f64>, kind: GradientNorm) -> Result<f64, Error> {
        if kind == GradientNorm::Absolute {
            return Ok(gradient.amax());
        }
        match &self.jacobian {
            Jacobian::Dense(j) => column_normalized_norm(
                gradient,
                j.iter().enumerate().map(|(i, &v)| (i / j.nrows(), v)),
            ),
            Jacobian::Sparse(j) => {
                column_normalized_norm(gradient, j.entries().map(|(_, c, v)| (c, v)))
            }
        }
    }
    fn predicted(&self, g: &DVector<f64>, step: &DVector<f64>) -> Result<f64, Error> {
        let js = match &self.jacobian {
            Jacobian::Dense(j) => j * step,
            Jacobian::Sparse(j) => j.multiply(step)?,
        };
        let value = -g.dot(step) - 0.5 * js.norm_squared();
        if !value.is_finite() {
            return Err(Error::NonFinite);
        }
        Ok(value)
    }
    fn step(
        &self,
        gradient: &DVector<f64>,
        damping: f64,
        options: &Options,
        blocks: Option<&[Range<usize>]>,
        workspace: &mut LinearSolveWorkspace,
    ) -> Result<(DVector<f64>, usize), Error> {
        match &self.jacobian {
            Jacobian::Dense(j) => {
                if let LinearSolver::DenseQr(qr) = options.linear_solver {
                    let lambda = match options.algorithm {
                        Algorithm::GaussNewton => 0.0,
                        Algorithm::LevenbergMarquardt => damping,
                    };
                    return Ok((crate::dense::qr_step(j, &self.residual, lambda, &qr)?, 0));
                }
                let step = match options.algorithm {
                    Algorithm::GaussNewton => {
                        let (m, n) = j.shape();
                        let svd = j.clone().svd(true, true);
                        let cutoff = f64::EPSILON * (m.max(n) as f64) * svd.singular_values.amax();
                        if m < n || svd.singular_values.iter().any(|s| *s <= cutoff) {
                            return Err(Error::LinearSolve);
                        }
                        svd.solve(&(-&self.residual), cutoff)
                            .map_err(|_| Error::LinearSolve)?
                    }
                    Algorithm::LevenbergMarquardt => {
                        let mut system = j.transpose() * j;
                        if !system.iter().all(|v| v.is_finite()) {
                            return Err(Error::NonFinite);
                        }
                        for k in 0..system.nrows() {
                            system[(k, k)] += damping * system[(k, k)].max(1.0);
                        }
                        if !system.iter().all(|v| v.is_finite()) {
                            return Err(Error::NonFinite);
                        }
                        match options.linear_solver {
                            LinearSolver::DenseSchur { retained } => {
                                schur_solve(&system, &(-gradient), retained)?
                            }
                            _ => system
                                .cholesky()
                                .ok_or(Error::LinearSolve)?
                                .solve(&(-gradient)),
                        }
                    }
                };
                Ok((step, 0))
            }
            Jacobian::Sparse(j) => {
                if matches!(options.linear_solver, LinearSolver::SparseLdl(_)) {
                    let LinearSolveWorkspace::Ldl(ldl_workspace) = workspace else {
                        return Err(Error::InvalidOptions);
                    };
                    let report = ldl_workspace.solve(j, &(-gradient), damping)?;
                    return Ok((report.x, 0));
                }
                if matches!(options.linear_solver, LinearSolver::SparseQr(_)) {
                    let LinearSolveWorkspace::Qr(qr_workspace) = workspace else {
                        return Err(Error::InvalidOptions);
                    };
                    let damping = match options.algorithm {
                        Algorithm::GaussNewton => 0.,
                        Algorithm::LevenbergMarquardt => damping,
                    };
                    return Ok((qr_workspace.solve(j, &self.residual, damping)?, 0));
                }
                let report = match options.linear_solver {
                    LinearSolver::SparseCg(cg) => {
                        conjugate_gradient(&NormalOperator::new(j, damping)?, &(-gradient), &cg)?
                    }
                    LinearSolver::SparseBlockCg(cg) => {
                        let partition = blocks.ok_or(Error::InvalidDimensions)?;
                        let preconditioner =
                            crate::preconditioner::BlockJacobi::from_normal(j, damping, partition)?;
                        crate::sparse::conjugate_gradient_preconditioned(
                            &NormalOperator::new(j, damping)?,
                            &preconditioner,
                            &(-gradient),
                            &cg,
                        )?
                    }
                    LinearSolver::BlockSchur {
                        retained,
                        block_size,
                        cg,
                    } => block_schur_solve(j, &(-gradient), damping, retained, block_size, &cg)?,
                    _ => return Err(Error::InvalidOptions),
                };
                Ok((report.x, report.iterations))
            }
        }
    }
}
fn linearize(
    problem: &impl LocalProblem,
    x: &DVector<f64>,
    n: usize,
    m: usize,
    options: &Options,
    loss_blocks: &[LossBlock],
) -> Result<Linearization, Error> {
    if !x.iter().all(|v| v.is_finite()) {
        return Err(Error::NonFinite);
    }
    let (mut r, mut jacobian) = match options.linear_solver {
        LinearSolver::Dense | LinearSolver::DenseQr(_) | LinearSolver::DenseSchur { .. } => {
            let mut r = DVector::from_element(m, f64::NAN);
            let mut j = DMatrix::from_element(m, n, f64::NAN);
            problem.evaluate_local(x, &mut r, &mut j)?;
            if j.shape() != (m, n) {
                return Err(Error::InvalidDimensions);
            }
            if !j.iter().all(|v| v.is_finite()) {
                return Err(Error::NonFinite);
            }
            (r, Jacobian::Dense(j))
        }
        LinearSolver::SparseCg(_)
        | LinearSolver::SparseBlockCg(_)
        | LinearSolver::SparseLdl(_)
        | LinearSolver::SparseQr(_)
        | LinearSolver::BlockSchur { .. } => {
            let (r, j) = problem.evaluate_sparse(x)?;
            if j.shape() != (m, n) {
                return Err(Error::InvalidDimensions);
            }
            (r, Jacobian::Sparse(j))
        }
    };
    if r.len() != m {
        return Err(Error::InvalidDimensions);
    }
    if !r.iter().all(|v| v.is_finite()) {
        return Err(Error::NonFinite);
    }
    let count = m - loss_blocks.iter().map(|b| b.rows.len() - 1).sum::<usize>();
    let mut objective_norms = Vec::with_capacity(count);
    let (cost, weights) = robust_model(&r, options.loss, loss_blocks, Some(&mut objective_norms))?;
    r.component_mul_assign(&weights);
    if !cost.is_finite() || !r.iter().chain(weights.iter()).all(|v| v.is_finite()) {
        return Err(Error::NonFinite);
    }
    match &mut jacobian {
        Jacobian::Dense(j) => {
            for i in 0..m {
                j.row_mut(i).scale_mut(weights[i]);
            }
            if !j.iter().all(|v| v.is_finite()) {
                return Err(Error::NonFinite);
            }
        }
        Jacobian::Sparse(j) => j.scale_rows(&weights)?,
    }
    Ok(Linearization {
        cost,
        objective_norms,
        residual: r,
        jacobian,
    })
}
pub(crate) fn robust_weights(
    residual: &DVector<f64>,
    fallback: Loss,
    blocks: &[LossBlock],
) -> Result<(f64, DVector<f64>), Error> {
    robust_model(residual, fallback, blocks, None)
}
fn robust_model(
    residual: &DVector<f64>,
    fallback: Loss,
    blocks: &[LossBlock],
    mut norms: Option<&mut Vec<f64>>,
) -> Result<(f64, DVector<f64>), Error> {
    let m = residual.len();
    let mut cursor = 0;
    let mut cost = CompensatedSum::default();
    let mut weights = DVector::zeros(m);
    let mut add = |norm: f64, loss: Loss, rows: Range<usize>| -> Result<(), Error> {
        if !norm.is_finite() {
            return Err(Error::NonFinite);
        }
        let (c, w) = loss.value_weight(norm);
        cost.add(c)?;
        weights.rows_mut(rows.start, rows.len()).fill(w.sqrt());
        if let Some(norms) = &mut norms {
            norms.push(norm);
        }
        Ok(())
    };
    for block in blocks {
        for i in cursor..block.rows.start {
            add(residual[i].abs(), fallback, i..i + 1)?;
        }
        let norm = residual.rows(block.rows.start, block.rows.len()).norm();
        add(norm, block.loss, block.rows.clone())?;
        cursor = block.rows.end;
    }
    for i in cursor..m {
        add(residual[i].abs(), fallback, i..i + 1)?;
    }
    Ok((cost.total()?, weights))
}
pub(crate) fn validated_loss_blocks(
    problem: &impl LocalProblem,
    m: usize,
) -> Result<Vec<LossBlock>, Error> {
    let mut loss_blocks = problem.loss_blocks();
    loss_blocks.sort_unstable_by_key(|b| b.rows.start);
    let mut end = 0;
    for block in &loss_blocks {
        if block.rows.start < end || block.rows.start >= block.rows.end || block.rows.end > m {
            return Err(Error::InvalidDimensions);
        }
        if !block.loss.valid() {
            return Err(Error::InvalidOptions);
        }
        end = block.rows.end;
    }
    Ok(loss_blocks)
}
/// Dense IRLS GN or gain-ratio LM. Rejected trials do not mutate the accepted state.
pub fn solve(
    problem: &impl Residual,
    initial: DVector<f64>,
    options: &Options,
) -> Result<Report, Error> {
    solve_local(&EuclideanProblem(problem), initial, options)
}
/// Optimize using tangent Jacobians, the problem's retraction and the selected linear backend.
pub fn solve_local(
    problem: &impl LocalProblem,
    initial: DVector<f64>,
    options: &Options,
) -> Result<Report, Error> {
    solve_local_model(problem, initial, options).map(|(report, _, _, _)| report)
}
pub(crate) struct GaussianSolution {
    pub report: Report,
    pub residual: DVector<f64>,
    pub jacobian: CsrMatrix,
    pub final_damping: f64,
    pub column_permutation: Option<Vec<usize>>,
}
pub(crate) fn gaussian_reduction(
    before: &DVector<f64>,
    after: &DVector<f64>,
) -> Result<f64, Error> {
    if before.len() != after.len() {
        return Err(Error::InvalidDimensions);
    }
    let mut sum = CompensatedSum::default();
    for (&a, &b) in before.iter().zip(after.iter()) {
        sum.add(Loss::Squared.reduction(a.abs(), b.abs()))?;
    }
    sum.total()
}
/// Ordered Gaussian cost without materializing a residual or zero vector.
/// Errors are retained until finish so callers can preserve callback precedence.
#[derive(Default)]
pub(crate) struct GaussianCostAccumulator {
    sum: CompensatedSum,
    error: Option<Error>,
}
impl GaussianCostAccumulator {
    pub(crate) fn add(&mut self, residuals: &DVector<f64>) {
        if self.error.is_some() {
            return;
        }
        for &value in residuals.iter() {
            if let Err(error) = self.sum.add(Loss::Squared.reduction(0., value.abs())) {
                self.error = Some(error);
                break;
            }
        }
    }
    pub(crate) fn finish(self) -> Result<f64, Error> {
        match self.error {
            Some(error) => Err(error),
            None => self.sum.total().map(|reduction| -reduction),
        }
    }
}
pub(crate) fn gaussian_gradient_norm(
    j: &CsrMatrix,
    r: &DVector<f64>,
    kind: GradientNorm,
) -> Result<f64, Error> {
    let gradient = j.transpose_multiply(r)?;
    gaussian_gradient_measure(j, &gradient, kind)
}
pub(crate) fn gaussian_gradient_measure(
    j: &CsrMatrix,
    gradient: &DVector<f64>,
    kind: GradientNorm,
) -> Result<f64, Error> {
    if kind == GradientNorm::Absolute {
        return Ok(gradient.amax());
    }
    column_normalized_norm(gradient, j.entries().map(|(_, c, v)| (c, v)))
}
// Safe squared accumulation has ample exponent margin even for usize::MAX
// entries. An extreme coefficient switches that column to scaled hypot.
fn column_normalized_norm(
    gradient: &DVector<f64>,
    entries: impl Iterator<Item = (usize, f64)>,
) -> Result<f64, Error> {
    const LOW: f64 = f64::from_bits((1023 - 450) << 52);
    const HIGH: f64 = f64::from_bits((1023 + 450) << 52);
    let mut values = DVector::<f64>::zeros(gradient.len());
    let mut scaled = vec![false; gradient.len()];
    for (column, value) in entries {
        if scaled[column] {
            values[column] = values[column].hypot(value);
        } else if value == 0. || (LOW..=HIGH).contains(&value.abs()) {
            values[column] += value * value;
        } else {
            values[column] = values[column].sqrt().hypot(value);
            scaled[column] = true;
        }
    }
    let mut result = 0.0_f64;
    for (i, &g) in gradient.iter().enumerate() {
        if !g.is_finite() {
            return Err(Error::NonFinite);
        }
        let scale = if scaled[i] {
            values[i]
        } else {
            values[i].sqrt()
        };
        if !scale.is_finite() {
            return Err(Error::NonFinite);
        }
        result = result.max(if scale == 0. {
            g.abs()
        } else {
            g.abs() / scale
        });
    }
    if !result.is_finite() {
        return Err(Error::NonFinite);
    }
    Ok(result)
}
/// Return the final accepted whitened Gaussian model for undamped square-root
/// rebuilds. General robust callers must not treat an IRLS model as Gaussian.
pub(crate) fn solve_local_gaussian_model(
    problem: &impl LocalProblem,
    initial: DVector<f64>,
    options: &Options,
    capture_ordering: bool,
) -> Result<GaussianSolution, Error> {
    if !matches!(options.loss, Loss::Squared)
        || problem
            .loss_blocks()
            .iter()
            .any(|block| !matches!(block.loss, Loss::Squared))
    {
        return Err(Error::InvalidOptions);
    }
    let (report, model, final_damping, workspace) = solve_local_model(problem, initial, options)?;
    let column_permutation = if capture_ordering {
        workspace.permutation().map(|p| p.to_vec())
    } else {
        None
    };
    let (r, j) = model.into_sparse()?;
    Ok(GaussianSolution {
        report,
        residual: r,
        jacobian: j,
        final_damping,
        column_permutation,
    })
}
fn solve_local_model(
    problem: &impl LocalProblem,
    initial: DVector<f64>,
    options: &Options,
) -> Result<(Report, Linearization, f64, LinearSolveWorkspace), Error> {
    let (ambient, n, m) = problem.dimensions();
    if ambient == 0 || m == 0 || initial.len() != ambient {
        return Err(Error::InvalidDimensions);
    }
    if !initial.iter().all(|v| v.is_finite()) {
        return Err(Error::NonFinite);
    }
    if options.max_iterations == 0
        || (options.nonmonotonic_gauss_newton
            && !matches!(options.algorithm, Algorithm::GaussNewton))
        || !options.loss.valid()
        || !options.initial_damping.is_finite()
        || options.initial_damping <= 0.0
        || [
            options.gradient_tolerance,
            options.step_tolerance,
            options.cost_tolerance,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(Error::InvalidOptions);
    }
    if matches!(
        options.linear_solver,
        LinearSolver::SparseCg(_)
            | LinearSolver::SparseBlockCg(_)
            | LinearSolver::SparseLdl(_)
            | LinearSolver::BlockSchur { .. }
    ) && matches!(options.algorithm, Algorithm::GaussNewton)
    {
        return Err(Error::InvalidOptions);
    }
    if let LinearSolver::DenseSchur { retained } = options.linear_solver {
        if retained == 0 || retained >= n || matches!(options.algorithm, Algorithm::GaussNewton) {
            return Err(Error::InvalidOptions);
        }
    }
    if let LinearSolver::DenseQr(qr) = options.linear_solver {
        qr.validate()?;
    }
    if let LinearSolver::SparseLdl(ldl) = options.linear_solver {
        ldl.validate()?;
    }
    if let LinearSolver::SparseQr(qr) = options.linear_solver {
        qr.validate()?;
    }
    if let LinearSolver::SparseCg(cg) | LinearSolver::SparseBlockCg(cg) = options.linear_solver {
        cg.validate()?;
    }
    if let LinearSolver::BlockSchur {
        retained,
        block_size,
        cg,
    } = options.linear_solver
    {
        if retained == 0
            || retained >= n
            || block_size == 0
            || !(n - retained).is_multiple_of(block_size)
        {
            return Err(Error::InvalidOptions);
        }
        cg.validate()?;
    }
    let block_layout = if n > 0 && matches!(options.linear_solver, LinearSolver::SparseBlockCg(_)) {
        let blocks = problem.tangent_blocks()?;
        crate::preconditioner::validate_partition(n, &blocks)?;
        Some(blocks)
    } else {
        None
    };
    let loss_blocks = validated_loss_blocks(problem, m)?;
    let mut x = initial;
    let mut model = linearize(problem, &x, n, m, options, &loss_blocks)?;
    let mut cost = model.cost;
    let initial_cost = cost;
    if n == 0 {
        return Ok((
            Report {
                x,
                initial_cost,
                cost,
                termination: Termination::Gradient,
                iterations: Vec::new(),
            },
            model,
            options.initial_damping,
            LinearSolveWorkspace::None,
        ));
    }
    let mut damping = options.initial_damping;
    let mut growth = 2.0;
    let mut history = Vec::new();
    let mut workspace = match options.linear_solver {
        LinearSolver::SparseLdl(ldl) => {
            LinearSolveWorkspace::Ldl(crate::sparse_direct::SparseLdlWorkspace::new(ldl)?)
        }
        LinearSolver::SparseQr(qr) => {
            LinearSolveWorkspace::Qr(crate::sparse_qr::SparseQrWorkspace::new(qr)?)
        }
        _ => LinearSolveWorkspace::None,
    };
    let mut termination = Termination::MaxIterations;
    for iteration in 0..options.max_iterations {
        let gradient = model.gradient()?;
        let gradient_norm = model.gradient_norm(&gradient, options.gradient_norm)?;
        if !gradient_norm.is_finite() {
            return Err(Error::NonFinite);
        }
        if gradient_norm <= options.gradient_tolerance {
            termination = Termination::Gradient;
            break;
        }
        let (step, linear_iterations) = model.step(
            &gradient,
            damping,
            options,
            block_layout.as_deref(),
            &mut workspace,
        )?;
        if !step.iter().all(|v| v.is_finite()) {
            return Err(Error::NonFinite);
        }
        let step_norm = step.norm();
        let scale = problem.step_scale(&x);
        if !scale.is_finite() || scale < 0.0 {
            return Err(Error::InvalidOptions);
        }

        let trial = problem.retract(&x, &step)?;
        if trial.len() != ambient {
            return Err(Error::InvalidDimensions);
        }
        let trial_model = linearize(problem, &trial, n, m, options, &loss_blocks)?;
        let trial_cost = trial_model.cost;
        let predicted = model.predicted(&gradient, &step)?;
        let actual = model.actual_reduction(&trial_model, options.loss, &loss_blocks)?;
        let ratio = if predicted > 0.0 {
            actual / predicted
        } else {
            f64::NEG_INFINITY
        };
        let small_step = step_norm <= options.step_tolerance * (options.step_tolerance + scale);
        let accepted = (actual > 0.0 && ratio > 0.0)
            || (options.nonmonotonic_gauss_newton
                && predicted > 0.0
                && !(actual <= 0.0 && small_step));
        history.push(Iteration {
            iteration,
            cost,
            trial_cost,
            actual_reduction: actual,
            gradient_norm,
            step_norm,
            damping,
            gain_ratio: ratio,
            accepted,
            linear_iterations,
        });
        if accepted {
            let previous = cost;
            x = trial;
            model = trial_model;
            cost = trial_cost;
            if matches!(options.algorithm, Algorithm::LevenbergMarquardt) {
                damping *= (1.0 - (2.0 * ratio - 1.0).powi(3)).max(1.0 / 3.0);
                damping = damping.max(1e-18);
                growth = 2.0;
            }
            if step_norm <= options.step_tolerance * (options.step_tolerance + scale) {
                termination = Termination::Step;
                break;
            }
            if actual > 0.0 && actual <= options.cost_tolerance * previous.max(1.0) {
                termination = Termination::Cost;
                break;
            }
        } else {
            // A numerically unresolvable trial is stagnation, not convergence.
            if step_norm <= options.step_tolerance * (options.step_tolerance + scale) {
                termination = Termination::NoProgress;
                break;
            }
            match options.algorithm {
                Algorithm::LevenbergMarquardt => {
                    damping *= growth;
                    growth *= 2.0;
                    if !damping.is_finite() {
                        return Err(Error::LinearSolve);
                    }
                }
                Algorithm::GaussNewton => {
                    termination = Termination::NoProgress;
                    break;
                }
            }
        }
    }
    Ok((
        Report {
            x,
            initial_cost,
            cost,
            termination,
            iterations: history,
        },
        model,
        damping,
        workspace,
    ))
}

#[cfg(test)]
mod stable_loss_tests {
    use super::*;
    #[test]
    fn streamed_gaussian_cost_preserves_compensation_bits_and_errors_across_blocks() {
        for values in [
            vec![],
            vec![0., -0.],
            vec![1e150, 1e-150, 3., -4., 1e150],
            vec![f64::from_bits(1), -f64::MIN_POSITIVE, 1.],
            vec![1e154, -1e154, 1e-154],
            vec![1e200, 3., 4.],
            vec![f64::NAN, 1.],
            vec![f64::INFINITY, 1.],
        ] {
            let residual = DVector::from_vec(values.clone());
            let expected = gaussian_reduction(&DVector::zeros(values.len()), &residual)
                .map(|value| (-value).to_bits());
            for width in [1, 2, 3, 8] {
                let mut cost = GaussianCostAccumulator::default();
                cost.add(&DVector::zeros(0));
                for block in values.chunks(width) {
                    cost.add(&DVector::from_column_slice(block));
                }
                assert_eq!(cost.finish().map(f64::to_bits), expected);
            }
        }
    }
    #[test]
    fn normalized_columns_handle_zero_tiny_huge_and_mixed_coefficients() {
        for coefficients in [
            vec![3., 4.],
            vec![3e-200, 4e-200],
            vec![3e200, 4e200],
            vec![3., 4e200, 3e200],
            vec![3e200, 4e200, 3.],
        ] {
            let reference = coefficients.iter().copied().fold(0.0_f64, f64::hypot);
            let g = DVector::from_element(1, 2. * reference);
            let norm =
                column_normalized_norm(&g, coefficients.iter().copied().map(|v| (0, v))).unwrap();
            assert!((norm - 2.).abs() < 1e-14);
        }
        assert_eq!(
            column_normalized_norm(
                &DVector::from_vec(vec![0., 7.]),
                [(0, 0.), (1, 0.)].into_iter()
            )
            .unwrap(),
            7.
        );
        assert_eq!(
            column_normalized_norm(&DVector::zeros(1), [(0, f64::INFINITY)].into_iter()),
            Err(Error::NonFinite)
        );
    }
    #[test]
    fn squared_and_huber_differences_preserve_small_changes_and_transition_signs() {
        assert_eq!(
            Loss::Squared.reduction(100_000_001., 100_000_000.),
            100_000_000.5
        );
        assert_eq!(Loss::Huber(2.).reduction(4., 3.), 2.);
        assert_eq!(Loss::Huber(2.).reduction(3., 1.), 3.5);
        assert_eq!(Loss::Huber(2.).reduction(1., 3.), -3.5);
        assert_eq!(Loss::Huber(2.).reduction(2., 1.), 1.5);
        assert_eq!(Loss::Squared.reduction(3., 3.), 0.);
    }
    #[test]
    fn cauchy_differences_handle_close_norms_and_extreme_ratio_in_both_directions() {
        // High precision log((1+before^2)/(1+after^2))/2 = about 2e-16.
        // Subtraction of separately rounded losses gives zero here.
        let reduction = Loss::Cauchy(1.).reduction(1e16, 1e16 - 2.);
        assert!((reduction - 2e-16).abs() < 1e-30);
        let decrease = Loss::Cauchy(1.).reduction(1e100, 0.);
        assert!((decrease - 230.25850929940458).abs() < 1e-12);
        assert_eq!(Loss::Cauchy(1.).reduction(0., 1e100), -decrease);
        assert_eq!(Loss::Cauchy(1.).reduction(1., 1.), 0.);
        assert!((Loss::Cauchy(1e154).reduction(1., 0.5) - 0.375).abs() < 1e-15);
        let tiny_change = Loss::Cauchy(1e154).reduction(1., 1. - f64::EPSILON);
        assert!((tiny_change - f64::EPSILON).abs() < 1e-30);
    }
    #[test]
    fn compensated_summation_retains_small_terms_and_rejects_nonfinite_totals() {
        let mut sum = CompensatedSum::default();
        for value in [1e16, 1., 1., -1e16] {
            sum.add(value).unwrap();
        }
        assert_eq!(sum.total().unwrap(), 2.);
        assert_eq!(
            CompensatedSum::default().add(f64::INFINITY).unwrap_err(),
            Error::NonFinite
        );
        let mut sum = CompensatedSum::default();
        sum.add(f64::MAX).unwrap();
        assert_eq!(sum.add(f64::MAX).unwrap_err(), Error::NonFinite);
    }
}
