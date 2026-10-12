use crate::schur::block_schur_solve;
use crate::sparse::{CgOptions, CsrMatrix, NormalOperator, conjugate_gradient, schur_solve};
use crate::{DMatrix, DVector};
use std::ops::Range;

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    InvalidDimensions,
    InvalidOptions,
    NonFinite,
    LinearSolve,
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
    fn step_scale(&self, _x: &DVector<f64>) -> f64 {
        1.0
    }
}
struct EuclideanProblem<'a, P>(&'a P);
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
        *r = (self.function)(x)?;
        if r.len() != self.residuals {
            return Err(Error::InvalidDimensions);
        }
        for k in 0..x.len() {
            let h = self.step * x[k].abs().max(1.0);
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[k] += h;
            minus[k] -= h;
            let a = (self.function)(&plus)?;
            let b = (self.function)(&minus)?;
            if a.len() != self.residuals || b.len() != self.residuals {
                return Err(Error::InvalidDimensions);
            }
            j.set_column(k, &((a - b) / (2.0 * h)));
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
/// Loss applies per scalar residual, rather than per residual block.
#[derive(Debug, Clone, Copy)]
pub enum Loss {
    Squared,
    Huber(f64),
    Cauchy(f64),
}
impl Loss {
    fn valid(self) -> bool {
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
}
#[derive(Debug, Clone, Copy)]
pub enum LinearSolver {
    Dense,
    /// Reference dense elimination of trailing tangent variables; LM only.
    DenseSchur {
        retained: usize,
    },
    SparseCg(CgOptions),
    /// CSR-based independent-block elimination and reduced PCG; LM only.
    BlockSchur {
        retained: usize,
        block_size: usize,
        cg: CgOptions,
    },
}
#[derive(Debug, Clone)]
pub struct Options {
    pub linear_solver: LinearSolver,
    pub algorithm: Algorithm,
    pub loss: Loss,
    pub max_iterations: usize,
    pub gradient_tolerance: f64,
    pub step_tolerance: f64,
    pub cost_tolerance: f64,
    pub initial_damping: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            linear_solver: LinearSolver::Dense,
            algorithm: Algorithm::LevenbergMarquardt,
            loss: Loss::Squared,
            max_iterations: 100,
            gradient_tolerance: 1e-10,
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
    pub gradient_norm: f64,
    pub step_norm: f64,
    pub damping: f64,
    pub gain_ratio: f64,
    pub accepted: bool,
    /// PCG iterations for this trial; zero for direct dense solves.
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
    residual: DVector<f64>,
    jacobian: Jacobian,
}
impl Linearization {
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
    ) -> Result<(DVector<f64>, usize), Error> {
        match &self.jacobian {
            Jacobian::Dense(j) => {
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
                let report = match options.linear_solver {
                    LinearSolver::SparseCg(cg) => {
                        conjugate_gradient(&NormalOperator::new(j, damping)?, &(-gradient), &cg)?
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
) -> Result<Linearization, Error> {
    if !x.iter().all(|v| v.is_finite()) {
        return Err(Error::NonFinite);
    }
    let (mut r, mut jacobian) = match options.linear_solver {
        LinearSolver::Dense | LinearSolver::DenseSchur { .. } => {
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
        LinearSolver::SparseCg(_) | LinearSolver::BlockSchur { .. } => {
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
    let mut cost = 0.0;
    let mut weights = DVector::zeros(m);
    for i in 0..m {
        if !r[i].is_finite() {
            return Err(Error::NonFinite);
        }
        let (c, w) = options.loss.value_weight(r[i]);
        cost += c;
        weights[i] = w.sqrt();
        r[i] *= weights[i];
    }
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
        residual: r,
        jacobian,
    })
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
    let (ambient, n, m) = problem.dimensions();
    if ambient == 0 || m == 0 || initial.len() != ambient {
        return Err(Error::InvalidDimensions);
    }
    if !initial.iter().all(|v| v.is_finite()) {
        return Err(Error::NonFinite);
    }
    if options.max_iterations == 0
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
        LinearSolver::SparseCg(_) | LinearSolver::BlockSchur { .. }
    ) && matches!(options.algorithm, Algorithm::GaussNewton)
    {
        return Err(Error::InvalidOptions);
    }
    if let LinearSolver::DenseSchur { retained } = options.linear_solver {
        if retained == 0 || retained >= n || matches!(options.algorithm, Algorithm::GaussNewton) {
            return Err(Error::InvalidOptions);
        }
    }
    if let LinearSolver::SparseCg(cg) = options.linear_solver {
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
    let mut x = initial;
    let mut model = linearize(problem, &x, n, m, options)?;
    let mut cost = model.cost;
    let initial_cost = cost;
    if n == 0 {
        return Ok(Report {
            x,
            initial_cost,
            cost,
            termination: Termination::Gradient,
            iterations: Vec::new(),
        });
    }
    let mut damping = options.initial_damping;
    let mut growth = 2.0;
    let mut history = Vec::new();
    let mut termination = Termination::MaxIterations;
    for iteration in 0..options.max_iterations {
        let gradient = model.gradient()?;
        let gradient_norm = gradient.amax();
        if !gradient_norm.is_finite() {
            return Err(Error::NonFinite);
        }
        if gradient_norm <= options.gradient_tolerance {
            termination = Termination::Gradient;
            break;
        }
        let (step, linear_iterations) = model.step(&gradient, damping, options)?;
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
        let trial_model = linearize(problem, &trial, n, m, options)?;
        let trial_cost = trial_model.cost;
        let predicted = model.predicted(&gradient, &step)?;
        let actual = cost - trial_cost;
        let ratio = if predicted > 0.0 {
            actual / predicted
        } else {
            f64::NEG_INFINITY
        };
        let accepted = actual > 0.0 && ratio > 0.0;
        history.push(Iteration {
            iteration,
            cost,
            trial_cost,
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
            if actual <= options.cost_tolerance * previous.max(1.0) {
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
    Ok(Report {
        x,
        initial_cost,
        cost,
        termination,
        iterations: history,
    })
}
