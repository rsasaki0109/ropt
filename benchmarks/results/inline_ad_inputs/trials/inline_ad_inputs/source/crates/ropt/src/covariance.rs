//! Selected marginal covariance from a full whitened local Jacobian via SVD.
//! No damping, pseudoinverse, automatic residual-variance estimate or sparse factorization.
use crate::solver::{robust_weights, validated_loss_blocks};
use crate::{DMatrix, DVector, Error, LocalProblem, Loss};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, Default)]
pub enum CovarianceModel {
    /// Local Gaussian/Gauss–Newton approximation; rejects robust factor losses.
    #[default]
    Gaussian,
    /// Inverse frozen-IRLS information, not robust sandwich covariance or calibrated confidence.
    /// Match fallback_loss to the scalar loss used by the nonlinear solve.
    FrozenIrls { fallback_loss: Loss },
}
#[derive(Debug, Clone, Copy, Default)]
pub enum CovarianceInput {
    #[default]
    Dense,
    /// Explicitly converts CSR into a dense Jacobian for global SVD rank validation.
    /// This is a sparse input path, not a sparse or matrix-free covariance algorithm.
    SparseToDense,
}
#[derive(Debug, Clone, Copy)]
pub struct CovarianceOptions {
    pub model: CovarianceModel,
    pub input: CovarianceInput,
    /// Singular values at or below this fraction of the largest are treated as unobservable.
    /// Positive and less than one; sensitive to coordinate units and scaling.
    pub relative_rank_tolerance: f64,
    /// Explicit multiplicative noise-variance scale. Defaults to known whitened noise (1).
    /// Does not automatically use residual sum of squares or degrees of freedom.
    pub variance_scale: f64,
    /// Total SVD iteration limit; zero (unbounded) is rejected.
    pub svd_max_iterations: usize,
}
impl Default for CovarianceOptions {
    fn default() -> Self {
        Self {
            model: CovarianceModel::Gaussian,
            input: CovarianceInput::Dense,
            relative_rank_tolerance: 1e-12,
            variance_scale: 1.0,
            svd_max_iterations: 10000,
        }
    }
}
impl CovarianceOptions {
    fn validate(&self) -> Result<(), Error> {
        if self.svd_max_iterations == 0
            || !self.relative_rank_tolerance.is_finite()
            || self.relative_rank_tolerance <= 0.0
            || self.relative_rank_tolerance >= 1.0
            || !self.variance_scale.is_finite()
            || self.variance_scale <= 0.0
        {
            return Err(Error::InvalidOptions);
        }
        if let CovarianceModel::FrozenIrls { fallback_loss } = self.model {
            if !fallback_loss.valid() {
                return Err(Error::InvalidOptions);
            }
        }
        Ok(())
    }
}
#[derive(Debug, Clone)]
pub struct TangentCovariance {
    /// Active tangent indices, in the requested order, at the evaluation layout.
    pub indices: Vec<usize>,
    pub matrix: DMatrix<f64>,
    pub model: CovarianceModel,
    /// Rank of the full weighted Jacobian, including unqueried active parameters.
    pub rank: usize,
    /// Spectral condition number of the Jacobian, not its squared normal matrix.
    pub condition_number: f64,
}
/// Compute variance_scale * selected((JᵀJ)^-1), retaining all other active variables
/// as uncertain nuisance variables. J is already whitened by the LocalProblem.
/// Coordinates are derivatives of that problem's retraction, not ambient quaternion coefficients.
/// The evaluation point need not be an optimum; this API does not certify convergence.
pub fn marginal_covariance(
    problem: &impl LocalProblem,
    x: &DVector<f64>,
    indices: &[usize],
    options: &CovarianceOptions,
) -> Result<TangentCovariance, Error> {
    options.validate()?;
    let (ambient, n, m) = problem.dimensions();
    if ambient == 0 || n == 0 || m == 0 || x.len() != ambient || indices.is_empty() {
        return Err(Error::InvalidDimensions);
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err(Error::NonFinite);
    }
    let mut seen = BTreeSet::new();
    if indices.iter().any(|&i| i >= n || !seen.insert(i)) {
        return Err(Error::InvalidDimensions);
    }
    let blocks = validated_loss_blocks(problem, m)?;
    let fallback = match options.model {
        CovarianceModel::Gaussian => {
            if blocks.iter().any(|b| !matches!(b.loss, Loss::Squared)) {
                return Err(Error::Evaluation(
                    "robust factors require explicit FrozenIrls covariance model".into(),
                ));
            }
            Loss::Squared
        }
        CovarianceModel::FrozenIrls { fallback_loss } => fallback_loss,
    };
    let (residual, mut j) = match options.input {
        CovarianceInput::Dense => {
            let mut r = DVector::from_element(m, f64::NAN);
            let mut j = DMatrix::from_element(m, n, f64::NAN);
            problem.evaluate_local(x, &mut r, &mut j)?;
            (r, j)
        }
        CovarianceInput::SparseToDense => {
            let (r, s) = problem.evaluate_sparse(x)?;
            if s.shape() != (m, n) {
                return Err(Error::InvalidDimensions);
            }
            let mut j = DMatrix::zeros(m, n);
            for (row, column, value) in s.entries() {
                j[(row, column)] = value;
            }
            (r, j)
        }
    };
    if residual.len() != m || j.shape() != (m, n) {
        return Err(Error::InvalidDimensions);
    }
    if !residual.iter().chain(j.iter()).all(|v| v.is_finite()) {
        return Err(Error::NonFinite);
    }
    if matches!(options.model, CovarianceModel::FrozenIrls { .. }) {
        let (cost, weights) = robust_weights(&residual, fallback, &blocks)?;
        if !cost.is_finite() || !weights.iter().all(|v| v.is_finite()) {
            return Err(Error::NonFinite);
        }
        for row in 0..m {
            j.row_mut(row).scale_mut(weights[row]);
        }
        if !j.iter().all(|v| v.is_finite()) {
            return Err(Error::NonFinite);
        }
    }
    // Normalize before SVD to avoid overflow/underflow in the decomposition.
    let scale = j.amax();
    if scale == 0.0 {
        return Err(Error::RankDeficient {
            rank: 0,
            parameters: n,
        });
    }
    for v in j.iter_mut() {
        *v /= scale;
    }
    let svd = j
        .try_svd(false, true, f64::EPSILON, options.svd_max_iterations)
        .ok_or(Error::LinearSolve)?;
    if !svd.singular_values.iter().all(|v| v.is_finite()) {
        return Err(Error::NonFinite);
    }
    let largest = svd.singular_values.amax();
    let cutoff = options.relative_rank_tolerance * largest;
    let rank = svd.singular_values.iter().filter(|&&s| s > cutoff).count();
    if rank < n {
        return Err(Error::RankDeficient {
            rank,
            parameters: n,
        });
    }
    let condition_number = largest / svd.singular_values.min();
    let vt = svd.v_t.ok_or(Error::LinearSolve)?;
    let variance_root = options.variance_scale.sqrt();
    // Only K selected rows of V/Σ are assembled; no full inverse covariance is formed.
    let projected = DMatrix::from_fn(indices.len(), n, |row, k| {
        vt[(k, indices[row])] / svd.singular_values[k] / scale * variance_root
    });
    if !condition_number.is_finite() || !projected.iter().all(|v| v.is_finite()) {
        return Err(Error::NonFinite);
    }
    let matrix = &projected * projected.transpose();
    if !matrix.iter().all(|v| v.is_finite()) {
        return Err(Error::NonFinite);
    }
    // Positive rank cannot give zero marginal variance in representable arithmetic.
    // Reject underflow rather than presenting an exactly certain active parameter.
    if matrix.diagonal().iter().any(|v| *v <= 0.0) {
        return Err(Error::LinearSolve);
    }
    Ok(TangentCovariance {
        indices: indices.to_vec(),
        matrix,
        model: options.model,
        rank,
        condition_number,
    })
}

/// Selected Euclidean covariance for the standalone Residual API.
/// SparseToDense still requires explicit sparse support and has no dense fallback.
pub fn marginal_covariance_euclidean(
    problem: &impl crate::Residual,
    x: &DVector<f64>,
    indices: &[usize],
    options: &CovarianceOptions,
) -> Result<TangentCovariance, Error> {
    marginal_covariance(
        &crate::solver::EuclideanProblem(problem),
        x,
        indices,
        options,
    )
}
