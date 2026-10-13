//! Measurement whitening. A transform W maps raw residuals and Jacobians into
//! normalized units. For covariance Sigma, W Sigma Wᵀ = I and WᵀW = Sigma^-1.
use crate::{DMatrix, DVector, Error};
#[derive(Debug, Clone)]
enum Whitening {
    Unit,
    Scalar(f64),
    Diagonal(DVector<f64>),
    Matrix(DMatrix<f64>),
}

#[cfg(test)]
mod workspace_tests {
    use super::*;

    fn bits(values: impl Iterator<Item = f64>) -> Vec<u64> {
        values.map(f64::to_bits).collect()
    }

    #[test]
    fn poisoned_scratch_and_changing_shapes_match_owned_products_bit_for_bit() {
        let mut workspace = WhiteningWorkspace::default();
        for n in [3, 3, 1, 17, 17, 2, 9, 3] {
            let w = DMatrix::from_fn(n, n, |r, c| {
                if r == c {
                    1.5
                } else {
                    ((r + c + 1) as f64 * 0.31).cos() * 0.2
                }
            });
            let noise = NoiseModel::sqrt_information(w.clone()).unwrap();
            let mut residual = DVector::from_fn(n, |r, _| (r as f64 * 0.4).sin());
            residual[0] = -0.;
            let expected_residual = &w * &residual;
            let mut blocks: Vec<_> = [1, 3, 12, 0, 3]
                .into_iter()
                .map(|cols| {
                    DMatrix::from_fn(n, cols, |r, c| {
                        if (r + c) % 5 == 0 {
                            -0.
                        } else {
                            ((r + 2 * c + 1) as f64 * 0.7).sin()
                        }
                    })
                })
                .collect();
            let expected_blocks: Vec<_> = blocks.iter().map(|block| &w * block).collect();
            workspace.residual.resize_vertically_mut(n, f64::NAN);
            workspace.block.resize_mut(n, 1, f64::NAN);
            workspace.residual.fill(f64::NAN);
            workspace.block.fill(f64::NAN);
            noise
                .whiten(&mut residual, &mut blocks, &mut workspace)
                .unwrap();
            assert_eq!(
                bits(residual.iter().copied()),
                bits(expected_residual.iter().copied())
            );
            for (actual, expected) in blocks.iter().zip(expected_blocks) {
                assert_eq!(actual.shape(), expected.shape());
                assert_eq!(bits(actual.iter().copied()), bits(expected.iter().copied()));
            }
        }
        // Finite semidefinite information remains allowed; exact-zero weights
        // still perform the original matrix product rather than dropping rows.
        let noise = NoiseModel::sqrt_information(DMatrix::zeros(3, 3)).unwrap();
        let mut r = DVector::from_element(3, f64::MAX);
        let mut blocks = vec![DMatrix::from_element(3, 2, f64::MAX)];
        noise.whiten(&mut r, &mut blocks, &mut workspace).unwrap();
        assert!(r.iter().chain(blocks[0].iter()).all(|&v| v == 0.));
    }

    #[test]
    fn overflow_errors_do_not_poison_a_later_whitening_call() {
        let mut workspace = WhiteningWorkspace::default();
        let bad = NoiseModel::sqrt_information(DMatrix::identity(3, 3) * 1.7e308).unwrap();
        for (value, derivative) in [(2., 0.), (0., 2.)] {
            let mut residual = DVector::from_element(3, value);
            let mut blocks = vec![DMatrix::from_element(3, 3, derivative)];
            assert_eq!(
                bad.whiten(&mut residual, &mut blocks, &mut workspace),
                Err(Error::NonFinite)
            );
            let w = DMatrix::from_row_slice(3, 3, &[1., 0.3, 0., 0., 2., -0.2, 0.1, 0., 3.]);
            let good = NoiseModel::sqrt_information(w.clone()).unwrap();
            let mut residual = DVector::from_vec(vec![1., -2., 3.]);
            let mut blocks = vec![DMatrix::identity(3, 3)];
            let expected_r = &w * &residual;
            let expected_j = &w * &blocks[0];
            good.whiten(&mut residual, &mut blocks, &mut workspace)
                .unwrap();
            assert_eq!(
                bits(residual.iter().copied()),
                bits(expected_r.iter().copied())
            );
            assert_eq!(
                bits(blocks[0].iter().copied()),
                bits(expected_j.iter().copied())
            );
        }
    }
}
#[derive(Debug, Clone)]
pub struct NoiseModel {
    whitening: Whitening,
}
// Owned by one graph evaluation; callbacks keep their existing owned buffers.
#[derive(Default)]
pub(crate) struct WhiteningWorkspace {
    residual: DVector<f64>,
    block: DMatrix<f64>,
}
impl Default for NoiseModel {
    fn default() -> Self {
        Self {
            whitening: Whitening::Unit,
        }
    }
}
impl NoiseModel {
    pub fn unit() -> Self {
        Self::default()
    }
    pub fn isotropic(sigma: f64) -> Result<Self, Error> {
        if !sigma.is_finite() {
            return Err(Error::NonFinite);
        }
        if sigma <= 0.0 {
            return Err(Error::InvalidOptions);
        }
        let weight = sigma.recip();
        if !weight.is_finite() {
            return Err(Error::NonFinite);
        }
        Ok(Self {
            whitening: Whitening::Scalar(weight),
        })
    }
    pub fn diagonal(sigmas: DVector<f64>) -> Result<Self, Error> {
        if sigmas.is_empty() {
            return Err(Error::InvalidDimensions);
        }
        if !sigmas.iter().all(|s| s.is_finite()) {
            return Err(Error::NonFinite);
        }
        if sigmas.iter().any(|s| *s <= 0.0) {
            return Err(Error::InvalidOptions);
        }
        let weights = sigmas.map(f64::recip);
        if !weights.iter().all(|v| v.is_finite()) {
            return Err(Error::NonFinite);
        }
        Ok(Self {
            whitening: Whitening::Diagonal(weights),
        })
    }
    /// Accepts a finite square W. Rank-deficient W is allowed and represents
    /// semidefinite information; callers are responsible for observability.
    pub fn sqrt_information(matrix: DMatrix<f64>) -> Result<Self, Error> {
        if matrix.nrows() == 0 || matrix.nrows() != matrix.ncols() {
            return Err(Error::InvalidDimensions);
        }
        if !matrix.iter().all(|v| v.is_finite()) {
            return Err(Error::NonFinite);
        }
        Ok(Self {
            whitening: Whitening::Matrix(matrix),
        })
    }
    /// Uses a Cholesky triangular solve to construct W=L^-1. Never inverts Sigma.
    pub fn from_covariance(covariance: DMatrix<f64>) -> Result<Self, Error> {
        let n = covariance.nrows();
        if n == 0 || covariance.ncols() != n {
            return Err(Error::InvalidDimensions);
        }
        if !covariance.iter().all(|v| v.is_finite()) {
            return Err(Error::NonFinite);
        }
        let scale = covariance.amax();
        if (&covariance - covariance.transpose()).amax() > 1e-12 * scale {
            return Err(Error::InvalidOptions);
        }
        let lower = covariance.cholesky().ok_or(Error::InvalidOptions)?.l();
        let transform = lower
            .solve_lower_triangular(&DMatrix::identity(n, n))
            .ok_or(Error::InvalidOptions)?;
        Self::sqrt_information(transform)
    }
    /// Factor SPD information as Omega=L*Lᵀ and whiten with W=Lᵀ.
    /// The transpose matters: Wᵀ*W must equal Omega, including correlations.
    pub fn from_information(information: DMatrix<f64>) -> Result<Self, Error> {
        let n = information.nrows();
        if n == 0 || information.ncols() != n {
            return Err(Error::InvalidDimensions);
        }
        if !information.iter().all(|v| v.is_finite()) {
            return Err(Error::NonFinite);
        }
        if (&information - information.transpose()).amax() > 1e-12 * information.amax() {
            return Err(Error::InvalidOptions);
        }
        let lower = information.cholesky().ok_or(Error::InvalidOptions)?.l();
        Self::sqrt_information(lower.transpose())
    }
    pub(crate) fn validate_dimension(&self, rows: usize) -> Result<(), Error> {
        let dimension = match &self.whitening {
            Whitening::Matrix(w) => Some(w.nrows()),
            Whitening::Diagonal(w) => Some(w.len()),
            _ => None,
        };
        if dimension.is_some_and(|n| n != rows) {
            return Err(Error::InvalidDimensions);
        }
        Ok(())
    }
    pub(crate) fn whiten(
        &self,
        residual: &mut DVector<f64>,
        blocks: &mut [DMatrix<f64>],
        workspace: &mut WhiteningWorkspace,
    ) -> Result<(), Error> {
        self.validate_dimension(residual.len())?;
        match &self.whitening {
            Whitening::Unit => {}
            Whitening::Scalar(w) => {
                residual.scale_mut(*w);
                for block in &mut *blocks {
                    block.scale_mut(*w);
                }
            }
            Whitening::Diagonal(w) => {
                residual.component_mul_assign(w);
                for block in &mut *blocks {
                    for i in 0..w.len() {
                        block.row_mut(i).scale_mut(w[i]);
                    }
                }
            }
            Whitening::Matrix(w) => {
                if workspace.residual.len() != residual.len() {
                    workspace.residual.resize_vertically_mut(residual.len(), 0.);
                }
                // The owned multiplication uses the same nalgebra GEMM kernel.
                // beta=0 overwrites scratch without reading its previous values.
                workspace.residual.gemm(1., w, &*residual, 0.);
                std::mem::swap(residual, &mut workspace.residual);
                for block in &mut *blocks {
                    if workspace.block.shape() != block.shape() {
                        workspace.block.resize_mut(block.nrows(), block.ncols(), 0.);
                    }
                    workspace.block.gemm(1., w, &*block, 0.);
                    std::mem::swap(block, &mut workspace.block);
                }
            }
        }
        if !residual
            .iter()
            .chain(blocks.iter().flat_map(|b| b.iter()))
            .all(|v| v.is_finite())
        {
            return Err(Error::NonFinite);
        }
        Ok(())
    }
}
