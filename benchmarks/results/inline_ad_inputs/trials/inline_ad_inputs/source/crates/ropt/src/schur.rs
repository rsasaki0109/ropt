//! Independent-block Schur elimination from a CSR Jacobian.
//! The reduced operator is matrix-free; the retained Jacobian and cross-block rows
//! are sparse, while each independent eliminated block is factored densely.
use crate::sparse::{
    CgOptions, CgReport, CsrMatrix, LinearOperator, NormalOperator, conjugate_gradient,
};
use crate::{DMatrix, DVector, Error};
use nalgebra::{Cholesky, Dyn};
use std::collections::BTreeMap;

struct EliminatedBlock {
    factor: Cholesky<f64, Dyn>,
    cross: Vec<(usize, DVector<f64>)>,
}
/// Active columns must be a retained prefix followed by equally sized independent
/// blocks. A residual row may touch at most one eliminated block. Unsupported
/// coupling is rejected rather than silently discarded.
pub struct BlockSchurSystem {
    retained_jacobian: CsrMatrix,
    retained: usize,
    block_size: usize,
    blocks: Vec<EliminatedBlock>,
    damping: f64,
    original_diagonal: DVector<f64>,
    reduced_diagonal: DVector<f64>,
}
impl BlockSchurSystem {
    pub fn new(
        jacobian: &CsrMatrix,
        damping: f64,
        retained: usize,
        block_size: usize,
    ) -> Result<Self, Error> {
        let (m, n) = jacobian.shape();
        if retained == 0
            || retained >= n
            || block_size == 0
            || !(n - retained).is_multiple_of(block_size)
        {
            return Err(Error::InvalidDimensions);
        }
        if !damping.is_finite() || damping < 0.0 {
            return Err(Error::InvalidOptions);
        }
        let count = (n - retained) / block_size;
        let diagonal = jacobian.normal_diagonal()?;
        let mut normals = vec![DMatrix::<f64>::zeros(block_size, block_size); count];
        let mut cross: Vec<BTreeMap<usize, DVector<f64>>> = vec![BTreeMap::new(); count];
        let mut retained_entries = Vec::new();
        for row in 0..m {
            let mut left = Vec::new();
            let mut right = Vec::new();
            let mut block = None;
            for (column, value) in jacobian.row(row)? {
                if column < retained {
                    left.push((column, value));
                    retained_entries.push((row, column, value));
                } else {
                    let index = (column - retained) / block_size;
                    if block.is_some_and(|previous| previous != index) {
                        return Err(Error::Evaluation(
                            "residual row couples eliminated Schur blocks".into(),
                        ));
                    }
                    block = Some(index);
                    right.push(((column - retained) % block_size, value));
                }
            }
            if let Some(index) = block {
                for &(a, x) in &right {
                    for &(b, y) in &right {
                        normals[index][(a, b)] += x * y;
                    }
                }
                for &(a, x) in &left {
                    let vector = cross[index]
                        .entry(a)
                        .or_insert_with(|| DVector::zeros(block_size));
                    for &(b, y) in &right {
                        vector[b] += x * y;
                    }
                }
            }
        }
        let mut blocks = Vec::with_capacity(count);
        let mut reduced_diagonal = diagonal.rows(0, retained).map(|v| v + damping * v.max(1.0));
        for (index, (mut normal, cross)) in normals.into_iter().zip(cross).enumerate() {
            for k in 0..block_size {
                normal[(k, k)] += damping * diagonal[retained + index * block_size + k].max(1.0);
            }
            if !normal.iter().all(|v| v.is_finite()) {
                return Err(Error::NonFinite);
            }
            let factor = normal.cholesky().ok_or(Error::LinearSolve)?;
            let cross: Vec<_> = cross.into_iter().collect();
            for (row, vector) in &cross {
                if !vector.iter().all(|v| v.is_finite()) {
                    return Err(Error::NonFinite);
                }
                reduced_diagonal[*row] -= vector.dot(&factor.solve(vector));
            }
            blocks.push(EliminatedBlock { factor, cross });
        }
        if !reduced_diagonal.iter().all(|v| v.is_finite()) {
            return Err(Error::NonFinite);
        }
        Ok(Self {
            retained_jacobian: CsrMatrix::from_triplets(m, retained, retained_entries)?,
            retained,
            block_size,
            blocks,
            damping,
            original_diagonal: diagonal,
            reduced_diagonal,
        })
    }
    pub fn eliminated_blocks(&self) -> usize {
        self.blocks.len()
    }
    /// Number of retained rows with nonzero coupling to eliminated blocks.
    pub fn cross_rows(&self) -> usize {
        self.blocks.iter().map(|b| b.cross.len()).sum()
    }
    pub fn reduced_rhs(&self, rhs: &DVector<f64>) -> Result<DVector<f64>, Error> {
        self.check_rhs(rhs)?;
        let mut out = rhs.rows(0, self.retained).into_owned();
        for (index, block) in self.blocks.iter().enumerate() {
            let solved = block.factor.solve(
                &rhs.rows(self.retained + index * self.block_size, self.block_size)
                    .into_owned(),
            );
            for (row, vector) in &block.cross {
                out[*row] -= vector.dot(&solved);
            }
        }
        finite(&out)?;
        Ok(out)
    }
    pub fn back_substitute(
        &self,
        rhs: &DVector<f64>,
        retained_step: &DVector<f64>,
    ) -> Result<DVector<f64>, Error> {
        self.check_rhs(rhs)?;
        if retained_step.len() != self.retained {
            return Err(Error::InvalidDimensions);
        }
        finite(retained_step)?;
        let mut out = DVector::zeros(rhs.len());
        out.rows_mut(0, self.retained).copy_from(retained_step);
        for (index, block) in self.blocks.iter().enumerate() {
            let start = self.retained + index * self.block_size;
            let mut right = rhs.rows(start, self.block_size).into_owned();
            for (row, vector) in &block.cross {
                right.axpy(-retained_step[*row], vector, 1.0);
            }
            let step = block.factor.solve(&right);
            out.rows_mut(start, self.block_size).copy_from(&step);
        }
        finite(&out)?;
        Ok(out)
    }
    fn check_rhs(&self, rhs: &DVector<f64>) -> Result<(), Error> {
        if rhs.len() != self.original_diagonal.len() {
            return Err(Error::InvalidDimensions);
        }
        finite(rhs)
    }
}
impl LinearOperator for BlockSchurSystem {
    fn dimension(&self) -> usize {
        self.retained
    }
    fn apply(&self, x: &DVector<f64>) -> Result<DVector<f64>, Error> {
        if x.len() != self.retained {
            return Err(Error::InvalidDimensions);
        }
        finite(x)?;
        let mut out = self
            .retained_jacobian
            .transpose_multiply(&self.retained_jacobian.multiply(x)?)?;
        for k in 0..self.retained {
            out[k] += self.damping * self.original_diagonal[k].max(1.0) * x[k];
        }
        for block in &self.blocks {
            let mut right = DVector::zeros(self.block_size);
            for (row, vector) in &block.cross {
                right.axpy(x[*row], vector, 1.0);
            }
            let solved = block.factor.solve(&right);
            for (row, vector) in &block.cross {
                out[*row] -= vector.dot(&solved);
            }
        }
        finite(&out)?;
        Ok(out)
    }
    fn diagonal(&self) -> Result<DVector<f64>, Error> {
        Ok(self.reduced_diagonal.clone())
    }
}
fn finite(v: &DVector<f64>) -> Result<(), Error> {
    if v.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(Error::NonFinite)
    }
}
/// Solve a damped normal system using independent-block elimination and reduced
/// PCG. The returned residual norm is verified against the full original system.
pub fn block_schur_solve(
    jacobian: &CsrMatrix,
    rhs: &DVector<f64>,
    damping: f64,
    retained: usize,
    block_size: usize,
    cg: &CgOptions,
) -> Result<CgReport, Error> {
    cg.validate()?;
    if rhs.len() != jacobian.shape().1 {
        return Err(Error::InvalidDimensions);
    }
    finite(rhs)?;
    let system = BlockSchurSystem::new(jacobian, damping, retained, block_size)?;
    let threshold = cg
        .absolute_tolerance
        .max(cg.relative_tolerance * rhs.norm());
    if !threshold.is_finite() {
        return Err(Error::NonFinite);
    }
    // Leave headroom for back-substitution rounding and verify the full residual.
    let reduced_options = CgOptions {
        absolute_tolerance: threshold * 0.1,
        relative_tolerance: 0.0,
        ..*cg
    };
    let result = conjugate_gradient(&system, &system.reduced_rhs(rhs)?, &reduced_options)?;
    let x = system.back_substitute(rhs, &result.x)?;
    let residual_norm = (rhs - NormalOperator::new(jacobian, damping)?.apply(&x)?).norm();
    if !residual_norm.is_finite() {
        return Err(Error::NonFinite);
    }
    if residual_norm > threshold {
        return Err(Error::LinearResidualTooLarge {
            residual_norm,
            tolerance: threshold,
        });
    }
    Ok(CgReport {
        x,
        iterations: result.iterations,
        residual_norm,
    })
}
