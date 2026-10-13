//! SPD preconditioners. Block Jacobi stores only principal normal blocks.
use crate::sparse::{CsrMatrix, finite_vector};
use crate::{DMatrix, DVector, Error};
use nalgebra::{Cholesky, Dyn};
use std::ops::Range;
/// Apply M^-1 to a residual. M must be symmetric positive definite and fixed
/// for one PCG solve; caller-supplied implementations must uphold this contract.
pub trait Preconditioner {
    fn dimension(&self) -> usize;
    fn apply(&self, residual: &DVector<f64>) -> Result<DVector<f64>, Error>;
    /// Fill an existing n-vector with M^-1*r. The default delegates to apply;
    /// implementations can override this to avoid allocations in every PCG iteration.
    fn apply_into(&self, residual: &DVector<f64>, out: &mut DVector<f64>) -> Result<(), Error> {
        if residual.len() != self.dimension() || out.len() != self.dimension() {
            return Err(Error::InvalidDimensions);
        }
        let result = self.apply(residual)?;
        if result.len() != self.dimension() {
            return Err(Error::InvalidDimensions);
        }
        finite_vector(&result)?;
        out.copy_from(&result);
        Ok(())
    }
}
#[derive(Debug, Clone)]
pub struct Jacobi {
    diagonal: DVector<f64>,
}
impl Jacobi {
    pub fn new(diagonal: DVector<f64>) -> Result<Self, Error> {
        if diagonal.is_empty() {
            return Err(Error::InvalidDimensions);
        }
        finite_vector(&diagonal)?;
        if diagonal.iter().any(|v| *v <= 0.0) {
            return Err(Error::LinearSolve);
        }
        Ok(Self { diagonal })
    }
}
impl Preconditioner for Jacobi {
    fn dimension(&self) -> usize {
        self.diagonal.len()
    }
    fn apply(&self, r: &DVector<f64>) -> Result<DVector<f64>, Error> {
        let mut z = DVector::zeros(self.dimension());
        self.apply_into(r, &mut z)?;
        Ok(z)
    }
    fn apply_into(&self, r: &DVector<f64>, z: &mut DVector<f64>) -> Result<(), Error> {
        if r.len() != self.dimension() || z.len() != self.dimension() {
            return Err(Error::InvalidDimensions);
        }
        finite_vector(r)?;
        for k in 0..self.dimension() {
            z[k] = r[k] / self.diagonal[k];
        }
        finite_vector(z)
    }
}
struct Block {
    range: Range<usize>,
    factor: Cholesky<f64, Dyn>,
}
pub struct BlockJacobi {
    dimension: usize,
    blocks: Vec<Block>,
}
pub(crate) fn validate_partition(n: usize, ranges: &[Range<usize>]) -> Result<(), Error> {
    let mut end = 0;
    for range in ranges {
        if range.start != end || range.start >= range.end || range.end > n {
            return Err(Error::InvalidDimensions);
        }
        end = range.end;
    }
    if end != n {
        return Err(Error::InvalidDimensions);
    }
    Ok(())
}
impl BlockJacobi {
    /// Factor each H_bb of H=JᵀJ+lambda*max(diag(JᵀJ),1), from canonical CSR.
    /// Ranges must cover all active columns exactly once, in contiguous order.
    /// Cross-block couplings remain in the full operator, but not in M.
    /// No full normal matrix, explicit inverse, extra jitter or fallback is used.
    pub fn from_normal(
        j: &CsrMatrix,
        damping: f64,
        ranges: &[Range<usize>],
    ) -> Result<Self, Error> {
        let (m, n) = j.shape();
        if n == 0 {
            return Err(Error::InvalidDimensions);
        }
        if !damping.is_finite() || damping < 0.0 {
            return Err(Error::InvalidOptions);
        }
        validate_partition(n, ranges)?;
        let mut mapping = vec![(0, 0); n];
        let mut matrices = Vec::new();
        for (block, range) in ranges.iter().enumerate() {
            matrices.push(DMatrix::<f64>::zeros(range.len(), range.len()));
            for col in range.clone() {
                mapping[col] = (block, col - range.start);
            }
        }
        // Reuse one row scratch buffer; canonical columns allow grouped block products.
        let mut row = Vec::new();
        for r in 0..m {
            row.clear();
            row.extend(j.row(r)?);
            for (i, &(col, value)) in row.iter().enumerate() {
                let (block, offset) = mapping[col];
                for &(other, weight) in &row[i..] {
                    let (other_block, other_offset) = mapping[other];
                    if other_block != block {
                        break;
                    }
                    let product = value * weight;
                    let matrix = &mut matrices[block];
                    matrix[(offset, other_offset)] += product;
                    if offset != other_offset {
                        matrix[(other_offset, offset)] += product;
                    }
                }
            }
        }
        let mut blocks = Vec::new();
        for (range, mut matrix) in ranges.iter().zip(matrices) {
            for k in 0..matrix.nrows() {
                matrix[(k, k)] += damping * matrix[(k, k)].max(1.0);
            }
            if !matrix.iter().all(|v| v.is_finite()) {
                return Err(Error::NonFinite);
            }
            let factor = matrix.cholesky().ok_or(Error::LinearSolve)?;
            blocks.push(Block {
                range: range.clone(),
                factor,
            });
        }
        Ok(Self {
            dimension: n,
            blocks,
        })
    }
    /// Stored dense factor entries; not total allocated memory or a runtime benchmark.
    pub fn stored_entries(&self) -> usize {
        self.blocks
            .iter()
            .map(|b| b.range.len() * b.range.len())
            .sum()
    }
}
impl Preconditioner for BlockJacobi {
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn apply(&self, r: &DVector<f64>) -> Result<DVector<f64>, Error> {
        let mut z = DVector::zeros(self.dimension);
        self.apply_into(r, &mut z)?;
        Ok(z)
    }
    fn apply_into(&self, r: &DVector<f64>, z: &mut DVector<f64>) -> Result<(), Error> {
        if r.len() != self.dimension || z.len() != self.dimension {
            return Err(Error::InvalidDimensions);
        }
        finite_vector(r)?;
        z.copy_from(r);
        for b in &self.blocks {
            b.factor
                .solve_mut(&mut z.rows_mut(b.range.start, b.range.len()));
        }
        finite_vector(z)
    }
}
