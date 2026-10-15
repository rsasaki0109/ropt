//! Canonical CSR matrices and Jacobi-preconditioned conjugate gradients.
//! The LM operator computes Jᵀ(Jv) without materializing a dense normal matrix.
use crate::{DVector, Error};

#[derive(Debug, Clone)]
pub struct CsrMatrix {
    rows: usize,
    columns: usize,
    offsets: Vec<usize>,
    indices: Vec<usize>,
    values: Vec<f64>,
}
impl CsrMatrix {
    /// Sort entries and sum duplicates. Exact zero entries are removed.
    /// Returns `InvalidDimensions` if the row-offset array cannot be represented.
    /// This checks allocation layout limits, not available memory.
    pub fn from_triplets(
        rows: usize,
        columns: usize,
        mut entries: Vec<(usize, usize, f64)>,
    ) -> Result<Self, Error> {
        let offset_count = rows.checked_add(1).ok_or(Error::InvalidDimensions)?;
        std::alloc::Layout::array::<usize>(offset_count).map_err(|_| Error::InvalidDimensions)?;
        if entries.iter().any(|&(r, c, _)| r >= rows || c >= columns) {
            return Err(Error::InvalidDimensions);
        }
        if entries.iter().any(|&(_, _, v)| !v.is_finite()) {
            return Err(Error::NonFinite);
        }
        entries.sort_unstable_by_key(|&(r, c, _)| (r, c));
        let mut combined: Vec<(usize, usize, f64)> = Vec::new();
        for (r, c, v) in entries {
            match combined.last_mut() {
                Some(last) if (last.0, last.1) == (r, c) => {
                    last.2 += v;
                    if !last.2.is_finite() {
                        return Err(Error::NonFinite);
                    }
                }
                _ => combined.push((r, c, v)),
            }
        }
        let mut offsets = vec![0; offset_count];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for (r, c, v) in combined {
            if v != 0.0 {
                offsets[r + 1] += 1;
                indices.push(c);
                values.push(v);
            }
        }
        for r in 0..rows {
            offsets[r + 1] += offsets[r];
        }
        Ok(Self {
            rows,
            columns,
            offsets,
            indices,
            values,
        })
    }
    /// Internal checked handoff for assembly that already emits canonical rows.
    /// No sorting, duplicate reduction or pruning is performed here.
    pub(crate) fn from_canonical(
        rows: usize,
        columns: usize,
        offsets: Vec<usize>,
        indices: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<Self, Error> {
        if rows.checked_add(1) != Some(offsets.len())
            || indices.len() != values.len()
            || offsets.first() != Some(&0)
            || offsets.last() != Some(&indices.len())
            || offsets.windows(2).any(|pair| pair[0] > pair[1])
            || indices.iter().any(|&column| column >= columns)
        {
            return Err(Error::InvalidDimensions);
        }
        for row in 0..rows {
            if indices[offsets[row]..offsets[row + 1]]
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            {
                return Err(Error::InvalidDimensions);
            }
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Error::NonFinite);
        }
        if values.contains(&0.) {
            return Err(Error::InvalidDimensions);
        }
        Ok(Self {
            rows,
            columns,
            offsets,
            indices,
            values,
        })
    }
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }
    /// Iterate canonical nonzero entries of one row.
    pub fn row(&self, row: usize) -> Result<impl Iterator<Item = (usize, f64)> + '_, Error> {
        if row >= self.rows {
            return Err(Error::InvalidDimensions);
        }
        Ok((self.offsets[row]..self.offsets[row + 1])
            .map(move |k| (self.indices[k], self.values[k])))
    }
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
    pub(crate) fn pattern(&self) -> (&[usize], &[usize]) {
        (&self.offsets, &self.indices)
    }
    pub(crate) fn values(&self) -> &[f64] {
        &self.values
    }
    pub fn entries(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        (0..self.rows).flat_map(move |r| {
            (self.offsets[r]..self.offsets[r + 1])
                .map(move |k| (r, self.indices[k], self.values[k]))
        })
    }
    pub fn multiply(&self, x: &DVector<f64>) -> Result<DVector<f64>, Error> {
        if x.len() != self.columns {
            return Err(Error::InvalidDimensions);
        }
        finite_vector(x)?;
        let mut out = DVector::zeros(self.rows);
        for r in 0..self.rows {
            for k in self.offsets[r]..self.offsets[r + 1] {
                out[r] += self.values[k] * x[self.indices[k]];
            }
        }
        finite_vector(&out)?;
        Ok(out)
    }
    pub fn transpose_multiply(&self, x: &DVector<f64>) -> Result<DVector<f64>, Error> {
        if x.len() != self.rows {
            return Err(Error::InvalidDimensions);
        }
        finite_vector(x)?;
        let mut out = DVector::zeros(self.columns);
        for (r, c, v) in self.entries() {
            out[c] += v * x[r];
        }
        finite_vector(&out)?;
        Ok(out)
    }
    pub fn normal_diagonal(&self) -> Result<DVector<f64>, Error> {
        let mut d = DVector::zeros(self.columns);
        for (_, c, v) in self.entries() {
            d[c] += v * v;
        }
        finite_vector(&d)?;
        Ok(d)
    }
    pub(crate) fn scale_rows(&mut self, weights: &DVector<f64>) -> Result<(), Error> {
        if weights.len() != self.rows {
            return Err(Error::InvalidDimensions);
        }
        for r in 0..self.rows {
            for k in self.offsets[r]..self.offsets[r + 1] {
                self.values[k] *= weights[r];
                if !self.values[k].is_finite() {
                    return Err(Error::NonFinite);
                }
            }
        }
        Ok(())
    }
}
pub(crate) fn finite_vector(x: &DVector<f64>) -> Result<(), Error> {
    if x.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(Error::NonFinite)
    }
}

#[cfg(test)]
mod canonical_tests {
    use super::*;

    #[test]
    fn canonical_handoff_checks_storage_and_matches_independent_products() {
        let j = CsrMatrix::from_canonical(3, 4, vec![0, 2, 2, 3], vec![0, 3, 2], vec![2., -1., 4.])
            .unwrap();
        assert_eq!(
            j.multiply(&DVector::from_vec(vec![1., 2., 3., 4.]))
                .unwrap(),
            DVector::from_vec(vec![-2., 0., 12.])
        );
        assert_eq!(
            j.transpose_multiply(&DVector::from_vec(vec![1., 2., 3.]))
                .unwrap(),
            DVector::from_vec(vec![2., 0., 12., -1.])
        );
        for (rows, columns, offsets, indices, values) in [
            (2, 3, vec![0], vec![], vec![]),
            (1, 2, vec![1, 1], vec![0], vec![1.]),
            (2, 2, vec![0, 2, 1], vec![0], vec![1.]),
            (1, 2, vec![0, 2], vec![0], vec![1.]),
            (1, 2, vec![0, 1], vec![2], vec![1.]),
            (1, 3, vec![0, 2], vec![2, 1], vec![1., 2.]),
            (1, 3, vec![0, 2], vec![1, 1], vec![1., 2.]),
            (1, 3, vec![0, 1], vec![0], vec![]),
            (usize::MAX, 0, vec![0], vec![], vec![]),
        ] {
            assert_eq!(
                CsrMatrix::from_canonical(rows, columns, offsets, indices, values).unwrap_err(),
                Error::InvalidDimensions
            );
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                CsrMatrix::from_canonical(1, 1, vec![0, 1], vec![0], vec![value]).unwrap_err(),
                Error::NonFinite
            );
        }
        for value in [0., -0.] {
            assert_eq!(
                CsrMatrix::from_canonical(1, 1, vec![0, 1], vec![0], vec![value]).unwrap_err(),
                Error::InvalidDimensions
            );
        }
        let empty = CsrMatrix::from_canonical(2, 0, vec![0, 0, 0], vec![], vec![]).unwrap();
        assert_eq!(
            empty.multiply(&DVector::zeros(0)).unwrap(),
            DVector::zeros(2)
        );
    }
}
/// Implementors must provide a symmetric positive definite operator and its positive diagonal.
pub trait LinearOperator {
    fn dimension(&self) -> usize;
    fn apply(&self, x: &DVector<f64>) -> Result<DVector<f64>, Error>;
    /// Write A*x into an existing n-vector. Implementors must fill every entry.
    /// The default preserves existing implementations; optimized implementations
    /// may reuse the caller's buffer without allocation or interior mutable state.
    fn apply_into(&self, x: &DVector<f64>, out: &mut DVector<f64>) -> Result<(), Error> {
        if x.len() != self.dimension() || out.len() != self.dimension() {
            return Err(Error::InvalidDimensions);
        }
        let result = self.apply(x)?;
        if result.len() != self.dimension() {
            return Err(Error::InvalidDimensions);
        }
        finite_vector(&result)?;
        out.copy_from(&result);
        Ok(())
    }
    fn diagonal(&self) -> Result<DVector<f64>, Error>;
}
#[derive(Debug, Clone, Copy)]
pub struct CgOptions {
    pub max_iterations: usize,
    pub relative_tolerance: f64,
    pub absolute_tolerance: f64,
}
impl Default for CgOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            relative_tolerance: 1e-10,
            absolute_tolerance: 1e-12,
        }
    }
}
impl CgOptions {
    pub(crate) fn validate(&self) -> Result<(), Error> {
        if self.max_iterations == 0
            || [self.relative_tolerance, self.absolute_tolerance]
                .iter()
                .any(|t| !t.is_finite() || *t < 0.0)
        {
            return Err(Error::InvalidOptions);
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CgReport {
    pub x: DVector<f64>,
    pub iterations: usize,
    pub residual_norm: f64,
}
// Preserve the usual dot/sqrt path, but avoid squaring away tiny residuals or
// overflowing on large vectors whose Euclidean norm is still representable.
fn cg_norm(vector: &DVector<f64>) -> Result<f64, Error> {
    let squared = vector.norm_squared();
    let norm = if squared.is_normal() {
        squared.sqrt()
    } else {
        vector.iter().fold(0_f64, |norm, value| norm.hypot(*value))
    };
    if norm.is_finite() {
        Ok(norm)
    } else {
        Err(Error::NonFinite)
    }
}
/// Zero-initialized PCG with a Jacobi preconditioner. A recursive residual alone
/// never establishes convergence: the true b-Ax residual is verified before return.
/// Extreme residual norms use scaled arithmetic; recurrence dot products can
/// still break down, in which case an error is returned rather than convergence.
pub fn conjugate_gradient(
    operator: &impl LinearOperator,
    b: &DVector<f64>,
    options: &CgOptions,
) -> Result<CgReport, Error> {
    let n = operator.dimension();
    if b.len() != n || n == 0 {
        return Err(Error::InvalidDimensions);
    }
    options.validate()?;
    finite_vector(b)?;
    let diagonal = operator.diagonal()?;
    if diagonal.len() != n {
        return Err(Error::InvalidDimensions);
    }
    finite_vector(&diagonal)?;
    if diagonal.iter().any(|v| *v <= 0.0) {
        return Err(Error::LinearSolve);
    }
    let preconditioner = crate::preconditioner::Jacobi::new(diagonal)?;
    conjugate_gradient_preconditioned(operator, &preconditioner, b, options)
}
/// Zero-initialized PCG with an explicit SPD preconditioner and checked true residual.
/// Does not certify global SPD; both operator and preconditioner must satisfy their contracts.
pub fn conjugate_gradient_preconditioned(
    operator: &impl LinearOperator,
    preconditioner: &impl crate::preconditioner::Preconditioner,
    b: &DVector<f64>,
    options: &CgOptions,
) -> Result<CgReport, Error> {
    let n = operator.dimension();
    if n == 0 || b.len() != n || preconditioner.dimension() != n {
        return Err(Error::InvalidDimensions);
    }
    options.validate()?;
    finite_vector(b)?;
    let precondition = |r: &DVector<f64>, z: &mut DVector<f64>| -> Result<(), Error> {
        preconditioner.apply_into(r, z)?;
        if z.len() != n {
            return Err(Error::InvalidDimensions);
        }
        finite_vector(z)
    };
    let norm = cg_norm(b)?;
    let threshold = options
        .absolute_tolerance
        .max(options.relative_tolerance * norm);
    if !threshold.is_finite() {
        return Err(Error::InvalidOptions);
    }
    let apply = |v: &DVector<f64>, out: &mut DVector<f64>| -> Result<(), Error> {
        operator.apply_into(v, out)?;
        if out.len() != n {
            return Err(Error::InvalidDimensions);
        }
        finite_vector(out)
    };
    let mut x = DVector::zeros(n);
    let mut ap = DVector::zeros(n);
    apply(&x, &mut ap)?;
    let mut r = b - &ap;
    let residual_norm = cg_norm(&r)?;
    if residual_norm <= threshold {
        return Ok(CgReport {
            x,
            iterations: 0,
            residual_norm,
        });
    }
    let mut z = DVector::zeros(n);
    precondition(&r, &mut z)?;
    let mut p = z.clone();
    let mut true_r = DVector::zeros(n);
    let mut rz = r.dot(&z);
    for iteration in 1..=options.max_iterations {
        apply(&p, &mut ap)?;
        let curvature = p.dot(&ap);
        if !curvature.is_finite() || !rz.is_finite() {
            return Err(Error::NonFinite);
        }
        if curvature <= 0.0 || rz <= 0.0 {
            return Err(Error::LinearSolve);
        }
        let alpha = rz / curvature;
        x.axpy(alpha, &p, 1.0);
        r.axpy(-alpha, &ap, 1.0);
        finite_vector(&x)?;
        finite_vector(&r)?;
        if cg_norm(&r)? <= threshold {
            apply(&x, &mut true_r)?;
            for k in 0..n {
                true_r[k] = b[k] - true_r[k];
            }
            let residual_norm = cg_norm(&true_r)?;
            if residual_norm <= threshold {
                return Ok(CgReport {
                    x,
                    iterations: iteration,
                    residual_norm,
                });
            }
            // Restart if rounding made the recursive residual too optimistic.
            r.copy_from(&true_r);
            precondition(&r, &mut z)?;
            p.copy_from(&z);
            rz = r.dot(&z);
            continue;
        }
        precondition(&r, &mut z)?;
        let next = r.dot(&z);
        if !next.is_finite() {
            return Err(Error::NonFinite);
        }
        if next <= 0.0 {
            return Err(Error::LinearSolve);
        }
        let beta = next / rz;
        p.axpy(1.0, &z, beta);
        rz = next;
    }
    Err(Error::LinearIterationLimit {
        iterations: options.max_iterations,
    })
}
/// Damped normal operator (JᵀJ + lambda*max(diag(JᵀJ),1)).
/// No normal-matrix entries are stored.
pub struct NormalOperator<'a> {
    jacobian: &'a CsrMatrix,
    damping: f64,
    diagonal: DVector<f64>,
}
impl<'a> NormalOperator<'a> {
    pub fn new(jacobian: &'a CsrMatrix, damping: f64) -> Result<Self, Error> {
        if !damping.is_finite() || damping < 0.0 {
            return Err(Error::InvalidOptions);
        }
        let diagonal = jacobian.normal_diagonal()?;
        Ok(Self {
            jacobian,
            damping,
            diagonal,
        })
    }
}
impl LinearOperator for NormalOperator<'_> {
    fn dimension(&self) -> usize {
        self.jacobian.columns
    }
    fn apply(&self, x: &DVector<f64>) -> Result<DVector<f64>, Error> {
        let mut out = DVector::zeros(self.dimension());
        self.apply_into(x, &mut out)?;
        Ok(out)
    }
    fn apply_into(&self, x: &DVector<f64>, out: &mut DVector<f64>) -> Result<(), Error> {
        if x.len() != self.dimension() || out.len() != self.dimension() {
            return Err(Error::InvalidDimensions);
        }
        finite_vector(x)?;
        out.fill(0.0);
        // Fuse J*x and J^T*(J*x) row by row; no residual-sized scratch vector.
        for row in 0..self.jacobian.rows {
            let range = self.jacobian.offsets[row]..self.jacobian.offsets[row + 1];
            let mut value = 0.0;
            for k in range.clone() {
                value += self.jacobian.values[k] * x[self.jacobian.indices[k]];
            }
            if !value.is_finite() {
                return Err(Error::NonFinite);
            }
            for k in range {
                out[self.jacobian.indices[k]] += self.jacobian.values[k] * value;
            }
        }
        for i in 0..x.len() {
            out[i] += self.damping * self.diagonal[i].max(1.0) * x[i];
        }
        finite_vector(out)
    }
    fn diagonal(&self) -> Result<DVector<f64>, Error> {
        let d = self.diagonal.map(|v| v + self.damping * v.max(1.0));
        finite_vector(&d)?;
        Ok(d)
    }
}

/// Dense reference Schur elimination of trailing variables from an SPD system.
/// Solves the eliminated block by Cholesky (never an explicit inverse), solves the
/// reduced system, then back-substitutes. This is not a sparse BA backend.
pub fn schur_solve(
    system: &crate::DMatrix<f64>,
    rhs: &DVector<f64>,
    retained: usize,
) -> Result<DVector<f64>, Error> {
    let n = system.nrows();
    if system.ncols() != n || rhs.len() != n || retained == 0 || retained >= n {
        return Err(Error::InvalidDimensions);
    }
    if !system.iter().chain(rhs.iter()).all(|v| v.is_finite()) {
        return Err(Error::NonFinite);
    }
    let norm = system.norm();
    if !norm.is_finite() {
        return Err(Error::NonFinite);
    }
    if (system - system.transpose()).norm() > 1e-12 * norm.max(1.0) {
        return Err(Error::LinearSolve);
    }
    let eliminated = n - retained;
    let a = system.view((0, 0), (retained, retained)).into_owned();
    let b = system
        .view((0, retained), (retained, eliminated))
        .into_owned();
    let c = system
        .view((retained, retained), (eliminated, eliminated))
        .into_owned();
    let factor = c.cholesky().ok_or(Error::LinearSolve)?;
    let cb = factor.solve(&b.transpose());
    let cy = factor.solve(&rhs.rows(retained, eliminated).into_owned());
    let reduced = a - &b * &cb;
    let reduced_rhs = rhs.rows(0, retained) - &b * &cy;
    if !reduced
        .iter()
        .chain(reduced_rhs.iter())
        .all(|v| v.is_finite())
    {
        return Err(Error::NonFinite);
    }
    let x = reduced
        .cholesky()
        .ok_or(Error::LinearSolve)?
        .solve(&reduced_rhs);
    let y = cy - cb * &x;
    let mut out = DVector::zeros(n);
    out.rows_mut(0, retained).copy_from(&x);
    out.rows_mut(retained, eliminated).copy_from(&y);
    finite_vector(&out)?;
    Ok(out)
}

impl LinearOperator for CsrMatrix {
    fn dimension(&self) -> usize {
        self.rows
    }
    fn apply(&self, x: &DVector<f64>) -> Result<DVector<f64>, Error> {
        if self.rows != self.columns {
            return Err(Error::InvalidDimensions);
        }
        self.multiply(x)
    }
    fn diagonal(&self) -> Result<DVector<f64>, Error> {
        if self.rows != self.columns {
            return Err(Error::InvalidDimensions);
        }
        let mut d = DVector::zeros(self.rows);
        for (r, c, v) in self.entries() {
            if r == c {
                d[r] = v;
            }
        }
        Ok(d)
    }
}
