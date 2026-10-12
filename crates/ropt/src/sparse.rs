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
    pub fn from_triplets(
        rows: usize,
        columns: usize,
        mut entries: Vec<(usize, usize, f64)>,
    ) -> Result<Self, Error> {
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
        let mut offsets = vec![0; rows + 1];
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
fn finite_vector(x: &DVector<f64>) -> Result<(), Error> {
    if x.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(Error::NonFinite)
    }
}
/// Implementors must provide a symmetric positive definite operator and its positive diagonal.
pub trait LinearOperator {
    fn dimension(&self) -> usize;
    fn apply(&self, x: &DVector<f64>) -> Result<DVector<f64>, Error>;
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
/// Zero-initialized PCG with a Jacobi preconditioner. A recursive residual alone
/// never establishes convergence: the true b-Ax residual is verified before return.
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
    let norm = b.norm();
    if !norm.is_finite() {
        return Err(Error::NonFinite);
    }
    let threshold = options
        .absolute_tolerance
        .max(options.relative_tolerance * norm);
    if !threshold.is_finite() {
        return Err(Error::InvalidOptions);
    }
    let apply = |v: &DVector<f64>| -> Result<DVector<f64>, Error> {
        let out = operator.apply(v)?;
        if out.len() != n {
            return Err(Error::InvalidDimensions);
        }
        finite_vector(&out)?;
        Ok(out)
    };
    let mut x = DVector::zeros(n);
    let mut r = b - &apply(&x)?;
    if r.norm() <= threshold {
        return Ok(CgReport {
            x,
            iterations: 0,
            residual_norm: r.norm(),
        });
    }
    let mut z = r.component_div(&diagonal);
    let mut p = z.clone();
    let mut rz = r.dot(&z);
    for iteration in 1..=options.max_iterations {
        let ap = apply(&p)?;
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
        if r.norm() <= threshold {
            let true_r = b - apply(&x)?;
            let residual_norm = true_r.norm();
            if residual_norm <= threshold {
                return Ok(CgReport {
                    x,
                    iterations: iteration,
                    residual_norm,
                });
            }
            // Restart if rounding made the recursive residual too optimistic.
            r = true_r;
            z = r.component_div(&diagonal);
            p = z.clone();
            rz = r.dot(&z);
            continue;
        }
        z = r.component_div(&diagonal);
        let next = r.dot(&z);
        let beta = next / rz;
        p = &z + beta * p;
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
        let mut out = self
            .jacobian
            .transpose_multiply(&self.jacobian.multiply(x)?)?;
        for i in 0..x.len() {
            out[i] += self.damping * self.diagonal[i].max(1.0) * x[i];
        }
        finite_vector(&out)?;
        Ok(out)
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
