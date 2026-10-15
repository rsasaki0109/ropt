//! Dense least-squares steps without forming normal equations.
use crate::{DMatrix, DVector, Error};
#[derive(Debug, Clone, Copy)]
pub struct QrOptions {
    /// Relative cutoff on pivoted R diagonal magnitudes, not singular values.
    pub relative_rank_tolerance: f64,
}
impl Default for QrOptions {
    fn default() -> Self {
        Self {
            relative_rank_tolerance: 1e-14,
        }
    }
}
impl QrOptions {
    pub(crate) fn validate(&self) -> Result<(), Error> {
        if !self.relative_rank_tolerance.is_finite()
            || self.relative_rank_tolerance <= 0.0
            || self.relative_rank_tolerance >= 1.0
        {
            return Err(Error::InvalidOptions);
        }
        Ok(())
    }
}
/// Solve min_d ||J*d+r||² + damping*sum(max(||J_col||²,1)*d_col²).
/// Zero damping is a full-column-rank GN step. Positive damping is scaled LM.
/// QR acts on [J; sqrt(damping)*D], where D=max(column norm,1).
/// Column pivots are undone before returning; no normal matrix or fallback is used.
pub fn qr_step(
    jacobian: &DMatrix<f64>,
    residual: &DVector<f64>,
    damping: f64,
    options: &QrOptions,
) -> Result<DVector<f64>, Error> {
    options.validate()?;
    let (m, n) = jacobian.shape();
    if m == 0 || n == 0 || residual.len() != m {
        return Err(Error::InvalidDimensions);
    }
    if !jacobian
        .iter()
        .chain(residual.iter())
        .all(|v| v.is_finite())
    {
        return Err(Error::NonFinite);
    }
    if !damping.is_finite() || damping < 0.0 {
        return Err(Error::InvalidOptions);
    }
    let augmented = damping > 0.0;
    let rows = if augmented {
        m.checked_add(n).ok_or(Error::InvalidDimensions)?
    } else {
        m
    };
    let mut a = DMatrix::zeros(rows, n);
    a.rows_mut(0, m).copy_from(jacobian);
    let mut b = DVector::zeros(rows);
    b.rows_mut(0, m).copy_from(&(-residual));
    if augmented {
        let root = damping.sqrt();
        for column in 0..n {
            // Stable norm avoids unnecessary overflow in sum(J_col²) or lambda*diag.
            let norm = jacobian
                .column(column)
                .iter()
                .fold(0.0_f64, |norm, &v| norm.hypot(v));
            let entry = root * norm.max(1.0);
            if !norm.is_finite() || !entry.is_finite() {
                return Err(Error::NonFinite);
            }
            a[(m + column, column)] = entry;
        }
    }
    let scale = a.amax();
    if scale == 0.0 {
        return Err(Error::RankDeficient {
            rank: 0,
            parameters: n,
        });
    }
    a.iter_mut().for_each(|v| *v /= scale);
    b.iter_mut().for_each(|v| *v /= scale);
    if !b.iter().all(|v| v.is_finite()) {
        return Err(Error::NonFinite);
    }
    let qr = a.col_piv_qr();
    let r = qr.r();
    if !r.iter().all(|v| v.is_finite()) {
        return Err(Error::NonFinite);
    }
    let diagonal = DVector::from_iterator(
        r.nrows().min(n),
        (0..r.nrows().min(n)).map(|i| r[(i, i)].abs()),
    );
    let cutoff = options.relative_rank_tolerance * diagonal.amax();
    let rank = diagonal.iter().filter(|&&v| v > cutoff).count();
    if rank < n {
        return Err(Error::RankDeficient {
            rank,
            parameters: n,
        });
    }
    qr.q_tr_mul(&mut b);
    if !b.iter().all(|v| v.is_finite()) {
        return Err(Error::NonFinite);
    }
    let mut step = r
        .solve_upper_triangular(&b.rows(0, n).into_owned())
        .ok_or(Error::LinearSolve)?;
    qr.p().inv_permute_rows(&mut step);
    if !step.iter().all(|v| v.is_finite()) {
        return Err(Error::NonFinite);
    }
    Ok(step)
}
