//! Dense square-root elimination of a linearized least-squares system.
//! Uses orthogonal projection, not a normal-matrix Schur complement or inverse.
use crate::{DMatrix, DVector, Error};

#[derive(Debug, Clone, Copy)]
pub struct MarginalizationOptions {
    /// Relative SVD cutoff for the eliminated Jacobian columns.
    pub relative_rank_tolerance: f64,
    pub svd_max_iterations: usize,
    /// Maximum entries in the input augmented matrix `[J,r]`. Not a peak-byte limit.
    pub max_entries: usize,
}
impl Default for MarginalizationOptions {
    fn default() -> Self {
        Self {
            relative_rank_tolerance: 1e-12,
            svd_max_iterations: 1000,
            max_entries: 1_000_000,
        }
    }
}
impl MarginalizationOptions {
    pub(crate) fn validate(self) -> Result<(), Error> {
        if !self.relative_rank_tolerance.is_finite()
            || self.relative_rank_tolerance <= 0.0
            || self.relative_rank_tolerance >= 1.0
            || self.svd_max_iterations == 0
            || self.max_entries == 0
        {
            return Err(Error::InvalidOptions);
        }
        Ok(())
    }
    pub(crate) fn check_size(self, rows: usize, columns: usize) -> Result<(), Error> {
        self.validate()?;
        if rows
            .checked_mul(columns.checked_add(1).ok_or(Error::InvalidDimensions)?)
            .is_none_or(|entries| entries > self.max_entries)
        {
            return Err(Error::Evaluation(
                "marginalization matrix entry limit exceeded".into(),
            ));
        }
        Ok(())
    }
}
/// ||jacobian*retained_delta + residual||² equals the minimized original
/// linearized squared residual, including any irreducible constant term.
#[derive(Debug, Clone)]
pub struct LinearizedPrior {
    pub jacobian: DMatrix<f64>,
    pub residual: DVector<f64>,
}
/// Eliminate the first `eliminated` columns from min ||J*d+r||².
/// Eliminated columns must have full rank; retained columns may be rank deficient.
/// No damping, rank truncation or pseudoinverse is used.
pub fn marginalize_linearized(
    j: &DMatrix<f64>,
    r: &DVector<f64>,
    eliminated: usize,
    options: &MarginalizationOptions,
) -> Result<LinearizedPrior, Error> {
    let (m, n) = j.shape();
    options.check_size(m, n)?;
    if m == 0 || r.len() != m || eliminated > n {
        return Err(Error::InvalidDimensions);
    }
    if !j.iter().chain(r.iter()).all(|v| v.is_finite()) {
        return Err(Error::NonFinite);
    }
    let retained = n - eliminated;
    let mut augmented = DMatrix::zeros(m, retained + 1);
    augmented
        .columns_mut(0, retained)
        .copy_from(&j.columns(eliminated, retained));
    augmented.set_column(retained, r);
    if eliminated > 0 {
        let mut e = j.columns(0, eliminated).into_owned();
        let scale = e.amax();
        let rank = if scale == 0.0 {
            0
        } else {
            e.iter_mut().for_each(|v| *v /= scale);
            let svd = e
                .clone()
                .try_svd(false, false, f64::EPSILON, options.svd_max_iterations)
                .ok_or(Error::LinearSolve)?;
            if !svd.singular_values.iter().all(|v| v.is_finite()) {
                return Err(Error::NonFinite);
            }
            let cutoff = options.relative_rank_tolerance * svd.singular_values.amax();
            svd.singular_values.iter().filter(|&&s| s > cutoff).count()
        };
        if rank < eliminated {
            return Err(Error::RankDeficient {
                rank,
                parameters: eliminated,
            });
        }
        e.qr().q_tr_mul(&mut augmented);
        augmented = augmented.rows(eliminated, m - eliminated).into_owned();
    }
    // QR of the augmented matrix compresses both retained information and the
    // constant residual component, even when the retained Jacobian is singular.
    let compressed = if augmented.nrows() == 0 {
        DMatrix::zeros(1, retained + 1)
    } else {
        let scale = augmented.amax();
        if scale > 0.0 {
            augmented.iter_mut().for_each(|v| *v /= scale);
        }
        let mut r = augmented.qr().r();
        if scale > 0.0 {
            r *= scale;
        }
        r
    };
    if !compressed.iter().all(|v| v.is_finite()) {
        return Err(Error::NonFinite);
    }
    Ok(LinearizedPrior {
        jacobian: compressed.columns(0, retained).into_owned(),
        residual: compressed.column(retained).into_owned(),
    })
}
