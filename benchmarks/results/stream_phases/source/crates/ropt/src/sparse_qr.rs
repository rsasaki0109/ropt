//! Ordered sparse Givens least squares. No numeric normal matrix or fallback.
use crate::{
    DVector, Error,
    incremental_qr::{IncrementalQr, IncrementalQrOptions, QrRowOrder},
    sparse::{CsrMatrix, finite_vector},
    sparse_direct::{SparseLdlOptions, SparseOrdering, column_ordering},
};

#[derive(Debug, Clone, Copy)]
pub struct SparseQrOptions {
    pub ordering: SparseOrdering,
    /// Relative cutoff on the ordered, unpivoted R diagonal; not an SVD certificate.
    pub relative_rank_tolerance: f64,
    /// Bound on symbolic ordering storage and numeric R entries, not peak memory.
    pub max_entries: usize,
}
impl Default for SparseQrOptions {
    fn default() -> Self {
        Self {
            ordering: SparseOrdering::MinimumDegree,
            relative_rank_tolerance: 1e-12,
            max_entries: 1_000_000,
        }
    }
}
impl SparseQrOptions {
    pub(crate) fn validate(self) -> Result<(), Error> {
        self.symbolic_options().validate()?;
        if !self.relative_rank_tolerance.is_finite()
            || self.relative_rank_tolerance <= 0.
            || self.relative_rank_tolerance >= 1.
        {
            return Err(Error::InvalidOptions);
        }
        Ok(())
    }
    fn symbolic_options(self) -> SparseLdlOptions {
        SparseLdlOptions {
            ordering: self.ordering,
            max_entries: self.max_entries,
            ..Default::default()
        }
    }
}

pub(crate) struct SparseQrWorkspace {
    options: SparseQrOptions,
    pattern: Option<QrRowOrder>,
    order: Vec<usize>,
    factor: IncrementalQr,
}
impl SparseQrWorkspace {
    pub(crate) fn new(options: SparseQrOptions) -> Result<Self, Error> {
        options.validate()?;
        Ok(Self {
            options,
            pattern: None,
            order: Vec::new(),
            factor: IncrementalQr::new(
                0,
                IncrementalQrOptions {
                    relative_rank_tolerance: options.relative_rank_tolerance,
                    max_entries: options.max_entries,
                },
            )?,
        })
    }
    pub(crate) fn permutation(&self) -> Option<&[usize]> {
        self.pattern.as_ref().map(|_| self.order.as_slice())
    }
    pub(crate) fn solve(
        &mut self,
        j: &CsrMatrix,
        r: &DVector<f64>,
        damping: f64,
    ) -> Result<DVector<f64>, Error> {
        crate::stream_profile::measure(4, || {
            let (m, n) = j.shape();
            if m == 0 || n == 0 || r.len() != m {
                return Err(Error::InvalidDimensions);
            }
            finite_vector(r)?;
            if !damping.is_finite() || damping < 0. {
                return Err(Error::InvalidOptions);
            }
            if self.pattern.as_ref().is_none_or(|p| !p.matches(j)) {
                let order = column_ordering(j, &self.options.symbolic_options())?;
                self.order = order;
                self.pattern = Some(QrRowOrder::new(j, &self.order)?);
            }
            self.factor.reset_for_rebuild(n, &self.order)?;
            let qr = &mut self.factor;
            if damping == 0. {
                qr.rebuild_cached_row_order(j, r, self.pattern.as_ref().unwrap())?;
            } else {
                // Keep observation rows first and sort the complete augmented model
                // during rebuilding; appending damping after R would cause extra fill.
                let rows = m.checked_add(n).ok_or(Error::InvalidDimensions)?;
                let mut entries: Vec<_> = j.entries().collect();
                let mut norms = DVector::<f64>::zeros(n);
                for (_, c, v) in j.entries() {
                    norms[c] = norms[c].hypot(v);
                }
                for c in 0..n {
                    let value = damping.sqrt() * norms[c].max(1.);
                    if !value.is_finite() {
                        return Err(Error::NonFinite);
                    }
                    entries.push((m + c, c, value));
                }
                let augmented = CsrMatrix::from_triplets(rows, n, entries)?;
                let mut residual = DVector::zeros(rows);
                residual.rows_mut(0, m).copy_from(r);
                qr.rebuild_in_place(&augmented, &residual)?;
            }
            qr.solve()
        })
    }
}
/// Solve `min_delta ||J delta + r||² + damping * sum(max(||J_col||²,1)*delta_col²)`.
/// The elimination ordering is structural; numeric column pivoting is not provided.
pub fn solve_least_squares(
    j: &CsrMatrix,
    r: &DVector<f64>,
    damping: f64,
    options: &SparseQrOptions,
) -> Result<DVector<f64>, Error> {
    SparseQrWorkspace::new(*options)?.solve(j, r, damping)
}

#[cfg(test)]
mod reuse_tests {
    use super::*;
    #[test]
    fn repeated_numeric_rebuilds_match_fresh_factors_after_overflow_and_shape_changes() {
        let options = SparseQrOptions::default();
        let mut workspace = SparseQrWorkspace::new(options).unwrap();
        for n in [2, 7, 3, 7, 1] {
            let j =
                CsrMatrix::from_triplets(n + 1, n, (0..n).map(|k| (k, k, 2. + k as f64)).collect())
                    .unwrap();
            for damping in [0., 1e-3, 0.] {
                let r = DVector::from_fn(n + 1, |k, _| (k as f64 + 0.7).sin());
                let expected = solve_least_squares(&j, &r, damping, &options).unwrap();
                assert_eq!(workspace.solve(&j, &r, damping).unwrap(), expected);
            }
            // A late constant row overflows after previous rows modified R/RHS.
            let mut r = DVector::zeros(n + 1);
            r[n] = f64::MAX;
            assert_eq!(workspace.solve(&j, &r, 0.), Err(Error::NonFinite));
            let r = DVector::from_element(n + 1, 1.);
            assert_eq!(
                workspace.solve(&j, &r, 0.).unwrap(),
                solve_least_squares(&j, &r, 0., &options).unwrap()
            );
        }
    }
}
