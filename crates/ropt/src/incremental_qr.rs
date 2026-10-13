//! Sparse row insertion into an upper-triangular square-root information factor.
//! Givens updates do not reconstruct a normal matrix or replay previous rows.
use crate::{
    DVector, Error,
    sparse::{CsrMatrix, finite_vector},
};

// Keep the squared sum far from overflow and loss of a nonzero norm to
// underflow. Outside this exponent range use the standard scaled hypot.
// The direct path has ordinary IEEE rounding; it need not match hypot bitwise.
fn givens_norm(a: f64, b: f64) -> f64 {
    const LOW: f64 = f64::from_bits((1023 - 450) << 52);
    const HIGH: f64 = f64::from_bits((1023 + 450) << 52);
    let scale = a.abs().max(b.abs());
    if (LOW..=HIGH).contains(&scale) {
        (a * a + b * b).sqrt()
    } else {
        a.hypot(b)
    }
}

// Both live suffixes have strictly ascending, unique column indices. Equal
// endpoints and a span of len - 1 prove that both contain the same entire
// contiguous range. Gapped ranges still require checking every interior index.
#[inline]
fn same_columns(upper: &[(usize, f64)], lower: &[(usize, f64)]) -> bool {
    if upper.len() != lower.len() {
        return false;
    }
    match upper.len() {
        0 => true,
        1 => upper[0].0 == lower[0].0,
        len => {
            let last = len - 1;
            if upper[0].0 != lower[0].0 || upper[last].0 != lower[last].0 {
                return false;
            }
            upper[last].0 - upper[0].0 == last
                || upper[1..last]
                    .iter()
                    .zip(&lower[1..last])
                    .all(|(a, b)| a.0 == b.0)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IncrementalQrOptions {
    /// Relative diagonal cutoff on unpivoted R; not an SVD rank certificate.
    pub relative_rank_tolerance: f64,
    /// Stored entries in R, including diagonals. Not a peak-memory limit.
    pub max_entries: usize,
}
impl Default for IncrementalQrOptions {
    fn default() -> Self {
        Self {
            relative_rank_tolerance: 1e-12,
            max_entries: 1_000_000,
        }
    }
}
#[derive(Debug, Clone)]
pub struct IncrementalQr {
    // Sorted nonzero entries. A populated row always starts with its diagonal.
    rows: Vec<Vec<(usize, f64)>>,
    rhs: DVector<f64>,
    constant_cost: f64,
    entries: usize,
    options: IncrementalQrOptions,
    column_order: Option<ColumnOrder>,
}
#[derive(Debug, Clone)]
struct ColumnOrder {
    to_original: Vec<usize>,
    to_internal: Vec<usize>,
}
impl IncrementalQr {
    pub fn new(columns: usize, options: IncrementalQrOptions) -> Result<Self, Error> {
        if !options.relative_rank_tolerance.is_finite()
            || options.relative_rank_tolerance <= 0.
            || options.relative_rank_tolerance >= 1.
            || options.max_entries == 0
        {
            return Err(Error::InvalidOptions);
        }
        let mut out = Self {
            rows: Vec::new(),
            rhs: DVector::zeros(0),
            constant_cost: 0.,
            entries: 0,
            options,
            column_order: None,
        };
        out.extend_columns(columns)?;
        Ok(out)
    }
    pub fn dimension(&self) -> usize {
        self.rows.len()
    }
    pub fn stored_entries(&self) -> usize {
        self.entries
    }
    /// Extend with unconstrained trailing columns. Earlier column indices stay fixed.
    pub fn extend_columns(&mut self, columns: usize) -> Result<(), Error> {
        if columns < self.dimension() {
            return Err(Error::InvalidDimensions);
        }
        if columns > self.options.max_entries {
            return Err(Error::SparseFillLimit {
                limit: self.options.max_entries,
            });
        }
        if let Some(order) = &mut self.column_order {
            order.to_original.extend(self.rows.len()..columns);
            order.to_internal.extend(self.rows.len()..columns);
        }
        self.rows.resize_with(columns, Vec::new);
        self.rhs.resize_vertically_mut(columns, 0.);
        Ok(())
    }
    /// Add min ||J*d+residual||² rows. On failure the factor is unchanged.
    pub fn append(&mut self, j: &CsrMatrix, residual: &DVector<f64>) -> Result<(), Error> {
        let mut candidate = self.clone();
        candidate.append_in_place(j, residual)?;
        *self = candidate;
        Ok(())
    }
    pub(crate) fn append_in_place(
        &mut self,
        j: &CsrMatrix,
        residual: &DVector<f64>,
    ) -> Result<(), Error> {
        self.append_rows_in_place(j, residual, 0..j.shape().0)
    }
    pub(crate) fn rebuild_in_place(
        &mut self,
        j: &CsrMatrix,
        residual: &DVector<f64>,
    ) -> Result<(), Error> {
        // A complete new model can group rows by their first pivot, avoiding
        // late rotations through already populated downstream rows. Preserve
        // original order within each group. Incremental append stays in input
        // order; only a fresh factor may use this rebuild path.
        if self.entries != 0 || self.constant_cost != 0. || self.rhs.iter().any(|&v| v != 0.) {
            return Err(Error::InvalidOptions);
        }
        let (offsets, indices) = j.pattern();
        let mut rows = (0..j.shape().0).collect::<Vec<_>>();
        rows.sort_by_key(|&r| {
            if offsets[r] == offsets[r + 1] {
                j.shape().1
            } else if let Some(order) = &self.column_order {
                indices[offsets[r]..offsets[r + 1]]
                    .iter()
                    .map(|&c| order.to_internal[c])
                    .min()
                    .unwrap()
            } else {
                indices[offsets[r]]
            }
        });
        self.append_rows_in_place(j, residual, rows.into_iter())
    }
    /// Internal column ordering only. Input Jacobians, returned steps, supplied
    /// cost vectors and future appended columns retain original coordinates.
    pub(crate) fn set_column_order(&mut self, permutation: Vec<usize>) -> Result<(), Error> {
        if self.entries != 0 || self.constant_cost != 0. || self.rhs.iter().any(|&v| v != 0.) {
            return Err(Error::InvalidOptions);
        }
        if permutation.len() != self.dimension() {
            return Err(Error::InvalidDimensions);
        }
        let mut inverse = vec![usize::MAX; permutation.len()];
        for (internal, &original) in permutation.iter().enumerate() {
            if original >= inverse.len() || inverse[original] != usize::MAX {
                return Err(Error::InvalidDimensions);
            }
            inverse[original] = internal;
        }
        self.column_order = Some(ColumnOrder {
            to_original: permutation,
            to_internal: inverse,
        });
        Ok(())
    }
    pub(crate) fn column_permutation(&self) -> Option<&[usize]> {
        self.column_order
            .as_ref()
            .map(|order| order.to_original.as_slice())
    }
    fn append_rows_in_place(
        &mut self,
        j: &CsrMatrix,
        residual: &DVector<f64>,
        rows: impl Iterator<Item = usize>,
    ) -> Result<(), Error> {
        let (m, n) = j.shape();
        if n != self.dimension() || m != residual.len() {
            return Err(Error::InvalidDimensions);
        }
        finite_vector(residual)?;
        // Reuse merge buffers across all rotations; no per-column tree nodes
        // or temporary set union. Keep ascending traversal and exact-zero policy.
        let mut upper_row = Vec::new();
        let mut lower_row = Vec::new();
        let mut row = Vec::new();
        let max_entries = self.options.max_entries;
        for r in rows {
            let residual = residual[r];
            row.clear();
            row.extend(j.row(r)?);
            if let Some(order) = &self.column_order {
                for (column, _) in &mut row {
                    *column = order.to_internal[*column];
                }
                row.sort_unstable_by_key(|&(column, _)| column);
            }
            let mut b = -residual;
            let mut row_start = 0;
            while let Some(&(k, below)) = row.get(row_start) {
                if below == 0. {
                    row_start += 1;
                    continue;
                }
                let old = &mut self.rows[k];
                let diagonal = old.first().map_or(0., |&(_, value)| value);
                let norm = givens_norm(diagonal, below);
                if !norm.is_finite() || norm == 0. {
                    return Err(Error::NonFinite);
                }
                let (c, s) = (diagonal / norm, below / norm);
                upper_row.clear();
                lower_row.clear();
                let has_diagonal = !old.is_empty();
                let (mut i, mut j, mut write) = (
                    usize::from(has_diagonal),
                    row_start + 1,
                    usize::from(has_diagonal),
                );
                let mut entries = self.entries;
                if has_diagonal {
                    old[0].1 = norm;
                }
                // Equal live suffixes need no union merge or lower-row copy.
                // Prove contiguous ranges from their endpoints, otherwise
                // compare every column; equal lengths alone are insufficient.
                let aligned = has_diagonal && same_columns(&old[1..], &row[row_start + 1..]);
                if aligned {
                    // Independent column rotations share only reductions.
                    // Delay the finite check until this suffix is complete; an
                    // invalid private candidate is discarded before compaction
                    // or any further rotation/RHS update can expose its values.
                    let (mut removed, mut lower_zero, mut finite) = (0_usize, false, true);
                    for (upper, lower) in old[1..].iter_mut().zip(&mut row[row_start + 1..]) {
                        let a = c * upper.1 + s * lower.1;
                        let v = -s * upper.1 + c * lower.1;
                        finite &= a.is_finite() & v.is_finite();
                        removed += usize::from(a == 0.);
                        lower_zero |= v == 0.;
                        upper.1 = a;
                        lower.1 = v;
                    }
                    if !finite {
                        return Err(Error::NonFinite);
                    }
                    entries -= removed;
                    if removed != 0 {
                        old.retain(|&(_, value)| value != 0.);
                    }
                    // Consume the diagonal by advancing a cursor rather than
                    // shifting the row. Compact only exact-zero cancellations.
                    row_start += 1;
                    if lower_zero {
                        let mut write = row_start;
                        for read in row_start..row.len() {
                            if row[read].1 != 0. {
                                row[write] = row[read];
                                write += 1;
                            }
                        }
                        row.truncate(write);
                    }
                    self.entries = entries;
                } else {
                    // Compact existing columns in place until the first new interior
                    // column. Writes never reach unread entries. Preserve ascending
                    // arithmetic, including explicit zero operands and cancellations.
                    while i < old.len() && (j == row.len() || old[i].0 <= row[j].0) {
                        let (col, upper) = old[i];
                        let lower = if j < row.len() && row[j].0 == col {
                            let value = row[j].1;
                            j += 1;
                            value
                        } else {
                            0.
                        };
                        let a = c * upper + s * lower;
                        let v = -s * upper + c * lower;
                        if !a.is_finite() || !v.is_finite() {
                            return Err(Error::NonFinite);
                        }
                        if a == 0. {
                            entries -= 1;
                        } else {
                            old[write] = (col, a);
                            write += 1;
                        }
                        if v != 0. {
                            lower_row.push((col, v));
                        }
                        i += 1;
                    }
                    let in_place = has_diagonal && i == old.len();
                    if in_place {
                        old.truncate(write);
                        // New trailing columns cannot overwrite unread storage.
                        for &(col, lower) in &row[j..] {
                            let a = c * 0. + s * lower;
                            let v = -s * 0. + c * lower;
                            if !a.is_finite() || !v.is_finite() {
                                return Err(Error::NonFinite);
                            }
                            if a != 0. {
                                if entries >= max_entries {
                                    return Err(Error::SparseFillLimit { limit: max_entries });
                                }
                                entries += 1;
                                old.push((col, a));
                            }
                            if v != 0. {
                                lower_row.push((col, v));
                            }
                        }
                    } else {
                        if has_diagonal {
                            upper_row.extend_from_slice(&old[..write]);
                        } else {
                            upper_row.push((k, norm));
                        }
                        // A new interior column requires merging the remaining suffix.
                        // Already rotated/compacted columns are copied, never replayed.
                        let mut rotate_entry =
                            |col: usize, upper: f64, lower: f64| -> Result<(), Error> {
                                let a = c * upper + s * lower;
                                let v = -s * upper + c * lower;
                                if !a.is_finite() || !v.is_finite() {
                                    return Err(Error::NonFinite);
                                }
                                // Match the old ascending map-update entry checks, including
                                // transient fill before later cancellations and the diagonal.
                                if a == 0. && upper != 0. {
                                    entries -= 1;
                                }
                                if a != 0. && upper == 0. {
                                    if entries >= max_entries {
                                        return Err(Error::SparseFillLimit { limit: max_entries });
                                    }
                                    entries += 1;
                                }
                                if a != 0. {
                                    upper_row.push((col, a));
                                }
                                if v != 0. {
                                    lower_row.push((col, v));
                                }
                                Ok(())
                            };
                        while i < old.len() && j < row.len() {
                            let (upper_col, upper) = old[i];
                            let (lower_col, lower) = row[j];
                            match upper_col.cmp(&lower_col) {
                                std::cmp::Ordering::Less => {
                                    rotate_entry(upper_col, upper, 0.)?;
                                    i += 1;
                                }
                                std::cmp::Ordering::Greater => {
                                    rotate_entry(lower_col, 0., lower)?;
                                    j += 1;
                                }
                                std::cmp::Ordering::Equal => {
                                    rotate_entry(upper_col, upper, lower)?;
                                    i += 1;
                                    j += 1;
                                }
                            }
                        }
                        // Tails remain in the same ascending union order, using the
                        // same arithmetic and transient entry checks as paired columns.
                        for &(col, upper) in &old[i..] {
                            rotate_entry(col, upper, 0.)?;
                        }
                        for &(col, lower) in &row[j..] {
                            rotate_entry(col, 0., lower)?;
                        }
                    }
                    if diagonal == 0. {
                        if entries >= self.options.max_entries {
                            return Err(Error::SparseFillLimit {
                                limit: self.options.max_entries,
                            });
                        }
                        entries += 1;
                    }
                    self.entries = entries;
                    if !in_place {
                        std::mem::swap(old, &mut upper_row);
                    }
                    std::mem::swap(&mut row, &mut lower_row);
                    row_start = 0;
                }
                let old = self.rhs[k];
                self.rhs[k] = c * old + s * b;
                b = -s * old + c * b;
                if !self.rhs[k].is_finite() || !b.is_finite() {
                    return Err(Error::NonFinite);
                }
            }
            self.constant_cost += 0.5 * b * b;
            if !self.constant_cost.is_finite() {
                return Err(Error::NonFinite);
            }
        }
        Ok(())
    }
    /// Solve the full-column-rank cached model. No damping or pseudoinverse.
    pub fn solve(&self) -> Result<DVector<f64>, Error> {
        let largest = (0..self.dimension())
            .map(|k| self.rows[k].first().map_or(0., |&(_, a)| a.abs()))
            .fold(0., f64::max);
        let cutoff = largest * self.options.relative_rank_tolerance;
        let rank = (0..self.dimension())
            .filter(|&k| self.rows[k].first().is_some_and(|&(_, a)| a.abs() > cutoff))
            .count();
        if rank < self.dimension() {
            return Err(Error::RankDeficient {
                rank,
                parameters: self.dimension(),
            });
        }
        if let Some(order) = &self.column_order {
            let mut x = DVector::zeros(self.dimension());
            for k in (0..self.dimension()).rev() {
                let mut value = self.rhs[k];
                for &(col, coefficient) in self.rows[k].iter().skip(1) {
                    value -= coefficient * x[order.to_original[col]];
                }
                x[order.to_original[k]] = value / self.rows[k][0].1;
            }
            finite_vector(&x)?;
            return Ok(x);
        }
        let mut x = self.rhs.clone();
        for k in (0..self.dimension()).rev() {
            for &(col, value) in self.rows[k].iter().skip(1) {
                x[k] -= value * x[col];
            }
            x[k] /= self.rows[k][0].1;
        }
        finite_vector(&x)?;
        Ok(x)
    }
    /// Solve R^T R * step = -gradient without forming the normal matrix.
    /// Used only for opt-in near-solution refinement with a frozen QR factor.
    pub(crate) fn solve_gradient(&self, gradient: &DVector<f64>) -> Result<DVector<f64>, Error> {
        if gradient.len() != self.dimension() {
            return Err(Error::InvalidDimensions);
        }
        finite_vector(gradient)?;
        let largest = self
            .rows
            .iter()
            .map(|row| row.first().map_or(0., |&(_, v)| v.abs()))
            .fold(0.0_f64, f64::max);
        let cutoff = largest * self.options.relative_rank_tolerance;
        let rank = self
            .rows
            .iter()
            .filter(|row| row.first().is_some_and(|&(_, v)| v.abs() > cutoff))
            .count();
        if rank < self.dimension() {
            return Err(Error::RankDeficient {
                rank,
                parameters: self.dimension(),
            });
        }
        let original = |col: usize| {
            self.column_order
                .as_ref()
                .map_or(col, |order| order.to_original[col])
        };
        let mut internal = DVector::from_iterator(
            self.dimension(),
            (0..self.dimension()).map(|k| -gradient[original(k)]),
        );
        for k in 0..self.dimension() {
            internal[k] /= self.rows[k][0].1;
            let value = internal[k];
            for &(col, coefficient) in self.rows[k].iter().skip(1) {
                internal[col] -= coefficient * value;
            }
        }
        for k in (0..self.dimension()).rev() {
            for &(col, coefficient) in self.rows[k].iter().skip(1) {
                internal[k] -= coefficient * internal[col];
            }
            internal[k] /= self.rows[k][0].1;
        }
        finite_vector(&internal)?;
        if self.column_order.is_none() {
            return Ok(internal);
        }
        let mut result = DVector::zeros(self.dimension());
        for k in 0..self.dimension() {
            result[original(k)] = internal[k];
        }
        Ok(result)
    }
    /// Squared residual cost of the cached linear model at a supplied delta.
    pub fn cost(&self, x: &DVector<f64>) -> Result<f64, Error> {
        if x.len() != self.dimension() {
            return Err(Error::InvalidDimensions);
        }
        finite_vector(x)?;
        let mut cost = self.constant_cost;
        // The finite R entries contribute only signed zeros at the exact origin.
        // Keep row-order accumulation and overflow checks, without traversing R.
        // Test components exactly: a nonzero vector's norm can underflow to zero.
        if x.iter().all(|&value| value == 0.) {
            for &rhs in self.rhs.iter() {
                let residual = -rhs;
                cost += 0.5 * residual * residual;
            }
            return if cost.is_finite() {
                Ok(cost)
            } else {
                Err(Error::NonFinite)
            };
        }
        for (k, row) in self.rows.iter().enumerate() {
            let residual = row.iter().fold(-self.rhs[k], |r, &(col, a)| {
                let original = self
                    .column_order
                    .as_ref()
                    .map_or(col, |order| order.to_original[col]);
                r + a * x[original]
            });
            cost += 0.5 * residual * residual;
        }
        if !cost.is_finite() {
            return Err(Error::NonFinite);
        }
        Ok(cost)
    }
}

#[cfg(test)]
mod norm_tests {
    use super::givens_norm;

    fn power(exponent: i32) -> f64 {
        if exponent < -1022 {
            f64::from_bits(1_u64 << (exponent + 1074))
        } else {
            f64::from_bits(((exponent + 1023) as u64) << 52)
        }
    }

    #[test]
    fn squared_norm_matches_hypot_within_two_ulps_across_exponents_and_signs() {
        let mut state = 0x54d3c71a8912f0b9_u64;
        for exponent in -1074..=1023 {
            let scale = power(exponent);
            if scale == 0. || !scale.is_finite() {
                continue;
            }
            for _ in 0..8 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let fraction = (state >> 11) as f64 / ((1_u64 << 53) as f64);
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let ratio = (state >> 11) as f64 / ((1_u64 << 53) as f64);
                for sign in [-1., 1.] {
                    let a = sign * scale * (0.5 + fraction * 0.5);
                    let b = scale * ratio;
                    let expected = a.hypot(b);
                    let actual = givens_norm(a, b);
                    assert!(
                        actual.to_bits().abs_diff(expected.to_bits()) <= 2,
                        "exponent={exponent} a={a:e} b={b:e} actual={actual:e} expected={expected:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn extreme_scales_boundaries_zeros_and_nonfinite_inputs_keep_hypot_behavior() {
        for exponent in [-1074, -1022, -451, -450, 450, 451, 1023] {
            let scale = power(exponent);
            for a in [
                scale,
                f64::from_bits(scale.to_bits().saturating_sub(1)),
                f64::from_bits(scale.to_bits() + 1),
            ] {
                for b in [0., -0., f64::from_bits(1), -a, a * 0.75, a] {
                    let expected = a.hypot(b);
                    let actual = givens_norm(a, b);
                    assert_eq!(actual.is_finite(), expected.is_finite());
                    assert_eq!(actual == 0., expected == 0.);
                    if actual.is_finite() {
                        assert!(actual.to_bits().abs_diff(expected.to_bits()) <= 2);
                    }
                }
            }
        }
        for (a, b) in [
            (0., -0.),
            (f64::MAX, f64::MAX),
            (f64::INFINITY, 1.),
            (f64::INFINITY, f64::NAN),
            (f64::NAN, 1.),
        ] {
            let actual = givens_norm(a, b);
            let expected = a.hypot(b);
            assert_eq!(actual.is_nan(), expected.is_nan());
            if !actual.is_nan() {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }
}

#[cfg(test)]
mod column_tests {
    use super::same_columns;

    #[test]
    fn sorted_subset_masks_exhaustively_check_gaps_widths_and_max_indices() {
        for base in [0, usize::MAX - 6] {
            for a in 0_u8..128 {
                let upper = (0..7)
                    .filter(|&col| a & (1 << col) != 0)
                    .map(|col| (base + col, 1.))
                    .collect::<Vec<_>>();
                for b in 0_u8..128 {
                    let lower = (0..7)
                        .filter(|&col| b & (1 << col) != 0)
                        .map(|col| (base + col, -3.))
                        .collect::<Vec<_>>();
                    // Equality of subset masks is an independent structural
                    // oracle; coefficients have no role in column matching.
                    assert_eq!(same_columns(&upper, &lower), a == b);
                }
            }
        }
    }

    #[test]
    fn wide_contiguous_and_gapped_suffixes_check_endpoints_and_interior() {
        for len in [63, 64, 65, 127, 128, 257] {
            let contiguous = (0..len).map(|col| (col, 1.)).collect::<Vec<_>>();
            assert!(same_columns(&contiguous, &contiguous));
            let mut changed_end = contiguous.clone();
            changed_end[len - 1].0 += 1;
            assert!(!same_columns(&contiguous, &changed_end));
            let gapped = (0..len).map(|col| (2 * col, 1.)).collect::<Vec<_>>();
            assert!(same_columns(&gapped, &gapped));
            let mut changed_middle = gapped.clone();
            changed_middle[len / 2].0 += 1;
            assert!(!same_columns(&gapped, &changed_middle));
            assert!(!same_columns(&gapped, &gapped[..len - 1]));
        }
    }
}

#[cfg(test)]
mod rebuild_tests {
    use super::*;
    use crate::DMatrix;

    fn csr(j: &DMatrix<f64>) -> CsrMatrix {
        CsrMatrix::from_triplets(
            j.nrows(),
            j.ncols(),
            (0..j.nrows())
                .flat_map(|r| {
                    (0..j.ncols())
                        .filter_map(move |c| (j[(r, c)] != 0.).then_some((r, c, j[(r, c)])))
                })
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn arbitrary_column_orders_keep_external_coordinates_through_append_and_extension() {
        for n in [4, 7, 16] {
            let dense = DMatrix::from_fn(3 * n, n, |r, c| {
                if r == c {
                    3.
                } else if (r + 2 * c) % 5 == 0 {
                    ((r + c + 1) as f64).sin()
                } else {
                    0.
                }
            });
            let residual = DVector::from_fn(3 * n, |r, _| ((r + 1) as f64).cos());
            let mut qr = IncrementalQr::new(n, Default::default()).unwrap();
            let mut order = (0..n).rev().collect::<Vec<_>>();
            order.rotate_left(1);
            qr.set_column_order(order.clone()).unwrap();
            qr.rebuild_in_place(&csr(&dense), &residual).unwrap();
            let expected = reference(&dense, &residual);
            let gradient = dense.transpose() * &residual;
            assert!((&qr.solve_gradient(&gradient).unwrap() - &expected).norm() < 1e-10);
            assert!((&qr.solve().unwrap() - &expected).norm() < 1e-11);
            for x in [
                expected,
                DVector::zeros(n),
                DVector::from_fn(n, |i, _| 0.1 * (i + 1) as f64),
            ] {
                assert!(
                    (qr.cost(&x).unwrap() - 0.5 * (&dense * &x + &residual).norm_squared()).abs()
                        < 1e-10
                );
            }
            qr.extend_columns(n + 1).unwrap();
            assert_eq!(qr.column_permutation().unwrap(), [order, vec![n]].concat());
            assert!(matches!(qr.solve(), Err(Error::RankDeficient { .. })));
            let extra = CsrMatrix::from_triplets(1, n + 1, vec![(0, 0, 0.5), (0, n, 2.)]).unwrap();
            qr.append(&extra, &DVector::from_element(1, -0.7)).unwrap();
            let combined = DMatrix::from_fn(3 * n + 1, n + 1, |r, c| {
                if r < 3 * n {
                    if c < n { dense[(r, c)] } else { 0. }
                } else if c == 0 {
                    0.5
                } else if c == n {
                    2.
                } else {
                    0.
                }
            });
            let r = DVector::from_fn(3 * n + 1, |i, _| if i < 3 * n { residual[i] } else { -0.7 });
            let expected = reference(&combined, &r);
            let actual = qr.solve().unwrap();
            assert!((&actual - &expected).norm() < 1e-11);
            assert!(
                (qr.cost(&actual).unwrap() - 0.5 * (&combined * actual + r).norm_squared()).abs()
                    < 1e-10
            );
            let before = qr.solve().unwrap();
            assert_eq!(
                qr.append(&extra, &DVector::from_element(1, f64::INFINITY))
                    .unwrap_err(),
                Error::NonFinite
            );
            assert_eq!(qr.solve().unwrap(), before);
        }
    }

    #[test]
    fn invalid_column_orders_leave_a_fresh_order_unchanged() {
        let mut qr = IncrementalQr::new(3, Default::default()).unwrap();
        qr.set_column_order(vec![2, 0, 1]).unwrap();
        for bad in [vec![], vec![0, 1], vec![0, 1, 3], vec![0, 1, 1]] {
            assert_eq!(
                qr.set_column_order(bad).unwrap_err(),
                Error::InvalidDimensions
            );
            assert_eq!(qr.column_permutation(), Some([2, 0, 1].as_slice()));
        }
        let j = CsrMatrix::from_triplets(3, 3, vec![(0, 0, 1.), (1, 1, 1.), (2, 2, 1.)]).unwrap();
        qr.rebuild_in_place(&j, &DVector::from_vec(vec![-1., -2., -3.]))
            .unwrap();
        assert_eq!(qr.solve().unwrap(), DVector::from_vec(vec![1., 2., 3.]));
        assert_eq!(
            qr.set_column_order(vec![0, 1, 2]).unwrap_err(),
            Error::InvalidOptions
        );
    }

    fn reference(j: &DMatrix<f64>, residual: &DVector<f64>) -> DVector<f64> {
        let mut x = j
            .clone()
            .try_svd(true, true, f64::EPSILON * f64::EPSILON, 1000)
            .unwrap()
            .solve(&(-residual), 1e-12)
            .unwrap();
        // These test matrices contain a strong diagonal and are well conditioned.
        // SVD's disconnected-support vectors can leave a ~1e-10 original
        // gradient. Refine only this oracle with an independent dense SPD solve;
        // production rebuilds still use orthogonal Givens rotations exclusively.
        let normal = (j.transpose() * j).cholesky().unwrap();
        for _ in 0..3 {
            let gradient = j.transpose() * (j * &x + residual);
            if gradient.norm() < 1e-12 {
                break;
            }
            x -= normal.solve(&gradient);
        }
        assert!((j.transpose() * (j * &x + residual)).norm() < 1e-12);
        x
    }

    #[test]
    fn grouped_rebuild_and_later_extension_match_refined_dense_reference_and_original_energy() {
        for n in [4, 7, 16] {
            let m = 3 * n + 3;
            let dense = DMatrix::from_fn(m, n, |r, c| {
                if r == 0 || r == m - 1 {
                    0.
                } else if r <= n {
                    if c == n - r { 3. } else { 0. }
                } else if (r + 2 * c) % 5 == 0 {
                    ((r + c + 1) as f64).sin()
                } else {
                    0.
                }
            });
            let residual = DVector::from_fn(m, |r, _| ((r + 1) as f64).cos());
            let expected = reference(&dense, &residual);
            for scale in [1e-100, 1., 1e100] {
                let mut qr = IncrementalQr::new(n, Default::default()).unwrap();
                qr.rebuild_in_place(&csr(&(&dense * scale)), &(&residual * scale))
                    .unwrap();
                let solved = qr.solve().unwrap();
                let qr_gradient = dense.transpose() * (&dense * &solved + &residual);
                let svd_gradient = dense.transpose() * (&dense * &expected + &residual);
                assert!(svd_gradient.norm() < 1e-12);
                assert!(qr_gradient.norm() < 1e-12);
                assert!(
                    (&solved - &expected).norm() < 1e-11,
                    "n={n}, scale={scale}, actual={solved:?}, svd={expected:?}, gap={}, qr_gradient={}, svd_gradient={}",
                    (&solved - &expected).norm(),
                    qr_gradient.norm(),
                    svd_gradient.norm()
                );
                qr.extend_columns(n + 1).unwrap();
                assert!(matches!(qr.solve(), Err(Error::RankDeficient { .. })));
                qr.append(
                    &CsrMatrix::from_triplets(
                        1,
                        n + 1,
                        vec![(0, 0, 0.5 * scale), (0, n, 2. * scale)],
                    )
                    .unwrap(),
                    &DVector::from_element(1, -0.7 * scale),
                )
                .unwrap();
                let combined = DMatrix::from_fn(m + 1, n + 1, |r, c| {
                    if r < m {
                        if c < n { dense[(r, c)] } else { 0. }
                    } else if c == 0 {
                        0.5
                    } else if c == n {
                        2.
                    } else {
                        0.
                    }
                });
                let r = DVector::from_fn(m + 1, |i, _| if i < m { residual[i] } else { -0.7 });
                let expected = reference(&combined, &r);
                let solved = qr.solve().unwrap();
                let qr_gradient = combined.transpose() * (&combined * &solved + &r);
                let svd_gradient = combined.transpose() * (&combined * &expected + &r);
                assert!(svd_gradient.norm() < 1e-12);
                assert!(qr_gradient.norm() < 1e-12);
                assert!(
                    (&solved - &expected).norm() < 1e-11,
                    "extended n={n}, scale={scale}, gap={}, qr_gradient={}, svd_gradient={}",
                    (&solved - &expected).norm(),
                    qr_gradient.norm(),
                    svd_gradient.norm()
                );
                for x in [
                    solved,
                    DVector::zeros(n + 1),
                    DVector::from_element(n + 1, 0.3),
                ] {
                    let original = 0.5 * (&combined * &x + &r).norm_squared();
                    let cached = qr.cost(&x).unwrap() / scale / scale;
                    assert!((original - cached).abs() < 1e-11 * original.max(1.));
                }
            }
        }
    }

    #[test]
    fn rebuild_rejects_existing_models_bad_inputs_rank_loss_and_fill_overflow() {
        let j =
            CsrMatrix::from_triplets(2, 2, vec![(0, 0, 1.), (0, 1, 2.), (1, 0, 2.), (1, 1, 4.)])
                .unwrap();
        let mut qr = IncrementalQr::new(2, Default::default()).unwrap();
        assert_eq!(
            qr.rebuild_in_place(&j, &DVector::zeros(1)).unwrap_err(),
            Error::InvalidDimensions
        );
        assert_eq!(
            qr.rebuild_in_place(&j, &DVector::from_element(2, f64::INFINITY))
                .unwrap_err(),
            Error::NonFinite
        );
        qr.rebuild_in_place(&j, &DVector::zeros(2)).unwrap();
        assert!(matches!(qr.solve(), Err(Error::RankDeficient { .. })));
        assert_eq!(
            qr.rebuild_in_place(&j, &DVector::zeros(2)).unwrap_err(),
            Error::InvalidOptions
        );
        let dense = DMatrix::from_row_slice(3, 3, &[1., 1., 1., 1., 2., 3., 1., 3., 2.]);
        let mut limited = IncrementalQr::new(
            3,
            IncrementalQrOptions {
                max_entries: 3,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            limited
                .rebuild_in_place(&csr(&dense), &DVector::zeros(3))
                .unwrap_err(),
            Error::SparseFillLimit { limit: 3 }
        );
        let mut constants = IncrementalQr::new(0, Default::default()).unwrap();
        let empty = CsrMatrix::from_triplets(1, 0, vec![]).unwrap();
        constants
            .append(&empty, &DVector::from_element(1, 2.))
            .unwrap();
        assert_eq!(
            constants
                .rebuild_in_place(&empty, &DVector::zeros(1))
                .unwrap_err(),
            Error::InvalidOptions
        );
    }

    #[test]
    fn rebuild_keeps_exact_extreme_power_of_two_models_finite() {
        for exponent in [-800, 800] {
            let scale = 2.0_f64.powi(exponent);
            let dense = DMatrix::from_fn(5, 4, |r, c| {
                if r < 4 && c == 3 - r {
                    if r % 2 == 0 { scale } else { -scale }
                } else {
                    0.
                }
            });
            let expected = DVector::from_vec(vec![1., 2., 3., 4.]);
            let r = -&dense * &expected;
            let mut qr = IncrementalQr::new(4, Default::default()).unwrap();
            qr.rebuild_in_place(&csr(&dense), &r).unwrap();
            assert_eq!(qr.solve().unwrap(), expected);
            assert_eq!(qr.cost(&expected).unwrap(), 0.);
        }
    }
}
