//! Scalar-frontal Gaussian Bayes forest with selective square-root re-elimination.
//!
//! Existing conditional rows on the touched-to-root paths and new rows are
//! re-eliminated by sparse QR. Unaffected conditionals remain shared. This is a
//! fixed-order Gaussian backend, not selective nonlinear relinearization.
mod editable;
use crate::{
    DVector, Error,
    incremental_qr::{IncrementalQr, IncrementalQrOptions},
    sparse::{CsrMatrix, finite_vector},
};
pub use editable::{EditableBayesTree, EditableBayesTreeUpdate, GaussianFactor, GaussianFactorId};
use std::{collections::BTreeSet, sync::Arc};

#[derive(Debug)]
struct Conditional {
    // Internal elimination coordinates; starts at its frontal diagonal if nonempty.
    row: Vec<(usize, f64)>,
    rhs: f64,
}

/// Work in the last installed Gaussian update. Counts are scalar frontals,
/// not variable blocks, nonlinear callbacks, rotations or memory measurements.
/// `input_rows` counts selected old conditional rows plus new measurement rows;
/// when constructing from an existing QR snapshot it counts its triangular rows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BayesTreeUpdate {
    pub reeliminated_cliques: usize,
    pub reused_cliques: usize,
    pub input_rows: usize,
}

#[derive(Debug, Clone)]
pub struct GaussianBayesTree {
    conditionals: Vec<Arc<Conditional>>,
    // Filled symbolic separators ensure the running-intersection property even
    // when numeric cancellation removed an entry from R.
    separators: Vec<Arc<[usize]>>,
    parents: Vec<Option<usize>>,
    to_original: Vec<usize>,
    to_internal: Vec<usize>,
    options: IncrementalQrOptions,
    entries: usize,
    symbolic_entries: usize,
    constant_cost: f64,
    last_update: BayesTreeUpdate,
}
impl GaussianBayesTree {
    pub fn new(columns: usize, options: IncrementalQrOptions) -> Result<Self, Error> {
        Self::from_qr(IncrementalQr::new(columns, options)?)
    }
    /// Consume a factor, preserving its internal ordering and original coordinates.
    /// Like IncrementalQr, incomplete models are allowed; solve checks rank.
    pub fn from_qr(qr: IncrementalQr) -> Result<Self, Error> {
        let factor = qr.into_factor();
        let n = factor.rows.len();
        let permutation = factor.permutation.unwrap_or_else(|| (0..n).collect());
        let mut inverse = vec![0; n];
        for (i, &original) in permutation.iter().enumerate() {
            inverse[original] = i;
        }
        let entries = factor.rows.iter().map(Vec::len).sum();
        let mut tree = Self {
            conditionals: factor
                .rows
                .into_iter()
                .zip(factor.rhs.iter())
                .map(|(row, &rhs)| Arc::new(Conditional { row, rhs }))
                .collect(),
            separators: Vec::new(),
            parents: Vec::new(),
            to_original: permutation,
            to_internal: inverse,
            options: factor.options,
            entries,
            symbolic_entries: 0,
            constant_cost: factor.constant_cost,
            last_update: BayesTreeUpdate {
                reeliminated_cliques: n,
                reused_cliques: 0,
                input_rows: n,
            },
        };
        tree.rebuild_topology()?;
        Ok(tree)
    }
    pub fn dimension(&self) -> usize {
        self.conditionals.len()
    }
    pub fn stored_entries(&self) -> usize {
        self.entries
    }
    /// Scalar frontals plus symbolic separator entries, including cancelled fill.
    /// The same max_entries limit applies separately to this and numeric R.
    pub fn symbolic_entries(&self) -> usize {
        self.symbolic_entries
    }
    pub fn last_update(&self) -> BayesTreeUpdate {
        self.last_update
    }
    /// Internal scalar clique index, for a supplied original tangent column.
    pub fn clique_for_column(&self, column: usize) -> Result<usize, Error> {
        self.to_internal
            .get(column)
            .copied()
            .ok_or(Error::InvalidDimensions)
    }
    pub fn frontal_column(&self, clique: usize) -> Result<usize, Error> {
        self.to_original
            .get(clique)
            .copied()
            .ok_or(Error::InvalidDimensions)
    }
    pub fn parent(&self, clique: usize) -> Result<Option<usize>, Error> {
        self.parents
            .get(clique)
            .copied()
            .ok_or(Error::InvalidDimensions)
    }
    /// Internal clique indices, sorted in elimination order. Includes symbolic
    /// fill entries whose numeric coefficient may be zero.
    pub fn separator(&self, clique: usize) -> Result<&[usize], Error> {
        self.separators
            .get(clique)
            .map(AsRef::as_ref)
            .ok_or(Error::InvalidDimensions)
    }
    pub(crate) fn column_permutation(&self) -> &[usize] {
        &self.to_original
    }
    pub fn extend_columns(&mut self, columns: usize) -> Result<(), Error> {
        if columns < self.dimension() {
            return Err(Error::InvalidDimensions);
        }
        if columns > self.options.max_entries {
            return Err(Error::SparseFillLimit {
                limit: self.options.max_entries,
            });
        }
        let old = self.dimension();
        let symbolic_entries = self
            .symbolic_entries
            .checked_add(columns - old)
            .ok_or(Error::InvalidDimensions)?;
        if symbolic_entries > self.options.max_entries {
            return Err(Error::SparseFillLimit {
                limit: self.options.max_entries,
            });
        }
        for k in old..columns {
            self.conditionals.push(Arc::new(Conditional {
                row: Vec::new(),
                rhs: 0.,
            }));
            self.separators.push(Arc::from([]));
            self.parents.push(None);
            self.to_original.push(k);
            self.to_internal.push(k);
        }
        self.symbolic_entries = symbolic_entries;
        Ok(())
    }
    /// Add ||J*x+residual||² to the same frozen chart. On any error, all owned
    /// tree state is unchanged. Observability is checked separately by solve.
    pub fn append(
        &mut self,
        j: &CsrMatrix,
        residual: &DVector<f64>,
    ) -> Result<BayesTreeUpdate, Error> {
        let mut candidate = self.clone();
        candidate.append_in_place(j, residual)?;
        let report = candidate.last_update;
        *self = candidate;
        Ok(report)
    }
    pub(crate) fn append_in_place(
        &mut self,
        j: &CsrMatrix,
        residual: &DVector<f64>,
    ) -> Result<(), Error> {
        let (m, n) = j.shape();
        if n != self.dimension() || m != residual.len() {
            return Err(Error::InvalidDimensions);
        }
        finite_vector(residual)?;
        let mut affected = BTreeSet::new();
        for (_, column, _) in j.entries() {
            let mut current = Some(self.to_internal[column]);
            while let Some(k) = current {
                if !affected.insert(k) {
                    break;
                }
                current = self.parents[k];
            }
        }
        let selected = affected.into_iter().collect::<Vec<_>>();
        let mut local_index = vec![usize::MAX; n];
        for (local, &global) in selected.iter().enumerate() {
            local_index[global] = local;
        }
        let rows = selected
            .len()
            .checked_add(m)
            .ok_or(Error::InvalidDimensions)?;
        let mut triplets = Vec::new();
        let mut r = DVector::zeros(rows);
        let mut removed_entries = 0;
        for (local, &global) in selected.iter().enumerate() {
            let conditional = &self.conditionals[global];
            removed_entries += conditional.row.len();
            r[local] = -conditional.rhs;
            for &(column, value) in &conditional.row {
                if local_index[column] == usize::MAX {
                    return Err(Error::InvalidDimensions);
                }
                triplets.push((local, local_index[column], value));
            }
        }
        for (row, column, value) in j.entries() {
            triplets.push((
                selected.len() + row,
                local_index[self.to_internal[column]],
                value,
            ));
        }
        r.rows_mut(selected.len(), m).copy_from(residual);
        let local_j = CsrMatrix::from_triplets(rows, selected.len(), triplets)?;
        let mut qr = IncrementalQr::new(selected.len(), self.options)?;
        qr.rebuild_in_place(&local_j, &r)?;
        let factor = qr.into_factor();
        let entries =
            self.entries - removed_entries + factor.rows.iter().map(Vec::len).sum::<usize>();
        if entries > self.options.max_entries {
            return Err(Error::SparseFillLimit {
                limit: self.options.max_entries,
            });
        }
        let constant = self.constant_cost + factor.constant_cost;
        if !constant.is_finite() {
            return Err(Error::NonFinite);
        }
        for (local, row) in factor.rows.into_iter().enumerate() {
            let global = selected[local];
            self.conditionals[global] = Arc::new(Conditional {
                row: row
                    .into_iter()
                    .map(|(column, value)| (selected[column], value))
                    .collect(),
                rhs: factor.rhs[local],
            });
        }
        self.entries = entries;
        self.constant_cost = constant;
        self.update_topology(&selected, &local_index)?;
        self.last_update = BayesTreeUpdate {
            reeliminated_cliques: selected.len(),
            reused_cliques: n - selected.len(),
            input_rows: rows,
        };
        Ok(())
    }
    fn rebuild_topology(&mut self) -> Result<(), Error> {
        let n = self.dimension();
        let mut filled: Vec<BTreeSet<usize>> = self
            .conditionals
            .iter()
            .map(|conditional| {
                conditional
                    .row
                    .iter()
                    .skip(1)
                    .map(|&(col, _)| col)
                    .collect()
            })
            .collect();
        let mut count = n;
        for k in 0..n {
            count = count
                .checked_add(filled[k].len())
                .ok_or(Error::InvalidDimensions)?;
            if count > self.options.max_entries {
                return Err(Error::SparseFillLimit {
                    limit: self.options.max_entries,
                });
            }
            if let Some(&parent) = filled[k].first() {
                let tail = filled[k].iter().copied().skip(1).collect::<Vec<_>>();
                filled[parent].extend(tail);
            }
        }
        self.symbolic_entries = count;
        self.separators = filled
            .into_iter()
            .map(|set| Arc::from(set.into_iter().collect::<Vec<_>>()))
            .collect();
        self.parents = self
            .separators
            .iter()
            .map(|separator| separator.first().copied())
            .collect();
        Ok(())
    }
    fn update_topology(&mut self, selected: &[usize], local_index: &[usize]) -> Result<(), Error> {
        // Keep previous symbolic fill conservatively. Adding factors may add
        // edges; a rounded numeric cancellation must not remove dependencies.
        let mut filled: Vec<BTreeSet<usize>> = selected
            .iter()
            .map(|&k| {
                self.separators[k]
                    .iter()
                    .copied()
                    .chain(self.conditionals[k].row.iter().skip(1).map(|&(col, _)| col))
                    .collect()
            })
            .collect();
        let mut count = self.symbolic_entries;
        for &k in selected {
            count -= self.separators[k].len();
        }
        for local in 0..selected.len() {
            count = count
                .checked_add(filled[local].len())
                .ok_or(Error::InvalidDimensions)?;
            if count > self.options.max_entries {
                return Err(Error::SparseFillLimit {
                    limit: self.options.max_entries,
                });
            }
            if let Some(&parent) = filled[local].first() {
                let parent_local = local_index[parent];
                if parent_local == usize::MAX {
                    return Err(Error::InvalidDimensions);
                }
                let tail = filled[local].iter().copied().skip(1).collect::<Vec<_>>();
                filled[parent_local].extend(tail);
            }
        }
        for (&k, separator) in selected.iter().zip(filled) {
            self.parents[k] = separator.first().copied();
            self.separators[k] = Arc::from(separator.into_iter().collect::<Vec<_>>());
        }
        self.symbolic_entries = count;
        Ok(())
    }
    /// Full back-substitution. No damping, pseudoinverse or normal matrix.
    pub fn solve(&self) -> Result<DVector<f64>, Error> {
        let largest = self
            .conditionals
            .iter()
            .map(|c| c.row.first().map_or(0., |&(_, v)| v.abs()))
            .fold(0_f64, f64::max);
        let cutoff = largest * self.options.relative_rank_tolerance;
        let rank = self
            .conditionals
            .iter()
            .filter(|c| c.row.first().is_some_and(|&(_, v)| v.abs() > cutoff))
            .count();
        if rank < self.dimension() {
            return Err(Error::RankDeficient {
                rank,
                parameters: self.dimension(),
            });
        }
        let mut x = DVector::zeros(self.dimension());
        for k in (0..self.dimension()).rev() {
            let c = &self.conditionals[k];
            let value = c.row.iter().skip(1).fold(c.rhs, |value, &(col, a)| {
                value - a * x[self.to_original[col]]
            });
            x[self.to_original[k]] = value / c.row[0].1;
        }
        finite_vector(&x)?;
        Ok(x)
    }
    pub fn cost(&self, x: &DVector<f64>) -> Result<f64, Error> {
        if x.len() != self.dimension() {
            return Err(Error::InvalidDimensions);
        }
        finite_vector(x)?;
        let mut cost = self.constant_cost;
        for conditional in &self.conditionals {
            let r = conditional
                .row
                .iter()
                .fold(-conditional.rhs, |r, &(col, a)| {
                    r + a * x[self.to_original[col]]
                });
            cost += 0.5 * r * r;
        }
        if cost.is_finite() {
            Ok(cost)
        } else {
            Err(Error::NonFinite)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn permuted_factor_and_new_columns_use_original_coordinates() {
        let mut qr = IncrementalQr::new(3, Default::default()).unwrap();
        qr.set_column_order(vec![2, 0, 1]).unwrap();
        let j = CsrMatrix::from_triplets(3, 3, vec![(0, 0, 1.), (1, 1, 1.), (2, 2, 1.)]).unwrap();
        qr.append(&j, &DVector::from_vec(vec![-1., -2., -3.]))
            .unwrap();
        let mut tree = GaussianBayesTree::from_qr(qr).unwrap();
        assert_eq!(tree.frontal_column(0).unwrap(), 2);
        assert_eq!(tree.clique_for_column(0).unwrap(), 1);
        tree.extend_columns(4).unwrap();
        tree.append(
            &CsrMatrix::from_triplets(1, 4, vec![(0, 0, -1.), (0, 3, 1.)]).unwrap(),
            &DVector::from_element(1, -4.),
        )
        .unwrap();
        let x = DVector::from_vec(vec![1., 2., 3., 5.]);
        assert!((tree.solve().unwrap() - &x).norm() < 1e-12);
        assert!(tree.cost(&x).unwrap() < 1e-24);
    }
    #[test]
    fn cancelled_fill_preserves_running_intersection_and_orphan_identity() {
        // Numeric R is not chordal: row 0 has separator [1,2], but row 1
        // has no numeric 2 entry. Topology must fill row 1's separator.
        let j = CsrMatrix::from_triplets(
            3,
            3,
            vec![(0, 0, 1.), (0, 1, 1.), (0, 2, 1.), (1, 1, 1.), (2, 2, 1.)],
        )
        .unwrap();
        let mut qr = IncrementalQr::new(3, Default::default()).unwrap();
        qr.append(&j, &DVector::zeros(3)).unwrap();
        let mut tree = GaussianBayesTree::from_qr(qr).unwrap();
        assert_eq!(tree.separator(1).unwrap(), &[2]);
        let orphan = tree.conditionals[0].clone();
        let update = CsrMatrix::from_triplets(1, 3, vec![(0, 1, 2.)]).unwrap();
        assert_eq!(
            tree.append(&update, &DVector::from_element(1, -1.))
                .unwrap()
                .reeliminated_cliques,
            2
        );
        assert!(Arc::ptr_eq(&orphan, &tree.conditionals[0]));
        let x = tree.solve().unwrap();
        assert!(
            (j.transpose_multiply(&j.multiply(&x).unwrap()).unwrap()
                + update
                    .transpose_multiply(
                        &(update.multiply(&x).unwrap() - DVector::from_element(1, 1.))
                    )
                    .unwrap())
            .norm()
                < 1e-12
        );
    }
}
