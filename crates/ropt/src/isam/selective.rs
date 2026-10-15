//! Factor-owned Gaussian rows in one product of per-variable tangent charts.
use super::{GaussianCache, IsamOptions, anchor_coordinates};
use crate::{
    DVector, Error, LocalProblem,
    bayes_tree::{EditableBayesTree, GaussianFactor},
    factor::{FactorGraph, VariableId},
    fixed_lag::{Key, SmootherFactor, WindowVariable, build_graph},
    incremental_qr::IncrementalQr,
    incremental_qr::IncrementalQrOptions,
    sparse::CsrMatrix,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

/// Work in a threshold-selected update. Counts describe factor linearizations,
/// not individual AD chunk callbacks. The explicit editable backend reports
/// Gaussian clique work separately; back-substitution remains global.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectiveRelinearizationReport {
    pub passes: usize,
    /// Distinct nonfixed keys whose anchors moved during this update.
    pub variables_relinearized: Vec<Key>,
    /// Selected factor evaluations summed over passes, including repeated factors.
    /// Excludes the initial evaluation of newly appended factors.
    pub factor_relinearizations: usize,
    /// Factors neither newly added nor selected in any pass, with shared row data.
    pub retained_factors: usize,
    /// Local or raw Gaussian steps still exceed the configured threshold after
    /// the pass cap. False is not a nonlinear convergence certificate.
    pub threshold_remaining: bool,
}

#[derive(Clone)]
pub(super) struct FactorCache {
    blocks: Vec<Arc<GaussianFactor>>,
    entries: usize,
    max_passes: usize,
}
pub(super) struct SelectiveContext<'a> {
    pub factors: &'a [SmootherFactor],
    pub order: &'a [Key],
    pub options: &'a IsamOptions,
    pub new_count: usize,
}
pub(super) struct SelectiveResult {
    pub qr: GaussianCache,
    pub coordinates: (DVector<f64>, f64),
    pub report: SelectiveRelinearizationReport,
    pub distinct_factors: usize,
}
fn capture(
    j: &CsrMatrix,
    r: &DVector<f64>,
    factors: &[SmootherFactor],
    limit: usize,
) -> Result<Vec<Arc<GaussianFactor>>, Error> {
    if j.shape().0 != r.len()
        || factors
            .iter()
            .try_fold(0usize, |sum, f| sum.checked_add(f.residuals))
            != Some(r.len())
    {
        return Err(Error::InvalidDimensions);
    }
    if r.len()
        .checked_add(j.nnz())
        .is_none_or(|entries| entries > limit)
    {
        return Err(Error::SparseFillLimit { limit });
    }
    let mut start = 0;
    let mut blocks = Vec::with_capacity(factors.len());
    let (row_offsets, columns) = j.pattern();
    for factor in factors {
        // Each factor owns consecutive canonical rows. Copy its contiguous
        // coefficients once, with exact lengths, retaining global column IDs.
        let end = start + factor.residuals;
        let base = row_offsets[start];
        let stop = row_offsets[end];
        let offsets = row_offsets[start..=end]
            .iter()
            .map(|&offset| offset - base)
            .collect();
        blocks.push(Arc::new(GaussianFactor::new(
            CsrMatrix::from_canonical(
                factor.residuals,
                j.shape().1,
                offsets,
                columns[base..stop].to_vec(),
                j.values()[base..stop].to_vec(),
            )?,
            r.rows(start, factor.residuals).into_owned(),
        )?));
        start = end;
    }
    Ok(blocks)
}
impl FactorCache {
    pub fn empty_like(&self) -> Self {
        Self::new(self.max_passes)
    }
    pub fn new(max_passes: usize) -> Self {
        Self {
            blocks: Vec::new(),
            entries: 0,
            max_passes,
        }
    }
    pub fn reset(
        &mut self,
        j: &CsrMatrix,
        r: &DVector<f64>,
        factors: &[SmootherFactor],
        limit: usize,
    ) -> Result<(), Error> {
        let blocks = capture(j, r, factors, limit)?;
        self.entries = r.len() + j.nnz();
        self.blocks = blocks;
        Ok(())
    }
    pub fn append(
        &mut self,
        j: &CsrMatrix,
        r: &DVector<f64>,
        factors: &[SmootherFactor],
        limit: usize,
    ) -> Result<(), Error> {
        let blocks = capture(j, r, factors, limit)?;
        let entries = self
            .entries
            .checked_add(r.len())
            .and_then(|n| n.checked_add(j.nnz()))
            .filter(|&n| n <= limit)
            .ok_or(Error::SparseFillLimit { limit })?;
        self.blocks.extend(blocks);
        self.entries = entries;
        Ok(())
    }
    fn assemble(&self, columns: usize) -> Result<(DVector<f64>, CsrMatrix), Error> {
        let mut residual = Vec::new();
        let mut offsets = vec![0];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for block in &self.blocks {
            residual.extend_from_slice(block.residual().as_slice());
            let base = values.len();
            let (block_offsets, block_columns) = block.jacobian().pattern();
            offsets.extend(block_offsets.iter().skip(1).map(|&offset| base + offset));
            indices.extend_from_slice(block_columns);
            values.extend_from_slice(block.jacobian().values());
        }
        let j = CsrMatrix::from_canonical(residual.len(), columns, offsets, indices, values)?;
        Ok((DVector::from_vec(residual), j))
    }
    fn shared_factors(
        &self,
        indices: impl Iterator<Item = usize>,
    ) -> Result<Vec<(u64, Arc<GaussianFactor>)>, Error> {
        indices
            .map(|index| {
                Ok((
                    u64::try_from(index).map_err(|_| Error::InvalidDimensions)?,
                    self.blocks[index].clone(),
                ))
            })
            .collect()
    }
    pub fn install_new(&self, tree: &mut EditableBayesTree, count: usize) -> Result<(), Error> {
        let start = self
            .blocks
            .len()
            .checked_sub(count)
            .ok_or(Error::InvalidDimensions)?;
        tree.update_shared(&[], self.shared_factors(start..self.blocks.len())?)?;
        Ok(())
    }
    pub fn build_tree(
        &self,
        columns: usize,
        permutation: Option<&[usize]>,
        options: IncrementalQrOptions,
    ) -> Result<EditableBayesTree, Error> {
        let mut tree = EditableBayesTree::new_ordered(
            permutation.map_or_else(|| (0..columns).collect(), <[usize]>::to_vec),
            options,
        )?;
        tree.update_shared(&[], self.shared_factors(0..self.blocks.len())?)?;
        Ok(tree)
    }
    pub fn relinearize(
        &mut self,
        variables: &mut BTreeMap<Key, WindowVariable>,
        anchors: &mut BTreeMap<Key, WindowVariable>,
        context: SelectiveContext<'_>,
        mut keys: BTreeSet<Key>,
        mut gaussian: GaussianCache,
    ) -> Result<SelectiveResult, Error> {
        if self.blocks.len() != context.factors.len()
            || context.new_count > self.blocks.len()
            || keys.is_empty()
        {
            return Err(Error::InvalidDimensions);
        }
        let mut touched =
            (self.blocks.len() - context.new_count..self.blocks.len()).collect::<BTreeSet<_>>();
        let mut moved = BTreeSet::new();
        let mut report = SelectiveRelinearizationReport {
            passes: 0,
            variables_relinearized: Vec::new(),
            factor_relinearizations: 0,
            retained_factors: 0,
            threshold_remaining: false,
        };
        let permutation = gaussian.column_permutation().map(<[usize]>::to_vec);
        for _ in 0..self.max_passes {
            for &key in &keys {
                let variable = variables.get(&key).ok_or(Error::InvalidDimensions)?;
                if variable.fixed {
                    return Err(Error::InvalidDimensions);
                }
                anchors.insert(key, variable.clone());
                moved.insert(key);
            }
            let indices = context
                .factors
                .iter()
                .enumerate()
                .filter(|(_, factor)| factor.keys.iter().any(|key| keys.contains(key)))
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            let factors = indices
                .iter()
                .map(|&index| context.factors[index].clone())
                .collect::<Vec<_>>();
            let (graph, ids) = build_graph(anchors, &factors, context.order)?;
            let (r, j) = graph.evaluate_sparse(&graph.initial())?;
            let replacement = capture(&j, &r, &factors, context.options.qr.max_entries)?;
            let removed = indices.iter().try_fold(0usize, |sum, &index| {
                sum.checked_add(self.blocks[index].entries()?)
                    .ok_or(Error::InvalidDimensions)
            })?;
            let entries = (self.entries - removed)
                .checked_add(r.len())
                .and_then(|n| n.checked_add(j.nnz()))
                .filter(|&n| n <= context.options.qr.max_entries)
                .ok_or(Error::SparseFillLimit {
                    limit: context.options.qr.max_entries,
                })?;
            for (&index, block) in indices.iter().zip(replacement) {
                self.blocks[index] = block;
                touched.insert(index);
            }
            self.entries = entries;
            if let GaussianCache::Editable(tree) = &mut gaussian {
                tree.update_shared(&[], self.shared_factors(indices.iter().copied())?)?;
            } else {
                let (r, j) = self.assemble(graph.dimensions().1)?;
                let mut qr = IncrementalQr::new(j.shape().1, context.options.qr)?;
                if let Some(order) = &permutation {
                    qr.set_column_order(order.clone())?;
                }
                qr.rebuild_in_place(&j, &r)?;
                gaussian = GaussianCache::Qr(qr);
            }
            let delta = gaussian.solve()?;
            let state = graph.retract(&graph.initial(), &delta)?;
            for (&key, &id) in &ids {
                variables.get_mut(&key).unwrap().value = graph.value(&state, id)?;
            }
            keys = threshold_keys(
                variables,
                anchors,
                &graph,
                &ids,
                &delta,
                context.options.relinearization_threshold,
            )?;
            report.passes += 1;
            report.factor_relinearizations += indices.len();
            report.threshold_remaining = !keys.is_empty();
            if keys.is_empty() || report.passes == self.max_passes {
                report.retained_factors = self.blocks.len() - touched.len();
                report.variables_relinearized = moved.into_iter().collect();
                return Ok(SelectiveResult {
                    qr: gaussian,
                    coordinates: anchor_coordinates(variables, anchors, &graph, &ids)?,
                    report,
                    distinct_factors: touched.len(),
                });
            }
        }
        Err(Error::InvalidOptions)
    }
}
pub(super) fn threshold_keys(
    variables: &BTreeMap<Key, WindowVariable>,
    anchors: &BTreeMap<Key, WindowVariable>,
    graph: &FactorGraph,
    ids: &BTreeMap<Key, VariableId>,
    delta: &DVector<f64>,
    threshold: f64,
) -> Result<BTreeSet<Key>, Error> {
    let mut keys = BTreeSet::new();
    for (&key, &id) in ids {
        if let Some(range) = graph.tangent_range(id)? {
            let variable = &variables[&key];
            let local = variable
                .kind
                .local(&anchors[&key].value, &variable.value)?
                .norm();
            let raw = delta.rows(range.start, range.len()).norm();
            if !local.is_finite() || !raw.is_finite() {
                return Err(Error::NonFinite);
            }
            if local.max(raw) > threshold {
                keys.insert(key);
            }
        }
    }
    Ok(keys)
}

#[cfg(test)]
mod capture_tests {
    use super::*;

    fn recipes(rows: &[usize]) -> Vec<SmootherFactor> {
        rows.iter()
            .map(|&n| SmootherFactor::analytic(&[], n, Default::default(), |_| unreachable!()))
            .collect()
    }

    // Preserve the preceding row-iterator assembly as an independent reference.
    fn reference(j: &CsrMatrix, r: &DVector<f64>, rows: &[usize]) -> Vec<GaussianFactor> {
        let mut start = 0;
        rows.iter()
            .map(|&n| {
                let mut offsets = vec![0];
                let (mut columns, mut values) = (Vec::new(), Vec::new());
                for row in start..start + n {
                    for (column, value) in j.row(row).unwrap() {
                        columns.push(column);
                        values.push(value);
                    }
                    offsets.push(values.len());
                }
                let block = GaussianFactor::new(
                    CsrMatrix::from_canonical(n, j.shape().1, offsets, columns, values).unwrap(),
                    r.rows(start, n).into_owned(),
                )
                .unwrap();
                start += n;
                block
            })
            .collect()
    }

    #[test]
    fn consecutive_capture_preserves_empty_rows_global_columns_and_coefficient_bits() {
        for columns in [7, 100_000] {
            let j = CsrMatrix::from_triplets(
                6,
                columns,
                vec![
                    (0, 0, 1e-200),
                    (0, columns - 1, -2.),
                    (2, 3, 1e200),
                    (4, 1, -1e-100),
                    (4, 3, 3.),
                    (4, columns - 1, 5.),
                ],
            )
            .unwrap();
            let r = DVector::from_vec(vec![-0., 1., -2., 0., 1e-200, -1e200]);
            for rows in [&[1, 3, 2][..], &[2, 1, 1, 2], &[6], &[1, 1, 1, 1, 1, 1]] {
                let factors = recipes(rows);
                let actual = capture(&j, &r, &factors, r.len() + j.nnz()).unwrap();
                let expected = reference(&j, &r, rows);
                for (a, b) in actual.iter().zip(&expected) {
                    assert_eq!(a.jacobian().shape(), b.jacobian().shape());
                    assert_eq!(a.jacobian().pattern(), b.jacobian().pattern());
                    assert_eq!(
                        a.jacobian()
                            .values()
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        b.jacobian()
                            .values()
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        a.residual().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        b.residual().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
                let mut cache = FactorCache::new(1);
                cache.reset(&j, &r, &factors, r.len() + j.nnz()).unwrap();
                let (assembled_r, assembled_j) = cache.assemble(columns).unwrap();
                assert_eq!(assembled_j.pattern(), j.pattern());
                assert_eq!(assembled_j.values(), j.values());
                assert_eq!(
                    assembled_r.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    r.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
        // Constant-only observations, including factors with no active columns.
        let j = CsrMatrix::from_triplets(3, 0, vec![]).unwrap();
        let r = DVector::from_vec(vec![1., -0., 2.]);
        let blocks = capture(&j, &r, &recipes(&[1, 2]), 3).unwrap();
        assert_eq!(blocks.iter().map(|b| b.jacobian().nnz()).sum::<usize>(), 0);
        assert_eq!(blocks[1].residual()[0].to_bits(), (-0_f64).to_bits());
    }

    #[test]
    fn capture_checks_dimensions_fill_and_finiteness_before_committing_cache() {
        let j = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 1.), (1, 1, 2.)]).unwrap();
        let r = DVector::from_vec(vec![1., 2.]);
        let good = recipes(&[1, 1]);
        let mut cache = FactorCache::new(1);
        cache.reset(&j, &r, &good, 4).unwrap();
        let saved = cache.blocks.clone();
        let bad_residual = DVector::from_vec(vec![1., f64::NAN]);
        assert!(matches!(
            cache.reset(&j, &r, &recipes(&[3]), 4),
            Err(Error::InvalidDimensions)
        ));
        assert!(matches!(
            cache.reset(&j, &r, &recipes(&[0, 2]), 4),
            Err(Error::InvalidDimensions)
        ));
        assert!(matches!(
            cache.reset(&j, &r, &recipes(&[usize::MAX, 1]), 4),
            Err(Error::InvalidDimensions)
        ));
        assert!(matches!(
            cache.reset(&j, &bad_residual, &good, 3),
            Err(Error::SparseFillLimit { limit: 3 })
        ));
        assert!(matches!(
            cache.reset(&j, &bad_residual, &good, 4),
            Err(Error::NonFinite)
        ));
        assert!(matches!(
            cache.append(&j, &r, &good, 7),
            Err(Error::SparseFillLimit { limit: 7 })
        ));
        assert_eq!(cache.entries, 4);
        assert_eq!(cache.blocks.len(), saved.len());
        assert!(
            cache
                .blocks
                .iter()
                .zip(&saved)
                .all(|(a, b)| Arc::ptr_eq(a, b))
        );
    }
}
