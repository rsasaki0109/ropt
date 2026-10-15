//! Factor replacement by multifrontal square-root elimination and cached messages.
use super::{Conditional, GaussianBayesTree};
use crate::{
    DVector, Error,
    incremental_qr::{IncrementalQr, IncrementalQrOptions},
    sparse::{CsrMatrix, finite_vector},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub type GaussianFactorId = u64;

/// An owned Gaussian observation block: cost = 0.5 * ||J*x+residual||².
/// Columns are original parameter coordinates, not elimination coordinates.
#[derive(Debug, Clone)]
pub struct GaussianFactor {
    jacobian: CsrMatrix,
    residual: DVector<f64>,
}

impl GaussianFactor {
    pub fn new(jacobian: CsrMatrix, residual: DVector<f64>) -> Result<Self, Error> {
        if jacobian.shape().0 == 0 || jacobian.shape().0 != residual.len() {
            return Err(Error::InvalidDimensions);
        }
        finite_vector(&residual)?;
        Ok(Self { jacobian, residual })
    }
    pub fn jacobian(&self) -> &CsrMatrix {
        &self.jacobian
    }
    pub fn residual(&self) -> &DVector<f64> {
        &self.residual
    }
    pub(crate) fn entries(&self) -> Result<usize, Error> {
        self.jacobian
            .nnz()
            .checked_add(self.residual.len())
            .ok_or(Error::InvalidDimensions)
    }
}

/// Scalar-clique work for the last committed factor update. Global metadata
/// analysis and full back-substitution are not included in these counts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EditableBayesTreeUpdate {
    pub reeliminated_cliques: usize,
    pub reused_cliques: usize,
    /// Original observation rows gathered by affected cliques.
    pub input_factor_rows: usize,
    /// Separator rows gathered, including newly computed child messages.
    pub input_separator_rows: usize,
    /// Unaffected child messages reused at the affected region's boundary.
    pub reused_boundary_messages: usize,
}
#[derive(Debug)]
struct Message {
    // Sorted internal coordinates. Each row represents row*x-rhs.
    rows: Vec<Vec<(usize, f64)>>,
    rhs: Vec<f64>,
    constant_cost: f64,
}
impl Message {
    fn empty() -> Self {
        Self {
            rows: Vec::new(),
            rhs: Vec::new(),
            constant_cost: 0.,
        }
    }
    fn entries(&self) -> Result<usize, Error> {
        self.rows
            .iter()
            .try_fold(self.rhs.len(), |count, row| count.checked_add(row.len()))
            .ok_or(Error::InvalidDimensions)
    }
}
type Topology = (
    Vec<Arc<[usize]>>,
    Vec<Option<usize>>,
    Arc<BTreeMap<GaussianFactorId, Option<usize>>>,
    usize,
);

#[derive(Debug, Clone)]
struct Incidence {
    // Sorted lists preserve the original child/factor row-gathering order.
    children: Vec<Arc<[usize]>>,
    factors: Vec<Arc<[GaussianFactorId]>>,
}
impl Incidence {
    fn new(columns: usize) -> Self {
        Self {
            children: vec![Arc::from([]); columns],
            factors: vec![Arc::from([]); columns],
        }
    }
    fn extend(&mut self, columns: usize) {
        self.children.resize(columns, Arc::from([]));
        self.factors.resize(columns, Arc::from([]));
    }
}
#[derive(Default)]
struct IncidenceDelta<T> {
    removed: Vec<T>,
    added: Vec<T>,
}
fn update_incidence_row<T: Copy + Ord>(
    old: &[T],
    delta: IncidenceDelta<T>,
) -> Result<Arc<[T]>, Error> {
    // Deltas arrive in ascending unique child/ID order. Merge once per changed
    // row, including high-degree roots, without repeated insertion or sorting.
    if delta.removed.windows(2).any(|p| p[0] >= p[1])
        || delta.added.windows(2).any(|p| p[0] >= p[1])
    {
        return Err(Error::InvalidDimensions);
    }
    let capacity = old
        .len()
        .checked_sub(delta.removed.len())
        .and_then(|n| n.checked_add(delta.added.len()))
        .ok_or(Error::InvalidDimensions)?;
    let mut row = Vec::with_capacity(capacity);
    let (mut removed, mut added) = (0, 0);
    for &entry in old {
        if let Some(&next) = delta.removed.get(removed) {
            if next < entry {
                return Err(Error::InvalidDimensions);
            }
            if next == entry {
                removed += 1;
                continue;
            }
        }
        while delta.added.get(added).is_some_and(|&next| next < entry) {
            row.push(delta.added[added]);
            added += 1;
        }
        if delta.added.get(added) == Some(&entry) {
            return Err(Error::InvalidDimensions);
        }
        row.push(entry);
    }
    if removed != delta.removed.len() {
        return Err(Error::InvalidDimensions);
    }
    row.extend_from_slice(&delta.added[added..]);
    Ok(Arc::from(row))
}

/// Fixed-order scalar Gaussian Bayes forest with original factor ownership and
/// cached square-root separator messages. Factor replacement/deletion replays
/// only affected original factors and child messages; unchanged cliques share
/// immutable conditionals/messages. No QR downdate or normal matrix is used.
///
/// Symbolic topology is reused for same-support replacements and extended for
/// additions; deletion or changed support requires complete analysis. Metadata
/// bookkeeping and solve still visit global state. This is a
/// Gaussian backend used by IsamSmoother's explicit selective Bayes-tree policy.
#[derive(Debug, Clone)]
pub struct EditableBayesTree {
    tree: GaussianBayesTree,
    messages: Vec<Arc<Message>>,
    // Candidate/snapshot clones share metadata until a map is changed. In
    // particular, column extension changes neither map, and same-support
    // replacement keeps ownership shared. Detaching a changed map is still O(n).
    factors: Arc<BTreeMap<GaussianFactorId, Arc<GaussianFactor>>>,
    owners: Arc<BTreeMap<GaussianFactorId, Option<usize>>>,
    incidence: Arc<Incidence>,
    factor_entries: usize,
    message_entries: usize,
    last_update: EditableBayesTreeUpdate,
}
impl EditableBayesTree {
    pub fn new(columns: usize, options: IncrementalQrOptions) -> Result<Self, Error> {
        if columns > options.max_entries {
            return Err(Error::SparseFillLimit {
                limit: options.max_entries,
            });
        }
        Self::new_ordered((0..columns).collect(), options)
    }
    /// Fixed permutation maps internal elimination indices to original columns.
    pub fn new_ordered(
        permutation: Vec<usize>,
        options: IncrementalQrOptions,
    ) -> Result<Self, Error> {
        let n = permutation.len();
        let mut qr = IncrementalQr::new(n, options)?;
        qr.set_column_order(permutation)?;
        let empty_message = Arc::new(Message::empty());
        Ok(Self {
            tree: GaussianBayesTree::from_qr(qr)?,
            messages: vec![empty_message; n],
            factors: Arc::new(BTreeMap::new()),
            owners: Arc::new(BTreeMap::new()),
            incidence: Arc::new(Incidence::new(n)),
            factor_entries: 0,
            message_entries: 0,
            last_update: EditableBayesTreeUpdate {
                reused_cliques: n,
                ..Default::default()
            },
        })
    }
    pub fn dimension(&self) -> usize {
        self.tree.dimension()
    }
    pub fn factor_count(&self) -> usize {
        self.factors.len()
    }
    pub fn factor(&self, id: GaussianFactorId) -> Option<&GaussianFactor> {
        self.factors.get(&id).map(AsRef::as_ref)
    }
    /// Numeric conditional coefficients, separately bounded by max_entries.
    pub fn stored_entries(&self) -> usize {
        self.tree.stored_entries()
    }
    /// Original nonzero coefficients plus residual entries, separately bounded.
    pub fn factor_entries(&self) -> usize {
        self.factor_entries
    }
    /// Cached message nonzero coefficients plus right-hand sides, separately bounded.
    pub fn separator_entries(&self) -> usize {
        self.message_entries
    }
    pub fn last_update(&self) -> EditableBayesTreeUpdate {
        self.last_update
    }
    pub fn column_permutation(&self) -> &[usize] {
        &self.tree.to_original
    }
    pub fn parent(&self, clique: usize) -> Result<Option<usize>, Error> {
        self.tree.parent(clique)
    }
    pub fn separator(&self, clique: usize) -> Result<&[usize], Error> {
        self.tree.separator(clique)
    }
    pub fn solve(&self) -> Result<DVector<f64>, Error> {
        self.tree.solve()
    }
    pub fn cost(&self, x: &DVector<f64>) -> Result<f64, Error> {
        self.tree.cost(x)
    }
    /// Extend with unconstrained trailing original columns; earlier order is fixed.
    pub fn extend_columns(&mut self, columns: usize) -> Result<(), Error> {
        let mut candidate = self.clone();
        candidate.tree.extend_columns(columns)?;
        candidate
            .messages
            .resize_with(columns, || Arc::new(Message::empty()));
        if candidate.incidence.children.len() != columns {
            Arc::make_mut(&mut candidate.incidence).extend(columns);
        }
        *self = candidate;
        Ok(())
    }
    /// Atomically remove existing IDs and insert/replace Gaussian blocks. Lists
    /// must contain unique, disjoint IDs. Replacement columns must match the
    /// current dimension; measurements keep their supplied original coordinates.
    /// Incomplete models can be installed: solve checks observability separately.
    /// Any dimension, finite-value, fill or arithmetic error leaves state unchanged.
    ///
    /// max_entries separately limits original coefficients+residuals, cached
    /// message coefficients+RHS, numeric conditionals, symbolic structure, and
    /// each local gathered coefficient+residual workspace. It is not peak memory.
    pub fn update(
        &mut self,
        removed: &[GaussianFactorId],
        replacements: Vec<(GaussianFactorId, GaussianFactor)>,
    ) -> Result<EditableBayesTreeUpdate, Error> {
        self.update_shared(
            removed,
            replacements
                .into_iter()
                .map(|(id, factor)| (id, Arc::new(factor)))
                .collect(),
        )
    }
    pub(crate) fn update_shared(
        &mut self,
        removed: &[GaussianFactorId],
        replacements: Vec<(GaussianFactorId, Arc<GaussianFactor>)>,
    ) -> Result<EditableBayesTreeUpdate, Error> {
        crate::stream_profile::measure(7, || {
            let mut candidate = self.clone();
            candidate.update_in_place(removed, replacements)?;
            let report = candidate.last_update;
            *self = candidate;
            Ok(report)
        })
    }
    fn topology(&self) -> Result<Topology, Error> {
        let n = self.dimension();
        let mut filled = vec![BTreeSet::new(); n];
        let mut owners = BTreeMap::new();
        for (&id, factor) in self.factors.iter() {
            // Whole-block support is conservative when rows are disconnected.
            let support = factor
                .jacobian
                .entries()
                .map(|(_, column, _)| self.tree.to_internal[column])
                .collect::<BTreeSet<_>>();
            let owner = support.first().copied();
            owners.insert(id, owner);
            if let Some(k) = owner {
                filled[k].extend(support.into_iter().skip(1));
            }
        }
        let mut count = n;
        for k in 0..n {
            count = count
                .checked_add(filled[k].len())
                .ok_or(Error::InvalidDimensions)?;
            if count > self.tree.options.max_entries {
                return Err(Error::SparseFillLimit {
                    limit: self.tree.options.max_entries,
                });
            }
            if let Some(&parent) = filled[k].first() {
                let tail = filled[k].iter().copied().skip(1).collect::<Vec<_>>();
                filled[parent].extend(tail);
            }
        }
        let separators = filled
            .into_iter()
            .map(|set| Arc::from(set.into_iter().collect::<Vec<_>>()))
            .collect::<Vec<Arc<[usize]>>>();
        let parents = separators
            .iter()
            .map(|separator| separator.first().copied())
            .collect();
        Ok((separators, parents, Arc::new(owners), count))
    }
    fn factor_support(&self, factor: &GaussianFactor) -> BTreeSet<usize> {
        factor
            .jacobian
            .entries()
            .map(|(_, column, _)| self.tree.to_internal[column])
            .collect()
    }
    /// Adding supports cannot shrink symbolic fill. Same-support replacements
    /// change numeric rows only. Propagate added separator tails in elimination
    /// order; removals or changed supports use the complete pruning analysis.
    fn added_topology(
        &self,
        removed: &[GaussianFactorId],
        replacements: &[(GaussianFactorId, Arc<GaussianFactor>)],
    ) -> Result<Option<Topology>, Error> {
        if !removed.is_empty() {
            return Ok(None);
        }
        let mut additions = Vec::new();
        for (id, factor) in replacements {
            let support = self.factor_support(factor);
            if let Some(old) = self.factors.get(id) {
                if self.factor_support(old) != support {
                    return Ok(None);
                }
            } else {
                additions.push((*id, support));
            }
        }
        let mut separators = self.tree.separators.clone();
        let mut parents = self.tree.parents.clone();
        let mut owners = self.owners.clone();
        let mut count = self.tree.symbolic_entries;
        let mut pending = BTreeMap::<usize, BTreeSet<usize>>::new();
        for (id, support) in additions {
            let owner = support.first().copied();
            Arc::make_mut(&mut owners).insert(id, owner);
            if let Some(k) = owner {
                pending
                    .entry(k)
                    .or_default()
                    .extend(support.into_iter().skip(1));
            }
        }
        while let Some((k, added)) = pending.pop_first() {
            let filled = separators[k]
                .iter()
                .copied()
                .chain(added)
                .collect::<BTreeSet<_>>();
            if filled.len() == separators[k].len() {
                continue;
            }
            count = count
                .checked_add(filled.len() - separators[k].len())
                .ok_or(Error::InvalidDimensions)?;
            if count > self.tree.options.max_entries {
                return Err(Error::SparseFillLimit {
                    limit: self.tree.options.max_entries,
                });
            }
            parents[k] = filled.first().copied();
            if let Some(parent) = parents[k] {
                pending
                    .entry(parent)
                    .or_default()
                    .extend(filled.iter().copied().skip(1));
            }
            separators[k] = Arc::from(filled.into_iter().collect::<Vec<_>>());
        }
        Ok(Some((separators, parents, owners, count)))
    }
    fn updated_incidence(
        &self,
        parents: &[Option<usize>],
        owners: &BTreeMap<GaussianFactorId, Option<usize>>,
        changed: &BTreeSet<GaussianFactorId>,
    ) -> Result<Arc<Incidence>, Error> {
        let n = self.dimension();
        if parents.len() != n
            || self.incidence.children.len() != n
            || self.incidence.factors.len() != n
        {
            return Err(Error::InvalidDimensions);
        }
        // Fresh full-model construction has no previous ownership to patch.
        // Enumerate once rather than creating a delta-map entry for each row.
        if self.owners.is_empty() && owners.values().any(Option::is_some) {
            let mut children = vec![Vec::new(); n];
            let mut factors = vec![Vec::new(); n];
            for (child, &parent) in parents.iter().enumerate() {
                if let Some(parent) = parent {
                    children[parent].push(child);
                }
            }
            for (&id, &owner) in owners {
                if let Some(owner) = owner {
                    factors[owner].push(id);
                }
            }
            return Ok(Arc::new(Incidence {
                children: children
                    .into_iter()
                    .enumerate()
                    .map(|(k, row)| {
                        if row.is_empty() {
                            self.incidence.children[k].clone()
                        } else {
                            Arc::from(row)
                        }
                    })
                    .collect(),
                factors: factors
                    .into_iter()
                    .enumerate()
                    .map(|(k, row)| {
                        if row.is_empty() {
                            self.incidence.factors[k].clone()
                        } else {
                            Arc::from(row)
                        }
                    })
                    .collect(),
            }));
        }
        let mut children = BTreeMap::<usize, IncidenceDelta<usize>>::new();
        let mut factors = BTreeMap::<usize, IncidenceDelta<GaussianFactorId>>::new();
        for (child, (&old, &new)) in self.tree.parents.iter().zip(parents).enumerate() {
            if old != new {
                if let Some(parent) = old {
                    children.entry(parent).or_default().removed.push(child);
                }
                if let Some(parent) = new {
                    children.entry(parent).or_default().added.push(child);
                }
            }
        }
        for &id in changed {
            let old = self.owners.get(&id).copied().flatten();
            let new = owners.get(&id).copied().flatten();
            if old != new {
                if let Some(owner) = old {
                    factors.entry(owner).or_default().removed.push(id);
                }
                if let Some(owner) = new {
                    factors.entry(owner).or_default().added.push(id);
                }
            }
        }
        let mut incidence = self.incidence.clone();
        if !children.is_empty() || !factors.is_empty() {
            let index = Arc::make_mut(&mut incidence);
            for (parent, delta) in children {
                let row = index
                    .children
                    .get_mut(parent)
                    .ok_or(Error::InvalidDimensions)?;
                *row = update_incidence_row(row, delta)?;
            }
            for (owner, delta) in factors {
                let row = index
                    .factors
                    .get_mut(owner)
                    .ok_or(Error::InvalidDimensions)?;
                *row = update_incidence_row(row, delta)?;
            }
        }
        Ok(incidence)
    }
    fn gather_clique(
        &self,
        columns: &[usize],
        factors: &[GaussianFactorId],
        children: &[usize],
        rows: usize,
        workspace_entries: usize,
    ) -> Result<(CsrMatrix, DVector<f64>, f64), Error> {
        // The projection is injective. Original CSR rows may need a small
        // per-row sort after permutation, but never duplicate reduction.
        // Child messages already use sorted internal coordinates, so their
        // projection preserves order. Keep factor/child/residual order exactly.
        let coefficients = workspace_entries
            .checked_sub(rows)
            .ok_or(Error::InvalidDimensions)?;
        let mut offsets = Vec::with_capacity(rows.checked_add(1).ok_or(Error::InvalidDimensions)?);
        let mut indices = Vec::with_capacity(coefficients);
        let mut values = Vec::with_capacity(coefficients);
        let mut residual = Vec::with_capacity(rows);
        let mut projected = Vec::new();
        offsets.push(0);
        for &id in factors {
            let factor = &self.factors[&id];
            for (row, &r) in factor.residual.iter().enumerate() {
                projected.clear();
                for (column, value) in factor.jacobian.row(row)? {
                    let column = columns
                        .binary_search(&self.tree.to_internal[column])
                        .map_err(|_| Error::InvalidDimensions)?;
                    projected.push((column, value));
                }
                projected.sort_unstable_by_key(|&(column, _)| column);
                indices.extend(projected.iter().map(|&(column, _)| column));
                values.extend(projected.iter().map(|&(_, value)| value));
                offsets.push(values.len());
                residual.push(r);
            }
        }
        let mut constant = 0.;
        for &child in children {
            let message = &self.messages[child];
            constant += message.constant_cost;
            for (row, &rhs) in message.rows.iter().zip(&message.rhs) {
                for &(column, value) in row {
                    indices.push(
                        columns
                            .binary_search(&column)
                            .map_err(|_| Error::InvalidDimensions)?,
                    );
                    values.push(value);
                }
                offsets.push(values.len());
                residual.push(-rhs);
            }
        }
        if !constant.is_finite() {
            return Err(Error::NonFinite);
        }
        let j = CsrMatrix::from_canonical(rows, columns.len(), offsets, indices, values)?;
        Ok((j, DVector::from_vec(residual), constant))
    }
    fn update_in_place(
        &mut self,
        removed: &[GaussianFactorId],
        replacements: Vec<(GaussianFactorId, Arc<GaussianFactor>)>,
    ) -> Result<(), Error> {
        let n = self.dimension();
        let limit = self.tree.options.max_entries;
        let mut changed = BTreeSet::new();
        let mut pending = Vec::new();
        for &id in removed {
            if !changed.insert(id) || !self.factors.contains_key(&id) {
                return Err(Error::Evaluation(
                    "duplicate or unknown removed Gaussian factor ID".into(),
                ));
            }
            if let Some(Some(owner)) = self.owners.get(&id) {
                pending.push(*owner);
            }
        }
        for (id, factor) in &replacements {
            if !changed.insert(*id) {
                return Err(Error::Evaluation(
                    "duplicate or overlapping Gaussian factor ID".into(),
                ));
            }
            if factor.jacobian.shape().1 != n {
                return Err(Error::InvalidDimensions);
            }
            if let Some(Some(owner)) = self.owners.get(id) {
                pending.push(*owner);
            }
        }
        if changed.is_empty() {
            self.last_update = EditableBayesTreeUpdate {
                reused_cliques: n,
                ..Default::default()
            };
            return Ok(());
        }
        let incremental_topology = self.added_topology(removed, &replacements)?;
        {
            let factors = Arc::make_mut(&mut self.factors);
            for &id in removed {
                factors.remove(&id);
            }
            for (id, factor) in replacements {
                factors.insert(id, factor);
            }
        }
        let entries = self.factors.values().try_fold(0usize, |count, factor| {
            count
                .checked_add(factor.entries()?)
                .ok_or(Error::InvalidDimensions)
        })?;
        if entries > limit {
            return Err(Error::SparseFillLimit { limit });
        }
        let (separators, parents, owners, symbolic_entries) = match incremental_topology {
            Some(topology) => topology,
            None => self.topology()?,
        };
        for id in &changed {
            if let Some(Some(owner)) = owners.get(id) {
                pending.push(*owner);
            }
        }
        for (k, separator) in separators.iter().enumerate() {
            if self.tree.separators[k].as_ref() != separator.as_ref() {
                pending.push(k);
            }
        }
        let mut affected = BTreeSet::new();
        while let Some(k) = pending.pop() {
            if affected.insert(k) {
                pending.extend(self.tree.parents[k]);
                pending.extend(parents[k]);
            }
        }
        let incidence = self.updated_incidence(&parents, &owners, &changed)?;
        let children = &incidence.children;
        let buckets = &incidence.factors;
        let mut report = EditableBayesTreeUpdate {
            reeliminated_cliques: affected.len(),
            reused_cliques: n - affected.len(),
            ..Default::default()
        };
        for &k in &affected {
            let mut columns = Vec::with_capacity(separators[k].len() + 1);
            columns.push(k);
            columns.extend(separators[k].iter().copied());
            let mut rows = 0usize;
            let mut workspace_entries = 0usize;
            for &id in buckets[k].iter() {
                let factor = &self.factors[&id];
                rows = rows
                    .checked_add(factor.residual.len())
                    .ok_or(Error::InvalidDimensions)?;
                workspace_entries = workspace_entries
                    .checked_add(factor.entries()?)
                    .ok_or(Error::InvalidDimensions)?;
                report.input_factor_rows = report
                    .input_factor_rows
                    .checked_add(factor.residual.len())
                    .ok_or(Error::InvalidDimensions)?;
            }
            for &child in children[k].iter() {
                let message = &self.messages[child];
                rows = rows
                    .checked_add(message.rows.len())
                    .ok_or(Error::InvalidDimensions)?;
                workspace_entries = workspace_entries
                    .checked_add(message.entries()?)
                    .ok_or(Error::InvalidDimensions)?;
                report.input_separator_rows = report
                    .input_separator_rows
                    .checked_add(message.rows.len())
                    .ok_or(Error::InvalidDimensions)?;
                if !affected.contains(&child) {
                    report.reused_boundary_messages += 1;
                }
            }
            if workspace_entries > limit {
                return Err(Error::SparseFillLimit { limit });
            }
            let (j, r, carried_constant) =
                self.gather_clique(&columns, &buckets[k], &children[k], rows, workspace_entries)?;
            let mut qr = IncrementalQr::new(columns.len(), self.tree.options)?;
            qr.rebuild_in_place(&j, &r)?;
            let factor = qr.into_factor();
            let mut eliminated = factor.rows.into_iter().enumerate();
            let (_, frontal) = eliminated.next().ok_or(Error::InvalidDimensions)?;
            let conditional = Conditional {
                row: frontal
                    .into_iter()
                    .map(|(column, value)| (columns[column], value))
                    .collect(),
                rhs: factor.rhs[0],
            };
            self.tree.entries = (self.tree.entries - self.tree.conditionals[k].row.len())
                .checked_add(conditional.row.len())
                .ok_or(Error::InvalidDimensions)?;
            if self.tree.entries > limit {
                return Err(Error::SparseFillLimit { limit });
            }
            let mut message = Message {
                rows: Vec::new(),
                rhs: Vec::new(),
                constant_cost: carried_constant + factor.constant_cost,
            };
            for (row, coefficients) in eliminated {
                if coefficients.is_empty() {
                    message.constant_cost += 0.5 * factor.rhs[row] * factor.rhs[row];
                } else {
                    message.rows.push(
                        coefficients
                            .into_iter()
                            .map(|(column, value)| (columns[column], value))
                            .collect(),
                    );
                    message.rhs.push(factor.rhs[row]);
                }
            }
            if !message.constant_cost.is_finite() {
                return Err(Error::NonFinite);
            }
            self.message_entries = (self.message_entries - self.messages[k].entries()?)
                .checked_add(message.entries()?)
                .ok_or(Error::InvalidDimensions)?;
            if self.message_entries > limit {
                return Err(Error::SparseFillLimit { limit });
            }
            self.tree.conditionals[k] = Arc::new(conditional);
            self.messages[k] = Arc::new(message);
        }
        // Only current roots contribute marginalized constants. Adding every
        // orphan's original rows or every message constant would double-count.
        let mut constant = 0.;
        for (&id, &owner) in owners.iter() {
            if owner.is_none() {
                for &value in self.factors[&id].residual.iter() {
                    constant += 0.5 * value * value;
                }
            }
        }
        for (k, parent) in parents.iter().enumerate() {
            if parent.is_none() {
                constant += self.messages[k].constant_cost;
            }
        }
        if !constant.is_finite() {
            return Err(Error::NonFinite);
        }
        self.tree.constant_cost = constant;
        self.tree.separators = separators;
        self.tree.parents = parents;
        self.tree.symbolic_entries = symbolic_entries;
        self.owners = owners;
        self.incidence = incidence;
        self.factor_entries = entries;
        self.last_update = report;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn incidence_deltas_match_complete_sorted_enumeration_and_share_unchanged_rows() {
        let mut tree =
            EditableBayesTree::new_ordered(vec![4, 0, 3, 1, 5, 2], Default::default()).unwrap();
        let permutation = tree.column_permutation().to_vec();
        let block = |support: &[usize], scale: f64| {
            GaussianFactor::new(
                CsrMatrix::from_triplets(
                    1,
                    6,
                    support
                        .iter()
                        .map(|&k| (0, permutation[k], scale))
                        .collect(),
                )
                .unwrap(),
                DVector::from_element(1, -0.5),
            )
            .unwrap()
        };
        let check = |tree: &EditableBayesTree| {
            let mut children = vec![Vec::new(); tree.dimension()];
            let mut factors = vec![Vec::new(); tree.dimension()];
            for (child, parent) in tree.tree.parents.iter().enumerate() {
                if let Some(parent) = parent {
                    children[*parent].push(child);
                }
            }
            for (&id, &owner) in tree.owners.iter() {
                if let Some(owner) = owner {
                    factors[owner].push(id);
                }
            }
            for k in 0..tree.dimension() {
                assert_eq!(tree.incidence.children[k].as_ref(), children[k]);
                assert_eq!(tree.incidence.factors[k].as_ref(), factors[k]);
            }
        };
        tree.update(
            &[],
            (0..6).rev().map(|k| (k as u64, block(&[k], 1.))).collect(),
        )
        .unwrap();
        tree.update(&[], vec![(42, block(&[0, 3, 5], 1.))]).unwrap();
        check(&tree);
        let old = tree.incidence.clone();
        tree.update(&[], vec![(42, block(&[0, 3, 5], 2.))]).unwrap();
        check(&tree);
        assert!(Arc::ptr_eq(&old, &tree.incidence));
        tree.update(&[], vec![(u64::MAX, block(&[], 1.))]).unwrap();
        check(&tree);
        assert!(Arc::ptr_eq(&old, &tree.incidence));
        tree.update(&[], vec![(9, block(&[0, 1], 1.))]).unwrap();
        check(&tree);
        assert_eq!(tree.parent(0).unwrap(), Some(1));
        assert!(!Arc::ptr_eq(&old, &tree.incidence));
        for k in 0..6 {
            if old.children[k].as_ref() == tree.incidence.children[k].as_ref() {
                assert!(Arc::ptr_eq(&old.children[k], &tree.incidence.children[k]));
            }
            if old.factors[k].as_ref() == tree.incidence.factors[k].as_ref() {
                assert!(Arc::ptr_eq(&old.factors[k], &tree.incidence.factors[k]));
            }
        }
        for i in 0..12 {
            tree.update(
                &[],
                vec![(100 + i, block(&[(i as usize) % 5, 5], 1. + i as f64 * 0.1))],
            )
            .unwrap();
            check(&tree);
        }
        tree.update(&[], vec![(42, block(&[2, 4], 3.))]).unwrap();
        check(&tree);
        tree.update(&[9, 103, u64::MAX], vec![]).unwrap();
        check(&tree);
        let snapshot = tree.clone();
        let overflow = GaussianFactor::new(
            CsrMatrix::from_triplets(1, 6, vec![]).unwrap(),
            DVector::from_element(1, 1e308),
        )
        .unwrap();
        assert!(matches!(
            tree.update(&[], vec![(42, overflow)]),
            Err(Error::NonFinite)
        ));
        assert!(Arc::ptr_eq(&snapshot.incidence, &tree.incidence));
        assert_eq!(tree.solve().unwrap(), snapshot.solve().unwrap());
        check(&tree);
        tree.extend_columns(7).unwrap();
        check(&tree);
        for k in 0..6 {
            assert!(Arc::ptr_eq(
                &snapshot.incidence.children[k],
                &tree.incidence.children[k]
            ));
            assert!(Arc::ptr_eq(
                &snapshot.incidence.factors[k],
                &tree.incidence.factors[k]
            ));
        }
        tree.update(
            &[],
            vec![(
                200,
                GaussianFactor::new(
                    CsrMatrix::from_triplets(1, 7, vec![(0, 6, 1.)]).unwrap(),
                    DVector::from_element(1, -4.),
                )
                .unwrap(),
            )],
        )
        .unwrap();
        check(&tree);
        assert_eq!(tree.solve().unwrap()[6], 4.);
        assert_eq!(snapshot.dimension(), 6);
    }
    #[test]
    fn shared_metadata_detaches_only_on_change_and_failed_updates_preserve_snapshots() {
        let prior = |n: usize, column: usize, target: f64| {
            GaussianFactor::new(
                CsrMatrix::from_triplets(1, n, vec![(0, column, 1.)]).unwrap(),
                DVector::from_element(1, -target),
            )
            .unwrap()
        };
        let mut tree = EditableBayesTree::new(2, Default::default()).unwrap();
        tree.update(
            &[],
            vec![
                (0, prior(2, 0, 1.)),
                (1, prior(2, 1, 2.)),
                (
                    10,
                    GaussianFactor::new(
                        CsrMatrix::from_triplets(1, 2, vec![(0, 0, 1.), (0, 1, -1.)]).unwrap(),
                        DVector::from_element(1, -0.5),
                    )
                    .unwrap(),
                ),
            ],
        )
        .unwrap();
        let mut snapshot = tree.clone();
        let old_mean = snapshot.solve().unwrap();
        assert!((&old_mean - DVector::from_vec(vec![1.5, 1.5])).norm() < 1e-14);
        assert!(Arc::ptr_eq(&tree.factors, &snapshot.factors));
        assert!(Arc::ptr_eq(&tree.owners, &snapshot.owners));
        tree.extend_columns(3).unwrap();
        assert!(Arc::ptr_eq(&tree.factors, &snapshot.factors));
        assert!(Arc::ptr_eq(&tree.owners, &snapshot.owners));
        assert_eq!(snapshot.dimension(), 2);
        assert!(matches!(tree.solve(), Err(Error::RankDeficient { .. })));
        tree.update(&[], vec![(0, prior(3, 0, 4.))]).unwrap();
        assert!(!Arc::ptr_eq(&tree.factors, &snapshot.factors));
        assert!(Arc::ptr_eq(&tree.owners, &snapshot.owners));
        assert_eq!(snapshot.factor(0).unwrap().residual()[0], -1.);
        assert_eq!(snapshot.solve().unwrap(), old_mean);
        tree.update(&[], vec![(2, prior(3, 2, -1.))]).unwrap();
        assert!(!Arc::ptr_eq(&tree.owners, &snapshot.owners));
        assert!((tree.solve().unwrap() - DVector::from_vec(vec![3.5, 2.5, -1.])).norm() < 1e-14);
        let retained = tree.clone();
        // This reaches mutation of both private candidate maps before the
        // unowned constant factor overflows. Neither public map may detach.
        let overflow = GaussianFactor::new(
            CsrMatrix::from_triplets(1, 3, vec![]).unwrap(),
            DVector::from_element(1, 1e308),
        )
        .unwrap();
        assert!(matches!(
            tree.update(&[], vec![(0, overflow)]),
            Err(Error::NonFinite)
        ));
        assert!(Arc::ptr_eq(&tree.factors, &retained.factors));
        assert!(Arc::ptr_eq(&tree.owners, &retained.owners));
        assert!(Arc::ptr_eq(&tree.incidence, &retained.incidence));
        assert_eq!(tree.solve().unwrap(), retained.solve().unwrap());
        assert_eq!(tree.last_update(), retained.last_update());
        let old_owners = tree.owners.clone();
        tree.update(&[10], vec![]).unwrap();
        assert!(!Arc::ptr_eq(&tree.owners, &old_owners));
        assert!((tree.solve().unwrap() - DVector::from_vec(vec![4., 2., -1.])).norm() < 1e-14);
        assert_eq!(retained.factor_count(), 4);
        assert_eq!(snapshot.solve().unwrap(), old_mean);
        // An independent mutation of the original clone cannot affect either
        // already diverged model.
        snapshot.update(&[], vec![(1, prior(2, 1, 5.))]).unwrap();
        assert!((snapshot.solve().unwrap() - DVector::from_vec(vec![2.5, 3.5])).norm() < 1e-14);
        assert!((tree.solve().unwrap() - DVector::from_vec(vec![4., 2., -1.])).norm() < 1e-14);
        assert!(
            (retained.solve().unwrap() - DVector::from_vec(vec![3.5, 2.5, -1.])).norm() < 1e-14
        );
    }
    #[test]
    fn canonical_gather_matches_triplets_for_permuted_rows_and_boundary_messages() {
        let mut tree =
            EditableBayesTree::new_ordered(vec![3, 0, 5, 2, 1, 4], Default::default()).unwrap();
        for (id, factor) in [
            (
                7,
                GaussianFactor::new(
                    CsrMatrix::from_triplets(
                        3,
                        6,
                        vec![(0, 0, 2.), (0, 1, 3.), (0, 5, -1.), (2, 4, 1e-150)],
                    )
                    .unwrap(),
                    DVector::from_vec(vec![0.4, -0., 3.]),
                )
                .unwrap(),
            ),
            (
                8,
                GaussianFactor::new(
                    CsrMatrix::from_triplets(1, 6, vec![(0, 1, 1e150), (0, 5, -4.)]).unwrap(),
                    DVector::from_vec(vec![-7.]),
                )
                .unwrap(),
            ),
        ] {
            Arc::make_mut(&mut tree.factors).insert(id, Arc::new(factor));
        }
        tree.messages[0] = Arc::new(Message {
            rows: vec![vec![(2, 4.), (4, -2.)], vec![(4, 8.), (5, 1.)]],
            rhs: vec![0., 2.],
            constant_cost: 0.75,
        });
        let columns = [1, 2, 4, 5];
        let local = columns
            .iter()
            .enumerate()
            .map(|(index, &column)| (column, index))
            .collect::<BTreeMap<_, _>>();
        let mut triplets = Vec::new();
        let mut residual = Vec::new();
        for id in [7, 8] {
            let factor = &tree.factors[&id];
            let start = residual.len();
            for (row, column, value) in factor.jacobian.entries() {
                triplets.push((start + row, local[&tree.tree.to_internal[column]], value));
            }
            residual.extend_from_slice(factor.residual.as_slice());
        }
        for (row, rhs) in tree.messages[0].rows.iter().zip(&tree.messages[0].rhs) {
            for &(column, value) in row {
                triplets.push((residual.len(), local[&column], value));
            }
            residual.push(-rhs);
        }
        let rows = residual.len();
        let entries = triplets.len() + rows;
        let expected = CsrMatrix::from_triplets(rows, columns.len(), triplets).unwrap();
        let (actual, r, constant) = tree
            .gather_clique(&columns, &[7, 8], &[0], rows, entries)
            .unwrap();
        assert_eq!(actual.pattern(), expected.pattern());
        assert_eq!(actual.values(), expected.values());
        assert_eq!(
            r.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            residual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(constant, 0.75);
        assert!(matches!(
            tree.gather_clique(&columns[..3], &[7, 8], &[0], rows, entries),
            Err(Error::InvalidDimensions)
        ));
        let (empty, residual, constant) = tree.gather_clique(&[], &[], &[], 0, 0).unwrap();
        assert_eq!(empty.shape(), (0, 0));
        assert!(residual.is_empty());
        assert_eq!(constant, 0.);
    }
    #[test]
    fn orphan_conditionals_and_nonzero_separator_messages_share_identity() {
        let leaf = |column: usize, target: f64| {
            GaussianFactor::new(
                CsrMatrix::from_triplets(2, 4, vec![(0, column, 1.), (1, column, 1.), (1, 3, -1.)])
                    .unwrap(),
                DVector::from_vec(vec![-target, -0.5]),
            )
            .unwrap()
        };
        let mut tree = EditableBayesTree::new(4, Default::default()).unwrap();
        tree.update(
            &[],
            vec![(0, leaf(0, 1.)), (1, leaf(1, 2.)), (2, leaf(2, 3.))],
        )
        .unwrap();
        let conditional = tree.tree.conditionals[1].clone();
        let message = tree.messages[1].clone();
        let factor = tree.factors[&1].clone();
        assert!(!message.rows.is_empty());
        for root in [-2., 0., 4.] {
            let actual = message
                .rows
                .iter()
                .zip(&message.rhs)
                .map(|(row, rhs)| {
                    0.5 * (row
                        .iter()
                        .map(|&(col, a)| {
                            assert_eq!(col, 3);
                            a * root
                        })
                        .sum::<f64>()
                        - rhs)
                        .powi(2)
                })
                .sum::<f64>()
                + message.constant_cost;
            assert!((actual - 0.25 * (root + 0.5 - 2.).powi(2)).abs() < 1e-13);
        }
        tree.update(&[], vec![(0, leaf(0, 4.))]).unwrap();
        assert!(Arc::ptr_eq(&conditional, &tree.tree.conditionals[1]));
        assert!(Arc::ptr_eq(&message, &tree.messages[1]));
        assert!(Arc::ptr_eq(&factor, &tree.factors[&1]));
    }
    #[test]
    fn incremental_symbolic_tails_match_full_pruning_and_reuse_unchanged_supports() {
        let mut tree =
            EditableBayesTree::new_ordered(vec![4, 2, 7, 0, 8, 5, 3, 6, 1], Default::default())
                .unwrap();
        let permutation = tree.column_permutation().to_vec();
        let block = |columns: &[usize], scale: f64| {
            GaussianFactor::new(
                CsrMatrix::from_triplets(
                    1,
                    9,
                    columns
                        .iter()
                        .map(|&column| (0, permutation[column], scale))
                        .collect(),
                )
                .unwrap(),
                DVector::from_element(1, -1.),
            )
            .unwrap()
        };
        let check = |tree: &EditableBayesTree| {
            let (separators, parents, owners, count) = tree.topology().unwrap();
            assert_eq!(tree.tree.separators, separators);
            assert_eq!(tree.tree.parents, parents);
            assert_eq!(tree.owners, owners);
            assert_eq!(tree.tree.symbolic_entries, count);
        };
        tree.update(&[], vec![(1, block(&[0, 5, 8], 1.))]).unwrap();
        check(&tree);
        tree.update(&[], vec![(2, block(&[0, 3], 1.))]).unwrap();
        check(&tree);
        assert_eq!(tree.parent(0).unwrap(), Some(3));
        assert_eq!(tree.separator(3).unwrap(), &[5, 8]);
        let old = tree.tree.separators.clone();
        tree.update(&[], vec![(2, block(&[0, 3], 7.))]).unwrap();
        check(&tree);
        for (a, b) in old.iter().zip(&tree.tree.separators) {
            assert!(Arc::ptr_eq(a, b));
        }
        for index in 0..20 {
            let a = index % 7;
            let b = 1 + (index * 3) % 8;
            tree.update(
                &[],
                vec![(10 + index as u64, block(&[a, b], 1. + index as f64 * 0.1))],
            )
            .unwrap();
            check(&tree);
        }
        tree.update(&[1], vec![]).unwrap();
        check(&tree);
        tree.update(&[], vec![(2, block(&[8], 1.))]).unwrap();
        check(&tree);
    }
}
