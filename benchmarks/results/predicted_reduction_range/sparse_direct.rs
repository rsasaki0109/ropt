//! Sparse symmetric LDL^T of damped normal equations. No pivoting, inverse or jitter.
use crate::sparse::{CsrMatrix, LinearOperator, NormalOperator, checked_norm, finite_vector};
use crate::{DVector, Error};
use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SparseOrdering {
    Natural,
    /// Deterministic greedy minimum degree on the evolving scalar elimination graph.
    /// This is not approximate minimum degree or COLAMD.
    #[default]
    MinimumDegree,
    /// Greedy minimum degree on contiguous equal-size variable blocks, keeping
    /// each block's scalar coordinates together. Dimension must be divisible.
    BlockMinimumDegree {
        block_size: usize,
    },
}
#[derive(Debug, Clone, Copy)]
pub struct SparseLdlOptions {
    pub ordering: SparseOrdering,
    /// Maximum stored lower-triangle entries including diagonal and structural zeros.
    /// Applies to normal assembly and factor fill; not a byte or RSS limit.
    pub max_entries: usize,
    pub relative_tolerance: f64,
    pub absolute_tolerance: f64,
}
impl Default for SparseLdlOptions {
    fn default() -> Self {
        Self {
            ordering: SparseOrdering::MinimumDegree,
            max_entries: 1_000_000,
            relative_tolerance: 1e-10,
            absolute_tolerance: 1e-12,
        }
    }
}
impl SparseLdlOptions {
    pub(crate) fn validate(self) -> Result<(), Error> {
        if self.max_entries == 0
            || matches!(
                self.ordering,
                SparseOrdering::BlockMinimumDegree { block_size: 0 }
            )
            || [self.relative_tolerance, self.absolute_tolerance]
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(Error::InvalidOptions);
        }
        Ok(())
    }
}
// Keep short structural columns contiguous; bound insertion shifts for large
// columns by promoting to a tree. Neither representation stores numeric zeros.
enum PatternColumn {
    Small(Vec<usize>),
    Large(BTreeSet<usize>),
}
impl PatternColumn {
    fn diagonal(row: usize) -> Self {
        let mut rows = Vec::with_capacity(8);
        rows.push(row);
        Self::Small(rows)
    }
    fn iter(&self) -> impl Iterator<Item = &usize> {
        let (small, large) = match self {
            Self::Small(rows) => (Some(rows), None),
            Self::Large(rows) => (None, Some(rows)),
        };
        small
            .into_iter()
            .flatten()
            .chain(large.into_iter().flatten())
    }
    // Adjacency operations use the same ascending hybrid storage as patterns.
    // Their logical edge budget remains checked by minimum_degree before fill.
    fn len(&self) -> usize {
        match self {
            Self::Small(rows) => rows.len(),
            Self::Large(rows) => rows.len(),
        }
    }
    fn contains(&self, row: &usize) -> bool {
        match self {
            Self::Small(rows) => rows.binary_search(row).is_ok(),
            Self::Large(rows) => rows.contains(row),
        }
    }
    fn insert_neighbor(&mut self, row: usize) {
        match self {
            Self::Small(rows) => {
                if let Err(position) = rows.binary_search(&row) {
                    if rows.len() < 64 {
                        rows.insert(position, row);
                    } else {
                        let mut tree = rows.iter().copied().collect::<BTreeSet<_>>();
                        tree.insert(row);
                        *self = Self::Large(tree);
                    }
                }
            }
            Self::Large(rows) => {
                rows.insert(row);
            }
        }
    }
    fn remove(&mut self, row: &usize) {
        match self {
            Self::Small(rows) => {
                if let Ok(position) = rows.binary_search(row) {
                    rows.remove(position);
                }
            }
            Self::Large(rows) => {
                rows.remove(row);
            }
        }
    }
    fn clear(&mut self) {
        match self {
            Self::Small(rows) => rows.clear(),
            Self::Large(rows) => rows.clear(),
        }
    }
    fn into_rows(self) -> Vec<usize> {
        match self {
            Self::Small(rows) => rows,
            Self::Large(rows) => rows.into_iter().collect(),
        }
    }
}
type Columns = Vec<PatternColumn>;
fn add_pattern_entry(
    column: &mut PatternColumn,
    row: usize,
    entries: &mut usize,
    limit: usize,
) -> Result<(), Error> {
    match column {
        PatternColumn::Small(rows) => {
            let position = match rows.binary_search(&row) {
                Ok(_) => return Ok(()),
                Err(position) => position,
            };
            if *entries >= limit {
                return Err(Error::SparseFillLimit { limit });
            }
            if rows.len() < 64 {
                rows.insert(position, row);
            } else {
                let mut tree = rows.iter().copied().collect::<BTreeSet<_>>();
                tree.insert(row);
                *column = PatternColumn::Large(tree);
            }
        }
        PatternColumn::Large(rows) => {
            if rows.contains(&row) {
                return Ok(());
            }
            if *entries >= limit {
                return Err(Error::SparseFillLimit { limit });
            }
            rows.insert(row);
        }
    }
    *entries += 1;
    Ok(())
}
// Insert a strictly ascending structural suffix with an advancing cursor.
// Fill suffixes start at the existing diagonal of the destination column.
fn add_pattern_entries(
    column: &mut PatternColumn,
    sorted: &[usize],
    entries: &mut usize,
    limit: usize,
) -> Result<(), Error> {
    if sorted.is_empty() {
        return Ok(());
    }
    match column {
        PatternColumn::Small(rows) => {
            let mut position = 0;
            for (index, &row) in sorted.iter().enumerate() {
                while position < rows.len() && rows[position] < row {
                    position += 1;
                }
                if position < rows.len() && rows[position] == row {
                    position += 1;
                    continue;
                }
                if *entries >= limit {
                    return Err(Error::SparseFillLimit { limit });
                }
                if rows.len() < 64 {
                    rows.insert(position, row);
                    position += 1;
                    *entries += 1;
                } else {
                    let mut tree = rows.iter().copied().collect::<BTreeSet<_>>();
                    tree.insert(row);
                    *entries += 1;
                    *column = PatternColumn::Large(tree);
                    return add_pattern_entries(column, &sorted[index + 1..], entries, limit);
                }
            }
        }
        PatternColumn::Large(_) => {
            for &row in sorted {
                add_pattern_entry(column, row, entries, limit)?;
            }
        }
    }
    Ok(())
}
fn assemble_pattern(j: &CsrMatrix, limit: usize) -> Result<(Columns, usize), Error> {
    let (m, n) = j.shape();
    if n == 0 {
        return Err(Error::InvalidDimensions);
    }
    if n > limit {
        return Err(Error::SparseFillLimit { limit });
    }
    let mut columns = (0..n).map(PatternColumn::diagonal).collect::<Columns>();
    let mut entries = n;
    let (offsets, indices) = j.pattern();
    for r in 0..m {
        let row = &indices[offsets[r]..offsets[r + 1]];
        // Repeated supports introduce no new edges. Numeric values still
        // contribute separately, in their original row order during refactor.
        if r > 0 && row == &indices[offsets[r - 1]..offsets[r]] {
            continue;
        }
        for (k, &a) in row.iter().enumerate() {
            for &b in &row[k..] {
                add_pattern_entry(&mut columns[a], b, &mut entries, limit)?;
            }
        }
    }
    Ok((columns, entries))
}
fn minimum_degree(
    columns: &Columns,
    mut entries: usize,
    limit: usize,
) -> Result<Vec<usize>, Error> {
    let n = columns.len();
    let mut adjacent = (0..n)
        .map(|_| PatternColumn::Small(Vec::new()))
        .collect::<Columns>();
    for (col, values) in columns.iter().enumerate() {
        for &row in values.iter() {
            if row != col {
                adjacent[col].insert_neighbor(row);
                adjacent[row].insert_neighbor(col);
            }
        }
    }
    let mut active = vec![true; n];
    let mut heap = (0..n)
        .map(|k| Reverse((adjacent[k].len(), k)))
        .collect::<BinaryHeap<_>>();
    let mut order = Vec::with_capacity(n);
    let mut neighbors = Vec::new();
    while order.len() < n {
        let key = loop {
            let Reverse((degree, key)) = heap.pop().ok_or(Error::LinearSolve)?;
            if active[key] && degree == adjacent[key].len() {
                break key;
            }
        };
        active[key] = false;
        order.push(key);
        neighbors.clear();
        neighbors.extend(adjacent[key].iter().copied());
        for &other in &neighbors {
            adjacent[other].remove(&key);
        }
        for (i, &a) in neighbors.iter().enumerate() {
            for &b in &neighbors[i + 1..] {
                if !adjacent[a].contains(&b) {
                    if entries >= limit {
                        return Err(Error::SparseFillLimit { limit });
                    }
                    adjacent[a].insert_neighbor(b);
                    adjacent[b].insert_neighbor(a);
                    entries += 1;
                }
            }
        }
        adjacent[key].clear();
        for &other in &neighbors {
            heap.push(Reverse((adjacent[other].len(), other)));
        }
    }
    Ok(order)
}
/// Reusable symbolic analysis for an exact stored CSR pattern. Numeric values
/// and damping may change; any offset/index/shape change requires new analysis.
#[derive(Debug)]
pub struct SparseLdlSymbolic {
    shape: (usize, usize),
    pattern_offsets: Vec<usize>,
    pattern_indices: Vec<usize>,
    permutation: Vec<usize>,
    inverse: Vec<usize>,
    offsets: Vec<usize>,
    rows: Vec<usize>,
    normal_entries: usize,
    normal_offsets: Vec<usize>,
    normal_rows: Vec<usize>,
    max_off_diagonal: usize,
    update_targets: Option<Vec<usize>>,
    assembly_targets: Option<Vec<usize>>,
}
pub(crate) fn column_ordering(
    j: &CsrMatrix,
    options: &SparseLdlOptions,
) -> Result<Vec<usize>, Error> {
    options.validate()?;
    if let SparseOrdering::BlockMinimumDegree { block_size } = options.ordering {
        let (m, n) = j.shape();
        if n != 0 && n <= options.max_entries && n.is_multiple_of(block_size) {
            let (offsets, indices) = j.pattern();
            // Prove the scalar assembly budget without constructing its graph.
            // Count all row pairs conservatively, including duplicate supports.
            let upper = (0..m).try_fold(n, |sum, r| {
                let width = offsets[r + 1] - offsets[r];
                sum.checked_add(width.checked_mul(width.saturating_sub(1))? / 2)
            });
            if upper.is_some_and(|count| count <= options.max_entries) {
                let blocks = n / block_size;
                let mut coarse = (0..blocks)
                    .map(PatternColumn::diagonal)
                    .collect::<Columns>();
                let mut entries = blocks;
                let mut support = Vec::new();
                for r in 0..m {
                    support.clear();
                    for &column in &indices[offsets[r]..offsets[r + 1]] {
                        let block = column / block_size;
                        if support.last() != Some(&block) {
                            support.push(block);
                        }
                    }
                    for (k, &a) in support.iter().enumerate() {
                        for &b in &support[k + 1..] {
                            add_pattern_entry(
                                &mut coarse[a],
                                b,
                                &mut entries,
                                options.max_entries,
                            )?;
                        }
                    }
                }
                return Ok(minimum_degree(&coarse, entries, options.max_entries)?
                    .into_iter()
                    .flat_map(|block| {
                        (0..block_size).map(move |offset| block * block_size + offset)
                    })
                    .collect());
            }
        }
    }
    // A loose/overflowed bound does not reject otherwise eligible sparse input:
    // retain exact scalar counting and all its original budget/error behavior.
    let (columns, entries) = assemble_pattern(j, options.max_entries)?;
    choose_ordering(&columns, entries, options)
}
fn choose_ordering(
    columns: &Columns,
    normal_entries: usize,
    options: &SparseLdlOptions,
) -> Result<Vec<usize>, Error> {
    let n = columns.len();
    Ok(match options.ordering {
        SparseOrdering::Natural => (0..n).collect(),
        SparseOrdering::MinimumDegree => {
            minimum_degree(columns, normal_entries, options.max_entries)?
        }
        SparseOrdering::BlockMinimumDegree { block_size } => {
            if !n.is_multiple_of(block_size) {
                return Err(Error::InvalidDimensions);
            }
            let blocks = n / block_size;
            let mut coarse = (0..blocks)
                .map(PatternColumn::diagonal)
                .collect::<Columns>();
            let mut entries = blocks;
            for (col, values) in columns.iter().enumerate() {
                for &row in values.iter() {
                    add_pattern_entry(
                        &mut coarse[col / block_size],
                        row / block_size,
                        &mut entries,
                        options.max_entries,
                    )?;
                }
            }
            minimum_degree(&coarse, entries, options.max_entries)?
                .into_iter()
                .flat_map(|block| (0..block_size).map(move |offset| block * block_size + offset))
                .collect()
        }
    })
}
impl SparseLdlSymbolic {
    pub fn analyze(j: &CsrMatrix, options: &SparseLdlOptions) -> Result<Self, Error> {
        options.validate()?;
        let (columns, normal_entries) = assemble_pattern(j, options.max_entries)?;
        let n = columns.len();
        let permutation = choose_ordering(&columns, normal_entries, options)?;
        // Exact original-coordinate normal graph, independent of residual row
        // layout and numerical values. Keep equality checks collision-free.
        let mut normal_offsets = Vec::with_capacity(n + 1);
        let mut normal_rows = Vec::with_capacity(normal_entries);
        normal_offsets.push(0);
        for column in &columns {
            normal_rows.extend(column.iter().copied());
            normal_offsets.push(normal_rows.len());
        }
        let mut inverse = vec![0; n];
        for (new, &old) in permutation.iter().enumerate() {
            inverse[old] = new;
        }
        let mut work = (0..n)
            .map(|_| PatternColumn::Small(Vec::with_capacity(8)))
            .collect::<Columns>();
        let mut entries = 0;
        for (old_col, column) in columns.into_iter().enumerate() {
            for &old_row in column.iter() {
                let (a, b) = (inverse[old_col], inverse[old_row]);
                add_pattern_entry(
                    &mut work[a.min(b)],
                    a.max(b),
                    &mut entries,
                    options.max_entries,
                )?;
            }
        }
        let mut offsets = Vec::with_capacity(n + 1);
        let mut rows = Vec::with_capacity(normal_entries);
        let mut max_off_diagonal = 0;
        offsets.push(0);
        for k in 0..n {
            let column =
                std::mem::replace(&mut work[k], PatternColumn::Small(Vec::new())).into_rows();
            if column.first() != Some(&k) {
                return Err(Error::InvalidDimensions);
            }
            max_off_diagonal = max_off_diagonal.max(column.len() - 1);
            for (i, &a) in column[1..].iter().enumerate() {
                add_pattern_entries(
                    &mut work[a],
                    &column[i + 1..],
                    &mut entries,
                    options.max_entries,
                )?;
            }
            rows.extend(column);
            offsets.push(rows.len());
        }
        let (pattern_offsets, pattern_indices) = j.pattern();
        let mut symbolic = Self {
            shape: j.shape(),
            pattern_offsets: pattern_offsets.to_vec(),
            pattern_indices: pattern_indices.to_vec(),
            permutation,
            inverse,
            offsets,
            rows,
            normal_entries,
            normal_offsets,
            normal_rows,
            max_off_diagonal,
            update_targets: None,
            assembly_targets: None,
        };
        // Bound optional operation-index storage independently by the same entry
        // budget. Dense/high-fill factors can fall back to binary searches.
        let operations = (0..n).try_fold(0_usize, |sum, k| {
            let off = symbolic.offsets[k + 1] - symbolic.offsets[k] - 1;
            sum.checked_add(off.checked_mul(off.checked_add(1)?)? / 2)
        });
        if let Some(operations) = operations.filter(|&count| count <= options.max_entries) {
            let mut targets = Vec::with_capacity(operations);
            for k in 0..n {
                for i in symbolic.offsets[k] + 1..symbolic.offsets[k + 1] {
                    let column = symbolic.rows[i];
                    let start = symbolic.offsets[column];
                    let destinations = &symbolic.rows[start..symbolic.offsets[column + 1]];
                    // The first Schur destination is this column's diagonal.
                    debug_assert_eq!(destinations.first(), Some(&column));
                    targets.push(start);
                    let mut position = 1;
                    for &row in &symbolic.rows[i + 1..symbolic.offsets[k + 1]] {
                        // Common clique suffixes are contiguous. When another
                        // row intervenes, search only the remaining suffix.
                        if destinations.get(position) != Some(&row) {
                            position += destinations[position..]
                                .binary_search(&row)
                                .expect("symbolic fill contains every Schur entry");
                        }
                        targets.push(start + position);
                        position += 1;
                    }
                }
            }
            symbolic.update_targets = Some(targets);
        }
        symbolic.refresh_assembly(j, options.max_entries);
        Ok(symbolic)
    }
    // A different CSR descriptor can share the exact same normal graph. Only
    // that equality permits reusing ordering, fill and numeric update targets.
    fn rebind_pattern(&mut self, j: &CsrMatrix, limit: usize) -> Result<bool, Error> {
        if j.shape().1 != self.dimension() {
            return Ok(false);
        }
        let (columns, entries) = assemble_pattern(j, limit)?;
        if entries != self.normal_entries
            || columns.iter().enumerate().any(|(col, rows)| {
                rows.iter().copied().ne(self.normal_rows
                    [self.normal_offsets[col]..self.normal_offsets[col + 1]]
                    .iter()
                    .copied())
            })
        {
            return Ok(false);
        }
        self.refresh_assembly(j, limit);
        let (offsets, indices) = j.pattern();
        self.pattern_offsets.clear();
        self.pattern_offsets.extend_from_slice(offsets);
        self.pattern_indices.clear();
        self.pattern_indices.extend_from_slice(indices);
        self.shape = j.shape();
        Ok(true)
    }
    fn refresh_assembly(&mut self, j: &CsrMatrix, limit: usize) {
        let (pattern_offsets, pattern_indices) = j.pattern();
        let remaining = limit - self.update_entries();
        let operations = (0..j.shape().0).try_fold(0_usize, |sum, row| {
            let count = pattern_offsets[row + 1] - pattern_offsets[row];
            sum.checked_add(count.checked_mul(count.checked_add(1)?)? / 2)
        });
        if let Some(operations) = operations.filter(|&count| count <= remaining) {
            let mut targets = self.assembly_targets.take().unwrap_or_default();
            targets.clear();
            targets.reserve_exact(operations);
            let mut previous_start = 0;
            for row in 0..j.shape().0 {
                let start = targets.len();
                let support = &pattern_indices[pattern_offsets[row]..pattern_offsets[row + 1]];
                if row > 0
                    && support == &pattern_indices[pattern_offsets[row - 1]..pattern_offsets[row]]
                {
                    // Each residual retains its complete plan and operation budget;
                    // only identical column-to-destination lookups are copied.
                    targets.extend_from_within(previous_start..start);
                } else {
                    for a in pattern_offsets[row]..pattern_offsets[row + 1] {
                        let ca = self.inverse[pattern_indices[a]];
                        for &original in &pattern_indices[a..pattern_offsets[row + 1]] {
                            let cb = self.inverse[original];
                            targets.push(self.entry(ca.min(cb), ca.max(cb)));
                        }
                    }
                }
                previous_start = start;
            }
            self.assembly_targets = Some(targets);
        } else {
            self.assembly_targets = None;
        }
    }
    pub fn dimension(&self) -> usize {
        self.permutation.len()
    }
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }
    pub fn normal_entries(&self) -> usize {
        self.normal_entries
    }
    pub fn factor_entries(&self) -> usize {
        self.rows.len()
    }
    /// Cached Schur-update destinations. Optional plan storage is at most max_entries;
    /// larger operation counts use searches without changing the factor or arithmetic.
    pub fn update_entries(&self) -> usize {
        self.update_targets.as_ref().map_or(0, Vec::len)
    }
    pub fn has_update_plan(&self) -> bool {
        self.update_targets.is_some()
    }
    pub fn assembly_entries(&self) -> usize {
        self.assembly_targets.as_ref().map_or(0, Vec::len)
    }
    pub fn matches_pattern(&self, j: &CsrMatrix) -> bool {
        let (offsets, indices) = j.pattern();
        self.shape == j.shape()
            && self.pattern_offsets == offsets
            && self.pattern_indices == indices
    }
    fn entry(&self, column: usize, row: usize) -> usize {
        let start = self.offsets[column];
        start
            + self.rows[start..self.offsets[column + 1]]
                .binary_search(&row)
                .expect("symbolic fill contains every normal and Schur entry")
    }
    /// Construct an independently owned numeric factor with this symbolic analysis.
    /// For repeated numeric buffer reuse, use SparseLdlWorkspace instead.
    pub fn factor_normal(&self, j: &CsrMatrix, damping: f64) -> Result<SparseLdlFactor, Error> {
        let mut values = vec![0.0; self.factor_entries()];
        let mut scratch = vec![0.0; self.max_off_diagonal];
        factor_numeric(self, j, damping, &mut values, &mut scratch)?;
        let mut lower = Vec::with_capacity(self.dimension());
        let mut diagonal = DVector::zeros(self.dimension());
        for k in 0..self.dimension() {
            let start = self.offsets[k];
            diagonal[k] = values[start];
            lower.push(
                (start + 1..self.offsets[k + 1])
                    .map(|i| (self.rows[i], values[i]))
                    .collect(),
            );
        }
        Ok(SparseLdlFactor {
            lower,
            diagonal,
            permutation: self.permutation.clone(),
            normal_entries: self.normal_entries,
        })
    }
}
fn factor_numeric(
    symbolic: &SparseLdlSymbolic,
    j: &CsrMatrix,
    damping: f64,
    values: &mut [f64],
    scratch: &mut [f64],
) -> Result<(), Error> {
    if !symbolic.matches_pattern(j) {
        return Err(Error::Evaluation(
            "sparse LDL symbolic pattern mismatch".into(),
        ));
    }
    if !damping.is_finite() || damping < 0.0 {
        return Err(Error::InvalidOptions);
    }
    values.fill(0.0);
    let (offsets, indices) = j.pattern();
    let numeric = j.values();
    let mut assembly_index = 0;
    for row in 0..j.shape().0 {
        for a in offsets[row]..offsets[row + 1] {
            let ca = symbolic.inverse[indices[a]];
            for b in a..offsets[row + 1] {
                let cb = symbolic.inverse[indices[b]];
                let target = if let Some(targets) = &symbolic.assembly_targets {
                    targets[assembly_index]
                } else {
                    symbolic.entry(ca.min(cb), ca.max(cb))
                };
                assembly_index += 1;
                values[target] += numeric[a] * numeric[b];
                if !values[target].is_finite() {
                    return Err(Error::NonFinite);
                }
            }
        }
    }
    let mut update_index = 0;
    for k in 0..symbolic.dimension() {
        let diagonal = &mut values[symbolic.offsets[k]];
        *diagonal += damping * diagonal.max(1.0);
        if !diagonal.is_finite() {
            return Err(Error::NonFinite);
        }
    }
    for k in 0..symbolic.dimension() {
        let (start, end) = (symbolic.offsets[k], symbolic.offsets[k + 1]);
        let pivot = values[start];
        if !pivot.is_finite() {
            return Err(Error::NonFinite);
        }
        if pivot <= 0.0 {
            return Err(Error::LinearSolve);
        }
        let count = end - start - 1;
        scratch[..count].copy_from_slice(&values[start + 1..end]);
        for value in &mut values[start + 1..end] {
            *value /= pivot;
            if !value.is_finite() {
                return Err(Error::NonFinite);
            }
        }
        // Preserve the previous right-looking arithmetic order: l_i * original a_jk.
        for i in start + 1..end {
            let column = symbolic.rows[i];
            for q in i..end {
                let target = if let Some(targets) = &symbolic.update_targets {
                    targets[update_index]
                } else {
                    symbolic.entry(column, symbolic.rows[q])
                };
                update_index += 1;
                values[target] += -values[i] * scratch[q - start - 1];
                if !values[target].is_finite() {
                    return Err(Error::NonFinite);
                }
            }
        }
    }
    Ok(())
}
/// Unit lower-triangle columns, diagonal pivots and original-coordinate permutation.
/// Construction rejects nonpositive pivots. This does not provide indefinite pivoting.
#[derive(Debug)]
pub struct SparseLdlFactor {
    lower: Vec<Vec<(usize, f64)>>,
    diagonal: DVector<f64>,
    permutation: Vec<usize>,
    normal_entries: usize,
}
impl SparseLdlFactor {
    pub fn factor_normal(
        j: &CsrMatrix,
        damping: f64,
        options: &SparseLdlOptions,
    ) -> Result<Self, Error> {
        SparseLdlSymbolic::analyze(j, options)?.factor_normal(j, damping)
    }
    pub fn dimension(&self) -> usize {
        self.diagonal.len()
    }
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }
    pub fn normal_entries(&self) -> usize {
        self.normal_entries
    }
    pub fn factor_entries(&self) -> usize {
        self.dimension() + self.lower.iter().map(Vec::len).sum::<usize>()
    }
    /// Triangular solve in original coordinates. For a checked original-system
    /// residual, use solve_normal; this method alone solves the computed factors.
    pub fn solve(&self, rhs: &DVector<f64>) -> Result<DVector<f64>, Error> {
        if rhs.len() != self.dimension() {
            return Err(Error::InvalidDimensions);
        }
        finite_vector(rhs)?;
        let mut y =
            DVector::from_iterator(self.dimension(), self.permutation.iter().map(|&k| rhs[k]));
        for (k, column) in self.lower.iter().enumerate() {
            let value = y[k];
            for &(row, coefficient) in column {
                y[row] -= coefficient * value;
            }
        }
        for k in 0..self.dimension() {
            y[k] /= self.diagonal[k];
        }
        for (k, column) in self.lower.iter().enumerate().rev() {
            for &(row, coefficient) in column {
                y[k] -= coefficient * y[row];
            }
        }
        finite_vector(&y)?;
        let mut x = DVector::zeros(self.dimension());
        for (k, &old) in self.permutation.iter().enumerate() {
            x[old] = y[k];
        }
        Ok(x)
    }
}
#[derive(Debug)]
pub struct SparseLdlReport {
    pub x: DVector<f64>,
    pub residual_norm: f64,
    /// Corrections using the computed factors and original-operator residual (0..=3).
    pub refinement_steps: usize,
    pub normal_entries: usize,
    pub factor_entries: usize,
}
/// Counts of one checked solve; entry counts include diagonals/structural zeros.
#[derive(Debug, Clone, Copy)]
pub struct SparseLdlStats {
    pub residual_norm: f64,
    pub refinement_steps: usize,
    pub normal_entries: usize,
    pub factor_entries: usize,
}
/// Symbolic analysis and numeric scratch reused across linear solves. Exact CSR
/// pattern changes trigger checked pattern analysis. Equal normal graphs reuse
/// ordering/fill with refreshed CSR assembly targets. No global/shared cache.
#[derive(Debug)]
pub struct SparseLdlWorkspace {
    options: SparseLdlOptions,
    symbolic: Option<SparseLdlSymbolic>,
    values: Vec<f64>,
    column_scratch: Vec<f64>,
    solution: DVector<f64>,
    residual: DVector<f64>,
    analyses: usize,
    ready: bool,
}
impl SparseLdlWorkspace {
    pub fn new(options: SparseLdlOptions) -> Result<Self, Error> {
        options.validate()?;
        Ok(Self {
            options,
            symbolic: None,
            values: Vec::new(),
            column_scratch: Vec::new(),
            solution: DVector::zeros(0),
            residual: DVector::zeros(0),
            analyses: 0,
            ready: false,
        })
    }
    /// Successful CSR-pattern analyses, including normal-graph rebindings.
    /// Value-only refactors do not increment this counter.
    pub fn analyses(&self) -> usize {
        self.analyses
    }
    pub fn symbolic(&self) -> Option<&SparseLdlSymbolic> {
        self.symbolic.as_ref()
    }
    /// Reuse ordering/fill and numeric buffers when the stored pattern is identical.
    /// A failed refactor invalidates the numeric factor until the next successful call.
    pub fn refactor(&mut self, j: &CsrMatrix, damping: f64) -> Result<(), Error> {
        self.ready = false;
        if !damping.is_finite() || damping < 0.0 {
            return Err(Error::InvalidOptions);
        }
        if self.symbolic.as_ref().is_none_or(|s| !s.matches_pattern(j)) {
            let rebound = if let Some(symbolic) = &mut self.symbolic {
                symbolic.rebind_pattern(j, self.options.max_entries)?
            } else {
                false
            };
            if !rebound {
                let symbolic = SparseLdlSymbolic::analyze(j, &self.options)?;
                self.values.resize(symbolic.factor_entries(), 0.0);
                self.column_scratch.resize(symbolic.max_off_diagonal, 0.0);
                self.solution
                    .resize_vertically_mut(symbolic.dimension(), 0.0);
                self.residual
                    .resize_vertically_mut(symbolic.dimension(), 0.0);
                self.symbolic = Some(symbolic);
            }
            self.analyses = self.analyses.saturating_add(1);
        }
        factor_numeric(
            self.symbolic.as_ref().ok_or(Error::LinearSolve)?,
            j,
            damping,
            &mut self.values,
            &mut self.column_scratch,
        )?;
        self.ready = true;
        Ok(())
    }
    /// Solve the computed factors into caller storage, without checking the original
    /// normal system. Use solve_into or solve for the true-residual check.
    pub fn solve_factored_into(
        &mut self,
        rhs: &DVector<f64>,
        x: &mut DVector<f64>,
    ) -> Result<(), Error> {
        if !self.ready {
            return Err(Error::LinearSolve);
        }
        let s = self.symbolic.as_ref().ok_or(Error::LinearSolve)?;
        if rhs.len() != s.dimension() || x.len() != s.dimension() {
            return Err(Error::InvalidDimensions);
        }
        factored_solution(s, &self.values, rhs, &mut self.solution)?;
        for (k, &old) in s.permutation.iter().enumerate() {
            x[old] = self.solution[k];
        }
        Ok(())
    }
    /// Refactor, solve and verify the true rhs-A*x residual with the original CSR
    /// operator. Up to three factor-based corrections are attempted only when this
    /// residual exceeds the requested tolerance; failure still returns an error.
    /// Numeric buffers are reused; constructing its diagonal still allocates.
    pub fn solve_into(
        &mut self,
        j: &CsrMatrix,
        rhs: &DVector<f64>,
        damping: f64,
        x: &mut DVector<f64>,
    ) -> Result<SparseLdlStats, Error> {
        if rhs.len() != j.shape().1 || x.len() != rhs.len() {
            return Err(Error::InvalidDimensions);
        }
        finite_vector(rhs)?;
        self.refactor(j, damping)?;
        self.solve_factored_into(rhs, x)?;
        let operator = NormalOperator::new(j, damping)?;
        let rhs_norm = checked_norm(rhs)?;
        let tolerance = self
            .options
            .absolute_tolerance
            .max(self.options.relative_tolerance * rhs_norm);
        if !tolerance.is_finite() {
            return Err(Error::InvalidOptions);
        }
        let s = self.symbolic.as_ref().ok_or(Error::LinearSolve)?;
        for refinement_steps in 0..=3 {
            operator.apply_into(x, &mut self.residual)?;
            for k in 0..rhs.len() {
                self.residual[k] = rhs[k] - self.residual[k];
            }
            let residual_norm = checked_norm(&self.residual)?;
            if residual_norm <= tolerance {
                return Ok(SparseLdlStats {
                    residual_norm,
                    refinement_steps,
                    normal_entries: s.normal_entries(),
                    factor_entries: s.factor_entries(),
                });
            }
            if refinement_steps == 3 {
                return Err(Error::LinearResidualTooLarge {
                    residual_norm,
                    tolerance,
                });
            }
            factored_solution(s, &self.values, &self.residual, &mut self.solution)?;
            for (k, &old) in s.permutation.iter().enumerate() {
                x[old] += self.solution[k];
            }
            finite_vector(x)?;
        }
        unreachable!("bounded refinement loop always returns")
    }
    pub fn solve(
        &mut self,
        j: &CsrMatrix,
        rhs: &DVector<f64>,
        damping: f64,
    ) -> Result<SparseLdlReport, Error> {
        let mut x = DVector::zeros(j.shape().1);
        let stats = self.solve_into(j, rhs, damping, &mut x)?;
        Ok(SparseLdlReport {
            x,
            residual_norm: stats.residual_norm,
            refinement_steps: stats.refinement_steps,
            normal_entries: stats.normal_entries,
            factor_entries: stats.factor_entries,
        })
    }
}
fn factored_solution(
    s: &SparseLdlSymbolic,
    values: &[f64],
    rhs: &DVector<f64>,
    solution: &mut DVector<f64>,
) -> Result<(), Error> {
    finite_vector(rhs)?;
    for (k, &old) in s.permutation.iter().enumerate() {
        solution[k] = rhs[old];
    }
    for k in 0..s.dimension() {
        let v = solution[k];
        for i in s.offsets[k] + 1..s.offsets[k + 1] {
            solution[s.rows[i]] -= values[i] * v;
        }
    }
    for k in 0..s.dimension() {
        solution[k] /= values[s.offsets[k]];
    }
    for k in (0..s.dimension()).rev() {
        for i in s.offsets[k] + 1..s.offsets[k + 1] {
            solution[k] -= values[i] * solution[s.rows[i]];
        }
    }
    finite_vector(solution)
}
/// One-shot checked solve. For repeated numeric and symbolic reuse, keep a workspace.
pub fn solve_normal(
    j: &CsrMatrix,
    rhs: &DVector<f64>,
    damping: f64,
    options: &SparseLdlOptions,
) -> Result<SparseLdlReport, Error> {
    SparseLdlWorkspace::new(*options)?.solve(j, rhs, damping)
}

#[cfg(test)]
mod adjacency_tests {
    use super::*;

    #[test]
    fn sorted_neighbors_preserve_membership_through_promotion_removal_and_clear() {
        let mut neighbors = PatternColumn::Small(Vec::new());
        for value in (0..64).rev() {
            neighbors.insert_neighbor(value);
            neighbors.insert_neighbor(value);
        }
        assert!(matches!(neighbors, PatternColumn::Small(_)));
        assert_eq!(neighbors.len(), 64);
        assert_eq!(
            neighbors.iter().copied().collect::<Vec<_>>(),
            (0..64).collect::<Vec<_>>()
        );
        neighbors.remove(&30);
        neighbors.remove(&200);
        assert!(!neighbors.contains(&30));
        assert!(neighbors.contains(&31));
        neighbors.insert_neighbor(30);
        neighbors.insert_neighbor(100);
        assert!(matches!(neighbors, PatternColumn::Large(_)));
        let mut expected: Vec<_> = (0..64).chain([100]).collect();
        for value in (0..64).step_by(2) {
            neighbors.remove(&value);
            expected.retain(|&v| v != value);
        }
        assert_eq!(neighbors.len(), expected.len());
        assert_eq!(neighbors.iter().copied().collect::<Vec<_>>(), expected);
        neighbors.clear();
        assert_eq!(neighbors.len(), 0);
        neighbors.insert_neighbor(7);
        neighbors.insert_neighbor(1);
        neighbors.insert_neighbor(7);
        assert_eq!(neighbors.iter().copied().collect::<Vec<_>>(), vec![1, 7]);
    }

    // Dense boolean graph plus a full scan selects the minimum (degree, key).
    // It shares neither the heap nor the hybrid representation with production.
    fn reference(n: usize, edges: &[(usize, usize)], limit: usize) -> Result<Vec<usize>, Error> {
        let mut graph = vec![vec![false; n]; n];
        let mut entries = n;
        for &(a, b) in edges {
            if !graph[a][b] {
                graph[a][b] = true;
                graph[b][a] = true;
                entries += 1;
            }
        }
        let mut active = vec![true; n];
        let mut order = Vec::new();
        for _ in 0..n {
            let key = (0..n)
                .filter(|&k| active[k])
                .min_by_key(|&k| (graph[k].iter().filter(|&&v| v).count(), k))
                .unwrap();
            active[key] = false;
            order.push(key);
            let neighbors: Vec<_> = (0..n).filter(|&k| graph[key][k]).collect();
            for &other in &neighbors {
                graph[other][key] = false;
            }
            for (i, &a) in neighbors.iter().enumerate() {
                for &b in &neighbors[i + 1..] {
                    if !graph[a][b] {
                        if entries >= limit {
                            return Err(Error::SparseFillLimit { limit });
                        }
                        graph[a][b] = true;
                        graph[b][a] = true;
                        entries += 1;
                    }
                }
            }
            graph[key].fill(false);
        }
        Ok(order)
    }

    #[test]
    fn greedy_order_and_cumulative_fill_limits_match_independent_dense_graph() {
        let fixtures = [
            (1, vec![]),
            (8, vec![(1, 2), (4, 5), (1, 2)]),
            (
                24,
                (0..24)
                    .flat_map(|k| [(k, (k + 1) % 24), (k, (k + 5) % 24)])
                    .collect(),
            ),
            (
                72,
                (1..72)
                    .flat_map(|k| [(0, k), (k, if k == 71 { 1 } else { k + 1 })])
                    .collect(),
            ),
            (
                72,
                (0..72)
                    .flat_map(|a| (a + 1..72).map(move |b| (a, b)))
                    .collect(),
            ),
            (96, (1..96).rev().map(|k| (0, k)).collect()),
        ];
        for (n, edges) in fixtures {
            let mut columns: Columns = (0..n).map(PatternColumn::diagonal).collect();
            let mut entries = n;
            for &(a, b) in &edges {
                add_pattern_entry(&mut columns[a.min(b)], a.max(b), &mut entries, usize::MAX)
                    .unwrap();
            }
            for limit in [
                entries,
                (entries + 10).min(n * (n + 1) / 2),
                n * (n + 1) / 2,
            ] {
                assert_eq!(
                    minimum_degree(&columns, entries, limit),
                    reference(n, &edges, limit),
                    "n={n}, limit={limit}"
                );
            }
        }
    }
}

#[cfg(test)]
mod normal_rebind_tests {
    use super::*;
    use crate::DMatrix;

    fn dense_graph(extra_joint_rows: usize, reverse_rows: bool, huge: bool) -> CsrMatrix {
        let n = 6;
        let m = n + 1 + extra_joint_rows;
        let mut entries = Vec::new();
        for row in 0..m {
            let target = if reverse_rows { m - 1 - row } else { row };
            if row < n {
                entries.push((target, row, 1.));
            } else {
                for col in 0..n {
                    entries.push((
                        target,
                        col,
                        if huge && row == n + extra_joint_rows {
                            f64::MAX
                        } else {
                            0.2 + ((col + row + 1) as f64 * 0.17).cos()
                        },
                    ));
                }
            }
        }
        CsrMatrix::from_triplets(m, n, entries).unwrap()
    }

    #[test]
    fn equal_normal_graph_rebinds_rows_and_plan_budget_without_changing_arithmetic() {
        // 35 update destinations leave 28 assembly slots. One joint row fits;
        // repeated joint rows require the same original search fallback.
        let options = SparseLdlOptions {
            max_entries: 63,
            ..Default::default()
        };
        let original = dense_graph(0, false, false);
        let static_symbolic = SparseLdlSymbolic::analyze(&original, &options).unwrap();
        let changed = dense_graph(1, false, false);
        assert!(!static_symbolic.matches_pattern(&changed));
        assert!(matches!(
            static_symbolic.factor_normal(&changed, 0.01),
            Err(Error::Evaluation(_))
        ));
        let mut workspace = SparseLdlWorkspace::new(options).unwrap();
        let rhs = DVector::from_fn(6, |i, _| (i as f64 * 0.3).sin());
        let mut x = DVector::zeros(6);
        workspace.solve_into(&original, &rhs, 0.01, &mut x).unwrap();
        let rows_pointer = workspace.symbolic.as_ref().unwrap().rows.as_ptr();
        let updates_pointer = workspace
            .symbolic
            .as_ref()
            .unwrap()
            .update_targets
            .as_ref()
            .unwrap()
            .as_ptr();
        for (index, j) in [
            original,
            changed,
            dense_graph(1, true, false),
            dense_graph(0, false, false),
        ]
        .into_iter()
        .enumerate()
        {
            let stats = workspace.solve_into(&j, &rhs, 0.01, &mut x).unwrap();
            let fresh = solve_normal(&j, &rhs, 0.01, &options).unwrap();
            assert_eq!(x, fresh.x);
            assert_eq!(stats.residual_norm, fresh.residual_norm);
            assert_eq!(stats.refinement_steps, fresh.refinement_steps);
            assert_eq!(stats.normal_entries, fresh.normal_entries);
            assert_eq!(stats.factor_entries, fresh.factor_entries);
            let symbolic = workspace.symbolic.as_ref().unwrap();
            assert!(symbolic.matches_pattern(&j));
            assert_eq!(symbolic.rows.as_ptr(), rows_pointer);
            assert_eq!(
                symbolic.update_targets.as_ref().unwrap().as_ptr(),
                updates_pointer
            );
            assert_eq!(
                symbolic.assembly_targets.is_some(),
                index == 0 || index == 3
            );
            assert_eq!(workspace.analyses(), index + 1);
            let mut dense = DMatrix::zeros(j.shape().0, 6);
            for (r, c, v) in j.entries() {
                dense[(r, c)] = v;
            }
            let mut h = dense.transpose() * dense;
            for k in 0..6 {
                h[(k, k)] += 0.01 * h[(k, k)].max(1.);
            }
            assert!((&x - h.cholesky().unwrap().solve(&rhs)).norm() < 1e-12);
        }
    }

    #[test]
    fn equal_counts_with_different_normal_edges_require_full_analysis() {
        let make = |edges: &[(usize, usize)]| {
            let mut entries: Vec<_> = (0..6).map(|i| (i, i, 1.)).collect();
            for (r, &(a, b)) in edges.iter().enumerate() {
                entries.extend([(6 + r, a, -1.), (6 + r, b, 1.)]);
            }
            CsrMatrix::from_triplets(8, 6, entries).unwrap()
        };
        let a = make(&[(0, 1), (2, 3)]);
        let b = make(&[(0, 2), (1, 3)]);
        let options = SparseLdlOptions::default();
        let mut symbolic = SparseLdlSymbolic::analyze(&a, &options).unwrap();
        let other = SparseLdlSymbolic::analyze(&b, &options).unwrap();
        assert_eq!(symbolic.normal_entries(), other.normal_entries());
        assert!(!symbolic.rebind_pattern(&b, options.max_entries).unwrap());
        assert!(symbolic.matches_pattern(&a));
        let smaller =
            CsrMatrix::from_triplets(3, 3, vec![(0, 0, 1.), (1, 1, 1.), (2, 2, 1.)]).unwrap();
        assert!(
            !symbolic
                .rebind_pattern(&smaller, options.max_entries)
                .unwrap()
        );
        let mut workspace = SparseLdlWorkspace::new(options).unwrap();
        for (i, j) in [a, b].into_iter().enumerate() {
            let rhs = DVector::from_element(6, 1.);
            let solved = workspace.solve(&j, &rhs, 0.01).unwrap();
            assert_eq!(solved.x, solve_normal(&j, &rhs, 0.01, &options).unwrap().x);
            assert_eq!(workspace.analyses(), i + 1);
        }
    }

    #[test]
    fn failed_numeric_factor_after_rebinding_cannot_expose_old_solution_and_recovers() {
        let options = SparseLdlOptions::default();
        let mut workspace = SparseLdlWorkspace::new(options).unwrap();
        let original = dense_graph(0, false, false);
        let rhs = DVector::from_element(6, 1.);
        let expected = workspace.solve(&original, &rhs, 0.01).unwrap().x;
        let bad = dense_graph(1, false, true);
        assert_eq!(workspace.refactor(&bad, 0.01), Err(Error::NonFinite));
        assert!(workspace.symbolic().unwrap().matches_pattern(&bad));
        assert_eq!(workspace.analyses(), 2);
        let mut x = DVector::zeros(6);
        assert_eq!(
            workspace.solve_factored_into(&rhs, &mut x),
            Err(Error::LinearSolve)
        );
        assert_eq!(workspace.solve(&original, &rhs, 0.01).unwrap().x, expected);
        assert_eq!(workspace.analyses(), 3);
    }
}

#[cfg(test)]
mod update_cursor_tests {
    use super::*;

    #[test]
    fn destination_cursor_matches_binary_lookup_for_cliques_gaps_and_all_orders() {
        for n in [6, 9, 72] {
            for dense in [false, true] {
                let mut entries = (0..n).map(|k| (k, k, 2.)).collect::<Vec<_>>();
                let mut row = n;
                let mut supports = vec![vec![0, 2, n - 1], vec![1, 3, n - 2], vec![0, 1, n - 3]];
                if dense {
                    supports.push((0..n).collect());
                }
                for support in supports {
                    for col in support {
                        entries.push((row, col, 1.));
                    }
                    row += 1;
                }
                let j = CsrMatrix::from_triplets(row, n, entries).unwrap();
                for ordering in [
                    SparseOrdering::Natural,
                    SparseOrdering::MinimumDegree,
                    SparseOrdering::BlockMinimumDegree { block_size: 3 },
                ] {
                    let s = SparseLdlSymbolic::analyze(
                        &j,
                        &SparseLdlOptions {
                            ordering,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    let mut expected = Vec::new();
                    for k in 0..n {
                        for i in s.offsets[k] + 1..s.offsets[k + 1] {
                            for q in i..s.offsets[k + 1] {
                                expected.push(s.entry(s.rows[i], s.rows[q]));
                            }
                        }
                    }
                    assert_eq!(s.update_targets.as_ref().unwrap(), &expected);
                }
            }
        }
        let j = CsrMatrix::from_triplets(6, 6, (0..6).map(|k| (k, k, 1.)).collect()).unwrap();
        let s = SparseLdlSymbolic::analyze(&j, &Default::default()).unwrap();
        assert!(s.has_update_plan());
        assert_eq!(s.update_entries(), 0);
    }
}

#[cfg(test)]
mod coarse_ordering_tests {
    use super::*;
    #[test]
    fn direct_coarse_graph_matches_scalar_projection_orders_and_budget_failures() {
        for n in [1, 6, 15, 30] {
            let j = CsrMatrix::from_triplets(
                3 * n,
                n,
                (0..3 * n)
                    .flat_map(|r| {
                        (0..n).filter_map(move |c| {
                            (r % 7 != 0 && ((r + 2 * c) % 5 == 0 || r % n == c))
                                .then_some((r, c, 1.))
                        })
                    })
                    .collect(),
            )
            .unwrap();
            for block_size in [1, 2, 3, 5] {
                for max_entries in [n, n + 3, n * (n + 1) / 2, 10_000] {
                    let options = SparseLdlOptions {
                        ordering: SparseOrdering::BlockMinimumDegree { block_size },
                        max_entries,
                        ..Default::default()
                    };
                    let expected =
                        assemble_pattern(&j, max_entries).and_then(|(columns, entries)| {
                            choose_ordering(&columns, entries, &options)
                        });
                    assert_eq!(column_ordering(&j, &options), expected);
                }
            }
        }
    }
}
