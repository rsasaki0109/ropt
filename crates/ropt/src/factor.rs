//! Factor graph with dense and sparse Jacobian assembly, with typed variable IDs, explicit local derivatives and fixed blocks.
//! Numeric differentiation is opt-in at factor registration; never an implicit fallback.
use crate::manifold::{SE2, SE3, SO2, SO3, Tangent6};
use crate::noise::{NoiseModel, WhiteningWorkspace};
use crate::sparse::CsrMatrix;
use crate::{DMatrix, DVector, Error, LocalProblem, Loss, LossBlock};
use nalgebra::{Quaternion, UnitQuaternion, Vector2, Vector3};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
mod autodiff;
pub(crate) use autodiff::AutoDiffLayout;

static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VariableId {
    graph: u64,
    index: usize,
}
/// Storage conventions: SO2 `[angle]`, SE2 `[x,y,angle]`, SO3 `[w,x,y,z]`,
/// SE3 `[tx,ty,tz,w,x,y,z]`. Rotation tangent increments are right/body increments.
#[derive(Debug, Clone, Copy)]
pub enum VariableKind {
    Euclidean(usize),
    SO2,
    SO3,
    SE2,
    /// SE2 with a Cartesian right chart; Exp/Log on SE2 remain unchanged.
    SE2Cartesian,
    SE3,
}
impl VariableKind {
    pub fn dimensions(self) -> (usize, usize) {
        match self {
            Self::Euclidean(n) => (n, n),
            Self::SO2 => (1, 1),
            Self::SO3 => (4, 3),
            Self::SE2 | Self::SE2Cartesian => (3, 3),
            Self::SE3 => (7, 6),
        }
    }
    pub(crate) fn validate(self, x: &DVector<f64>) -> Result<(), Error> {
        self.validate_slice(x.as_slice())
    }
    fn validate_slice(self, values: &[f64]) -> Result<(), Error> {
        let x = nalgebra::DVectorView::from_slice(values, values.len());
        if x.len() != self.dimensions().0 || x.is_empty() {
            return Err(Error::InvalidDimensions);
        }
        if !x.iter().all(|v| v.is_finite()) {
            return Err(Error::NonFinite);
        }
        let offset = match self {
            Self::SO3 => Some(0),
            Self::SE3 => Some(3),
            _ => None,
        };
        if let Some(k) = offset {
            let norm = x.rows(k, 4).norm();
            if (norm - 1.0).abs() > 1e-8 {
                return Err(Error::Evaluation("quaternion must have unit norm".into()));
            }
        }
        Ok(())
    }
    /// Jet-valued right retraction in the same storage and tangent conventions as `retract`.
    /// Small-angle expansions keep derivatives finite at zero rotation.
    pub fn retract_autodiff(
        self,
        x: &DVector<f64>,
        d: &[crate::autodiff::DynamicJet],
    ) -> Result<Vec<crate::autodiff::DynamicJet>, Error> {
        crate::manifold_autodiff::retract(self, x, d)
    }
    pub fn retract(self, x: &DVector<f64>, d: &DVector<f64>) -> Result<DVector<f64>, Error> {
        self.validate(x)?;
        if d.len() != self.dimensions().1 {
            return Err(Error::InvalidDimensions);
        }
        if !d.iter().all(|v| v.is_finite()) {
            return Err(Error::NonFinite);
        }
        let out = match self {
            Self::Euclidean(_) => x + d,
            Self::SO2 => DVector::from_element(1, SO2::exp(x[0]).retract(d[0]).log()),
            Self::SO3 => {
                so3_values(decode_so3(x.as_slice(), 0).retract(Vector3::new(d[0], d[1], d[2])))
            }
            Self::SE2 => se2_values(read_se2(x)?.retract(Vector3::new(d[0], d[1], d[2]))),
            Self::SE2Cartesian => {
                se2_values(read_se2(x)?.retract_cartesian(Vector3::new(d[0], d[1], d[2])))
            }
            Self::SE3 => {
                se3_values(read_se3(x)?.retract(Tangent6::from_column_slice(d.as_slice())))
            }
        };
        self.validate(&out)?;
        Ok(out)
    }
    // Private graph path: x is already validated, and storage ranges have the
    // kind's dimensions. Use the same Lie operations as the owned public API.
    fn retract_validated_into(self, x: &[f64], d: &[f64], out: &mut [f64]) -> Result<(), Error> {
        if !d.iter().all(|v| v.is_finite()) {
            return Err(Error::NonFinite);
        }
        match self {
            Self::Euclidean(_) => {
                for ((result, &value), &delta) in out.iter_mut().zip(x).zip(d) {
                    *result = value + delta;
                }
            }
            Self::SO2 => out[0] = SO2::exp(x[0]).retract(d[0]).log(),
            Self::SO3 => {
                let rotation = decode_so3(x, 0).retract(Vector3::new(d[0], d[1], d[2]));
                let q = rotation.0.quaternion();
                out.copy_from_slice(&[q.w, q.i, q.j, q.k]);
            }
            Self::SE2 | Self::SE2Cartesian => {
                let pose = SE2 {
                    translation: Vector2::new(x[0], x[1]),
                    rotation: SO2::exp(x[2]),
                };
                let step = Vector3::new(d[0], d[1], d[2]);
                let pose = if matches!(self, Self::SE2Cartesian) {
                    pose.retract_cartesian(step)
                } else {
                    pose.retract(step)
                };
                out.copy_from_slice(&[
                    pose.translation[0],
                    pose.translation[1],
                    pose.rotation.log(),
                ]);
            }
            Self::SE3 => {
                let pose = SE3 {
                    translation: Vector3::new(x[0], x[1], x[2]),
                    rotation: decode_so3(x, 3),
                }
                .retract(Tangent6::from_column_slice(d));
                let q = pose.rotation.0.quaternion();
                out.copy_from_slice(&[
                    pose.translation[0],
                    pose.translation[1],
                    pose.translation[2],
                    q.w,
                    q.i,
                    q.j,
                    q.k,
                ]);
            }
        }
        self.validate_slice(out)
    }
    /// Principal right-local coordinates in the selected chart. SE2Cartesian
    /// uses relative translation and angle; other Lie kinds use group Log.
    pub fn local(self, anchor: &DVector<f64>, value: &DVector<f64>) -> Result<DVector<f64>, Error> {
        self.validate(anchor)?;
        self.validate(value)?;
        let out = match self {
            Self::Euclidean(_) => value - anchor,
            Self::SO2 => DVector::from_element(1, SO2::exp(anchor[0]).local(SO2::exp(value[0]))),
            Self::SO3 => {
                DVector::from_column_slice(read_so3(anchor)?.local(read_so3(value)?).as_slice())
            }
            Self::SE2 => {
                DVector::from_column_slice(read_se2(anchor)?.local(read_se2(value)?).as_slice())
            }
            Self::SE2Cartesian => DVector::from_column_slice(
                read_se2(anchor)?
                    .local_cartesian(read_se2(value)?)
                    .as_slice(),
            ),
            Self::SE3 => {
                DVector::from_column_slice(read_se3(anchor)?.local(read_se3(value)?).as_slice())
            }
        };
        if !out.iter().all(|v| v.is_finite()) {
            return Err(Error::NonFinite);
        }
        Ok(out)
    }
    /// Forward-AD local coordinates in the same chart as `local`.
    /// The principal rotation chart is discontinuous at pi.
    pub fn local_autodiff(
        self,
        anchor: &DVector<f64>,
        value: &[crate::autodiff::DynamicJet],
    ) -> Result<Vec<crate::autodiff::DynamicJet>, Error> {
        crate::manifold_autodiff::local(self, anchor, value)
    }
}
fn decode_so3(x: &[f64], k: usize) -> SO3 {
    SO3(UnitQuaternion::new_normalize(Quaternion::new(
        x[k],
        x[k + 1],
        x[k + 2],
        x[k + 3],
    )))
}
pub fn read_so3(x: &DVector<f64>) -> Result<SO3, Error> {
    VariableKind::SO3.validate(x)?;
    Ok(decode_so3(x.as_slice(), 0))
}
pub fn so3_values(r: SO3) -> DVector<f64> {
    let q = r.0.quaternion();
    DVector::from_vec(vec![q.w, q.i, q.j, q.k])
}
pub fn se2_values(p: SE2) -> DVector<f64> {
    DVector::from_vec(vec![p.translation[0], p.translation[1], p.rotation.log()])
}
pub fn read_se2(x: &DVector<f64>) -> Result<SE2, Error> {
    VariableKind::SE2.validate(x)?;
    Ok(SE2 {
        translation: Vector2::new(x[0], x[1]),
        rotation: SO2::exp(x[2]),
    })
}
pub fn se3_values(p: SE3) -> DVector<f64> {
    let q = p.rotation.0.quaternion();
    DVector::from_vec(vec![
        p.translation[0],
        p.translation[1],
        p.translation[2],
        q.w,
        q.i,
        q.j,
        q.k,
    ])
}
pub fn read_se3(x: &DVector<f64>) -> Result<SE3, Error> {
    VariableKind::SE3.validate(x)?;
    Ok(SE3 {
        translation: Vector3::new(x[0], x[1], x[2]),
        rotation: decode_so3(x.as_slice(), 3),
    })
}

struct Variable {
    kind: VariableKind,
    ambient: Range<usize>,
    tangent: Option<Range<usize>>,
    fixed: bool,
}
type NumericFn = dyn Fn(&[DVector<f64>]) -> Result<DVector<f64>, Error>;
type AnalyticFn = dyn Fn(&[DVector<f64>]) -> Result<(DVector<f64>, Vec<DMatrix<f64>>), Error>;
enum Evaluation {
    Analytic(Box<AnalyticFn>),
    Central { step: f64, function: Box<NumericFn> },
}
/// Noise is applied before loss. Some(loss) replaces the global scalar loss for
/// this entire whitened factor block; None keeps the global per-scalar fallback.
#[derive(Debug, Clone, Default)]
pub struct FactorOptions {
    pub noise: NoiseModel,
    pub loss: Option<Loss>,
}
impl FactorOptions {
    fn validate(&self, m: usize) -> Result<(), Error> {
        self.noise.validate_dimension(m)?;
        if self.loss.is_some_and(|loss| !loss.valid()) {
            return Err(Error::InvalidOptions);
        }
        Ok(())
    }
}
struct Factor {
    options: FactorOptions,
    variables: Vec<VariableId>,
    residuals: usize,
    evaluation: Evaluation,
}
/// Dense and CSR assembly. Factors own callbacks; registration validates connectivity.
/// Residual-count overflow is rejected before changing the graph.
pub struct FactorGraph {
    identity: u64,
    variables: Vec<Variable>,
    factors: Vec<Factor>,
    initial: Vec<f64>,
    tangent_size: usize,
    residual_size: usize,
}
impl Default for FactorGraph {
    fn default() -> Self {
        Self {
            identity: NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed),
            variables: Vec::new(),
            factors: Vec::new(),
            initial: Vec::new(),
            tangent_size: 0,
            residual_size: 0,
        }
    }
}
impl FactorGraph {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_variable(
        &mut self,
        kind: VariableKind,
        value: DVector<f64>,
    ) -> Result<VariableId, Error> {
        kind.validate(&value)?;
        let start = self.initial.len();
        self.initial.extend_from_slice(value.as_slice());
        let (_, n) = kind.dimensions();
        let t = self.tangent_size;
        self.tangent_size += n;
        let id = VariableId {
            graph: self.identity,
            index: self.variables.len(),
        };
        self.variables.push(Variable {
            kind,
            ambient: start..self.initial.len(),
            tangent: Some(t..t + n),
            fixed: false,
        });
        Ok(id)
    }
    pub fn set_fixed(&mut self, id: VariableId, fixed: bool) -> Result<(), Error> {
        if id.graph != self.identity {
            return Err(Error::InvalidDimensions);
        }
        let variable = self
            .variables
            .get_mut(id.index)
            .ok_or(Error::InvalidDimensions)?;
        if variable.fixed == fixed {
            return Ok(());
        }
        variable.fixed = fixed;
        self.tangent_size = 0;
        for v in &mut self.variables {
            v.tangent = if v.fixed {
                None
            } else {
                let t = self.tangent_size;
                self.tangent_size += v.kind.dimensions().1;
                Some(t..self.tangent_size)
            };
        }
        Ok(())
    }
    fn check_factor(&self, ids: &[VariableId], m: usize) -> Result<(), Error> {
        if ids.is_empty()
            || m == 0
            // CSR also needs a terminal row offset. Validate before registering
            // callbacks or modifying the factor list and residual count.
            || self
                .residual_size
                .checked_add(m)
                .and_then(|rows| rows.checked_add(1))
                .is_none()
            || ids
                .iter()
                .any(|id| id.graph != self.identity || id.index >= self.variables.len())
        {
            return Err(Error::InvalidDimensions);
        }
        for (i, id) in ids.iter().enumerate() {
            if ids[..i].contains(id) {
                return Err(Error::Evaluation("duplicate factor variable".into()));
            }
        }
        Ok(())
    }
    /// Jacobian blocks are with respect to each variable's right tangent increments,
    /// in the same order as ids (including fixed variables).
    pub fn add_analytic_factor<F>(
        &mut self,
        ids: &[VariableId],
        m: usize,
        function: F,
    ) -> Result<(), Error>
    where
        F: Fn(&[DVector<f64>]) -> Result<(DVector<f64>, Vec<DMatrix<f64>>), Error> + 'static,
    {
        self.add_analytic_factor_with_options(ids, m, FactorOptions::default(), function)
    }
    /// Register an analytic factor with whitening and an optional joint robust loss.
    pub fn add_analytic_factor_with_options<F>(
        &mut self,
        ids: &[VariableId],
        m: usize,
        options: FactorOptions,
        function: F,
    ) -> Result<(), Error>
    where
        F: Fn(&[DVector<f64>]) -> Result<(DVector<f64>, Vec<DMatrix<f64>>), Error> + 'static,
    {
        self.check_factor(ids, m)?;
        options.validate(m)?;
        self.factors.push(Factor {
            options,
            variables: ids.to_vec(),
            residuals: m,
            evaluation: Evaluation::Analytic(Box::new(function)),
        });
        self.residual_size += m;
        Ok(())
    }
    /// Explicit central differences around manifold retractions; intended for verification
    /// and small examples, not an automatic performance path.
    pub fn add_numeric_factor<F>(
        &mut self,
        ids: &[VariableId],
        m: usize,
        step: f64,
        function: F,
    ) -> Result<(), Error>
    where
        F: Fn(&[DVector<f64>]) -> Result<DVector<f64>, Error> + 'static,
    {
        self.add_numeric_factor_with_options(ids, m, step, FactorOptions::default(), function)
    }
    /// Central differences stay explicit; whitening is applied after differentiating.
    /// Each perturbation output and derivative column is validated before the next
    /// callback, preserving the first detected dimension or numerical error.
    /// `step` and `2*step` must be finite and positive. Active perturbations must
    /// be distinct from the base storage and from each other; otherwise evaluation
    /// returns `InvalidOptions` before perturbation callbacks. Fixed blocks are skipped.
    pub fn add_numeric_factor_with_options<F>(
        &mut self,
        ids: &[VariableId],
        m: usize,
        step: f64,
        options: FactorOptions,
        function: F,
    ) -> Result<(), Error>
    where
        F: Fn(&[DVector<f64>]) -> Result<DVector<f64>, Error> + 'static,
    {
        self.check_factor(ids, m)?;
        options.validate(m)?;
        if !step.is_finite() || step <= 0.0 || !(2.0 * step).is_finite() {
            return Err(Error::InvalidOptions);
        }
        self.factors.push(Factor {
            options,
            variables: ids.to_vec(),
            residuals: m,
            evaluation: Evaluation::Central {
                step,
                function: Box::new(function),
            },
        });
        self.residual_size += m;
        Ok(())
    }
    /// Runtime AD of residuals evaluated on right-retracted ambient variable blocks.
    /// Callbacks receive the documented storage coordinates; Jacobians use tangent coordinates.
    pub fn add_autodiff_factor<F>(
        &mut self,
        ids: &[VariableId],
        m: usize,
        function: F,
    ) -> Result<(), Error>
    where
        F: Fn(&[&[crate::autodiff::DynamicJet]]) -> Result<Vec<crate::autodiff::DynamicJet>, Error>
            + 'static,
    {
        self.add_autodiff_factor_with_options(ids, m, 32, FactorOptions::default(), function)
    }
    /// Chunked runtime AD with the same whitening and joint loss policies as analytic factors.
    pub fn add_autodiff_factor_with_options<F>(
        &mut self,
        ids: &[VariableId],
        m: usize,
        chunk_size: usize,
        options: FactorOptions,
        function: F,
    ) -> Result<(), Error>
    where
        F: Fn(&[&[crate::autodiff::DynamicJet]]) -> Result<Vec<crate::autodiff::DynamicJet>, Error>
            + 'static,
    {
        self.check_factor(ids, m)?;
        if chunk_size == 0 {
            return Err(Error::InvalidOptions);
        }
        let kinds = ids
            .iter()
            .map(|id| self.variable_kind(*id))
            .collect::<Result<Vec<_>, Error>>()?;
        let layout = AutoDiffLayout::new(kinds, m, chunk_size)?;
        self.add_analytic_factor_with_options(ids, m, options, move |values| {
            layout.evaluate(values, &function)
        })
    }

    pub fn variable_kind(&self, id: VariableId) -> Result<VariableKind, Error> {
        if id.graph != self.identity {
            return Err(Error::InvalidDimensions);
        }
        Ok(self
            .variables
            .get(id.index)
            .ok_or(Error::InvalidDimensions)?
            .kind)
    }
    /// Active tangent columns at the graph's current fixed-variable layout.
    /// None denotes a fixed variable; ranges change after set_fixed.
    pub fn tangent_range(&self, id: VariableId) -> Result<Option<Range<usize>>, Error> {
        self.variable_kind(id)?;
        Ok(self.variables[id.index].tangent.clone())
    }
    /// Marginal covariance of the requested active variables in their right tangents.
    /// Output ordering follows ids, then each block's tangent coordinates. Fixed
    /// variables are conditioned constants and cannot be queried as estimated variables.
    pub fn marginal_covariance(
        &self,
        x: &DVector<f64>,
        ids: &[VariableId],
        options: &crate::covariance::CovarianceOptions,
    ) -> Result<crate::covariance::TangentCovariance, Error> {
        let indices = self.covariance_indices(ids)?;
        crate::covariance::marginal_covariance(self, x, &indices, options)
    }
    /// Validate query blocks without evaluating any factor callbacks.
    pub(crate) fn covariance_indices(&self, ids: &[VariableId]) -> Result<Vec<usize>, Error> {
        if ids.is_empty() {
            return Err(Error::InvalidDimensions);
        }
        let mut indices = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for &id in ids {
            if !seen.insert(id.index) {
                return Err(Error::InvalidDimensions);
            }
            let range = self.tangent_range(id)?.ok_or_else(|| {
                Error::Evaluation("fixed variables have no estimated covariance".into())
            })?;
            indices.extend(range);
        }
        Ok(indices)
    }
    /// Validate an independent eliminated suffix and return (retained columns, block size).
    /// Fixed IDs are excluded; active eliminated blocks must have equal tangent sizes.
    pub fn schur_layout(&self, eliminated: &[VariableId]) -> Result<(usize, usize), Error> {
        let mut seen = std::collections::BTreeSet::new();
        let mut active = std::collections::BTreeSet::new();
        let mut ranges = Vec::new();
        for id in eliminated {
            self.variable_kind(*id)?;
            if !seen.insert(id.index) {
                return Err(Error::InvalidDimensions);
            }
            if let Some(range) = &self.variables[id.index].tangent {
                active.insert(id.index);
                ranges.push(range.clone());
            }
        }
        ranges.sort_unstable_by_key(|r| r.start);
        let first = ranges.first().ok_or(Error::InvalidDimensions)?;
        let retained = first.start;
        let block_size = first.len();
        let mut end = retained;
        if retained == 0 {
            return Err(Error::InvalidDimensions);
        }
        for range in ranges {
            if range.start != end || range.len() != block_size {
                return Err(Error::InvalidDimensions);
            }
            end = range.end;
        }
        if end != self.tangent_size {
            return Err(Error::InvalidDimensions);
        }
        for factor in &self.factors {
            if factor
                .variables
                .iter()
                .filter(|id| active.contains(&id.index))
                .count()
                > 1
            {
                return Err(Error::Evaluation(
                    "factor couples eliminated Schur variables".into(),
                ));
            }
        }
        Ok((retained, block_size))
    }
    pub fn initial(&self) -> DVector<f64> {
        DVector::from_vec(self.initial.clone())
    }
    pub fn value(&self, x: &DVector<f64>, id: VariableId) -> Result<DVector<f64>, Error> {
        if x.len() != self.initial.len() || id.graph != self.identity {
            return Err(Error::InvalidDimensions);
        }
        let v = self
            .variables
            .get(id.index)
            .ok_or(Error::InvalidDimensions)?;
        let out = x.rows(v.ambient.start, v.ambient.len()).into_owned();
        v.kind.validate(&out)?;
        Ok(out)
    }
}
fn check_residual(r: &DVector<f64>, m: usize) -> Result<(), Error> {
    if r.len() != m {
        return Err(Error::InvalidDimensions);
    }
    if !r.iter().all(|v| v.is_finite()) {
        return Err(Error::NonFinite);
    }
    Ok(())
}
pub(crate) fn check_factor_output(
    residual: &DVector<f64>,
    blocks: &[DMatrix<f64>],
    residuals: usize,
    dimensions: impl ExactSizeIterator<Item = usize>,
) -> Result<(), Error> {
    check_residual(residual, residuals)?;
    if blocks.len() != dimensions.len() {
        return Err(Error::InvalidDimensions);
    }
    for (dimension, block) in dimensions.zip(blocks) {
        if block.shape() != (residuals, dimension) {
            return Err(Error::InvalidDimensions);
        }
        if !block.iter().all(|value| value.is_finite()) {
            return Err(Error::NonFinite);
        }
    }
    Ok(())
}
impl LocalProblem for FactorGraph {
    fn dimensions(&self) -> (usize, usize, usize) {
        (self.initial.len(), self.tangent_size, self.residual_size)
    }
    fn tangent_blocks(&self) -> Result<Vec<Range<usize>>, Error> {
        Ok(self
            .variables
            .iter()
            .filter_map(|v| v.tangent.clone())
            .collect())
    }
    fn loss_blocks(&self) -> Vec<LossBlock> {
        let mut row = 0;
        let mut blocks = Vec::new();
        for factor in &self.factors {
            if let Some(loss) = factor.options.loss {
                blocks.push(LossBlock {
                    rows: row..row + factor.residuals,
                    loss,
                });
            }
            row += factor.residuals;
        }
        blocks
    }

    fn retract(&self, x: &DVector<f64>, d: &DVector<f64>) -> Result<DVector<f64>, Error> {
        if x.len() != self.initial.len() || d.len() != self.tangent_size {
            return Err(Error::InvalidDimensions);
        }
        let mut out = x.clone();
        for v in &self.variables {
            let value = &x.as_slice()[v.ambient.clone()];
            // Preserve validation and failure order, including fixed/disconnected
            // blocks. A failed update never escapes this owned output.
            v.kind.validate_slice(value)?;
            if let Some(t) = &v.tangent {
                v.kind.retract_validated_into(
                    value,
                    &d.as_slice()[t.clone()],
                    &mut out.as_mut_slice()[v.ambient.clone()],
                )?;
            }
        }
        Ok(out)
    }
    fn evaluate_local(
        &self,
        x: &DVector<f64>,
        r: &mut DVector<f64>,
        j: &mut DMatrix<f64>,
    ) -> Result<(), Error> {
        if r.len() != self.residual_size
            || j.shape() != (self.residual_size, self.tangent_size)
            || x.len() != self.initial.len()
        {
            return Err(Error::InvalidDimensions);
        }
        j.fill(0.0);
        self.visit_factors(x, |row, f, residual, blocks| {
            r.rows_mut(row, f.residuals).copy_from(&residual);
            for (id, block) in f.variables.iter().zip(blocks) {
                if let Some(t) = &self.variables[id.index].tangent {
                    j.view_mut((row, t.start), (f.residuals, t.len()))
                        .copy_from(&block);
                }
            }
            Ok(())
        })
    }
    fn evaluate_sparse(&self, x: &DVector<f64>) -> Result<(DVector<f64>, CsrMatrix), Error> {
        let mut residuals = DVector::zeros(self.residual_size);
        let mut offsets = Vec::with_capacity(self.residual_size + 1);
        offsets.push(0);
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut order = Vec::new();
        self.visit_factors(x, |row, f, residual, blocks| {
            residuals.rows_mut(row, f.residuals).copy_from(&residual);
            // Callback/whitening slots retain the caller's order. Only assembly
            // visits distinct active blocks in their current tangent-column order.
            order.clear();
            order.extend(
                (0..f.variables.len())
                    .filter(|&slot| self.variables[f.variables[slot].index].tangent.is_some()),
            );
            order.sort_unstable_by_key(|&slot| f.variables[slot].index);
            for r in 0..f.residuals {
                for &slot in &order {
                    let t = self.variables[f.variables[slot].index]
                        .tangent
                        .as_ref()
                        .unwrap();
                    let block = &blocks[slot];
                    for c in 0..t.len() {
                        let v = block[(r, c)];
                        if v != 0.0 {
                            indices.push(t.start + c);
                            values.push(v);
                        }
                    }
                }
                offsets.push(indices.len());
            }
            Ok(())
        })?;
        Ok((
            residuals,
            CsrMatrix::from_canonical(
                self.residual_size,
                self.tangent_size,
                offsets,
                indices,
                values,
            )?,
        ))
    }
}

impl FactorGraph {
    fn visit_factors<F>(&self, x: &DVector<f64>, mut consume: F) -> Result<(), Error>
    where
        F: FnMut(usize, &Factor, DVector<f64>, Vec<DMatrix<f64>>) -> Result<(), Error>,
    {
        if x.len() != self.initial.len() {
            return Err(Error::InvalidDimensions);
        }
        // Validate every block, including fixed/disconnected variables, without
        // allocating owned copies that would immediately be discarded.
        for variable in &self.variables {
            variable
                .kind
                .validate_slice(&x.as_slice()[variable.ambient.clone()])?;
        }
        // Callbacks keep the owned-vector API. Reuse per-slot storage across
        // factors, passing only the current arity; no values escape the callback.
        let mut inputs = Vec::<DVector<f64>>::new();
        let mut whitening = WhiteningWorkspace::default();
        let mut row = 0;
        for f in &self.factors {
            inputs.resize_with(inputs.len().max(f.variables.len()), || DVector::zeros(0));
            for (id, buffer) in f.variables.iter().zip(&mut inputs) {
                let ambient = &self.variables[id.index].ambient;
                if buffer.len() != ambient.len() {
                    buffer.resize_vertically_mut(ambient.len(), 0.);
                }
                buffer.copy_from(&x.rows(ambient.start, ambient.len()));
            }
            let values = &inputs[..f.variables.len()];
            let (mut residual, mut blocks) = match &f.evaluation {
                Evaluation::Analytic(function) => function(values)?,
                Evaluation::Central { step, function } => {
                    let residual = function(values)?;
                    check_residual(&residual, f.residuals)?;
                    let mut blocks = Vec::new();
                    for (index, id) in f.variables.iter().enumerate() {
                        let kind = self.variables[id.index].kind;
                        let n = kind.dimensions().1;
                        let mut block = DMatrix::zeros(f.residuals, n);
                        if self.variables[id.index].fixed {
                            blocks.push(block);
                            continue;
                        }
                        for k in 0..n {
                            let mut d = DVector::zeros(n);
                            d[k] = *step;
                            let mut plus = values.to_vec();
                            let mut minus = values.to_vec();
                            plus[index] = kind.retract(&values[index], &d)?;
                            minus[index] = kind.retract(&values[index], &(-d))?;
                            // Comparing both samples also catches a canonicalized
                            // Lie-group storage value that differs from the base.
                            if plus[index] == values[index]
                                || minus[index] == values[index]
                                || plus[index] == minus[index]
                            {
                                return Err(Error::InvalidOptions);
                            }
                            let a = function(&plus)?;
                            check_residual(&a, f.residuals)?;
                            let b = function(&minus)?;
                            check_residual(&b, f.residuals)?;
                            let column = (a - b) / (2.0 * step);
                            crate::sparse::finite_vector(&column)?;
                            block.set_column(k, &column);
                        }
                        blocks.push(block);
                    }
                    (residual, blocks)
                }
            };
            check_factor_output(
                &residual,
                &blocks,
                f.residuals,
                f.variables
                    .iter()
                    .map(|id| self.variables[id.index].kind.dimensions().1),
            )?;
            f.options
                .noise
                .whiten(&mut residual, &mut blocks, &mut whitening)?;
            consume(row, f, residual, blocks)?;
            row += f.residuals;
        }
        Ok(())
    }
}
