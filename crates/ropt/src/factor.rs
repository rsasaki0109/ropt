//! Factor graph with dense and sparse Jacobian assembly, with typed variable IDs, explicit local derivatives and fixed blocks.
//! Numeric differentiation is opt-in at factor registration; never an implicit fallback.
use crate::manifold::{SE2, SE3, SO2, SO3, Tangent6};
use crate::sparse::CsrMatrix;
use crate::{DMatrix, DVector, Error, LocalProblem};
use nalgebra::{Quaternion, UnitQuaternion, Vector2, Vector3};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
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
    SE3,
}
impl VariableKind {
    pub fn dimensions(self) -> (usize, usize) {
        match self {
            Self::Euclidean(n) => (n, n),
            Self::SO2 => (1, 1),
            Self::SO3 => (4, 3),
            Self::SE2 => (3, 3),
            Self::SE3 => (7, 6),
        }
    }
    fn validate(self, x: &DVector<f64>) -> Result<(), Error> {
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
            Self::SO3 => so3_values(decode_so3(x, 0).retract(Vector3::new(d[0], d[1], d[2]))),
            Self::SE2 => se2_values(read_se2(x)?.retract(Vector3::new(d[0], d[1], d[2]))),
            Self::SE3 => {
                se3_values(read_se3(x)?.retract(Tangent6::from_column_slice(d.as_slice())))
            }
        };
        self.validate(&out)?;
        Ok(out)
    }
}
fn decode_so3(x: &DVector<f64>, k: usize) -> SO3 {
    SO3(UnitQuaternion::new_normalize(Quaternion::new(
        x[k],
        x[k + 1],
        x[k + 2],
        x[k + 3],
    )))
}
pub fn read_so3(x: &DVector<f64>) -> Result<SO3, Error> {
    VariableKind::SO3.validate(x)?;
    Ok(decode_so3(x, 0))
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
        rotation: decode_so3(x, 3),
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
struct Factor {
    variables: Vec<VariableId>,
    residuals: usize,
    evaluation: Evaluation,
}
/// Dense and CSR assembly. Factors own callbacks; registration validates connectivity.
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
        self.variables
            .get_mut(id.index)
            .ok_or(Error::InvalidDimensions)?
            .fixed = fixed;
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
        self.check_factor(ids, m)?;
        self.factors.push(Factor {
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
        self.check_factor(ids, m)?;
        if !step.is_finite() || step <= 0.0 {
            return Err(Error::InvalidOptions);
        }
        self.factors.push(Factor {
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
impl LocalProblem for FactorGraph {
    fn dimensions(&self) -> (usize, usize, usize) {
        (self.initial.len(), self.tangent_size, self.residual_size)
    }
    fn retract(&self, x: &DVector<f64>, d: &DVector<f64>) -> Result<DVector<f64>, Error> {
        if x.len() != self.initial.len() || d.len() != self.tangent_size {
            return Err(Error::InvalidDimensions);
        }
        let mut out = x.clone();
        for (i, v) in self.variables.iter().enumerate() {
            let value = self.value(
                x,
                VariableId {
                    graph: self.identity,
                    index: i,
                },
            )?;
            if let Some(t) = &v.tangent {
                let delta = d.rows(t.start, t.len()).into_owned();
                let updated = v.kind.retract(&value, &delta)?;
                out.rows_mut(v.ambient.start, v.ambient.len())
                    .copy_from(&updated);
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
        let mut entries = Vec::new();
        self.visit_factors(x, |row, f, residual, blocks| {
            residuals.rows_mut(row, f.residuals).copy_from(&residual);
            for (id, block) in f.variables.iter().zip(blocks) {
                if let Some(t) = &self.variables[id.index].tangent {
                    for r in 0..f.residuals {
                        for c in 0..t.len() {
                            let v = block[(r, c)];
                            if v != 0.0 {
                                entries.push((row + r, t.start + c, v));
                            }
                        }
                    }
                }
            }
            Ok(())
        })?;
        Ok((
            residuals,
            CsrMatrix::from_triplets(self.residual_size, self.tangent_size, entries)?,
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
        for i in 0..self.variables.len() {
            self.value(
                x,
                VariableId {
                    graph: self.identity,
                    index: i,
                },
            )?;
        }
        let mut row = 0;
        for f in &self.factors {
            let values = f
                .variables
                .iter()
                .map(|id| self.value(x, *id))
                .collect::<Result<Vec<_>, _>>()?;
            let (residual, blocks) = match &f.evaluation {
                Evaluation::Analytic(function) => function(&values)?,
                Evaluation::Central { step, function } => {
                    let residual = function(&values)?;
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
                            let mut plus = values.clone();
                            let mut minus = values.clone();
                            plus[index] = kind.retract(&values[index], &d)?;
                            minus[index] = kind.retract(&values[index], &(-d))?;
                            let a = function(&plus)?;
                            let b = function(&minus)?;
                            check_residual(&a, f.residuals)?;
                            check_residual(&b, f.residuals)?;
                            block.set_column(k, &((a - b) / (2.0 * step)));
                        }
                        blocks.push(block);
                    }
                    (residual, blocks)
                }
            };
            check_residual(&residual, f.residuals)?;
            if blocks.len() != f.variables.len() {
                return Err(Error::InvalidDimensions);
            }
            for (id, block) in f.variables.iter().zip(&blocks) {
                let v = &self.variables[id.index];
                if block.shape() != (f.residuals, v.kind.dimensions().1) {
                    return Err(Error::InvalidDimensions);
                }
                if !block.iter().all(|v| v.is_finite()) {
                    return Err(Error::NonFinite);
                }
            }
            consume(row, f, residual, blocks)?;
            row += f.residuals;
        }
        Ok(())
    }
}
