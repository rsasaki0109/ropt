//! Transactional batch fixed-lag smoothing with square-root tangent priors.
//! Each update reoptimizes the window. This is not iSAM or a Bayes-tree update.
use crate::autodiff::DynamicJet;
use crate::factor::{FactorGraph, FactorOptions, VariableId, VariableKind};
use crate::marginalization::{MarginalizationOptions, marginalize_linearized};
use crate::{
    DMatrix, DVector, Error, Iteration, LinearSolver, LocalProblem, Loss, Options, QrOptions,
    Termination, solve_local,
};
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

/// Application-owned key, unique among live variables. Retired keys may be reused.
pub type Key = u64;
#[derive(Debug, Clone)]
pub struct WindowVariable {
    pub key: Key,
    pub kind: VariableKind,
    pub value: DVector<f64>,
    /// None keeps a persistent variable (for example shared calibration).
    pub timestamp: Option<f64>,
    pub fixed: bool,
}
type AnalyticFn = dyn Fn(&[DVector<f64>]) -> Result<(DVector<f64>, Vec<DMatrix<f64>>), Error>;
type AutoDiffFn = dyn Fn(&[&[DynamicJet]]) -> Result<Vec<DynamicJet>, Error>;
#[derive(Clone)]
enum Evaluation {
    Analytic(Rc<AnalyticFn>),
    AutoDiff {
        chunk_size: usize,
        function: Rc<AutoDiffFn>,
    },
}
/// Owned factor recipe. Callbacks must be deterministic; callback side effects
/// cannot be rolled back when an update fails.
#[derive(Clone)]
pub struct SmootherFactor {
    pub(crate) keys: Vec<Key>,
    pub(crate) residuals: usize,
    pub(crate) options: FactorOptions,
    evaluation: Evaluation,
}
impl SmootherFactor {
    pub(crate) fn evaluate(
        &self,
        values: &[DVector<f64>],
        kinds: impl Iterator<Item = VariableKind>,
    ) -> Result<(DVector<f64>, Vec<DMatrix<f64>>), Error> {
        match &self.evaluation {
            Evaluation::Analytic(function) => function(values),
            Evaluation::AutoDiff {
                chunk_size,
                function,
            } => crate::factor::AutoDiffLayout::new(kinds.collect(), self.residuals, *chunk_size)?
                .evaluate(values, function.as_ref()),
        }
    }
    pub fn analytic<F>(keys: &[Key], residuals: usize, options: FactorOptions, function: F) -> Self
    where
        F: Fn(&[DVector<f64>]) -> Result<(DVector<f64>, Vec<DMatrix<f64>>), Error> + 'static,
    {
        Self {
            keys: keys.to_vec(),
            residuals,
            options,
            evaluation: Evaluation::Analytic(Rc::new(function)),
        }
    }
    pub fn autodiff<F>(
        keys: &[Key],
        residuals: usize,
        chunk_size: usize,
        options: FactorOptions,
        function: F,
    ) -> Self
    where
        F: Fn(&[&[DynamicJet]]) -> Result<Vec<DynamicJet>, Error> + 'static,
    {
        Self {
            keys: keys.to_vec(),
            residuals,
            options,
            evaluation: Evaluation::AutoDiff {
                chunk_size,
                function: Rc::new(function),
            },
        }
    }
    fn register(&self, graph: &mut FactorGraph, ids: &[VariableId]) -> Result<(), Error> {
        match &self.evaluation {
            Evaluation::Analytic(function) => {
                let function = function.clone();
                graph.add_analytic_factor_with_options(
                    ids,
                    self.residuals,
                    self.options.clone(),
                    move |values| function(values),
                )
            }
            Evaluation::AutoDiff {
                chunk_size,
                function,
            } => {
                let function = function.clone();
                graph.add_autodiff_factor_with_options(
                    ids,
                    self.residuals,
                    *chunk_size,
                    self.options.clone(),
                    move |values| function(values),
                )
            }
        }
    }
}
#[derive(Default)]
pub struct SmootherUpdate {
    pub variables: Vec<WindowVariable>,
    pub factors: Vec<SmootherFactor>,
    /// Refresh existing timestamps. Times cannot move backwards or exceed now.
    pub timestamps: Vec<(Key, f64)>,
}
#[derive(Debug, Clone, Copy, Default)]
pub enum MarginalizationLoss {
    /// Reject robust factors when they enter marginalization.
    #[default]
    Gaussian,
    /// Freeze robust IRLS weights at the current estimate; the prior is Gaussian.
    FrozenIrls,
}
#[derive(Debug, Clone)]
pub struct FixedLagOptions {
    pub lag: f64,
    pub solver: Options,
    pub marginalization: MarginalizationOptions,
    pub marginalization_loss: MarginalizationLoss,
}
impl Default for FixedLagOptions {
    fn default() -> Self {
        Self {
            lag: 2.0,
            solver: Options {
                linear_solver: LinearSolver::DenseQr(QrOptions::default()),
                ..Options::default()
            },
            marginalization: MarginalizationOptions::default(),
            marginalization_loss: MarginalizationLoss::Gaussian,
        }
    }
}
#[derive(Debug, Clone, Default)]
pub struct MarginalizationSummary {
    pub removed_factors: usize,
    pub separator_variables: usize,
    pub eliminated_tangent: usize,
    pub separator_tangent: usize,
    pub prior_rows: usize,
}
#[derive(Debug)]
pub struct FixedLagReport {
    /// Live estimates after marginalization. Expired keys are absent.
    pub estimates: BTreeMap<Key, DVector<f64>>,
    pub marginalized: Vec<Key>,
    /// Costs of the optimized window before expiry, including carried constants.
    /// After nonlinear marginalization this is a surrogate, not all-history cost.
    pub initial_cost: f64,
    pub final_cost: f64,
    pub termination: Termination,
    pub iterations: Vec<Iteration>,
    pub marginalization: MarginalizationSummary,
}
pub struct FixedLagSmoother {
    options: FixedLagOptions,
    variables: BTreeMap<Key, WindowVariable>,
    factors: Vec<SmootherFactor>,
    last_time: Option<f64>,
    constant_cost: f64,
}
impl FixedLagSmoother {
    pub fn new(options: FixedLagOptions) -> Result<Self, Error> {
        if !options.lag.is_finite() || options.lag <= 0.0 {
            return Err(Error::InvalidOptions);
        }
        options.marginalization.validate()?;
        Ok(Self {
            options,
            variables: BTreeMap::new(),
            factors: Vec::new(),
            last_time: None,
            constant_cost: 0.0,
        })
    }
    pub fn estimate(&self, key: Key) -> Option<&DVector<f64>> {
        self.variables.get(&key).map(|v| &v.value)
    }
    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }
    pub fn factor_count(&self) -> usize {
        self.factors.len()
    }
    pub fn last_time(&self) -> Option<f64> {
        self.last_time
    }
    /// Selected right-tangent covariance of the live window and its frozen prior.
    /// This is the current surrogate model, not exact nonlinear all-history covariance.
    pub fn marginal_covariance(
        &self,
        keys: &[Key],
        options: &crate::covariance::CovarianceOptions,
    ) -> Result<crate::covariance::TangentCovariance, Error> {
        let order = self.variables.keys().copied().collect::<Vec<_>>();
        let (graph, ids) = build_graph(&self.variables, &self.factors, &order)?;
        let query = keys
            .iter()
            .map(|key| ids.get(key).copied().ok_or(Error::InvalidDimensions))
            .collect::<Result<Vec<_>, _>>()?;
        graph.marginal_covariance(&graph.initial(), &query, options)
    }
    /// Optimize with new observations, then marginalize timestamps strictly older
    /// than now-lag. On any error, owned estimates, factors and time are unchanged.
    /// All factors must reference live or newly supplied keys, never expired keys.
    pub fn update(&mut self, now: f64, update: SmootherUpdate) -> Result<FixedLagReport, Error> {
        if !now.is_finite() || self.last_time.is_some_and(|t| now < t) {
            return Err(Error::InvalidOptions);
        }
        let cutoff = now - self.options.lag;
        if !cutoff.is_finite() {
            return Err(Error::InvalidOptions);
        }
        let mut variables = self.variables.clone();
        for v in update.variables {
            v.kind.validate(&v.value)?;
            if variables.contains_key(&v.key) {
                return Err(Error::Evaluation("duplicate live key".into()));
            }
            if v.timestamp
                .is_some_and(|t| !t.is_finite() || t > now || t < cutoff)
            {
                return Err(Error::Evaluation(
                    "new variable timestamp is outside the window".into(),
                ));
            }
            variables.insert(v.key, v);
        }
        let mut refreshed = BTreeSet::new();
        for (key, time) in update.timestamps {
            let v = variables.get_mut(&key).ok_or(Error::InvalidDimensions)?;
            if !refreshed.insert(key)
                || !time.is_finite()
                || time > now
                || time < cutoff
                || v.timestamp.is_some_and(|t| time < t)
            {
                return Err(Error::InvalidOptions);
            }
            v.timestamp = Some(time);
        }
        let mut factors = self.factors.clone();
        factors.extend(update.factors);
        let order = variables.keys().copied().collect::<Vec<_>>();
        let (graph, ids) = build_graph(&variables, &factors, &order)?;
        let (_, n, m) = graph.dimensions();
        // Bound dense window/augmented-QR matrices too, before evaluating callbacks.
        self.options
            .marginalization
            .check_size(m.checked_add(n).ok_or(Error::InvalidDimensions)?, n)?;
        let solved = solve_local(&graph, graph.initial(), &self.options.solver)?;
        if matches!(
            solved.termination,
            Termination::NoProgress | Termination::MaxIterations
        ) {
            return Err(Error::Evaluation(format!(
                "fixed-lag optimization did not converge: {:?}",
                solved.termination
            )));
        }
        for (&key, &id) in &ids {
            variables.get_mut(&key).unwrap().value = graph.value(&solved.x, id)?;
        }
        let expired = variables
            .iter()
            .filter_map(|(&key, v)| v.timestamp.filter(|&t| t < cutoff).map(|_| key))
            .collect::<BTreeSet<_>>();
        let mut constant_cost = self.constant_cost;
        let summary = marginalize(
            &mut variables,
            &mut factors,
            &expired,
            &self.options,
            &mut constant_cost,
        )?;
        let estimates = variables
            .iter()
            .map(|(&key, v)| (key, v.value.clone()))
            .collect();
        let initial_cost = solved.initial_cost + self.constant_cost;
        let final_cost = solved.cost + self.constant_cost;
        if !initial_cost.is_finite() || !final_cost.is_finite() || !constant_cost.is_finite() {
            return Err(Error::NonFinite);
        }
        self.variables = variables;
        self.factors = factors;
        self.last_time = Some(now);
        self.constant_cost = constant_cost;
        Ok(FixedLagReport {
            estimates,
            marginalized: expired.into_iter().collect(),
            initial_cost,
            final_cost,
            termination: solved.termination,
            iterations: solved.iterations,
            marginalization: summary,
        })
    }
}
pub(crate) fn build_graph(
    variables: &BTreeMap<Key, WindowVariable>,
    factors: &[SmootherFactor],
    order: &[Key],
) -> Result<(FactorGraph, BTreeMap<Key, VariableId>), Error> {
    crate::stream_profile::measure(0, || {
        let mut graph = FactorGraph::new();
        let mut ids = BTreeMap::new();
        for key in order {
            let v = variables.get(key).ok_or(Error::InvalidDimensions)?;
            let id = graph.add_variable(v.kind, v.value.clone())?;
            graph.set_fixed(id, v.fixed)?;
            ids.insert(*key, id);
        }
        for f in factors {
            let connected = f
                .keys
                .iter()
                .map(|key| ids.get(key).copied().ok_or(Error::InvalidDimensions))
                .collect::<Result<Vec<_>, _>>()?;
            f.register(&mut graph, &connected)?;
        }
        Ok((graph, ids))
    })
}
fn marginalize(
    variables: &mut BTreeMap<Key, WindowVariable>,
    factors: &mut Vec<SmootherFactor>,
    expired: &BTreeSet<Key>,
    options: &FixedLagOptions,
    constant_cost: &mut f64,
) -> Result<MarginalizationSummary, Error> {
    let touched = factors
        .iter()
        .filter(|f| f.keys.iter().any(|k| expired.contains(k)))
        .cloned()
        .collect::<Vec<_>>();
    let connected = touched
        .iter()
        .flat_map(|f| f.keys.iter().copied())
        .collect::<BTreeSet<_>>();
    let eliminated = connected
        .iter()
        .filter(|k| expired.contains(k) && !variables[k].fixed)
        .copied()
        .collect::<Vec<_>>();
    let separator = connected
        .iter()
        .filter(|k| !expired.contains(k) && !variables[k].fixed)
        .copied()
        .collect::<Vec<_>>();
    let mut summary = MarginalizationSummary {
        removed_factors: touched.len(),
        separator_variables: separator.len(),
        ..MarginalizationSummary::default()
    };
    if !touched.is_empty() {
        let mut order = eliminated
            .iter()
            .chain(&separator)
            .copied()
            .collect::<Vec<_>>();
        order.extend(connected.iter().filter(|k| variables[k].fixed).copied());
        let (graph, _) = build_graph(variables, &touched, &order)?;
        let (_, n, m) = graph.dimensions();
        options.marginalization.check_size(m, n)?;
        let mut residual = DVector::zeros(m);
        let mut j = DMatrix::zeros(m, n);
        graph.evaluate_local(&graph.initial(), &mut residual, &mut j)?;
        match options.marginalization_loss {
            MarginalizationLoss::Gaussian => {
                if touched.iter().any(|f| {
                    !matches!(f.options.loss.unwrap_or(options.solver.loss), Loss::Squared)
                }) {
                    return Err(Error::Evaluation(
                        "robust marginalization requires explicit FrozenIrls".into(),
                    ));
                }
            }
            MarginalizationLoss::FrozenIrls => {
                let blocks = crate::solver::validated_loss_blocks(&graph, m)?;
                let (cost, weights) =
                    crate::solver::robust_weights(&residual, options.solver.loss, &blocks)?;
                if !cost.is_finite() {
                    return Err(Error::NonFinite);
                }
                for row in 0..m {
                    residual[row] *= weights[row];
                    j.row_mut(row).scale_mut(weights[row]);
                }
                // Keep the robust objective offset at this linearization point.
                *constant_cost += cost - 0.5 * residual.norm_squared();
            }
        }
        let ne = eliminated
            .iter()
            .map(|k| variables[k].kind.dimensions().1)
            .sum();
        summary.eliminated_tangent = ne;
        summary.separator_tangent = n - ne;
        let prior = marginalize_linearized(&j, &residual, ne, &options.marginalization)?;
        if separator.is_empty() {
            *constant_cost += 0.5 * prior.residual.norm_squared();
        } else {
            let kinds = separator
                .iter()
                .map(|k| variables[k].kind)
                .collect::<Vec<_>>();
            let anchors = separator
                .iter()
                .map(|k| variables[k].value.clone())
                .collect::<Vec<_>>();
            summary.prior_rows = prior.residual.len();
            let rows = prior.residual.len();
            factors.push(SmootherFactor::autodiff(
                &separator,
                rows,
                32,
                FactorOptions {
                    loss: Some(Loss::Squared),
                    ..FactorOptions::default()
                },
                move |values| {
                    let mut delta = Vec::new();
                    for ((&kind, anchor), value) in kinds.iter().zip(&anchors).zip(values) {
                        delta.extend(kind.local_autodiff(anchor, value)?);
                    }
                    Ok((0..rows)
                        .map(|row| {
                            delta.iter().enumerate().fold(
                                DynamicJet::constant(prior.residual[row]),
                                |sum, (col, v)| sum + prior.jacobian[(row, col)] * v,
                            )
                        })
                        .collect())
                },
            ));
        }
    }
    factors.retain(|f| !f.keys.iter().any(|k| expired.contains(k)));
    variables.retain(|key, _| !expired.contains(key));
    Ok(summary)
}
