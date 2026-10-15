//! Initial iSAM-style incremental square-root smoothing.
//! New Gaussian rows update cached sparse R using Givens rotations. Default
//! threshold and periodic refreshes run the batch solver. An explicit selective
//! mode reevaluates incident factors, retaining unrelated rows, then rebuilds QR.
//! An optional Gaussian Bayes tree selectively re-eliminates appended row paths.
mod probe;
mod selective;
use crate::bayes_tree::{
    BayesTreeUpdate, EditableBayesTree, EditableBayesTreeUpdate, GaussianBayesTree,
};
use crate::factor::{FactorGraph, VariableId};
use crate::fixed_lag::{Key, SmootherFactor, SmootherUpdate, WindowVariable, build_graph};
use crate::incremental_qr::{IncrementalQr, IncrementalQrOptions};
use crate::{
    Algorithm, DVector, Error, LinearSolver, LocalProblem, Loss, Options, QrOptions, Termination,
    solver::{GaussianSolution, solve_local_gaussian_model},
};
pub use selective::SelectiveRelinearizationReport;
use selective::{FactorCache, SelectiveContext, threshold_keys};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct IsamOptions {
    pub relinearization_threshold: f64,
    /// Full refresh at this update interval. Earlier threshold refreshes use the
    /// full batch solver unless selective relinearization was explicitly enabled.
    pub relinearization_interval: usize,
    pub qr: IncrementalQrOptions,
    pub batch_solver: Options,
}
impl Default for IsamOptions {
    fn default() -> Self {
        Self {
            relinearization_threshold: 0.1,
            relinearization_interval: 10,
            qr: IncrementalQrOptions::default(),
            batch_solver: Options {
                linear_solver: LinearSolver::DenseQr(QrOptions::default()),
                ..Options::default()
            },
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsamUpdateKind {
    Incremental,
    /// Full batch refresh or explicit threshold-selected factor evaluation.
    /// selective_update distinguishes the bounded selective path.
    Relinearized,
}
/// Why nonlinear factors were relinearized. Explicit requests take
/// precedence over the interval; the threshold is checked only on cached solves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsamRelinearizationReason {
    Forced,
    Interval,
    Threshold,
    /// Explicitly requested variable anchors; does not advance the arrival count.
    Requested,
}
/// Explicit full nonlinear evaluation at the committed estimates. Gradient
/// measures use current whitened right-tangent Jacobian columns, not cached R.
#[derive(Debug, Clone)]
pub struct IsamNonlinearSummary {
    pub cost: f64,
    pub gradient_norm: f64,
    pub norm: crate::GradientNorm,
    /// Per nonfixed variable, the maximum measured tangent gradient component.
    pub variable_gradients: BTreeMap<Key, f64>,
}
#[derive(Debug)]
pub struct IsamReport {
    pub estimates: BTreeMap<Key, DVector<f64>>,
    pub kind: IsamUpdateKind,
    pub relinearization_reason: Option<IsamRelinearizationReason>,
    /// Distinct recipes evaluated for the installed update: new factors, selected
    /// factors, or the whole graph. Excludes batch LM callback counts and repeated
    /// selective passes; selective_update reports those separately.
    pub factors_linearized: usize,
    pub stored_entries: usize,
    pub linearized_cost: f64,
    pub max_anchor_delta: f64,
    /// LM/GN trials in a full batch refresh; zero on cached/selected updates or initial convergence.
    pub batch_trials: usize,
    /// Effective initial/final LM damping; None on cached updates and Gauss-Newton.
    pub batch_initial_damping: Option<f64>,
    pub batch_final_damping: Option<f64>,
    /// Accepted optional frozen-QR Gaussian refinement steps after batch LM.
    pub refinement_steps: usize,
    /// Optional scalar-frontal Gaussian tree work. Full nonlinear refreshes still
    /// rebuild every conditional; cached updates re-eliminate affected paths.
    pub bayes_tree_update: Option<BayesTreeUpdate>,
    /// Optional threshold-selected factor evaluation, preserving unrelated rows.
    /// The ordinary selected mode rebuilds QR globally; the explicit editable
    /// mode reports affected clique replacement in editable_bayes_tree_update.
    pub selective_update: Option<SelectiveRelinearizationReport>,
    /// Original-factor replacement through cached separator messages. Counts
    /// scalar Gaussian cliques; symbolic analysis/back-substitution remain global.
    pub editable_bayes_tree_update: Option<EditableBayesTreeUpdate>,
}
#[derive(Clone)]
enum GaussianCache {
    Qr(IncrementalQr),
    Tree(GaussianBayesTree),
    Editable(EditableBayesTree),
}
impl GaussianCache {
    fn into_tree(self) -> Result<Self, Error> {
        match self {
            Self::Qr(qr) => Ok(Self::Tree(GaussianBayesTree::from_qr(qr)?)),
            Self::Editable(_) => Err(Error::InvalidOptions),
            tree => Ok(tree),
        }
    }
    fn tree_update(&self) -> Option<BayesTreeUpdate> {
        match self {
            Self::Qr(_) => None,
            Self::Editable(_) => None,
            Self::Tree(tree) => Some(tree.last_update()),
        }
    }
    fn editable_update(&self) -> Option<EditableBayesTreeUpdate> {
        match self {
            Self::Editable(tree) => Some(tree.last_update()),
            _ => None,
        }
    }
    fn column_permutation(&self) -> Option<&[usize]> {
        match self {
            Self::Qr(qr) => qr.column_permutation(),
            Self::Tree(tree) => Some(tree.column_permutation()),
            Self::Editable(tree) => Some(tree.column_permutation()),
        }
    }
    fn extend_columns(&mut self, n: usize) -> Result<(), Error> {
        match self {
            Self::Qr(qr) => qr.extend_columns(n),
            Self::Tree(tree) => tree.extend_columns(n),
            Self::Editable(tree) => tree.extend_columns(n),
        }
    }
    fn append_in_place(
        &mut self,
        j: &crate::sparse::CsrMatrix,
        r: &DVector<f64>,
    ) -> Result<(), Error> {
        match self {
            Self::Qr(qr) => qr.append_in_place(j, r),
            Self::Tree(tree) => tree.append_in_place(j, r),
            Self::Editable(_) => Err(Error::InvalidOptions),
        }
    }
    fn rebuild_in_place(
        &mut self,
        j: &crate::sparse::CsrMatrix,
        r: &DVector<f64>,
    ) -> Result<(), Error> {
        match self {
            Self::Qr(qr) => qr.rebuild_in_place(j, r),
            Self::Tree(_) | Self::Editable(_) => Err(Error::InvalidOptions),
        }
    }
    fn set_column_order(&mut self, order: Vec<usize>) -> Result<(), Error> {
        match self {
            Self::Qr(qr) => qr.set_column_order(order),
            Self::Tree(_) | Self::Editable(_) => Err(Error::InvalidOptions),
        }
    }
    fn solve(&self) -> Result<DVector<f64>, Error> {
        match self {
            Self::Qr(qr) => qr.solve(),
            Self::Tree(tree) => tree.solve(),
            Self::Editable(tree) => tree.solve(),
        }
    }
    fn solve_gradient(&self, g: &DVector<f64>) -> Result<DVector<f64>, Error> {
        match self {
            Self::Qr(qr) => qr.solve_gradient(g),
            Self::Tree(_) | Self::Editable(_) => Err(Error::InvalidOptions),
        }
    }
    fn cost(&self, x: &DVector<f64>) -> Result<f64, Error> {
        match self {
            Self::Qr(qr) => qr.cost(x),
            Self::Tree(tree) => tree.cost(x),
            Self::Editable(tree) => tree.cost(x),
        }
    }
    fn stored_entries(&self) -> usize {
        match self {
            Self::Qr(qr) => qr.stored_entries(),
            Self::Tree(tree) => tree.stored_entries(),
            Self::Editable(tree) => tree.stored_entries(),
        }
    }
}
pub struct IsamSmoother {
    options: IsamOptions,
    variables: BTreeMap<Key, WindowVariable>,
    anchors: BTreeMap<Key, WindowVariable>,
    factors: Vec<SmootherFactor>,
    order: Vec<Key>,
    qr: GaussianCache,
    use_bayes_tree: bool,
    updates: usize,
    next_batch_damping: Option<f64>,
    reuse_batch_ordering: bool,
    max_refinement_steps: usize,
    selective_cache: Option<FactorCache>,
}
impl IsamSmoother {
    pub fn new(options: IsamOptions) -> Result<Self, Error> {
        if !options.relinearization_threshold.is_finite()
            || options.relinearization_threshold <= 0.
            || options.relinearization_interval == 0
            || !matches!(options.batch_solver.loss, Loss::Squared)
        {
            return Err(Error::InvalidOptions);
        }
        let qr = GaussianCache::Qr(IncrementalQr::new(0, options.qr)?);
        Ok(Self {
            options,
            variables: BTreeMap::new(),
            anchors: BTreeMap::new(),
            factors: Vec::new(),
            order: Vec::new(),
            qr,
            use_bayes_tree: false,
            updates: 0,
            next_batch_damping: None,
            reuse_batch_ordering: false,
            max_refinement_steps: 0,
            selective_cache: None,
        })
    }
    /// Opt into fixed-order Gaussian Bayes-tree re-elimination for cached rows.
    /// Existing unaffected conditional branches are shared on row append.
    /// Nonlinear refreshes, including explicit selective factor evaluation,
    /// rebuild the full tree. Back-substitution and bookkeeping remain global;
    /// no iSAM2 performance equivalence or selective row replacement is implied.
    pub fn with_bayes_tree(mut self) -> Result<Self, Error> {
        self.qr = self.qr.into_tree()?;
        self.use_bayes_tree = true;
        Ok(self)
    }
    /// Opt into threshold-selected nonlinear factor relinearization, before the
    /// first update. Factors incident on selected variables are evaluated at a
    /// mixed anchor: selected variables move to their current estimates; all
    /// other anchors stay fixed. The full Gaussian QR is then rebuilt.
    ///
    /// This performs bounded undamped Gaussian steps, not a converged batch
    /// solve or selective Bayes-tree factor replacement. Check threshold_remaining
    /// in the report; a small anchor delta does not imply nonlinear stationarity.
    /// Forced and interval refreshes retain the full batch policy. Cached residual
    /// and nonzero coefficient counts together are limited by qr.max_entries.
    pub fn with_selective_relinearization(mut self, max_passes: usize) -> Result<Self, Error> {
        if self.updates != 0 || max_passes == 0 || max_passes > 200 {
            return Err(Error::InvalidOptions);
        }
        self.selective_cache = Some(FactorCache::new(max_passes));
        Ok(self)
    }
    /// Opt into selected nonlinear factor evaluation with affected Gaussian
    /// clique replacement, before the first update. Unaffected separator
    /// messages/conditionals are shared; symbolic analysis, coordinate handling
    /// and back-substitution remain global. Pass bounds and nonlinear convergence
    /// limitations are the same as with_selective_relinearization.
    /// Forced/interval refreshes rebuild the full original-factor tree.
    /// This selects the editable backend instead of the append-only Bayes tree.
    pub fn with_selective_bayes_tree(self, max_passes: usize) -> Result<Self, Error> {
        let mut smoother = self.with_selective_relinearization(max_passes)?;
        smoother.qr = GaussianCache::Editable(EditableBayesTree::new(0, smoother.options.qr)?);
        smoother.use_bayes_tree = false;
        Ok(smoother)
    }
    /// Opt into carrying adaptive LM damping between successful full refreshes.
    /// The first refresh uses the configured initial value. Ordinary updates and
    /// failed refreshes leave this state unchanged. The default restarts damping.
    /// Rejections still adapt damping normally; no tolerance/backend fallback occurs.
    pub fn with_batch_damping_reuse(mut self) -> Result<Self, Error> {
        if !matches!(
            self.options.batch_solver.algorithm,
            Algorithm::LevenbergMarquardt
        ) || !self.options.batch_solver.initial_damping.is_finite()
            || self.options.batch_solver.initial_damping <= 0.
        {
            return Err(Error::InvalidOptions);
        }
        if self.next_batch_damping.is_none() {
            self.next_batch_damping = Some(self.options.batch_solver.initial_damping);
        }
        Ok(self)
    }
    /// None means the default restart policy; Some is the next refresh's LM damping.
    pub fn next_batch_damping(&self) -> Option<f64> {
        self.next_batch_damping
    }
    /// Opt into using the sparse batch solver's elimination ordering for fresh
    /// QR caches. Original parameter coordinates and column extension are kept.
    /// A refresh converged without a linear solve extends the previous ordering.
    pub fn with_batch_ordering_reuse(mut self) -> Result<Self, Error> {
        if !matches!(
            self.options.batch_solver.linear_solver,
            LinearSolver::SparseLdl(_) | LinearSolver::SparseQr(_)
        ) {
            return Err(Error::InvalidOptions);
        }
        self.reuse_batch_ordering = true;
        Ok(self)
    }
    /// Opt into bounded frozen-QR refinement of an accepted Gaussian batch
    /// solution. Every accepted step requires positive predicted and actual
    /// improvement and a smaller gradient measure. The QR factor stays at the
    /// batch solution's anchor; cached updates and linearized cost use that anchor.
    /// Backtracking is bounded to eight trials per step. Refinement stops at
    /// one hundredth of the configured gradient tolerance, or on a small step.
    pub fn with_final_refinement(mut self, max_steps: usize) -> Result<Self, Error> {
        if max_steps == 0 || max_steps > 200 {
            return Err(Error::InvalidOptions);
        }
        self.max_refinement_steps = max_steps;
        Ok(self)
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
    /// Evaluate every installed nonlinear factor at committed estimates. This
    /// explicit probe neither changes anchors/caches nor advances the update
    /// counter. Callbacks run again; it is not an inexpensive cached cost query.
    /// A small gradient is a stationarity measure, not an observability certificate.
    pub fn nonlinear_summary(
        &self,
        norm: crate::GradientNorm,
    ) -> Result<IsamNonlinearSummary, Error> {
        probe::summary(&self.variables, &self.factors, &self.order, norm)
    }
    /// Reevaluate factors incident on explicitly supplied nonfixed keys and
    /// solve the resulting Gaussian model. Requires a selected-factor mode and
    /// unique, known, nonempty keys. Its configured bounded threshold passes
    /// can additionally select other variables. No observation is appended and
    /// the arrival counter/damping schedule stays unchanged. Owned state is
    /// unchanged on any error; callback side effects cannot be rolled back.
    pub fn relinearize_selected(&mut self, keys: &[Key]) -> Result<IsamReport, Error> {
        let mut selected = std::collections::BTreeSet::new();
        if keys.is_empty() || self.selective_cache.is_none() {
            return Err(Error::InvalidOptions);
        }
        for &key in keys {
            if !selected.insert(key)
                || self
                    .variables
                    .get(&key)
                    .is_none_or(|variable| variable.fixed)
            {
                return Err(Error::InvalidDimensions);
            }
        }
        let mut variables = self.variables.clone();
        let mut anchors = self.anchors.clone();
        let mut cache = self.selective_cache.clone().unwrap();
        let result = cache.relinearize(
            &mut variables,
            &mut anchors,
            SelectiveContext {
                factors: &self.factors,
                order: &self.order,
                options: &self.options,
                new_count: 0,
            },
            selected,
            self.qr.clone(),
        )?;
        let mut qr = result.qr;
        let linearized_cost = qr.cost(&result.coordinates.0)?;
        if self.use_bayes_tree {
            qr = qr.into_tree()?;
        }
        let report = IsamReport {
            estimates: variables
                .iter()
                .map(|(&key, value)| (key, value.value.clone()))
                .collect(),
            kind: IsamUpdateKind::Relinearized,
            relinearization_reason: Some(IsamRelinearizationReason::Requested),
            factors_linearized: result.distinct_factors,
            stored_entries: qr.stored_entries(),
            linearized_cost,
            max_anchor_delta: result.coordinates.1,
            batch_trials: 0,
            batch_initial_damping: None,
            batch_final_damping: None,
            refinement_steps: 0,
            bayes_tree_update: qr.tree_update(),
            selective_update: Some(result.report),
            editable_bayes_tree_update: qr.editable_update(),
        };
        self.variables = variables;
        self.anchors = anchors;
        self.selective_cache = Some(cache);
        self.qr = qr;
        Ok(report)
    }
    /// Append variables/factors. Timestamps are not used by this all-history
    /// smoother; timestamp refresh requests are rejected. Fixed flags cannot change.
    /// Without a full refresh, estimates solve a cached linearization, not the
    /// fully relinearized nonlinear objective. Owned state is unchanged on error.
    /// Forced/interval refreshes start from committed estimates and new seeds,
    /// without first evaluating new factors or solving at potentially stale anchors.
    /// Threshold-triggered refreshes start from the successful cached solve.
    /// In selective mode, unrelated factors keep their previous charts/rows,
    /// while incident factors use the updated product of per-variable anchors.
    pub fn update(
        &mut self,
        update: SmootherUpdate,
        force_relinearize: bool,
    ) -> Result<IsamReport, Error> {
        if !update.timestamps.is_empty() {
            return Err(Error::InvalidOptions);
        }
        if update
            .factors
            .iter()
            .any(|f| !matches!(f.options.loss.unwrap_or(Loss::Squared), Loss::Squared))
        {
            return Err(Error::Evaluation(
                "initial iSAM backend supports Gaussian factors only".into(),
            ));
        }
        let updates = self.updates.checked_add(1).ok_or(Error::InvalidOptions)?;
        let mut variables = self.variables.clone();
        let mut anchors = self.anchors.clone();
        let mut order = self.order.clone();
        let mut selective_cache = self.selective_cache.clone();
        let editable_mode = matches!(self.qr, GaussianCache::Editable(_));
        for v in update.variables {
            v.kind.validate(&v.value)?;
            if variables.contains_key(&v.key) {
                return Err(Error::Evaluation("duplicate live key".into()));
            }
            order.push(v.key);
            anchors.insert(v.key, v.clone());
            variables.insert(v.key, v);
        }
        let (new_graph, new_ids) = build_graph(&anchors, &update.factors, &order)?;
        let (_, n, _) = new_graph.dimensions();
        let new_count = update.factors.len();
        let mut reason = if force_relinearize {
            Some(IsamRelinearizationReason::Forced)
        } else if updates.is_multiple_of(self.options.relinearization_interval) {
            Some(IsamRelinearizationReason::Interval)
        } else {
            None
        };
        // A scheduled rebuild does not depend on observability or residual
        // validity at obsolete anchors. Do not copy/solve the discarded cache.
        let mut qr = if reason.is_some() {
            GaussianCache::Qr(IncrementalQr::new(n, self.options.qr)?)
        } else {
            self.qr.clone()
        };
        let mut cached_coordinates = None;
        let mut selected_keys = None;
        if reason.is_none() {
            qr.extend_columns(n)?;
            let (r, j) = new_graph.evaluate_sparse(&new_graph.initial())?;
            if let Some(cache) = &mut selective_cache {
                cache.append(&j, &r, &update.factors, self.options.qr.max_entries)?;
            }
            if let GaussianCache::Editable(tree) = &mut qr {
                selective_cache
                    .as_ref()
                    .unwrap()
                    .install_new(tree, new_count)?;
            } else {
                qr.append_in_place(&j, &r)?;
            }
            let delta = qr.solve()?;
            let state = new_graph.retract(&new_graph.initial(), &delta)?;
            // This graph already has the full insertion-order variable layout.
            for (&key, &id) in &new_ids {
                variables.get_mut(&key).unwrap().value = new_graph.value(&state, id)?;
            }
            let raw_delta = new_graph
                .tangent_blocks()?
                .iter()
                .map(|range| delta.rows(range.start, range.len()).norm())
                .fold(0.0_f64, f64::max);
            let coordinates = anchor_coordinates(&variables, &anchors, &new_graph, &new_ids)?;
            let max_delta = coordinates.1.max(raw_delta);
            if max_delta > self.options.relinearization_threshold {
                reason = Some(IsamRelinearizationReason::Threshold);
                if selective_cache.is_some() {
                    selected_keys = Some(threshold_keys(
                        &variables,
                        &anchors,
                        &new_graph,
                        &new_ids,
                        &delta,
                        self.options.relinearization_threshold,
                    )?);
                }
            } else {
                cached_coordinates = Some(coordinates);
            }
        }
        let refresh = reason.is_some() && selected_keys.is_none();
        let mut selective_update = None;
        let mut selective_factors_linearized = 0;
        let mut next_batch_damping = self.next_batch_damping;
        let (mut batch_trials, mut batch_initial_damping, mut batch_final_damping) =
            (0, None, None);
        let mut refinement_steps = 0;
        if let Some(keys) = selected_keys {
            let factors = self
                .factors
                .iter()
                .chain(&update.factors)
                .cloned()
                .collect::<Vec<_>>();
            let result = selective_cache.as_mut().unwrap().relinearize(
                &mut variables,
                &mut anchors,
                SelectiveContext {
                    factors: &factors,
                    order: &order,
                    options: &self.options,
                    new_count,
                },
                keys,
                qr,
            )?;
            qr = result.qr;
            cached_coordinates = Some(result.coordinates);
            selective_factors_linearized = result.distinct_factors;
            selective_update = Some(result.report);
        } else if refresh {
            // Keep the committed recipes untouched until success; ordinary
            // updates never clone or register the old factor history.
            let factors = self
                .factors
                .iter()
                .chain(&update.factors)
                .cloned()
                .collect::<Vec<_>>();
            let (graph, ids) = build_graph(&variables, &factors, &order)?;
            let (_, n, m) = graph.dimensions();
            if matches!(
                self.options.batch_solver.linear_solver,
                LinearSolver::Dense | LinearSolver::DenseQr(_) | LinearSolver::DenseSchur { .. }
            ) && m
                .checked_add(n)
                .and_then(|rows| rows.checked_mul(n))
                .is_none_or(|entries| entries > self.options.qr.max_entries)
            {
                return Err(Error::Evaluation(
                    "iSAM batch refresh matrix entry limit exceeded".into(),
                ));
            }
            let mut batch_options = self.options.batch_solver.clone();
            if let Some(damping) = next_batch_damping {
                batch_options.initial_damping = damping;
            }
            let GaussianSolution {
                report: solved,
                residual: mut r,
                jacobian: mut j,
                final_damping,
                column_permutation,
            } = solve_local_gaussian_model(
                &graph,
                graph.initial(),
                &batch_options,
                self.reuse_batch_ordering,
            )?;
            let stalled = solved.termination == Termination::NoProgress;
            if solved.termination == Termination::MaxIterations
                || (stalled && self.max_refinement_steps == 0)
            {
                return Err(Error::Evaluation(format!(
                    "iSAM batch refresh did not converge: {:?}, trials={}, cost={:.17e}, last gradient={:?}",
                    solved.termination,
                    solved.iterations.len(),
                    solved.cost,
                    solved.iterations.last().map(|it| it.gradient_norm),
                )));
            }
            batch_trials = solved.iterations.len();
            if matches!(batch_options.algorithm, Algorithm::LevenbergMarquardt) {
                batch_initial_damping = Some(batch_options.initial_damping);
                batch_final_damping = Some(final_damping);
            }
            if next_batch_damping.is_some() {
                next_batch_damping = Some(final_damping);
            }
            let mut state = solved.x;
            let cache_state = (self.max_refinement_steps != 0).then(|| state.clone());
            let mut final_gradient_norm = if stalled {
                crate::solver::gaussian_gradient_norm(&j, &r, batch_options.gradient_norm)?
            } else {
                0.
            };
            // The solver already evaluated the model at its final accepted state.
            // Squared loss leaves whitening intact with unit IRLS weights.
            qr = GaussianCache::Qr(IncrementalQr::new(n, self.options.qr)?);
            if self.reuse_batch_ordering {
                if let Some(permutation) = column_permutation {
                    qr.set_column_order(permutation)?;
                } else if let Some(previous) = self.qr.column_permutation() {
                    if previous.len() > n {
                        return Err(Error::InvalidDimensions);
                    }
                    let mut permutation = previous.to_vec();
                    permutation.extend(previous.len()..n);
                    qr.set_column_order(permutation)?;
                }
            }
            qr.rebuild_in_place(&j, &r)?;
            if let Some(cache) = &mut selective_cache {
                cache.reset(&j, &r, &factors, self.options.qr.max_entries)?;
            }
            // Require an observable undamped model even when batch LM used damping.
            let mut step = qr.solve()?;
            let mut gradient = if self.max_refinement_steps != 0 {
                j.transpose_multiply(&r)?
            } else {
                DVector::zeros(0)
            };
            let mut previous_gradient_norm = if self.max_refinement_steps != 0 {
                crate::solver::gaussian_gradient_measure(
                    &j,
                    &gradient,
                    batch_options.gradient_norm,
                )?
            } else {
                0.
            };
            for _ in 0..self.max_refinement_steps {
                if step.norm() <= self.options.batch_solver.step_tolerance {
                    break;
                }
                if previous_gradient_norm <= batch_options.gradient_tolerance * 1e-2 {
                    break;
                }
                let js = j.multiply(&step)?;
                let predicted = -gradient.dot(&step) - 0.5 * js.norm_squared();
                if !predicted.is_finite() {
                    return Err(Error::NonFinite);
                }
                if predicted <= 0. {
                    break;
                }
                let mut accepted = None;
                let mut fraction = 1.0;
                for _ in 0..8 {
                    let scaled = &step * fraction;
                    let scaled_prediction = -fraction * gradient.dot(&step)
                        - 0.5 * fraction * fraction * js.norm_squared();
                    if !scaled_prediction.is_finite() {
                        return Err(Error::NonFinite);
                    }
                    if scaled_prediction <= 0. {
                        break;
                    }
                    let trial = graph.retract(&state, &scaled)?;
                    let (trial_r, trial_j) = graph.evaluate_sparse(&trial)?;
                    let actual = crate::solver::gaussian_reduction(&r, &trial_r)?;
                    let gain_ratio = actual / scaled_prediction;
                    if !gain_ratio.is_finite() {
                        return Err(Error::NonFinite);
                    }
                    let trial_gradient = trial_j.transpose_multiply(&trial_r)?;
                    let trial_gradient_norm = crate::solver::gaussian_gradient_measure(
                        &trial_j,
                        &trial_gradient,
                        batch_options.gradient_norm,
                    )?;
                    if actual > 0.
                        && gain_ratio > 0.
                        && trial_gradient_norm < previous_gradient_norm
                    {
                        accepted =
                            Some((trial, trial_r, trial_j, trial_gradient, trial_gradient_norm));
                        break;
                    }
                    fraction *= 0.5;
                }
                let Some((trial, trial_r, trial_j, trial_gradient, trial_gradient_norm)) = accepted
                else {
                    break;
                };
                final_gradient_norm = trial_gradient_norm;
                previous_gradient_norm = trial_gradient_norm;
                state = trial;
                j = trial_j;
                r = trial_r;
                gradient = trial_gradient;
                step = qr.solve_gradient(&gradient)?;
                refinement_steps += 1;
            }
            if stalled {
                let norm = final_gradient_norm;
                if norm > batch_options.gradient_tolerance {
                    return Err(Error::Evaluation(format!(
                        "iSAM Gaussian refinement did not meet gradient target: norm={norm:.17e}, target={:.17e}, accepted={refinement_steps}",
                        batch_options.gradient_tolerance
                    )));
                }
            }
            for (&key, &id) in &ids {
                variables.get_mut(&key).unwrap().value = graph.value(&state, id)?;
            }
            anchors = variables.clone();
            if refinement_steps != 0 {
                for (&key, &id) in &ids {
                    anchors.get_mut(&key).unwrap().value =
                        graph.value(cache_state.as_ref().unwrap(), id)?;
                }
            }
        }
        // Share principal local coordinates between threshold, report and cost.
        // The raw QR step can wind beyond the principal chart; keep its separate
        // threshold check. Refreshes changed anchors and must recompute Local,
        // even at identical poses where floating Lie operations may be nonzero.
        let (delta, max_anchor_delta) = match cached_coordinates {
            Some(coordinates) => coordinates,
            None => anchor_coordinates(&variables, &anchors, &new_graph, &new_ids)?,
        };
        if editable_mode && !matches!(qr, GaussianCache::Editable(_)) {
            qr = GaussianCache::Editable(selective_cache.as_ref().unwrap().build_tree(
                n,
                qr.column_permutation(),
                self.options.qr,
            )?);
        }
        let linearized_cost = qr.cost(&delta)?;
        let estimates = variables
            .iter()
            .map(|(&k, v)| (k, v.value.clone()))
            .collect();
        let stored_entries = qr.stored_entries();
        if self.use_bayes_tree {
            qr = qr.into_tree()?;
        }
        let bayes_tree_update = qr.tree_update();
        let editable_bayes_tree_update = qr.editable_update();
        self.variables = variables;
        self.anchors = anchors;
        self.order = order;
        self.factors.extend(update.factors);
        self.qr = qr;
        self.updates = updates;
        self.next_batch_damping = next_batch_damping;
        self.selective_cache = selective_cache;
        Ok(IsamReport {
            estimates,
            kind: if reason.is_some() {
                IsamUpdateKind::Relinearized
            } else {
                IsamUpdateKind::Incremental
            },
            relinearization_reason: reason,
            factors_linearized: if refresh {
                self.factors.len()
            } else if selective_update.is_some() {
                selective_factors_linearized
            } else {
                new_count
            },
            stored_entries,
            linearized_cost,
            max_anchor_delta,
            batch_trials,
            batch_initial_damping,
            batch_final_damping,
            refinement_steps,
            bayes_tree_update,
            selective_update,
            editable_bayes_tree_update,
        })
    }
}
fn anchor_coordinates(
    values: &BTreeMap<Key, WindowVariable>,
    anchors: &BTreeMap<Key, WindowVariable>,
    graph: &FactorGraph,
    ids: &BTreeMap<Key, VariableId>,
) -> Result<(DVector<f64>, f64), Error> {
    let mut delta = DVector::zeros(graph.dimensions().1);
    let mut maximum = 0.0_f64;
    // Key order matches the previous norm reduction. Column ranges follow graph
    // insertion order, which need not match the numeric keys.
    for (&key, &id) in ids {
        if let Some(range) = graph.tangent_range(id)? {
            let v = &values[&key];
            let local = v.kind.local(&anchors[&key].value, &v.value)?;
            maximum = maximum.max(local.norm());
            delta.rows_mut(range.start, range.len()).copy_from(&local);
        }
    }
    if !maximum.is_finite() {
        return Err(Error::NonFinite);
    }
    Ok((delta, maximum))
}
