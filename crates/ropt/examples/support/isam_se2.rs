//! Nonlinear accuracy validation, deliberately separate from timing benchmarks.
use nalgebra::{Matrix3, Vector2, Vector3};
use ropt::{
    DMatrix, DVector, LinearSolver, Options, Termination,
    factor::*,
    fixed_lag::*,
    isam::*,
    manifold::*,
    noise::NoiseModel,
    pose_graph::*,
    solve_local,
    sparse_direct::{SparseLdlOptions, SparseOrdering},
};
use std::{collections::BTreeMap, io::Write};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy)]
struct Edge {
    a: usize,
    b: usize,
    measured: SE2,
    information: Matrix3<f64>,
}
impl Edge {
    fn options(self) -> Result<FactorOptions> {
        Ok(FactorOptions {
            noise: NoiseModel::from_information(DMatrix::from_column_slice(
                3,
                3,
                self.information.as_slice(),
            ))?,
            ..Default::default()
        })
    }
    fn recipe(self) -> Result<SmootherFactor> {
        let relative = RelativePose2::new(self.measured)?;
        Ok(SmootherFactor::analytic(
            &[self.a as u64, self.b as u64],
            3,
            self.options()?,
            move |v| {
                let (r, ja, jb) = relative.linearize(read_se2(&v[0])?, read_se2(&v[1])?)?;
                Ok((
                    DVector::from_column_slice(r.as_slice()),
                    vec![
                        DMatrix::from_column_slice(3, 3, ja.as_slice()),
                        DMatrix::from_column_slice(3, 3, jb.as_slice()),
                    ],
                ))
            },
        ))
    }
    // Original measurement-frame coordinates, evaluated independently of the
    // library's residual, whitening, autodiff and sparse linearization paths.
    fn energy(self, poses: &[SE2]) -> f64 {
        let (a, b, z) = (poses[self.a], poses[self.b], self.measured);
        let (s, c) = a.rotation.log().sin_cos();
        let d = b.translation - a.translation;
        let x = c * d.x + s * d.y - z.translation.x;
        let y = -s * d.x + c * d.y - z.translation.y;
        let (sz, cz) = z.rotation.log().sin_cos();
        let angle = b.rotation.log() - a.rotation.log() - z.rotation.log();
        let r = Vector3::new(
            cz * x + sz * y,
            -sz * x + cz * y,
            angle.sin().atan2(angle.cos()),
        );
        0.5 * r.dot(&(self.information * r))
    }
}
fn energy(edges: &[Edge], poses: &[SE2]) -> f64 {
    edges.iter().map(|e| e.energy(poses)).sum()
}
fn gradient(edges: &[Edge], poses: &[SE2], h: f64) -> Vec<f64> {
    let mut perturbed = poses.to_vec();
    let mut gradient = Vec::with_capacity(3 * (poses.len() - 1));
    // Central differences are a validation oracle, never the smoother's Jacobian.
    for k in 1..poses.len() {
        for coordinate in 0..3 {
            let mut d = Vector3::zeros();
            d[coordinate] = h;
            perturbed[k] = poses[k].retract(d);
            let plus: f64 = edges
                .iter()
                .filter(|e| e.a == k || e.b == k)
                .map(|e| e.energy(&perturbed))
                .sum();
            perturbed[k] = poses[k].retract(-d);
            let minus: f64 = edges
                .iter()
                .filter(|e| e.a == k || e.b == k)
                .map(|e| e.energy(&perturbed))
                .sum();
            gradient.push((plus - minus) / (2. * d[coordinate]));
            perturbed[k] = poses[k];
        }
    }
    gradient
}
fn gradient_infinity(edges: &[Edge], poses: &[SE2], h: f64) -> f64 {
    gradient(edges, poses, h)
        .iter()
        .map(|x| x.abs())
        .fold(0.0_f64, f64::max)
}
fn gradient_step_difference(edges: &[Edge], poses: &[SE2]) -> f64 {
    gradient(edges, poses, 1e-6)
        .iter()
        .zip(gradient(edges, poses, 5e-7))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}
fn batch(edges: &[Edge], seeds: &[SE2], options: &Options) -> Result<(Vec<SE2>, f64)> {
    let mut graph = FactorGraph::new();
    let mut ids = Vec::new();
    for (k, &seed) in seeds.iter().enumerate() {
        let id = graph.add_variable(VariableKind::SE2, se2_values(seed))?;
        if k == 0 {
            graph.set_fixed(id, true)?;
        }
        ids.push(id);
    }
    for &edge in edges {
        // Deliberately use AD and dense QR, rather than the incremental analytic
        // factors and the sparse normal-equation backend used for refreshes.
        add_relative_pose2_factor(
            &mut graph,
            ids[edge.a],
            ids[edge.b],
            edge.measured,
            edge.options()?,
        )?;
    }
    let solved = solve_local(&graph, graph.initial(), options)?;
    if matches!(
        solved.termination,
        Termination::NoProgress | Termination::MaxIterations
    ) {
        return Err(format!("batch reference did not converge: {:?}", solved.termination).into());
    }
    Ok((
        ids.iter()
            .map(|&id| read_se2(&graph.value(&solved.x, id)?))
            .collect::<std::result::Result<_, ropt::Error>>()?,
        solved.cost,
    ))
}
fn variable(key: usize, pose: SE2) -> WindowVariable {
    WindowVariable {
        key: key as u64,
        kind: VariableKind::SE2,
        value: se2_values(pose),
        timestamp: None,
        fixed: key == 0,
    }
}
fn poses(estimates: &BTreeMap<u64, DVector<f64>>) -> Result<Vec<SE2>> {
    Ok((0..estimates.len())
        .map(|k| read_se2(&estimates[&(k as u64)]))
        .collect::<std::result::Result<_, _>>()?)
}
#[derive(Debug, Default)]
pub struct AccuracySummary {
    pub incremental: usize,
    pub forced: usize,
    pub interval: usize,
    pub threshold: usize,
    pub checkpoints: usize,
    pub max_pose_difference: f64,
    pub max_cost_difference: f64,
    pub max_refreshed_gradient: f64,
    pub max_incremental_gradient: f64,
    pub max_cached_cost_gap: f64,
    pub max_gradient_step_difference: f64,
    pub final_qr_entries: usize,
}
pub trait UpdateObserver {
    fn start(&mut self);
    fn finish(&mut self, frame: usize, report: &IsamReport);
}
impl UpdateObserver for () {
    fn start(&mut self) {}
    fn finish(&mut self, _: usize, _: &IsamReport) {}
}
pub fn validate(frames: usize, output: &mut impl Write) -> Result<AccuracySummary> {
    validate_observed(frames, output, &mut ())
}
/// Observe complete updates while keeping nonlinear and dense-reference checks
/// outside the observed region. The workload and quality gates are unchanged.
pub fn validate_observed(
    frames: usize,
    output: &mut impl Write,
    observer: &mut impl UpdateObserver,
) -> Result<AccuracySummary> {
    if !(25..=300).contains(&frames) || !frames.is_multiple_of(25) {
        return Err("frames must be a multiple of 25 in 25..=300".into());
    }
    let options = Options {
        cost_tolerance: 1e-15,
        // Half the independent 1e-6 stationarity gate, leaving a numerical margin.
        // Near the noise floor, the default 1e-10 can correctly report NoProgress.
        gradient_tolerance: 5e-7,
        max_iterations: 200,
        ..Default::default()
    };
    let mut smoother = IsamSmoother::new(IsamOptions {
        batch_solver: Options {
            linear_solver: LinearSolver::SparseLdl(SparseLdlOptions {
                ordering: SparseOrdering::BlockMinimumDegree { block_size: 3 },
                ..Default::default()
            }),
            ..options.clone()
        },
        ..Default::default()
    })?;
    let origin = SE2 {
        translation: Vector2::new(2., -1.),
        rotation: SO2::exp(0.4),
    };
    smoother.update(
        SmootherUpdate {
            variables: vec![variable(0, origin)],
            ..Default::default()
        },
        false,
    )?;
    let mut truth = vec![origin];
    let mut seeds = vec![origin];
    let mut edges = Vec::new();
    let mut summary = AccuracySummary::default();
    let information = Matrix3::new(400., 30., 10., 30., 500., -15., 10., -15., 2500.);
    writeln!(
        output,
        "frame,reason,nonlinear_cost,cached_cost,gradient_inf,batch_pose_difference,batch_cost_difference,qr_entries"
    )?;
    for k in 1..=frames {
        let phase = k as f64;
        let increment = SE2::exp(Vector3::new(0.6, 0.02, 0.04 + 0.005 * (phase * 0.07).sin()));
        truth.push(truth[k - 1].compose(increment));
        let measured = increment.retract(Vector3::new(
            0.008 * (phase * 0.37).sin(),
            0.006 * (phase * 0.21).cos(),
            0.002 * (phase * 0.19).sin(),
        ));
        seeds.push(seeds[k - 1].compose(measured));
        let seed = read_se2(
            smoother
                .estimate((k - 1) as u64)
                .ok_or("missing previous key")?,
        )?
        .compose(measured)
        .retract(Vector3::new(0.02 * phase.cos(), 0.01 * phase.sin(), 0.002));
        let mut added = vec![Edge {
            a: k - 1,
            b: k,
            measured,
            information,
        }];
        let checkpoint = k.is_multiple_of(25);
        if checkpoint {
            // One genuinely delayed old-to-old constraint, and a second loop
            // connecting an existing non-root pose to the newest pose.
            for (a, b) in [(k / 3, k - 5), (k / 5, k)] {
                added.push(Edge {
                    a,
                    b,
                    measured: truth[a].inverse().compose(truth[b]).retract(Vector3::new(
                        0.003 * phase.sin(),
                        0.002 * phase.cos(),
                        0.0005 * phase.sin(),
                    )),
                    information: information * 4.,
                });
            }
        }
        let recipes = added
            .iter()
            .map(|e| e.recipe())
            .collect::<Result<Vec<_>>>()?;
        edges.extend(added);
        let update = SmootherUpdate {
            variables: vec![variable(k, seed)],
            factors: recipes,
            ..Default::default()
        };
        observer.start();
        let report = smoother.update(update, checkpoint)?;
        observer.finish(k, &report);
        match report.relinearization_reason {
            None => summary.incremental += 1,
            Some(IsamRelinearizationReason::Forced) => summary.forced += 1,
            Some(IsamRelinearizationReason::Interval) => summary.interval += 1,
            Some(IsamRelinearizationReason::Threshold) => summary.threshold += 1,
        }
        let estimates = poses(&report.estimates)?;
        if report.estimates[&0] != se2_values(origin) {
            return Err("fixed root changed".into());
        }
        let cost = energy(&edges, &estimates);
        let gradient = gradient_infinity(&edges, &estimates, 1e-6);
        let gap = (cost - report.linearized_cost).abs();
        if ![cost, gradient, gap].iter().all(|x| x.is_finite()) {
            return Err(format!("non-finite independent validation at {k}").into());
        }
        summary.max_cached_cost_gap = summary.max_cached_cost_gap.max(gap);
        if report.kind == IsamUpdateKind::Relinearized {
            summary.max_refreshed_gradient = summary.max_refreshed_gradient.max(gradient);
            if gradient > 1e-6 || gap > 1e-10 * cost.max(1.) {
                return Err(format!(
                    "refresh accuracy failed at {k}: gradient={gradient}, cost gap={gap}"
                )
                .into());
            }
        } else {
            summary.max_incremental_gradient = summary.max_incremental_gradient.max(gradient);
        }
        let (mut mean_difference, mut cost_difference) = (String::new(), String::new());
        if checkpoint {
            let (reference, batch_cost) = batch(
                &edges,
                &seeds,
                &Options {
                    linear_solver: LinearSolver::DenseQr(Default::default()),
                    ..options.clone()
                },
            )?;
            let reference_gradient = gradient_infinity(&edges, &reference, 1e-6);
            let step_difference = gradient_step_difference(&edges, &estimates)
                .max(gradient_step_difference(&edges, &reference));
            summary.max_gradient_step_difference =
                summary.max_gradient_step_difference.max(step_difference);
            let difference = estimates
                .iter()
                .zip(&reference)
                .map(|(a, b)| a.local(*b).norm())
                .fold(0.0_f64, f64::max);
            let delta_cost = (cost - energy(&edges, &reference)).abs();
            if ![
                reference_gradient,
                difference,
                delta_cost,
                batch_cost,
                step_difference,
            ]
            .iter()
            .all(|x| x.is_finite())
                || reference_gradient > 1e-6
                || step_difference > 2e-7
                || difference > 2e-6
                || delta_cost > 1e-9 * cost.max(1.)
                || (batch_cost - energy(&edges, &reference)).abs() > 1e-10 * batch_cost.max(1.)
            {
                return Err(format!("batch agreement failed at {k}: pose={difference}, cost={delta_cost}, reference gradient={reference_gradient}").into());
            }
            summary.checkpoints += 1;
            summary.max_pose_difference = summary.max_pose_difference.max(difference);
            summary.max_cost_difference = summary.max_cost_difference.max(delta_cost);
            mean_difference = format!("{difference:.17e}");
            cost_difference = format!("{delta_cost:.17e}");
        }
        summary.final_qr_entries = report.stored_entries;
        writeln!(
            output,
            "{k},{:?},{cost:.17e},{:.17e},{gradient:.17e},{mean_difference},{cost_difference},{}",
            report.relinearization_reason, report.linearized_cost, report.stored_entries
        )?;
    }
    if summary.incremental == 0
        || summary.forced != frames / 25
        || summary.checkpoints != frames / 25
    {
        return Err("expected incremental and forced updates were not exercised".into());
    }
    Ok(summary)
}
