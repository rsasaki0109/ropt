#[allow(dead_code)]
#[path = "real_se2_benchmark.rs"]
mod source;
use ropt::{factor::*, fixed_lag::*, isam::*, *};
pub use source::Counter;
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Default)]
struct Work {
    calls: usize,
    queries: usize,
    requested: usize,
    recovery: usize,
    batch: usize,
    refinement: usize,
    factors: usize,
    relinearizations: usize,
    passes: usize,
    eliminated: usize,
    reused: usize,
    boundary: usize,
}
impl Work {
    fn record(&mut self, report: &IsamReport) {
        self.calls += 1;
        self.batch += report.batch_trials;
        self.refinement += report.refinement_steps;
        self.factors += report.factors_linearized;
        if let Some(selected) = &report.selective_update {
            self.relinearizations += selected.factor_relinearizations;
            self.passes += selected.passes;
        }
        if let Some(tree) = report.editable_bayes_tree_update {
            self.eliminated += tree.reeliminated_cliques;
            self.reused += tree.reused_cliques;
            self.boundary += tree.reused_boundary_messages;
        }
    }
}
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    run_with_differentiation(None)
}
/// Explicit AD workload selected by the entry point; analytic entry points stay unchanged.
#[allow(dead_code)] // Used by the explicit AD benchmark entry point.
pub fn run_autodiff(
    chunk_size: usize,
    edge_stride: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    run_with_differentiation(Some((chunk_size, edge_stride)))
}
fn run_with_differentiation(
    autodiff: Option<(usize, usize)>,
) -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        return Err("requires release".into());
    }
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 6 {
        return Err("usage: benchmark_selective_se2_isam input NEW_DIRECTORY repeats warmup full|global|editable selected_cap".into());
    }
    let repeats: usize = args[2].parse()?;
    let warmup: usize = args[3].parse()?;
    let policy = args[4].as_str();
    let cap: usize = args[5].parse()?;
    if repeats == 0
        || warmup.checked_add(repeats).is_none()
        || !matches!(policy, "full" | "global" | "editable")
        || cap > 198
    {
        return Err("invalid arguments".into());
    }
    let target = 8e-8;
    let (root, frames) = source::load_with_differentiation(
        &args[0],
        |relative, a, b| relative.linearize_ambient(a.as_slice(), b.as_slice()),
        VariableKind::SE2Cartesian,
        autodiff,
    )?;
    let factor_count: usize = frames.iter().map(|f| f.factors.len()).sum();
    let ad_count = autodiff.map_or(0, |(_, stride)| factor_count.div_ceil(stride));
    let out = PathBuf::from(&args[1]);
    fs::create_dir(&out)?;
    let options = IsamOptions {
        relinearization_interval: usize::MAX,
        batch_solver: Options {
            algorithm: Algorithm::GaussNewton,
            nonmonotonic_gauss_newton: true,
            linear_solver: LinearSolver::SparseQr(SparseQrOptions {
                ordering: SparseOrdering::BlockMinimumDegree { block_size: 3 },
                ..Default::default()
            }),
            gradient_norm: GradientNorm::ColumnNormalized,
            gradient_tolerance: target,
            step_tolerance: 1e-12,
            cost_tolerance: 1e-15,
            max_iterations: 200,
            ..Default::default()
        },
        ..Default::default()
    };
    fs::write(
        out.join("options.txt"),
        format!(
            "{options:?}\npolicy={policy}; requested_cap={cap}; passes_per_call=1; target={target}; key_gate=target/2; full_recovery_explicit; allocator=false\ncomplete update, probes, owned reports and recovery timed; parsing, factor construction and serialization excluded\n"
        ),
    )?;
    if let Some((chunk, stride)) = autodiff {
        fs::write(
            out.join("differentiation.txt"),
            format!(
                "runtime_ad=true; chunk_size={chunk}; edge_stride={stride}; factors={factor_count}; ad_factors={ad_count}; analytic_factors={}; original_edges_and_information_unchanged=true\n",
                factor_count - ad_count
            ),
        )?;
    }
    let mut samples = BufWriter::new(File::create(out.join("samples.csv"))?);
    let mut updates = BufWriter::new(File::create(out.join("updates.csv"))?);
    let mut poses = BufWriter::new(File::create(out.join("poses.csv"))?);
    writeln!(
        samples,
        "sample,elapsed_ns,updates,allocations,requested_bytes"
    )?;
    writeln!(poses, "frame,key,x,y,angle")?;
    writeln!(
        updates,
        "sample,frame,elapsed_ns,iterations,cost,entries,allocations,requested_bytes,batch_trials,initial_damping,final_damping,refinement_steps,queries,requested_calls,forced_recovery,recovery_reason,factors_linearized,factor_relinearizations,selected_passes,gaussian_reeliminated,gaussian_reused,boundary_messages,gradient_norm,cached_cost"
    )?;
    for round in 0..warmup + repeats {
        let smoother = IsamSmoother::new(options.clone())?
            .with_batch_ordering_reuse()?
            .with_final_refinement(4)?;
        let mut smoother = match policy {
            "editable" => smoother.with_selective_bayes_tree(1)?,
            "global" => smoother.with_selective_relinearization(1)?,
            _ => smoother,
        };
        smoother.update(
            SmootherUpdate {
                variables: vec![root.clone()],
                ..Default::default()
            },
            false,
        )?;
        let mut total = 0_u128;
        for (index, frame) in frames.iter().enumerate() {
            let update = || SmootherUpdate {
                variables: vec![frame.variable.clone()],
                factors: frame.factors.clone(),
                ..Default::default()
            };
            ropt::stream_profile::reset();
            let mut profile_ns = [0_u128; 4];
            let mut profile_calls = [0_usize; 4];
            let mut profile_errors = [0_usize; 4];
            macro_rules! profile_call {
                ($slot:expr, $call:expr) => {{
                    ropt::stream_profile::enter($slot);
                    let t = Instant::now();
                    let result = $call;
                    profile_ns[$slot] += t.elapsed().as_nanos();
                    profile_calls[$slot] += 1;
                    profile_errors[$slot] += usize::from(result.is_err());
                    result
                }};
            }
            let start = Instant::now();
            let mut work = Work::default();
            let mut recovery_reason = "";
            let mut report = match profile_call!(0, smoother.update(update(), policy == "full")) {
                Ok(report) => report,
                Err(error) if policy != "full" => {
                    eprintln!(
                        "frame {} initial partial update rejected: {error:?}",
                        index + 1
                    );
                    work.calls += 1;
                    work.recovery = 1;
                    recovery_reason = "initial_error";
                    profile_call!(2, smoother.update(update(), true))?
                }
                Err(error) => return Err(error.into()),
            };
            work.record(&report);
            let mut probe = profile_call!(
                3,
                smoother.nonlinear_summary(GradientNorm::ColumnNormalized)
            )?;
            work.queries += 1;
            if policy != "full" {
                for _ in 0..cap {
                    if probe.gradient_norm <= target {
                        break;
                    }
                    let keys = probe
                        .variable_gradients
                        .iter()
                        .filter(|(_, g)| **g > target * 0.5)
                        .map(|(&key, _)| key)
                        .collect::<Vec<_>>();
                    match profile_call!(1, smoother.relinearize_selected(&keys)) {
                        Ok(result) => {
                            report = result;
                            work.record(&report);
                            work.requested += 1;
                        }
                        Err(error) => {
                            eprintln!("frame {} requested update rejected: {error:?}", index + 1);
                            work.calls += 1;
                            recovery_reason = "requested_error";
                            break;
                        }
                    }
                    probe = profile_call!(
                        3,
                        smoother.nonlinear_summary(GradientNorm::ColumnNormalized)
                    )?;
                    work.queries += 1;
                }
                if probe.gradient_norm > target {
                    report = profile_call!(2, smoother.update(Default::default(), true))?;
                    work.record(&report);
                    work.recovery += 1;
                    if recovery_reason.is_empty() {
                        recovery_reason = "selected_cap";
                    }
                    probe = profile_call!(
                        3,
                        smoother.nonlinear_summary(GradientNorm::ColumnNormalized)
                    )?;
                    work.queries += 1;
                }
            }
            if probe.gradient_norm > target {
                return Err(format!(
                    "frame {} fails internal gradient target: {:.17e}",
                    index + 1,
                    probe.gradient_norm
                )
                .into());
            }
            let elapsed = start.elapsed().as_nanos();
            total += elapsed;
            eprintln!(
                "stream_profile round={} frame={} total={} initial={} requested={} recovery={} probe={} initial_calls={} requested_calls={} recovery_calls={} probe_calls={} initial_errors={} requested_errors={} recovery_errors={} probe_errors={}",
                round,
                index + 1,
                elapsed,
                profile_ns[0],
                profile_ns[1],
                profile_ns[2],
                profile_ns[3],
                profile_calls[0],
                profile_calls[1],
                profile_calls[2],
                profile_calls[3],
                profile_errors[0],
                profile_errors[1],
                profile_errors[2],
                profile_errors[3]
            );
            let names = [
                "graph_build",
                "sparse_evaluation",
                "cache_solve",
                "batch_solver",
                "sparse_qr_workspace",
                "qr_rebuild",
                "factor_capture",
                "tree_update",
                "tree_build",
                "coordinates",
                "selected_relinearize",
            ];
            for (api, records) in ropt::stream_profile::snapshot().iter().enumerate() {
                for (name, r) in names.iter().zip(records) {
                    if r.calls != 0 {
                        eprintln!(
                            "stream_detail round={} frame={} api={} phase={} ns={} calls={} errors={}",
                            round,
                            index + 1,
                            api,
                            name,
                            r.ns,
                            r.calls,
                            r.errors
                        );
                    }
                }
            }
            if round >= warmup {
                writeln!(
                    updates,
                    "{},{},{elapsed},{},{:.17e},{},0,0,{},,,{},{},{},{},{},{},{},{},{},{},{},{:.17e},{:.17e}",
                    round - warmup,
                    index + 1,
                    work.calls,
                    probe.cost,
                    report.stored_entries,
                    work.batch,
                    work.refinement,
                    work.queries,
                    work.requested,
                    work.recovery,
                    recovery_reason,
                    work.factors,
                    work.relinearizations,
                    work.passes,
                    work.eliminated,
                    work.reused,
                    work.boundary,
                    probe.gradient_norm,
                    report.linearized_cost
                )?;
            }
            if round == warmup {
                for (key, value) in report.estimates {
                    writeln!(
                        poses,
                        "{},{key},{:.17e},{:.17e},{:.17e}",
                        index + 1,
                        value[0],
                        value[1],
                        value[2]
                    )?;
                }
            }
        }
        if round >= warmup {
            writeln!(samples, "{},{total},{},0,0", round - warmup, frames.len())?;
        }
    }
    samples.flush()?;
    updates.flush()?;
    poses.flush()?;
    Ok(())
}
