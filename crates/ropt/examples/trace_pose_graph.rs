//! Iteration diagnostics for the explicit analytic/block-LDL configuration.
//! This runs one solve, including unsuccessful termination; it is not a benchmark.
use ropt::{
    g2o::{PoseGraph2, PoseJacobian},
    *,
};
use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::PathBuf,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let input = PathBuf::from(
        args.next()
            .ok_or("usage: trace_pose_graph input.g2o NEW_DIRECTORY cost-tolerance")?,
    );
    let output = PathBuf::from(args.next().ok_or("missing output")?);
    let tolerance: f64 = args.next().ok_or("missing cost tolerance")?.parse()?;
    if args.next().is_some() || !tolerance.is_finite() || tolerance < 0. {
        return Err("invalid arguments".into());
    }
    let data = PoseGraph2::from_reader(BufReader::new(File::open(input)?))?;
    let loaded = data.build_with_jacobians(None, PoseJacobian::Analytic)?;
    std::fs::create_dir(&output)?;
    let options = Options {
        linear_solver: LinearSolver::SparseLdl(SparseLdlOptions {
            ordering: SparseOrdering::BlockMinimumDegree { block_size: 3 },
            ..Default::default()
        }),
        cost_tolerance: tolerance,
        max_iterations: 300,
        ..Default::default()
    };
    let report = solve_local(&loaded.graph, loaded.graph.initial(), &options)?;
    let mut trace = BufWriter::new(File::create(output.join("iterations.csv"))?);
    writeln!(
        trace,
        "iteration,cost,trial_cost,cost_difference,actual_reduction,gradient_norm,step_norm,damping,gain_ratio,accepted"
    )?;
    for i in &report.iterations {
        writeln!(
            trace,
            "{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{}",
            i.iteration,
            i.cost,
            i.trial_cost,
            i.cost - i.trial_cost,
            i.actual_reduction,
            i.gradient_norm,
            i.step_norm,
            i.damping,
            i.gain_ratio,
            i.accepted
        )?;
    }
    trace.flush()?;
    let mut poses = BufWriter::new(File::create(output.join("poses.g2o"))?);
    data.write(&mut poses, &loaded.poses(&report.x)?)?;
    poses.flush()?;
    let (r, j) = loaded.graph.evaluate_sparse(&report.x)?;
    let gradient = j.transpose_multiply(&r)?.amax();
    let mut stats = File::create(output.join("summary.txt"))?;
    writeln!(
        stats,
        "options={options:?}\ntermination={:?}\nnonlinear_trials={}\ncost={:.17e}\nfinal_tangent_gradient={:.17e}",
        report.termination,
        report.iterations.len(),
        report.cost,
        gradient
    )?;
    println!(
        "{:?}: {} trials, gradient {gradient:.3e}; {}",
        report.termination,
        report.iterations.len(),
        output.display()
    );
    Ok(())
}
