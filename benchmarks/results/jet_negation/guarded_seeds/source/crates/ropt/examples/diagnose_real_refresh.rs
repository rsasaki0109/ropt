//! Untimed diagnosis of one original-information recorded SE2 batch refresh.
//! Use prior committed estimates as vertices; measurements/information stay intact.
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
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err(
            "usage: diagnose_real_refresh INPUT.g2o NEW_DIRECTORY ldl|qr|sparse-gn INITIAL_DAMPING"
                .into(),
        );
    }
    let data = PoseGraph2::from_reader(BufReader::new(File::open(&args[0])?))?;
    let loaded = data.build_with_jacobians(None, PoseJacobian::Analytic)?;
    let backend = match args[2].as_str() {
        "ldl" => LinearSolver::SparseLdl(SparseLdlOptions {
            ordering: SparseOrdering::BlockMinimumDegree { block_size: 3 },
            ..Default::default()
        }),
        "qr" => LinearSolver::DenseQr(Default::default()),
        "sparse-gn" => LinearSolver::SparseQr(SparseQrOptions {
            ordering: SparseOrdering::BlockMinimumDegree { block_size: 3 },
            ..Default::default()
        }),
        _ => return Err("unknown backend".into()),
    };
    let options = Options {
        linear_solver: backend,
        algorithm: if args[2] == "sparse-gn" {
            Algorithm::GaussNewton
        } else {
            Algorithm::LevenbergMarquardt
        },
        nonmonotonic_gauss_newton: args[2] == "sparse-gn",
        initial_damping: args[3].parse()?,
        gradient_tolerance: 8e-8,
        gradient_norm: GradientNorm::ColumnNormalized,
        cost_tolerance: 1e-15,
        step_tolerance: 1e-12,
        max_iterations: 200,
        ..Default::default()
    };
    let out = PathBuf::from(&args[1]);
    std::fs::create_dir(&out)?;
    let report = solve_local(&loaded.graph, loaded.graph.initial(), &options)?;
    let mut trace = BufWriter::new(File::create(out.join("iterations.csv"))?);
    writeln!(
        trace,
        "iteration,cost,trial_cost,actual_reduction,gradient_norm,step_norm,damping,gain_ratio,accepted"
    )?;
    for i in &report.iterations {
        writeln!(
            trace,
            "{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{}",
            i.iteration,
            i.cost,
            i.trial_cost,
            i.actual_reduction,
            i.gradient_norm,
            i.step_norm,
            i.damping,
            i.gain_ratio,
            i.accepted
        )?;
    }
    trace.flush()?;
    let mut output = BufWriter::new(File::create(out.join("poses.g2o"))?);
    data.write(&mut output, &loaded.poses(&report.x)?)?;
    output.flush()?;
    let summary = format!(
        "options={options:?}\ntermination={:?}\ntrials={}\ninitial_cost={:.17e}\nfinal_cost={:.17e}\n",
        report.termination,
        report.iterations.len(),
        report.initial_cost,
        report.cost
    );
    std::fs::write(out.join("summary.txt"), &summary)?;
    print!("{summary}");
    Ok(())
}
