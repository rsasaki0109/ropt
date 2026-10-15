//! Release-only solve timings. Parsing, graph construction and validation are untimed.
use ropt::{
    g2o::{PoseGraph2, PoseJacobian},
    sparse::CgOptions,
    *,
};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        return Err("timing requires cargo run --release".into());
    }
    let mut input = None;
    let mut output = None;
    let mut repeats = 15_usize;
    let mut warmup = 3_usize;
    let mut backends = "dense,qr,scalar_cg,block_cg".to_string();
    let mut pose_jacobians = PoseJacobian::AutoDiff;
    let mut cg_relative_tolerance = 1e-8_f64;
    let mut cg_absolute_tolerance = 1e-10_f64;
    let mut cost_tolerance = Options::default().cost_tolerance;
    let mut max_iterations = Options::default().max_iterations;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input = Some(PathBuf::from(args.next().ok_or("missing input")?)),
            "--output" => output = Some(PathBuf::from(args.next().ok_or("missing output")?)),
            "--repeats" => repeats = args.next().ok_or("missing repeats")?.parse()?,
            "--warmup" => warmup = args.next().ok_or("missing warmup")?.parse()?,
            "--cost-tolerance" => {
                cost_tolerance = args.next().ok_or("missing cost tolerance")?.parse()?;
            }
            "--max-iterations" => {
                max_iterations = args.next().ok_or("missing iteration limit")?.parse()?;
            }
            "--backends" => backends = args.next().ok_or("missing backend list")?,
            "--pose-jacobians" => {
                pose_jacobians = match args.next().as_deref() {
                    Some("autodiff") => PoseJacobian::AutoDiff,
                    Some("analytic") => PoseJacobian::Analytic,
                    _ => return Err("--pose-jacobians requires autodiff or analytic".into()),
                }
            }
            "--cg-relative-tolerance" => {
                cg_relative_tolerance = args
                    .next()
                    .ok_or("missing CG relative tolerance")?
                    .parse()?
            }
            "--cg-absolute-tolerance" => {
                cg_absolute_tolerance = args
                    .next()
                    .ok_or("missing CG absolute tolerance")?
                    .parse()?
            }
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    if repeats == 0 || max_iterations == 0 {
        return Err("repeats and iteration limit must be positive".into());
    }
    if !cg_relative_tolerance.is_finite()
        || !cg_absolute_tolerance.is_finite()
        || cg_relative_tolerance < 0.0
        || cg_absolute_tolerance < 0.0
    {
        return Err("CG tolerances must be finite and nonnegative".into());
    }
    if !cost_tolerance.is_finite() || cost_tolerance < 0.0 {
        return Err("cost tolerance must be finite and nonnegative".into());
    }
    let input = input.ok_or("--input file.g2o is required")?;
    let output = output.ok_or("--output NEW_DIRECTORY is required")?;
    let data = PoseGraph2::from_reader(BufReader::new(File::open(input)?))?;
    let loaded = data.build_with_jacobians(None, pose_jacobians)?;
    let initial = loaded.graph.initial();
    let cg = CgOptions {
        max_iterations: 10000,
        relative_tolerance: cg_relative_tolerance,
        absolute_tolerance: cg_absolute_tolerance,
    };
    let available = [
        ("dense", LinearSolver::Dense),
        ("qr", LinearSolver::DenseQr(QrOptions::default())),
        ("scalar_cg", LinearSolver::SparseCg(cg)),
        ("block_cg", LinearSolver::SparseBlockCg(cg)),
        (
            "sparse_ldl",
            LinearSolver::SparseLdl(SparseLdlOptions::default()),
        ),
        (
            "sparse_ldl_natural",
            LinearSolver::SparseLdl(SparseLdlOptions {
                ordering: SparseOrdering::Natural,
                ..SparseLdlOptions::default()
            }),
        ),
        (
            "sparse_ldl_block",
            LinearSolver::SparseLdl(SparseLdlOptions {
                ordering: SparseOrdering::BlockMinimumDegree { block_size: 3 },
                ..Default::default()
            }),
        ),
    ];
    let mut solvers = Vec::new();
    for name in backends.split(',') {
        let solver = available
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .copied()
            .ok_or_else(|| format!("unknown backend {name}"))?;
        if solvers.iter().any(|(candidate, _)| *candidate == name) {
            return Err(format!("duplicate backend {name}").into());
        }
        solvers.push(solver);
    }
    let options = |linear_solver| Options {
        linear_solver,
        cost_tolerance,
        max_iterations,
        ..Options::default()
    };
    // An extra untimed solve supplies a consistency reference, not a timing sample.
    // Sparse-only selections do not allocate a dense Jacobian or normal matrix.
    let reference_solver = solvers
        .iter()
        .find(|(name, _)| *name == "dense")
        .unwrap_or(&solvers[0])
        .1;
    let reference = solve_local(&loaded.graph, initial.clone(), &options(reference_solver))?;
    let reference_poses = loaded.poses(&reference.x)?;
    let validate = |report: &Report| -> Result<(), Box<dyn std::error::Error>> {
        if matches!(
            report.termination,
            Termination::NoProgress | Termination::MaxIterations
        ) || !report.cost.is_finite()
            || report.cost > report.initial_cost
            || (report.cost - reference.cost).abs() > 1e-9 * reference.cost.abs().max(1.0)
        {
            return Err("nonconverged or inconsistent final cost".into());
        }
        let poses = loaded.poses(&report.x)?;
        for id in &loaded.anchors {
            if loaded.graph.value(&initial, loaded.variables[id])?
                != loaded.graph.value(&report.x, loaded.variables[id])?
            {
                return Err("fixed anchor changed".into());
            }
        }
        for (id, pose) in poses {
            let r = reference_poses[&id];
            let error = (pose.translation - r.translation).norm_squared()
                + r.rotation.local(pose.rotation).powi(2);
            if !error.is_finite() || error.sqrt() > 1e-6 {
                return Err("final poses differ from the consistency reference".into());
            }
        }
        Ok(())
    };
    validate(&reference)?;
    fs::create_dir(&output)?;
    let new_file = |name: &str| {
        File::options()
            .write(true)
            .create_new(true)
            .open(output.join(name))
    };
    let mut config = new_file("options.txt")?;
    writeln!(
        config,
        "dimensions={:?}; anchors={:?}; warmup={warmup}; repeats={repeats}",
        loaded.graph.dimensions(),
        loaded.anchors
    )?;
    writeln!(config, "consistency_reference={reference_solver:?}")?;
    writeln!(config, "pose_jacobians={pose_jacobians:?}")?;
    for (name, solver) in &solvers {
        writeln!(config, "{name}: {:?}", options(*solver))?;
    }
    let mut samples = BufWriter::new(new_file("samples.csv")?);
    writeln!(
        samples,
        "backend,sample,elapsed_ns,initial_cost,final_cost,termination,nonlinear_trials,accepted_trials,linear_iterations"
    )?;
    for round in 0..warmup + repeats {
        // Rotate the first backend each round to reduce fixed-order drift.
        for offset in 0..solvers.len() {
            let (name, solver) = solvers[(round + offset) % solvers.len()];
            let x = initial.clone();
            let opts = options(solver);
            let start = Instant::now();
            let report = solve_local(std::hint::black_box(&loaded.graph), x, &opts)?;
            let elapsed = start.elapsed().as_nanos();
            validate(&report)?;
            if round < warmup {
                continue;
            }
            let sample = round - warmup;
            writeln!(
                samples,
                "{name},{sample},{elapsed},{:.17e},{:.17e},{:?},{},{},{}",
                report.initial_cost,
                report.cost,
                report.termination,
                report.iterations.len(),
                report.iterations.iter().filter(|i| i.accepted).count(),
                report
                    .iterations
                    .iter()
                    .map(|i| i.linear_iterations)
                    .sum::<usize>()
            )?;
            if sample == 0 {
                let mut file = BufWriter::new(new_file(&format!("{name}.g2o"))?);
                data.write(&mut file, &loaded.poses(&report.x)?)?;
                file.flush()?;
            }
        }
    }
    samples.flush()?;
    println!(
        "Saved {} timed solves and {} checked solutions to {}",
        repeats * solvers.len(),
        solvers.len(),
        output.display()
    );
    Ok(())
}
