//! Accuracy-matched all-history nonlinear update calls on a shared input stream.
#[path = "allocation_counter.rs"]
mod allocation;
pub use allocation::Counter;
use ropt::{
    factor::*, fixed_lag::*, isam::*, noise::NoiseModel, pose_graph::*, sparse_direct::*, *,
};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

pub(super) struct Frame {
    pub variable: WindowVariable,
    pub factors: Vec<SmootherFactor>,
}
pub(super) fn load(
    path: &str,
    linearize: impl Fn(RelativePose2, &DVector<f64>, &DVector<f64>) -> Result<Pose2Linearization, Error>
    + Copy
    + 'static,
    kind: VariableKind,
) -> Result<(WindowVariable, Vec<Frame>), Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut tokens = text.split_whitespace();
    let mut next = || tokens.next().ok_or("truncated stream");
    if next()? != "ROOT" {
        return Err("missing ROOT".into());
    }
    let root = WindowVariable {
        key: 0,
        kind,
        value: DVector::from_vec(
            (0..3)
                .map(|_| next()?.parse::<f64>().map_err(Into::into))
                .collect::<Result<_, Box<dyn std::error::Error>>>()?,
        ),
        timestamp: None,
        fixed: true,
    };
    read_se2(&root.value)?;
    let mut frames = Vec::new();
    while let Ok(tag) = next() {
        if tag != "FRAME" {
            return Err("missing FRAME".into());
        }
        let key: u64 = next()?.parse()?;
        if key != frames.len() as u64 + 1 {
            return Err("nonsequential key".into());
        }
        let value = DVector::from_vec(
            (0..3)
                .map(|_| next()?.parse::<f64>().map_err(Into::into))
                .collect::<Result<_, Box<dyn std::error::Error>>>()?,
        );
        read_se2(&value)?;
        let count: usize = next()?.parse()?;
        if count == 0 || count > 10_000 {
            return Err("invalid edge count".into());
        }
        let mut factors = Vec::new();
        for _ in 0..count {
            if next()? != "EDGE" {
                return Err("missing EDGE".into());
            }
            let a: u64 = next()?.parse()?;
            let b: u64 = next()?.parse()?;
            if a >= b || b > key {
                return Err("invalid endpoints".into());
            }
            let z = DVector::from_vec(
                (0..3)
                    .map(|_| next()?.parse::<f64>().map_err(Into::into))
                    .collect::<Result<_, Box<dyn std::error::Error>>>()?,
            );
            let mut w = [0.; 6];
            for v in &mut w {
                *v = next()?.parse()?;
            }
            let noise = NoiseModel::from_information(DMatrix::from_row_slice(
                3,
                3,
                &[w[0], w[1], w[2], w[1], w[3], w[4], w[2], w[4], w[5]],
            ))?;
            let relative = RelativePose2::new(read_se2(&z)?)?;
            factors.push(SmootherFactor::analytic(
                &[a, b],
                3,
                FactorOptions {
                    noise,
                    ..Default::default()
                },
                move |v| {
                    let (r, ja, jb) = linearize(relative, &v[0], &v[1])?;
                    Ok((
                        DVector::from_column_slice(r.as_slice()),
                        vec![
                            DMatrix::from_column_slice(3, 3, ja.as_slice()),
                            DMatrix::from_column_slice(3, 3, jb.as_slice()),
                        ],
                    ))
                },
            ));
        }
        frames.push(Frame {
            variable: WindowVariable {
                key,
                kind,
                value,
                timestamp: None,
                fixed: false,
            },
            factors,
        });
    }
    if frames.is_empty() || frames.len() > 10_000 {
        return Err("invalid frame count".into());
    }
    Ok((root, frames))
}
#[allow(dead_code)] // Existing benchmark entry points select this typed-group path.
pub fn run(
    policy: &str,
    constructor: impl Fn(IsamOptions) -> Result<IsamSmoother, Error>,
    telemetry: impl Fn(&IsamReport) -> Option<(usize, f64, f64)>,
    batch_solver: Option<Options>,
) -> Result<(), Box<dyn std::error::Error>> {
    run_with_linearization(
        policy,
        constructor,
        telemetry,
        batch_solver,
        |relative, a, b| relative.linearize(read_se2(a)?, read_se2(b)?),
    )
}
#[allow(dead_code)] // Shared module is compiled separately by each benchmark entry.
pub fn run_with_linearization(
    policy: &str,
    constructor: impl Fn(IsamOptions) -> Result<IsamSmoother, Error>,
    telemetry: impl Fn(&IsamReport) -> Option<(usize, f64, f64)>,
    batch_solver: Option<Options>,
    linearize: impl Fn(RelativePose2, &DVector<f64>, &DVector<f64>) -> Result<Pose2Linearization, Error>
    + Copy
    + 'static,
) -> Result<(), Box<dyn std::error::Error>> {
    run_with_chart(
        policy,
        constructor,
        telemetry,
        batch_solver,
        linearize,
        VariableKind::SE2,
        false,
    )
}
#[allow(dead_code)] // Explicit benchmark chart selector.
pub fn run_with_chart(
    policy: &str,
    constructor: impl Fn(IsamOptions) -> Result<IsamSmoother, Error>,
    telemetry: impl Fn(&IsamReport) -> Option<(usize, f64, f64)>,
    batch_solver: Option<Options>,
    linearize: impl Fn(RelativePose2, &DVector<f64>, &DVector<f64>) -> Result<Pose2Linearization, Error>
    + Copy
    + 'static,
    kind: VariableKind,
    report_gn_trials: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if !matches!(kind, VariableKind::SE2 | VariableKind::SE2Cartesian) {
        return Err("requires SE2 chart".into());
    }
    if cfg!(debug_assertions) {
        return Err("requires release".into());
    }
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(4..=5).contains(&args.len()) {
        return Err("usage: benchmark_isam_se2 input NEW_DIRECTORY repeats warmup [alloc]".into());
    }
    let repeats: usize = args[2].parse()?;
    let warmup: usize = args[3].parse()?;
    let alloc = args.len() == 5;
    if repeats == 0 || warmup.checked_add(repeats).is_none() || (alloc && args[4] != "alloc") {
        return Err("invalid arguments".into());
    }
    let (root, frames) = load(&args[0], linearize, kind)?;
    let out = PathBuf::from(&args[1]);
    fs::create_dir(&out)?;
    let options = IsamOptions {
        batch_solver: batch_solver.unwrap_or(Options {
            linear_solver: LinearSolver::SparseLdl(SparseLdlOptions {
                ordering: SparseOrdering::BlockMinimumDegree { block_size: 3 },
                ..Default::default()
            }),
            initial_damping: 1e-5,
            gradient_tolerance: 8e-8,
            gradient_norm: GradientNorm::ColumnNormalized,
            cost_tolerance: 1e-15,
            step_tolerance: 1e-12,
            max_iterations: 200,
            ..Default::default()
        }),
        ..Default::default()
    };
    fs::write(
        out.join("options.txt"),
        format!(
            "{options:?}\npolicy={policy}\nforce every update; full report timed; allocator={alloc}\n"
        ),
    )?;
    let mut samples = BufWriter::new(File::create(out.join("samples.csv"))?);
    let mut updates = BufWriter::new(File::create(out.join("updates.csv"))?);
    let mut poses = BufWriter::new(File::create(out.join("poses.csv"))?);
    writeln!(
        samples,
        "sample,elapsed_ns,updates,allocations,requested_bytes"
    )?;
    writeln!(
        updates,
        "sample,frame,elapsed_ns,iterations,cost,entries,allocations,requested_bytes,batch_trials,initial_damping,final_damping,refinement_steps"
    )?;
    writeln!(poses, "frame,key,x,y,angle")?;
    for round in 0..warmup + repeats {
        let mut isam = constructor(options.clone())?;
        isam.update(
            SmootherUpdate {
                variables: vec![root.clone()],
                ..Default::default()
            },
            false,
        )?;
        let (mut total, mut calls, mut bytes) = (0_u128, 0_u64, 0_u64);
        for (index, frame) in frames.iter().enumerate() {
            let update = SmootherUpdate {
                variables: vec![frame.variable.clone()],
                factors: frame.factors.clone(),
                ..Default::default()
            };
            if alloc {
                allocation::start();
            }
            let start = Instant::now();
            let result = isam.update(update, true);
            let ns = start.elapsed().as_nanos();
            let (a, b) = if alloc { allocation::stop() } else { (0, 0) };
            let result = result?;
            total += ns;
            calls += a;
            bytes += b;
            let (trials, initial, final_value) = telemetry(&result).map_or_else(
                || {
                    (
                        if report_gn_trials {
                            result.batch_trials.to_string()
                        } else {
                            String::new()
                        },
                        String::new(),
                        String::new(),
                    )
                },
                |(n, a, b)| (n.to_string(), format!("{a:.17e}"), format!("{b:.17e}")),
            );
            if round >= warmup {
                writeln!(
                    updates,
                    "{},{},{ns},1,{:.17e},{},{a},{b},{trials},{initial},{final_value},{}",
                    round - warmup,
                    index + 1,
                    result.linearized_cost,
                    result.stored_entries,
                    result.refinement_steps
                )?;
            }
            if round == warmup {
                for (key, v) in result.estimates {
                    writeln!(
                        poses,
                        "{},{key},{:.17e},{:.17e},{:.17e}",
                        index + 1,
                        v[0],
                        v[1],
                        v[2]
                    )?;
                }
            }
        }
        if round >= warmup {
            writeln!(
                samples,
                "{},{total},{},{calls},{bytes}",
                round - warmup,
                frames.len()
            )?;
        }
    }
    samples.flush()?;
    updates.flush()?;
    poses.flush()?;
    Ok(())
}
