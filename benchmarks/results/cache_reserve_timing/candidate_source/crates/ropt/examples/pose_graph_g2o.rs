//! Optimize 2D g2o files with AD relative SE2 factors and matrix-free sparse LM.
use ropt::{
    g2o::{PoseGraph2, PoseJacobian},
    sparse::CgOptions,
    *,
};
use std::fs::File;
use std::io::{BufReader, BufWriter};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = None;
    let mut output = None;
    let mut dense = false;
    let mut ldl = false;
    let mut analytic = false;
    let mut block_ordering = false;
    let mut max_iterations = 100;
    let mut cost_tolerance = Options::default().cost_tolerance;
    let mut loss = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input = Some(args.next().ok_or("missing input path")?),
            "--output" => output = Some(args.next().ok_or("missing output path")?),
            "--dense" => dense = true,
            "--ldl" => ldl = true,
            "--analytic" => analytic = true,
            "--block-ordering" => block_ordering = true,
            "--max-iterations" => {
                max_iterations = args.next().ok_or("missing iteration count")?.parse()?
            }
            "--cost-tolerance" => {
                cost_tolerance = args.next().ok_or("missing cost tolerance")?.parse()?
            }
            "--huber" => {
                loss = Some(Loss::Huber(
                    args.next().ok_or("missing Huber scale")?.parse()?,
                ))
            }
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    let input=input.ok_or("usage: pose_graph_g2o --input file.g2o [--output new.g2o] [--dense | --ldl] [--analytic] [--block-ordering] [--max-iterations N] [--cost-tolerance C] [--huber scale]")?;
    let data = PoseGraph2::from_reader(BufReader::new(File::open(input)?))?;
    let loaded = data.build_with_jacobians(
        loss,
        if analytic {
            PoseJacobian::Analytic
        } else {
            PoseJacobian::AutoDiff
        },
    )?;
    let initial = loaded.graph.initial();
    if dense && ldl {
        return Err("choose either --dense or --ldl".into());
    }
    if block_ordering && !ldl {
        return Err("--block-ordering requires --ldl".into());
    }
    let linear_solver = if ldl {
        LinearSolver::SparseLdl(SparseLdlOptions {
            ordering: if block_ordering {
                SparseOrdering::BlockMinimumDegree { block_size: 3 }
            } else {
                SparseOrdering::MinimumDegree
            },
            ..Default::default()
        })
    } else if dense {
        LinearSolver::Dense
    } else {
        LinearSolver::SparseCg(CgOptions {
            max_iterations: 10000,
            relative_tolerance: 1e-8,
            absolute_tolerance: 1e-10,
        })
    };
    let report = solve_local(
        &loaded.graph,
        initial.clone(),
        &Options {
            max_iterations,
            cost_tolerance,
            linear_solver,
            ..Options::default()
        },
    )?;
    let (_, tangent, residuals) = loaded.graph.dimensions();
    println!(
        "vertices={}, edges={}, residuals={}, active tangent={}, anchors={:?}",
        data.vertices().len(),
        data.edges().len(),
        residuals,
        tangent,
        loaded.anchors
    );
    println!(
        "cost: {:.12e} -> {:.12e}; termination={:?}; nonlinear trials={}; linear iterations={}",
        report.initial_cost,
        report.cost,
        report.termination,
        report.iterations.len(),
        report
            .iterations
            .iter()
            .map(|i| i.linear_iterations)
            .sum::<usize>()
    );
    for key in &loaded.anchors {
        let id = loaded.variables[key];
        if loaded.graph.value(&initial, id)? != loaded.graph.value(&report.x, id)? {
            return Err("anchor changed".into());
        }
    }
    if !report.cost.is_finite() || report.cost > report.initial_cost {
        return Err("optimization increased cost".into());
    }
    if let Some(path) = output {
        // Output is opt-in and must be a new file, so input data cannot be overwritten.
        let file = File::options().write(true).create_new(true).open(path)?;
        let mut writer = BufWriter::new(file);
        data.write(&mut writer, &loaded.poses(&report.x)?)?;
        std::io::Write::flush(&mut writer)?;
    }
    if matches!(
        report.termination,
        Termination::MaxIterations | Termination::NoProgress
    ) {
        return Err(format!(
            "solver stopped without convergence: {:?}",
            report.termination
        )
        .into());
    }
    Ok(())
}
