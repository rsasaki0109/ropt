//! Sparse factor storage and complete LM solves with both elimination orders.
use ropt::{g2o::PoseGraph2, sparse_direct::SparseLdlFactor, *};
use std::fs::File;
use std::io::BufReader;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = std::env::args()
        .nth(1)
        .ok_or("usage: sparse_ldl input.g2o")?;
    let graph = PoseGraph2::from_reader(BufReader::new(File::open(input)?))?
        .build(None)?
        .graph;
    let (_, j) = graph.evaluate_sparse(&graph.initial())?;
    println!(
        "ordering,columns,jacobian_entries,normal_lower_entries,factor_entries,cost,nonlinear_trials,termination"
    );
    let mut reference: Option<Report> = None;
    for ordering in [SparseOrdering::Natural, SparseOrdering::MinimumDegree] {
        let cfg = SparseLdlOptions {
            ordering,
            ..SparseLdlOptions::default()
        };
        let factor = SparseLdlFactor::factor_normal(&j, 1e-3, &cfg)?;
        let report = solve_local(
            &graph,
            graph.initial(),
            &Options {
                linear_solver: LinearSolver::SparseLdl(cfg),
                ..Options::default()
            },
        )?;
        if matches!(
            report.termination,
            Termination::NoProgress | Termination::MaxIterations
        ) {
            return Err("sparse LDL did not converge".into());
        }
        if let Some(r) = &reference {
            if (report.cost - r.cost).abs() > 1e-9 || (&report.x - &r.x).norm() > 1e-6 {
                return Err("elimination orders disagree".into());
            }
        }
        println!(
            "{ordering:?},{},{},{},{},{:.17e},{},{:?}",
            factor.dimension(),
            j.nnz(),
            factor.normal_entries(),
            factor.factor_entries(),
            report.cost,
            report.iterations.len(),
            report.termination
        );
        reference = Some(report);
    }
    Ok(())
}
