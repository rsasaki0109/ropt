//! Anchored synthetic SE2 loop using AD on right tangents and g2o measurement coordinates.
use ropt::{g2o::PoseGraph2, sparse::CgOptions, *};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = PoseGraph2::from_reader(include_bytes!("../tests/data/loop_se2.g2o").as_slice())?;
    let loaded = data.build(None)?;
    for linear_solver in [
        LinearSolver::Dense,
        LinearSolver::SparseCg(CgOptions::default()),
    ] {
        let report = solve_local(
            &loaded.graph,
            loaded.graph.initial(),
            &Options {
                linear_solver,
                ..Options::default()
            },
        )?;
        println!(
            "anchored SE2 loop: cost={:.3e} -> {:.3e}, termination={:?}",
            report.initial_cost, report.cost, report.termination
        );
        if report.cost > 1e-12 {
            return Err("pose loop did not converge".into());
        }
    }
    Ok(())
}
