//! PCG iteration comparison on a deterministic GNSS/odometry fixture; not a timing benchmark.
#[path = "support/gnss.rs"]
mod gnss;
use ropt::{sparse::CgOptions, *};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    for (loss_name, loss) in [
        ("squared", Loss::Squared),
        ("joint Huber", Loss::Huber(2.5)),
    ] {
        let scene = gnss::scene(Some(loss))?;
        println!(
            "{loss_name}: 36 poses, {} GNSS fixes ({} outliers), lever arm={:?}",
            scene.fixes.len(),
            scene.fixes.iter().filter(|f| f.2).count(),
            scene.lever_arm.as_slice()
        );
        let blocks = scene.graph.tangent_blocks()?;
        println!(
            "{} active 6D blocks, {} stored preconditioner entries versus {} entries in a full normal matrix",
            blocks.len(),
            blocks.iter().map(|b| b.len() * b.len()).sum::<usize>(),
            scene.graph.dimensions().1.pow(2)
        );
        let mut reference: Option<Report> = None;
        for (name, linear_solver) in [
            (
                "scalar Jacobi",
                LinearSolver::SparseCg(CgOptions::default()),
            ),
            (
                "block Jacobi",
                LinearSolver::SparseBlockCg(CgOptions::default()),
            ),
        ] {
            let report = solve_local(
                &scene.graph,
                scene.graph.initial(),
                &Options {
                    linear_solver,
                    ..Options::default()
                },
            )?;
            let work = report
                .iterations
                .iter()
                .map(|i| i.linear_iterations)
                .sum::<usize>();
            println!(
                "{name}: cost={:.9}, RMSE={:.6} m, endpoint error={:.6} m, nonlinear trials={}, PCG iterations={}, termination={:?}",
                report.cost,
                scene.position_rmse(&report.x)?,
                scene.endpoint_error(&report.x)?,
                report.iterations.len(),
                work,
                report.termination
            );
            if matches!(
                report.termination,
                Termination::NoProgress | Termination::MaxIterations
            ) {
                return Err("preconditioned solve did not converge".into());
            }
            if let Some(r) = &reference {
                if (report.cost - r.cost).abs() > 1e-8 || (&report.x - &r.x).norm() > 1e-6 {
                    return Err("preconditioners produced different solutions".into());
                }
            }
            reference = Some(report);
        }
    }
    Ok(())
}
