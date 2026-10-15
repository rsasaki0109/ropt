//! Per-observation whitening and joint robust loss, with fixed synthetic outliers.
#[path = "support/robust_ba.rs"]
mod ba;
use ropt::{sparse::CgOptions, *};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rms = Vec::new();
    for (name, loss) in [
        ("squared", Loss::Squared),
        ("joint Huber", Loss::Huber(2.0)),
    ] {
        let scene = ba::scene(loss)?;
        let (retained, block_size) = scene.graph.schur_layout(&scene.points)?;
        let report = solve_local(
            &scene.graph,
            scene.graph.initial(),
            &Options {
                linear_solver: LinearSolver::BlockSchur {
                    retained,
                    block_size,
                    cg: CgOptions::default(),
                },
                max_iterations: 200,
                ..Options::default()
            },
        )?;
        let value = scene.inlier_rms(&report.x)?;
        rms.push(value);
        println!(
            "{name}: inlier RMS={value:.6} pixels, cost={:.6}, iterations={}, termination={:?}",
            report.cost,
            report.iterations.len(),
            report.termination
        );
    }
    if rms[1] >= rms[0] {
        return Err("robust model did not improve the inlier fit".into());
    }
    Ok(())
}
