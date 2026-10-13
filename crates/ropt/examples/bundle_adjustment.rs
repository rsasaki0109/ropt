//! Deterministic noise-free BA. Two known camera poses anchor world pose and metric scale.
#[path = "support/ba.rs"]
mod ba;
use ropt::{sparse::CgOptions, *};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let autodiff = std::env::args().skip(1).any(|arg| arg == "--autodiff");
    let scene = if autodiff {
        ba::scene_with_autodiff(0.0, true)?
    } else {
        ba::scene()?
    };
    println!(
        "Jacobian: {}",
        if autodiff {
            "forward AD on right tangents"
        } else {
            "analytic"
        }
    );
    let initial = scene.graph.initial();
    let initial_rms = scene.rms(&initial)?;
    let (retained, block_size) = scene.graph.schur_layout(&scene.points)?;
    println!(
        "4 cameras, 12 points, 48 observations; two camera poses fixed; initial RMS={initial_rms:.6} pixels"
    );
    for (name, linear_solver) in [
        ("dense", LinearSolver::Dense),
        (
            "block Schur",
            LinearSolver::BlockSchur {
                retained,
                block_size,
                cg: CgOptions::default(),
            },
        ),
    ] {
        let report = solve_local(
            &scene.graph,
            initial.clone(),
            &Options {
                linear_solver,
                max_iterations: 100,
                ..Options::default()
            },
        )?;
        let rms = scene.rms(&report.x)?;
        let error = scene.max_state_error(&report.x)?;
        println!(
            "{name}: RMS={rms:.3e} pixels, max state error={error:.3e}, termination={:?}, iterations={}",
            report.termination,
            report.iterations.len()
        );
        if rms > 1e-6 || error > 1e-5 {
            return Err("BA did not recover the synthetic solution".into());
        }
    }
    Ok(())
}
