//! Jointly estimate shared intrinsics/distortion, three camera poses and 25 landmarks.
#[path = "support/calibration_ba.rs"]
mod ba;
use ropt::{sparse::CgOptions, *};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scene = ba::scene(0.0)?;
    let initial = scene.graph.initial();
    let (retained, block_size) = scene.graph.schur_layout(&scene.points)?;
    println!(
        "6 cameras (3 known), 25 points, 150 observations, shared 8-parameter calibration; initial RMS={:.6} pixels",
        scene.rms(&initial)?
    );
    for (name, linear_solver) in [
        ("dense", LinearSolver::Dense),
        ("dense QR", LinearSolver::DenseQr(QrOptions::default())),
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
                ..Options::default()
            },
        )?;
        let rms = scene.rms(&report.x)?;
        let state_error = scene.max_state_error(&report.x)?;
        let calibration_error = scene.max_calibration_error(&report.x)?;
        let model = scene.model(&report.x)?;
        println!(
            "{name}: RMS={rms:.3e}, max state error={state_error:.3e}, max physical calibration error={calibration_error:.3e}, {:?} in {} trials",
            report.termination,
            report.iterations.len()
        );
        println!(
            "  fx={:.9}, fy={:.9}, cx={:.9}, cy={:.9}; [k1,k2,p1,p2]={:?}",
            model.pinhole.fx,
            model.pinhole.fy,
            model.pinhole.cx,
            model.pinhole.cy,
            model.distortion
        );
        if rms > 1e-6
            || state_error > 1e-5
            || calibration_error > 1e-4
            || matches!(
                report.termination,
                Termination::NoProgress | Termination::MaxIterations
            )
        {
            return Err("synthetic calibration BA did not recover truth".into());
        }
    }
    Ok(())
}
