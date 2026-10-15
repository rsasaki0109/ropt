//! Export actual deterministic solver checkpoints for README visualization.
//! Each checkpoint reruns from the same initial state with an iteration cap;
//! this avoids adding snapshot allocations to the production solver.
#[path = "support/ba.rs"]
mod ba;
use nalgebra::Vector3;
use ropt::{factor::read_se3, sparse::CgOptions, *};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};
fn vector(out: &mut impl Write, values: &[f64]) -> std::io::Result<()> {
    write!(out, "[")?;
    for (i, value) in values.iter().enumerate() {
        if i > 0 {
            write!(out, ",")?;
        }
        write!(out, "{value:.17e}")?;
    }
    write!(out, "]")
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scene = ba::scene()?;
    let initial = scene.graph.initial();
    let (retained, block_size) = scene.graph.schur_layout(&scene.points)?;
    let options = Options {
        linear_solver: LinearSolver::BlockSchur {
            retained,
            block_size,
            cg: CgOptions::default(),
        },
        ..Options::default()
    };
    let reference = solve_local(&scene.graph, initial.clone(), &options)?;
    if scene.rms(&reference.x)? > 1e-7 || scene.max_state_error(&reference.x)? > 1e-6 {
        return Err("BA trace did not converge".into());
    }
    let mut states = vec![initial.clone()];
    for count in 1..=reference.iterations.len() {
        let checkpoint = solve_local(
            &scene.graph,
            initial.clone(),
            &Options {
                max_iterations: count,
                ..options.clone()
            },
        )?;
        let expected = &reference.iterations[count - 1];
        let expected_cost = if expected.accepted {
            expected.trial_cost
        } else {
            expected.cost
        };
        if (checkpoint.cost - expected_cost).abs() > 1e-12 * expected_cost.abs().max(1.0) {
            return Err("checkpoint disagrees with full solve".into());
        }
        states.push(checkpoint.x);
    }
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "docs/assets/ba_trace.json".into());
    if let Some(parent) = Path::new(&path)
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = BufWriter::new(File::create(&path)?);
    write!(
        out,
        "{{\"schema\":1,\"solver\":\"LM + block Schur\",\"synthetic\":true,\"camera_count\":4,\"point_count\":12,\"observation_count\":48,\"fixed_cameras\":[0,1],\"truth_points\":["
    )?;
    for (i, p) in scene.truth_points.iter().enumerate() {
        if i > 0 {
            write!(out, ",")?;
        }
        vector(&mut out, p.as_slice())?;
    }
    write!(out, "],\"observed_camera\":2,\"observations\":[")?;
    for (i, (_, _, pixel)) in scene
        .measurements
        .iter()
        .filter(|(camera, _, _)| *camera == 2)
        .enumerate()
    {
        if i > 0 {
            write!(out, ",")?;
        }
        vector(&mut out, pixel.as_slice())?;
    }
    write!(out, "],\"states\":[")?;
    for (step, x) in states.iter().enumerate() {
        if step > 0 {
            write!(out, ",")?;
        }
        write!(
            out,
            "{{\"iteration\":{step},\"rms\":{:.17e},\"max_state_error\":{:.17e},\"accepted\":{},\"points\":[",
            scene.rms(x)?,
            scene.max_state_error(x)?,
            step == 0 || reference.iterations[step - 1].accepted
        )?;
        for (i, id) in scene.points.iter().enumerate() {
            if i > 0 {
                write!(out, ",")?;
            }
            vector(&mut out, scene.graph.value(x, *id)?.as_slice())?;
        }
        write!(out, "],\"cameras\":[")?;
        for (i, id) in scene.cameras.iter().enumerate() {
            if i > 0 {
                write!(out, ",")?;
            }
            let pose = read_se3(&scene.graph.value(x, *id)?)?;
            write!(out, "{{\"center\":")?;
            vector(&mut out, pose.translation.as_slice())?;
            write!(out, ",\"frustum\":[")?;
            for (j, (a, b)) in [(0.0, 0.0), (640.0, 0.0), (640.0, 480.0), (0.0, 480.0)]
                .iter()
                .enumerate()
            {
                if j > 0 {
                    write!(out, ",")?;
                }
                let p = pose.translation
                    + pose.rotation.0
                        * Vector3::new(
                            (a - scene.calibration.cx) / scene.calibration.fx * 0.5,
                            (b - scene.calibration.cy) / scene.calibration.fy * 0.5,
                            0.5,
                        );
                vector(&mut out, p.as_slice())?;
            }
            write!(out, "]}}")?;
        }
        write!(out, "],\"projected\":[")?;
        let pose = read_se3(&scene.graph.value(x, scene.cameras[2])?)?;
        for (i, id) in scene.points.iter().enumerate() {
            if i > 0 {
                write!(out, ",")?;
            }
            let point = Vector3::from_column_slice(scene.graph.value(x, *id)?.as_slice());
            let pixel = scene
                .calibration
                .project(pose.rotation.0.inverse() * (point - pose.translation))?;
            vector(&mut out, pixel.as_slice())?;
        }
        write!(out, "]}}")?;
    }
    writeln!(out, "]}}")?;
    out.flush()?;
    println!("Exported {} solver checkpoints to {path}", states.len());
    Ok(())
}
