//! Sparse Givens row updates, with periodic nonlinear refresh and late loop closure.
use nalgebra::{Vector2, Vector3};
use ropt::{
    Error, factor::*, fixed_lag::*, isam::*, manifold::*, noise::NoiseModel, pose_graph::*,
};
fn between(a: u64, b: u64, m: SE2, sigma: f64) -> Result<SmootherFactor, Error> {
    let r = RelativePose2::new(m)?;
    Ok(SmootherFactor::autodiff(
        &[a, b],
        3,
        6,
        FactorOptions {
            noise: NoiseModel::isotropic(sigma)?,
            ..Default::default()
        },
        move |v| r.residual_autodiff(v[0], v[1]),
    ))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut smoother = IsamSmoother::new(IsamOptions::default())?;
    let origin = SE2 {
        translation: Vector2::new(2., -1.),
        rotation: SO2::exp(0.4),
    };
    let target = se2_values(origin);
    smoother.update(
        SmootherUpdate {
            variables: vec![WindowVariable {
                key: 0,
                kind: VariableKind::SE2,
                value: target.clone(),
                timestamp: None,
                fixed: true,
            }],
            factors: vec![SmootherFactor::autodiff(
                &[0],
                3,
                3,
                Default::default(),
                move |v| VariableKind::SE2.local_autodiff(&target, v[0]),
            )],
            ..Default::default()
        },
        false,
    )?;
    let increment = SE2::exp(Vector3::new(0.6, 0.05, 0.06));
    let mut truth = origin;
    let mut incremental = 0;
    let mut rebuilt = 0;
    println!("frame,update_kind,factors_linearized,stored_qr_entries,endpoint_error");
    for frame in 1..=40_u64 {
        truth = truth.compose(increment);
        let phase = frame as f64;
        let measured = increment.retract(Vector3::new(
            0.012 * phase.sin(),
            0.008 * (phase * 0.5).cos(),
            0.003 * (phase * 0.4).sin(),
        ));
        let seed = read_se2(
            smoother
                .estimate(frame - 1)
                .ok_or("missing previous state")?,
        )?
        .compose(measured);
        let mut factors = vec![between(frame - 1, frame, measured, 0.05)?];
        if frame % 10 == 0 {
            factors.push(between(0, frame, origin.inverse().compose(truth), 0.01)?);
        }
        let report = smoother.update(
            SmootherUpdate {
                variables: vec![WindowVariable {
                    key: frame,
                    kind: VariableKind::SE2,
                    value: se2_values(seed),
                    timestamp: None,
                    fixed: false,
                }],
                factors,
                ..Default::default()
            },
            false,
        )?;
        match report.kind {
            IsamUpdateKind::Incremental => incremental += 1,
            IsamUpdateKind::Relinearized => rebuilt += 1,
        }
        let error = (read_se2(&report.estimates[&frame])?.translation - truth.translation).norm();
        if frame % 10 == 0 || frame == 1 {
            println!(
                "{frame},{:?},{},{},{error:.8}",
                report.kind, report.factors_linearized, report.stored_entries
            );
        }
    }
    let report = smoother.update(SmootherUpdate::default(), true)?;
    let error = (read_se2(&report.estimates[&40])?.translation - truth.translation).norm();
    if error > 0.03 || incremental == 0 || rebuilt == 0 {
        return Err("incremental/loop regression failed".into());
    }
    eprintln!(
        "40 poses added: {incremental} incremental updates, {rebuilt} automatic full refreshes; final explicit refresh; endpoint error={error:.8} m; live variables={}, factors={}",
        smoother.variable_count(),
        smoother.factor_count()
    );
    Ok(())
}
