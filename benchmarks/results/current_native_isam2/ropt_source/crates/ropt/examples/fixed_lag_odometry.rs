//! Synthetic SE2 odometry + Cartesian position observations in a five-second window.
use nalgebra::{Vector2, Vector3};
use ropt::factor::{FactorOptions, VariableKind, read_se2, se2_values};
use ropt::fixed_lag::*;
use ropt::manifold::{SE2, SO2};
use ropt::noise::NoiseModel;
use ropt::pose_graph::RelativePose2;
use ropt::{DVector, Error};

fn position(key: u64, measurement: Vector2<f64>) -> Result<SmootherFactor, Error> {
    Ok(SmootherFactor::autodiff(
        &[key],
        2,
        3,
        FactorOptions {
            noise: NoiseModel::diagonal(DVector::from_element(2, 0.1))?,
            ..FactorOptions::default()
        },
        move |v| Ok(vec![&v[0][0] - measurement.x, &v[0][1] - measurement.y]),
    ))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut smoother = FixedLagSmoother::new(FixedLagOptions {
        lag: 5.,
        ..FixedLagOptions::default()
    })?;
    let mut truth = SE2 {
        rotation: SO2::exp(0.4),
        translation: Vector2::new(2., -1.),
    };
    smoother.update(
        0.,
        SmootherUpdate {
            variables: vec![WindowVariable {
                key: 0,
                kind: VariableKind::SE2,
                value: se2_values(truth),
                timestamp: Some(0.),
                fixed: true,
            }],
            factors: vec![position(0, truth.translation)?],
            ..SmootherUpdate::default()
        },
    )?;
    let increment = SE2::exp(Vector3::new(0.6, 0.05, 0.06));
    let mut max_error = 0.0_f64;
    let mut max_variables = 0;
    let mut max_factors = 0;
    let mut finalized = Vec::new();
    println!("frame,live_variables,live_factors,expired,position_error,window_surrogate_cost");
    for frame in 1..=80_u64 {
        truth = truth.compose(increment);
        let phase = frame as f64;
        let measured = increment.retract(Vector3::new(
            0.015 * phase.sin(),
            0.02 * (phase * 0.7).cos(),
            0.005 * (phase * 0.3).sin(),
        ));
        let seed = read_se2(
            smoother
                .estimate(frame - 1)
                .ok_or("previous pose expired")?,
        )?
        .compose(measured);
        let residual = RelativePose2::new(measured)?;
        let between = SmootherFactor::autodiff(
            &[frame - 1, frame],
            3,
            6,
            FactorOptions {
                noise: NoiseModel::diagonal(DVector::from_vec(vec![0.05, 0.05, 0.02]))?,
                ..FactorOptions::default()
            },
            move |v| residual.residual_autodiff(v[0], v[1]),
        );
        let observation = truth.translation
            + Vector2::new(0.06 * (phase * 0.5).sin(), 0.06 * (phase * 0.7).cos());
        let report = smoother.update(
            phase,
            SmootherUpdate {
                variables: vec![WindowVariable {
                    key: frame,
                    kind: VariableKind::SE2,
                    value: se2_values(seed),
                    timestamp: Some(phase),
                    fixed: false,
                }],
                factors: vec![between, position(frame, observation)?],
                ..SmootherUpdate::default()
            },
        )?;
        let estimate = read_se2(&report.estimates[&frame])?;
        let error = (estimate.translation - truth.translation).norm();
        max_error = max_error.max(error);
        max_variables = max_variables.max(smoother.variable_count());
        max_factors = max_factors.max(smoother.factor_count());
        if frame % 10 == 0 {
            println!(
                "{frame},{},{},{},{error:.8},{:.8}",
                smoother.variable_count(),
                smoother.factor_count(),
                report.marginalized.len(),
                report.final_cost
            );
        }
        // Applications can retain final poses while the smoother stays bounded.
        // These snapshots are never revised by later window optimizations.
        finalized.extend(report.marginalized_estimates);
        if max_variables > 6 || max_factors > 12 || error > 0.3 {
            return Err("window/error regression failed".into());
        }
    }
    if finalized.len() + smoother.variable_count() != 81 {
        return Err("missing trajectory snapshots".into());
    }
    for (_, value) in &finalized {
        read_se2(value)?;
    }
    eprintln!(
        "retained {} finalized poses outside the live window",
        finalized.len()
    );
    let covariance = smoother.marginal_covariance(&[80], &Default::default())?;
    eprintln!(
        "80 frames; maximum live variables={max_variables}, factors={max_factors}, position error={max_error:.8} m; latest right-tangent stddev={:?}",
        covariance.matrix.diagonal().map(f64::sqrt).as_slice()
    );
    Ok(())
}
