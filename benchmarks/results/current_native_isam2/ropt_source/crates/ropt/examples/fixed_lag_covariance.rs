//! Retired SE3 poses and body-tangent covariance from an anchored odometry chain.
//! Synthetic exact measurements; this illustrates uncertainty propagation, not
//! empirical coverage or performance. Covariance explicitly uses dense SVD.
use nalgebra::Vector3;
use ropt::{
    DVector,
    covariance::CovarianceOptions,
    factor::{FactorOptions, VariableKind, se3_values},
    fixed_lag::*,
    manifold::{SE3, SO3, Tangent6},
    noise::NoiseModel,
    pose_graph::RelativePose3,
};
use std::collections::BTreeSet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let adaptive = match arguments.as_slice() {
        [] => false,
        [flag] if flag == "--adaptive" => true,
        _ => return Err("usage: fixed_lag_covariance [--adaptive]".into()),
    };
    let mut smoother = FixedLagSmoother::new(FixedLagOptions {
        lag: 2.,
        ..Default::default()
    })?;
    let covariance_options = CovarianceOptions::default();
    let root = SE3 {
        translation: Vector3::new(2., -1., 0.4),
        rotation: SO3::exp(Vector3::new(0.2, -0.3, 0.5)),
    };
    let increment = SE3::exp(Tangent6::from_row_slice(&[
        0.6, 0.1, -0.05, 0.02, -0.03, 0.08,
    ]));
    let mut previous = root;
    let mut archived = Vec::new();
    let mut active = BTreeSet::new();
    if adaptive {
        println!(
            "frame,lag,live_variables,live_factors,retired_count,std_body_tx,std_body_ty,std_body_tz,std_body_rx,std_body_ry,std_body_rz"
        );
    } else {
        println!(
            "frame,key,x,y,z,qw,qx,qy,qz,std_body_tx,std_body_ty,std_body_tz,std_body_rx,std_body_ry,std_body_rz,cross_covariance_norm"
        );
    }
    for frame in 1..=12_u64 {
        if adaptive {
            match frame {
                5 => smoother.set_lag(4.)?,
                9 => smoother.set_lag(1.)?,
                _ => {}
            }
        }
        active.insert(frame);
        let cutoff = frame as f64 - smoother.lag();
        let retiring = active
            .iter()
            .copied()
            .filter(|&key| (key as f64) < cutoff)
            .collect::<Vec<_>>();
        let pose = previous.compose(increment);
        let mut variables = vec![WindowVariable {
            key: frame,
            kind: VariableKind::SE3,
            value: se3_values(pose),
            timestamp: Some(frame as f64),
            fixed: false,
        }];
        if frame == 1 {
            // A fixed persistent root supplies the gauge. It has no estimated covariance.
            variables.push(WindowVariable {
                key: 0,
                kind: VariableKind::SE3,
                value: se3_values(root),
                timestamp: None,
                fixed: true,
            });
        }
        let residual = RelativePose3::new(increment)?;
        let factor = SmootherFactor::autodiff(
            &[frame - 1, frame],
            6,
            6,
            FactorOptions {
                noise: NoiseModel::diagonal(DVector::from_vec(vec![
                    0.05, 0.06, 0.07, 0.01, 0.02, 0.03,
                ]))?,
                ..Default::default()
            },
            move |v| residual.residual_autodiff(v[0], v[1]),
        );
        let keys = retiring.iter().copied().chain([frame]).collect::<Vec<_>>();
        let (report, covariance) = smoother.update_with_covariance(
            frame as f64,
            SmootherUpdate {
                variables,
                factors: vec![factor],
                ..Default::default()
            },
            &keys,
            &covariance_options,
        )?;

        // The last requested block is always the latest pose. Compare it with
        // an independent query after Schur elimination and layout compaction.
        let latest_offset = covariance.matrix.nrows() - 6;
        let latest_before = covariance
            .matrix
            .view((latest_offset, latest_offset), (6, 6));
        let latest_after = smoother.marginal_covariance(&[frame], &covariance_options)?;
        if (&latest_after.matrix - latest_before).norm() > 1e-9 * latest_before.norm().max(1.) {
            return Err("retained covariance changed during marginalization".into());
        }
        if report.marginalized != retiring {
            return Err("unexpected retirement".into());
        }
        for (index, &retired) in retiring.iter().enumerate() {
            if smoother.estimate(retired).is_some() {
                return Err("retired variable still live".into());
            }
            let value = report.marginalized_estimates[&retired].clone();
            let marginal = covariance
                .matrix
                .view((6 * index, 6 * index), (6, 6))
                .into_owned();
            let cross_norm = covariance
                .matrix
                .view((6 * index, latest_offset), (6, 6))
                .norm();
            if cross_norm <= 0.
                || marginal
                    .diagonal()
                    .iter()
                    .any(|&v| !v.is_finite() || v <= 0.)
            {
                return Err("invalid uncertainty snapshot".into());
            }
            if !adaptive {
                print!("{frame},{retired}");
                for number in value
                    .iter()
                    .copied()
                    .chain(marginal.diagonal().iter().map(|v| v.sqrt()))
                    .chain([cross_norm])
                {
                    print!(",{number:.9}");
                }
                println!();
            }
            // An application may write these owned values/matrices to durable
            // storage. They describe this retirement point and are never revised.
            archived.push((retired, value, marginal));
            active.remove(&retired);
        }
        if adaptive {
            print!(
                "{frame},{},{},{},{}",
                smoother.lag(),
                smoother.variable_count(),
                smoother.factor_count(),
                retiring.len()
            );
            for variance in latest_after.matrix.diagonal().iter() {
                if !variance.is_finite() || *variance <= 0. {
                    return Err("invalid latest-pose variance".into());
                }
                print!(",{:.9}", variance.sqrt());
            }
            println!();
        }
        if smoother.variable_count() != active.len() + 1 || smoother.factor_count() != active.len()
        {
            return Err("window growth exceeded expected bounds".into());
        }
        previous = pose;
    }
    let expected_retired = if adaptive { 10 } else { 9 };
    if archived.len() != expected_retired || archived.len() + smoother.variable_count() != 13 {
        return Err("incomplete trajectory archive".into());
    }
    eprintln!(
        "12 updates; {} retired pose/covariance snapshots; {} live variables including the fixed root; pre/post covariance checks passed",
        archived.len(),
        smoother.variable_count()
    );
    Ok(())
}
