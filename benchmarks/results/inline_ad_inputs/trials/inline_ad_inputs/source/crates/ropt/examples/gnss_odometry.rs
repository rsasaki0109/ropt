//! GNSS/odometry trajectory refinement in a metric local world frame, with a known initial pose.
#[path = "support/gnss.rs"]
mod gnss;
use ropt::{
    covariance::{CovarianceInput, CovarianceModel, CovarianceOptions},
    factor::read_se3,
    sparse::CgOptions,
    *,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut errors = Vec::new();
    for (name, loss) in [
        ("odometry only", None),
        ("GNSS squared", Some(Loss::Squared)),
        ("GNSS joint Huber", Some(Loss::Huber(2.5))),
    ] {
        let scene = gnss::scene(loss)?;
        let report = solve_local(
            &scene.graph,
            scene.graph.initial(),
            &Options {
                linear_solver: LinearSolver::SparseCg(CgOptions::default()),
                ..Options::default()
            },
        )?;
        let rmse = scene.position_rmse(&report.x)?;
        let endpoint = scene.endpoint_error(&report.x)?;
        errors.push(rmse);
        println!(
            "{name}: body-position RMSE={rmse:.6} m (all 36 timestamps), endpoint error={endpoint:.6} m, termination={:?}, trials={}",
            report.termination,
            report.iterations.len()
        );
        println!(
            "  9 available GNSS fixes ({} outliers); known lever arm={:?}",
            scene.fixes.iter().filter(|f| f.2).count(),
            scene.lever_arm.as_slice()
        );
        let last = *scene.poses.last().unwrap();
        let covariance = scene.graph.marginal_covariance(
            &report.x,
            &[last],
            &CovarianceOptions {
                input: CovarianceInput::SparseToDense,
                model: if matches!(loss, Some(Loss::Huber(_))) {
                    CovarianceModel::FrozenIrls {
                        fallback_loss: Loss::Squared,
                    }
                } else {
                    CovarianceModel::Gaussian
                },
                ..CovarianceOptions::default()
            },
        )?;
        let rotation = read_se3(&scene.graph.value(&report.x, last)?)?
            .rotation
            .0
            .to_rotation_matrix();
        let r = DMatrix::from_column_slice(3, 3, rotation.matrix().as_slice());
        let world_position = &r * covariance.matrix.view((0, 0), (3, 3)) * r.transpose();
        let diagonal = world_position.diagonal();
        if !diagonal.iter().all(|v| v.is_finite() && *v > 0.0) {
            return Err("invalid propagated position covariance".into());
        }
        println!(
            "  endpoint local model world-position std={:?} m; {:?}; J condition={:.3e}",
            diagonal.map(f64::sqrt).as_slice(),
            covariance.model,
            covariance.condition_number
        );
        if matches!(
            report.termination,
            Termination::MaxIterations | Termination::NoProgress
        ) {
            return Err("trajectory refinement did not converge".into());
        }
    }
    if errors[2] >= errors[0] || errors[2] >= errors[1] {
        return Err("robust GNSS did not improve this synthetic trajectory".into());
    }
    Ok(())
}
