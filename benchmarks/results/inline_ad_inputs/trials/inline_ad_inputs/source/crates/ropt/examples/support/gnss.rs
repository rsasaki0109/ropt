//! Synthetic local Cartesian GNSS fixes with two gross outliers and biased odometry.
use nalgebra::Vector3;
use ropt::{factor::*, manifold::*, navigation::*, noise::NoiseModel, pose_graph::*, *};
pub struct Scene {
    pub graph: FactorGraph,
    pub poses: Vec<VariableId>,
    pub truth: Vec<SE3>,
    pub lever_arm: Vector3<f64>,
    pub fixes: Vec<(usize, Vector3<f64>, bool)>,
}
pub fn scene(gnss_loss: Option<Loss>) -> Result<Scene, Error> {
    let n = 36;
    let lever_arm = Vector3::new(0.3, -0.12, 0.8);
    let truth = (0..n)
        .map(|i| {
            let k = i as f64;
            SE3 {
                translation: Vector3::new(0.45 * k, 1.1 * (0.11 * k).sin(), 0.3 * (0.16 * k).sin()),
                rotation: SO3::exp(Vector3::new(
                    0.025 * (0.1 * k).sin(),
                    0.035 * (0.08 * k).cos(),
                    0.04 * k,
                )),
            }
        })
        .collect::<Vec<_>>();
    let measurements = (1..n)
        .map(|i| {
            let k = i as f64;
            truth[i - 1]
                .inverse()
                .compose(truth[i])
                .retract(Tangent6::new(
                    0.012,
                    0.005,
                    0.003,
                    0.0006 * k.sin(),
                    -0.0004 * k.cos(),
                    0.0008,
                ))
        })
        .collect::<Vec<_>>();
    let mut initial = vec![truth[0]];
    for &relative in &measurements {
        initial.push(initial.last().unwrap().compose(relative));
    }
    let mut graph = FactorGraph::new();
    let poses = initial
        .into_iter()
        .map(|p| graph.add_variable(VariableKind::SE3, se3_values(p)))
        .collect::<Result<Vec<_>, _>>()?;
    graph.set_fixed(poses[0], true)?;
    let odometry_noise = NoiseModel::diagonal(DVector::from_vec(vec![
        0.04, 0.04, 0.06, 0.012, 0.012, 0.012,
    ]))?;
    for (i, &relative) in measurements.iter().enumerate() {
        add_relative_pose3_factor(
            &mut graph,
            poses[i],
            poses[i + 1],
            relative,
            FactorOptions {
                noise: odometry_noise.clone(),
                loss: Some(Loss::Squared),
            },
        )?;
    }
    let covariance = DMatrix::from_row_slice(
        3,
        3,
        &[
            0.0144, 0.004, 0.002, 0.004, 0.0144, -0.001, 0.002, -0.001, 0.04,
        ],
    );
    let gnss_noise = NoiseModel::from_covariance(covariance)?;
    let mut fixes = Vec::new();
    for index in [4, 8, 12, 16, 20, 24, 28, 32, 35] {
        let k = index as f64;
        let pose = truth[index];
        let mut observed = pose.translation
            + pose.rotation.0 * lever_arm
            + Vector3::new(
                0.04 * (0.7 * k).sin(),
                0.035 * (0.5 * k).cos(),
                0.06 * (0.3 * k).sin(),
            );
        let outlier = match index {
            12 => Some(Vector3::new(3.0, -2.0, 1.0)),
            28 => Some(Vector3::new(-2.5, 2.0, -1.2)),
            _ => None,
        };
        if let Some(offset) = outlier {
            observed += offset;
        }
        fixes.push((index, observed, outlier.is_some()));
        if let Some(loss) = gnss_loss {
            add_position3_factor(
                &mut graph,
                poses[index],
                observed,
                lever_arm,
                FactorOptions {
                    noise: gnss_noise.clone(),
                    loss: Some(loss),
                },
            )?;
        }
    }
    Ok(Scene {
        graph,
        poses,
        truth,
        lever_arm,
        fixes,
    })
}
impl Scene {
    /// Ground-truth body-position RMSE over all timestamps, including corrupted fix times.
    pub fn position_rmse(&self, x: &DVector<f64>) -> Result<f64, Error> {
        let squared = self
            .poses
            .iter()
            .zip(&self.truth)
            .map(|(&id, truth)| {
                Ok(
                    (read_se3(&self.graph.value(x, id)?)?.translation - truth.translation)
                        .norm_squared(),
                )
            })
            .collect::<Result<Vec<_>, Error>>()?
            .iter()
            .sum::<f64>();
        Ok((squared / self.poses.len() as f64).sqrt())
    }
    pub fn endpoint_error(&self, x: &DVector<f64>) -> Result<f64, Error> {
        let p = read_se3(&self.graph.value(x, *self.poses.last().unwrap())?)?;
        Ok((p.translation - self.truth.last().unwrap().translation).norm())
    }
}
