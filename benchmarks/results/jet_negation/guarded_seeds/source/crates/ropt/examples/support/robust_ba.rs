//! Deterministic BA with heterogeneous measurement noise and two gross outliers.
use nalgebra::{Vector2, Vector3};
use ropt::{camera::*, factor::*, manifold::*, noise::NoiseModel, *};
pub struct Scene {
    pub graph: FactorGraph,
    pub points: Vec<VariableId>,
    cameras: Vec<VariableId>,
    inliers: Vec<(usize, usize, Vector2<f64>)>,
    calibration: Pinhole,
}
pub fn scene(loss: Loss) -> Result<Scene, Error> {
    let calibration = Pinhole::new(420.0, 415.0, 320.0, 240.0)?;
    let poses: Vec<_> = (0..4)
        .map(|i| {
            SE3::exp(Tangent6::new(
                i as f64 * 0.5,
                0.03 * i as f64,
                0.0,
                0.01 * i as f64,
                -0.02 * i as f64,
                0.03 * i as f64,
            ))
        })
        .collect();
    let world: Vec<_> = (0..12)
        .map(|i| {
            Vector3::new(
                -0.5 + (i % 4) as f64 * 0.65,
                -0.6 + (i / 4) as f64 * 0.55,
                4.0 + (i % 3) as f64 * 0.5,
            )
        })
        .collect();
    let mut graph = FactorGraph::new();
    let mut cameras = Vec::new();
    let mut points = Vec::new();
    for (i, pose) in poses.iter().enumerate() {
        let value = if i < 2 {
            *pose
        } else {
            pose.retract(Tangent6::new(0.04, -0.02, 0.03, 0.005, -0.01, 0.01))
        };
        let id = graph.add_variable(VariableKind::SE3, se3_values(value))?;
        if i < 2 {
            graph.set_fixed(id, true)?;
        }
        cameras.push(id);
    }
    for (i, point) in world.iter().enumerate() {
        points.push(graph.add_variable(
            VariableKind::Euclidean(3),
            DVector::from_column_slice(
                (point + Vector3::new(0.03 * (i as f64).sin(), -0.02, 0.05)).as_slice(),
            ),
        )?);
    }
    let mut inliers = Vec::new();
    for (camera, pose) in poses.iter().enumerate() {
        for (point, p) in world.iter().enumerate() {
            let index = (camera * 12 + point + 1) as f64;
            let sigma = if (camera + point) % 3 == 0 { 1.2 } else { 0.5 };
            let noise = Vector2::new(index.sin() * sigma, (index * 1.7).cos() * sigma * 1.2) * 0.2;
            let mut observed =
                calibration.project(pose.rotation.0.inverse() * (p - pose.translation))? + noise;
            if (camera, point) == (2, 6) {
                observed += Vector2::new(35.0, -25.0);
            } else if (camera, point) == (3, 9) {
                observed += Vector2::new(-30.0, 20.0);
            } else {
                inliers.push((camera, point, observed));
            }
            let options = FactorOptions {
                noise: NoiseModel::diagonal(DVector::from_vec(vec![sigma, sigma * 1.2]))?,
                loss: Some(loss),
            };
            add_reprojection_factor_with_options(
                &mut graph,
                cameras[camera],
                points[point],
                calibration,
                observed,
                options,
            )?;
        }
    }
    Ok(Scene {
        graph,
        points,
        cameras,
        inliers,
        calibration,
    })
}
impl Scene {
    pub fn inlier_rms(&self, x: &DVector<f64>) -> Result<f64, Error> {
        let mut squared = 0.0;
        for &(camera, point, pixel) in &self.inliers {
            let pose = read_se3(&self.graph.value(x, self.cameras[camera])?)?;
            let p = Vector3::from_column_slice(self.graph.value(x, self.points[point])?.as_slice());
            squared += self
                .calibration
                .reprojection(pose, p, pixel)?
                .residual
                .norm_squared();
        }
        Ok((squared / self.inliers.len() as f64).sqrt())
    }
}
