//! Deterministic synthetic BA fixture, shared by the example and regression tests.
use nalgebra::{Vector2, Vector3};
use ropt::{camera::*, factor::*, manifold::*, *};
pub struct Scene {
    pub graph: FactorGraph,
    pub cameras: Vec<VariableId>,
    pub points: Vec<VariableId>,
    pub truth_cameras: Vec<SE3>,
    pub truth_points: Vec<Vector3<f64>>,
    pub measurements: Vec<(usize, usize, Vector2<f64>)>,
    pub calibration: Pinhole,
}
pub fn scene() -> Result<Scene, Error> {
    scene_with_noise(0.0)
}
pub fn scene_with_noise(amplitude: f64) -> Result<Scene, Error> {
    if !amplitude.is_finite() || amplitude < 0.0 {
        return Err(Error::InvalidOptions);
    }
    let calibration = Pinhole::new(420.0, 415.0, 320.0, 240.0)?;
    let truth_cameras: Vec<_> = (0..4)
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
    let truth_points: Vec<_> = (0..12)
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
    for (i, pose) in truth_cameras.iter().enumerate() {
        let value = if i < 2 {
            *pose
        } else {
            pose.retract(Tangent6::new(0.08, -0.04, 0.05, 0.01, -0.015, 0.02))
        };
        let id = graph.add_variable(VariableKind::SE3, se3_values(value))?;
        if i < 2 {
            graph.set_fixed(id, true)?;
        }
        cameras.push(id);
    }
    let mut points = Vec::new();
    for (i, point) in truth_points.iter().enumerate() {
        let perturbation = Vector3::new(0.04 * (i as f64 + 1.0).sin(), -0.03, 0.08);
        points.push(graph.add_variable(
            VariableKind::Euclidean(3),
            DVector::from_column_slice((point + perturbation).as_slice()),
        )?);
    }
    let mut measurements = Vec::new();
    for (camera, pose) in truth_cameras.iter().enumerate() {
        for (point, p) in truth_points.iter().enumerate() {
            let index = (camera * truth_points.len() + point + 1) as f64;
            let noise = Vector2::new(index.sin(), (index * 1.7).cos()) * amplitude;
            let pixel =
                calibration.project(pose.rotation.0.inverse() * (p - pose.translation))? + noise;
            add_reprojection_factor(
                &mut graph,
                cameras[camera],
                points[point],
                calibration,
                pixel,
            )?;
            measurements.push((camera, point, pixel));
        }
    }
    Ok(Scene {
        graph,
        cameras,
        points,
        truth_cameras,
        truth_points,
        measurements,
        calibration,
    })
}
impl Scene {
    pub fn rms(&self, x: &DVector<f64>) -> Result<f64, Error> {
        let mut squared = 0.0;
        for &(camera, point, observed) in &self.measurements {
            let pose = read_se3(&self.graph.value(x, self.cameras[camera])?)?;
            let p = Vector3::from_column_slice(self.graph.value(x, self.points[point])?.as_slice());
            squared += self
                .calibration
                .reprojection(pose, p, observed)?
                .residual
                .norm_squared();
        }
        Ok((squared / self.measurements.len() as f64).sqrt())
    }
    pub fn max_state_error(&self, x: &DVector<f64>) -> Result<f64, Error> {
        let mut error: f64 = 0.0;
        for (id, truth) in self.cameras.iter().zip(&self.truth_cameras) {
            error = error.max(truth.local(read_se3(&self.graph.value(x, *id)?)?).norm());
        }
        for (id, truth) in self.points.iter().zip(&self.truth_points) {
            error = error.max(
                (Vector3::from_column_slice(self.graph.value(x, *id)?.as_slice()) - truth).norm(),
            );
        }
        Ok(error)
    }
}
