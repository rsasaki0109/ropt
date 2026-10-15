//! Synthetic shared calibration BA with three known, diverse metric camera poses.
use nalgebra::{Vector2, Vector3};
use ropt::{camera::*, factor::*, manifold::*, *};
pub struct Scene {
    pub graph: FactorGraph,
    pub calibration: VariableId,
    pub cameras: Vec<VariableId>,
    pub points: Vec<VariableId>,
    pub truth_calibration: BrownConrady,
    pub truth_cameras: Vec<SE3>,
    pub truth_points: Vec<Vector3<f64>>,
    measurements: Vec<(usize, usize, Vector2<f64>)>,
}
pub fn scene(noise: f64) -> Result<Scene, Error> {
    if !noise.is_finite() || noise < 0.0 {
        return Err(Error::InvalidOptions);
    }
    let truth_calibration = BrownConrady::new(
        Pinhole::new(420.0, 415.0, 320.0, 240.0)?,
        [-0.12, 0.035, 0.004, -0.003],
    )?;
    let initial_calibration = BrownConrady::new(
        Pinhole::new(450.0, 390.0, 324.0, 235.0)?,
        [-0.07, 0.015, 0.0, 0.0],
    )?;
    let truth_cameras = [
        ([-0.7, -0.35, 0.0], [0.05, -0.08, 0.0]),
        ([0.75, -0.2, 0.15], [-0.1, 0.12, 0.03]),
        ([0.05, 0.65, -0.05], [0.12, -0.05, -0.08]),
        ([-0.45, 0.25, 0.25], [0.04, 0.1, 0.08]),
        ([0.5, 0.45, -0.1], [-0.08, -0.1, 0.04]),
        ([0.0, -0.55, 0.12], [0.09, 0.08, -0.06]),
    ]
    .iter()
    .map(|(t, r)| SE3 {
        translation: Vector3::from_column_slice(t),
        rotation: SO3::exp(Vector3::from_column_slice(r)),
    })
    .collect::<Vec<_>>();
    let truth_points = (0..25)
        .map(|i| {
            Vector3::new(
                -1.6 + (i % 5) as f64 * 0.8,
                -1.2 + (i / 5) as f64 * 0.6,
                3.5 + (i % 4) as f64 * 0.45,
            )
        })
        .collect::<Vec<_>>();
    let mut graph = FactorGraph::new();
    let calibration = graph.add_variable(
        VariableKind::Euclidean(8),
        initial_calibration.parameters()?,
    )?;
    let mut cameras = Vec::new();
    for (i, &truth) in truth_cameras.iter().enumerate() {
        let pose = if i < 3 {
            truth
        } else {
            truth.retract(Tangent6::new(0.03, -0.02, 0.025, 0.006, -0.008, 0.005))
        };
        let id = graph.add_variable(VariableKind::SE3, se3_values(pose))?;
        if i < 3 {
            graph.set_fixed(id, true)?;
        }
        cameras.push(id);
    }
    let points = truth_points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            graph.add_variable(
                VariableKind::Euclidean(3),
                DVector::from_column_slice(
                    (p + Vector3::new(0.03 * (i as f64 + 1.0).sin(), -0.025, 0.05)).as_slice(),
                ),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut measurements = Vec::new();
    for (i, &pose) in truth_cameras.iter().enumerate() {
        for (j, &point) in truth_points.iter().enumerate() {
            let index = (i * truth_points.len() + j + 1) as f64;
            let observed = truth_calibration.reprojection(pose, point, Vector2::zeros())?
                + noise * Vector2::new(index.sin(), (index * 1.7).cos());
            add_calibration_reprojection_factor(
                &mut graph,
                cameras[i],
                points[j],
                calibration,
                observed,
                FactorOptions::default(),
            )?;
            measurements.push((i, j, observed));
        }
    }
    Ok(Scene {
        graph,
        calibration,
        cameras,
        points,
        truth_calibration,
        truth_cameras,
        truth_points,
        measurements,
    })
}
impl Scene {
    pub fn model(&self, x: &DVector<f64>) -> Result<BrownConrady, Error> {
        BrownConrady::from_parameters(self.graph.value(x, self.calibration)?.as_slice())
    }
    pub fn rms(&self, x: &DVector<f64>) -> Result<f64, Error> {
        let model = self.model(x)?;
        let mut squared = 0.0;
        for &(i, j, obs) in &self.measurements {
            let pose = read_se3(&self.graph.value(x, self.cameras[i])?)?;
            let point = Vector3::from_column_slice(self.graph.value(x, self.points[j])?.as_slice());
            squared += model.reprojection(pose, point, obs)?.norm_squared();
        }
        Ok((squared / self.measurements.len() as f64).sqrt())
    }
    pub fn max_state_error(&self, x: &DVector<f64>) -> Result<f64, Error> {
        let mut error: f64 = 0.0;
        for (&id, &truth) in self.cameras.iter().zip(&self.truth_cameras) {
            error = error.max(truth.local(read_se3(&self.graph.value(x, id)?)?).norm());
        }
        for (&id, truth) in self.points.iter().zip(&self.truth_points) {
            error = error.max(
                (Vector3::from_column_slice(self.graph.value(x, id)?.as_slice()) - truth).norm(),
            );
        }
        Ok(error)
    }
    pub fn max_calibration_error(&self, x: &DVector<f64>) -> Result<f64, Error> {
        let model = self.model(x)?;
        let truth = self.truth_calibration;
        let p = model.pinhole;
        let q = truth.pinhole;
        Ok([p.fx - q.fx, p.fy - q.fy, p.cx - q.cx, p.cy - q.cy]
            .into_iter()
            .chain(
                model
                    .distortion
                    .into_iter()
                    .zip(truth.distortion)
                    .map(|(a, b)| a - b),
            )
            .fold(0.0_f64, |e, d| e.max(d.abs())))
    }
}
