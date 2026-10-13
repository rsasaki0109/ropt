//! Pinhole and radial/tangential cameras with fixed or optimized calibration.
//! Camera poses transform camera coordinates to world coordinates. Landmarks are
//! in world coordinates. Tangent order is translation then rotation.
use crate::{Error, manifold::SE3};
use nalgebra::{Matrix3, SMatrix, Vector2, Vector3};
#[derive(Debug, Clone, Copy)]
pub struct Pinhole {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}
impl Pinhole {
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Result<Self, Error> {
        let camera = Self { fx, fy, cx, cy };
        camera.validate()?;
        Ok(camera)
    }
    fn validate(self) -> Result<(), Error> {
        if ![self.fx, self.fy, self.cx, self.cy]
            .iter()
            .all(|x| x.is_finite())
        {
            return Err(Error::NonFinite);
        }
        if self.fx <= 0.0 || self.fy <= 0.0 {
            return Err(Error::InvalidOptions);
        }
        Ok(())
    }
    /// Rejects points behind the camera or within 1e-8 of the projection plane.
    pub fn project(self, point: Vector3<f64>) -> Result<Vector2<f64>, Error> {
        self.validate()?;
        if !point.iter().all(|x| x.is_finite()) {
            return Err(Error::NonFinite);
        }
        if point.z <= 1e-8 {
            return Err(Error::Evaluation(
                "point behind camera or too close to projection plane".into(),
            ));
        }
        let pixel = Vector2::new(
            self.fx * point.x / point.z + self.cx,
            self.fy * point.y / point.z + self.cy,
        );
        if !pixel.iter().all(|x| x.is_finite()) {
            return Err(Error::NonFinite);
        }
        Ok(pixel)
    }
    /// Jet-valued reprojection with ambient SE3 `[tx,ty,tz,w,x,y,z]` and world point.
    /// Combine with graph AD to obtain right-tangent pose derivatives.
    pub fn reprojection_autodiff(
        self,
        pose: &[crate::autodiff::DynamicJet],
        point: &[crate::autodiff::DynamicJet],
        observation: Vector2<f64>,
    ) -> Result<Vec<crate::autodiff::DynamicJet>, Error> {
        self.validate()?;
        if pose.len() != 7 || point.len() != 3 {
            return Err(Error::InvalidDimensions);
        }
        if !observation.iter().all(|x| x.is_finite()) {
            return Err(Error::NonFinite);
        }
        for v in pose.iter().chain(point) {
            v.partials()?;
        }
        let p = camera_point_autodiff(pose, point)?;
        let out = vec![
            self.fx * &p[0] / &p[2] + self.cx - observation.x,
            self.fy * &p[1] / &p[2] + self.cy - observation.y,
        ];
        for v in &out {
            v.partials()?;
        }
        Ok(out)
    }
    /// Returns (pixel prediction - observation, pose tangent Jacobian, point Jacobian).
    pub fn reprojection(
        self,
        camera_to_world: SE3,
        world_point: Vector3<f64>,
        observation: Vector2<f64>,
    ) -> Result<Reprojection, Error> {
        if !observation.iter().all(|x| x.is_finite()) {
            return Err(Error::NonFinite);
        }
        let rotation = camera_to_world.rotation.0.inverse().to_rotation_matrix();
        let point = rotation * (world_point - camera_to_world.translation);
        let residual = self.project(point)? - observation;
        let projection = SMatrix::<f64, 2, 3>::new(
            self.fx / point.z,
            0.0,
            -self.fx * point.x / (point.z * point.z),
            0.0,
            self.fy / point.z,
            -self.fy * point.y / (point.z * point.z),
        );
        let skew = Matrix3::new(
            0.0, -point.z, point.y, point.z, 0.0, -point.x, -point.y, point.x, 0.0,
        );
        let mut local = SMatrix::<f64, 3, 6>::zeros();
        local
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(-Matrix3::identity()));
        local.fixed_view_mut::<3, 3>(0, 3).copy_from(&skew);
        let pose_jacobian = projection * local;
        let point_jacobian = projection * rotation.matrix();
        if !pose_jacobian
            .iter()
            .chain(point_jacobian.iter())
            .all(|x| x.is_finite())
        {
            return Err(Error::NonFinite);
        }
        Ok(Reprojection {
            residual,
            pose_jacobian,
            point_jacobian,
        })
    }
}
#[derive(Debug, Clone)]
pub struct Reprojection {
    pub residual: Vector2<f64>,
    pub pose_jacobian: SMatrix<f64, 2, 6>,
    pub point_jacobian: SMatrix<f64, 2, 3>,
}

/// Register an analytic reprojection factor connecting one SE3 camera-to-world
/// pose and one Euclidean(3) world landmark. No implicit numerical derivatives.
pub fn add_reprojection_factor(
    graph: &mut crate::factor::FactorGraph,
    pose: crate::factor::VariableId,
    point: crate::factor::VariableId,
    camera: Pinhole,
    observation: Vector2<f64>,
) -> Result<(), Error> {
    add_reprojection_factor_with_options(
        graph,
        pose,
        point,
        camera,
        observation,
        crate::factor::FactorOptions::default(),
    )
}
/// Analytic reprojection with measurement whitening and an optional joint pixel loss.
pub fn add_reprojection_factor_with_options(
    graph: &mut crate::factor::FactorGraph,
    pose: crate::factor::VariableId,
    point: crate::factor::VariableId,
    camera: Pinhole,
    observation: Vector2<f64>,
    options: crate::factor::FactorOptions,
) -> Result<(), Error> {
    use crate::factor::{VariableKind, read_se3};
    camera.validate()?;
    if !observation.iter().all(|x| x.is_finite()) {
        return Err(Error::NonFinite);
    }
    if !matches!(graph.variable_kind(pose)?, VariableKind::SE3)
        || !matches!(graph.variable_kind(point)?, VariableKind::Euclidean(3))
    {
        return Err(Error::InvalidDimensions);
    }
    graph.add_analytic_factor_with_options(&[pose, point], 2, options, move |values| {
        let projection = camera.reprojection(
            read_se3(&values[0])?,
            Vector3::from_column_slice(values[1].as_slice()),
            observation,
        )?;
        Ok((
            crate::DVector::from_column_slice(projection.residual.as_slice()),
            vec![
                crate::DMatrix::from_column_slice(2, 6, projection.pose_jacobian.as_slice()),
                crate::DMatrix::from_column_slice(2, 3, projection.point_jacobian.as_slice()),
            ],
        ))
    })
}

/// Register a reprojection factor whose local Jacobians are generated by forward AD.
pub fn add_autodiff_reprojection_factor(
    graph: &mut crate::factor::FactorGraph,
    pose: crate::factor::VariableId,
    point: crate::factor::VariableId,
    camera: Pinhole,
    observation: Vector2<f64>,
) -> Result<(), Error> {
    add_autodiff_reprojection_factor_with_options(
        graph,
        pose,
        point,
        camera,
        observation,
        crate::factor::FactorOptions::default(),
    )
}
/// AD reprojection with measurement whitening and an optional joint pixel loss.
pub fn add_autodiff_reprojection_factor_with_options(
    graph: &mut crate::factor::FactorGraph,
    pose: crate::factor::VariableId,
    point: crate::factor::VariableId,
    camera: Pinhole,
    observation: Vector2<f64>,
    options: crate::factor::FactorOptions,
) -> Result<(), Error> {
    use crate::factor::VariableKind;
    camera.validate()?;
    if !observation.iter().all(|x| x.is_finite()) {
        return Err(Error::NonFinite);
    }
    if !matches!(graph.variable_kind(pose)?, VariableKind::SE3)
        || !matches!(graph.variable_kind(point)?, VariableKind::Euclidean(3))
    {
        return Err(Error::InvalidDimensions);
    }
    graph.add_autodiff_factor_with_options(&[pose, point], 2, 32, options, move |b| {
        camera.reprojection_autodiff(b[0], b[1], observation)
    })
}

/// Transform a world point into the camera frame using a normalized unit quaternion.
fn camera_point_autodiff(
    pose: &[crate::autodiff::DynamicJet],
    point: &[crate::autodiff::DynamicJet],
) -> Result<[crate::autodiff::DynamicJet; 3], Error> {
    if pose.len() != 7 || point.len() != 3 {
        return Err(Error::InvalidDimensions);
    }
    for v in pose.iter().chain(point) {
        v.partials()?;
    }
    use crate::autodiff::DynamicJet as J;
    let norm = pose[3..]
        .iter()
        .fold(J::constant(0.0), |a, b| a + b * b)
        .sqrt();
    if (norm.value() - 1.0).abs() > 1e-8 {
        return Err(Error::Evaluation("quaternion must have unit norm".into()));
    }
    let q = [
        &pose[3] / &norm,
        -&pose[4] / &norm,
        -&pose[5] / &norm,
        -&pose[6] / &norm,
    ];
    let v: [J; 3] = std::array::from_fn(|i| &point[i] - &pose[i]);
    let p = crate::manifold_autodiff::rotate(&q, &v);
    if p[2].value() <= 1e-8 {
        return Err(Error::Evaluation(
            "point behind camera or too close to projection plane".into(),
        ));
    }
    for v in &p {
        v.partials()?;
    }
    Ok(p)
}

mod calibration;
pub use calibration::{
    BrownConrady, add_calibration_reprojection_factor, brown_conrady_reprojection_autodiff,
};
