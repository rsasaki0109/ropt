//! Four-coefficient Brown–Conrady camera: two radial and two tangential terms.
use super::{Pinhole, camera_point_autodiff};
use crate::autodiff::DynamicJet as J;
use crate::factor::{FactorGraph, FactorOptions, VariableId, VariableKind};
use crate::manifold::SE3;
use crate::{DVector, Error};
use nalgebra::{Vector2, Vector3};

/// Forward distortion in normalized image coordinates; no skew or inverse undistortion.
/// Coefficient order `[k1,k2,p1,p2]`, with p1 contributing y-directed decentering.
#[derive(Debug, Clone, Copy)]
pub struct BrownConrady {
    pub pinhole: Pinhole,
    pub distortion: [f64; 4],
}
impl BrownConrady {
    pub fn new(pinhole: Pinhole, distortion: [f64; 4]) -> Result<Self, Error> {
        let model = Self {
            pinhole,
            distortion,
        };
        model.validate()?;
        Ok(model)
    }
    fn validate(self) -> Result<(), Error> {
        self.pinhole.validate()?;
        if !self.distortion.iter().all(|x| x.is_finite()) {
            return Err(Error::NonFinite);
        }
        Ok(())
    }
    /// Euclidean optimization storage `[ln(fx),ln(fy),cx,cy,k1,k2,p1,p2]`.
    /// Finite representable exponentials keep focal lengths positive.
    pub fn parameters(self) -> Result<DVector<f64>, Error> {
        self.validate()?;
        let p = self.pinhole;
        Ok(DVector::from_vec(vec![
            p.fx.ln(),
            p.fy.ln(),
            p.cx,
            p.cy,
            self.distortion[0],
            self.distortion[1],
            self.distortion[2],
            self.distortion[3],
        ]))
    }
    pub fn from_parameters(parameters: &[f64]) -> Result<Self, Error> {
        if parameters.len() != 8 {
            return Err(Error::InvalidDimensions);
        }
        if !parameters.iter().all(|x| x.is_finite()) {
            return Err(Error::NonFinite);
        }
        Self::new(
            Pinhole::new(
                parameters[0].exp(),
                parameters[1].exp(),
                parameters[2],
                parameters[3],
            )?,
            [parameters[4], parameters[5], parameters[6], parameters[7]],
        )
    }
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
        let x = point.x / point.z;
        let y = point.y / point.z;
        let r2 = x * x + y * y;
        let [k1, k2, p1, p2] = self.distortion;
        let radial = 1.0 + k1 * r2 + k2 * r2 * r2;
        let xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        let yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        let pixel = Vector2::new(
            self.pinhole.fx * xd + self.pinhole.cx,
            self.pinhole.fy * yd + self.pinhole.cy,
        );
        if !pixel.iter().all(|x| x.is_finite()) {
            return Err(Error::NonFinite);
        }
        Ok(pixel)
    }
    pub fn reprojection(
        self,
        pose: SE3,
        point: Vector3<f64>,
        observed: Vector2<f64>,
    ) -> Result<Vector2<f64>, Error> {
        if !observed.iter().all(|x| x.is_finite()) {
            return Err(Error::NonFinite);
        }
        let r = self.project(pose.rotation.0.inverse() * (point - pose.translation))? - observed;
        if !r.iter().all(|x| x.is_finite()) {
            return Err(Error::NonFinite);
        }
        Ok(r)
    }
    /// Fixed-calibration AD path for custom factors; calibration jets are constants.
    pub fn reprojection_autodiff(
        self,
        pose: &[J],
        point: &[J],
        observed: Vector2<f64>,
    ) -> Result<Vec<J>, Error> {
        let c = self
            .parameters()?
            .iter()
            .copied()
            .map(J::constant)
            .collect::<Vec<_>>();
        brown_conrady_reprojection_autodiff(&c, pose, point, observed)
    }
}
/// Parameterized forward AD reprojection. Calibration storage uses log focal lengths.
/// Poses transform camera to world and store `[tx,ty,tz,w,x,y,z]`.
pub fn brown_conrady_reprojection_autodiff(
    calibration: &[J],
    pose: &[J],
    point: &[J],
    observed: Vector2<f64>,
) -> Result<Vec<J>, Error> {
    if calibration.len() != 8 {
        return Err(Error::InvalidDimensions);
    }
    if !observed.iter().all(|x| x.is_finite()) {
        return Err(Error::NonFinite);
    }
    for v in calibration {
        v.partials()?;
    }
    let fx = calibration[0].exp();
    let fy = calibration[1].exp();
    fx.partials()?;
    fy.partials()?;
    if fx.value() <= 0.0 || fy.value() <= 0.0 {
        return Err(Error::InvalidOptions);
    }
    let p = camera_point_autodiff(pose, point)?;
    let x = &p[0] / &p[2];
    let y = &p[1] / &p[2];
    let r2 = &x * &x + &y * &y;
    let radial = 1.0 + &calibration[4] * &r2 + &calibration[5] * &r2 * &r2;
    let xd =
        &x * &radial + 2.0 * &calibration[6] * &x * &y + &calibration[7] * (&r2 + 2.0 * &x * &x);
    let yd = &y * radial + &calibration[6] * (&r2 + 2.0 * &y * &y) + 2.0 * &calibration[7] * x * y;
    let r = vec![
        fx * xd + &calibration[2] - observed.x,
        fy * yd + &calibration[3] - observed.y,
    ];
    for v in &r {
        v.partials()?;
    }
    Ok(r)
}
/// Register a factor on `(SE3 pose, Euclidean(3) point, Euclidean(8) calibration)`.
/// Shared calibration belongs in the retained prefix for landmark block Schur.
/// Freeze the calibration ID to use a fixed distorted camera in the same graph.
pub fn add_calibration_reprojection_factor(
    graph: &mut FactorGraph,
    pose: VariableId,
    point: VariableId,
    calibration: VariableId,
    observed: Vector2<f64>,
    options: FactorOptions,
) -> Result<(), Error> {
    if !matches!(graph.variable_kind(pose)?, VariableKind::SE3)
        || !matches!(graph.variable_kind(point)?, VariableKind::Euclidean(3))
        || !matches!(
            graph.variable_kind(calibration)?,
            VariableKind::Euclidean(8)
        )
    {
        return Err(Error::InvalidDimensions);
    }
    if !observed.iter().all(|x| x.is_finite()) {
        return Err(Error::NonFinite);
    }
    graph.add_autodiff_factor_with_options(&[pose, point, calibration], 2, 17, options, move |b| {
        brown_conrady_reprojection_autodiff(b[2], b[0], b[1], observed)
    })
}
