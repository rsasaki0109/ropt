//! Cartesian absolute position constraints with body-frame antenna lever arms.
//! GNSS fixes must already be converted to a common local metric world frame.
use crate::autodiff::DynamicJet as J;
use crate::factor::{FactorGraph, FactorOptions, VariableId, VariableKind, read_se3};
use crate::manifold::SE3;
use crate::{DMatrix, DVector, Error};
use nalgebra::{Matrix3, SMatrix, Vector3};
/// Prediction is t_world_body + R_world_body * lever_body.
#[derive(Debug, Clone, Copy)]
pub struct Position3 {
    observation: Vector3<f64>,
    lever_arm: Vector3<f64>,
}
#[derive(Debug, Clone)]
pub struct PositionResidual {
    pub residual: Vector3<f64>,
    pub pose_jacobian: SMatrix<f64, 3, 6>,
}
impl Position3 {
    pub fn new(observation: Vector3<f64>, lever_arm: Vector3<f64>) -> Result<Self, Error> {
        if !observation
            .iter()
            .chain(lever_arm.iter())
            .all(|v| v.is_finite())
        {
            return Err(Error::NonFinite);
        }
        Ok(Self {
            observation,
            lever_arm,
        })
    }
    /// World-frame residual and right/body tangent Jacobian `[R, -R*[lever]_x]`.
    pub fn evaluate(self, pose: SE3) -> Result<PositionResidual, Error> {
        VariableKind::SE3.validate(&crate::factor::se3_values(pose))?;
        let rotation = pose.rotation.0.to_rotation_matrix();
        let residual = pose.translation + rotation * self.lever_arm - self.observation;
        let l = self.lever_arm;
        let skew = Matrix3::new(0.0, -l.z, l.y, l.z, 0.0, -l.x, -l.y, l.x, 0.0);
        let mut j = SMatrix::<f64, 3, 6>::zeros();
        j.fixed_view_mut::<3, 3>(0, 0).copy_from(rotation.matrix());
        j.fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(-rotation.matrix() * skew));
        if !residual.iter().chain(j.iter()).all(|v| v.is_finite()) {
            return Err(Error::NonFinite);
        }
        Ok(PositionResidual {
            residual,
            pose_jacobian: j,
        })
    }
    /// Jet-valued ambient SE3 block for custom AD factors.
    pub fn residual_autodiff(self, pose: &[J]) -> Result<Vec<J>, Error> {
        use crate::manifold_autodiff::{normalize_rotation, rotate};
        if pose.len() != 7 {
            return Err(Error::InvalidDimensions);
        }
        for v in pose {
            v.partials()?;
        }
        let q = normalize_rotation(&pose[3..])?;
        let rotated = rotate(
            &q,
            &self
                .lever_arm
                .as_slice()
                .iter()
                .copied()
                .map(J::constant)
                .collect::<Vec<_>>(),
        );
        let r = (0..3)
            .map(|i| &pose[i] + &rotated[i] - self.observation[i])
            .collect::<Vec<_>>();
        for v in &r {
            v.partials()?;
        }
        Ok(r)
    }
}
/// Analytic absolute position factor with world-coordinate covariance whitening.
/// Use a joint loss for a GNSS fix to downweight its full whitened three-vector.
pub fn add_position3_factor(
    graph: &mut FactorGraph,
    pose: VariableId,
    observation: Vector3<f64>,
    lever_arm: Vector3<f64>,
    options: FactorOptions,
) -> Result<(), Error> {
    if !matches!(graph.variable_kind(pose)?, VariableKind::SE3) {
        return Err(Error::InvalidDimensions);
    }
    let cost = Position3::new(observation, lever_arm)?;
    graph.add_analytic_factor_with_options(&[pose], 3, options, move |values| {
        let result = cost.evaluate(read_se3(&values[0])?)?;
        Ok((
            DVector::from_column_slice(result.residual.as_slice()),
            vec![DMatrix::from_column_slice(
                3,
                6,
                result.pose_jacobian.as_slice(),
            )],
        ))
    })
}
