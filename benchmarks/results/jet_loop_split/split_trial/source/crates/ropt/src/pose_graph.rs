//! SE2 relative measurements in g2o coordinates and SE3 relative Lie Log residuals.
use crate::Error;
use crate::autodiff::DynamicJet as J;
use crate::factor::{FactorGraph, FactorOptions, VariableId, VariableKind};
use crate::manifold::SE2;
use nalgebra::{Matrix3, Vector3};
pub type Pose2Linearization = (Vector3<f64>, Matrix3<f64>, Matrix3<f64>);

/// Residual is the translation and principal angle of Z^-1 * (A^-1 * B).
/// Information matrices must be expressed in these measurement-frame coordinates.
#[derive(Debug, Clone, Copy)]
pub struct RelativePose2 {
    measurement: SE2,
    measurement_angle: f64,
    measurement_cosine: f64,
    measurement_sine: f64,
}
impl RelativePose2 {
    pub fn new(measurement: SE2) -> Result<Self, Error> {
        let angle = measurement.rotation.log();
        if !measurement.translation.iter().all(|x| x.is_finite()) || !angle.is_finite() {
            return Err(Error::NonFinite);
        }
        Ok(Self {
            measurement,
            measurement_angle: angle,
            measurement_cosine: angle.cos(),
            measurement_sine: angle.sin(),
        })
    }
    pub fn residual(self, a: SE2, b: SE2) -> Result<Vector3<f64>, Error> {
        let delta = self.measurement.inverse().compose(a.inverse().compose(b));
        let r = Vector3::new(
            delta.translation.x,
            delta.translation.y,
            delta.rotation.log(),
        );
        if !r.iter().all(|x| x.is_finite()) {
            return Err(Error::NonFinite);
        }
        Ok(r)
    }
    /// Exact analytic right-tangent Jacobians in g2o measurement coordinates.
    /// The principal angle is differentiable away from its pi branch cut.
    pub fn linearize(self, a: SE2, b: SE2) -> Result<Pose2Linearization, Error> {
        self.linearize_parameters(
            a.translation,
            b.translation,
            a.rotation.log(),
            b.rotation.log(),
        )
    }
    /// Analytic right-tangent Jacobians directly from ambient [x, y, angle].
    /// Principal input angles avoid a sin/cos -> group -> atan2 round trip.
    /// Other finite angles use the group chart's range reduction before subtraction.
    pub fn linearize_ambient(self, a: &[f64], b: &[f64]) -> Result<Pose2Linearization, Error> {
        if a.len() != 3 || b.len() != 3 {
            return Err(Error::InvalidDimensions);
        }
        if !a.iter().chain(b).all(|value| value.is_finite()) {
            return Err(Error::NonFinite);
        }
        let principal = |angle: f64| {
            if (-std::f64::consts::PI..=std::f64::consts::PI).contains(&angle) {
                angle
            } else {
                crate::manifold::SO2::exp(angle).log()
            }
        };
        self.linearize_parameters(
            nalgebra::Vector2::new(a[0], a[1]),
            nalgebra::Vector2::new(b[0], b[1]),
            principal(a[2]),
            principal(b[2]),
        )
    }
    fn linearize_parameters(
        self,
        a: nalgebra::Vector2<f64>,
        b: nalgebra::Vector2<f64>,
        ta: f64,
        tb: f64,
    ) -> Result<Pose2Linearization, Error> {
        let z = self.measurement_angle;
        let (ca, sa, cz, sz) = (
            ta.cos(),
            ta.sin(),
            self.measurement_cosine,
            self.measurement_sine,
        );
        let difference = b - a;
        let qx = ca * difference.x + sa * difference.y;
        let qy = -sa * difference.x + ca * difference.y;
        let x = qx - self.measurement.translation.x;
        let y = qy - self.measurement.translation.y;
        let angle = tb - ta - z;
        let r = Vector3::new(
            cz * x + sz * y,
            -sz * x + cz * y,
            angle.sin().atan2(angle.cos()),
        );
        let ja = Matrix3::new(
            -cz,
            -sz,
            cz * qy - sz * qx,
            sz,
            -cz,
            -sz * qy - cz * qx,
            0.,
            0.,
            -1.,
        );
        let theta = tb - ta - z;
        let jb = Matrix3::new(
            theta.cos(),
            -theta.sin(),
            0.,
            theta.sin(),
            theta.cos(),
            0.,
            0.,
            0.,
            1.,
        );
        if !r
            .iter()
            .chain(ja.iter())
            .chain(jb.iter())
            .all(|v| v.is_finite())
        {
            return Err(Error::NonFinite);
        }
        Ok((r, ja, jb))
    }
    /// Ambient SE2 inputs `[x,y,angle]`; graph AD differentiates right retractions.
    pub fn residual_autodiff(self, a: &[J], b: &[J]) -> Result<Vec<J>, Error> {
        if a.len() != 3 || b.len() != 3 {
            return Err(Error::InvalidDimensions);
        }
        for x in a.iter().chain(b) {
            x.partials()?;
        }
        let dx = &b[0] - &a[0];
        let dy = &b[1] - &a[1];
        let c = a[2].cos();
        let s = a[2].sin();
        let x = &c * &dx + &s * &dy - self.measurement.translation.x;
        let y = -&s * dx + c * dy - self.measurement.translation.y;
        let z = self.measurement_angle;
        let angle = &b[2] - &a[2] - z;
        let r = vec![
            self.measurement_cosine * &x + self.measurement_sine * &y,
            -self.measurement_sine * x + self.measurement_cosine * y,
            angle.sin().atan2(&angle.cos()),
        ];
        for x in &r {
            x.partials()?;
        }
        Ok(r)
    }
}
/// Add an explicitly AD relative pose factor with local tangent Jacobians.
pub fn add_relative_pose2_factor(
    graph: &mut FactorGraph,
    a: VariableId,
    b: VariableId,
    measurement: SE2,
    options: FactorOptions,
) -> Result<(), Error> {
    if !matches!(
        graph.variable_kind(a)?,
        VariableKind::SE2 | VariableKind::SE2Cartesian
    ) || !matches!(
        graph.variable_kind(b)?,
        VariableKind::SE2 | VariableKind::SE2Cartesian
    ) {
        return Err(Error::InvalidDimensions);
    }
    let cost = RelativePose2::new(measurement)?;
    graph.add_autodiff_factor_with_options(&[a, b], 3, 6, options, move |values| {
        cost.residual_autodiff(values[0], values[1])
    })
}

/// Explicit analytic alternative to the AD SE2 factor; identical residual coordinates.
pub fn add_relative_pose2_factor_analytic(
    graph: &mut FactorGraph,
    a: VariableId,
    b: VariableId,
    measurement: SE2,
    options: FactorOptions,
) -> Result<(), Error> {
    if !matches!(
        graph.variable_kind(a)?,
        VariableKind::SE2 | VariableKind::SE2Cartesian
    ) || !matches!(
        graph.variable_kind(b)?,
        VariableKind::SE2 | VariableKind::SE2Cartesian
    ) {
        return Err(Error::InvalidDimensions);
    }
    let cost = RelativePose2::new(measurement)?;
    graph.add_analytic_factor_with_options(&[a, b], 3, options, move |v| {
        let (r, ja, jb) = cost.linearize(
            crate::factor::read_se2(&v[0])?,
            crate::factor::read_se2(&v[1])?,
        )?;
        Ok((
            crate::DVector::from_column_slice(r.as_slice()),
            vec![
                crate::DMatrix::from_column_slice(3, 3, ja.as_slice()),
                crate::DMatrix::from_column_slice(3, 3, jb.as_slice()),
            ],
        ))
    })
}

/// SE3 relative residual Log(Z^-1 * (A^-1 * B)), translation then rotation.
/// Noise must be expressed in this six-dimensional measurement Lie tangent.
#[derive(Debug, Clone, Copy)]
pub struct RelativePose3 {
    measurement: crate::manifold::SE3,
}
impl RelativePose3 {
    pub fn new(measurement: crate::manifold::SE3) -> Result<Self, Error> {
        VariableKind::SE3.validate(&crate::factor::se3_values(measurement))?;
        Ok(Self { measurement })
    }
    pub fn residual(
        self,
        a: crate::manifold::SE3,
        b: crate::manifold::SE3,
    ) -> Result<crate::manifold::Tangent6, Error> {
        VariableKind::SE3.validate(&crate::factor::se3_values(a))?;
        VariableKind::SE3.validate(&crate::factor::se3_values(b))?;
        let r = self.measurement.local(a.inverse().compose(b));
        if !r.iter().all(|v| v.is_finite()) {
            return Err(Error::NonFinite);
        }
        Ok(r)
    }
    /// Ambient pose blocks; registration differentiates twelve right-tangent columns.
    pub fn residual_autodiff(self, a: &[J], b: &[J]) -> Result<Vec<J>, Error> {
        use crate::manifold_autodiff::{
            compose_rotation, conjugate, normalize_rotation, pose_log, rotate,
        };
        if a.len() != 7 || b.len() != 7 {
            return Err(Error::InvalidDimensions);
        }
        for v in a.iter().chain(b) {
            v.partials()?;
        }
        let qa = conjugate(&normalize_rotation(&a[3..])?);
        let qb = normalize_rotation(&b[3..])?;
        let m = self.measurement.rotation.0.quaternion();
        let qz = [m.w, -m.i, -m.j, -m.k].map(J::constant);
        let difference: [J; 3] = std::array::from_fn(|i| &b[i] - &a[i]);
        let relative_translation = rotate(&qa, &difference);
        let translated: [J; 3] =
            std::array::from_fn(|i| &relative_translation[i] - self.measurement.translation[i]);
        let delta_t = rotate(&qz, &translated);
        let delta_q = compose_rotation(&qz, &compose_rotation(&qa, &qb));
        pose_log(&delta_t, &delta_q)
    }
}
/// Register a forward-AD SE3 relative Lie Log factor; no numerical differentiation.
pub fn add_relative_pose3_factor(
    graph: &mut FactorGraph,
    a: VariableId,
    b: VariableId,
    measurement: crate::manifold::SE3,
    options: FactorOptions,
) -> Result<(), Error> {
    if !matches!(graph.variable_kind(a)?, VariableKind::SE3)
        || !matches!(graph.variable_kind(b)?, VariableKind::SE3)
    {
        return Err(Error::InvalidDimensions);
    }
    let cost = RelativePose3::new(measurement)?;
    graph.add_autodiff_factor_with_options(&[a, b], 6, 12, options, move |values| {
        cost.residual_autodiff(values[0], values[1])
    })
}
