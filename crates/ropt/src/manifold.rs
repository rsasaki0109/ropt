//! Right retractions: retract(x,d) = x * Exp(d); local(x,y) = Log(x^-1 * y).
//! SE(3) tangent order is [translation, rotation]. Factor graphs use these
//! retractions to optimize in tangent coordinates.
use nalgebra::{Matrix3, SVector, UnitComplex, UnitQuaternion, Vector2, Vector3};
pub type Tangent6 = SVector<f64, 6>;
#[derive(Debug, Clone, Copy)]
pub struct SO2(pub UnitComplex<f64>);
impl SO2 {
    pub fn exp(angle: f64) -> Self {
        Self(UnitComplex::new(angle))
    }
    pub fn log(self) -> f64 {
        self.0.angle()
    }
    pub fn compose(self, b: Self) -> Self {
        Self(self.0 * b.0)
    }
    pub fn inverse(self) -> Self {
        Self(self.0.inverse())
    }
    pub fn retract(self, d: f64) -> Self {
        self.compose(Self::exp(d))
    }
    pub fn local(self, b: Self) -> f64 {
        self.inverse().compose(b).log()
    }
}
#[derive(Debug, Clone, Copy)]
pub struct SO3(pub UnitQuaternion<f64>);
impl SO3 {
    pub fn exp(w: Vector3<f64>) -> Self {
        Self(UnitQuaternion::from_scaled_axis(w))
    }
    pub fn log(self) -> Vector3<f64> {
        self.0.scaled_axis()
    }
    pub fn compose(self, b: Self) -> Self {
        Self(self.0 * b.0)
    }
    pub fn inverse(self) -> Self {
        Self(self.0.inverse())
    }
    pub fn retract(self, d: Vector3<f64>) -> Self {
        self.compose(Self::exp(d))
    }
    pub fn local(self, b: Self) -> Vector3<f64> {
        self.inverse().compose(b).log()
    }
}
#[derive(Debug, Clone, Copy)]
pub struct SE2 {
    pub rotation: SO2,
    pub translation: Vector2<f64>,
}
impl SE2 {
    /// Right/body Cartesian chart: compose a translation and an SO2 increment.
    /// Distinct from the group exponential; its differential at zero is identity.
    pub fn retract_cartesian(self, d: Vector3<f64>) -> Self {
        self.compose(Self {
            translation: Vector2::new(d[0], d[1]),
            rotation: SO2::exp(d[2]),
        })
    }
    /// Inverse Cartesian chart, with principal relative rotation.
    pub fn local_cartesian(self, b: Self) -> Vector3<f64> {
        let translation = self.rotation.inverse().0 * (b.translation - self.translation);
        Vector3::new(
            translation[0],
            translation[1],
            self.rotation.local(b.rotation),
        )
    }
    pub fn exp(v: Vector3<f64>) -> Self {
        let w = v[2];
        let (a, b) = if w.abs() < 1e-6 {
            (1.0 - w * w / 6.0, w / 2.0 - w.powi(3) / 24.0)
        } else {
            // Half-angle identity avoids cancellation in 1-cos(w).
            (w.sin() / w, 2.0 * (w / 2.0).sin().powi(2) / w)
        };
        Self {
            rotation: SO2::exp(w),
            translation: Vector2::new(a * v[0] - b * v[1], b * v[0] + a * v[1]),
        }
    }
    pub fn log(self) -> Vector3<f64> {
        let w = self.rotation.log();
        let a = if w.abs() < 1e-6 {
            1.0 - w * w / 12.0
        } else {
            (w / 2.0) / (w / 2.0).tan()
        };
        Vector3::new(
            a * self.translation[0] + w * 0.5 * self.translation[1],
            -w * 0.5 * self.translation[0] + a * self.translation[1],
            w,
        )
    }
    pub fn compose(self, b: Self) -> Self {
        Self {
            rotation: self.rotation.compose(b.rotation),
            translation: self.translation + self.rotation.0 * b.translation,
        }
    }
    pub fn inverse(self) -> Self {
        let rotation = self.rotation.inverse();
        Self {
            rotation,
            translation: rotation.0 * (-self.translation),
        }
    }
    pub fn retract(self, d: Vector3<f64>) -> Self {
        self.compose(Self::exp(d))
    }
    pub fn local(self, b: Self) -> Vector3<f64> {
        self.inverse().compose(b).log()
    }
}
fn skew(w: Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -w[2], w[1], w[2], 0.0, -w[0], -w[1], w[0], 0.0)
}
#[derive(Debug, Clone, Copy)]
pub struct SE3 {
    pub rotation: SO3,
    pub translation: Vector3<f64>,
}
impl SE3 {
    pub fn exp(v: Tangent6) -> Self {
        let t = v.fixed_rows::<3>(0).into_owned();
        let w = v.fixed_rows::<3>(3).into_owned();
        let theta = w.norm();
        let k = skew(w);
        let (a, b) = if theta < 1e-5 {
            (
                0.5 - theta * theta / 24.0,
                1.0 / 6.0 - theta * theta / 120.0,
            )
        } else {
            (
                (1.0 - theta.cos()) / theta.powi(2),
                (theta - theta.sin()) / theta.powi(3),
            )
        };
        Self {
            rotation: SO3::exp(w),
            translation: (Matrix3::identity() + a * k + b * k * k) * t,
        }
    }
    pub fn log(self) -> Tangent6 {
        let w = self.rotation.log();
        let theta = w.norm();
        let k = skew(w);
        let a = if theta < 1e-5 {
            1.0 / 12.0 + theta * theta / 720.0
        } else {
            (1.0 - 0.5 * theta / (theta / 2.0).tan()) / (theta * theta)
        };
        let t = (Matrix3::identity() - 0.5 * k + a * k * k) * self.translation;
        Tangent6::new(t[0], t[1], t[2], w[0], w[1], w[2])
    }
    pub fn compose(self, b: Self) -> Self {
        Self {
            rotation: self.rotation.compose(b.rotation),
            translation: self.translation + self.rotation.0 * b.translation,
        }
    }
    pub fn inverse(self) -> Self {
        let rotation = self.rotation.inverse();
        Self {
            rotation,
            translation: rotation.0 * (-self.translation),
        }
    }
    pub fn retract(self, d: Tangent6) -> Self {
        self.compose(Self::exp(d))
    }
    pub fn local(self, b: Self) -> Tangent6 {
        self.inverse().compose(b).log()
    }
}
#[derive(Debug, Clone, Copy)]
pub struct Euclidean<const N: usize>(pub SVector<f64, N>);
impl<const N: usize> Euclidean<N> {
    pub fn retract(self, d: SVector<f64, N>) -> Self {
        Self(self.0 + d)
    }
    pub fn local(self, b: Self) -> SVector<f64, N> {
        b.0 - self.0
    }
}
