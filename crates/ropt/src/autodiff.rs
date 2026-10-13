//! Fixed-dimensional forward-mode jets. Dimension is checked by the type system.
//! For runtime sizes, see DynamicJet and DynamicAutoDiff in this module.
pub use crate::dynamic_autodiff::{DynamicAutoDiff, DynamicJet};
use crate::{DMatrix, DVector, Error, Residual};
use std::ops::{Add, Div, Mul, Neg, Sub};
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Jet<const N: usize> {
    pub value: f64,
    pub derivative: [f64; N],
}
impl<const N: usize> Jet<N> {
    pub fn constant(value: f64) -> Self {
        Self {
            value,
            derivative: [0.0; N],
        }
    }
    pub fn variable(value: f64, index: usize) -> Self {
        let mut a = Self::constant(value);
        a.derivative[index] = 1.0;
        a
    }
    fn chain(self, value: f64, slope: f64) -> Self {
        Self {
            value,
            derivative: self.derivative.map(|d| d * slope),
        }
    }
    pub fn sin(self) -> Self {
        self.chain(self.value.sin(), self.value.cos())
    }
    pub fn cos(self) -> Self {
        self.chain(self.value.cos(), -self.value.sin())
    }
    pub fn tan(self) -> Self {
        let value = self.value.tan();
        self.chain(value, 1.0 + value * value)
    }
    pub fn atan(self) -> Self {
        self.chain(self.value.atan(), atan_slope(self.value))
    }
    pub fn powi(self, n: i32) -> Self {
        let (value, slope) = power_value_slope(self.value, n);
        self.chain(value, slope)
    }
    pub fn atan2(self, x: Self) -> Self {
        let (dy, dx) = atan2_slopes(self.value, x.value);
        Self {
            value: self.value.atan2(x.value),
            derivative: std::array::from_fn(|i| dy * self.derivative[i] + dx * x.derivative[i]),
        }
    }
    pub fn exp(self) -> Self {
        self.chain(self.value.exp(), self.value.exp())
    }
    pub fn ln(self) -> Self {
        self.chain(self.value.ln(), 1.0 / self.value)
    }
    pub fn sqrt(self) -> Self {
        self.chain(self.value.sqrt(), 0.5 / self.value.sqrt())
    }
}
impl<const N: usize> Add for Jet<N> {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            value: self.value + b.value,
            derivative: std::array::from_fn(|i| self.derivative[i] + b.derivative[i]),
        }
    }
}
impl<const N: usize> Sub for Jet<N> {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self {
            value: self.value - b.value,
            derivative: std::array::from_fn(|i| self.derivative[i] - b.derivative[i]),
        }
    }
}
#[allow(clippy::suspicious_arithmetic_impl)]
impl<const N: usize> Mul for Jet<N> {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self {
            value: self.value * b.value,
            derivative: std::array::from_fn(|i| {
                self.derivative[i] * b.value + self.value * b.derivative[i]
            }),
        }
    }
}
#[allow(clippy::suspicious_arithmetic_impl)]
impl<const N: usize> Div for Jet<N> {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        Self {
            value: self.value / b.value,
            derivative: std::array::from_fn(|i| {
                self.derivative[i] / b.value - (self.value / b.value) * (b.derivative[i] / b.value)
            }),
        }
    }
}
impl<const N: usize> Neg for Jet<N> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            derivative: self.derivative.map(|d| -d),
        }
    }
}
/// Residual closure receives seeded jets and returns M scalar residual jets.
pub struct AutoDiff<const N: usize, const M: usize, F>(pub F);
impl<const N: usize, const M: usize, F: Fn([Jet<N>; N]) -> [Jet<N>; M]> Residual
    for AutoDiff<N, M, F>
{
    fn dimensions(&self) -> (usize, usize) {
        (N, M)
    }
    fn evaluate(
        &self,
        x: &DVector<f64>,
        r: &mut DVector<f64>,
        j: &mut DMatrix<f64>,
    ) -> Result<(), Error> {
        if x.len() != N || r.len() != M || j.shape() != (M, N) {
            return Err(Error::InvalidDimensions);
        }
        if !x.iter().all(|v| v.is_finite()) {
            return Err(Error::NonFinite);
        }
        let result = (self.0)(std::array::from_fn(|i| Jet::variable(x[i], i)));
        if result
            .iter()
            .any(|jet| !jet.value.is_finite() || !jet.derivative.iter().all(|d| d.is_finite()))
        {
            return Err(Error::NonFinite);
        }

        for row in 0..M {
            r[row] = result[row].value;
            for col in 0..N {
                j[(row, col)] = result[row].derivative[col];
            }
        }
        Ok(())
    }
}

// Common scalar formulas, shared by fixed and dynamic AD.
pub(crate) fn power_value_slope(value: f64, n: i32) -> (f64, f64) {
    let result = value.powi(n);
    let slope = if n == 0 {
        0.0
    } else if n == i32::MIN {
        (n as f64) * result / value
    } else {
        (n as f64) * value.powi(n - 1)
    };
    (result, slope)
}
pub(crate) fn atan2_slopes(y: f64, x: f64) -> (f64, f64) {
    let scale = x.abs().max(y.abs());
    let xn = x / scale;
    let yn = y / scale;
    let denominator = xn * xn + yn * yn;
    ((xn / denominator) / scale, (-yn / denominator) / scale)
}

pub(crate) fn atan_slope(value: f64) -> f64 {
    if value.abs() <= 1.0 {
        1.0 / (1.0 + value * value)
    } else {
        let inv = value.recip();
        let squared = inv * inv;
        squared / (1.0 + squared)
    }
}
