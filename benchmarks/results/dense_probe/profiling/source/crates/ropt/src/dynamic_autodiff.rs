//! Runtime-sized forward-mode AD with bounded derivative chunks.
//! Zero-dimensional constants broadcast into active derivative spaces. Arithmetic
//! errors are carried to the adapter rather than panicking or truncating vectors.
use crate::{DMatrix, DVector, Error, Residual};
use std::ops::{Add, Div, Mul, Neg, Sub};
const INDEXED_WIDTH_LIMIT: usize = 16;
#[derive(Debug, Clone, Copy, PartialEq)]
enum Failure {
    Dimension,
    NonFinite,
}
#[derive(Clone, Copy)]
enum BinaryOperation {
    Add,
    Sub,
    Mul,
    Div,
}
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicJet {
    value: f64,
    derivative: Vec<f64>,
    failure: Option<Failure>,
}
impl DynamicJet {
    pub fn constant(value: f64) -> Self {
        Self::from_parts(value, Vec::new())
    }
    pub fn variable(value: f64, index: usize, dimension: usize) -> Result<Self, Error> {
        if dimension == 0 || index >= dimension {
            return Err(Error::InvalidDimensions);
        }
        if !value.is_finite() {
            return Err(Error::NonFinite);
        }
        let mut derivative = vec![0.0; dimension];
        derivative[index] = 1.0;
        Ok(Self::from_parts(value, derivative))
    }
    pub fn value(&self) -> f64 {
        self.value
    }
    /// Empty partials mean a constant in any derivative space.
    pub fn partials(&self) -> Result<&[f64], Error> {
        match self.failure {
            Some(Failure::Dimension) => Err(Error::InvalidDimensions),
            Some(Failure::NonFinite) => Err(Error::NonFinite),
            None => Ok(&self.derivative),
        }
    }
    fn from_parts(value: f64, derivative: Vec<f64>) -> Self {
        let failure = if !value.is_finite() || !derivative.iter().all(|d| d.is_finite()) {
            Some(Failure::NonFinite)
        } else {
            None
        };
        Self {
            value,
            derivative,
            failure,
        }
    }
    fn invalid(failure: Failure) -> Self {
        Self {
            value: f64::NAN,
            derivative: Vec::new(),
            failure: Some(failure),
        }
    }
    fn unary(&self, value: f64, slope: f64) -> Self {
        if let Some(failure) = self.failure {
            return Self::invalid(failure);
        }
        Self::from_parts(value, self.derivative.iter().map(|d| d * slope).collect())
    }
    fn binary(&self, b: &Self, value: f64, left: f64, right: f64) -> Self {
        if let Some(failure) = self.failure.or(b.failure) {
            return Self::invalid(failure);
        }
        let a_len = self.derivative.len();
        let b_len = b.derivative.len();
        if a_len != 0 && b_len != 0 && a_len != b_len {
            return Self::invalid(Failure::Dimension);
        }
        let dimension = a_len.max(b_len);
        let derivative = (0..dimension)
            .map(|i| {
                self.derivative.get(i).copied().unwrap_or(0.0) * left
                    + b.derivative.get(i).copied().unwrap_or(0.0) * right
            })
            .collect();
        Self::from_parts(value, derivative)
    }
    fn binary_terms(&self, b: &Self, operation: BinaryOperation) -> (f64, f64, f64) {
        match operation {
            BinaryOperation::Add => (self.value + b.value, 1., 1.),
            BinaryOperation::Sub => (self.value - b.value, 1., -1.),
            BinaryOperation::Mul => (self.value * b.value, b.value, self.value),
            BinaryOperation::Div => (
                self.value / b.value,
                1. / b.value,
                -(self.value / b.value) / b.value,
            ),
        }
    }
    fn binary_width(&self, b: &Self) -> Result<usize, Failure> {
        if let Some(failure) = self.failure.or(b.failure) {
            return Err(failure);
        }
        let a_len = self.derivative.len();
        let b_len = b.derivative.len();
        if a_len != 0 && b_len != 0 && a_len != b_len {
            return Err(Failure::Dimension);
        }
        Ok(a_len.max(b_len))
    }
    fn binary_left_owned(mut self, b: &Self, operation: BinaryOperation) -> Self {
        let (value, left, right) = self.binary_terms(b, operation);
        let width = match self.binary_width(b) {
            Ok(width) => width,
            Err(failure) => return Self::invalid(failure),
        };
        self.derivative.resize(width, 0.);
        // Short vectors keep the compact indexed loop; wide vectors avoid checks
        // per element. Both paths reuse the same owned buffer at every width.
        // Width validation guarantees either a broadcast constant or equal lengths.
        // Keep zero products explicit: they still detect nonfinite slopes.
        if width <= INDEXED_WIDTH_LIMIT {
            for (i, d) in self.derivative.iter_mut().enumerate() {
                *d = *d * left + b.derivative.get(i).copied().unwrap_or(0.) * right;
            }
        } else if b.derivative.is_empty() {
            for d in &mut self.derivative {
                *d = *d * left + 0. * right;
            }
        } else {
            for (d, rhs) in self.derivative.iter_mut().zip(&b.derivative) {
                *d = *d * left + rhs * right;
            }
        }
        Self::from_parts(value, self.derivative)
    }
    fn binary_right_owned(&self, mut b: Self, operation: BinaryOperation) -> Self {
        let (value, left, right) = self.binary_terms(&b, operation);
        let width = match self.binary_width(&b) {
            Ok(width) => width,
            Err(failure) => return Self::invalid(failure),
        };
        b.derivative.resize(width, 0.);
        if width <= INDEXED_WIDTH_LIMIT {
            for (i, d) in b.derivative.iter_mut().enumerate() {
                *d = self.derivative.get(i).copied().unwrap_or(0.) * left + *d * right;
            }
        } else if self.derivative.is_empty() {
            for d in &mut b.derivative {
                *d = 0. * left + *d * right;
            }
        } else {
            for (lhs, d) in self.derivative.iter().zip(&mut b.derivative) {
                *d = lhs * left + *d * right;
            }
        }
        Self::from_parts(value, b.derivative)
    }
    fn binary_owned(self, b: Self, operation: BinaryOperation) -> Self {
        // Prefer an already active buffer. Operand order and zero products stay exact.
        if self.derivative.len() >= b.derivative.len() {
            self.binary_left_owned(&b, operation)
        } else {
            self.binary_right_owned(b, operation)
        }
    }
    fn neg_owned(mut self) -> Self {
        if let Some(failure) = self.failure {
            return Self::invalid(failure);
        }
        // Negating a validated finite jet cannot overflow. Keep the original
        // multiplication so signed-zero and derivative bits remain unchanged.
        self.value = -self.value;
        for d in &mut self.derivative {
            *d *= -1.;
        }
        self
    }
    fn neg_borrowed(&self) -> Self {
        if let Some(failure) = self.failure {
            return Self::invalid(failure);
        }
        Self {
            value: -self.value,
            derivative: self.derivative.iter().map(|d| d * -1.).collect(),
            failure: None,
        }
    }
    fn add_jet(&self, b: &Self) -> Self {
        self.binary(b, self.value + b.value, 1.0, 1.0)
    }
    fn sub_jet(&self, b: &Self) -> Self {
        self.binary(b, self.value - b.value, 1.0, -1.0)
    }
    fn mul_jet(&self, b: &Self) -> Self {
        self.binary(b, self.value * b.value, b.value, self.value)
    }
    fn div_jet(&self, b: &Self) -> Self {
        self.binary(
            b,
            self.value / b.value,
            1.0 / b.value,
            -(self.value / b.value) / b.value,
        )
    }
    pub fn sin(&self) -> Self {
        self.unary(self.value.sin(), self.value.cos())
    }
    pub fn cos(&self) -> Self {
        self.unary(self.value.cos(), -self.value.sin())
    }
    pub fn tan(&self) -> Self {
        let value = self.value.tan();
        self.unary(value, 1.0 + value * value)
    }
    pub fn atan(&self) -> Self {
        self.unary(self.value.atan(), crate::autodiff::atan_slope(self.value))
    }
    pub fn exp(&self) -> Self {
        let value = self.value.exp();
        self.unary(value, value)
    }
    pub fn ln(&self) -> Self {
        self.unary(self.value.ln(), 1.0 / self.value)
    }
    pub fn sqrt(&self) -> Self {
        let value = self.value.sqrt();
        self.unary(value, 0.5 / value)
    }
    pub fn powi(&self, n: i32) -> Self {
        let (value, slope) = crate::autodiff::power_value_slope(self.value, n);
        self.unary(value, slope)
    }
    /// y.atan2(x); the active-variable derivative at the origin is undefined.
    pub fn atan2(&self, x: &Self) -> Self {
        let (dy, dx) = crate::autodiff::atan2_slopes(self.value, x.value);
        self.binary(x, self.value.atan2(x.value), dy, dx)
    }
}
macro_rules! jet_operator {
    ($trait:ident,$method:ident,$helper:ident) => {
        impl $trait for DynamicJet {
            type Output = DynamicJet;
            fn $method(self, b: DynamicJet) -> DynamicJet {
                self.binary_owned(b, BinaryOperation::$trait)
            }
        }
        impl $trait<&DynamicJet> for DynamicJet {
            type Output = DynamicJet;
            fn $method(self, b: &DynamicJet) -> DynamicJet {
                self.binary_left_owned(b, BinaryOperation::$trait)
            }
        }
        impl $trait<DynamicJet> for &DynamicJet {
            type Output = DynamicJet;
            fn $method(self, b: DynamicJet) -> DynamicJet {
                self.binary_right_owned(b, BinaryOperation::$trait)
            }
        }
        impl $trait<&DynamicJet> for &DynamicJet {
            type Output = DynamicJet;
            fn $method(self, b: &DynamicJet) -> DynamicJet {
                self.$helper(b)
            }
        }
        impl $trait<f64> for DynamicJet {
            type Output = DynamicJet;
            fn $method(self, b: f64) -> DynamicJet {
                self.binary_left_owned(&DynamicJet::constant(b), BinaryOperation::$trait)
            }
        }
        impl $trait<f64> for &DynamicJet {
            type Output = DynamicJet;
            fn $method(self, b: f64) -> DynamicJet {
                self.$helper(&DynamicJet::constant(b))
            }
        }
        impl $trait<DynamicJet> for f64 {
            type Output = DynamicJet;
            fn $method(self, b: DynamicJet) -> DynamicJet {
                DynamicJet::constant(self).binary_right_owned(b, BinaryOperation::$trait)
            }
        }
        impl $trait<&DynamicJet> for f64 {
            type Output = DynamicJet;
            fn $method(self, b: &DynamicJet) -> DynamicJet {
                DynamicJet::constant(self).$helper(b)
            }
        }
    };
}
jet_operator!(Add, add, add_jet);
jet_operator!(Sub, sub, sub_jet);
jet_operator!(Mul, mul, mul_jet);
jet_operator!(Div, div, div_jet);
impl Neg for DynamicJet {
    type Output = Self;
    fn neg(self) -> Self {
        self.neg_owned()
    }
}
impl Neg for &DynamicJet {
    type Output = DynamicJet;
    fn neg(self) -> DynamicJet {
        self.neg_borrowed()
    }
}
impl From<f64> for DynamicJet {
    fn from(value: f64) -> Self {
        Self::constant(value)
    }
}

/// Runtime-sized Euclidean residual adapter. Its deterministic callback is invoked
/// once per derivative chunk, with identical primal inputs. Constants broadcast;
/// only supplied input jets should be used for seeding in the callback.
pub struct DynamicAutoDiff<F> {
    parameters: usize,
    residuals: usize,
    chunk_size: usize,
    function: F,
}
impl<F: Fn(&[DynamicJet]) -> Result<Vec<DynamicJet>, Error>> DynamicAutoDiff<F> {
    pub fn new(parameters: usize, residuals: usize, function: F) -> Result<Self, Error> {
        if parameters == 0 || residuals == 0 {
            return Err(Error::InvalidDimensions);
        }
        Ok(Self {
            parameters,
            residuals,
            chunk_size: 32,
            function,
        })
    }
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Result<Self, Error> {
        if chunk_size == 0 {
            return Err(Error::InvalidOptions);
        }
        self.chunk_size = chunk_size;
        Ok(self)
    }
}
impl<F: Fn(&[DynamicJet]) -> Result<Vec<DynamicJet>, Error>> Residual for DynamicAutoDiff<F> {
    fn dimensions(&self) -> (usize, usize) {
        (self.parameters, self.residuals)
    }
    fn evaluate(
        &self,
        x: &DVector<f64>,
        r: &mut DVector<f64>,
        j: &mut DMatrix<f64>,
    ) -> Result<(), Error> {
        if x.len() != self.parameters
            || r.len() != self.residuals
            || j.shape() != (self.residuals, self.parameters)
        {
            return Err(Error::InvalidDimensions);
        }
        if !x.iter().all(|v| v.is_finite()) {
            return Err(Error::NonFinite);
        }
        for start in (0..self.parameters).step_by(self.chunk_size) {
            let width = (self.parameters - start).min(self.chunk_size);
            let end = start + width;
            let inputs = x
                .iter()
                .enumerate()
                .map(|(i, value)| {
                    if i >= start && i < end {
                        DynamicJet::variable(*value, i - start, width)
                    } else {
                        Ok(DynamicJet::constant(*value))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let outputs = (self.function)(&inputs)?;
            if outputs.len() != self.residuals {
                return Err(Error::InvalidDimensions);
            }
            for (row, output) in outputs.iter().enumerate() {
                let partials = output.partials()?;
                if !partials.is_empty() && partials.len() != width {
                    return Err(Error::InvalidDimensions);
                }
                if start == 0 {
                    r[row] = output.value();
                } else if r[row].to_bits() != output.value().to_bits() {
                    return Err(Error::Evaluation(
                        "autodiff residual values changed between derivative chunks".into(),
                    ));
                }
                for col in 0..width {
                    j[(row, start + col)] = partials.get(col).copied().unwrap_or(0.0);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod reuse_tests {
    use super::*;
    fn bits(jet: &DynamicJet) -> (u64, Result<Vec<u64>, Error>) {
        (
            jet.value().to_bits(),
            jet.partials()
                .map(|v| v.iter().map(|x| x.to_bits()).collect()),
        )
    }
    fn reference(a: &DynamicJet, b: &DynamicJet, op: BinaryOperation) -> DynamicJet {
        match op {
            BinaryOperation::Add => a.add_jet(b),
            BinaryOperation::Sub => a.sub_jet(b),
            BinaryOperation::Mul => a.mul_jet(b),
            BinaryOperation::Div => a.div_jet(b),
        }
    }

    #[test]
    fn owned_forms_match_allocating_reference_bits_errors_and_borrowed_inputs() {
        for a_len in [0, 1, 2, 6, 7, 12, 13, 16, 17, 32, 65] {
            for b_len in [0, a_len, 3, 7] {
                for (a_value, b_value) in [
                    (0.7, 1.2),
                    (-0., 0.),
                    (1e308, 1e308),
                    (1e308, 1.),
                    (1e-308, 1e-308),
                    (f64::NAN, 1.),
                ] {
                    let a = DynamicJet::from_parts(
                        a_value,
                        (0..a_len)
                            .map(|i| match i % 4 {
                                0 => -0.,
                                1 => 0.,
                                2 => 1e-150,
                                _ => -1e150,
                            })
                            .collect(),
                    );
                    let b = DynamicJet::from_parts(
                        b_value,
                        (0..b_len)
                            .map(|i| if i % 2 == 0 { -0. } else { 0.75 })
                            .collect(),
                    );
                    for op in [
                        BinaryOperation::Add,
                        BinaryOperation::Sub,
                        BinaryOperation::Mul,
                        BinaryOperation::Div,
                    ] {
                        let expected = bits(&reference(&a, &b, op));
                        let original_a = bits(&a);
                        let original_b = bits(&b);
                        assert_eq!(bits(&a.clone().binary_owned(b.clone(), op)), expected);
                        assert_eq!(bits(&a.clone().binary_left_owned(&b, op)), expected);
                        assert_eq!(bits(&a.binary_right_owned(b.clone(), op)), expected);
                        assert_eq!(bits(&a), original_a);
                        assert_eq!(bits(&b), original_b);
                    }
                }
            }
        }
    }

    #[test]
    fn negation_preserves_extreme_finite_bits_and_carried_failures() {
        for width in [0, 1, 6, 16, 17, 32, 65] {
            for value in [-f64::MAX, -0., 0., f64::MIN_POSITIVE, f64::MAX] {
                let jet = DynamicJet::from_parts(
                    value,
                    (0..width)
                        .map(|i| [-f64::MAX, -0., 0., f64::from_bits(1), f64::MAX][i % 5])
                        .collect(),
                );
                let before = bits(&jet);
                let expected = bits(&jet.unary(-value, -1.));
                assert_eq!(bits(&(-jet.clone())), expected);
                assert_eq!(bits(&(-&jet)), expected);
                assert_eq!(bits(&jet), before);
            }
        }
        for jet in [
            DynamicJet::invalid(Failure::Dimension),
            DynamicJet::from_parts(f64::NAN, vec![1.]),
            DynamicJet::from_parts(0., vec![f64::INFINITY]),
        ] {
            let expected = bits(&jet.unary(-jet.value, -1.));
            assert_eq!(bits(&(-jet.clone())), expected);
            assert_eq!(bits(&(-&jet)), expected);
        }
    }

    #[test]
    fn consumed_active_buffer_is_reused_for_either_operand_and_negation() {
        for width in [1, 2, 6, 7, 12, 16, 17, 32, 65] {
            let a = DynamicJet::variable(2., width - 1, width).unwrap();
            let pointer = a.derivative.as_ptr();
            let result = a + 3.;
            assert_eq!(result.derivative.as_ptr(), pointer);
            let b = DynamicJet::variable(2., width - 1, width).unwrap();
            let pointer = b.derivative.as_ptr();
            let result = 3. - b;
            assert_eq!(result.derivative.as_ptr(), pointer);
            let a = DynamicJet::variable(2., width - 1, width).unwrap();
            let pointer = a.derivative.as_ptr();
            let result = -a;
            assert_eq!(result.derivative.as_ptr(), pointer);
            let b = DynamicJet::variable(2., width - 1, width).unwrap();
            let pointer = b.derivative.as_ptr();
            let result = DynamicJet::constant(3.) / b;
            assert_eq!(result.derivative.as_ptr(), pointer);
            let a = DynamicJet::variable(2., width - 1, width).unwrap();
            let pointer = a.derivative.as_ptr();
            let result = a * DynamicJet::constant(3.);
            assert_eq!(result.derivative.as_ptr(), pointer);
        }
    }
}
