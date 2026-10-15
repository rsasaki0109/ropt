//! Compensated arithmetic for ill-conditioned information whitening only.
use crate::{DMatrix, DVector, Error};

#[derive(Clone, Copy, Default, Debug)]
struct Wide {
    hi: f64,
    lo: f64,
}
impl Wide {
    fn new(value: f64) -> Self {
        Self { hi: value, lo: 0. }
    }
    fn sum(a: f64, b: f64) -> Self {
        let hi = a + b;
        let tail = hi - a;
        Self {
            hi,
            lo: (a - (hi - tail)) + (b - tail),
        }
    }
    fn add(self, other: Self) -> Self {
        let s = Self::sum(self.hi, other.hi);
        Self::sum(s.hi, s.lo + self.lo + other.lo)
    }
    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
    fn sub(self, other: Self) -> Self {
        self.add(other.neg())
    }
    fn mul(self, other: Self) -> Self {
        let hi = self.hi * other.hi;
        let lo = self.hi.mul_add(other.hi, -hi)
            + self.hi * other.lo
            + self.lo * other.hi
            + self.lo * other.lo;
        Self::sum(hi, lo)
    }
    fn div(self, other: Self) -> Self {
        let first = Self::new(self.hi / other.hi);
        let second = Self::new(self.sub(other.mul(first)).hi / other.hi);
        let result = first.add(second);
        result.add(Self::new(self.sub(other.mul(result)).hi / other.hi))
    }
    fn sqrt(self) -> Self {
        let first = Self::new(self.hi.sqrt());
        first.add(self.sub(first.mul(first)).div(Self::new(2. * first.hi)))
    }
    fn finite(self) -> bool {
        self.hi.is_finite() && self.lo.is_finite()
    }
}

#[derive(Debug, Clone)]
pub(super) struct PreciseMatrix {
    n: usize,
    // Row-major upper square-root factor. Both coefficient components are kept.
    values: Vec<Wide>,
}
impl PreciseMatrix {
    pub(super) fn from_information(information: &DMatrix<f64>) -> Result<Self, Error> {
        let n = information.nrows();
        let mut lower = vec![Wide::default(); n.checked_mul(n).ok_or(Error::InvalidDimensions)?];
        for row in 0..n {
            for col in 0..=row {
                let mut value = Wide::new(information[(row, col)]);
                for k in 0..col {
                    value = value.sub(lower[row * n + k].mul(lower[col * n + k]));
                }
                if !value.finite() {
                    return Err(Error::NonFinite);
                }
                let entry = if row == col {
                    if value.hi <= 0. {
                        return Err(Error::InvalidOptions);
                    }
                    value.sqrt()
                } else {
                    value.div(lower[col * n + col])
                };
                if !entry.finite() {
                    return Err(Error::NonFinite);
                }
                lower[row * n + col] = entry;
            }
        }
        let mut values = vec![Wide::default(); n * n];
        for row in 0..n {
            for col in row..n {
                values[row * n + col] = lower[col * n + row];
            }
        }
        Ok(Self { n, values })
    }
    pub(super) fn dimension(&self) -> usize {
        self.n
    }
    pub(super) fn multiply_vector(
        &self,
        x: &DVector<f64>,
        out: &mut DVector<f64>,
    ) -> Result<(), Error> {
        for row in 0..self.n {
            let mut value = Wide::default();
            for col in row..self.n {
                value = value.add(self.values[row * self.n + col].mul(Wide::new(x[col])));
            }
            if !value.finite() {
                return Err(Error::NonFinite);
            }
            out[row] = value.hi + value.lo;
        }
        Ok(())
    }
    pub(super) fn multiply_block(
        &self,
        x: &DMatrix<f64>,
        out: &mut DMatrix<f64>,
    ) -> Result<(), Error> {
        for c in 0..x.ncols() {
            for row in 0..self.n {
                let mut value = Wide::default();
                for k in row..self.n {
                    value = value.add(self.values[row * self.n + k].mul(Wide::new(x[(k, c)])));
                }
                if !value.finite() {
                    return Err(Error::NonFinite);
                }
                out[(row, c)] = value.hi + value.lo;
            }
        }
        Ok(())
    }
}
