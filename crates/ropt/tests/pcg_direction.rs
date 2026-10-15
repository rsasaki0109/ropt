use ropt::{
    DVector, Error,
    preconditioner::{Jacobi, Preconditioner},
    sparse::*,
};
use std::cell::Cell;

struct GuardedOperator {
    matrix: CsrMatrix,
    calls: Cell<usize>,
}
impl LinearOperator for GuardedOperator {
    fn dimension(&self) -> usize {
        self.matrix.shape().0
    }
    fn diagonal(&self) -> Result<DVector<f64>, Error> {
        self.matrix.diagonal()
    }
    fn apply(&self, x: &DVector<f64>) -> Result<DVector<f64>, Error> {
        self.calls.set(self.calls.get() + 1);
        if x.iter().any(|v| !v.is_finite()) {
            return Err(Error::Evaluation(
                "operator received nonfinite direction".into(),
            ));
        }
        self.matrix.multiply(x)
    }
}

struct ScaledIdentity(f64);
impl Preconditioner for ScaledIdentity {
    fn dimension(&self) -> usize {
        1
    }
    fn apply(&self, r: &DVector<f64>) -> Result<DVector<f64>, Error> {
        Ok(r * self.0)
    }
}

fn options() -> CgOptions {
    CgOptions {
        max_iterations: 2,
        absolute_tolerance: 0.,
        relative_tolerance: 0.,
    }
}

#[test]
fn overflowing_beta_does_not_reach_the_operator() {
    // SPD: the determinant is 1e280 > 0 and both diagonal entries are positive.
    // With identity preconditioning, r.dot(z) grows from about 1e-300 to
    // 1e20 after the first step; their ratio overflows.
    let operator = GuardedOperator {
        matrix: CsrMatrix::from_triplets(
            2,
            2,
            vec![(0, 0, 1e-20), (0, 1, 1e140), (1, 0, 1e140), (1, 1, 2e300)],
        )
        .unwrap(),
        calls: Cell::new(0),
    };
    let pre = Jacobi::new(DVector::from_element(2, 1.)).unwrap();
    assert_eq!(
        conjugate_gradient_preconditioned(
            &operator,
            &pre,
            &DVector::from_vec(vec![1e-150, 0.]),
            &options()
        )
        .unwrap_err(),
        Error::NonFinite
    );
    assert_eq!(operator.calls.get(), 2);
}

#[test]
fn finite_beta_with_overflowing_direction_does_not_reach_the_operator() {
    // SPD: determinant = 2 - 1 = 1. The fixed SPD preconditioner is
    // M = diag(1e-300, 1). After one step beta is finite (about 1e300),
    // but multiplying the old direction's first entry (about 1e150) overflows.
    let operator = GuardedOperator {
        matrix: CsrMatrix::from_triplets(
            2,
            2,
            vec![(0, 0, 1e-300), (0, 1, 1.), (1, 0, 1.), (1, 1, 2e300)],
        )
        .unwrap(),
        calls: Cell::new(0),
    };
    let pre = Jacobi::new(DVector::from_vec(vec![1e-300, 1.])).unwrap();
    assert_eq!(
        conjugate_gradient_preconditioned(
            &operator,
            &pre,
            &DVector::from_vec(vec![1e-150, 0.]),
            &options()
        )
        .unwrap_err(),
        Error::NonFinite
    );
    assert_eq!(operator.calls.get(), 2);
}

#[test]
fn invalid_initial_preconditioned_inner_products_stop_before_direction_evaluation() {
    for (scale, rhs, expected) in [
        (0., 1., Error::LinearSolve),
        (-1., 1., Error::LinearSolve),
        (1., 1e200, Error::NonFinite),
    ] {
        let operator = GuardedOperator {
            matrix: CsrMatrix::from_triplets(1, 1, vec![(0, 0, 1.)]).unwrap(),
            calls: Cell::new(0),
        };
        assert_eq!(
            conjugate_gradient_preconditioned(
                &operator,
                &ScaledIdentity(scale),
                &DVector::from_element(1, rhs),
                &options()
            )
            .unwrap_err(),
            expected
        );
        assert_eq!(operator.calls.get(), 1);
    }
}
