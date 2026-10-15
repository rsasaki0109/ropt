use ropt::{DVector, Error, preconditioner::Preconditioner, sparse::*};
use std::cell::Cell;

struct IdentityPreconditioner {
    dimension: usize,
    calls: Cell<usize>,
}
impl Preconditioner for IdentityPreconditioner {
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn apply(&self, r: &DVector<f64>) -> Result<DVector<f64>, Error> {
        self.calls.set(self.calls.get() + 1);
        Ok(r.clone())
    }
}

struct CheckedOperator {
    matrix: CsrMatrix,
    calls: Cell<usize>,
    fail_final_probe: bool,
}
impl LinearOperator for CheckedOperator {
    fn dimension(&self) -> usize {
        self.matrix.shape().0
    }
    fn diagonal(&self) -> Result<DVector<f64>, Error> {
        self.matrix.diagonal()
    }
    fn apply(&self, x: &DVector<f64>) -> Result<DVector<f64>, Error> {
        let calls = self.calls.get() + 1;
        self.calls.set(calls);
        if self.fail_final_probe && calls == 3 {
            return Err(Error::Evaluation("final residual probe failed".into()));
        }
        self.matrix.multiply(x)
    }
}

fn one_step() -> CgOptions {
    CgOptions {
        max_iterations: 1,
        relative_tolerance: 0.,
        absolute_tolerance: 0.,
    }
}

#[test]
fn final_true_residual_can_establish_convergence_despite_recursive_roundoff() {
    // With the identity preconditioner, one scalar step produces x = 1.
    // Its recursive residual is -1.3877787807814457e-17, while b - A*x is zero.
    let matrix = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 0.1)]).unwrap();
    let operator = CheckedOperator {
        matrix,
        calls: Cell::new(0),
        fail_final_probe: false,
    };
    let pre = IdentityPreconditioner {
        dimension: 1,
        calls: Cell::new(0),
    };
    let b = DVector::from_element(1, 0.1);
    let report = conjugate_gradient_preconditioned(&operator, &pre, &b, &one_step()).unwrap();
    assert_eq!(report.iterations, 1);
    assert_eq!(report.x[0], 1.);
    assert_eq!(report.residual_norm, 0.);
    assert_eq!(operator.matrix.multiply(&report.x).unwrap(), b);
    assert_eq!(pre.calls.get(), 1);
    assert_eq!(operator.calls.get(), 3);
}

#[test]
fn unmet_final_tolerance_returns_limit_without_preparing_another_iteration() {
    let matrix =
        CsrMatrix::from_triplets(2, 2, vec![(0, 0, 4.), (0, 1, 1.), (1, 0, 1.), (1, 1, 3.)])
            .unwrap();
    let operator = CheckedOperator {
        matrix,
        calls: Cell::new(0),
        fail_final_probe: false,
    };
    let pre = IdentityPreconditioner {
        dimension: 2,
        calls: Cell::new(0),
    };
    let b = DVector::from_vec(vec![1., 2.]);
    assert_eq!(
        conjugate_gradient_preconditioned(&operator, &pre, &b, &one_step()).unwrap_err(),
        Error::LinearIterationLimit { iterations: 1 }
    );
    assert_eq!(operator.calls.get(), 3);
    assert_eq!(pre.calls.get(), 1);
    // An independent one-step result is x = [1/4, 1/2]; its true residual
    // [-1/2, 1/4] is nonzero, so reporting convergence would be incorrect.
    let candidate = DVector::from_vec(vec![0.25, 0.5]);
    assert_eq!(
        &b - operator.matrix.multiply(&candidate).unwrap(),
        DVector::from_vec(vec![-0.5, 0.25])
    );
}

#[test]
fn final_residual_evaluation_error_is_propagated() {
    let matrix = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 0.1)]).unwrap();
    let operator = CheckedOperator {
        matrix,
        calls: Cell::new(0),
        fail_final_probe: true,
    };
    let pre = IdentityPreconditioner {
        dimension: 1,
        calls: Cell::new(0),
    };
    assert_eq!(
        conjugate_gradient_preconditioned(
            &operator,
            &pre,
            &DVector::from_element(1, 0.1),
            &one_step()
        )
        .unwrap_err(),
        Error::Evaluation("final residual probe failed".into())
    );
    assert_eq!(operator.calls.get(), 3);
    assert_eq!(pre.calls.get(), 1);
}
