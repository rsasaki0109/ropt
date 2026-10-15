use ropt::{preconditioner::*, sparse::*, *};
use std::cell::Cell;

fn jacobian() -> (CsrMatrix, DMatrix<f64>) {
    let j = DMatrix::from_row_slice(
        4,
        3,
        &[2.0, 0.3, 0.0, 0.0, 0.0, 0.0, -0.4, 1.5, 0.2, 0.7, 0.0, 1.2],
    );
    let entries = (0..4)
        .flat_map(|r| (0..3).map(move |c| (r, c)))
        .filter_map(|(r, c)| (j[(r, c)] != 0.0).then_some((r, c, j[(r, c)])))
        .collect();
    (CsrMatrix::from_triplets(4, 3, entries).unwrap(), j)
}
#[test]
fn reusable_outputs_match_independent_dense_products_and_cholesky() {
    let (j, dense) = jacobian();
    let op = NormalOperator::new(&j, 0.2).unwrap();
    let mut h = dense.transpose() * dense;
    for k in 0..3 {
        h[(k, k)] += 0.2 * h[(k, k)].max(1.0);
    }
    let blocks = BlockJacobi::from_normal(&j, 0.2, &[0..2, 2..3]).unwrap();
    let scalar = Jacobi::new(h.diagonal()).unwrap();
    let mut out = DVector::from_element(3, f64::NAN);
    let pointer = out.as_ptr();
    for x in [
        DVector::zeros(3),
        DVector::from_vec(vec![0.3, -1.2, 2.0]),
        DVector::from_vec(vec![-0.7, 0.5, 0.0]),
    ] {
        op.apply_into(&x, &mut out).unwrap();
        assert!((&out - &h * &x).norm() < 1e-12);
        assert_eq!(out.as_ptr(), pointer);
        scalar.apply_into(&x, &mut out).unwrap();
        assert!((&out - x.component_div(&h.diagonal())).norm() < 1e-12);
        blocks.apply_into(&x, &mut out).unwrap();
        let expected = h
            .view((0, 0), (2, 2))
            .into_owned()
            .cholesky()
            .unwrap()
            .solve(&x.rows(0, 2).into_owned());
        assert!((out.rows(0, 2) - expected).norm() < 1e-12);
        assert!((out[2] - x[2] / h[(2, 2)]).abs() < 1e-12);
        assert_eq!(out.as_ptr(), pointer);
    }
    for pre in [&scalar as &dyn Preconditioner, &blocks] {
        assert_eq!(
            pre.apply_into(&DVector::zeros(2), &mut out),
            Err(Error::InvalidDimensions)
        );
        assert_eq!(
            pre.apply_into(&DVector::zeros(3), &mut DVector::zeros(4)),
            Err(Error::InvalidDimensions)
        );
        assert_eq!(
            pre.apply_into(&DVector::from_element(3, f64::NAN), &mut out),
            Err(Error::NonFinite)
        );
    }
    assert_eq!(
        op.apply_into(&DVector::zeros(2), &mut out),
        Err(Error::InvalidDimensions)
    );
    assert_eq!(
        op.apply_into(&DVector::zeros(3), &mut DVector::zeros(4)),
        Err(Error::InvalidDimensions)
    );
    assert_eq!(
        op.apply_into(&DVector::from_element(3, f64::INFINITY), &mut out),
        Err(Error::NonFinite)
    );
}
struct BufferedOperator(DMatrix<f64>);
impl LinearOperator for BufferedOperator {
    fn dimension(&self) -> usize {
        self.0.nrows()
    }
    fn apply(&self, _: &DVector<f64>) -> Result<DVector<f64>, Error> {
        Err(Error::Evaluation("allocating apply called".into()))
    }
    fn apply_into(&self, x: &DVector<f64>, out: &mut DVector<f64>) -> Result<(), Error> {
        out.copy_from(&(&self.0 * x));
        Ok(())
    }
    fn diagonal(&self) -> Result<DVector<f64>, Error> {
        Ok(self.0.diagonal())
    }
}
struct BufferedPre {
    diagonal: DVector<f64>,
    pointer: Cell<usize>,
}
impl Preconditioner for BufferedPre {
    fn dimension(&self) -> usize {
        self.diagonal.len()
    }
    fn apply(&self, _: &DVector<f64>) -> Result<DVector<f64>, Error> {
        Err(Error::Evaluation("allocating preconditioner called".into()))
    }
    fn apply_into(&self, r: &DVector<f64>, out: &mut DVector<f64>) -> Result<(), Error> {
        let pointer = out.as_ptr() as usize;
        if self.pointer.get() != 0 {
            assert_eq!(self.pointer.get(), pointer);
        }
        self.pointer.set(pointer);
        for k in 0..r.len() {
            out[k] = r[k] / self.diagonal[k];
        }
        Ok(())
    }
}
#[test]
fn pcg_uses_buffered_overrides_and_verifies_the_actual_linear_residual() {
    let h = DMatrix::from_row_slice(3, 3, &[4.0, 1.0, 0.2, 1.0, 3.0, 0.5, 0.2, 0.5, 2.0]);
    let b = DVector::from_vec(vec![1.0, 2.0, -0.4]);
    let pre = BufferedPre {
        diagonal: h.diagonal(),
        pointer: Cell::new(0),
    };
    let result = conjugate_gradient_preconditioned(
        &BufferedOperator(h.clone()),
        &pre,
        &b,
        &CgOptions::default(),
    )
    .unwrap();
    assert!((&result.x - h.clone().cholesky().unwrap().solve(&b)).norm() < 1e-11);
    let residual = (&b - &h * &result.x).norm();
    assert!((result.residual_norm - residual).abs() < 1e-14);
    assert!(residual <= 1e-12_f64.max(1e-10 * b.norm()));
}
struct Broken(u8);
impl LinearOperator for Broken {
    fn dimension(&self) -> usize {
        1
    }
    fn apply(&self, x: &DVector<f64>) -> Result<DVector<f64>, Error> {
        Ok(x.clone())
    }
    fn diagonal(&self) -> Result<DVector<f64>, Error> {
        Ok(DVector::from_element(1, 1.0))
    }
    fn apply_into(&self, _: &DVector<f64>, out: &mut DVector<f64>) -> Result<(), Error> {
        match self.0 {
            0 => {
                out.resize_vertically_mut(2, 0.0);
                Ok(())
            }
            1 => {
                out.fill(f64::NAN);
                Ok(())
            }
            _ => Err(Error::Evaluation("buffered failure".into())),
        }
    }
}
impl Preconditioner for Broken {
    fn dimension(&self) -> usize {
        1
    }
    fn apply(&self, r: &DVector<f64>) -> Result<DVector<f64>, Error> {
        Ok(r.clone())
    }
    fn apply_into(&self, r: &DVector<f64>, out: &mut DVector<f64>) -> Result<(), Error> {
        LinearOperator::apply_into(self, r, out)
    }
}
#[test]
fn buffer_contract_failures_are_checked_after_overrides() {
    let b = DVector::from_element(1, 1.0);
    let j = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 1.0)]).unwrap();
    let op = NormalOperator::new(&j, 0.1).unwrap();
    let pre = Jacobi::new(op.diagonal().unwrap()).unwrap();
    for (behavior, error) in [
        (0, Error::InvalidDimensions),
        (1, Error::NonFinite),
        (2, Error::Evaluation("buffered failure".into())),
    ] {
        assert_eq!(
            conjugate_gradient_preconditioned(&Broken(behavior), &pre, &b, &CgOptions::default())
                .unwrap_err(),
            error
        );
        assert_eq!(
            conjugate_gradient_preconditioned(&op, &Broken(behavior), &b, &CgOptions::default())
                .unwrap_err(),
            error
        );
    }
}
struct Legacy;
impl LinearOperator for Legacy {
    fn dimension(&self) -> usize {
        2
    }
    fn apply(&self, x: &DVector<f64>) -> Result<DVector<f64>, Error> {
        Ok(2.0 * x)
    }
    fn diagonal(&self) -> Result<DVector<f64>, Error> {
        Ok(DVector::from_element(2, 2.0))
    }
}
impl Preconditioner for Legacy {
    fn dimension(&self) -> usize {
        2
    }
    fn apply(&self, r: &DVector<f64>) -> Result<DVector<f64>, Error> {
        Ok(r / 2.0)
    }
}
#[test]
fn legacy_implementations_keep_working_and_arithmetic_overflow_is_an_error() {
    let x = DVector::from_vec(vec![0.2, -0.3]);
    let mut out = DVector::from_element(2, f64::NAN);
    LinearOperator::apply_into(&Legacy, &x, &mut out).unwrap();
    assert_eq!(out, 2.0 * &x);
    Preconditioner::apply_into(&Legacy, &x, &mut out).unwrap();
    assert_eq!(out, &x / 2.0);
    let result =
        conjugate_gradient_preconditioned(&Legacy, &Legacy, &x, &CgOptions::default()).unwrap();
    assert!((result.x - &x / 2.0).norm() < 1e-14);
    let huge = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 1e154)]).unwrap();
    let op = NormalOperator::new(&huge, 0.0).unwrap();
    assert_eq!(
        op.apply_into(&DVector::from_element(1, 1e200), &mut DVector::zeros(1)),
        Err(Error::NonFinite)
    );
}
