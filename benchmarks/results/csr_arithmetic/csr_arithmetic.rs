use ropt::{DVector, Error, sparse::CsrMatrix};

#[test]
fn finite_operands_that_overflow_return_nonfinite_without_changing_matrix() {
    // Multiplication itself overflows; every input is finite.
    let matrix = CsrMatrix::from_triplets(1, 1, vec![(0, 0, f64::MAX)]).unwrap();
    for product in [
        matrix.multiply(&DVector::from_element(1, 2.)),
        matrix.transpose_multiply(&DVector::from_element(1, 2.)),
        matrix.normal_diagonal(),
    ] {
        assert_eq!(product.unwrap_err(), Error::NonFinite);
    }
    // Failed products leave a reusable matrix, including the original value.
    assert_eq!(matrix.entries().collect::<Vec<_>>(), vec![(0, 0, f64::MAX)]);
    assert_eq!(
        matrix.multiply(&DVector::from_element(1, 1.)).unwrap()[0],
        f64::MAX
    );
    assert_eq!(
        matrix
            .transpose_multiply(&DVector::from_element(1, 1.))
            .unwrap()[0],
        f64::MAX
    );

    // Individual products are finite; only their accumulation overflows.
    let large = f64::MAX * 0.75;
    let row = CsrMatrix::from_triplets(1, 2, vec![(0, 0, large), (0, 1, large)]).unwrap();
    let column = CsrMatrix::from_triplets(2, 1, vec![(0, 0, large), (1, 0, large)]).unwrap();
    assert_eq!(
        row.multiply(&DVector::from_element(2, 1.)).unwrap_err(),
        Error::NonFinite
    );
    assert_eq!(
        column
            .transpose_multiply(&DVector::from_element(2, 1.))
            .unwrap_err(),
        Error::NonFinite
    );
    // Each square fits in f64, but the column's sum of squares does not.
    let root = large.sqrt();
    assert!((root * root).is_finite());
    let column = CsrMatrix::from_triplets(2, 1, vec![(0, 0, root), (1, 0, root)]).unwrap();
    assert_eq!(column.normal_diagonal().unwrap_err(), Error::NonFinite);
}

#[test]
fn structurally_unused_nonfinite_vector_entries_are_rejected() {
    let matrix = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 3.)]).unwrap();
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let vector = DVector::from_vec(vec![2., value]);
        assert_eq!(matrix.multiply(&vector).unwrap_err(), Error::NonFinite);
        assert_eq!(
            matrix.transpose_multiply(&vector).unwrap_err(),
            Error::NonFinite
        );
    }
    let finite = DVector::from_vec(vec![2., 5.]);
    let expected = DVector::from_vec(vec![6., 0.]);
    assert_eq!(matrix.multiply(&finite).unwrap(), expected);
    assert_eq!(matrix.transpose_multiply(&finite).unwrap(), expected);
    assert_eq!(
        matrix.normal_diagonal().unwrap(),
        DVector::from_vec(vec![9., 0.])
    );
}

#[test]
fn duplicate_reduction_rejects_overflow_and_preserves_finite_cancellation() {
    for sign in [1., -1.] {
        let value = sign * f64::MAX;
        assert_eq!(
            CsrMatrix::from_triplets(1, 1, vec![(0, 0, value), (0, 0, value)]).unwrap_err(),
            Error::NonFinite
        );
    }
    let matrix = CsrMatrix::from_triplets(1, 1, vec![(0, 0, f64::MAX), (0, 0, -f64::MAX)]).unwrap();
    assert_eq!(matrix.nnz(), 0);
    assert_eq!(
        matrix
            .multiply(&DVector::from_element(1, f64::MAX))
            .unwrap()[0],
        0.
    );
    assert_eq!(matrix.normal_diagonal().unwrap()[0], 0.);
}
