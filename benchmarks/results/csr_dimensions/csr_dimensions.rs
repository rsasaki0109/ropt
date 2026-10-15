use ropt::{DVector, Error, sparse::CsrMatrix};

#[test]
fn unrepresentable_row_storage_returns_dimension_error() {
    for rows in [usize::MAX, isize::MAX as usize / size_of::<usize>()] {
        assert_eq!(
            CsrMatrix::from_triplets(rows, 0, vec![]).unwrap_err(),
            Error::InvalidDimensions
        );
    }
}

#[test]
fn empty_shapes_and_wide_sparse_metadata_remain_valid() {
    for (rows, columns) in [(0, 0), (0, 3), (2, 0), (0, usize::MAX)] {
        let matrix = CsrMatrix::from_triplets(rows, columns, vec![]).unwrap();
        assert_eq!(matrix.shape(), (rows, columns));
        assert_eq!(matrix.nnz(), 0);
        assert_eq!(matrix.entries().count(), 0);
    }
    let matrix = CsrMatrix::from_triplets(2, 0, vec![]).unwrap();
    assert_eq!(
        matrix.multiply(&DVector::zeros(0)).unwrap(),
        DVector::zeros(2)
    );
    assert_eq!(
        matrix.transpose_multiply(&DVector::zeros(2)).unwrap().len(),
        0
    );
}
