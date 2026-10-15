use ropt::{
    DVector, Error,
    preconditioner::{BlockJacobi, Preconditioner},
    sparse::CsrMatrix,
};
use std::alloc::{GlobalAlloc, Layout, System};

// Keep malformed-size regressions bounded even if constructor preflight is
// accidentally removed. A bad allocation request aborts this test binary
// instead of exhausting the machine; correct construction never reaches it.
struct BoundedAllocator;
const ALLOCATION_LIMIT: usize = 16 * 1024 * 1024;
// SAFETY: Oversized requests are rejected with null, as permitted by GlobalAlloc.
// All accepted layouts, sizes and pointers are forwarded unchanged to System.
unsafe impl GlobalAlloc for BoundedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() > ALLOCATION_LIMIT {
            return std::ptr::null_mut();
        }
        // SAFETY: The caller supplies a valid allocation layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() > ALLOCATION_LIMIT {
            return std::ptr::null_mut();
        }
        // SAFETY: The caller supplies a valid allocation layout.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > ALLOCATION_LIMIT {
            return std::ptr::null_mut();
        }
        // SAFETY: The original System allocation and valid new size are preserved.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: The original System allocation and layout are preserved.
        unsafe { System.dealloc(pointer, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: BoundedAllocator = BoundedAllocator;

#[test]
fn unrepresentable_column_mapping_returns_dimension_error() {
    for columns in [
        usize::MAX,
        isize::MAX as usize / size_of::<(usize, usize)>() + 1,
    ] {
        let matrix = CsrMatrix::from_triplets(0, columns, vec![]).unwrap();
        let block = 0..columns;
        assert!(matches!(
            BlockJacobi::from_normal(&matrix, 1., std::slice::from_ref(&block)),
            Err(Error::InvalidDimensions)
        ));
    }
}

#[test]
fn unrepresentable_dense_blocks_are_rejected_before_allocating_column_mapping() {
    // Both mappings fit Rust's allocation layout. The first square's element
    // count overflows usize; the second square's byte size exceeds isize::MAX.
    // The allocator cap makes these safe to run even on a regression.
    for columns in [
        1usize << (usize::BITS / 2),
        1usize << ((usize::BITS - 4) / 2),
    ] {
        let matrix = CsrMatrix::from_triplets(0, columns, vec![]).unwrap();
        let block = 0..columns;
        assert!(matches!(
            BlockJacobi::from_normal(&matrix, 1., std::slice::from_ref(&block)),
            Err(Error::InvalidDimensions)
        ));
    }
}

#[test]
fn empty_rows_with_valid_damped_blocks_still_apply_the_expected_inverse() {
    let matrix = CsrMatrix::from_triplets(0, 3, vec![]).unwrap();
    let r = DVector::from_vec(vec![1., -2., 3.]);
    for ranges in [vec![0..1, 1..3], std::iter::once(0..3).collect()] {
        let pre = BlockJacobi::from_normal(&matrix, 0.5, &ranges).unwrap();
        assert_eq!(pre.dimension(), 3);
        let actual = pre.apply(&r).unwrap();
        for k in 0..3 {
            assert!((actual[k] - 2. * r[k]).abs() < 1e-13);
        }
    }
}
