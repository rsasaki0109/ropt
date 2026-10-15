//! Process-wide allocation traffic for a separate, single-threaded diagnostic run.
//! Counts successful allocation/reallocation requests, not live or peak memory.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
pub struct Counter;
static ENABLED: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
fn record(pointer: *mut u8, bytes: usize) {
    if !pointer.is_null() && ENABLED.load(Ordering::Relaxed) {
        CALLS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    }
}
// SAFETY: All allocation operations and their original layouts/pointers are
// forwarded unchanged to System. Accounting never allocates or touches the memory.
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller supplies a valid allocation layout.
        let pointer = unsafe { System.alloc(layout) };
        record(pointer, layout.size());
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller supplies a valid allocation layout.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        record(pointer, layout.size());
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: The original System allocation and layout are passed unchanged.
        unsafe { System.dealloc(pointer, layout) };
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: The caller provides the original allocation and valid new size.
        let pointer = unsafe { System.realloc(pointer, layout, size) };
        record(pointer, size);
        pointer
    }
}
pub fn start() {
    assert!(!ENABLED.load(Ordering::Relaxed));
    CALLS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    ENABLED.store(true, Ordering::Relaxed);
}
pub fn stop() -> (u64, u64) {
    ENABLED.store(false, Ordering::Relaxed);
    (CALLS.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed))
}
