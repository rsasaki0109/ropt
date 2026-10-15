use std::{
    cell::{Cell, RefCell},
    time::Instant,
};
#[derive(Clone, Copy)]
pub struct Record {
    pub ns: u128,
    pub calls: u64,
    pub errors: u64,
}
const ZERO: Record = Record {
    ns: 0,
    calls: 0,
    errors: 0,
};
thread_local! {
    static DATA: RefCell<[[Record; 11]; 4]> = const { RefCell::new([[ZERO; 11]; 4]) };
    static API: Cell<usize> = const { Cell::new(0) };
}
pub fn reset() {
    DATA.with(|d| *d.borrow_mut() = [[ZERO; 11]; 4]);
}
pub fn enter(api: usize) {
    API.with(|a| a.set(api));
}
pub fn snapshot() -> [[Record; 11]; 4] {
    DATA.with(|d| *d.borrow())
}
pub fn measure<T, E>(slot: usize, f: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    let api = API.with(Cell::get);
    let start = Instant::now();
    let result = f();
    let ns = start.elapsed().as_nanos();
    DATA.with(|d| {
        let mut d = d.borrow_mut();
        let r = &mut d[api][slot];
        r.ns += ns;
        r.calls += 1;
        r.errors += u64::from(result.is_err());
    });
    result
}
