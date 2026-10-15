//! Frozen initial linear system: diagnostic timers, not complete-solve benchmarks.
use ropt::{g2o::PoseGraph2, preconditioner::*, sparse::*, *};
use std::cell::Cell;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;
struct Timed<T> {
    inner: T,
    calls: Cell<usize>,
    ns: Cell<u128>,
}
impl<T> Timed<T> {
    fn new(inner: T) -> Self {
        Self {
            inner,
            calls: Cell::new(0),
            ns: Cell::new(0),
        }
    }
    fn record(&self, start: Instant) {
        self.ns.set(self.ns.get() + start.elapsed().as_nanos());
        self.calls.set(self.calls.get() + 1);
    }
}
impl<T: LinearOperator> LinearOperator for Timed<T> {
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
    fn diagonal(&self) -> Result<DVector<f64>, Error> {
        self.inner.diagonal()
    }
    fn apply(&self, x: &DVector<f64>) -> Result<DVector<f64>, Error> {
        self.inner.apply(x)
    }
    fn apply_into(&self, x: &DVector<f64>, out: &mut DVector<f64>) -> Result<(), Error> {
        let start = Instant::now();
        let result = self.inner.apply_into(x, out);
        self.record(start);
        result
    }
}
impl<T: Preconditioner> Preconditioner for Timed<T> {
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
    fn apply(&self, r: &DVector<f64>) -> Result<DVector<f64>, Error> {
        self.inner.apply(r)
    }
    fn apply_into(&self, r: &DVector<f64>, out: &mut DVector<f64>) -> Result<(), Error> {
        let start = Instant::now();
        let result = self.inner.apply_into(r, out);
        self.record(start);
        result
    }
}
fn run(j: &CsrMatrix, name: &str, pre: impl Preconditioner) -> Result<(), Error> {
    let operator = Timed::new(NormalOperator::new(j, 1e-9)?);
    let pre = Timed::new(pre);
    let b = DVector::from_iterator(
        j.shape().1,
        (0..j.shape().1).map(|i| (i as f64 * 0.13).cos()),
    );
    let start = Instant::now();
    let r = conjugate_gradient_preconditioned(
        &operator,
        &pre,
        &b,
        &CgOptions {
            max_iterations: 10000,
            relative_tolerance: 1e-10,
            absolute_tolerance: 1e-12,
        },
    )?;
    let total = start.elapsed().as_nanos();
    println!(
        "{name},{total},{},{},{},{},{},{:.17e}",
        operator.ns.get(),
        pre.ns.get(),
        operator.calls.get(),
        pre.calls.get(),
        r.iterations,
        r.residual_norm
    );
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        return Err("profiling requires release build".into());
    }
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: profile_linear input.g2o")?;
    let data = PoseGraph2::from_reader(BufReader::new(File::open(path)?))?;
    let graph = data.build(None)?.graph;
    let (_, j) = graph.evaluate_sparse(&graph.initial())?;
    println!(
        "backend,total_ns,operator_ns,preconditioner_ns,operator_calls,preconditioner_calls,iterations,residual_norm"
    );
    run(
        &j,
        "scalar_cg",
        Jacobi::new(NormalOperator::new(&j, 1e-9)?.diagonal()?)?,
    )?;
    run(
        &j,
        "block_cg",
        BlockJacobi::from_normal(&j, 1e-9, &graph.tangent_blocks()?)?,
    )?;
    Ok(())
}
