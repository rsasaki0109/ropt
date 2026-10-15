//! Accuracy-matched SE2 benchmark, with the default damping restart policy.
#[path = "support/se2_benchmark.rs"]
mod benchmark;
#[global_allocator]
static ALLOCATOR: benchmark::Counter = benchmark::Counter;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    benchmark::run("restart", ropt::isam::IsamSmoother::new, |_| None)
}
