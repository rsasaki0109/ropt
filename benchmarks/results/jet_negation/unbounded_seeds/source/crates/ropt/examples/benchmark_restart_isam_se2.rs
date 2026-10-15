//! Same restart policy with batch-trial telemetry (extraction outside timers).
#[path = "support/se2_benchmark.rs"]
mod benchmark;
#[global_allocator]
static ALLOCATOR: benchmark::Counter = benchmark::Counter;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    benchmark::run("restart", ropt::isam::IsamSmoother::new, |report| {
        report
            .batch_initial_damping
            .zip(report.batch_final_damping)
            .map(|(a, b)| (report.batch_trials, a, b))
    })
}
