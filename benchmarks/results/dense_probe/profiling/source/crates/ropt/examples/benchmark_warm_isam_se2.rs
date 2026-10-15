//! Same shared SE2 timings with explicitly enabled adaptive damping reuse.
#[path = "support/se2_benchmark.rs"]
mod benchmark;
#[global_allocator]
static ALLOCATOR: benchmark::Counter = benchmark::Counter;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    benchmark::run(
        "reuse",
        |options| ropt::isam::IsamSmoother::new(options)?.with_batch_damping_reuse(),
        |report| {
            report
                .batch_initial_damping
                .zip(report.batch_final_damping)
                .map(|(a, b)| (report.batch_trials, a, b))
        },
    )
}
