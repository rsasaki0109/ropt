//! Recorded SE2 replay with column-normalized convergence and sparse ordering reuse.
#[path = "support/real_se2_benchmark.rs"]
mod benchmark;
#[global_allocator]
static ALLOCATOR: benchmark::Counter = benchmark::Counter;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    benchmark::run(
        "restart(1e-5);reuse_batch_ordering;final_refinement=4;frozen_qr",
        |options| {
            ropt::isam::IsamSmoother::new(options)?
                .with_batch_ordering_reuse()?
                .with_final_refinement(4)
        },
        |report| {
            report
                .batch_initial_damping
                .zip(report.batch_final_damping)
                .map(|(a, b)| (report.batch_trials, a, b))
        },
        None,
    )
}
