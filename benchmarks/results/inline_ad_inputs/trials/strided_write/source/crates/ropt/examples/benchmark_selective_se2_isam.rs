//! Explicit stationarity-controlled real SE2 policies; no stock-native timing claim.
#[path = "support/real_selective_se2.rs"]
mod benchmark;
#[global_allocator]
static ALLOCATOR: benchmark::Counter = benchmark::Counter;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    benchmark::run()
}
