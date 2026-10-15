//! Prepared LDL reuse allocation diagnostic; these instrumented times are not benchmarks.
#[path = "support/allocation_counter.rs"]
mod allocation;
use ropt::{g2o::PoseGraph2, sparse_direct::*, *};
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;
#[global_allocator]
static ALLOCATOR: allocation::Counter = allocation::Counter;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        return Err("profile_ldl requires release build".into());
    }
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: profile_ldl input.g2o")?;
    let graph = PoseGraph2::from_reader(BufReader::new(File::open(path)?))?
        .build(None)?
        .graph;
    let (_, j) = graph.evaluate_sparse(&graph.initial())?;
    let mut workspace = SparseLdlWorkspace::new(Default::default())?;
    let start = Instant::now();
    workspace.refactor(&j, 1e-3)?;
    let initial_ns = start.elapsed().as_nanos();
    let b = DVector::from_fn(j.shape().1, |k, _| (k as f64 * 0.13).cos());
    let mut x = DVector::zeros(b.len());
    allocation::start();
    let start = Instant::now();
    let result = (|| -> Result<(), Error> {
        for trial in 0..100 {
            workspace.refactor(&j, if trial % 2 == 0 { 1e-3 } else { 2e-3 })?;
            workspace.solve_factored_into(&b, &mut x)?;
        }
        Ok(())
    })();
    let repeated_ns = start.elapsed().as_nanos();
    let (calls, bytes) = allocation::stop();
    result?;
    if (calls, bytes) != (0, 0) || workspace.analyses() != 1 {
        return Err("prepared numeric reuse unexpectedly allocated or reanalyzed".into());
    }
    // Independent original-operator residual check outside the allocation counter.
    let stats = workspace.solve_into(&j, &b, 2e-3, &mut x)?;
    println!(
        "columns,jacobian_entries,normal_lower_entries,factor_entries,assembly_plan_entries,update_plan_entries,initial_analysis_factor_ns,prepared_100_refactor_solve_ns,allocation_calls,requested_bytes,symbolic_analyses,checked_residual_norm"
    );
    println!(
        "{},{},{},{},{},{},{initial_ns},{repeated_ns},{calls},{bytes},{},{:.17e}",
        j.shape().1,
        j.nnz(),
        stats.normal_entries,
        stats.factor_entries,
        workspace.symbolic().unwrap().assembly_entries(),
        workspace.symbolic().unwrap().update_entries(),
        workspace.analyses(),
        stats.residual_norm
    );
    Ok(())
}
