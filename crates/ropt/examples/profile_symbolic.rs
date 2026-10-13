//! Allocation diagnostic for fresh symbolic analyses; instrumented times are not benchmarks.
#[path = "support/allocation_counter.rs"]
mod allocation;
use ropt::{LocalProblem, g2o::PoseGraph2, sparse::CsrMatrix, sparse_direct::*};
use std::{fs::File, io::BufReader, time::Instant};
#[global_allocator]
static ALLOCATOR: allocation::Counter = allocation::Counter;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        return Err("profile_symbolic requires release".into());
    }
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        return Err("usage: profile_symbolic input.g2o|reverse-star:N positive_repeats".into());
    }
    let repeats: usize = args[1].parse()?;
    if repeats == 0 {
        return Err("repeats must be positive".into());
    }
    let (j, ordering) = if let Some(size) = args[0].strip_prefix("reverse-star:") {
        let n: usize = size.parse()?;
        if n < 2 {
            return Err("star requires at least two columns".into());
        }
        let mut entries = vec![(n - 1, 0, 1.)];
        for row in 0..n - 1 {
            entries.push((row, 0, -1.));
            entries.push((row, n - 1 - row, 1.));
        }
        (
            CsrMatrix::from_triplets(n, n, entries)?,
            SparseOrdering::MinimumDegree,
        )
    } else {
        let graph = PoseGraph2::from_reader(BufReader::new(File::open(&args[0])?))?
            .build(None)?
            .graph;
        (
            graph.evaluate_sparse(&graph.initial())?.1,
            SparseOrdering::BlockMinimumDegree { block_size: 3 },
        )
    };
    let options = SparseLdlOptions {
        ordering,
        ..Default::default()
    };
    allocation::start();
    let start = Instant::now();
    let result = (|| -> Result<_, ropt::Error> {
        let mut last = None;
        for _ in 0..repeats {
            last = Some(std::hint::black_box(SparseLdlSymbolic::analyze(
                &j, &options,
            )?));
        }
        last.ok_or(ropt::Error::InvalidOptions)
    })();
    let elapsed = start.elapsed().as_nanos();
    let (calls, bytes) = allocation::stop();
    let symbolic = result?;
    println!(
        "columns,jacobian_entries,repeats,elapsed_ns,allocation_calls,requested_bytes,normal_entries,factor_entries,assembly_entries,update_entries"
    );
    println!(
        "{},{},{repeats},{elapsed},{calls},{bytes},{},{},{},{}",
        j.shape().1,
        j.nnz(),
        symbolic.normal_entries(),
        symbolic.factor_entries(),
        symbolic.assembly_entries(),
        symbolic.update_entries()
    );
    println!("permutation={:?}", symbolic.permutation());
    Ok(())
}
