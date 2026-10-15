//! Separate allocation-traffic diagnostic; no latency or peak-memory claim.
#[path = "support/allocation_counter.rs"]
mod allocation;
use ropt::{DVector, bayes_tree::*, sparse::CsrMatrix};
use std::hint::black_box;

#[global_allocator]
static ALLOCATOR: allocation::Counter = allocation::Counter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        return Err("usage: profile_editable_clone columns copies".into());
    }
    let columns: usize = args[0].parse()?;
    let copies: usize = args[1].parse()?;
    if columns == 0 || copies == 0 {
        return Err("positive dimensions and copies required".into());
    }
    let mut tree = EditableBayesTree::new(columns, Default::default())?;
    let factors = (0..columns)
        .map(|column| {
            Ok((
                u64::try_from(column)?,
                GaussianFactor::new(
                    CsrMatrix::from_triplets(1, columns, vec![(0, column, 1.)])?,
                    DVector::from_element(1, -(column as f64) * 0.125),
                )?,
            ))
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    tree.update(&[], factors)?;
    let expected = DVector::from_iterator(columns, (0..columns).map(|c| c as f64 * 0.125));
    if tree.solve()? != expected || tree.cost(&expected)? != 0. {
        return Err("independent prior means or energy differ".into());
    }
    for _ in 0..8 {
        black_box(tree.clone());
    }
    // Include clone construction and destruction; exclude original model
    // creation, validation, warmup and serialization. Deallocations do not
    // contribute to this counter of successful alloc/realloc requests.
    allocation::start();
    for _ in 0..copies {
        let snapshot = black_box(tree.clone());
        assert_eq!(black_box(snapshot.dimension()), columns);
        assert_eq!(black_box(snapshot.factor_count()), columns);
    }
    let (calls, bytes) = allocation::stop();
    if tree.solve()? != expected {
        return Err("original means changed after cloning".into());
    }
    println!(
        "{{\"columns\":{columns},\"copies\":{copies},\"allocation_calls\":{calls},\"requested_bytes\":{bytes},\"independent_means_pass\":true}}"
    );
    Ok(())
}
