//! Allocation diagnostic for forced Gaussian refreshes through the public API.
//! Counts requested traffic, not live/peak memory; instrumented timers are diagnostic.
#[path = "support/allocation_counter.rs"]
mod allocation;
#[global_allocator]
static ALLOCATOR: allocation::Counter = allocation::Counter;
use ropt::{factor::*, fixed_lag::*, isam::*, *};
use std::{cell::Cell, hint::black_box, rc::Rc, time::Instant};

fn hash(state: u64, bits: u64) -> u64 {
    (state ^ bits).wrapping_mul(0x100000001b3)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        return Err("benchmark_factor_cache requires release".into());
    }
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("usage: benchmark_factor_cache blocks residual_rows refreshes".into());
    }
    let blocks: usize = args[0].parse()?;
    let rows: usize = args[1].parse()?;
    let refreshes: usize = args[2].parse()?;
    if !(1..=512).contains(&blocks) || ![1, 3, 16].contains(&rows) || !(1..=64).contains(&refreshes)
    {
        return Err("invalid workload bounds".into());
    }
    let width = rows.min(3);
    let calls = Rc::new(Cell::new(0));
    let mut update = SmootherUpdate::default();
    let mut expected_cost = 0.;
    for key in 0..blocks {
        update.variables.push(WindowVariable {
            key: key as u64,
            kind: VariableKind::Euclidean(width),
            value: DVector::zeros(width),
            timestamp: None,
            fixed: false,
        });
        let mut a = DMatrix::zeros(rows, width);
        for row in 0..width {
            for column in 0..width {
                a[(row, column)] = if row == column { 2. } else { 0.1 };
            }
        }
        let mut constant = DVector::zeros(rows);
        for row in width..rows {
            constant[row] = 0.01 * (row + 1) as f64;
            expected_cost += 0.5 * constant[row] * constant[row];
        }
        let count = calls.clone();
        update.factors.push(SmootherFactor::analytic(
            &[key as u64],
            rows,
            FactorOptions::default(),
            move |values| {
                count.set(count.get() + 1);
                Ok((&a * &values[0] + &constant, vec![a.clone()]))
            },
        ));
    }
    let options = IsamOptions {
        relinearization_threshold: 1e9,
        relinearization_interval: usize::MAX,
        batch_solver: Options {
            algorithm: Algorithm::GaussNewton,
            linear_solver: LinearSolver::SparseQr(Default::default()),
            gradient_norm: GradientNorm::ColumnNormalized,
            gradient_tolerance: 1e-10,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut smoother = IsamSmoother::new(options)?.with_selective_bayes_tree(1)?;
    smoother.update(update, true)?;
    for _ in 0..4 {
        black_box(smoother.update(SmootherUpdate::default(), true)?);
    }
    let mut elapsed = 0;
    let (mut allocations, mut bytes, mut callbacks) = (0, 0, 0);
    let mut fingerprint = None;
    for _ in 0..refreshes {
        let before = calls.get();
        allocation::start();
        let start = Instant::now();
        let result = smoother.update(SmootherUpdate::default(), true);
        elapsed += start.elapsed().as_nanos();
        let (count, requested) = allocation::stop();
        let report = black_box(result?);
        allocations += count;
        bytes += requested;
        let called = calls.get() - before;
        assert_eq!(called, blocks);
        callbacks += called;
        assert_eq!(report.factors_linearized, blocks);
        assert_eq!(report.batch_trials, 0);
        assert_eq!(report.refinement_steps, 0);
        assert_eq!(report.estimates.len(), blocks);
        let summary = smoother.nonlinear_summary(GradientNorm::ColumnNormalized)?;
        assert_eq!(summary.gradient_norm, 0.);
        assert!((summary.cost - expected_cost).abs() <= 1e-12 * expected_cost.max(1.));
        assert!((report.linearized_cost - expected_cost).abs() <= 1e-12 * expected_cost.max(1.));
        let mut estimates = 0xcbf29ce484222325;
        for (&key, value) in &report.estimates {
            assert!(value.iter().all(|&v| v == 0.));
            estimates = hash(estimates, key);
            for value in value {
                estimates = hash(estimates, value.to_bits());
            }
        }
        let current = (
            summary.cost.to_bits(),
            summary.gradient_norm.to_bits(),
            estimates,
            report.linearized_cost.to_bits(),
            report.max_anchor_delta.to_bits(),
            report.stored_entries,
            report.editable_bayes_tree_update.unwrap(),
        );
        if let Some(previous) = fingerprint {
            assert_eq!(current, previous);
        }
        fingerprint = Some(current);
    }
    let (cost, gradient, estimates, cached_cost, delta, entries, work) = fingerprint.unwrap();
    println!(
        "blocks,rows,refreshes,elapsed_ns,allocation_calls,requested_bytes,callback_calls,cost_bits,gradient_bits,estimate_bits,cached_cost_bits,anchor_delta_bits,entries,reeliminated_cliques,input_factor_rows,input_separator_rows"
    );
    println!(
        "{blocks},{rows},{refreshes},{elapsed},{allocations},{bytes},{callbacks},{cost:016x},{gradient:016x},{estimates:016x},{cached_cost:016x},{delta:016x},{entries},{},{},{}",
        work.reeliminated_cliques, work.input_factor_rows, work.input_separator_rows
    );
    Ok(())
}
