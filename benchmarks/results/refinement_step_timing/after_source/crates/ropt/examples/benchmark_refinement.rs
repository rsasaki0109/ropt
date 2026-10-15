//! Requested allocation traffic for complete nonlinear refreshes, not peak memory.
#[path = "support/allocation_counter.rs"]
mod allocation;
#[global_allocator]
static ALLOCATOR: allocation::Counter = allocation::Counter;
use ropt::{factor::*, fixed_lag::*, isam::*, *};
use std::{cell::Cell, hint::black_box, rc::Rc};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if cfg!(debug_assertions) || args.len() != 4 {
        return Err(
            "release usage: benchmark_refinement qr|tree|editable blocks cap repeats".into(),
        );
    }
    let blocks: usize = args[1].parse()?;
    let cap: usize = args[2].parse()?;
    let repeats: usize = args[3].parse()?;
    if !["qr", "tree", "editable"].contains(&args[0].as_str())
        || !(1..=512).contains(&blocks)
        || ![1, 4].contains(&cap)
        || !(1..=100).contains(&repeats)
    {
        return Err("invalid diagnostic bounds".into());
    }
    let calls = Rc::new(Cell::new(0usize));
    let mut allocations = 0;
    let mut bytes = 0;
    let mut fingerprint = None;
    let mut expected_steps = None;
    let mut expected_calls = None;
    for _ in 0..repeats {
        let options = IsamOptions {
            relinearization_interval: usize::MAX,
            relinearization_threshold: 100.,
            batch_solver: Options {
                linear_solver: LinearSolver::SparseLdl(SparseLdlOptions {
                    ordering: SparseOrdering::MinimumDegree,
                    ..Default::default()
                }),
                gradient_tolerance: 1e-3,
                cost_tolerance: 1e-5,
                initial_damping: 1e-2,
                ..Default::default()
            },
            ..Default::default()
        };
        let smoother = IsamSmoother::new(options)?.with_final_refinement(cap)?;
        let mut smoother = match args[0].as_str() {
            "tree" => smoother.with_bayes_tree()?,
            "editable" => smoother.with_selective_bayes_tree(1)?,
            _ => smoother,
        };
        let mut update = SmootherUpdate::default();
        for key in 0..blocks {
            update.variables.push(WindowVariable {
                key: key as u64,
                value: DVector::from_element(1, 1.5),
                kind: VariableKind::Euclidean(1),
                timestamp: None,
                fixed: false,
            });
            let calls = calls.clone();
            update.factors.push(SmootherFactor::analytic(
                &[key as u64],
                1,
                Default::default(),
                move |v| {
                    calls.set(calls.get() + 1);
                    let x = v[0][0];
                    Ok((
                        DVector::from_element(1, x * x - 4.),
                        vec![DMatrix::from_element(1, 1, 2. * x)],
                    ))
                },
            ));
        }
        calls.set(0);
        allocation::start();
        let result = smoother.update(update, true);
        let (count, requested) = allocation::stop();
        let report = black_box(result?);
        allocations += count;
        bytes += requested;
        assert!(report.refinement_steps > 0 && report.refinement_steps <= cap);
        let callback_count = calls.get();
        if let Some(previous) = expected_calls {
            assert_eq!(previous, callback_count);
        }
        expected_calls = Some(callback_count);
        if let Some(previous) = expected_steps {
            assert_eq!(previous, report.refinement_steps);
        }
        expected_steps = Some(report.refinement_steps);
        let probe = smoother.nonlinear_summary(GradientNorm::Absolute)?;
        assert!(probe.gradient_norm <= 1e-3);
        let cached = smoother.update(Default::default(), false)?;
        assert_eq!(cached.kind, IsamUpdateKind::Incremental);
        let mut hash = 0xcbf29ce484222325u64;
        for value in report.estimates.values().chain(cached.estimates.values()) {
            assert!((value[0] - 2.).abs() < 1e-6);
            hash = (hash ^ value[0].to_bits()).wrapping_mul(0x100000001b3);
        }
        for value in [
            report.linearized_cost,
            cached.linearized_cost,
            probe.cost,
            probe.gradient_norm,
        ] {
            hash = (hash ^ value.to_bits()).wrapping_mul(0x100000001b3);
        }
        if let Some(previous) = fingerprint {
            assert_eq!(previous, hash);
        }
        fingerprint = Some(hash);
    }
    println!(
        "backend,blocks,cap,repeats,allocations,requested_bytes,callbacks_per_refresh,refinement_steps,fingerprint"
    );
    println!(
        "{},{blocks},{cap},{repeats},{allocations},{bytes},{},{},{:016x}",
        args[0],
        expected_calls.unwrap(),
        expected_steps.unwrap(),
        fingerprint.unwrap()
    );
    Ok(())
}
