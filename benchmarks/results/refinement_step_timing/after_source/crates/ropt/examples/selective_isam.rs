//! Threshold-selected nonlinear factors, followed by an explicit full refresh.
use ropt::{factor::*, fixed_lag::*, isam::*, *};
use std::{cell::Cell, rc::Rc};

fn main() -> Result<(), Error> {
    let calls = Rc::new(Cell::new(0usize));
    let mut smoother = IsamSmoother::new(IsamOptions {
        relinearization_interval: 100,
        batch_solver: Options {
            algorithm: Algorithm::GaussNewton,
            linear_solver: LinearSolver::SparseQr(Default::default()),
            cost_tolerance: 1e-15,
            step_tolerance: 1e-12,
            ..Default::default()
        },
        ..Default::default()
    })?
    .with_selective_relinearization(4)?;
    let mut first = SmootherUpdate::default();
    for key in 0..1024 {
        first.variables.push(WindowVariable {
            key,
            kind: VariableKind::Euclidean(1),
            value: DVector::from_element(1, 1.),
            timestamp: None,
            fixed: false,
        });
        let count = calls.clone();
        first.factors.push(SmootherFactor::autodiff(
            &[key],
            1,
            1,
            Default::default(),
            move |v| {
                count.set(count.get() + 1);
                Ok(vec![v[0][0].powi(2) - 1.])
            },
        ));
    }
    smoother.update(first, false)?;
    calls.set(0);
    let count = calls.clone();
    let report = smoother.update(
        SmootherUpdate {
            factors: vec![SmootherFactor::autodiff(
                &[0],
                1,
                1,
                Default::default(),
                move |v| {
                    count.set(count.get() + 1);
                    Ok(vec![&v[0][0] - 3.])
                },
            )],
            ..Default::default()
        },
        false,
    )?;
    let work = report
        .selective_update
        .as_ref()
        .ok_or(Error::InvalidOptions)?;
    let x = report.estimates[&0][0];
    println!(
        "selected variables: {:?}; retained factors: {}; passes: {}; factor callbacks: {}",
        work.variables_relinearized,
        work.retained_factors,
        work.passes,
        calls.get()
    );
    println!(
        "bounded selected estimate: {x:.10}; threshold remaining: {}; nonlinear gradient: {:.3e}",
        work.threshold_remaining,
        2. * x * (x * x - 1.) + x - 3.
    );
    // Threshold clearance only limits anchor movement. Request a full nonlinear
    // solve when needed, and inspect the stopping criteria and final gradient.
    let full = smoother.update(Default::default(), true)?;
    let x = full.estimates[&0][0];
    println!(
        "full batch estimate: {x:.10}; nonlinear gradient: {:.3e}",
        2. * x * (x * x - 1.) + x - 3.
    );
    Ok(())
}
