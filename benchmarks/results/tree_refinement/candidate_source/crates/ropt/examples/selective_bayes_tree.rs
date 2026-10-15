//! Synthetic nonlinear star with retained informative separator messages.
use ropt::{factor::*, fixed_lag::*, isam::*, *};
use std::{cell::Cell, rc::Rc};
fn main() -> Result<(), Error> {
    let leaves = 1024_u64;
    let root = 100000_u64;
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
    .with_selective_bayes_tree(4)?;
    let mut initial = SmootherUpdate::default();
    for key in 0..leaves {
        initial.variables.push(WindowVariable {
            key,
            kind: VariableKind::Euclidean(1),
            value: DVector::from_element(1, 1.),
            timestamp: None,
            fixed: false,
        });
        let count = calls.clone();
        initial.factors.push(SmootherFactor::autodiff(
            &[key],
            1,
            1,
            Default::default(),
            move |v| {
                count.set(count.get() + 1);
                Ok(vec![v[0][0].powi(2) - 1.])
            },
        ));
        let count = calls.clone();
        initial.factors.push(SmootherFactor::autodiff(
            &[key, root],
            1,
            1,
            Default::default(),
            move |v| {
                count.set(count.get() + 1);
                Ok(vec![&v[0][0] - &v[1][0] - 1.])
            },
        ));
    }
    initial.variables.push(WindowVariable {
        key: root,
        kind: VariableKind::Euclidean(1),
        value: DVector::zeros(1),
        timestamp: None,
        fixed: false,
    });
    initial.factors.push(SmootherFactor::autodiff(
        &[root],
        1,
        1,
        Default::default(),
        |v| Ok(vec![1000. * &v[0][0]]),
    ));
    smoother.update(initial, false)?;
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
    let nonlinear = report
        .selective_update
        .as_ref()
        .ok_or(Error::InvalidOptions)?;
    let gaussian = report
        .editable_bayes_tree_update
        .ok_or(Error::InvalidOptions)?;
    println!(
        "selected keys: {:?}; nonlinear factors retained: {}; passes: {}; callbacks: {}",
        nonlinear.variables_relinearized,
        nonlinear.retained_factors,
        nonlinear.passes,
        calls.get()
    );
    println!(
        "Gaussian cliques: {} re-eliminated, {} reused; {} boundary messages reused",
        gaussian.reeliminated_cliques, gaussian.reused_cliques, gaussian.reused_boundary_messages
    );
    let center = report.estimates[&root][0];
    let mut original =
        0.5 * (1000. * center).powi(2) + 0.5 * (report.estimates[&0][0] - 3.).powi(2);
    let mut gradient = 0_f64;
    let mut root_gradient = 1e6 * center;
    for key in 0..leaves {
        let x = report.estimates[&key][0];
        let between = x - center - 1.;
        original += 0.5 * ((x * x - 1.).powi(2) + between * between);
        root_gradient -= between;
        let g = 2. * x * (x * x - 1.) + between + if key == 0 { x - 3. } else { 0. };
        gradient = gradient.max(g.abs());
    }
    gradient = gradient.max(root_gradient.abs());
    println!(
        "cached Gaussian cost: {:.10}; original nonlinear cost: {original:.10}; original gradient inf: {gradient:.3e}",
        report.linearized_cost
    );
    println!(
        "threshold remaining: {}; symbolic analysis/back-substitution remain global",
        nonlinear.threshold_remaining
    );
    Ok(())
}
