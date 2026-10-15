//! Deterministic actual LM checkpoints for the README animation; no interpolation.
use ropt::{
    autodiff::{AutoDiff, Jet},
    *,
};
use std::{
    fs::File,
    io::{BufWriter, Write},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: export_rosenbrock_trace OUTPUT.csv")?;
    let residual = AutoDiff::<2, 2, _>(|x: [Jet<2>; 2]| {
        [
            Jet::constant(10.) * (x[1] - x[0] * x[0]),
            Jet::constant(1.) - x[0],
        ]
    });
    let initial = DVector::from_vec(vec![-1.2, 1.]);
    let options = Options {
        max_iterations: 200,
        ..Default::default()
    };
    let reference = solve(&residual, initial.clone(), &options)?;
    if reference.cost > 1e-12 || (&reference.x - DVector::from_element(2, 1.)).norm() > 1e-6 {
        return Err("Rosenbrock trace did not converge".into());
    }
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "iteration,x,y,cost,accepted,gain_ratio,damping")?;
    let initial_cost = reference
        .iterations
        .first()
        .ok_or("missing initial trial")?
        .cost;
    writeln!(
        out,
        "0,-1.2,1.0,{initial_cost:.17e},true,0,{:.17e}",
        options.initial_damping
    )?;
    for (index, trial) in reference.iterations.iter().enumerate() {
        let state = solve(
            &residual,
            initial.clone(),
            &Options {
                max_iterations: index + 1,
                ..options.clone()
            },
        )?;
        let expected = if trial.accepted {
            trial.trial_cost
        } else {
            trial.cost
        };
        if (state.cost - expected).abs() > 1e-12 * expected.abs().max(1.) {
            return Err("checkpoint differs from full LM solve".into());
        }
        writeln!(
            out,
            "{},{:.17e},{:.17e},{:.17e},{},{:.17e},{:.17e}",
            index + 1,
            state.x[0],
            state.x[1],
            state.cost,
            trial.accepted,
            trial.gain_ratio,
            trial.damping
        )?;
    }
    out.flush()?;
    println!(
        "{} actual LM trials; final cost {:.3e}",
        reference.iterations.len(),
        reference.cost
    );
    Ok(())
}
