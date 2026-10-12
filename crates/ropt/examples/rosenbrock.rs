use ropt::{
    autodiff::{AutoDiff, Jet},
    *,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let residual = AutoDiff::<2, 2, _>(|x: [Jet<2>; 2]| {
        [
            Jet::constant(10.0) * (x[1] - x[0] * x[0]),
            Jet::constant(1.0) - x[0],
        ]
    });
    let report = solve(
        &residual,
        DVector::from_vec(vec![-1.2, 1.0]),
        &Options {
            max_iterations: 200,
            ..Options::default()
        },
    )?;
    println!(
        "x={} cost={} termination={:?} iterations={}",
        report.x.transpose(),
        report.cost,
        report.termination,
        report.iterations.len()
    );
    if report.cost > 1e-12 {
        return Err("Rosenbrock did not converge".into());
    }
    Ok(())
}
