//! Anchored 3-pose SE(2) chain. Explicit central differences are for this small demo.
use nalgebra::Vector3;
use ropt::{manifold::SE2, *};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let measured = SE2::exp(Vector3::new(1.0, 0.0, 0.1));
    let origin = SE2::exp(Vector3::zeros());
    let p = CentralDifference {
        parameters: 6,
        residuals: 6,
        step: 1e-6,
        function: move |x: &DVector<f64>| {
            let a = SE2::exp(Vector3::new(x[0], x[1], x[2]));
            let b = SE2::exp(Vector3::new(x[3], x[4], x[5]));
            let e1 = measured.local(origin.inverse().compose(a));
            let e2 = measured.local(a.inverse().compose(b));
            Ok(DVector::from_iterator(
                6,
                e1.iter().chain(e2.iter()).copied(),
            ))
        },
    };
    let result = solve(
        &p,
        DVector::from_vec(vec![0.8, 0.1, 0.05, 1.8, -0.1, 0.15]),
        &Options::default(),
    )?;
    println!(
        "anchored SE2 chain cost={} termination={:?}",
        result.cost, result.termination
    );
    if result.cost > 1e-12 {
        return Err("pose chain did not converge".into());
    }
    Ok(())
}
