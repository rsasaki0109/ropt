//! A one-trial numerical regression, not a timing benchmark or convergence claim.
use ropt::*;
struct AlmostParallel {
    jacobian: DMatrix<f64>,
    truth: DVector<f64>,
}
impl Residual for AlmostParallel {
    fn dimensions(&self) -> (usize, usize) {
        (2, 3)
    }
    fn evaluate(
        &self,
        x: &DVector<f64>,
        r: &mut DVector<f64>,
        j: &mut DMatrix<f64>,
    ) -> Result<(), Error> {
        *r = &self.jacobian * (x - &self.truth);
        *j = self.jacobian.clone();
        Ok(())
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let epsilon = 1e-8;
    let p = AlmostParallel {
        jacobian: DMatrix::from_row_slice(
            3,
            2,
            &[1.0, 1.0, 1.0, 1.0 + epsilon, 1.0, 1.0 - epsilon],
        ),
        truth: DVector::from_vec(vec![1.0, -1.0]),
    };
    let singular = p.jacobian.clone().svd(false, false).singular_values;
    println!(
        "nearly parallel columns: epsilon={epsilon:.1e}, Jacobian condition={:.3e}",
        singular.max() / singular.min()
    );
    println!("one trial, LM lambda=1e-20; stopping tolerances=0 to examine the tiny residual");
    for (name, algorithm, linear_solver) in [
        ("GN SVD", Algorithm::GaussNewton, LinearSolver::Dense),
        (
            "GN pivoted QR",
            Algorithm::GaussNewton,
            LinearSolver::DenseQr(QrOptions::default()),
        ),
        (
            "LM normal Cholesky",
            Algorithm::LevenbergMarquardt,
            LinearSolver::Dense,
        ),
        (
            "LM augmented QR",
            Algorithm::LevenbergMarquardt,
            LinearSolver::DenseQr(QrOptions::default()),
        ),
    ] {
        let result = solve(
            &p,
            DVector::zeros(2),
            &Options {
                algorithm,
                linear_solver,
                initial_damping: 1e-20,
                max_iterations: 1,
                gradient_tolerance: 0.0,
                step_tolerance: 0.0,
                cost_tolerance: 0.0,
                ..Options::default()
            },
        );
        match result {
            Ok(report) => {
                let error = (&report.x - &p.truth).norm();
                println!(
                    "{name}: x={:?}, error={error:.3e}, cost={:.3e}, termination={:?}",
                    report.x.as_slice(),
                    report.cost,
                    report.termination
                );
                if matches!(linear_solver, LinearSolver::DenseQr(_))
                    && (error > 0.01 || report.cost >= report.initial_cost)
                {
                    return Err("QR did not resolve the regression step".into());
                }
            }
            Err(e) => {
                println!("{name}: {e}");
                if !matches!(linear_solver, LinearSolver::Dense)
                    || !matches!(algorithm, Algorithm::LevenbergMarquardt)
                {
                    return Err(e.into());
                }
            }
        }
    }
    Ok(())
}
