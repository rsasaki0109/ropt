//! Runtime-sized samples and parameter blocks, with chunked AD and no numeric Jacobians.
use ropt::{
    autodiff::{DynamicAutoDiff, DynamicJet},
    factor::*,
    sparse::CgOptions,
    *,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let times: Vec<f64> = (0..40).map(|i| i as f64 * 0.15).collect();
    let observed: Vec<f64> = times.iter().map(|t| 2.0 * (-0.4 * t).exp() + 0.5).collect();
    let p = DynamicAutoDiff::new(3, times.len(), |x: &[DynamicJet]| {
        Ok(times
            .iter()
            .zip(&observed)
            .map(|(t, y)| &x[0] * (-&x[1] * *t).exp() + &x[2] - *y)
            .collect())
    })?
    .with_chunk_size(2)?;
    let initial = DVector::from_vec(vec![1.4, 0.25, 0.2]);
    let report = solve(&p, initial.clone(), &Options::default())?;
    println!(
        "Dynamic AD: a={:.8}, rate={:.8}, bias={:.8}, cost={:.3e}",
        report.x[0], report.x[1], report.x[2], report.cost
    );
    let mut graph = FactorGraph::new();
    let a = graph.add_variable(
        VariableKind::Euclidean(1),
        DVector::from_element(1, initial[0]),
    )?;
    let rate = graph.add_variable(
        VariableKind::Euclidean(1),
        DVector::from_element(1, initial[1]),
    )?;
    let bias = graph.add_variable(
        VariableKind::Euclidean(1),
        DVector::from_element(1, initial[2]),
    )?;
    for (&t, &y) in times.iter().zip(&observed) {
        graph.add_autodiff_factor(&[a, rate, bias], 1, move |b| {
            Ok(vec![&b[0][0] * (-&b[1][0] * t).exp() + &b[2][0] - y])
        })?;
    }
    let (retained, block_size) = graph.schur_layout(&[bias])?;
    let graph_report = solve_local(
        &graph,
        graph.initial(),
        &Options {
            linear_solver: LinearSolver::BlockSchur {
                retained,
                block_size,
                cg: CgOptions::default(),
            },
            ..Options::default()
        },
    )?;
    println!("AD factors + block Schur: cost={:.3e}", graph_report.cost);
    let truth = DVector::from_vec(vec![2.0, 0.4, 0.5]);
    if (&report.x - &truth).norm() > 1e-6 || (&graph_report.x - &truth).norm() > 1e-6 {
        return Err("dynamic AD fit did not recover the synthetic parameters".into());
    }
    Ok(())
}
