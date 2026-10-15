//! Inspect information reconstruction and whitening for a supplied raw residual.
use ropt::{factor::*, noise::NoiseModel, *};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: diagnose_information INPUT.txt")?;
    let v: Vec<f64> = std::fs::read_to_string(path)?
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    if v.len() != 9 {
        return Err("expected six information entries and three residual entries".into());
    }
    let information = DMatrix::from_row_slice(
        3,
        3,
        &[v[0], v[1], v[2], v[1], v[3], v[4], v[2], v[4], v[5]],
    );
    let initial = DVector::from_column_slice(&v[6..]);
    let mut graph = FactorGraph::new();
    let id = graph.add_variable(VariableKind::Euclidean(3), initial.clone())?;
    graph.add_analytic_factor_with_options(
        &[id],
        3,
        FactorOptions {
            noise: NoiseModel::from_information(information.clone())?,
            ..Default::default()
        },
        |v| Ok((v[0].clone(), vec![DMatrix::identity(3, 3)])),
    )?;
    let mut r = DVector::zeros(3);
    let mut w = DMatrix::zeros(3, 3);
    graph.evaluate_local(&initial, &mut r, &mut w)?;
    println!(
        "white_cost={:.17e}\nraw_float_cost={:.17e}",
        0.5 * r.norm_squared(),
        0.5 * initial.dot(&(&information * &initial))
    );
    for row in 0..3 {
        for col in 0..3 {
            println!("w_{row}_{col}={:.17e}", w[(row, col)]);
        }
    }
    let reconstructed = w.transpose() * w;
    for row in 0..3 {
        for col in 0..3 {
            println!(
                "omega_error_{row}_{col}={:.17e}",
                reconstructed[(row, col)] - information[(row, col)]
            );
        }
    }
    Ok(())
}
