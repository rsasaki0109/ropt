//! SE(3) Lie Log graph with forward AD local Jacobians and a fixed initial pose.
use ropt::{factor::*, manifold::*, pose_graph::add_relative_pose3_factor, *};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let truth = [
        SE3::exp(Tangent6::zeros()),
        SE3::exp(Tangent6::new(1.0, 0.1, 0.0, 0.0, 0.0, 0.2)),
        SE3::exp(Tangent6::new(2.0, 0.2, 0.1, 0.1, 0.0, 0.4)),
    ];
    let mut graph = FactorGraph::new();
    let mut ids = Vec::new();
    for (i, p) in truth.iter().enumerate() {
        let initial = if i == 0 {
            *p
        } else {
            p.retract(Tangent6::new(0.2, -0.1, 0.1, 0.05, 0.1, -0.1))
        };
        ids.push(graph.add_variable(VariableKind::SE3, se3_values(initial))?);
    }
    graph.set_fixed(ids[0], true)?;
    for (a, b) in [(0, 1), (1, 2), (0, 2)] {
        let measurement = truth[a].inverse().compose(truth[b]);
        add_relative_pose3_factor(
            &mut graph,
            ids[a],
            ids[b],
            measurement,
            FactorOptions::default(),
        )?;
    }
    for (name, linear_solver) in [
        ("dense", LinearSolver::Dense),
        (
            "sparse PCG",
            LinearSolver::SparseCg(ropt::sparse::CgOptions::default()),
        ),
    ] {
        let report = solve_local(
            &graph,
            graph.initial(),
            &Options {
                linear_solver,
                ..Options::default()
            },
        )?;
        println!(
            "SE3 graph ({name}): cost={} termination={:?} iterations={} linear_iterations={}",
            report.cost,
            report.termination,
            report.iterations.len(),
            report
                .iterations
                .iter()
                .map(|i| i.linear_iterations)
                .sum::<usize>()
        );
        if report.cost > 1e-12 {
            return Err("SE3 graph did not converge".into());
        }
    }
    Ok(())
}
