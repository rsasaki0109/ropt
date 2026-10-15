//! Separate explicit sparse-QR/nonmonotonic-GN replay; historical LM policy stays intact.
#[path = "support/real_se2_benchmark.rs"]
mod benchmark;
#[global_allocator]
static ALLOCATOR: benchmark::Counter = benchmark::Counter;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    benchmark::run_with_chart(
        "cartesian_se2;ambient_se2;sparse_qr;nonmonotonic_gauss_newton;reuse_batch_ordering;final_refinement=4;frozen_qr",
        |options| {
            ropt::isam::IsamSmoother::new(options)?
                .with_batch_ordering_reuse()?
                .with_final_refinement(4)
        },
        |_| None,
        Some(ropt::Options {
            algorithm: ropt::Algorithm::GaussNewton,
            nonmonotonic_gauss_newton: true,
            linear_solver: ropt::LinearSolver::SparseQr(ropt::SparseQrOptions {
                ordering: ropt::SparseOrdering::BlockMinimumDegree { block_size: 3 },
                ..Default::default()
            }),
            gradient_norm: ropt::GradientNorm::ColumnNormalized,
            gradient_tolerance: 8e-8,
            step_tolerance: 1e-12,
            cost_tolerance: 1e-15,
            max_iterations: 200,
            ..Default::default()
        }),
        |relative, a, b| relative.linearize_ambient(a.as_slice(), b.as_slice()),
        ropt::factor::VariableKind::SE2Cartesian,
        true,
    )
}
