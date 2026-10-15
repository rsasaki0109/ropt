//! Diagnostic phase timers and separate allocation counts; not benchmark timings.
#[path = "support/allocation_counter.rs"]
mod allocation;
use ropt::{g2o::PoseGraph2, sparse::*, *};
use std::cell::Cell;
use std::fs::File;
use std::io::BufReader;
use std::ops::Range;
use std::time::Instant;
#[global_allocator]
static ALLOCATOR: allocation::Counter = allocation::Counter;
#[derive(Default, Clone, Copy)]
struct Phases {
    evaluations: usize,
    evaluation_ns: u128,
    retractions: usize,
    retraction_ns: u128,
}
struct Measured<'a, P> {
    problem: &'a P,
    phases: Cell<Phases>,
}
impl<P: LocalProblem> LocalProblem for Measured<'_, P> {
    fn dimensions(&self) -> (usize, usize, usize) {
        self.problem.dimensions()
    }
    fn tangent_blocks(&self) -> Result<Vec<Range<usize>>, Error> {
        self.problem.tangent_blocks()
    }
    fn loss_blocks(&self) -> Vec<LossBlock> {
        self.problem.loss_blocks()
    }
    fn step_scale(&self, x: &DVector<f64>) -> f64 {
        self.problem.step_scale(x)
    }
    fn evaluate_local(
        &self,
        x: &DVector<f64>,
        r: &mut DVector<f64>,
        j: &mut DMatrix<f64>,
    ) -> Result<(), Error> {
        let start = Instant::now();
        let result = self.problem.evaluate_local(x, r, j);
        let mut phases = self.phases.get();
        phases.evaluations += 1;
        phases.evaluation_ns += start.elapsed().as_nanos();
        self.phases.set(phases);
        result
    }
    fn evaluate_sparse(&self, x: &DVector<f64>) -> Result<(DVector<f64>, CsrMatrix), Error> {
        let start = Instant::now();
        let result = self.problem.evaluate_sparse(x);
        let mut phases = self.phases.get();
        phases.evaluations += 1;
        phases.evaluation_ns += start.elapsed().as_nanos();
        self.phases.set(phases);
        result
    }
    fn retract(&self, x: &DVector<f64>, delta: &DVector<f64>) -> Result<DVector<f64>, Error> {
        let start = Instant::now();
        let result = self.problem.retract(x, delta);
        let mut phases = self.phases.get();
        phases.retractions += 1;
        phases.retraction_ns += start.elapsed().as_nanos();
        self.phases.set(phases);
        result
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        return Err("profiling requires a release build".into());
    }
    let input = std::env::args()
        .nth(1)
        .ok_or("usage: profile_pose_graph input.g2o")?;
    let data = PoseGraph2::from_reader(BufReader::new(File::open(input)?))?;
    let loaded = data.build(None)?;
    // Independently check that built-in repeated operator/preconditioner calls
    // make zero allocation requests after their inputs and factors are prepared.
    let (_, j) = loaded.graph.evaluate_sparse(&loaded.graph.initial())?;
    let operator = NormalOperator::new(&j, 1e-3)?;
    let scalar = ropt::preconditioner::Jacobi::new(operator.diagonal()?)?;
    let block =
        ropt::preconditioner::BlockJacobi::from_normal(&j, 1e-3, &loaded.graph.tangent_blocks()?)?;
    let input = DVector::from_element(operator.dimension(), 0.1);
    let mut output = DVector::zeros(operator.dimension());
    allocation::start();
    let hot_loop = (|| -> Result<(), Error> {
        use ropt::preconditioner::Preconditioner;
        for _ in 0..100 {
            operator.apply_into(std::hint::black_box(&input), &mut output)?;
            scalar.apply_into(std::hint::black_box(&input), &mut output)?;
            block.apply_into(std::hint::black_box(&input), &mut output)?;
        }
        Ok(())
    })();
    let hot_allocations = allocation::stop();
    hot_loop?;
    if hot_allocations != (0, 0) {
        return Err("built-in buffered hot loops unexpectedly allocated".into());
    }
    eprintln!(
        "Verified zero allocation requests for 100 prepared normal/Jacobi/block-Jacobi applications"
    );
    let cg = CgOptions {
        max_iterations: 10000,
        relative_tolerance: 1e-10,
        absolute_tolerance: 1e-12,
    };
    println!(
        "backend,total_ns,evaluation_ns,retraction_ns,evaluations,retractions,allocation_calls,requested_bytes,cost,nonlinear_trials,linear_iterations,termination"
    );
    for (name, linear_solver) in [
        ("scalar_cg", LinearSolver::SparseCg(cg)),
        ("block_cg", LinearSolver::SparseBlockCg(cg)),
        (
            "sparse_ldl",
            LinearSolver::SparseLdl(SparseLdlOptions::default()),
        ),
    ] {
        let problem = Measured {
            problem: &loaded.graph,
            phases: Cell::default(),
        };
        let options = Options {
            linear_solver,
            ..Options::default()
        };
        let initial = loaded.graph.initial();
        let start = Instant::now();
        let report = solve_local(&problem, initial, &options)?;
        let elapsed = start.elapsed().as_nanos();
        let phases = problem.phases.get();
        // A separate solve counts traffic; its instrumented timing is not reported.
        let initial = loaded.graph.initial();
        allocation::start();
        let counted = solve_local(&loaded.graph, initial, &options);
        let (calls, bytes) = allocation::stop();
        let counted = counted?;
        if (&report.x - &counted.x).norm() > 1e-8
            || (report.cost - counted.cost).abs() > 1e-10
            || matches!(
                report.termination,
                Termination::NoProgress | Termination::MaxIterations
            )
        {
            return Err("diagnostic solves failed consistency/convergence".into());
        }
        println!(
            "{name},{elapsed},{},{},{},{},{calls},{bytes},{:.17e},{},{},{:?}",
            phases.evaluation_ns,
            phases.retraction_ns,
            phases.evaluations,
            phases.retractions,
            report.cost,
            report.iterations.len(),
            report
                .iterations
                .iter()
                .map(|i| i.linear_iterations)
                .sum::<usize>(),
            report.termination
        );
    }
    Ok(())
}
