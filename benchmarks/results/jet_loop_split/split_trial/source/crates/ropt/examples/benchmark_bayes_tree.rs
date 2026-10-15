//! Matched Gaussian append/solve/cost timings; independent original-model checks.
use ropt::{bayes_tree::*, incremental_qr::*, sparse::CsrMatrix, *};
use std::{fmt::Write, fs, path::PathBuf, time::Instant};
#[derive(Clone)]
enum Backend {
    Qr(IncrementalQr),
    Tree(GaussianBayesTree),
}
struct Output {
    updates: String,
    means: String,
    total: u128,
}
fn run(
    mode: &str,
    branches: usize,
    leaves: usize,
    updates: usize,
) -> Result<Output, Box<dyn std::error::Error>> {
    let width = leaves.checked_add(1).ok_or("dimensions overflow")?;
    let n = branches.checked_mul(width).ok_or("dimensions overflow")?;
    let mut triplets = Vec::new();
    let mut raw = Vec::new();
    for k in 0..n {
        triplets.push((k, k, 2.));
        if k % width < leaves {
            triplets.push((k, k / width * width + leaves, -1.));
        }
        raw.push((k as f64 * 0.17).sin());
    }
    let j = CsrMatrix::from_triplets(n, n, triplets.clone())?;
    let mut qr = IncrementalQr::new(n, Default::default())?;
    qr.append(&j, &DVector::from_vec(raw.clone()))?;
    let mut backend = match mode {
        "qr" => Backend::Qr(qr),
        "tree" => Backend::Tree(GaussianBayesTree::from_qr(qr)?),
        _ => return Err("mode must be qr or tree".into()),
    };
    let mut result=Output {updates:"frame,elapsed_ns,cost,original_cost,gradient_inf,reeliminated_cliques,reused_cliques,input_rows,entries\n".into(),means:"frame,column,value\n".into(),total:0};
    for frame in 0..updates {
        let branch = frame % branches;
        let leaf = (frame * 17) % leaves;
        let column = branch * width + leaf;
        let mut added = vec![(0, column, 1.)];
        let mut r = vec![(frame as f64 * 0.23).cos() * 0.3, 0.02];
        if frame % 13 == 0 && branches > 1 {
            added.push((2, column, 0.7));
            added.push((2, ((branch + 1) % branches) * width + leaf, -0.7));
            r.push(0.1);
        }
        let j = CsrMatrix::from_triplets(r.len(), n, added.clone())?;
        let residual = DVector::from_vec(r.clone());
        let start = Instant::now();
        let (x, cost, work, entries) = match &mut backend {
            Backend::Qr(qr) => {
                qr.append(&j, &residual)?;
                let x = qr.solve()?;
                let cost = qr.cost(&x)?;
                (x, cost, BayesTreeUpdate::default(), qr.stored_entries())
            }
            Backend::Tree(tree) => {
                let work = tree.append(&j, &residual)?;
                let x = tree.solve()?;
                let cost = tree.cost(&x)?;
                (x, cost, work, tree.stored_entries())
            }
        };
        let elapsed = start.elapsed().as_nanos();
        result.total += elapsed;
        let old_rows = raw.len();
        triplets.extend(
            added
                .into_iter()
                .map(|(row, col, value)| (old_rows + row, col, value)),
        );
        raw.extend(r);
        let original = CsrMatrix::from_triplets(raw.len(), n, triplets.clone())?;
        let residual = original.multiply(&x)? + DVector::from_vec(raw.clone());
        let original_cost = 0.5 * residual.norm_squared();
        let gradient = original.transpose_multiply(&residual)?.amax();
        if !gradient.is_finite()
            || gradient > 1e-8
            || (cost - original_cost).abs() > 1e-10 * original_cost.max(1.)
        {
            return Err(format!(
                "quality failed at {frame}: gradient={gradient:e}, cost gap={:e}",
                (cost - original_cost).abs()
            )
            .into());
        }
        writeln!(
            result.updates,
            "{},{elapsed},{cost:.17e},{original_cost:.17e},{gradient:.17e},{},{},{},{entries}",
            frame + 1,
            work.reeliminated_cliques,
            work.reused_cliques,
            work.input_rows
        )?;
        for (column, value) in x.iter().enumerate() {
            writeln!(result.means, "{},{column},{value:.17e}", frame + 1)?;
        }
    }
    Ok(result)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 5 {
        return Err(
            "usage: benchmark_bayes_tree qr|tree branches leaves updates NEW_DIRECTORY".into(),
        );
    }
    let branches = args[1].parse::<usize>()?;
    let leaves = args[2].parse::<usize>()?;
    let updates = args[3].parse::<usize>()?;
    if branches == 0
        || leaves == 0
        || updates == 0
        || branches > 64
        || leaves > 1024
        || updates > 10000
    {
        return Err("invalid workload bounds".into());
    }
    let out = PathBuf::from(&args[4]);
    fs::create_dir(&out)?;
    // One complete untimed warm-up with every original-model check.
    run(&args[0], branches, leaves, updates)?;
    let result = run(&args[0], branches, leaves, updates)?;
    fs::write(out.join("updates.csv"), result.updates)?;
    fs::write(out.join("means.csv"), result.means)?;
    fs::write(
        out.join("samples.csv"),
        format!("elapsed_ns,updates\n{},{}\n", result.total, updates),
    )?;
    fs::write(
        out.join("options.txt"),
        format!(
            "mode={}\nbranches={branches}\nleaves={leaves}\nupdates={updates}\nwarmup_streams=1\nmeasured_streams=1\ngradient_gate=1e-8\ncost_gate=1e-10*max(1,cost)\nrank_tolerance=1e-12\nmax_entries=1000000\ntiming=public append clone + full solve + cached cost; original-model checks and CSV formatting excluded\n",
            args[0]
        ),
    )?;
    Ok(())
}
