//! Release update timings on a synthetic linear correlated Gaussian stream.
//! Fixture/oracle construction, recipe creation, quality checks and output are untimed.
#[path = "support/isam_stream.rs"]
mod fixture;
use ropt::{fixed_lag::*, isam::*, *};
use std::{
    cell::Cell,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    rc::Rc,
    time::Instant,
};
const LIMIT: usize = 32_000_000;
enum StreamReport {
    Incremental(IsamReport),
    Batch(FixedLagReport),
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        return Err("benchmark_isam requires release".into());
    }
    let mut args = std::env::args().skip(1);
    let mode = args
        .next()
        .ok_or("usage: benchmark_isam incremental|batch frames NEW_DIRECTORY repeats warmup")?;
    let frames: usize = args.next().ok_or("missing frames")?.parse()?;
    let out = PathBuf::from(args.next().ok_or("missing output")?);
    let repeats: usize = args.next().ok_or("missing repeats")?.parse()?;
    let warmup: usize = args.next().ok_or("missing warmup")?.parse()?;
    if args.next().is_some()
        || !matches!(mode.as_str(), "incremental" | "batch")
        || !(1..=1000).contains(&frames)
        || repeats == 0
        || warmup.checked_add(repeats).is_none()
    {
        return Err("invalid arguments".into());
    }
    fs::create_dir(&out)?;
    let references = (1..=frames).map(fixture::oracle).collect::<Vec<_>>();
    let solver = Options {
        linear_solver: LinearSolver::SparseLdl(SparseLdlOptions::default()),
        // Constant Gaussian Jacobians: use an almost undamped batch step,
        // rather than charge the baseline for several default LM trials.
        initial_damping: 1e-12,
        cost_tolerance: 1e-15,
        ..Default::default()
    };
    let isam_options = IsamOptions {
        relinearization_threshold: 1e9,
        relinearization_interval: usize::MAX,
        qr: ropt::incremental_qr::IncrementalQrOptions {
            max_entries: LIMIT,
            ..Default::default()
        },
        batch_solver: solver.clone(),
    };
    let batch_options = FixedLagOptions {
        solver,
        marginalization: ropt::marginalization::MarginalizationOptions {
            max_entries: LIMIT,
            ..Default::default()
        },
        ..Default::default()
    };
    fs::write(
        out.join("options.txt"),
        format!(
            "mode={mode}\nframes={frames}\nisam={isam_options:?}\nbatch={batch_options:?}\nall timestamps=None; batch retains all history; no expiry\ncallback counting enabled; allocator counting disabled\n"
        ),
    )?;
    let mut samples = BufWriter::new(File::create(out.join("samples.csv"))?);
    writeln!(
        samples,
        "sample,elapsed_ns,updates,callbacks,final_cost,max_pose_delta,max_gradient,final_qr_entries"
    )?;
    let mut updates = BufWriter::new(File::create(out.join("updates.csv"))?);
    writeln!(
        updates,
        "sample,frame,elapsed_ns,late_loop,kind,callbacks,cost,reference_cost,gradient_inf,max_pose_delta,qr_entries"
    )?;
    for round in 0..warmup + repeats {
        let calls = Rc::new(Cell::new(0));
        let mut isam = IsamSmoother::new(isam_options.clone())?;
        let mut batch = FixedLagSmoother::new(batch_options.clone())?;
        if mode == "incremental" {
            isam.update(fixture::update(0, calls.clone()), false)?;
        } else {
            batch.update(0., fixture::update(0, calls.clone()))?;
        }
        calls.set(0);
        let (mut elapsed, mut max_delta, mut max_gradient) = (0_u128, 0_f64, 0_f64);
        let (mut last_cost, mut last_entries) = (0., 0);
        for frame in 1..=frames {
            let update = fixture::update(frame, calls.clone());
            let before = calls.get();
            let start = Instant::now();
            let result = if mode == "incremental" {
                StreamReport::Incremental(isam.update(update, false)?)
            } else {
                StreamReport::Batch(batch.update(frame as f64, update)?)
            };
            let ns = start.elapsed().as_nanos();
            let (estimates, cost, entries, kind) = match result {
                StreamReport::Incremental(r) => {
                    if r.kind != IsamUpdateKind::Incremental {
                        return Err("unexpected nonlinear refresh in linear stream".into());
                    }
                    (
                        r.estimates,
                        r.linearized_cost,
                        r.stored_entries,
                        "Incremental",
                    )
                }
                StreamReport::Batch(r) => {
                    if matches!(
                        r.termination,
                        Termination::NoProgress | Termination::MaxIterations
                    ) {
                        return Err("batch did not converge".into());
                    }
                    (r.estimates, r.final_cost, 0, "Batch")
                }
            };
            elapsed += ns;
            let count = calls.get() - before;
            if mode == "incremental" && count != if frame.is_multiple_of(25) { 3 } else { 2 } {
                return Err("incremental callbacks replayed history".into());
            }
            let q = fixture::quality(&estimates, &references[frame - 1])?;
            if !q.anchor
                || !q.cost.is_finite()
                || q.gradient_inf > 1e-8
                || q.max_delta > 1e-8
                || (q.cost - q.reference_cost).abs() > 1e-9 * q.reference_cost.max(1.)
                || (cost - q.cost).abs() > 1e-9 * q.cost.max(1.)
            {
                return Err(format!(
                    "quality failed at {frame}: delta {}, gradient {}",
                    q.max_delta, q.gradient_inf
                )
                .into());
            }
            max_delta = max_delta.max(q.max_delta);
            max_gradient = max_gradient.max(q.gradient_inf);
            last_cost = q.cost;
            last_entries = entries;
            if round >= warmup {
                writeln!(
                    updates,
                    "{},{frame},{ns},{},{kind},{count},{:.17e},{:.17e},{:.17e},{:.17e},{entries}",
                    round - warmup,
                    frame.is_multiple_of(25),
                    q.cost,
                    q.reference_cost,
                    q.gradient_inf,
                    q.max_delta
                )?;
            }
            if round == warmup && frame == frames {
                let mut final_values = BufWriter::new(File::create(out.join("estimates.csv"))?);
                writeln!(final_values, "key,x,y")?;
                for (key, v) in estimates {
                    writeln!(final_values, "{key},{:.17e},{:.17e}", v[0], v[1])?;
                }
                final_values.flush()?;
            }
        }
        if round >= warmup {
            writeln!(
                samples,
                "{},{elapsed},{frames},{},{last_cost:.17e},{max_delta:.17e},{max_gradient:.17e},{last_entries}",
                round - warmup,
                calls.get()
            )?;
        }
    }
    samples.flush()?;
    updates.flush()?;
    println!(
        "Saved {repeats} {mode} streams ({frames} updates each) to {}",
        out.display()
    );
    Ok(())
}
