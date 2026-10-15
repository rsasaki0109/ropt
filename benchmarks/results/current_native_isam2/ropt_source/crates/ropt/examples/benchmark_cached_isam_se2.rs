//! Cached-policy update timings with the existing independent accuracy gates.
#[path = "support/allocation_counter.rs"]
mod allocation_counter;
#[path = "support/isam_se2.rs"]
mod fixture;
use fixture::UpdateObserver;
use ropt::isam::IsamReport;
use std::{fmt::Write, path::PathBuf, time::Instant};

#[global_allocator]
static ALLOCATOR: allocation_counter::Counter = allocation_counter::Counter;

struct Measurements {
    start: Option<Instant>,
    allocations: bool,
    updates: String,
    means: String,
}
impl UpdateObserver for Measurements {
    fn start(&mut self) {
        if self.allocations {
            allocation_counter::start();
        }
        self.start = Some(Instant::now());
    }
    fn finish(&mut self, frame: usize, report: &IsamReport) {
        let elapsed = self.start.take().unwrap().elapsed().as_nanos();
        let (calls, bytes) = if self.allocations {
            allocation_counter::stop()
        } else {
            (0, 0)
        };
        writeln!(
            self.updates,
            "{frame},{elapsed},{calls},{bytes},{:?},{},{},{:.17e},{:.17e},{},{:?},{:?}",
            report.relinearization_reason,
            report.factors_linearized,
            report.stored_entries,
            report.linearized_cost,
            report.max_anchor_delta,
            report.batch_trials,
            report.batch_initial_damping,
            report.batch_final_damping,
        )
        .unwrap();
        for (&key, value) in &report.estimates {
            for (coordinate, x) in value.iter().enumerate() {
                writeln!(self.means, "{frame},{key},{coordinate},{x:.17e}").unwrap();
            }
        }
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=3).contains(&args.len()) || args.get(2).is_some_and(|s| s != "--allocations") {
        return Err(
            "usage: benchmark_cached_isam_se2 frames new-output-directory [--allocations]".into(),
        );
    }
    let frames = args[0].parse()?;
    let output = PathBuf::from(&args[1]);
    std::fs::create_dir(&output)?;
    // One complete untimed stream, including all its accuracy checks.
    fixture::validate(frames, &mut std::io::sink())?;
    let mut measurements = Measurements {
        start: None,
        allocations: args.len() == 3,
        updates: "frame,elapsed_ns,allocation_calls,allocation_bytes,reason,factors_linearized,entries,cost,max_anchor_delta,batch_trials,initial_damping,final_damping\n".into(),
        means: "frame,key,coordinate,value\n".into(),
    };
    let mut trace = Vec::new();
    let summary = fixture::validate_observed(frames, &mut trace, &mut measurements)?;
    std::fs::write(output.join("updates.csv"), measurements.updates)?;
    std::fs::write(output.join("means.csv"), measurements.means)?;
    std::fs::write(output.join("quality.csv"), trace)?;
    std::fs::write(output.join("summary.txt"), format!("{summary:#?}\n"))?;
    std::fs::write(
        output.join("options.txt"),
        format!(
            "frames={frames}\nthreshold=0.1\ninterval=10\nforce_every=25\nbatch=SparseLdl(BlockMinimumDegree(3))\nbatch_damping=restart(1e-3)\ncost_tolerance=1e-15\ngradient_tolerance=5e-7\nmax_iterations=200\nroot_update_timed=false\nwarmup_streams=1\nmeasured_streams=1\nallocation_counting={}\n",
            measurements.allocations
        ),
    )?;
    Ok(())
}
