//! Isolated owned/borrowed negation; seeding and validation stay outside timing.
use ropt::autodiff::DynamicJet;
use std::{hint::black_box, time::Instant};
#[path = "support/allocation_counter.rs"]
mod allocation;
#[global_allocator]
static ALLOCATOR: allocation::Counter = allocation::Counter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err(
            "usage: profile_dynamic_negation width iterations timing|allocations owned|borrowed"
                .into(),
        );
    }
    let width: usize = args[0].parse()?;
    let iterations: usize = args[1].parse()?;
    let instrument = match args[2].as_str() {
        "timing" => false,
        "allocations" => true,
        _ => return Err("invalid mode".into()),
    };
    let borrowed = match args[3].as_str() {
        "owned" => false,
        "borrowed" => true,
        _ => return Err("invalid ownership".into()),
    };
    if width == 0 || iterations == 0 {
        return Err("invalid workload".into());
    }
    let initial = DynamicJet::variable(f64::MAX, width - 1, width)?;
    let mut jet = initial.clone();
    for _ in 0..4 {
        jet = if borrowed { -&jet } else { -jet };
    }
    if instrument {
        allocation::start();
    }
    let start = Instant::now();
    if borrowed {
        for _ in 0..iterations {
            jet = black_box(-black_box::<&DynamicJet>(&jet));
        }
    } else {
        for _ in 0..iterations {
            jet = black_box(-black_box(jet));
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    let (calls, bytes) = if instrument {
        allocation::stop()
    } else {
        (0, 0)
    };
    let expected = if iterations.is_multiple_of(2) {
        initial
    } else {
        -initial
    };
    assert_eq!(jet.value().to_bits(), expected.value().to_bits());
    assert!(
        jet.partials()?
            .iter()
            .zip(expected.partials()?)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    println!("dimension,iterations,ownership,jet_bytes,seconds,allocations,requested_bytes");
    println!(
        "{width},{iterations},{},{},{seconds:.17},{calls},{bytes}",
        args[3],
        std::mem::size_of::<DynamicJet>()
    );
    println!("value_bits={:016x}", jet.value().to_bits());
    println!(
        "partial_bits={}",
        jet.partials()?
            .iter()
            .map(|v| format!("{:016x}", v.to_bits()))
            .collect::<Vec<_>>()
            .join(",")
    );
    Ok(())
}
