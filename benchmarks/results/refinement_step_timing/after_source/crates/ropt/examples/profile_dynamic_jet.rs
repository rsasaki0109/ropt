//! Separate runtime-AD allocation/latency diagnostic across derivative widths.
use ropt::{
    DMatrix, DVector, Residual,
    autodiff::{DynamicAutoDiff, DynamicJet},
};
use std::{hint::black_box, time::Instant};
#[path = "support/allocation_counter.rs"]
mod allocation;
#[global_allocator]
static ALLOCATOR: allocation::Counter = allocation::Counter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: profile_dynamic_jet dimension iterations timing|allocations".into());
    }
    let dimension: usize = args[0].parse()?;
    let iterations: usize = args[1].parse()?;
    let instrument = match args[2].as_str() {
        "timing" => false,
        "allocations" => true,
        _ => return Err("invalid mode".into()),
    };
    if dimension == 0 || iterations == 0 {
        return Err("invalid workload".into());
    }
    let function = DynamicAutoDiff::new(dimension, 3, |jets: &[DynamicJet]| {
        let mut linear = DynamicJet::constant(0.);
        let mut nonlinear = DynamicJet::constant(0.);
        for (i, x) in jets.iter().enumerate() {
            linear = linear + x * ((i + 1) as f64 * 0.125);
            nonlinear = nonlinear + x.sin() * x.exp() + x * x;
        }
        let angle = nonlinear.atan2(&(&linear + 1.));
        Ok(vec![linear, nonlinear, angle])
    })?
    .with_chunk_size(dimension)?;
    let values = DVector::from_iterator(
        dimension,
        (0..dimension).map(|i| 0.2 + i as f64 / dimension as f64 * 0.3),
    );
    let mut residual = DVector::zeros(3);
    let mut jacobian = DMatrix::zeros(3, dimension);
    for _ in 0..4 {
        function.evaluate(&values, &mut residual, &mut jacobian)?;
    }
    let before_r = residual.clone();
    let before_j = jacobian.clone();
    if instrument {
        allocation::start();
    }
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(function.evaluate(black_box(&values), &mut residual, &mut jacobian))?;
        black_box(&residual);
        black_box(&jacobian);
    }
    let seconds = start.elapsed().as_secs_f64();
    let (calls, bytes) = if instrument {
        allocation::stop()
    } else {
        (0, 0)
    };
    assert_eq!(residual, before_r);
    assert_eq!(jacobian, before_j);
    // Independently check the physical residual and its analytic derivatives.
    let linear = values
        .iter()
        .enumerate()
        .map(|(i, x)| x * (i + 1) as f64 * 0.125)
        .sum::<f64>();
    let nonlinear = values
        .iter()
        .map(|x| x.sin() * x.exp() + x * x)
        .sum::<f64>();
    assert!((residual[0] - linear).abs() < 1e-12 * linear.max(1.));
    assert!((residual[1] - nonlinear).abs() < 1e-12 * nonlinear.max(1.));
    assert!((residual[2] - nonlinear.atan2(linear + 1.)).abs() < 1e-14);
    for (i, x) in values.iter().enumerate() {
        let a = (i + 1) as f64 * 0.125;
        let b = x.cos() * x.exp() + x.sin() * x.exp() + 2. * x;
        let angle =
            ((linear + 1.) * b - nonlinear * a) / ((linear + 1.).powi(2) + nonlinear.powi(2));
        assert!((jacobian[(0, i)] - a).abs() < 1e-14);
        assert!((jacobian[(1, i)] - b).abs() < 1e-14);
        assert!((jacobian[(2, i)] - angle).abs() < 1e-13);
    }
    println!("dimension,iterations,jet_bytes,seconds,allocations,requested_bytes");
    println!(
        "{dimension},{iterations},{},{seconds:.17},{calls},{bytes}",
        std::mem::size_of::<DynamicJet>()
    );
    println!(
        "residual_bits={}",
        residual
            .iter()
            .map(|v| format!("{:016x}", v.to_bits()))
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "jacobian_bits={}",
        jacobian
            .iter()
            .map(|v| format!("{:016x}", v.to_bits()))
            .collect::<Vec<_>>()
            .join(",")
    );
    Ok(())
}
