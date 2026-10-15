//! Synthetic nonlinear probe workload; setup/validation are outside query timers.
use ropt::factor::VariableKind;
use ropt::fixed_lag::{SmootherFactor, SmootherUpdate, WindowVariable};
use ropt::isam::{IsamOptions, IsamSmoother};
use ropt::{DMatrix, DVector, Error, GradientNorm};
use std::{cell::Cell, hint::black_box, rc::Rc, time::Instant};
#[path = "support/allocation_counter.rs"]
mod allocation;
#[global_allocator]
static ALLOCATOR: allocation::Counter = allocation::Counter;

fn hash(mut state: u64, word: u64) -> u64 {
    for byte in word.to_le_bytes() {
        state = (state ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
    state
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err(
            "usage: benchmark_autodiff_probe blocks ad_percent chunk queries timing|allocations"
                .into(),
        );
    }
    let blocks: usize = args[1].parse()?;
    let percent: usize = args[2].parse()?;
    let chunk: usize = args[3].parse()?;
    let queries: usize = args[4].parse()?;
    let count_allocations = match args[5].as_str() {
        "timing" => false,
        "allocations" => true,
        _ => return Err("invalid measurement mode".into()),
    };
    if blocks == 0 || percent > 100 || chunk == 0 || queries == 0 {
        return Err(Error::InvalidOptions.into());
    }
    let calls = Rc::new(Cell::new(0usize));
    let mut update = SmootherUpdate::default();
    update.variables.push(WindowVariable {
        key: 0,
        kind: VariableKind::Euclidean(1),
        value: DVector::from_element(1, 1.),
        timestamp: None,
        fixed: true,
    });
    let ad_factors = blocks * percent / 100;
    for i in 0..blocks {
        let key = i as u64 + 1;
        update.variables.push(WindowVariable {
            key,
            kind: VariableKind::Euclidean(2),
            value: DVector::from_vec(vec![0.2 + (i % 3) as f64 * 0.1, 0.1]),
            timestamp: None,
            fixed: false,
        });
        let counter = calls.clone();
        // Alternate callback slots; fixed blocks retain their derivative chunks.
        let active = i % 2;
        let fixed = 1 - active;
        let keys = if active == 0 { [key, 0] } else { [0, key] };
        let factor = if i < ad_factors {
            SmootherFactor::autodiff(&keys, 3, chunk, Default::default(), move |v| {
                counter.set(counter.get() + 1);
                let x = &v[active][0];
                let y = &v[active][1];
                Ok(vec![x - 0.3, y + 0.2, x * x + y.sin() - &v[fixed][0]])
            })
        } else {
            SmootherFactor::analytic(&keys, 3, Default::default(), move |v| {
                counter.set(counter.get() + 1);
                let x = v[active][0];
                let y = v[active][1];
                let mut j = vec![DMatrix::zeros(3, 0); 2];
                j[active] = DMatrix::from_row_slice(3, 2, &[1., 0., 0., 1., 2. * x, y.cos()]);
                j[fixed] = DMatrix::from_column_slice(3, 1, &[0., 0., -1.]);
                Ok((
                    DVector::from_vec(vec![x - 0.3, y + 0.2, x * x + y.sin() - v[fixed][0]]),
                    j,
                ))
            })
        };
        update.factors.push(factor);
    }
    let mut smoother = IsamSmoother::new(IsamOptions {
        relinearization_threshold: 1e10,
        relinearization_interval: usize::MAX,
        ..Default::default()
    })?;
    smoother.update(update, false)?;
    assert_eq!(smoother.variable_count(), blocks + 1);
    assert_eq!(smoother.factor_count(), blocks);
    assert_eq!(smoother.estimate(0).unwrap()[0], 1.);
    let mut estimates_hash = 0xcbf29ce484222325;
    let mut cost = 0.;
    let mut gradients = Vec::new();
    for key in 1..=blocks as u64 {
        let value = smoother.estimate(key).unwrap();
        let (x, y) = (value[0], value[1]);
        estimates_hash = hash(hash(estimates_hash, x.to_bits()), y.to_bits());
        let r = x * x + y.sin() - 1.;
        cost += 0.5 * ((x - 0.3).powi(2) + (y + 0.2).powi(2) + r * r);
        let gx = (x - 0.3 + 2. * x * r).abs();
        let gy = (y + 0.2 + y.cos() * r).abs();
        gradients.push((
            key,
            gx.max(gy),
            (gx / 1_f64.hypot(2. * x)).max(gy / 1_f64.hypot(y.cos())),
        ));
    }
    let expected_calls = ad_factors * 3usize.div_ceil(chunk) + blocks - ad_factors;
    println!(
        "norm,blocks,ad_factors,chunk,queries,seconds,cost_bits,gradient_bits,variable_bits,estimate_bits,callback_calls,allocation_calls,allocation_bytes"
    );
    for norm in [GradientNorm::Absolute, GradientNorm::ColumnNormalized] {
        for _ in 0..4 {
            black_box(smoother.nonlinear_summary(norm)?);
        }
        let mut elapsed = 0.;
        let mut callback_calls = 0;
        let mut allocation_calls = 0;
        let mut allocation_bytes = 0;
        let mut fingerprint = None;
        for _ in 0..queries {
            let before = calls.get();
            if count_allocations {
                allocation::start();
            }
            let start = Instant::now();
            let result = black_box(smoother.nonlinear_summary(norm)?);
            elapsed += start.elapsed().as_secs_f64();
            if count_allocations {
                let (c, b) = allocation::stop();
                allocation_calls += c;
                allocation_bytes += b;
            }
            let called = calls.get() - before;
            assert_eq!(called, expected_calls);
            callback_calls += called;
            assert!((result.cost - cost).abs() <= 2e-12 * cost.max(1.));
            assert_eq!(result.variable_gradients.len(), blocks);
            let mut max_gradient = 0_f64;
            let mut bits = 0xcbf29ce484222325;
            for &(key, absolute, normalized) in &gradients {
                let expected = if norm == GradientNorm::Absolute {
                    absolute
                } else {
                    normalized
                };
                let actual = result.variable_gradients[&key];
                assert!((actual - expected).abs() <= 2e-14 * expected.max(1.));
                max_gradient = max_gradient.max(expected);
                bits = hash(hash(bits, key), actual.to_bits());
            }
            assert!((result.gradient_norm - max_gradient).abs() <= 2e-14 * max_gradient.max(1.));
            let current = (result.cost.to_bits(), result.gradient_norm.to_bits(), bits);
            if let Some(previous) = fingerprint {
                assert_eq!(current, previous);
            }
            fingerprint = Some(current);
        }
        let (cost_bits, gradient_bits, bits) = fingerprint.unwrap();
        println!(
            "{norm:?},{blocks},{ad_factors},{chunk},{queries},{elapsed:.17},{cost_bits:016x},{gradient_bits:016x},{bits:016x},{estimates_hash:016x},{callback_calls},{allocation_calls},{allocation_bytes}"
        );
    }
    Ok(())
}
