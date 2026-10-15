//! Synthetic correlated Gaussian stream and a separate 2x2 block-Thomas oracle.
use nalgebra::{Matrix2, Vector2};
use ropt::{factor::*, fixed_lag::*, noise::NoiseModel, *};
use std::{cell::Cell, collections::BTreeMap, rc::Rc};
pub type V = Vector2<f64>;
type M = Matrix2<f64>;
pub fn covariance(mode: usize) -> M {
    match mode {
        0 => M::new(1., 0.2, 0.2, 2.),
        1 => M::new(0.5, -0.1, -0.1, 0.8),
        _ => M::new(0.3, 0.04, 0.04, 0.4),
    }
}
// Explicit cofactor inversion is independent of ropt's QR/LDL/whitening code.
fn inverse(a: M) -> M {
    let det = a[(0, 0)] * a[(1, 1)] - a[(0, 1)] * a[(1, 0)];
    assert!(det > 0. && det.is_finite());
    M::new(a[(1, 1)], -a[(0, 1)], -a[(1, 0)], a[(0, 0)]) / det
}
pub fn truth(k: usize) -> V {
    V::new(0.1 * k as f64, -0.03 * k as f64)
}
pub fn observed(k: usize) -> V {
    truth(k)
        + V::new(
            0.012 * (0.37 * k as f64).sin(),
            0.015 * (0.21 * k as f64).cos(),
        )
}
fn odometry(k: usize) -> V {
    V::new(
        0.1 + 0.002 * (0.51 * k as f64).sin(),
        -0.03 + 0.003 * (0.19 * k as f64).cos(),
    )
}
fn closure(k: usize) -> (usize, V) {
    let key = k / 2;
    (
        key,
        truth(key)
            + V::new(
                0.004 * (0.17 * k as f64).cos(),
                0.003 * (0.31 * k as f64).sin(),
            ),
    )
}
fn noise(mode: usize) -> NoiseModel {
    let c = covariance(mode);
    NoiseModel::from_covariance(DMatrix::from_column_slice(2, 2, c.as_slice())).unwrap()
}
fn observation(key: usize, target: V, mode: usize, calls: Rc<Cell<usize>>) -> SmootherFactor {
    SmootherFactor::analytic(
        &[key as u64],
        2,
        FactorOptions {
            noise: noise(mode),
            ..Default::default()
        },
        move |v| {
            calls.set(calls.get() + 1);
            Ok((
                &v[0] - DVector::from_column_slice(target.as_slice()),
                vec![DMatrix::identity(2, 2)],
            ))
        },
    )
}
fn between(a: usize, b: usize, target: V, mode: usize, calls: Rc<Cell<usize>>) -> SmootherFactor {
    SmootherFactor::analytic(
        &[a as u64, b as u64],
        2,
        FactorOptions {
            noise: noise(mode),
            ..Default::default()
        },
        move |v| {
            calls.set(calls.get() + 1);
            Ok((
                &v[1] - &v[0] - DVector::from_column_slice(target.as_slice()),
                vec![-DMatrix::identity(2, 2), DMatrix::identity(2, 2)],
            ))
        },
    )
}
pub fn update(k: usize, calls: Rc<Cell<usize>>) -> SmootherUpdate {
    let seed = if k == 0 { V::zeros() } else { observed(k) };
    let mut factors = vec![observation(k, seed, 0, calls.clone())];
    if k > 0 {
        factors.push(between(k - 1, k, odometry(k), 1, calls.clone()));
        if k.is_multiple_of(25) {
            let (old, target) = closure(k);
            factors.push(between(0, old, target, 2, calls));
        }
    }
    SmootherUpdate {
        variables: vec![WindowVariable {
            key: k as u64,
            kind: VariableKind::Euclidean(2),
            value: DVector::from_column_slice(seed.as_slice()),
            timestamp: None,
            fixed: k == 0,
        }],
        factors,
        ..Default::default()
    }
}
pub fn oracle(n: usize) -> Vec<V> {
    let (o, b, l) = (
        inverse(covariance(0)),
        inverse(covariance(1)),
        inverse(covariance(2)),
    );
    let mut diagonal = vec![o; n];
    let mut rhs = (1..=n).map(|k| o * observed(k)).collect::<Vec<_>>();
    for k in 1..=n {
        diagonal[k - 1] += b;
        rhs[k - 1] += b * odometry(k);
        if k > 1 {
            diagonal[k - 2] += b;
            rhs[k - 2] -= b * odometry(k);
        }
        if k.is_multiple_of(25) {
            let (old, target) = closure(k);
            diagonal[old - 1] += l;
            rhs[old - 1] += l * target;
        }
    }
    let mut inverses = Vec::with_capacity(n);
    for k in 0..n {
        if k > 0 {
            let multiplier = -b * inverses[k - 1];
            diagonal[k] -= multiplier * (-b);
            let previous = rhs[k - 1];
            rhs[k] -= multiplier * previous;
        }
        inverses.push(inverse(diagonal[k]));
    }
    let mut x = vec![V::zeros(); n + 1];
    for k in (0..n).rev() {
        x[k + 1] = inverses[k] * (rhs[k] + if k + 1 < n { b * x[k + 2] } else { V::zeros() });
    }
    x
}
pub struct Quality {
    pub cost: f64,
    pub reference_cost: f64,
    pub gradient_inf: f64,
    pub max_delta: f64,
    pub anchor: bool,
}
fn energy(x: &[V]) -> (f64, f64) {
    let (o, b, l) = (
        inverse(covariance(0)),
        inverse(covariance(1)),
        inverse(covariance(2)),
    );
    let mut cost = 0.;
    let mut gradient = vec![V::zeros(); x.len()];
    for k in 1..x.len() {
        let r = x[k] - observed(k);
        let w = o * r;
        cost += 0.5 * r.dot(&w);
        gradient[k] += w;
        let r = x[k] - x[k - 1] - odometry(k);
        let w = b * r;
        cost += 0.5 * r.dot(&w);
        gradient[k] += w;
        gradient[k - 1] -= w;
        if k.is_multiple_of(25) {
            let (old, target) = closure(k);
            let r = x[old] - x[0] - target;
            let w = l * r;
            cost += 0.5 * r.dot(&w);
            gradient[old] += w;
            gradient[0] -= w;
        }
    }
    let norm = gradient.iter().skip(1).map(|v| v.amax()).fold(0., f64::max);
    (cost, norm)
}
pub fn quality(estimates: &BTreeMap<u64, DVector<f64>>, reference: &[V]) -> Result<Quality, Error> {
    if estimates.len() != reference.len() {
        return Err(Error::InvalidDimensions);
    }
    let x = (0..reference.len())
        .map(|k| {
            let v = estimates.get(&(k as u64)).ok_or(Error::InvalidDimensions)?;
            if v.len() != 2 || !v.iter().all(|v| v.is_finite()) {
                return Err(Error::NonFinite);
            }
            Ok(V::new(v[0], v[1]))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let (cost, gradient_inf) = energy(&x);
    let reference_cost = energy(reference).0;
    let max_delta = x
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).norm())
        .fold(0., f64::max);
    Ok(Quality {
        cost,
        reference_cost,
        gradient_inf,
        max_delta,
        anchor: x[0] == V::zeros(),
    })
}
