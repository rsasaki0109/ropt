//! Jet-valued right retractions. Small-angle branches avoid sqrt(0) derivatives.
use crate::autodiff::DynamicJet as J;
use crate::factor::{VariableKind, read_se3, read_so3};
use crate::{DVector, Error};

pub(crate) fn cross(a: &[J], b: &[J]) -> [J; 3] {
    [
        &a[1] * &b[2] - &a[2] * &b[1],
        &a[2] * &b[0] - &a[0] * &b[2],
        &a[0] * &b[1] - &a[1] * &b[0],
    ]
}
/// Unit quaternion `[w,x,y,z]` acting on a three-vector.
pub(crate) fn rotate(q: &[J], v: &[J]) -> [J; 3] {
    let uv = cross(&q[1..], v);
    let uuv = cross(&q[1..], &uv);
    std::array::from_fn(|i| &v[i] + 2.0 * (&q[0] * &uv[i] + &uuv[i]))
}
fn rotation_exp(w: &[J]) -> [J; 4] {
    let s = w.iter().fold(J::constant(0.0), |a, b| a + b * b);
    let (c, a) = if s.value() < 1e-10 {
        (
            1.0 - &s / 8.0 + s.powi(2) / 384.0,
            0.5 - &s / 48.0 + s.powi(2) / 3840.0,
        )
    } else {
        let theta = s.sqrt();
        ((&theta * 0.5).cos(), (&theta * 0.5).sin() / theta)
    };
    [c, &a * &w[0], &a * &w[1], &a * &w[2]]
}
pub(crate) fn compose_rotation(q: &[J], b: &[J]) -> [J; 4] {
    let c = cross(&q[1..], &b[1..]);
    [
        &q[0] * &b[0] - &q[1] * &b[1] - &q[2] * &b[2] - &q[3] * &b[3],
        &q[0] * &b[1] + &b[0] * &q[1] + &c[0],
        &q[0] * &b[2] + &b[0] * &q[2] + &c[1],
        &q[0] * &b[3] + &b[0] * &q[3] + &c[2],
    ]
}
pub(crate) fn retract(kind: VariableKind, x: &DVector<f64>, d: &[J]) -> Result<Vec<J>, Error> {
    kind.validate(x)?;
    if d.len() != kind.dimensions().1 {
        return Err(Error::InvalidDimensions);
    }
    let mut width = 0;
    for v in d {
        let n = v.partials()?.len();
        if n != 0 {
            if width != 0 && width != n {
                return Err(Error::InvalidDimensions);
            }
            width = n;
        }
    }
    let angle = |base: f64, delta: &J| {
        let c = delta.cos();
        let s = delta.sin();
        (base.sin() * &c + base.cos() * &s).atan2(&(base.cos() * c - base.sin() * s))
    };
    let out = match kind {
        VariableKind::Euclidean(_) => x.iter().zip(d).map(|(x, d)| *x + d).collect(),
        VariableKind::SO2 => vec![angle(x[0], &d[0])],
        VariableKind::SE2 => {
            let w = &d[2];
            let (a, b) = if w.value().abs() < 1e-6 {
                (1.0 - w.powi(2) / 6.0, w / 2.0 - w.powi(3) / 24.0)
            } else {
                (w.sin() / w, 2.0 * (w / 2.0).sin().powi(2) / w)
            };
            let u = &a * &d[0] - &b * &d[1];
            let v = &b * &d[0] + &a * &d[1];
            vec![
                x[0] + x[2].cos() * &u - x[2].sin() * &v,
                x[1] + x[2].sin() * u + x[2].cos() * v,
                angle(x[2], w),
            ]
        }
        VariableKind::SO3 | VariableKind::SE3 => {
            // Normalize the base quaternion exactly as the f64 retraction does.
            let (q, offset) = if matches!(kind, VariableKind::SO3) {
                (read_so3(x)?.0, 0)
            } else {
                (read_se3(x)?.rotation.0, 3)
            };
            let q = q.quaternion();
            let q = [q.w, q.i, q.j, q.k].map(J::constant);
            let w = &d[offset..offset + 3];
            let rotation = compose_rotation(&q, &rotation_exp(w));
            if offset == 0 {
                rotation.to_vec()
            } else {
                let s = w.iter().fold(J::constant(0.0), |a, b| a + b * b);
                let (a, b) = if s.value() < 1e-10 {
                    (0.5 - &s / 24.0, 1.0 / 6.0 - &s / 120.0)
                } else {
                    let theta = s.sqrt();
                    (
                        (1.0 - theta.cos()) / &s,
                        (&theta - theta.sin()) / (&s * theta),
                    )
                };
                let kv = cross(w, &d[..3]);
                let kkv = cross(w, &kv);
                let v: [J; 3] = std::array::from_fn(|i| &d[i] + &a * &kv[i] + &b * &kkv[i]);
                let translation = rotate(&q, &v);
                translation
                    .iter()
                    .enumerate()
                    .map(|(i, v)| x[i] + v)
                    .chain(rotation)
                    .collect()
            }
        }
    };
    for v in &out {
        v.partials()?;
    }
    Ok(out)
}

/// Validate and normalize ambient unit quaternion jets, preserving their derivatives.
pub(crate) fn normalize_rotation(q: &[J]) -> Result<[J; 4], Error> {
    if q.len() != 4 {
        return Err(Error::InvalidDimensions);
    }
    for v in q {
        v.partials()?;
    }
    let s = q.iter().fold(J::constant(0.0), |a, b| a + b * b);
    if (s.value().sqrt() - 1.0).abs() > 1e-8 {
        return Err(Error::Evaluation("quaternion must have unit norm".into()));
    }
    let norm = s.sqrt();
    let out = std::array::from_fn(|i| &q[i] / &norm);
    for v in &out {
        v.partials()?;
    }
    Ok(out)
}
pub(crate) fn conjugate(q: &[J; 4]) -> [J; 4] {
    [q[0].clone(), -&q[1], -&q[2], -&q[3]]
}
/// Principal SO3 log, invariant to quaternion sign away from the pi cut.
pub(crate) fn rotation_log(q: &[J; 4]) -> Result<[J; 3], Error> {
    let q = normalize_rotation(q)?;
    let q = if q[0].value() < 0.0 { q.map(|v| -v) } else { q };
    let s = q[1..].iter().fold(J::constant(0.0), |a, b| a + b * b);
    let scale = if s.value() < 1e-12 {
        2.0 + &s / 3.0 + 3.0 * s.powi(2) / 20.0
    } else {
        let norm = s.sqrt();
        2.0 * norm.atan2(&q[0]) / norm
    };
    let out = std::array::from_fn(|i| &scale * &q[i + 1]);
    for v in &out {
        v.partials()?;
    }
    Ok(out)
}
/// SE3 log translation uses inverse left Jacobian, in [translation,rotation] order.
pub(crate) fn pose_log(t: &[J; 3], q: &[J; 4]) -> Result<Vec<J>, Error> {
    let w = rotation_log(q)?;
    let s = w.iter().fold(J::constant(0.0), |a, b| a + b * b);
    let a = if s.value() < 1e-10 {
        1.0 / 12.0 + &s / 720.0
    } else {
        let theta = s.sqrt();
        (1.0 - 0.5 * &theta / (&theta * 0.5).tan()) / s
    };
    let kt = cross(&w, t);
    let kkt = cross(&w, &kt);
    let translation: [J; 3] = std::array::from_fn(|i| &t[i] - 0.5 * &kt[i] + &a * &kkt[i]);
    let out = translation.into_iter().chain(w).collect::<Vec<_>>();
    for v in &out {
        v.partials()?;
    }
    Ok(out)
}

pub(crate) fn local(
    kind: VariableKind,
    anchor: &DVector<f64>,
    value: &[J],
) -> Result<Vec<J>, Error> {
    kind.validate(anchor)?;
    for v in value {
        v.partials()?;
    }
    kind.validate(&DVector::from_iterator(
        value.len(),
        value.iter().map(J::value),
    ))?;
    let out = match kind {
        VariableKind::Euclidean(_) => value.iter().zip(anchor).map(|(v, a)| v - *a).collect(),
        VariableKind::SO2 => {
            let w = &value[0] - anchor[0];
            vec![w.sin().atan2(&w.cos())]
        }
        VariableKind::SE2 => {
            let dx = &value[0] - anchor[0];
            let dy = &value[1] - anchor[1];
            let (c, s) = (anchor[2].cos(), anchor[2].sin());
            let x = c * &dx + s * &dy;
            let y = -s * dx + c * dy;
            let angle = &value[2] - anchor[2];
            let w = angle.sin().atan2(&angle.cos());
            let a = if w.value().abs() < 1e-6 {
                1.0 - w.powi(2) / 12.0
            } else {
                (&w * 0.5) / (&w * 0.5).tan()
            };
            vec![&a * &x + 0.5 * &w * &y, -0.5 * &w * x + a * y, w]
        }
        VariableKind::SO3 | VariableKind::SE3 => {
            let offset = if matches!(kind, VariableKind::SO3) {
                0
            } else {
                3
            };
            let q = if offset == 0 {
                read_so3(anchor)?.0
            } else {
                read_se3(anchor)?.rotation.0
            };
            let q = q.quaternion();
            let inverse = [q.w, -q.i, -q.j, -q.k].map(J::constant);
            let relative = compose_rotation(&inverse, &normalize_rotation(&value[offset..])?);
            if offset == 0 {
                rotation_log(&relative)?.to_vec()
            } else {
                let difference: [J; 3] = std::array::from_fn(|i| &value[i] - anchor[i]);
                pose_log(&rotate(&inverse, &difference), &relative)?
            }
        }
    };
    for v in &out {
        v.partials()?;
    }
    Ok(out)
}
