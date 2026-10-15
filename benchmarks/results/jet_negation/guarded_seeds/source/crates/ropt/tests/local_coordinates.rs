use ropt::autodiff::DynamicAutoDiff;
use ropt::{DMatrix, DVector, Residual, autodiff::DynamicJet as J, factor::*};

#[test]
fn local_and_ad_match_right_retractions_and_central_differences() {
    for (kind, anchor) in [
        (VariableKind::Euclidean(2), DVector::from_vec(vec![1., -2.])),
        (VariableKind::SO2, DVector::from_element(1, 2.9)),
        (VariableKind::SE2, DVector::from_vec(vec![1., -2., 0.7])),
        (
            VariableKind::SO3,
            so3_values(ropt::manifold::SO3::exp(nalgebra::Vector3::new(
                0.3, -0.2, 0.1,
            ))),
        ),
        (
            VariableKind::SE3,
            se3_values(ropt::manifold::SE3::exp(ropt::manifold::Tangent6::new(
                1., -0.5, 0.3, 0.2, -0.1, 0.4,
            ))),
        ),
    ] {
        let n = kind.dimensions().1;
        for scale in [0., 1e-8, 0.2, 1.2] {
            let offset = DVector::from_fn(n, |i, _| scale * ((i + 1) as f64).sin());
            let value = kind.retract(&anchor, &offset).unwrap();
            assert!((kind.local(&anchor, &value).unwrap() - &offset).norm() < 1e-10);
            let problem = DynamicAutoDiff::new(n, n, |d| {
                kind.local_autodiff(&anchor, &kind.retract_autodiff(&value, d)?)
            })
            .unwrap();
            let mut r = DVector::zeros(n);
            let mut jac = DMatrix::zeros(n, n);
            problem
                .evaluate(&DVector::zeros(n), &mut r, &mut jac)
                .unwrap();
            assert!((&r - kind.local(&anchor, &value).unwrap()).norm() < 1e-10);
            for col in 0..n {
                let mut d = DVector::zeros(n);
                d[col] = 1e-6;
                let numeric = (kind
                    .local(&anchor, &kind.retract(&value, &d).unwrap())
                    .unwrap()
                    - kind
                        .local(&anchor, &kind.retract(&value, &(-&d)).unwrap())
                        .unwrap())
                    / 2e-6;
                assert!((numeric - jac.column(col)).norm() < 2e-8);
            }
            if scale == 0. {
                assert!((jac - DMatrix::identity(n, n)).norm() < 1e-12);
            }
        }
    }
}
#[test]
fn local_charts_wrap_angles_and_ignore_quaternion_sign() {
    let k = VariableKind::SO2;
    assert!(
        (k.local(
            &DVector::from_element(1, 3.1),
            &DVector::from_element(1, -3.1)
        )
        .unwrap()[0]
            - (2. * std::f64::consts::PI - 6.2))
            .abs()
            < 1e-12
    );
    let k = VariableKind::SO3;
    let anchor = so3_values(ropt::manifold::SO3::exp(nalgebra::Vector3::new(
        0.1, 0.2, 0.3,
    )));
    let value = k
        .retract(&anchor, &DVector::from_vec(vec![0.2, -0.1, 0.3]))
        .unwrap();
    assert!(
        (k.local(&anchor, &value).unwrap() - k.local(&anchor, &(-&value)).unwrap()).norm() < 1e-12
    );
    let a = k
        .local_autodiff(
            &anchor,
            &value.iter().map(|v| J::constant(*v)).collect::<Vec<_>>(),
        )
        .unwrap();
    let b = k
        .local_autodiff(
            &anchor,
            &value.iter().map(|v| J::constant(-*v)).collect::<Vec<_>>(),
        )
        .unwrap();
    for (a, b) in a.iter().zip(b) {
        assert!((a.value() - b.value()).abs() < 1e-12);
    }
    assert!(k.local_autodiff(&anchor, &[J::constant(1.)]).is_err());
}
