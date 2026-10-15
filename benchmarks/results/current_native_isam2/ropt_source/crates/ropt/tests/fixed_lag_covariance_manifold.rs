use nalgebra::Vector3;
use ropt::{
    DMatrix, DVector,
    covariance::{CovarianceInput, CovarianceOptions},
    factor::*,
    fixed_lag::*,
    manifold::{SE3, SO3, Tangent6},
    noise::NoiseModel,
    pose_graph::RelativePose3,
};

fn between(a: u64, b: u64, measurement: SE3, sigma: &[f64]) -> SmootherFactor {
    let residual = RelativePose3::new(measurement).unwrap();
    SmootherFactor::autodiff(
        &[a, b],
        6,
        6,
        FactorOptions {
            noise: NoiseModel::diagonal(DVector::from_column_slice(sigma)).unwrap(),
            ..Default::default()
        },
        move |v| residual.residual_autodiff(v[0], v[1]),
    )
}

#[test]
fn retiring_se3_covariance_matches_independent_tangent_differences_after_compaction() {
    let root = SE3 {
        translation: Vector3::new(2., -1., 0.4),
        rotation: SO3::exp(Vector3::new(0.2, -0.3, 0.5)),
    };
    let first = SE3::exp(Tangent6::from_row_slice(&[1., 0.2, -0.3, 0.3, -0.1, 0.2]));
    let second = SE3::exp(Tangent6::from_row_slice(&[-0.4, 0.8, 0.1, -0.2, 0.4, 0.1]));
    let a = root.compose(first);
    let b = a.compose(second);
    let sigma_a = [0.1, 0.2, 0.3, 0.04, 0.05, 0.06];
    let sigma_b = [0.3, 0.15, 0.2, 0.03, 0.07, 0.02];
    let residual_a = RelativePose3::new(first).unwrap();
    let residual_b = RelativePose3::new(second).unwrap();
    // Independent f64 group residuals and central differences, with columns
    // in requested order [b:6, a:6, scalar:1], rather than the graph layout.
    let evaluate = |delta: &DVector<f64>| {
        let moved_b = b.retract(Tangent6::from_column_slice(&delta.as_slice()[..6]));
        let moved_a = a.retract(Tangent6::from_column_slice(&delta.as_slice()[6..12]));
        let ra = residual_a.residual(root, moved_a).unwrap();
        let rb = residual_b.residual(moved_a, moved_b).unwrap();
        let mut r = DVector::zeros(13);
        for k in 0..6 {
            r[k] = ra[k] / sigma_a[k];
            r[k + 6] = rb[k] / sigma_b[k];
        }
        r[12] = 2. * delta[12];
        r
    };
    let mut j = DMatrix::zeros(13, 13);
    let h = 1e-6;
    for k in 0..13 {
        let mut delta = DVector::zeros(13);
        delta[k] = h;
        let plus = evaluate(&delta);
        delta[k] = -h;
        j.set_column(k, &((plus - evaluate(&delta)) / (2. * h)));
    }
    let expected = (j.transpose() * j).try_inverse().unwrap();
    assert!(expected.view((0, 6), (6, 6)).norm() > 1e-3);

    for sign in [1., -1.] {
        for input in [CovarianceInput::Dense, CovarianceInput::SparseToDense] {
            let mut s = FixedLagSmoother::new(FixedLagOptions {
                lag: 1.,
                ..Default::default()
            })
            .unwrap();
            let pose = |key, value, timestamp, fixed| {
                let mut ambient = se3_values(value);
                for k in 3..7 {
                    ambient[k] *= sign;
                }
                WindowVariable {
                    key,
                    kind: VariableKind::SE3,
                    value: ambient,
                    timestamp,
                    fixed,
                }
            };
            s.update(
                1.,
                SmootherUpdate {
                    // Deliberately not in key or query order. Key 5 will expire;
                    // removing it changes the surviving active tangent indices.
                    variables: vec![
                        pose(30, b, Some(1.), false),
                        pose(10, root, None, true),
                        pose(5, a, Some(0.), false),
                        WindowVariable {
                            key: 20,
                            kind: VariableKind::Euclidean(1),
                            value: DVector::from_element(1, 4.),
                            timestamp: None,
                            fixed: false,
                        },
                    ],
                    factors: vec![
                        between(10, 5, first, &sigma_a),
                        between(5, 30, second, &sigma_b),
                        SmootherFactor::analytic(&[20], 1, Default::default(), |v| {
                            Ok((
                                DVector::from_element(1, 2. * (v[0][0] - 4.)),
                                vec![DMatrix::from_element(1, 1, 2.)],
                            ))
                        }),
                    ],
                    ..Default::default()
                },
            )
            .unwrap();
            let options = CovarianceOptions {
                input,
                ..Default::default()
            };
            let (report, covariance) = s
                .update_with_covariance(2., Default::default(), &[30, 5, 20], &options)
                .unwrap();
            assert_eq!(report.marginalized, vec![5]);
            assert_eq!(report.marginalized_estimates[&5].len(), 7);
            assert!(
                a.local(read_se3(&report.marginalized_estimates[&5]).unwrap())
                    .norm()
                    < 1e-10
            );
            assert_eq!(covariance.matrix.shape(), (13, 13));
            assert_eq!(covariance.rank, 13);
            assert_eq!(
                covariance.indices,
                (7..13).chain(0..6).chain([6]).collect::<Vec<_>>()
            );
            assert!((&covariance.matrix - &expected).norm() < 1e-8);

            let retained = s.marginal_covariance(&[30, 20], &options).unwrap();
            assert_eq!(retained.indices, (1..7).chain([0]).collect::<Vec<_>>());
            let selected = [0, 1, 2, 3, 4, 5, 12];
            let retained_expected =
                DMatrix::from_fn(7, 7, |r, c| expected[(selected[r], selected[c])]);
            assert!((retained.matrix - retained_expected).norm() < 1e-8);
            // The earlier covariance remains in its original layout.
            assert_eq!(covariance.indices[0], 7);
            assert!(s.estimate(5).is_none());
        }
    }
}
