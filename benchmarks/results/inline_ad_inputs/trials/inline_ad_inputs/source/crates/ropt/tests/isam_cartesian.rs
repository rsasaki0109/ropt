use ropt::{factor::*, fixed_lag::*, isam::*, *};
fn observation(target: [f64; 3]) -> SmootherFactor {
    SmootherFactor::autodiff(&[9], 3, 3, Default::default(), move |v| {
        Ok(v[0]
            .iter()
            .zip(target)
            .map(|(value, target)| value - target)
            .collect())
    })
}
#[test]
fn cartesian_local_coordinates_preserve_exact_cached_affine_observation_energy() {
    let mut smoother = IsamSmoother::new(IsamOptions {
        relinearization_threshold: 100.,
        relinearization_interval: 100,
        ..Default::default()
    })
    .unwrap()
    .with_bayes_tree()
    .unwrap();
    let first = smoother
        .update(
            SmootherUpdate {
                variables: vec![WindowVariable {
                    key: 9,
                    kind: VariableKind::SE2Cartesian,
                    value: DVector::from_vec(vec![0., 0., 0.1]),
                    timestamp: None,
                    fixed: false,
                }],
                factors: vec![observation([1., 2., 0.7])],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    assert_eq!(first.kind, IsamUpdateKind::Incremental);
    assert!((&first.estimates[&9] - DVector::from_vec(vec![1., 2., 0.7])).norm() < 1e-12);
    let second = smoother
        .update(
            SmootherUpdate {
                factors: vec![observation([-3., 1., -0.2])],
                ..Default::default()
            },
            false,
        )
        .unwrap();
    assert_eq!(second.kind, IsamUpdateKind::Incremental);
    let expected = DVector::from_vec(vec![-1., 1.5, 0.25]);
    assert!((&second.estimates[&9] - expected).norm() < 1e-12);
    assert!((second.linearized_cost - 0.25 * (4_f64.powi(2) + 1. + 0.9_f64.powi(2))).abs() < 1e-12);
    assert_eq!(second.factors_linearized, 1);
}
