use ropt::{factor::*, fixed_lag::*, manifold::*, *};

fn scalar(key: u64, time: Option<f64>) -> WindowVariable {
    WindowVariable {
        key,
        kind: VariableKind::Euclidean(1),
        value: DVector::zeros(1),
        timestamp: time,
        fixed: false,
    }
}
fn observation(key: u64, target: f64) -> SmootherFactor {
    SmootherFactor::analytic(&[key], 1, Default::default(), move |v| {
        Ok((
            DVector::from_element(1, v[0][0] - target),
            vec![DMatrix::identity(1, 1)],
        ))
    })
}
fn rotation(key: u64, target: DVector<f64>) -> SmootherFactor {
    SmootherFactor::autodiff(&[key], 3, 3, Default::default(), move |v| {
        VariableKind::SO3.local_autodiff(&target, v[0])
    })
}

#[test]
fn retired_manifold_times_follow_refresh_and_survive_failed_updates_and_key_reuse() {
    let target = so3_values(SO3::exp(nalgebra::Vector3::new(0.2, -0.1, 0.3)));
    let fixed_pose = se3_values(SE3::exp(Tangent6::new(1., 2., 3., 0.1, 0.2, 0.3)));
    let mut smoother = FixedLagSmoother::new(FixedLagOptions {
        lag: 2.,
        marginalization_loss: MarginalizationLoss::FrozenIrls,
        solver: Options {
            loss: Loss::Huber(1.),
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    let initial = smoother
        .update(
            2.,
            SmootherUpdate {
                variables: vec![
                    WindowVariable {
                        key: 90,
                        kind: VariableKind::SO3,
                        value: so3_values(SO3::exp(nalgebra::Vector3::zeros())),
                        timestamp: Some(0.),
                        fixed: false,
                    },
                    WindowVariable {
                        key: 1,
                        kind: VariableKind::SE3,
                        value: fixed_pose.clone(),
                        timestamp: Some(1.),
                        fixed: true,
                    },
                    scalar(200, None),
                ],
                factors: vec![rotation(90, target.clone()), observation(200, 10.)],
                ..Default::default()
            },
        )
        .unwrap();
    assert!(initial.marginalized_timestamps.is_empty());
    let boundary = smoother
        .update(
            3.,
            SmootherUpdate {
                timestamps: vec![(90, 1.5)],
                ..Default::default()
            },
        )
        .unwrap();
    assert!(boundary.marginalized_timestamps.is_empty());
    assert!(
        smoother
            .update_with_covariance(
                4.,
                SmootherUpdate {
                    timestamps: vec![(90, 3.5)],
                    ..Default::default()
                },
                &[999],
                &ropt::covariance::CovarianceOptions {
                    model: ropt::covariance::CovarianceModel::FrozenIrls {
                        fallback_loss: Loss::Huber(1.),
                    },
                    ..Default::default()
                }
            )
            .is_err()
    );
    assert_eq!(smoother.last_time(), Some(3.));
    let retired = smoother.update(4., Default::default()).unwrap();
    assert_eq!(retired.marginalized, vec![1, 90]);
    assert_eq!(
        retired
            .marginalized_timestamps
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        retired.marginalized
    );
    assert_eq!(retired.marginalized_timestamps[&1], 1.);
    assert_eq!(retired.marginalized_timestamps[&90], 1.5);
    assert_eq!(retired.marginalized_estimates[&1], fixed_pose);
    assert!(
        VariableKind::SO3
            .local(&target, &retired.marginalized_estimates[&90])
            .unwrap()
            .norm()
            < 1e-7
    );
    assert!(!retired.marginalized_timestamps.contains_key(&200));
    let old_rotation = retired.marginalized_estimates[&90].clone();
    let new_target = so3_values(SO3::exp(nalgebra::Vector3::new(-0.4, 0.2, 0.1)));
    let reused = smoother
        .update(
            5.,
            SmootherUpdate {
                variables: vec![WindowVariable {
                    key: 90,
                    kind: VariableKind::SO3,
                    value: old_rotation.clone(),
                    timestamp: Some(5.),
                    fixed: false,
                }],
                factors: vec![rotation(90, new_target.clone())],
                ..Default::default()
            },
        )
        .unwrap();
    assert!(reused.marginalized_timestamps.is_empty());
    assert!(
        VariableKind::SO3
            .local(&new_target, &reused.estimates[&90])
            .unwrap()
            .norm()
            < 1e-7
    );
    drop(smoother);
    assert_eq!(retired.marginalized_timestamps[&90], 1.5);
    assert_eq!(retired.marginalized_estimates[&90], old_rotation);
}

#[test]
fn robust_retirement_keeps_timestamp_and_constant_cost_with_a_persistent_variable() {
    let mut smoother = FixedLagSmoother::new(FixedLagOptions {
        lag: 1.,
        marginalization_loss: MarginalizationLoss::FrozenIrls,
        solver: Options {
            loss: Loss::Huber(1.),
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    let initial = smoother
        .update(
            0.,
            SmootherUpdate {
                variables: vec![scalar(0, Some(0.)), scalar(99, None)],
                factors: vec![
                    observation(0, 2.),
                    observation(0, 2.1),
                    observation(0, 20.),
                    observation(99, 10.),
                ],
                ..Default::default()
            },
        )
        .unwrap();
    let value = initial.estimates[&0][0];
    let retired = smoother.update(2., Default::default()).unwrap();
    assert_eq!(retired.marginalized_timestamps.len(), 1);
    assert_eq!(retired.marginalized_timestamps[&0], 0.);
    assert!((retired.marginalized_estimates[&0][0] - value).abs() < 1e-7);
    assert!((retired.final_cost - initial.final_cost).abs() < 1e-10);
    let later = smoother.update(3., Default::default()).unwrap();
    assert!(later.marginalized_timestamps.is_empty());
    assert!((later.final_cost - retired.final_cost).abs() < 1e-10);
    assert!((later.estimates[&99][0] - 10.).abs() < 1e-7);
    assert_eq!(retired.marginalized_timestamps[&0], 0.);
}
