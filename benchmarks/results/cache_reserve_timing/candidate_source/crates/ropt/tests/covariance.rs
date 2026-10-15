use nalgebra::Vector3;
use ropt::{
    covariance::*, factor::*, manifold::*, noise::NoiseModel,
    pose_graph::add_relative_pose3_factor, *,
};
use std::ops::Range;
fn coupled() -> (FactorGraph, VariableId, VariableId) {
    let mut g = FactorGraph::new();
    let x = g
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    let y = g
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    g.add_analytic_factor_with_options(
        &[x, y],
        1,
        FactorOptions {
            noise: NoiseModel::isotropic(2.0).unwrap(),
            loss: None,
        },
        |v| {
            Ok((
                DVector::from_element(1, v[0][0] - v[1][0]),
                vec![
                    DMatrix::<f64>::identity(1, 1),
                    -DMatrix::<f64>::identity(1, 1),
                ],
            ))
        },
    )
    .unwrap();
    g.add_analytic_factor_with_options(
        &[y],
        1,
        FactorOptions {
            noise: NoiseModel::isotropic(3.0).unwrap(),
            loss: None,
        },
        |v| Ok((v[0].clone(), vec![DMatrix::<f64>::identity(1, 1)])),
    )
    .unwrap();
    (g, x, y)
}
#[test]
fn selected_marginals_include_uncertain_nuisance_variables_and_preserve_order() {
    let (g, x, y) = coupled();
    let initial = g.initial();
    let covariance = g
        .marginal_covariance(&initial, &[x, y], &CovarianceOptions::default())
        .unwrap();
    let expected = DMatrix::from_row_slice(2, 2, &[13.0, 9.0, 9.0, 9.0]);
    assert!((&covariance.matrix - &expected).norm() < 1e-12);
    assert_eq!(covariance.rank, 2);
    let x_only = g
        .marginal_covariance(&initial, &[x], &CovarianceOptions::default())
        .unwrap();
    assert!((x_only.matrix[(0, 0)] - 13.0).abs() < 1e-12); // Not 4: y is uncertain, not conditioned away.
    let reversed = g
        .marginal_covariance(
            &initial,
            &[y, x],
            &CovarianceOptions {
                input: CovarianceInput::SparseToDense,
                ..CovarianceOptions::default()
            },
        )
        .unwrap();
    assert_eq!(reversed.indices, vec![1, 0]);
    assert!(
        (reversed.matrix - DMatrix::from_row_slice(2, 2, &[9.0, 9.0, 9.0, 13.0])).norm() < 1e-12
    );
    let scaled = g
        .marginal_covariance(
            &initial,
            &[x, y],
            &CovarianceOptions {
                variance_scale: 4.0,
                ..CovarianceOptions::default()
            },
        )
        .unwrap();
    assert!((scaled.matrix - 4.0 * expected).norm() < 1e-11);
}
#[test]
fn known_correlated_measurement_covariance_is_recovered_without_residual_scale_estimation() {
    let sigma = DMatrix::from_row_slice(
        3,
        3,
        &[0.04, 0.01, 0.005, 0.01, 0.09, -0.008, 0.005, -0.008, 0.16],
    );
    let mut g = FactorGraph::new();
    let id = g
        .add_variable(
            VariableKind::Euclidean(3),
            DVector::from_vec(vec![5.0, -3.0, 2.0]),
        )
        .unwrap();
    g.add_analytic_factor_with_options(
        &[id],
        3,
        FactorOptions {
            noise: NoiseModel::from_covariance(sigma.clone()).unwrap(),
            loss: None,
        },
        |v| Ok((v[0].clone(), vec![DMatrix::<f64>::identity(3, 3)])),
    )
    .unwrap();
    for input in [CovarianceInput::Dense, CovarianceInput::SparseToDense] {
        let c = g
            .marginal_covariance(
                &g.initial(),
                &[id],
                &CovarianceOptions {
                    input,
                    ..CovarianceOptions::default()
                },
            )
            .unwrap();
        assert!((c.matrix - &sigma).norm() < 1e-14);
    }
    let c = g
        .marginal_covariance(&DVector::zeros(3), &[id], &CovarianceOptions::default())
        .unwrap();
    assert!((c.matrix - sigma).norm() < 1e-14);
}
#[test]
fn fixed_variables_are_conditioned_and_layout_changes_are_resolved_from_ids() {
    let (mut g, x, y) = coupled();
    let point = g.initial();
    assert_eq!(g.tangent_range(x).unwrap(), Some(0..1));
    assert_eq!(g.tangent_range(y).unwrap(), Some(1..2));
    g.set_fixed(y, true).unwrap();
    assert_eq!(g.tangent_range(y).unwrap(), None);
    assert!(matches!(
        g.marginal_covariance(&point, &[y], &CovarianceOptions::default()),
        Err(Error::Evaluation(_))
    ));
    let covariance = g
        .marginal_covariance(&point, &[x], &CovarianceOptions::default())
        .unwrap();
    assert!((covariance.matrix[(0, 0)] - 4.0).abs() < 1e-12);
    g.set_fixed(y, false).unwrap();
    g.set_fixed(x, true).unwrap();
    assert_eq!(g.tangent_range(y).unwrap(), Some(0..1));
    let covariance = g
        .marginal_covariance(&point, &[y], &CovarianceOptions::default())
        .unwrap();
    assert!((covariance.matrix[(0, 0)] - 36.0 / 13.0).abs() < 1e-12);
    let (other, other_id, _) = coupled();
    assert_eq!(g.tangent_range(other_id), Err(Error::InvalidDimensions));
    assert!(other.tangent_range(x).is_err());
}
#[test]
fn singularity_is_checked_in_the_full_problem_even_when_only_observable_columns_are_queried() {
    let mut g = FactorGraph::new();
    let a = g
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    let _b = g
        .add_variable(VariableKind::Euclidean(1), DVector::zeros(1))
        .unwrap();
    g.add_analytic_factor(&[a], 2, |_| {
        Ok((
            DVector::zeros(2),
            vec![DMatrix::from_row_slice(2, 1, &[1.0, 0.0])],
        ))
    })
    .unwrap();
    assert_eq!(
        g.marginal_covariance(&g.initial(), &[a], &CovarianceOptions::default())
            .unwrap_err(),
        Error::RankDeficient {
            rank: 1,
            parameters: 2
        }
    );
    let mut g = FactorGraph::new();
    let id = g
        .add_variable(VariableKind::Euclidean(2), DVector::zeros(2))
        .unwrap();
    g.add_analytic_factor(&[id], 1, |_| {
        Ok((
            DVector::zeros(1),
            vec![DMatrix::from_row_slice(1, 2, &[1.0, 1.0])],
        ))
    })
    .unwrap();
    assert_eq!(
        g.marginal_covariance(&g.initial(), &[id], &CovarianceOptions::default())
            .unwrap_err(),
        Error::RankDeficient {
            rank: 1,
            parameters: 2
        }
    );
}
#[test]
fn gauge_freedom_fails_and_anchored_se3_covariance_uses_six_body_tangent_dimensions() {
    let mut g = FactorGraph::new();
    let a = g
        .add_variable(VariableKind::SE3, se3_values(SE3::exp(Tangent6::zeros())))
        .unwrap();
    let pose = SE3 {
        translation: Vector3::new(1.0, 0.1, 0.2),
        rotation: SO3::exp(Vector3::new(0.3, -0.2, 0.4)),
    };
    let b = g.add_variable(VariableKind::SE3, se3_values(pose)).unwrap();
    let sigma = DVector::from_vec(vec![0.1, 0.2, 0.3, 0.01, 0.02, 0.03]);
    add_relative_pose3_factor(
        &mut g,
        a,
        b,
        pose,
        FactorOptions {
            noise: NoiseModel::diagonal(sigma.clone()).unwrap(),
            loss: Some(Loss::Squared),
        },
    )
    .unwrap();
    assert!(matches!(
        g.marginal_covariance(&g.initial(), &[b], &CovarianceOptions::default()),
        Err(Error::RankDeficient { .. })
    ));
    g.set_fixed(a, true).unwrap();
    assert_eq!(g.tangent_range(b).unwrap(), Some(0..6));
    let expected = DMatrix::from_diagonal(&sigma.map(|s| s * s));
    for input in [CovarianceInput::Dense, CovarianceInput::SparseToDense] {
        let c = g
            .marginal_covariance(
                &g.initial(),
                &[b],
                &CovarianceOptions {
                    input,
                    ..CovarianceOptions::default()
                },
            )
            .unwrap();
        assert_eq!(c.matrix.shape(), (6, 6));
        assert!((c.matrix - &expected).norm() < 1e-12);
    }
    let r = pose.rotation.0.to_rotation_matrix();
    let world = r.matrix() * expected.fixed_view::<3, 3>(0, 0) * r.matrix().transpose();
    assert!((world.trace() - 0.14).abs() < 1e-12); // Body-coordinate translation covariance rotates into world coordinates.
}
#[test]
fn robust_covariance_requires_explicit_irls_and_uses_joint_weights_once() {
    let mut g = FactorGraph::new();
    let id = g
        .add_variable(
            VariableKind::Euclidean(2),
            DVector::from_vec(vec![6.0, 8.0]),
        )
        .unwrap();
    g.add_analytic_factor_with_options(
        &[id],
        2,
        FactorOptions {
            noise: NoiseModel::isotropic(2.0).unwrap(),
            loss: Some(Loss::Huber(2.0)),
        },
        |v| Ok((v[0].clone(), vec![DMatrix::<f64>::identity(2, 2)])),
    )
    .unwrap();
    assert!(matches!(
        g.marginal_covariance(&g.initial(), &[id], &CovarianceOptions::default()),
        Err(Error::Evaluation(_))
    ));
    // Whitened norm=5, rho'=2/5. JᵀJ=.25I, IRLS information=.1I, covariance=10I.
    for input in [CovarianceInput::Dense, CovarianceInput::SparseToDense] {
        let c = g
            .marginal_covariance(
                &g.initial(),
                &[id],
                &CovarianceOptions {
                    input,
                    model: CovarianceModel::FrozenIrls {
                        fallback_loss: Loss::Cauchy(1.0),
                    },
                    ..CovarianceOptions::default()
                },
            )
            .unwrap();
        assert!((c.matrix - 10.0 * DMatrix::<f64>::identity(2, 2)).norm() < 1e-12);
    }
    let c = g
        .marginal_covariance(
            &DVector::zeros(2),
            &[id],
            &CovarianceOptions {
                model: CovarianceModel::FrozenIrls {
                    fallback_loss: Loss::Squared,
                },
                ..CovarianceOptions::default()
            },
        )
        .unwrap();
    assert!((c.matrix - 4.0 * DMatrix::<f64>::identity(2, 2)).norm() < 1e-12);
}
struct MatrixProblem {
    j: DMatrix<f64>,
    residual: DVector<f64>,
    blocks: Vec<LossBlock>,
}
impl LocalProblem for MatrixProblem {
    fn dimensions(&self) -> (usize, usize, usize) {
        (self.j.ncols(), self.j.ncols(), self.j.nrows())
    }
    fn evaluate_local(
        &self,
        _: &DVector<f64>,
        r: &mut DVector<f64>,
        j: &mut DMatrix<f64>,
    ) -> Result<(), Error> {
        *r = self.residual.clone();
        *j = self.j.clone();
        Ok(())
    }
    fn retract(&self, x: &DVector<f64>, d: &DVector<f64>) -> Result<DVector<f64>, Error> {
        Ok(x + d)
    }
    fn loss_blocks(&self) -> Vec<LossBlock> {
        self.blocks.clone()
    }
}
#[test]
fn irls_scalar_fallback_only_affects_uncovered_rows() {
    let p = MatrixProblem {
        j: DMatrix::from_row_slice(4, 2, &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]),
        residual: DVector::from_vec(vec![10.0, 3.0, 4.0, 5.0]),
        blocks: vec![LossBlock {
            rows: 1..3,
            loss: Loss::Huber(2.0),
        }],
    };
    let c = marginal_covariance(
        &p,
        &DVector::zeros(2),
        &[0, 1],
        &CovarianceOptions {
            model: CovarianceModel::FrozenIrls {
                fallback_loss: Loss::Cauchy(1.0),
            },
            ..CovarianceOptions::default()
        },
    )
    .unwrap();
    let expected = DMatrix::from_diagonal(&DVector::from_vec(vec![
        1.0 / (1.0 / 101.0 + 0.4),
        1.0 / (0.4 + 1.0 / 26.0),
    ]));
    assert!((c.matrix - expected).norm() < 1e-12);
}
#[test]
fn rank_threshold_and_extreme_scale_do_not_produce_damped_or_pseudo_inverse_covariance() {
    let p = MatrixProblem {
        j: DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1e-10])),
        residual: DVector::zeros(2),
        blocks: vec![],
    };
    let options = CovarianceOptions {
        relative_rank_tolerance: 1e-8,
        ..CovarianceOptions::default()
    };
    assert_eq!(
        marginal_covariance(&p, &DVector::zeros(2), &[0], &options).unwrap_err(),
        Error::RankDeficient {
            rank: 1,
            parameters: 2
        }
    );
    let c =
        marginal_covariance(&p, &DVector::zeros(2), &[1], &CovarianceOptions::default()).unwrap();
    assert!((c.matrix[(0, 0)] / 1e20 - 1.0).abs() < 1e-14);
    assert!((c.condition_number / 1e10 - 1.0).abs() < 1e-14);
    for scale in [1e-100, 1e100] {
        let p = MatrixProblem {
            j: DMatrix::<f64>::identity(2, 2) * scale,
            residual: DVector::zeros(2),
            blocks: vec![],
        };
        let c = marginal_covariance(
            &p,
            &DVector::zeros(2),
            &[0, 1],
            &CovarianceOptions::default(),
        )
        .unwrap();
        assert!((c.matrix[(0, 0)] * scale * scale - 1.0).abs() < 1e-14);
    }
    let zero = MatrixProblem {
        j: DMatrix::zeros(2, 2),
        residual: DVector::zeros(2),
        blocks: vec![],
    };
    assert_eq!(
        marginal_covariance(
            &zero,
            &DVector::zeros(2),
            &[0],
            &CovarianceOptions::default()
        )
        .unwrap_err(),
        Error::RankDeficient {
            rank: 0,
            parameters: 2
        }
    );
}
#[test]
fn invalid_selections_options_metadata_and_nonfinite_evaluations_are_errors() {
    let (g, x, _) = coupled();
    let point = g.initial();
    for indices in [vec![], vec![0, 0], vec![2]] {
        assert_eq!(
            marginal_covariance(&g, &point, &indices, &CovarianceOptions::default()).unwrap_err(),
            Error::InvalidDimensions
        );
    }
    assert_eq!(
        g.marginal_covariance(&point, &[x, x], &CovarianceOptions::default())
            .unwrap_err(),
        Error::InvalidDimensions
    );
    for options in [
        CovarianceOptions {
            variance_scale: 0.0,
            ..CovarianceOptions::default()
        },
        CovarianceOptions {
            relative_rank_tolerance: f64::NAN,
            ..CovarianceOptions::default()
        },
        CovarianceOptions {
            relative_rank_tolerance: 1.0,
            ..CovarianceOptions::default()
        },
        CovarianceOptions {
            model: CovarianceModel::FrozenIrls {
                fallback_loss: Loss::Huber(-1.0),
            },
            ..CovarianceOptions::default()
        },
    ] {
        assert_eq!(
            g.marginal_covariance(&point, &[x], &options).unwrap_err(),
            Error::InvalidOptions
        );
    }
    let mut p = MatrixProblem {
        j: DMatrix::<f64>::identity(2, 2),
        residual: DVector::zeros(2),
        blocks: vec![],
    };
    for rows in [Range { start: 0, end: 3 }, Range { start: 1, end: 1 }] {
        p.blocks = vec![LossBlock {
            rows,
            loss: Loss::Squared,
        }];
        assert_eq!(
            marginal_covariance(&p, &point, &[0], &CovarianceOptions::default()).unwrap_err(),
            Error::InvalidDimensions
        );
    }
    p.blocks.clear();
    p.j[(0, 0)] = f64::NAN;
    assert_eq!(
        marginal_covariance(&p, &point, &[0], &CovarianceOptions::default()).unwrap_err(),
        Error::NonFinite
    );
    p.j = DMatrix::<f64>::identity(2, 2);
    p.residual[0] = f64::INFINITY;
    assert_eq!(
        marginal_covariance(&p, &point, &[0], &CovarianceOptions::default()).unwrap_err(),
        Error::NonFinite
    );
    p.residual = DVector::zeros(2);
    assert!(matches!(
        marginal_covariance(
            &p,
            &point,
            &[0],
            &CovarianceOptions {
                input: CovarianceInput::SparseToDense,
                ..CovarianceOptions::default()
            }
        ),
        Err(Error::Evaluation(_))
    ));
}

#[test]
fn standalone_euclidean_residual_covariance_matches_known_sigma_and_explicit_scalar_loss() {
    use ropt::autodiff::{AutoDiff, Jet};
    let residual =
        AutoDiff::<1, 1, _>(|x: [Jet<1>; 1]| [(x[0] - Jet::constant(2.0)) / Jet::constant(0.2)]);
    let point = DVector::from_element(1, 1.0);
    let gaussian =
        marginal_covariance_euclidean(&residual, &point, &[0], &CovarianceOptions::default())
            .unwrap();
    assert!((gaussian.matrix[(0, 0)] - 0.04).abs() < 1e-14);
    let irls = marginal_covariance_euclidean(
        &residual,
        &point,
        &[0],
        &CovarianceOptions {
            model: CovarianceModel::FrozenIrls {
                fallback_loss: Loss::Huber(1.0),
            },
            ..CovarianceOptions::default()
        },
    )
    .unwrap();
    assert!((irls.matrix[(0, 0)] - 0.2).abs() < 1e-14);
}

#[test]
fn svd_iteration_limits_fail_explicitly() {
    let p = MatrixProblem {
        j: DMatrix::from_row_slice(
            4,
            4,
            &[
                1.0, 0.2, 0.1, 0.4, 0.7, 2.0, 0.3, -0.2, 0.5, 0.1, 3.0, 0.4, 0.6, -0.3, 0.8, 4.0,
            ],
        ),
        residual: DVector::zeros(4),
        blocks: vec![],
    };
    assert_eq!(
        marginal_covariance(
            &p,
            &DVector::zeros(4),
            &[0],
            &CovarianceOptions {
                svd_max_iterations: 0,
                ..CovarianceOptions::default()
            }
        )
        .unwrap_err(),
        Error::InvalidOptions
    );
    assert_eq!(
        marginal_covariance(
            &p,
            &DVector::zeros(4),
            &[0],
            &CovarianceOptions {
                svd_max_iterations: 1,
                ..CovarianceOptions::default()
            }
        )
        .unwrap_err(),
        Error::LinearSolve
    );
    assert!(
        marginal_covariance(&p, &DVector::zeros(4), &[0], &CovarianceOptions::default()).is_ok()
    );
}
