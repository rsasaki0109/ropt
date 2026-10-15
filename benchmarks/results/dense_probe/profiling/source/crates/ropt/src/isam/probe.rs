//! Current-estimate nonlinear probes without graph registration or CSR assembly.
use super::IsamNonlinearSummary;
use crate::{
    DVector, Error, GradientNorm,
    factor::check_factor_output,
    fixed_lag::{Key, SmootherFactor, WindowVariable},
    noise::WhiteningWorkspace,
    sparse::finite_vector,
};
use std::{collections::BTreeMap, ops::Range};

/// Connectivity/options are already validated by the committed update. Every
/// variable and callback output is validated again, including fixed blocks.
pub(super) fn summary(
    variables: &BTreeMap<Key, WindowVariable>,
    factors: &[SmootherFactor],
    order: &[Key],
    norm: GradientNorm,
) -> Result<IsamNonlinearSummary, Error> {
    let profile_entry = std::time::Instant::now();
    let mut profile = [0_u128; 7];
    let mut profile_samples = 0;
    let mut layout = BTreeMap::<Key, (&WindowVariable, Option<Range<usize>>)>::new();
    let mut columns = 0usize;
    for key in order {
        let variable = variables.get(key).ok_or(Error::InvalidDimensions)?;
        variable.kind.validate(&variable.value)?;
        let range = if !variable.fixed {
            let end = columns
                .checked_add(variable.kind.dimensions().1)
                .ok_or(Error::InvalidDimensions)?;
            let range = columns..end;
            columns = end;
            Some(range)
        } else {
            None
        };
        layout.insert(*key, (variable, range));
    }
    factors.iter().try_fold(0usize, |n, f| {
        n.checked_add(f.residuals).ok_or(Error::InvalidDimensions)
    })?;
    let mut cost = crate::solver::GaussianCostAccumulator::default();
    let mut gradient = DVector::<f64>::zeros(columns);
    let mut scales = DVector::<f64>::zeros(columns);
    let mut inputs = Vec::<DVector<f64>>::new();
    let mut connections = Vec::new();
    let mut slots = Vec::new();
    let mut whitening = WhiteningWorkspace::default();
    profile[0] = profile_entry.elapsed().as_nanos();
    for (factor_index, factor) in factors.iter().enumerate() {
        let mut sample = if factor_index % 17 == 0 {
            profile_samples += 1;
            Some(std::time::Instant::now())
        } else { None };
        inputs.resize_with(inputs.len().max(factor.keys.len()), || DVector::zeros(0));
        connections.clear();
        for key in &factor.keys {
            connections.push(layout.get(key).ok_or(Error::InvalidDimensions)?);
        }
        for (connection, buffer) in connections.iter().zip(&mut inputs) {
            let value = &connection.0.value;
            if buffer.len() != value.len() {
                buffer.resize_vertically_mut(value.len(), 0.);
            }
            buffer.copy_from(value);
        }
        if let Some(start) = sample { profile[1] += start.elapsed().as_nanos(); sample = Some(std::time::Instant::now()); }
        let (mut residual, mut blocks) = factor.evaluate(
            &inputs[..factor.keys.len()],
            connections.iter().map(|connection| connection.0.kind),
        )?;
        if let Some(start) = sample { profile[2] += start.elapsed().as_nanos(); sample = Some(std::time::Instant::now()); }
        check_factor_output(
            &residual,
            &blocks,
            factor.residuals,
            connections
                .iter()
                .map(|connection| connection.0.kind.dimensions().1),
        )?;
        factor
            .options
            .noise
            .whiten(&mut residual, &mut blocks, &mut whitening)?;
        if let Some(start) = sample { profile[3] += start.elapsed().as_nanos(); sample = Some(std::time::Instant::now()); }
        cost.add(&residual);
        if let Some(start) = sample { profile[4] += start.elapsed().as_nanos(); sample = Some(std::time::Instant::now()); }
        // Callbacks retain caller slot order. Reduction visits active blocks in
        // tangent-column order, exactly as canonical CSR row assembly did.
        slots.clear();
        slots.extend(
            connections
                .iter()
                .enumerate()
                .filter_map(|(slot, connection)| {
                    connection.1.as_ref().map(|range| (slot, range.clone()))
                }),
        );
        slots.sort_unstable_by_key(|(_, range)| range.start);
        for row in 0..factor.residuals {
            for (slot, range) in &slots {
                for column in 0..range.len() {
                    let value = blocks[*slot][(row, column)];
                    if value != 0. {
                        let global = range.start + column;
                        gradient[global] += value * residual[row];
                        if norm == GradientNorm::ColumnNormalized {
                            scales[global] = scales[global].hypot(value);
                        }
                    }
                }
            }
        }
        if let Some(start) = sample { profile[5] += start.elapsed().as_nanos(); }
    }
    let finish_start = std::time::Instant::now();
    // Keep the compensated Gaussian cost and final finite checks. No callback
    // is skipped because a running reduction overflowed; CSR checked afterwards.
    let cost = cost.finish()?;
    finite_vector(&gradient)?;
    finite_vector(&scales)?;
    for (component, &scale) in gradient.iter_mut().zip(&scales) {
        *component = if scale == 0. {
            component.abs()
        } else {
            component.abs() / scale
        };
    }
    finite_vector(&gradient)?;
    let mut variable_gradients = BTreeMap::new();
    let mut gradient_norm = 0_f64;
    for (key, (_, range)) in layout {
        if let Some(range) = range {
            let value = gradient.rows(range.start, range.len()).amax();
            gradient_norm = gradient_norm.max(value);
            variable_gradients.insert(key, value);
        }
    }
    profile[6] = finish_start.elapsed().as_nanos();
    eprintln!("probe_profile factors={} sampled={} norm={:?} layout={} lookup={} callback={} validate_whiten={} cost={} gradient={} finish={}", factors.len(), profile_samples, norm, profile[0], profile[1], profile[2], profile[3], profile[4], profile[5], profile[6]);
    Ok(IsamNonlinearSummary {
        cost,
        gradient_norm,
        norm,
        variable_gradients,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DMatrix, LocalProblem,
        factor::{FactorOptions, VariableKind},
        fixed_lag::build_graph,
        isam::{IsamOptions, IsamSmoother},
        noise::NoiseModel,
    };
    use std::{cell::Cell, rc::Rc};

    fn legacy(
        variables: &BTreeMap<Key, WindowVariable>,
        factors: &[SmootherFactor],
        order: &[Key],
        norm: GradientNorm,
    ) -> Result<IsamNonlinearSummary, Error> {
        let (graph, ids) = build_graph(variables, factors, order)?;
        let (r, j) = graph.evaluate_sparse(&graph.initial())?;
        let cost = -crate::solver::gaussian_reduction(&DVector::zeros(r.len()), &r)?;
        let g = j.transpose_multiply(&r)?;
        let mut scales = DVector::<f64>::zeros(g.len());
        if norm == GradientNorm::ColumnNormalized {
            for (_, c, value) in j.entries() {
                scales[c] = scales[c].hypot(value);
            }
        }
        finite_vector(&scales)?;
        let values = DVector::from_iterator(
            g.len(),
            g.iter()
                .zip(&scales)
                .map(|(&g, &s)| if s == 0. { g.abs() } else { g.abs() / s }),
        );
        finite_vector(&values)?;
        let mut result = IsamNonlinearSummary {
            cost,
            gradient_norm: 0.,
            norm,
            variable_gradients: BTreeMap::new(),
        };
        for (key, id) in ids {
            if let Some(range) = graph.tangent_range(id)? {
                let value = values.rows(range.start, range.len()).amax();
                result.gradient_norm = result.gradient_norm.max(value);
                result.variable_gradients.insert(key, value);
            }
        }
        Ok(result)
    }

    fn variable(key: Key, kind: VariableKind, values: &[f64], fixed: bool) -> WindowVariable {
        WindowVariable {
            key,
            kind,
            value: DVector::from_column_slice(values),
            fixed,
            timestamp: None,
        }
    }

    #[test]
    fn direct_reductions_match_csr_bits_for_mixed_manifolds_fixed_slots_and_noise() {
        let order = vec![8, 11, 123, 7, 100, 99, 200];
        let variables = [
            variable(8, VariableKind::SO3, &[0.5, 0.5, 0.5, 0.5], false),
            variable(11, VariableKind::Euclidean(1), &[1.1], true),
            variable(123, VariableKind::Euclidean(2), &[5., -3.], false),
            variable(7, VariableKind::Euclidean(2), &[0.8, -1.2], false),
            variable(100, VariableKind::Euclidean(1), &[0.], false),
            variable(99, VariableKind::SO2, &[0.35], false),
            variable(200, VariableKind::SO3, &[1., 0., 0., 0.], true),
        ]
        .into_iter()
        .map(|v| (v.key, v))
        .collect::<BTreeMap<_, _>>();
        let information = DMatrix::from_row_slice(2, 2, &[4., 1., 1., 3.]);
        let factors = vec![
            SmootherFactor::analytic(
                &[11, 7, 8, 99],
                2,
                FactorOptions {
                    noise: NoiseModel::from_information(information.clone()).unwrap(),
                    ..Default::default()
                },
                |v| {
                    let q = &v[2];
                    Ok((
                        DVector::from_vec(vec![
                            q[0] + 2. * v[1][0] - 3. * v[1][1] - 0.3,
                            q[1] + v[3][0].sin() + v[0][0] - 0.7,
                        ]),
                        vec![
                            DMatrix::from_column_slice(2, 1, &[0., 1.]),
                            DMatrix::from_row_slice(2, 2, &[2., -3., 0., 0.]),
                            DMatrix::from_row_slice(
                                2,
                                3,
                                &[
                                    -0.5 * q[1],
                                    -0.5 * q[2],
                                    -0.5 * q[3],
                                    0.5 * q[0],
                                    -0.5 * q[3],
                                    0.5 * q[2],
                                ],
                            ),
                            DMatrix::from_column_slice(2, 1, &[0., v[3][0].cos()]),
                        ],
                    ))
                },
            ),
            SmootherFactor::analytic(&[100], 2, Default::default(), |v| {
                Ok((
                    DVector::from_vec(vec![1e-150 * v[0][0], 1e150 * v[0][0]]),
                    vec![DMatrix::from_column_slice(2, 1, &[1e-150, 1e150])],
                ))
            }),
            SmootherFactor::analytic(
                &[11],
                2,
                FactorOptions {
                    noise: NoiseModel::diagonal(DVector::from_vec(vec![2., 0.5])).unwrap(),
                    ..Default::default()
                },
                |_| {
                    Ok((
                        DVector::from_vec(vec![0.4, -0.3]),
                        vec![DMatrix::zeros(2, 1)],
                    ))
                },
            ),
        ];
        for norm in [GradientNorm::Absolute, GradientNorm::ColumnNormalized] {
            let direct = summary(&variables, &factors, &order, norm).unwrap();
            let csr = legacy(&variables, &factors, &order, norm).unwrap();
            assert_eq!(direct.cost.to_bits(), csr.cost.to_bits());
            assert_eq!(direct.gradient_norm.to_bits(), csr.gradient_norm.to_bits());
            for (key, value) in &csr.variable_gradients {
                assert_eq!(direct.variable_gradients[key].to_bits(), value.to_bits());
            }
            assert_eq!(direct.variable_gradients[&123], 0.);
            assert_eq!(direct.variable_gradients[&100], 0.);
            assert!(!direct.variable_gradients.contains_key(&11));
            let r = DVector::from_vec(vec![5.4, 0.5 + 0.35_f64.sin() + 1.1 - 0.7]);
            let expected = 0.5 * (r.transpose() * &information * &r)[(0, 0)] + 0.2;
            assert!((direct.cost - expected).abs() < 2e-13);
        }
        let mut invalid = variables.clone();
        invalid.get_mut(&200).unwrap().value[0] = 2.;
        assert_eq!(
            summary(&invalid, &factors, &order, GradientNorm::Absolute).unwrap_err(),
            legacy(&invalid, &factors, &order, GradientNorm::Absolute).unwrap_err()
        );
    }

    #[test]
    fn overflowing_cost_does_not_skip_later_callbacks_or_replace_their_errors() {
        let variables = [(1, variable(1, VariableKind::Euclidean(1), &[0.], false))]
            .into_iter()
            .collect();
        for later_failure in [false, true] {
            let calls = Rc::new(Cell::new(0));
            let counter = calls.clone();
            let factors = vec![
                SmootherFactor::analytic(&[1], 1, Default::default(), |_| {
                    Ok((DVector::from_element(1, 1e200), vec![DMatrix::zeros(1, 1)]))
                }),
                SmootherFactor::analytic(&[1], 1, Default::default(), move |_| {
                    counter.set(counter.get() + 1);
                    if later_failure {
                        Err(Error::Evaluation("later callback".into()))
                    } else {
                        Ok((DVector::zeros(1), vec![DMatrix::identity(1, 1)]))
                    }
                }),
            ];
            let direct = summary(&variables, &factors, &[1], GradientNorm::Absolute).unwrap_err();
            assert_eq!(calls.get(), 1);
            let expected = if later_failure {
                Error::Evaluation("later callback".into())
            } else {
                Error::NonFinite
            };
            assert_eq!(direct, expected);
            assert_eq!(
                legacy(&variables, &factors, &[1], GradientNorm::Absolute).unwrap_err(),
                expected
            );
            assert_eq!(calls.get(), 2);
        }
    }

    #[test]
    fn output_errors_and_overflow_match_csr_including_fixed_blocks() {
        let variables = [
            variable(2, VariableKind::Euclidean(1), &[0.], true),
            variable(1, VariableKind::Euclidean(1), &[0.], false),
        ]
        .into_iter()
        .map(|v| (v.key, v))
        .collect::<BTreeMap<_, _>>();
        for mode in 0..4 {
            let factors = vec![SmootherFactor::analytic(
                &[2, 1],
                1,
                Default::default(),
                move |_| {
                    let residual = DVector::from_element(
                        if mode == 0 { 2 } else { 1 },
                        if mode == 3 { 1e308 } else { 1. },
                    );
                    let fixed = if mode == 1 {
                        DMatrix::zeros(1, 2)
                    } else {
                        DMatrix::from_element(1, 1, if mode == 2 { f64::NAN } else { 1. })
                    };
                    Ok((residual, vec![fixed, DMatrix::identity(1, 1)]))
                },
            )];
            assert_eq!(
                summary(
                    &variables,
                    &factors,
                    &[2, 1],
                    GradientNorm::ColumnNormalized
                )
                .unwrap_err(),
                legacy(
                    &variables,
                    &factors,
                    &[2, 1],
                    GradientNorm::ColumnNormalized
                )
                .unwrap_err()
            );
        }
    }

    #[test]
    fn mixed_autodiff_keeps_graph_callback_semantics_and_current_energy() {
        let calls = Rc::new(Cell::new(0));
        let counter = calls.clone();
        let mut smoother = IsamSmoother::new(IsamOptions {
            relinearization_threshold: 10.,
            relinearization_interval: 1000,
            ..Default::default()
        })
        .unwrap();
        smoother
            .update(
                crate::fixed_lag::SmootherUpdate {
                    variables: vec![
                        variable(2, VariableKind::Euclidean(1), &[1.], true),
                        variable(8, VariableKind::Euclidean(1), &[1.], false),
                    ],
                    factors: vec![SmootherFactor::autodiff(
                        &[8, 2],
                        1,
                        1,
                        Default::default(),
                        move |v| {
                            counter.set(counter.get() + 1);
                            Ok(vec![v[0][0].clone() * v[0][0].clone() - v[1][0].clone()])
                        },
                    )],
                    ..Default::default()
                },
                false,
            )
            .unwrap();
        smoother
            .update(
                crate::fixed_lag::SmootherUpdate {
                    factors: vec![SmootherFactor::analytic(&[8], 1, Default::default(), |v| {
                        Ok((
                            DVector::from_element(1, v[0][0] - 3.),
                            vec![DMatrix::identity(1, 1)],
                        ))
                    })],
                    ..Default::default()
                },
                false,
            )
            .unwrap();
        let x = smoother.estimate(8).unwrap()[0];
        for norm in [GradientNorm::Absolute, GradientNorm::ColumnNormalized] {
            let before = calls.get();
            let result = smoother.nonlinear_summary(norm).unwrap();
            let actual_calls = calls.get() - before;
            let before = calls.get();
            let reference = legacy(
                &smoother.variables,
                &smoother.factors,
                &smoother.order,
                norm,
            )
            .unwrap();
            assert_eq!(actual_calls, calls.get() - before);
            assert!(actual_calls > 0);
            assert_eq!(result.cost.to_bits(), reference.cost.to_bits());
            assert_eq!(
                result.gradient_norm.to_bits(),
                reference.gradient_norm.to_bits()
            );
            let cost = 0.5 * ((x * x - 1.).powi(2) + (x - 3.).powi(2));
            let gradient = (2. * x * (x * x - 1.) + x - 3.).abs();
            let expected = if norm == GradientNorm::Absolute {
                gradient
            } else {
                gradient / (4. * x * x + 1.).sqrt()
            };
            assert!((result.cost - cost).abs() < 1e-14);
            assert!((result.gradient_norm - expected).abs() < 1e-14);
        }
    }
    #[test]
    fn autodiff_chunks_match_csr_for_every_manifold_and_fixed_layout() {
        let cases = [
            (VariableKind::Euclidean(2), vec![0.4, -0.7]),
            (VariableKind::SO2, vec![0.3]),
            (VariableKind::SO3, vec![0.5, 0.5, 0.5, 0.5]),
            (VariableKind::SE2, vec![0.4, -0.7, 0.3]),
            (VariableKind::SE2Cartesian, vec![0.4, -0.7, 0.3]),
            (VariableKind::SE3, vec![0.4, -0.7, 0.1, 0.5, 0.5, 0.5, 0.5]),
        ];
        for (kind, value) in cases {
            for fixed in [false, true] {
                for chunk in [1, 2, 32] {
                    let variables = [
                        variable(20, kind, &value, fixed),
                        variable(9, VariableKind::SO3, &[1., 0., 0., 0.], true),
                        variable(4, VariableKind::Euclidean(1), &[3.], false),
                    ]
                    .into_iter()
                    .map(|v| (v.key, v))
                    .collect();
                    let calls = Rc::new(Cell::new(0));
                    let counter = calls.clone();
                    let factors = vec![SmootherFactor::autodiff(
                        &[9, 20],
                        2,
                        chunk,
                        FactorOptions {
                            noise: NoiseModel::from_information(DMatrix::from_row_slice(
                                2,
                                2,
                                &[4., 1., 1., 3.],
                            ))
                            .unwrap(),
                            ..Default::default()
                        },
                        move |v| {
                            counter.set(counter.get() + 1);
                            assert_eq!(v.len(), 2);
                            assert_eq!(v[0].len(), 4);
                            let x = &v[1][0];
                            let y = v[1].last().unwrap();
                            Ok(vec![x * x + y.sin() - &v[0][0], x.exp() + y * y - 0.3])
                        },
                    )];
                    for norm in [GradientNorm::Absolute, GradientNorm::ColumnNormalized] {
                        let before = calls.get();
                        let direct = summary(&variables, &factors, &[20, 4, 9], norm).unwrap();
                        let direct_calls = calls.get() - before;
                        let before = calls.get();
                        let csr = legacy(&variables, &factors, &[20, 4, 9], norm).unwrap();
                        assert_eq!(direct_calls, calls.get() - before);
                        // Fixed tangent blocks still participate in derivative chunks.
                        assert_eq!(direct_calls, (3 + kind.dimensions().1).div_ceil(chunk));
                        assert_eq!(direct.cost.to_bits(), csr.cost.to_bits());
                        assert_eq!(direct.gradient_norm.to_bits(), csr.gradient_norm.to_bits());
                        assert_eq!(
                            direct.variable_gradients.len(),
                            csr.variable_gradients.len()
                        );
                        for (key, value) in csr.variable_gradients {
                            assert_eq!(direct.variable_gradients[&key].to_bits(), value.to_bits());
                        }
                        assert_eq!(direct.variable_gradients[&4], 0.);
                        assert_eq!(direct.variable_gradients.contains_key(&20), !fixed);
                    }
                }
            }
        }
    }

    #[test]
    fn autodiff_failures_match_csr_without_changing_callback_counts() {
        use crate::autodiff::DynamicJet as J;
        let variables = [
            variable(2, VariableKind::Euclidean(1), &[0.], true),
            variable(1, VariableKind::Euclidean(1), &[0.], false),
        ]
        .into_iter()
        .map(|v| (v.key, v))
        .collect();
        for mode in 0..5 {
            let calls = Rc::new(Cell::new(0));
            let counter = calls.clone();
            let factors = vec![SmootherFactor::autodiff(
                &[2, 1],
                1,
                1,
                Default::default(),
                move |v| {
                    counter.set(counter.get() + 1);
                    match mode {
                        0 => Ok(vec![]),
                        1 => Ok(vec![J::constant(f64::NAN)]),
                        2 => Ok(vec![J::variable(0., 0, 7)?]),
                        3 => Ok(vec![v[0][0].sqrt()]), // Invalid fixed-block derivative.
                        _ => Err(Error::Evaluation("probe failure".into())),
                    }
                },
            )];
            let before = calls.get();
            let direct =
                summary(&variables, &factors, &[1, 2], GradientNorm::Absolute).unwrap_err();
            let direct_calls = calls.get() - before;
            let before = calls.get();
            assert_eq!(
                direct,
                legacy(&variables, &factors, &[1, 2], GradientNorm::Absolute).unwrap_err()
            );
            assert_eq!(direct_calls, calls.get() - before);
        }
    }
}
