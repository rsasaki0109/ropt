//! Real SE2 replay with explicit runtime AD recipes and unchanged quality policy.
#[allow(dead_code)]
#[path = "support/real_selective_se2.rs"]
mod benchmark;
#[global_allocator]
static ALLOCATOR: benchmark::Counter = benchmark::Counter;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Two benchmark-only selectors; the six replay arguments follow unchanged.
    let chunk = std::env::var("ROPT_BENCH_AD_CHUNK")?.parse()?;
    let stride = std::env::var("ROPT_BENCH_AD_STRIDE")?.parse()?;
    benchmark::run_autodiff(chunk, stride)
}

#[cfg(test)]
#[allow(dead_code)]
#[path = "support/real_se2_benchmark.rs"]
mod fixture;

#[cfg(test)]
mod tests {
    use super::fixture;
    use ropt::{factor::*, fixed_lag::*, isam::*, pose_graph::*, *};

    #[test]
    fn parser_preserves_original_problem_across_explicit_ad_selections() {
        let path =
            std::env::temp_dir().join(format!("ropt-ad-replay-{}.stream", std::process::id()));
        std::fs::write(&path, "ROOT 0 0 0\nFRAME 1 1 .2 .1 1\nEDGE 0 1 1 0 0 4 .2 .1 5 .3 6\nFRAME 2 2 .4 .2 2\nEDGE 1 2 1 0 0 4 .2 .1 5 .3 6\nEDGE 0 2 2.1 .1 .05 4 .2 .1 5 .3 6\n").unwrap();
        let mut reference: Option<(Vec<DVector<f64>>, f64)> = None;
        for selection in [None, Some((6, 2)), Some((1, 1))] {
            let (root, frames) = fixture::load_with_differentiation(
                path.to_str().unwrap(),
                |r, a, b| r.linearize_ambient(a.as_slice(), b.as_slice()),
                VariableKind::SE2Cartesian,
                selection,
            )
            .unwrap();
            assert_eq!(root.value.as_slice(), &[0., 0., 0.]);
            assert!(root.fixed);
            assert_eq!(frames.len(), 2);
            assert_eq!(frames[0].variable.value.as_slice(), &[1., 0.2, 0.1]);
            assert_eq!(frames[1].variable.value.as_slice(), &[2., 0.4, 0.2]);
            assert_eq!((frames[0].factors.len(), frames[1].factors.len()), (1, 2));
            let mut smoother = IsamSmoother::new(IsamOptions::default()).unwrap();
            smoother
                .update(
                    SmootherUpdate {
                        variables: vec![root],
                        ..Default::default()
                    },
                    false,
                )
                .unwrap();
            for frame in frames {
                smoother
                    .update(
                        SmootherUpdate {
                            variables: vec![frame.variable],
                            factors: frame.factors,
                            ..Default::default()
                        },
                        true,
                    )
                    .unwrap();
            }
            let values: Vec<_> = (0..3)
                .map(|k| smoother.estimate(k).unwrap().clone())
                .collect();
            let summary = smoother
                .nonlinear_summary(GradientNorm::ColumnNormalized)
                .unwrap();
            assert!(summary.gradient_norm < 1e-7);
            assert_eq!(values[0].as_slice(), &[0., 0., 0.]);
            // Physical objective independently uses the original measurements/information.
            let information =
                DMatrix::from_row_slice(3, 3, &[4., 0.2, 0.1, 0.2, 5., 0.3, 0.1, 0.3, 6.]);
            let mut cost = 0.;
            for (a, b, z) in [
                (0, 1, [1., 0., 0.]),
                (1, 2, [1., 0., 0.]),
                (0, 2, [2.1, 0.1, 0.05]),
            ] {
                let measurement = read_se2(&DVector::from_column_slice(&z)).unwrap();
                let r = RelativePose2::new(measurement)
                    .unwrap()
                    .residual(read_se2(&values[a]).unwrap(), read_se2(&values[b]).unwrap())
                    .unwrap();
                cost += 0.5 * (r.transpose() * &information * r)[(0, 0)];
            }
            assert!((summary.cost - cost).abs() < 1e-12);
            if let Some((previous, previous_cost)) = &reference {
                assert!((summary.cost - previous_cost).abs() < 1e-12);
                for (a, b) in values.iter().zip(previous) {
                    assert!((a - b).norm() < 1e-8);
                }
            } else {
                reference = Some((values, summary.cost));
            }
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn invalid_ad_selection_fails_before_loading_a_file() {
        for selection in [Some((0, 1)), Some((6, 0))] {
            let error = fixture::load_with_differentiation(
                "not-a-stream",
                |r, a, b| r.linearize_ambient(a.as_slice(), b.as_slice()),
                VariableKind::SE2Cartesian,
                selection,
            )
            .err()
            .unwrap();
            assert!(matches!(
                error.downcast_ref::<Error>(),
                Some(Error::InvalidOptions)
            ));
        }
    }
}
