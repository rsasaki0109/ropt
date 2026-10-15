use ropt::{sparse::CsrMatrix, *};

struct ScaledParameter(f64);
impl LocalProblem for ScaledParameter {
    fn dimensions(&self) -> (usize, usize, usize) {
        (1, 1, 2)
    }
    fn evaluate_local(
        &self,
        x: &DVector<f64>,
        r: &mut DVector<f64>,
        j: &mut DMatrix<f64>,
    ) -> Result<(), Error> {
        r[0] = self.0 * x[0] - 2.;
        r[1] = 4.;
        j.fill(0.);
        j[(0, 0)] = self.0;
        Ok(())
    }
    fn evaluate_sparse(&self, x: &DVector<f64>) -> Result<(DVector<f64>, CsrMatrix), Error> {
        Ok((
            DVector::from_vec(vec![self.0 * x[0] - 2., 4.]),
            CsrMatrix::from_triplets(2, 1, vec![(0, 0, self.0)])?,
        ))
    }
    fn retract(&self, x: &DVector<f64>, delta: &DVector<f64>) -> Result<DVector<f64>, Error> {
        Ok(x + delta)
    }
}

#[test]
fn column_normalization_has_the_same_stopping_measure_under_parameter_unit_changes() {
    for scale in [1e-3, 1., 1e3] {
        for backend in [
            LinearSolver::DenseQr(Default::default()),
            LinearSolver::SparseLdl(Default::default()),
        ] {
            let initial = DVector::from_element(1, 1. / scale);
            let report = solve_local(
                &ScaledParameter(scale),
                initial.clone(),
                &Options {
                    gradient_norm: GradientNorm::ColumnNormalized,
                    gradient_tolerance: 1.1,
                    linear_solver: backend,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(report.termination, Termination::Gradient);
            assert_eq!(report.x, initial);
            assert!(report.iterations.is_empty());
        }
    }
    assert_eq!(Options::default().gradient_norm, GradientNorm::Absolute);
    let report = solve_local(
        &ScaledParameter(1e3),
        DVector::from_element(1, 1e-3),
        &Options {
            gradient_tolerance: 1.1,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!report.iterations.is_empty());
    assert_eq!(report.iterations[0].gradient_norm, 1e3);
}

#[test]
fn reported_normalized_gradient_matches_original_column_projection() {
    for scale in [1e-3, 1., 1e3] {
        let report = solve_local(
            &ScaledParameter(scale),
            DVector::zeros(1),
            &Options {
                gradient_norm: GradientNorm::ColumnNormalized,
                gradient_tolerance: 1e-10,
                cost_tolerance: 0.,
                step_tolerance: 0.,
                max_iterations: 100,
                linear_solver: LinearSolver::SparseLdl(Default::default()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(report.iterations[0].gradient_norm, 2.);
        assert!((scale * report.x[0] - 2.).abs() < 1e-8);
        assert!(!matches!(
            report.termination,
            Termination::NoProgress | Termination::MaxIterations
        ));
    }
}
