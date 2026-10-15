use ropt::{sparse::CsrMatrix, *};

struct OffsetProblem {
    loss: Loss,
    reverse: bool,
    wrong_jacobian: bool,
}
impl LocalProblem for OffsetProblem {
    fn dimensions(&self) -> (usize, usize, usize) {
        (1, 1, 4)
    }
    fn loss_blocks(&self) -> Vec<LossBlock> {
        vec![LossBlock {
            rows: 0..2,
            loss: self.loss,
        }]
    }
    fn evaluate_local(
        &self,
        x: &DVector<f64>,
        r: &mut DVector<f64>,
        j: &mut DMatrix<f64>,
    ) -> Result<(), Error> {
        // Large fixed scalar-loss terms mask joint-loss improvements. Keep the
        // active rows first so this isolates objective arithmetic rather than
        // cancellation in an orthogonal solve against a huge constant RHS.
        let (a, b) = if self.reverse { (4., 3.) } else { (3., 4.) };
        r.copy_from(&DVector::from_vec(vec![
            a * (x[0] - 1.),
            b * (x[0] - 1.),
            1e9,
            1e9,
        ]));
        j.fill(0.);
        let sign = if self.wrong_jacobian { -1. } else { 1. };
        j[(0, 0)] = sign * a;
        j[(1, 0)] = sign * b;
        Ok(())
    }
    fn evaluate_sparse(&self, x: &DVector<f64>) -> Result<(DVector<f64>, CsrMatrix), Error> {
        let mut r = DVector::zeros(4);
        let mut j = DMatrix::zeros(4, 1);
        self.evaluate_local(x, &mut r, &mut j)?;
        Ok((
            r,
            CsrMatrix::from_triplets(4, 1, vec![(0, 0, j[(0, 0)]), (1, 0, j[(1, 0)])])?,
        ))
    }
    fn retract(&self, x: &DVector<f64>, step: &DVector<f64>) -> Result<DVector<f64>, Error> {
        Ok(x + step)
    }
}

#[test]
fn constant_offsets_cannot_mask_scalar_or_joint_true_loss_improvements() {
    for loss in [Loss::Squared, Loss::Huber(1.), Loss::Cauchy(1.)] {
        for reverse in [false, true] {
            let problem = OffsetProblem {
                loss,
                reverse,
                wrong_jacobian: false,
            };
            for backend in [
                LinearSolver::Dense,
                LinearSolver::DenseQr(Default::default()),
                LinearSolver::SparseLdl(Default::default()),
            ] {
                let report = solve_local(
                    &problem,
                    DVector::zeros(1),
                    &Options {
                        linear_solver: backend,
                        cost_tolerance: 0.,
                        max_iterations: 50,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert!((report.x[0] - 1.).abs() < 1e-8, "{loss:?}: {report:?}");
                assert!(
                    report
                        .iterations
                        .iter()
                        .any(|i| i.accepted && i.cost == i.trial_cost && i.actual_reduction > 0.)
                );
                assert!(
                    report
                        .iterations
                        .iter()
                        .all(|i| !i.accepted || (i.actual_reduction > 0. && i.gain_ratio > 0.))
                );
                assert!(!matches!(
                    report.termination,
                    Termination::NoProgress | Termination::MaxIterations
                ));
            }
        }
    }
}

#[test]
fn increasing_true_loss_is_rejected_even_when_aggregate_cost_is_unchanged() {
    for loss in [Loss::Squared, Loss::Huber(1.), Loss::Cauchy(1.)] {
        let problem = OffsetProblem {
            loss,
            reverse: false,
            wrong_jacobian: true,
        };
        let report = solve_local(
            &problem,
            DVector::zeros(1),
            &Options {
                max_iterations: 1,
                cost_tolerance: 0.,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(report.x[0], 0.);
        let i = &report.iterations[0];
        assert!(!i.accepted && i.actual_reduction < 0. && i.gain_ratio < 0.);
    }
}

struct ScalarOffset;
impl Residual for ScalarOffset {
    fn dimensions(&self) -> (usize, usize) {
        (1, 2)
    }
    fn evaluate(
        &self,
        x: &DVector<f64>,
        r: &mut DVector<f64>,
        j: &mut DMatrix<f64>,
    ) -> Result<(), Error> {
        r[0] = x[0] - 1.;
        r[1] = 1e8;
        j[(0, 0)] = 1.;
        j[(1, 0)] = 0.;
        Ok(())
    }
}
#[test]
fn gauss_newton_can_accept_a_decrease_hidden_by_a_constant_cost() {
    let report = solve(
        &ScalarOffset,
        DVector::zeros(1),
        &Options {
            algorithm: Algorithm::GaussNewton,
            cost_tolerance: 0.,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(report.x[0], 1.);
    let i = &report.iterations[0];
    assert_eq!(i.cost, i.trial_cost);
    assert_eq!(i.actual_reduction, 0.5);
    assert!(i.accepted);
}
