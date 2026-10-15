use ropt::{factor::VariableKind, fixed_lag::*, isam::*, *};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

type Trace = Rc<RefCell<Vec<(u64, f64)>>>;

struct Harness {
    smoother: IsamSmoother,
    trace: Trace,
    fail_at: Rc<Cell<Option<usize>>>,
}

impl Harness {
    fn factor(&self, key: u64, target: f64) -> SmootherFactor {
        let trace = self.trace.clone();
        let fail_at = self.fail_at.clone();
        SmootherFactor::analytic(&[key], 1, Default::default(), move |v| {
            let index = trace.borrow().len();
            trace.borrow_mut().push((key, v[0][0]));
            if fail_at.get() == Some(index) {
                return Err(Error::Evaluation("injected refinement failure".into()));
            }
            Ok((
                DVector::from_element(1, v[0][0].powi(2) - target),
                vec![DMatrix::from_element(1, 1, 2. * v[0][0])],
            ))
        })
    }

    fn new(editable: bool, refine: bool) -> Self {
        let smoother = IsamSmoother::new(IsamOptions {
            relinearization_interval: 2,
            batch_solver: Options {
                algorithm: Algorithm::GaussNewton,
                linear_solver: LinearSolver::SparseQr(Default::default()),
                cost_tolerance: 1e-5,
                gradient_tolerance: 1e-12,
                step_tolerance: 1e-14,
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap();
        let smoother = if editable {
            smoother.with_selective_bayes_tree(4).unwrap()
        } else {
            smoother.with_selective_relinearization(4).unwrap()
        };
        let smoother = if refine {
            smoother.with_final_refinement(4).unwrap()
        } else {
            smoother
        };
        let mut harness = Self {
            smoother,
            trace: Rc::new(RefCell::new(Vec::new())),
            fail_at: Rc::new(Cell::new(None)),
        };
        let initial = SmootherUpdate {
            variables: vec![variable(0, 1.)],
            factors: vec![harness.factor(0, 1.)],
            ..Default::default()
        };
        harness.smoother.update(initial, false).unwrap();
        harness.trace.borrow_mut().clear();
        harness
    }

    fn update(&self) -> SmootherUpdate {
        SmootherUpdate {
            variables: vec![variable(7, 2.)],
            factors: vec![self.factor(0, 4.), self.factor(7, 4.)],
            ..Default::default()
        }
    }
}

fn variable(key: u64, value: f64) -> WindowVariable {
    WindowVariable {
        key,
        kind: VariableKind::Euclidean(1),
        value: DVector::from_element(1, value),
        timestamp: None,
        fixed: false,
    }
}

#[test]
fn failed_refinement_preserves_state_and_schedule_then_retries_cleanly() {
    for editable in [false, true] {
        // Calibrate the batch-only callback prefix instead of hard-coding an
        // iteration/evaluation count. The next callback belongs to refinement.
        let mut batch = Harness::new(editable, false);
        batch.smoother.update(batch.update(), true).unwrap();
        let batch_trace = batch.trace.borrow().clone();
        let mut reference = Harness::new(editable, true);
        let expected = reference.smoother.update(reference.update(), true).unwrap();
        assert!(expected.refinement_steps > 0);
        assert!(reference.trace.borrow().len() > batch_trace.len());
        assert!(reference.trace.borrow().starts_with(&batch_trace));

        let successful_trace = reference.trace.borrow().clone();
        for failure_index in batch_trace.len()..successful_trace.len() {
            let mut failing = Harness::new(editable, true);
            failing.fail_at.set(Some(failure_index));
            let error = failing.smoother.update(failing.update(), true).unwrap_err();
            assert!(
                matches!(error, Error::Evaluation(ref message) if message == "injected refinement failure")
            );
            assert_eq!(failing.trace.borrow().len(), failure_index + 1);
            assert_eq!(
                &failing.trace.borrow()[..failure_index],
                &successful_trace[..failure_index]
            );
            assert_eq!(failing.smoother.factor_count(), 1);
            assert_eq!(failing.smoother.estimate(0).unwrap()[0], 1.);
            assert!(failing.smoother.estimate(7).is_none());

            failing.fail_at.set(None);
            failing.trace.borrow_mut().clear();
            let retried = failing.smoother.update(failing.update(), false).unwrap();
            assert_eq!(
                retried.relinearization_reason,
                Some(IsamRelinearizationReason::Interval)
            );
            assert_eq!(retried.estimates, expected.estimates);
            assert_eq!(retried.refinement_steps, expected.refinement_steps);
            assert_eq!(retried.linearized_cost, expected.linearized_cost);
            assert_eq!(failing.smoother.factor_count(), 3);
            let next = failing.smoother.update(Default::default(), false).unwrap();
            assert_eq!(next.relinearization_reason, None);
            assert_eq!(next.estimates, retried.estimates);
        }
    }
}
