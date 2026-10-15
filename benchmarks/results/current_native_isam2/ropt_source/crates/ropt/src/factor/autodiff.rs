//! Shared tangent-space AD evaluation for graphs and current-estimate probes.
use super::VariableKind;
use crate::{
    DMatrix, DVector, Error,
    autodiff::{DynamicAutoDiff, DynamicJet},
};

pub(crate) struct AutoDiffLayout {
    kinds: Vec<VariableKind>,
    dimensions: Vec<usize>,
    parameters: usize,
    residuals: usize,
    chunk_size: usize,
}

impl AutoDiffLayout {
    pub(crate) fn new(
        kinds: Vec<VariableKind>,
        residuals: usize,
        chunk_size: usize,
    ) -> Result<Self, Error> {
        if chunk_size == 0 {
            return Err(Error::InvalidOptions);
        }
        let dimensions: Vec<_> = kinds.iter().map(|kind| kind.dimensions().1).collect();
        let parameters = dimensions.iter().sum();
        Ok(Self {
            kinds,
            dimensions,
            parameters,
            residuals,
            chunk_size,
        })
    }

    pub(crate) fn evaluate<F>(
        &self,
        values: &[DVector<f64>],
        function: &F,
    ) -> Result<(DVector<f64>, Vec<DMatrix<f64>>), Error>
    where
        F: Fn(&[&[DynamicJet]]) -> Result<Vec<DynamicJet>, Error> + ?Sized,
    {
        let residual = DynamicAutoDiff::new(self.parameters, self.residuals, |inputs| {
            match (self.kinds.as_slice(), values) {
                ([kind], [value]) => {
                    let block = kind.retract_autodiff(value, inputs)?;
                    return function(&[block.as_slice()]);
                }
                ([first, second], [a, b]) => {
                    let n = first.dimensions().1;
                    let a = first.retract_autodiff(a, &inputs[..n])?;
                    let b = second.retract_autodiff(b, &inputs[n..])?;
                    return function(&[a.as_slice(), b.as_slice()]);
                }
                _ => {}
            }
            let mut start = 0;
            let blocks = self
                .kinds
                .iter()
                .zip(values)
                .map(|(&kind, value)| {
                    let n = kind.dimensions().1;
                    let block = kind.retract_autodiff(value, &inputs[start..start + n]);
                    start += n;
                    block
                })
                .collect::<Result<Vec<_>, Error>>()?;
            function(&blocks.iter().map(Vec::as_slice).collect::<Vec<_>>())
        })?
        .with_chunk_size(self.chunk_size)?;
        residual.evaluate_blocks_at_zero(&self.dimensions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    type JetBits = (u64, Vec<u64>);
    type InputBits = Vec<Vec<JetBits>>;
    fn bits(inputs: &[&[DynamicJet]]) -> InputBits {
        inputs
            .iter()
            .map(|block| {
                block
                    .iter()
                    .map(|jet| {
                        (
                            jet.value().to_bits(),
                            jet.partials()
                                .unwrap()
                                .iter()
                                .map(|v| v.to_bits())
                                .collect(),
                        )
                    })
                    .collect()
            })
            .collect()
    }
    fn residual(inputs: &[&[DynamicJet]]) -> Result<Vec<DynamicJet>, Error> {
        let mut a = DynamicJet::constant(-0.);
        let mut b = DynamicJet::constant(0.);
        for (slot, block) in inputs.iter().enumerate() {
            for value in *block {
                a = a + (slot + 1) as f64 * value;
                b = b + value.sin() + value * value;
            }
        }
        Ok(vec![a, b])
    }

    #[test]
    fn inline_and_general_slots_match_heap_reference_bits_and_callback_trace() {
        let euclidean = (VariableKind::Euclidean(2), vec![0.3, -0.2]);
        let pose = (VariableKind::SE2Cartesian, vec![1., -2., 0.7]);
        let rotation = (VariableKind::SO3, vec![1., 0., 0., 0.]);
        let cases = [
            vec![euclidean.clone()],
            vec![pose.clone()],
            vec![rotation.clone()],
            vec![euclidean.clone(), pose.clone()],
            vec![rotation.clone(), euclidean.clone()],
            vec![pose.clone(), rotation.clone(), euclidean.clone()],
        ];
        for case in cases {
            let kinds: Vec<_> = case.iter().map(|(kind, _)| *kind).collect();
            let values: Vec<_> = case
                .iter()
                .map(|(_, value)| DVector::from_vec(value.clone()))
                .collect();
            for chunk in [1, 2, 6, 32] {
                let layout = AutoDiffLayout::new(kinds.clone(), 2, chunk).unwrap();
                let actual_trace = RefCell::new(Vec::new());
                let actual = layout
                    .evaluate(&values, &|inputs| {
                        actual_trace.borrow_mut().push(bits(inputs));
                        residual(inputs)
                    })
                    .unwrap();
                let expected_trace = RefCell::new(Vec::new());
                let reference = DynamicAutoDiff::new(layout.parameters, 2, |inputs| {
                    let mut offset = 0;
                    let blocks = kinds
                        .iter()
                        .zip(&values)
                        .map(|(&kind, value)| {
                            let n = kind.dimensions().1;
                            let block = kind.retract_autodiff(value, &inputs[offset..offset + n]);
                            offset += n;
                            block
                        })
                        .collect::<Result<Vec<_>, Error>>()?;
                    let views: Vec<_> = blocks.iter().map(Vec::as_slice).collect();
                    expected_trace.borrow_mut().push(bits(&views));
                    residual(&views)
                })
                .unwrap()
                .with_chunk_size(chunk)
                .unwrap()
                .evaluate_blocks_at_zero(&layout.dimensions)
                .unwrap();
                assert_eq!(actual.0.map(f64::to_bits), reference.0.map(f64::to_bits));
                for (a, b) in actual.1.iter().zip(&reference.1) {
                    assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
                }
                assert_eq!(actual_trace.into_inner(), expected_trace.into_inner());
            }
        }
    }

    #[test]
    fn inline_slots_keep_validation_and_later_callback_error_order() {
        let layout = AutoDiffLayout::new(vec![VariableKind::Euclidean(1); 2], 1, 1).unwrap();
        let calls = RefCell::new(0);
        let callback = |inputs: &[&[DynamicJet]]| {
            let mut count = calls.borrow_mut();
            *count += 1;
            if *count == 2 {
                return Err(Error::Evaluation("second chunk".into()));
            }
            Ok(vec![&inputs[0][0] + &inputs[1][0]])
        };
        // Validation stops before any callback, in original slot order.
        assert_eq!(
            layout.evaluate(
                &[DVector::zeros(2), DVector::from_element(1, f64::NAN)],
                &callback
            ),
            Err(Error::InvalidDimensions)
        );
        assert_eq!(*calls.borrow(), 0);
        assert_eq!(
            layout.evaluate(
                &[DVector::zeros(1), DVector::from_element(1, f64::NAN)],
                &callback
            ),
            Err(Error::NonFinite)
        );
        assert_eq!(*calls.borrow(), 0);
        assert_eq!(
            layout.evaluate(&[DVector::zeros(1), DVector::zeros(1)], &callback),
            Err(Error::Evaluation("second chunk".into()))
        );
        assert_eq!(*calls.borrow(), 2);
    }
}
