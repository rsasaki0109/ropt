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
