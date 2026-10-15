//! Common robotics tangent widths avoid a heap allocation per Jet operation.
use std::{fmt, ops::Deref};

const INLINE_CAPACITY: usize = 6;

#[derive(Clone)]
pub(super) enum Derivatives {
    Inline {
        values: [f64; INLINE_CAPACITY],
        len: usize,
    },
    Heap(Vec<f64>),
}

impl Derivatives {
    pub(super) fn from_fn(len: usize, mut value: impl FnMut(usize) -> f64) -> Self {
        if len <= INLINE_CAPACITY {
            let mut values = [0.; INLINE_CAPACITY];
            for (i, slot) in values[..len].iter_mut().enumerate() {
                *slot = value(i);
            }
            Self::Inline { values, len }
        } else {
            Self::Heap((0..len).map(value).collect())
        }
    }
    pub(super) fn empty() -> Self {
        Self::Inline {
            values: [0.; INLINE_CAPACITY],
            len: 0,
        }
    }
}

impl Deref for Derivatives {
    type Target = [f64];
    fn deref(&self) -> &[f64] {
        match self {
            Self::Inline { values, len } => &values[..*len],
            Self::Heap(values) => values,
        }
    }
}

impl PartialEq for Derivatives {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

// Keep DynamicJet's existing Debug representation independent of storage.
impl fmt::Debug for Derivatives {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.deref().fmt(f)
    }
}
