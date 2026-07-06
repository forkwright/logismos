//! `Shape` — inline-stored dimension list.

use smallvec::{SmallVec, smallvec};

/// Tensor shape. Rank-6 inline storage covers every tensor through
/// Phase 11 (decoder rank 4, DiT rank 5, U-Net rank 5; headroom 1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape(SmallVec<[usize; 6]>);

impl Shape {
    /// Construct from a slice.
    #[must_use]
    pub fn new(dims: &[usize]) -> Self {
        Self(SmallVec::from_slice(dims))
    }

    /// Construct the scalar (0-dim) shape.
    #[must_use]
    pub fn scalar() -> Self {
        Self(smallvec![])
    }

    /// Dims as a slice.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Number of axes (length of the dims slice).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Total number of elements.
    #[must_use]
    pub fn elem_count(&self) -> usize {
        if self.0.is_empty() {
            1
        } else {
            self.0.iter().product()
        }
    }
}

impl From<&[usize]> for Shape {
    fn from(v: &[usize]) -> Self {
        Self::new(v)
    }
}

impl From<Vec<usize>> for Shape {
    fn from(v: Vec<usize>) -> Self {
        Self(SmallVec::from_vec(v))
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(v: [usize; N]) -> Self {
        Self(SmallVec::from_slice(&v))
    }
}
