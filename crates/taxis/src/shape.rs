//! `Shape` — inline-stored dimension list.

use smallvec::{SmallVec, smallvec};

use crate::error::{GeometryOverflowSnafu, Result};

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

    /// Total number of elements when the shape is usable allocation geometry.
    ///
    /// # Errors
    ///
    /// [`crate::Error::GeometryOverflow`] when the dimension product cannot
    /// be represented by `usize`.
    pub fn checked_elem_count(&self) -> Result<usize> {
        self.0.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim).ok_or_else(|| {
                GeometryOverflowSnafu {
                    op: "Shape::checked_elem_count",
                    msg: format!("dimension product overflows at dimension {dim}"),
                }
                .build()
            })
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_elem_count_empty_shape_is_one() -> Result<()> {
        assert_eq!(Shape::scalar().checked_elem_count()?, 1);
        Ok(())
    }

    #[test]
    fn checked_elem_count_normal_product() -> Result<()> {
        assert_eq!(Shape::new(&[2, 3, 4]).checked_elem_count()?, 24);
        Ok(())
    }

    #[test]
    fn checked_elem_count_rejects_overflow() {
        // WHY(forkwright/logismos#58): a raw `.iter().product()` wraps
        // modulo 2^64 in a release build. These two dims are chosen so
        // the wrapped product is a deceptively small, plausible-looking
        // value (2) rather than an obviously-broken one — exactly the
        // silent shape/length confusion a caller has no way to detect.
        // Confirmed below via `wrapping_mul` so the fixture documents
        // its own target rather than asserting a magic number.
        let dims = [usize::MAX / 2 + 2, 2];
        assert_eq!(
            dims[0].wrapping_mul(dims[1]),
            2,
            "sanity: this dim pair must wrap to a small value, not saturate cleanly"
        );
        let shape = Shape::new(&dims);
        assert!(matches!(
            shape.checked_elem_count(),
            Err(crate::Error::GeometryOverflow { .. })
        ));
    }
}
