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
    ///
    /// The product is checked, not raw: a plain `.iter().product()`
    /// panics on overflow in a debug build and wraps to an arbitrary
    /// (possibly small, plausible-looking) value in release, where a
    /// wrapped count can silently equal an unrelated storage length.
    /// This saturates to [`usize::MAX`] instead, which no real
    /// storage length can ever match, so downstream `elem_count`-based
    /// validation (e.g. [`crate::Tensor::try_from_cpu`]) reliably
    /// rejects an overflowing shape instead of silently accepting one.
    #[must_use]
    pub fn elem_count(&self) -> usize {
        self.0
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .unwrap_or(usize::MAX)
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
    fn elem_count_empty_shape_is_one() {
        assert_eq!(Shape::scalar().elem_count(), 1);
    }

    #[test]
    fn elem_count_normal_product() {
        assert_eq!(Shape::new(&[2, 3, 4]).elem_count(), 24);
    }

    #[test]
    fn elem_count_overflow_saturates_instead_of_wrapping() {
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
        assert_eq!(
            shape.elem_count(),
            usize::MAX,
            "overflow must saturate to usize::MAX, not wrap to {}",
            dims[0].wrapping_mul(dims[1])
        );
    }
}
