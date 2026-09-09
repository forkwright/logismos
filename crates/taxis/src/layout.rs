//! `Layout` — shape + stride (element counts) + offset.

use smallvec::SmallVec;

use crate::error::{GeometryOverflowSnafu, LayoutRankMismatchSnafu, Result};
use crate::shape::Shape;

/// Layout describing how tensor elements sit in storage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    shape: Shape,
    stride: SmallVec<[usize; 6]>,
    start_offset: usize,
}

impl Layout {
    /// Canonical row-major contiguous layout for `shape`.
    pub(crate) fn contiguous(shape: Shape) -> Result<Self> {
        shape.checked_elem_count()?;
        let dims = shape.dims();
        let mut stride: SmallVec<[usize; 6]> = SmallVec::with_capacity(dims.len());
        stride.resize(dims.len(), 0);
        let mut acc = 1usize;
        for (index, &dimension) in dims.iter().enumerate().rev() {
            stride[index] = acc;
            acc = acc.checked_mul(dimension).ok_or_else(|| {
                GeometryOverflowSnafu {
                    operation: "contiguous layout stride",
                }
                .build()
            })?;
        }
        Ok(Self {
            shape,
            stride,
            start_offset: 0,
        })
    }

    /// Construct a checked layout from explicit parts.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::LayoutRankMismatch`] for incoherent ranks or
    /// [`crate::Error::GeometryOverflow`] for an unrepresentable span.
    pub fn from_parts(
        shape: Shape,
        stride: SmallVec<[usize; 6]>,
        start_offset: usize,
    ) -> Result<Self> {
        if shape.rank() != stride.len() {
            return LayoutRankMismatchSnafu {
                dimensions: shape.rank(),
                strides: stride.len(),
            }
            .fail();
        }
        shape.checked_elem_count()?;
        let _span = shape.dims().iter().zip(&stride).try_fold(
            start_offset,
            |end, (&dimension, &step)| {
                if dimension == 0 {
                    return Ok(end);
                }
                dimension
                    .checked_sub(1)
                    .and_then(|width| width.checked_mul(step))
                    .and_then(|width| end.checked_add(width))
                    .ok_or_else(|| {
                        GeometryOverflowSnafu {
                            operation: "layout storage span",
                        }
                        .build()
                    })
            },
        )?;
        Ok(Self {
            shape,
            stride,
            start_offset,
        })
    }

    /// Per-axis extent (this layout's view over the storage).
    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Axis extents as a raw slice (shortcut for `shape().dims()`).
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Strides (elements).
    #[must_use]
    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    /// Offset of the first element, in elements.
    #[must_use]
    pub fn start_offset(&self) -> usize {
        self.start_offset
    }

    /// Return this layout's exact logical element count.
    pub fn checked_elem_count(&self) -> Result<usize> {
        self.shape.checked_elem_count()
    }

    /// Is the layout canonical row-major contiguous from offset 0?
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        if self.start_offset != 0 || self.shape.rank() != self.stride.len() {
            return false;
        }
        let mut expected = 1usize;
        for (&dimension, &stride) in self.shape.dims().iter().rev().zip(self.stride.iter().rev()) {
            if dimension != 1 && stride != expected {
                return false;
            }
            let Some(next) = expected.checked_mul(dimension) else {
                return false;
            };
            expected = next;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_3d_strides() -> Result<()> {
        let layout = Layout::contiguous(Shape::new(&[2, 3, 4]))?;
        assert_eq!(layout.stride(), &[12, 4, 1]);
        assert!(layout.is_contiguous());
        assert_eq!(layout.checked_elem_count()?, 24);
        Ok(())
    }

    #[test]
    fn valid_strided_layout_is_not_contiguous() -> Result<()> {
        let layout = Layout::from_parts(Shape::new(&[2, 2]), SmallVec::from_slice(&[3, 1]), 0)?;
        assert!(!layout.is_contiguous());
        Ok(())
    }

    #[test]
    fn offset_layout_is_not_contiguous() -> Result<()> {
        let layout = Layout::from_parts(Shape::new(&[2, 3]), SmallVec::from_slice(&[3, 1]), 3)?;
        assert!(!layout.is_contiguous());
        Ok(())
    }

    #[test]
    fn explicit_layout_rejects_incoherent_rank() {
        let err = Layout::from_parts(Shape::new(&[2, 3]), SmallVec::from_slice(&[1]), 0);
        assert!(matches!(err, Err(crate::Error::LayoutRankMismatch { .. })));
    }

    #[test]
    fn explicit_layout_rejects_overflowing_span() {
        let err = Layout::from_parts(Shape::new(&[2]), SmallVec::from_slice(&[usize::MAX]), 1);
        assert!(matches!(err, Err(crate::Error::GeometryOverflow { .. })));
    }
}
