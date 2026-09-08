//! `Layout` — shape + stride (element counts) + offset.

use smallvec::SmallVec;

use crate::error::{GeometryOverflowSnafu, Result, ShapeMismatchSnafu};
use crate::shape::Shape;

/// Layout describing how tensor elements sit in storage.
///
/// Strides are in **elements**, not bytes, matching candle's choice
/// (`candle-core/src/layout.rs:6`). Bytes are recovered by
/// multiplying through `DType::size_in_bytes_exact`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    shape: Shape,
    stride: SmallVec<[usize; 6]>,
    start_offset: usize,
    elem_count: usize,
    required_storage_len: usize,
}

impl Layout {
    /// Canonical row-major contiguous layout for `shape`.
    #[must_use]
    pub(crate) fn try_contiguous(shape: Shape) -> Result<Self> {
        let dims = shape.dims();
        let mut stride: SmallVec<[usize; 6]> = SmallVec::with_capacity(dims.len());
        stride.resize(dims.len(), 0);
        let elem_count = shape.checked_elem_count()?;
        if elem_count == 0 {
            return Self::try_from_parts(shape, stride, 0);
        }
        let mut acc: usize = 1;
        for (i, &d) in dims.iter().enumerate().rev() {
            if let Some(slot) = stride.get_mut(i) {
                *slot = acc;
            }
            acc = acc.checked_mul(d).ok_or_else(|| {
                GeometryOverflowSnafu {
                    op: "Layout::try_contiguous",
                    msg: format!("stride product overflows at dimension {d}"),
                }
                .build()
            })?;
        }
        Self::try_from_parts(shape, stride, 0)
    }

    /// Construct an explicit layout after validating its rank and addressable
    /// geometry.
    ///
    /// # Errors
    ///
    /// [`crate::Error::ShapeMismatch`] when the stride and dimension ranks
    /// differ. [`crate::Error::GeometryOverflow`] when the shape or largest
    /// reachable storage offset cannot be represented by `usize`.
    pub fn try_from_parts(
        shape: Shape,
        stride: SmallVec<[usize; 6]>,
        start_offset: usize,
    ) -> Result<Self> {
        if shape.rank() != stride.len() {
            return ShapeMismatchSnafu {
                op: "Layout::try_from_parts",
                msg: format!(
                    "shape rank {} does not match stride rank {}",
                    shape.rank(),
                    stride.len()
                ),
            }
            .fail();
        }
        let elem_count = shape.checked_elem_count()?;
        let required_storage_len = if elem_count == 0 {
            0
        } else {
            let largest_offset = shape.dims().iter().zip(&stride).try_fold(
                start_offset,
                |offset, (&dim, &axis_stride)| {
                    let extent = dim.checked_sub(1).ok_or_else(|| {
                        GeometryOverflowSnafu {
                            op: "Layout::try_from_parts",
                            msg: "zero dimension reached nonempty geometry".to_string(),
                        }
                        .build()
                    })?;
                    let axis_offset = extent.checked_mul(axis_stride).ok_or_else(|| {
                        GeometryOverflowSnafu {
                            op: "Layout::try_from_parts",
                            msg: format!("axis extent {extent} × stride {axis_stride} overflows"),
                        }
                        .build()
                    })?;
                    offset.checked_add(axis_offset).ok_or_else(|| {
                        GeometryOverflowSnafu {
                            op: "Layout::try_from_parts",
                            msg: "largest reachable storage offset overflows".to_string(),
                        }
                        .build()
                    })
                },
            )?;
            largest_offset.checked_add(1).ok_or_else(|| {
                GeometryOverflowSnafu {
                    op: "Layout::try_from_parts",
                    msg: "required storage length overflows".to_string(),
                }
                .build()
            })?
        };
        Ok(Self {
            shape,
            stride,
            start_offset,
            elem_count,
            required_storage_len,
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

    /// Number of logical elements.
    #[must_use]
    pub fn elem_count(&self) -> usize {
        self.elem_count
    }

    /// Minimum storage elements needed to address every element in this view.
    #[must_use]
    pub fn required_storage_len(&self) -> usize {
        self.required_storage_len
    }

    /// Is the layout canonical row-major contiguous from offset 0?
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        if self.start_offset != 0 {
            return false;
        }
        if self.elem_count == 0 {
            return true;
        }
        let mut expected: usize = 1;
        for (&dim, &stride) in self.shape.dims().iter().rev().zip(self.stride.iter().rev()) {
            if dim == 1 {
                continue;
            }
            if stride != expected {
                return false;
            }
            let Some(next_expected) = expected.checked_mul(dim) else {
                return false;
            };
            expected = next_expected;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_3d_strides() -> Result<()> {
        let l = Layout::try_contiguous(Shape::new(&[2, 3, 4]))?;
        assert_eq!(l.stride(), &[12, 4, 1]);
        assert!(l.is_contiguous());
        assert_eq!(l.elem_count(), 24);
        assert_eq!(l.required_storage_len(), 24);
        Ok(())
    }

    #[test]
    fn contiguous_with_unit_dim() -> Result<()> {
        let l = Layout::try_contiguous(Shape::new(&[2, 1, 4]))?;
        assert!(l.is_contiguous());
        Ok(())
    }

    #[test]
    fn non_zero_start_offset_is_not_contiguous() -> Result<()> {
        // WHY(forkwright/logismos#58): the only prior tests build via
        // `Layout::contiguous`, where `start_offset` is always 0 — the
        // `start_offset != 0` branch in `is_contiguous` had zero
        // coverage. `from_parts` is the only way to set a non-zero
        // offset (e.g. a future view/slice API). This is the
        // negative-case fixture for that branch: it fails if the
        // `start_offset != 0 { return false }` guard is ever dropped.
        let base = Layout::try_contiguous(Shape::new(&[2, 3]))?;
        let sliced =
            Layout::try_from_parts(base.shape().clone(), SmallVec::from_slice(base.stride()), 3)?;
        assert!(!sliced.is_contiguous());
        Ok(())
    }

    #[test]
    fn rejects_rank_mismatch_and_offset_overflow() {
        assert!(matches!(
            Layout::try_from_parts(Shape::new(&[2, 3]), SmallVec::new(), 0),
            Err(crate::Error::ShapeMismatch { .. })
        ));
        assert!(matches!(
            Layout::try_from_parts(Shape::new(&[2]), SmallVec::from_slice(&[1]), usize::MAX),
            Err(crate::Error::GeometryOverflow { .. })
        ));
    }

    #[test]
    fn empty_layout_is_valid_without_overflowing_unused_strides() -> Result<()> {
        let layout = Layout::try_contiguous(Shape::new(&[0, usize::MAX, 2]))?;
        assert_eq!(layout.elem_count(), 0);
        assert_eq!(layout.required_storage_len(), 0);
        assert!(layout.is_contiguous());
        Ok(())
    }

    #[test]
    fn scalar_and_noncontiguous_layouts_remain_valid() -> Result<()> {
        let scalar = Layout::try_contiguous(Shape::scalar())?;
        assert_eq!(scalar.elem_count(), 1);
        assert!(scalar.is_contiguous());

        let transposed =
            Layout::try_from_parts(Shape::new(&[2, 3]), SmallVec::from_slice(&[1, 2]), 0)?;
        assert_eq!(transposed.required_storage_len(), 6);
        assert!(!transposed.is_contiguous());
        Ok(())
    }
}
