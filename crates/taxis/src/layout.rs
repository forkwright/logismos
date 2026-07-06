//! `Layout` — shape + stride (element counts) + offset.

use smallvec::SmallVec;

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
}

impl Layout {
    /// Canonical row-major contiguous layout for `shape`.
    #[must_use]
    pub(crate) fn contiguous(shape: Shape) -> Self {
        let dims = shape.dims();
        let mut stride: SmallVec<[usize; 6]> = SmallVec::with_capacity(dims.len());
        stride.resize(dims.len(), 0);
        let mut acc: usize = 1;
        for (i, &d) in dims.iter().enumerate().rev() {
            if let Some(slot) = stride.get_mut(i) {
                *slot = acc;
            }
            acc = acc.saturating_mul(d);
        }
        Self {
            shape,
            stride,
            start_offset: 0,
        }
    }

    /// Construct from explicit parts.
    #[must_use]
    pub fn from_parts(shape: Shape, stride: SmallVec<[usize; 6]>, start_offset: usize) -> Self {
        Self {
            shape,
            stride,
            start_offset,
        }
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
        self.shape.elem_count()
    }

    /// Is the layout canonical row-major contiguous from offset 0?
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        if self.start_offset != 0 {
            return false;
        }
        let mut expected: usize = 1;
        for (&dim, &stride) in self.shape.dims().iter().rev().zip(self.stride.iter().rev()) {
            if dim == 1 {
                continue;
            }
            if stride != expected {
                return false;
            }
            expected = expected.saturating_mul(dim);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_3d_strides() {
        let l = Layout::contiguous(Shape::new(&[2, 3, 4]));
        assert_eq!(l.stride(), &[12, 4, 1]);
        assert!(l.is_contiguous());
        assert_eq!(l.elem_count(), 24);
    }

    #[test]
    fn contiguous_with_unit_dim() {
        let l = Layout::contiguous(Shape::new(&[2, 1, 4]));
        assert!(l.is_contiguous());
    }
}
