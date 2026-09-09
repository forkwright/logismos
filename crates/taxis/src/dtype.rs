//! Runtime dtype enumeration.

use crate::error::{GeometryOverflowSnafu, Result};

/// Runtime dtype tag.
///
/// `#[non_exhaustive]` so the public surface can grow without breaking
/// downstream matches. Phase 1 ships the dtypes the matmul and norm
/// kernels actually consume; `F8*` and `I4` are declared so the
/// loader + quant crates can reference them, but no Phase-1 kernel
/// dispatches on them.
#[expect(
    missing_docs,
    reason = "variant names are standard dtype tags (F32/F16/BF16/I8/...) -- documented in the enum doc comment above"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    F32,
    F16,
    BF16,
    F8E4M3,
    F8E5M2,
    I32,
    I8,
    /// Packed 4-bit signed, two elements per byte.
    I4,
    U8,
}

impl DType {
    /// Size of one element in bits. Useful for sub-byte dtypes.
    #[must_use]
    pub(crate) fn size_in_bits(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 32,
            Self::F16 | Self::BF16 => 16,
            Self::F8E4M3 | Self::F8E5M2 | Self::I8 | Self::U8 => 8,
            Self::I4 => 4,
        }
    }

    /// Size of one element in whole bytes; `None` for sub-byte dtypes.
    #[must_use]
    pub fn size_in_bytes_exact(self) -> Option<usize> {
        match self {
            Self::I4 => None,
            _ => Some(self.size_in_bits() / 8),
        }
    }

    /// Return the exact byte count for `elem_count` elements, rounded up.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::GeometryOverflow`] if the byte count cannot
    /// fit in `usize`.
    pub fn checked_byte_count(self, elem_count: usize) -> Result<usize> {
        match self.size_in_bytes_exact() {
            Some(bytes_per_element) => elem_count.checked_mul(bytes_per_element).ok_or_else(|| {
                GeometryOverflowSnafu {
                    operation: "dtype byte count",
                }
                .build()
            }),
            None => Ok(elem_count / 2 + elem_count % 2),
        }
    }

    /// True when this dtype is supported end-to-end by the Phase-1
    /// compute path (matmul, rms_norm, softmax, rope).
    #[must_use]
    pub fn is_phase1_compute(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }
}
