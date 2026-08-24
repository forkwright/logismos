//! Runtime dtype enumeration.

use hipcore::BytePod;

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

    /// Total byte count for `elem_count` elements, rounded up.
    #[must_use]
    pub fn byte_count(self, elem_count: usize) -> usize {
        (self.size_in_bits() * elem_count).div_ceil(8)
    }

    /// True when this dtype is supported end-to-end by the Phase-1
    /// compute path (matmul, rms_norm, softmax, rope).
    #[must_use]
    pub fn is_phase1_compute(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }
}

/// Binds a host element type to the runtime [`DType`] tag it represents.
///
/// [`HipStorage::from_host`](crate::storage::HipStorage::from_host)
/// derives its stored `dtype` from `T::DTYPE` instead of accepting a
/// free [`DType`] parameter, so a caller cannot construct storage whose
/// declared dtype disagrees with the byte layout of the data actually
/// copied — the mismatch is unrepresentable rather than checked.
///
/// # Safety
///
/// Implementors must guarantee `DTYPE.size_in_bytes_exact()` is
/// `Some(size_of::<Self>())`. Violating this lets `from_host` construct
/// a [`crate::storage::HipStorage`] whose `dtype` / `elem_count` /
/// byte length are mutually inconsistent — the exact corruption this
/// trait exists to foreclose.
pub unsafe trait DTyped: BytePod {
    /// Runtime dtype tag this Rust type represents.
    const DTYPE: DType;
}

// SAFETY: `DType::F32.size_in_bytes_exact() == Some(4) == size_of::<f32>()`.
unsafe impl DTyped for f32 {
    const DTYPE: DType = DType::F32;
}
// SAFETY: `DType::F16.size_in_bytes_exact() == Some(2) == size_of::<half::f16>()`.
unsafe impl DTyped for half::f16 {
    const DTYPE: DType = DType::F16;
}
// SAFETY: `DType::BF16.size_in_bytes_exact() == Some(2) == size_of::<half::bf16>()`.
unsafe impl DTyped for half::bf16 {
    const DTYPE: DType = DType::BF16;
}
// SAFETY: `DType::I32.size_in_bytes_exact() == Some(4) == size_of::<i32>()`.
unsafe impl DTyped for i32 {
    const DTYPE: DType = DType::I32;
}
// SAFETY: `DType::I8.size_in_bytes_exact() == Some(1) == size_of::<i8>()`.
unsafe impl DTyped for i8 {
    const DTYPE: DType = DType::I8;
}
// SAFETY: `DType::U8.size_in_bytes_exact() == Some(1) == size_of::<u8>()`.
unsafe impl DTyped for u8 {
    const DTYPE: DType = DType::U8;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_dtyped_size_matches<T: DTyped>() {
        assert_eq!(
            T::DTYPE.size_in_bytes_exact(),
            Some(core::mem::size_of::<T>()),
            "DTyped mapping for {:?} disagrees with size_of::<T>()",
            T::DTYPE
        );
    }

    /// Pins the exact invariant `HipStorage::from_host` relies on as
    /// its ONLY source of `dtype`: if any `DTyped` impl declared a tag
    /// whose byte size does not match its Rust type, `from_host` would
    /// silently reproduce a dtype/byte-layout mismatch. There is no
    /// second, independent `dtype` input left to cross-check against —
    /// this mapping table IS the contract.
    #[test]
    fn dtyped_mapping_matches_declared_byte_size() {
        assert_dtyped_size_matches::<f32>();
        assert_dtyped_size_matches::<half::f16>();
        assert_dtyped_size_matches::<half::bf16>();
        assert_dtyped_size_matches::<i32>();
        assert_dtyped_size_matches::<i8>();
        assert_dtyped_size_matches::<u8>();
    }

    #[test]
    fn dtyped_mapping_assigns_the_expected_tag() {
        assert_eq!(<f32 as DTyped>::DTYPE, DType::F32);
        assert_eq!(<half::f16 as DTyped>::DTYPE, DType::F16);
        assert_eq!(<half::bf16 as DTyped>::DTYPE, DType::BF16);
        assert_eq!(<i32 as DTyped>::DTYPE, DType::I32);
        assert_eq!(<i8 as DTyped>::DTYPE, DType::I8);
        assert_eq!(<u8 as DTyped>::DTYPE, DType::U8);
    }
}
