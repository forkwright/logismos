//! # cache
//!
//! KV cache layouts for decoder inference.
//!
//! The legacy `flat` default feature ships `FlatKvCache` — a
//! layer-indexed ring of CPU-backed `taxis::Tensor` slots with an
//! append-style `put` + a `get` that slices the whole cached range
//! for a given layer. No eviction, no sharing, no prefix reuse.
//!
//! [`PagedKvPool`] is a separate private CPU transaction owner for native
//! decoder execution. It does not implement the legacy tensor trait because
//! its borrowed append lifecycle is the correctness boundary.
//!
//! ## Shape model
//!
//! The flat feature's `CacheLayout` carries its tensor geometry:
//! `{ num_layers, num_kv_heads, head_dim, max_seq_len, dtype }`.
//! It stores K and V separately per layer, as CPU tensors with
//! shape `[max_seq_len, num_kv_heads * head_dim]`. Per-layer
//! "written-length" state (`lens[layer]`) tracks how many rows have
//! been appended. A subsequent `get_kv(layer, 0..len)` returns two
//! sliced views.
//! Paged geometry and transaction ownership are separate from this legacy
//! tensor layout and do not initialize a device runtime.
//!
//! Legacy flat geometry is constructed only through [`CacheLayout::new`], and
//! [`FlatKvCache::new`] remains fallible because its validated K/V backing
//! still requires allocator reservations. This is the intentional migration
//! from public field literals and infallible flat-cache allocation.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::cast_sign_loss
)]

pub mod error;
#[cfg(feature = "flat")]
pub mod flat;
mod paged;

#[cfg(feature = "flat")]
use taxis::Tensor;

pub use crate::error::{Error, Result};
#[cfg(feature = "flat")]
pub use crate::flat::{CacheLayout, FlatKvCache};
#[cfg(feature = "gpu")]
pub use crate::paged::{
    NativePagedAppend, NativePagedKvBuffers, NativePagedKvPlan, NativePagedKvPool,
    NativePagedKvPoolBindingError, NativePagedLayerKv,
};
pub use crate::paged::{
    PagedAppend, PagedKvGeometry, PagedKvPlan, PagedKvPool, PagedLayerKv, PagedPreparedCommit,
};

/// Legacy unshared tensor KV-cache contract.
///
/// Paged append transactions use their own explicit lifecycle, not this
/// tensor-oriented put/get/reset interface. Sharing and eviction are not
/// hidden implementation details of this trait.
#[cfg(feature = "flat")]
pub trait KvCache {
    /// Append `k` and `v` tensors to the given `layer_idx` slot.
    ///
    /// `k` and `v` must have identical leading shape `[n_tokens, ..]`
    /// and match the cache's declared dtype. The cache will refuse
    /// writes that would exceed `max_seq_len`.
    fn put(&mut self, layer_idx: usize, k: &Tensor, v: &Tensor) -> Result<()>;

    /// Read back the first `len` tokens for `layer_idx` as a
    /// `(k, v)` pair.
    fn get(&self, layer_idx: usize, len: usize) -> Result<(Tensor, Tensor)>;

    /// Current written length for the given layer, or `None` if
    /// `layer_idx` is out of range for this cache. Kept distinct from
    /// `Some(0)`, which means the layer is in range but nothing has been
    /// written to it yet.
    fn len_of(&self, layer_idx: usize) -> Option<usize>;

    /// Number of layers this cache was sized for.
    fn num_layers(&self) -> usize;

    /// Reset every layer's written-length back to zero. The underlying
    /// storage is not reallocated — subsequent `put`s overwrite in
    /// place.
    fn reset(&mut self);
}

#[cfg(all(test, feature = "flat"))]
mod tests {
    use super::*;

    #[test]
    fn cache_layout_row_elems_multiplies_heads_by_width() -> Result<()> {
        let layout = CacheLayout::new(2, 4, 8, 16, taxis::DType::F16)?;
        assert_eq!(layout.row_elems(), 32);
        Ok(())
    }

    #[test]
    fn cache_layout_rejects_zero_configured_dimension() {
        for result in [
            CacheLayout::new(0, 4, 8, 16, taxis::DType::F16),
            CacheLayout::new(2, 0, 8, 16, taxis::DType::F16),
            CacheLayout::new(2, 4, 0, 16, taxis::DType::F16),
            CacheLayout::new(2, 4, 8, 0, taxis::DType::F16),
        ] {
            assert!(matches!(result, Err(Error::FlatZeroDimension { .. })));
        }
    }

    #[test]
    fn cache_layout_rejects_overflowing_row() {
        let result = CacheLayout::new(1, usize::MAX, 2, 1, taxis::DType::F16);
        assert!(matches!(result, Err(Error::FlatArithmetic { .. })));
    }

    #[test]
    fn cache_layout_rejects_overflowing_dtype_bytes() {
        let result = CacheLayout::new(1, usize::MAX, 1, 1, taxis::DType::F32);
        assert!(matches!(result, Err(Error::Taxis { .. })));
    }

    #[test]
    fn cache_layout_rejects_overflowing_per_layer_context() {
        let result = CacheLayout::new(1, 1, 1, usize::MAX, taxis::DType::F32);
        assert!(matches!(result, Err(Error::FlatArithmetic { .. })));
    }

    #[test]
    fn cache_layout_rejects_overflowing_all_layer_backing() {
        let result = CacheLayout::new(usize::MAX, 1, 1, 1, taxis::DType::F32);
        assert!(matches!(result, Err(Error::FlatArithmetic { .. })));
    }
}
