//! # cache
//!
//! KV cache layouts for decoder inference.
//!
//! Phase 2 ships a single implementation: [`FlatKvCache`] — a
//! layer-indexed ring of CPU-backed `taxis::Tensor` slots with an
//! append-style `put` + a `get` that slices the whole cached range
//! for a given layer. No eviction, no sharing, no prefix reuse.
//!
//! The paged allocator (vLLM-style) + radix cache (SGLang-style) land
//! in Phases 6 and 12 respectively; both are behind the same public
//! [`KvCache`] trait so callers can swap layouts without touching the
//! forward-pass code.
//!
//! ## Shape model
//!
//! [`CacheLayout`] carries the invariants a decoder needs:
//! `{ num_layers, num_kv_heads, head_dim, max_seq_len, dtype }`.
//! Phase 2 stores K and V separately per layer, as CPU tensors with
//! shape `[max_seq_len, num_kv_heads * head_dim]`. Per-layer
//! "written-length" state (`lens[layer]`) tracks how many rows have
//! been appended. A subsequent `get_kv(layer, 0..len)` returns two
//! sliced views.
//!
//! Phase 3 (Stella) runs on CPU at first. Phase 4 moves onto HIP.
//! Switching the storage kind is a one-liner on `alloc_slot` once
//! `taxis::Tensor` grows a device-side allocator backend.

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
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

pub mod error;
pub mod flat;

use taxis::Tensor;

pub use crate::error::{Error, Result};
pub use crate::flat::{CacheLayout, FlatKvCache};

/// Abstract KV-cache contract.
///
/// Kept as a trait so Phase 6 / 12 layouts can be substituted without
/// touching the forward-pass code. Every concrete impl lives behind
/// the same two primitive operations (put + get); the paged + radix
/// variants add sharing / eviction as implementation details.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_layout_row_elems_multiplies_heads_by_width() {
        let layout = CacheLayout {
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 8,
            max_seq_len: 16,
            dtype: taxis::DType::F16,
        };
        assert_eq!(layout.row_elems(), 32);
    }
}
