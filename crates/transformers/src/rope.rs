//! RoPE table (halves-rotation, HF-Qwen2 convention).
//!
//! A [`RopeTable`] is precomputed once per model and sliced per
//! sequence at forward-time. The shape convention is:
//!
//! - `cos`, `sin`: `[max_seq_len, head_dim / 2]` fp32.
//!
//! The `kernels::cpu_f32::rope_halves_in_place` routine takes a
//! per-row (already gathered) cos/sin slice, so `RopeTable::gather` in
//! this module materialises the per-token slices for a given position
//! vector before calling into the kernel. This matches the HF Qwen2
//! `apply_rotary_pos_emb` exactly.

use kernels::cpu_f32::build_rope_table_f32;

/// Precomputed rotary-position-embedding table.
#[derive(Debug, Clone)]
pub struct RopeTable {
    /// Cosine table, flat `[max_seq * head_dim / 2]`.
    pub cos: Vec<f32>,
    /// Sine table, flat `[max_seq * head_dim / 2]`.
    pub sin: Vec<f32>,
    /// Maximum sequence length covered.
    pub max_seq: usize,
    /// Per-head dimension (must be even).
    pub head_dim: usize,
}

impl RopeTable {
    /// Build a fresh table for the given sequence length, head dim and
    /// rope base theta.
    #[must_use]
    pub fn new(max_seq: usize, head_dim: usize, theta: f64) -> Self {
        let (cos, sin) = build_rope_table_f32(max_seq, head_dim, theta);
        Self {
            cos,
            sin,
            max_seq,
            head_dim,
        }
    }

    /// Gather per-row cos/sin for each position in `positions`. Returns
    /// `(cos_rows, sin_rows)` each of length `positions.len() * head_dim / 2`.
    #[must_use]
    pub(crate) fn gather(&self, positions: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let half = self.head_dim / 2;
        let mut cos_rows = Vec::with_capacity(positions.len() * half);
        let mut sin_rows = Vec::with_capacity(positions.len() * half);
        for &pos in positions {
            debug_assert!(pos < self.max_seq);
            let start = pos * half;
            let end = (pos + 1) * half;
            debug_assert!(end <= self.cos.len());
            debug_assert!(end <= self.sin.len());
            // SAFETY: `RopeTable::new` builds both tables as
            // `max_seq * head_dim / 2`, and attention forward only gathers
            // positions below `max_seq`. The debug assertions keep malformed
            // table construction or caller bugs visible during development
            // while keeping this per-token gather on the hot path branch-free.
            let cos_row = unsafe { self.cos.get_unchecked(start..end) };
            // SAFETY: same invariant as the cosine table above.
            let sin_row = unsafe { self.sin.get_unchecked(start..end) };
            cos_rows.extend_from_slice(cos_row);
            sin_rows.extend_from_slice(sin_row);
        }
        (cos_rows, sin_rows)
    }
}
