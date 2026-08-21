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

use crate::error::{Result, ShapeSnafu};

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
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] if any position in `positions` is `>= self.max_seq`,
    /// or if the table itself is shorter than its declared `max_seq * head_dim
    /// / 2` (malformed construction).
    pub(crate) fn gather(&self, positions: &[usize]) -> Result<(Vec<f32>, Vec<f32>)> {
        let half = self.head_dim / 2;
        let mut cos_rows = Vec::with_capacity(positions.len() * half);
        let mut sin_rows = Vec::with_capacity(positions.len() * half);
        for &pos in positions {
            if pos >= self.max_seq {
                return ShapeSnafu {
                    message: format!(
                        "RopeTable::gather: position {pos} >= max_seq {}",
                        self.max_seq
                    ),
                }
                .fail();
            }
            let start = pos * half;
            let end = start + half;
            let cos_row = self.cos.get(start..end).ok_or_else(|| {
                ShapeSnafu {
                    message: format!(
                        "RopeTable::gather: cos range {start}..{end} exceeds table length {}",
                        self.cos.len()
                    ),
                }
                .build()
            })?;
            let sin_row = self.sin.get(start..end).ok_or_else(|| {
                ShapeSnafu {
                    message: format!(
                        "RopeTable::gather: sin range {start}..{end} exceeds table length {}",
                        self.sin.len()
                    ),
                }
                .build()
            })?;
            cos_rows.extend_from_slice(cos_row);
            sin_rows.extend_from_slice(sin_row);
        }
        Ok((cos_rows, sin_rows))
    }
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "test assertions use expect()/expect_err() directly"
    )]

    use super::*;

    #[test]
    fn gather_in_range_matches_table_rows() {
        let table = RopeTable::new(4, 4, 10_000.0);
        let (cos_rows, sin_rows) = table.gather(&[0, 1, 2]).expect("in-range gather");
        assert_eq!(cos_rows.len(), 3 * 2);
        assert_eq!(sin_rows.len(), 3 * 2);
        assert_eq!(cos_rows[0..2], table.cos[0..2]);
        assert_eq!(sin_rows[2..4], table.sin[2..4]);
    }

    #[test]
    fn gather_out_of_range_position_is_shape_error_not_ub() {
        // WHY: this is the regression test for forkwright/logismos#28 —
        // `gather` used to guard `pos < max_seq` with `debug_assert!` only
        // and then read via `get_unchecked`, which is undefined behaviour
        // in a release build on an out-of-range position. It must be a
        // returned `Err` in every profile.
        let table = RopeTable::new(4, 4, 10_000.0);
        let err = table
            .gather(&[0, 4])
            .expect_err("position 4 >= max_seq 4 must error");
        assert!(err.to_string().contains("position 4 >= max_seq 4"));
    }
}
