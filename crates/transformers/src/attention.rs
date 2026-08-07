//! Qwen2-style multi-head attention with Grouped-Query Attention.
//!
//! Matches the HF `Qwen2Attention` forward exactly at fp32 precision:
//!
//! 1. Q/K/V projections with biases.
//! 2. Reshape Q to `(S, H, D)`, K/V to `(S, H_kv, D)`.
//! 3. RoPE (halves rotation) applied to Q and K.
//! 4. GQA expansion: each KV head repeated `H / H_kv` times.
//! 5. Scaled dot-product attention: `softmax(Q Kᵀ / sqrt(D)) @ V`.
//! 6. Additive padding mask (encoder, non-causal; caller decides).
//! 7. Merge heads, output projection (no bias).
//!
//! Phase-3 scope: single-batch (`B = 1`), fp32, CPU only. A batch loop
//! around this block handles `B > 1`.

use kernels::cpu_f32;
use num_traits::ToPrimitive;

use crate::error::{Error, Result};
use crate::rope::RopeTable;

/// Attention config shared between Qwen2 and Stella.
#[derive(Debug, Clone, Copy)]
pub struct QwenAttentionConfig {
    /// Model hidden size.
    pub hidden: usize,
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of key/value heads (<= `n_heads`; GQA groups = `n_heads / n_kv_heads`).
    pub n_kv_heads: usize,
    /// Per-head dimension (`hidden / n_heads`).
    pub head_dim: usize,
}

impl QwenAttentionConfig {
    /// Total K/V width (`n_kv_heads * head_dim`).
    #[must_use]
    pub(crate) fn kv_width(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

/// Per-layer attention weights, all fp32, all untransposed (HF layout
/// `[out, in]`). Biases are the HF convention: present on Q, K, V only.
#[derive(Debug, Clone)]
pub struct QwenAttentionWeights {
    /// `q_proj.weight` — `[hidden, hidden]` flat.
    pub wq: Vec<f32>,
    /// `q_proj.bias` — `[hidden]`.
    pub bq: Vec<f32>,
    /// `k_proj.weight` — `[kv_width, hidden]`.
    pub wk: Vec<f32>,
    /// `k_proj.bias` — `[kv_width]`.
    pub bk: Vec<f32>,
    /// `v_proj.weight` — `[kv_width, hidden]`.
    pub wv: Vec<f32>,
    /// `v_proj.bias` — `[kv_width]`.
    pub bv: Vec<f32>,
    /// `o_proj.weight` — `[hidden, hidden]`. No bias.
    pub wo: Vec<f32>,
}

/// Owning attention block.
#[derive(Debug, Clone)]
pub struct QwenAttention {
    /// Configuration.
    pub cfg: QwenAttentionConfig,
    /// Weights.
    pub weights: QwenAttentionWeights,
}

impl QwenAttention {
    /// Build a new attention block.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] if any weight shape contradicts the config.
    pub fn new(cfg: QwenAttentionConfig, weights: QwenAttentionWeights) -> Result<Self> {
        let h = cfg.hidden;
        let kvw = cfg.kv_width();
        check_shape("wq", weights.wq.len(), h * h)?;
        check_shape("bq", weights.bq.len(), h)?;
        check_shape("wk", weights.wk.len(), kvw * h)?;
        check_shape("bk", weights.bk.len(), kvw)?;
        check_shape("wv", weights.wv.len(), kvw * h)?;
        check_shape("bv", weights.bv.len(), kvw)?;
        check_shape("wo", weights.wo.len(), h * h)?;
        if !cfg.n_heads.is_multiple_of(cfg.n_kv_heads) {
            return Err(Error::Shape(format!(
                "attention: n_heads {} not a multiple of n_kv_heads {}",
                cfg.n_heads, cfg.n_kv_heads
            )));
        }
        if cfg.head_dim * cfg.n_heads != cfg.hidden {
            return Err(Error::Shape(format!(
                "attention: head_dim({}) * n_heads({}) != hidden({})",
                cfg.head_dim, cfg.n_heads, cfg.hidden
            )));
        }
        Ok(Self { cfg, weights })
    }

    /// Forward pass, encoder-style (no KV cache).
    ///
    /// Arguments:
    /// - `x`: `[seq, hidden]` input, fp32.
    /// - `mask`: `[seq]` attention mask (u8; 1 = real, 0 = pad).
    /// - `rope`: precomputed rope table, sized at least `seq`.
    ///
    /// Returns `[seq, hidden]` fp32.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] on input-shape disagreement with the config, or if
    /// any internal head-slicing range falls outside an allocated buffer.
    pub fn forward(&self, x: &[f32], mask: &[u8], rope: &RopeTable) -> Result<Vec<f32>> {
        let cfg = self.cfg;
        let hidden = cfg.hidden;
        let n_h = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let d = cfg.head_dim;
        let kv_width = cfg.kv_width();
        let groups = n_h / n_kv;

        if !x.len().is_multiple_of(hidden) {
            return Err(Error::Shape(format!(
                "attention.forward: x.len()={} not multiple of hidden={}",
                x.len(),
                hidden
            )));
        }
        let seq = x.len() / hidden;
        if mask.len() != seq {
            return Err(Error::Shape(format!(
                "attention.forward: mask.len()={} != seq={}",
                mask.len(),
                seq
            )));
        }
        if seq > rope.max_seq {
            return Err(Error::Shape(format!(
                "attention.forward: seq {} > rope.max_seq {}",
                seq, rope.max_seq
            )));
        }
        if rope.head_dim != d {
            return Err(Error::Shape(format!(
                "attention.forward: rope.head_dim {} != cfg.head_dim {}",
                rope.head_dim, d
            )));
        }

        // ---- Q/K/V projections -------------------------------------------
        // x: [seq, hidden]; wq: [hidden, hidden] (HF layout, rows = out-dim)
        // y = x @ wq^T + bq -> [seq, hidden]
        let q = cpu_f32::linear_t(
            x,
            &self.weights.wq,
            Some(&self.weights.bq),
            seq,
            hidden,
            hidden,
        );
        let k = cpu_f32::linear_t(
            x,
            &self.weights.wk,
            Some(&self.weights.bk),
            seq,
            kv_width,
            hidden,
        );
        let v = cpu_f32::linear_t(
            x,
            &self.weights.wv,
            Some(&self.weights.bv),
            seq,
            kv_width,
            hidden,
        );

        // ---- Reshape to heads and apply RoPE -----------------------------
        // q: [seq, n_h, d]; we need RoPE on each (seq, d) slice. We flatten
        // to rows = seq * n_h, each row of length d, with the positional
        // index being `row / n_h`. Materialise per-row cos/sin via gather.
        let positions_q: Vec<usize> = (0..seq).flat_map(|s| std::iter::repeat_n(s, n_h)).collect();
        let positions_k: Vec<usize> = (0..seq)
            .flat_map(|s| std::iter::repeat_n(s, n_kv))
            .collect();
        let (cos_q, sin_q) = rope.gather(&positions_q)?;
        let (cos_k, sin_k) = rope.gather(&positions_k)?;

        let mut q_rope = q;
        cpu_f32::rope_halves_in_place(&mut q_rope, &cos_q, &sin_q, seq * n_h, d);
        let mut k_rope = k;
        cpu_f32::rope_halves_in_place(&mut k_rope, &cos_k, &sin_k, seq * n_kv, d);

        // ---- Transpose to [heads, seq, head_dim] for the attention matmul.
        // Current q: [seq, n_h, d] (stride n_h*d, d, 1). We want [n_h, seq, d].
        let q_hsd = transpose_seq_heads(&q_rope, seq, n_h, d)?;
        let k_hsd = transpose_seq_heads(&k_rope, seq, n_kv, d)?;
        let v_hsd = transpose_seq_heads(&v, seq, n_kv, d)?;

        // ---- GQA expansion: repeat each kv head `groups` times.
        let k_expanded = repeat_kv(&k_hsd, n_kv, groups, seq, d)?;
        let v_expanded = repeat_kv(&v_hsd, n_kv, groups, seq, d)?;

        // ---- Scaled dot-product: scores = q @ k^T / sqrt(d) -> [n_h, seq, seq]
        let scale = d.to_f32().unwrap_or(f32::INFINITY).sqrt().recip();
        let mut scores = vec![0.0f32; n_h * seq * seq];
        for h in 0..n_h {
            let start = h * seq * d;
            let end = (h + 1) * seq * d;
            let q_h = checked_slice(&q_hsd, start, end, "attention query head")?;
            let k_h = checked_slice(&k_expanded, start, end, "attention key head")?;
            // scores_h = q_h @ k_h^T -> [seq, seq]
            let sc = cpu_f32::linear_t(q_h, k_h, None, seq, seq, d);
            let score_start = h * seq * seq;
            let score_end = (h + 1) * seq * seq;
            let dst =
                checked_slice_mut(&mut scores, score_start, score_end, "attention score head")?;
            for (d_slot, v) in dst.iter_mut().zip(sc.iter()) {
                *d_slot = v * scale;
            }
        }

        // ---- Additive mask: -inf where mask[j] == 0, for every row i.
        // Mask shape per head: [seq, seq] where column j is masked.
        // Build a [seq, seq] broadcast mask once, then reuse across heads.
        let mut col_mask = vec![1u8; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                let slot = col_mask
                    .get_mut(i * seq + j)
                    .ok_or_else(|| Error::Shape("attention column mask slot".to_string()))?;
                *slot = mask
                    .get(j)
                    .copied()
                    .ok_or_else(|| Error::Shape("attention mask column".to_string()))?;
            }
        }
        // Apply per head; kernels::mask_additive_in_place expects scores
        // `[rows, n]` and mask `[rows_mask, n]` broadcasting rows_mask -> rows.
        for h in 0..n_h {
            let start = h * seq * seq;
            let end = (h + 1) * seq * seq;
            let head = checked_slice_mut(&mut scores, start, end, "attention masked score head")?;
            cpu_f32::mask_additive_in_place(head, &col_mask, seq, seq);
        }

        // ---- Softmax per head, per row.
        for h in 0..n_h {
            let start = h * seq * seq;
            let end = (h + 1) * seq * seq;
            let head = checked_slice_mut(&mut scores, start, end, "attention scores head")?;
            let sm = cpu_f32::softmax_last_dim(head, seq, seq);
            head.copy_from_slice(&sm);
        }

        // ---- Attention output: out = scores @ v -> [n_h, seq, d]
        let mut attn = vec![0.0f32; n_h * seq * d];
        for h in 0..n_h {
            let score_start = h * seq * seq;
            let score_end = (h + 1) * seq * seq;
            let value_start = h * seq * d;
            let value_end = (h + 1) * seq * d;
            let sh = checked_slice(&scores, score_start, score_end, "attention score rows")?;
            let vh = checked_slice(&v_expanded, value_start, value_end, "attention value rows")?;
            // attn_h = sh @ vh -> [seq, d]
            let ah = cpu_f32::linear(sh, vh, None, seq, d, seq);
            let dst =
                checked_slice_mut(&mut attn, value_start, value_end, "attention output rows")?;
            dst.copy_from_slice(&ah);
        }

        // ---- Merge heads: [n_h, seq, d] -> [seq, n_h*d] = [seq, hidden]
        let merged = merge_heads(&attn, n_h, seq, d)?;

        // ---- Output projection. wo: [hidden, hidden] no bias.
        let out = cpu_f32::linear_t(&merged, &self.weights.wo, None, seq, hidden, hidden);
        Ok(out)
    }
}

fn check_shape(name: &'static str, got: usize, expected: usize) -> Result<()> {
    if got == expected {
        Ok(())
    } else {
        Err(Error::Shape(format!(
            "{name}: expected {expected} elements, got {got}"
        )))
    }
}

/// `[seq, heads, head_dim] -> [heads, seq, head_dim]` (materialised).
///
/// # Errors
///
/// [`Error::Shape`] if `x` is shorter than `seq * heads * d`.
fn transpose_seq_heads(x: &[f32], seq: usize, heads: usize, d: usize) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; seq * heads * d];
    for s in 0..seq {
        for h in 0..heads {
            let src_start = (s * heads + h) * d;
            let src_end = (s * heads + h + 1) * d;
            let dst_start = (h * seq + s) * d;
            let dst_end = (h * seq + s + 1) * d;
            let src = checked_slice(x, src_start, src_end, "transpose source row")?;
            let dst = checked_slice_mut(&mut out, dst_start, dst_end, "transpose destination row")?;
            dst.copy_from_slice(src);
        }
    }
    Ok(out)
}

/// `[heads, seq, head_dim] -> [seq, heads, head_dim]` (materialised).
///
/// # Errors
///
/// [`Error::Shape`] if `attn` is shorter than `heads * seq * d`.
fn merge_heads(attn: &[f32], heads: usize, seq: usize, d: usize) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; seq * heads * d];
    for h in 0..heads {
        for s in 0..seq {
            let src_start = (h * seq + s) * d;
            let src_end = (h * seq + s + 1) * d;
            let dst_start = (s * heads + h) * d;
            let dst_end = (s * heads + h + 1) * d;
            let src = checked_slice(attn, src_start, src_end, "merge source row")?;
            let dst = checked_slice_mut(&mut out, dst_start, dst_end, "merge destination row")?;
            dst.copy_from_slice(src);
        }
    }
    Ok(out)
}

/// Repeat each KV head `groups` times along the head axis.
/// `kv`: `[n_kv, seq, d]` -> `[n_kv * groups, seq, d]`.
///
/// # Errors
///
/// [`Error::Shape`] if `kv` is shorter than `n_kv * seq * d`.
fn repeat_kv(kv: &[f32], n_kv: usize, groups: usize, seq: usize, d: usize) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; n_kv * groups * seq * d];
    for h in 0..n_kv {
        let src_start = h * seq * d;
        let src_end = (h + 1) * seq * d;
        let src = checked_slice(kv, src_start, src_end, "repeat_kv source head")?;
        for g in 0..groups {
            let dst_head = h * groups + g;
            let dst_start = dst_head * seq * d;
            let dst_end = (dst_head + 1) * seq * d;
            let dst =
                checked_slice_mut(&mut out, dst_start, dst_end, "repeat_kv destination head")?;
            dst.copy_from_slice(src);
        }
    }
    Ok(out)
}

/// Bounds-checked immutable sub-slice. Unlike an `unsafe`/`get_unchecked`
/// accessor, an out-of-range `start..end` is a returned [`Error::Shape`],
/// never undefined behaviour.
fn checked_slice<'a>(
    x: &'a [f32],
    start: usize,
    end: usize,
    context: &'static str,
) -> Result<&'a [f32]> {
    x.get(start..end).ok_or_else(|| {
        Error::Shape(format!(
            "{context}: range {start}..{end} exceeds slice length {}",
            x.len()
        ))
    })
}

/// Bounds-checked mutable sub-slice. See [`checked_slice`].
fn checked_slice_mut<'a>(
    x: &'a mut [f32],
    start: usize,
    end: usize,
    context: &'static str,
) -> Result<&'a mut [f32]> {
    let len = x.len();
    x.get_mut(start..end).ok_or_else(|| {
        Error::Shape(format!(
            "{context}: range {start}..{end} exceeds slice length {len}"
        ))
    })
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "test assertions use expect()/expect_err() directly"
    )]

    use super::*;

    #[test]
    fn transpose_roundtrips() {
        let seq = 3;
        let heads = 2;
        let d = 2;
        let x: Vec<f32> = (0..seq * heads * d).map(|v| v as f32).collect();
        let t = transpose_seq_heads(&x, seq, heads, d).expect("transpose");
        let back = merge_heads(&t, heads, seq, d).expect("merge");
        assert_eq!(back, x);
    }

    #[test]
    fn repeat_kv_basic() {
        let n_kv = 2;
        let seq = 1;
        let d = 2;
        let kv = vec![1.0_f32, 2.0, 10.0, 20.0];
        let rep = repeat_kv(&kv, n_kv, 3, seq, d).expect("repeat_kv");
        // heads: [kv0,kv0,kv0, kv1,kv1,kv1]
        assert_eq!(
            rep,
            vec![
                1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 10.0, 20.0, 10.0, 20.0, 10.0, 20.0
            ]
        );
    }

    #[test]
    fn forward_rejects_mismatched_rope_head_dim() {
        let cfg = QwenAttentionConfig {
            hidden: 4,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 2,
        };
        let weights = QwenAttentionWeights {
            wq: vec![0.0; 16],
            bq: vec![0.0; 4],
            wk: vec![0.0; 8],
            bk: vec![0.0; 2],
            wv: vec![0.0; 8],
            bv: vec![0.0; 2],
            wo: vec![0.0; 16],
        };
        let attention = QwenAttention::new(cfg, weights).expect("attention");
        let rope = RopeTable::new(1, 4, 1_000_000.0);

        let err = attention
            .forward(&[0.0; 4], &[1], &rope)
            .expect_err("mismatched rope head_dim should be rejected");
        assert!(
            err.to_string()
                .contains("rope.head_dim 4 != cfg.head_dim 2"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn checked_slice_reports_shape_error_instead_of_ub() {
        // WHY: this is the regression test for forkwright/logismos#61 —
        // checked_slice used to be `unsafe { x.get_unchecked(..) }` guarded
        // only by a `debug_assert!`, which is compiled out in release. An
        // out-of-range request must be a returned `Err`, in every profile,
        // not undefined behaviour.
        let x = [1.0_f32, 2.0, 3.0];
        let err = checked_slice(&x, 1, 10, "test range").expect_err("range exceeds slice");
        assert!(err.to_string().contains("test range"));
    }

    #[test]
    fn checked_slice_mut_reports_shape_error_instead_of_ub() {
        let mut x = [1.0_f32, 2.0, 3.0];
        let err =
            checked_slice_mut(&mut x, 2, 5, "test range mut").expect_err("range exceeds slice");
        assert!(err.to_string().contains("test range mut"));
    }
}
