//! ModernBERT transformer building blocks.
//!
//! Bidirectional encoder with alternating local (sliding-window) and
//! global attention layers. Activation is GELU. Normalization is
//! LayerNorm (not RMSNorm). RoPE uses two distinct theta values for
//! local and global layers.
//!
//! Phase 5 scope: fp32 CPU reference. Tolerance gate is 1e-3 against
//! Transformers `AutoModel` golden vectors.

use kernels::cpu_f32;

use crate::error::{Error, Result};
use crate::rope::RopeTable;

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

/// Per-row LayerNorm (mean-variance, additive bias optional).
///
/// `y[r,i] = (x[r,i] - mean(r)) / sqrt(var(r) + eps) * weight[i] + bias[i]`
///
/// ModernBERT's `norm_bias = false` in the default config so `bias` is empty
/// in that case — the caller passes an empty slice.
#[must_use]
pub fn layer_norm_f32(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    rows: usize,
    n: usize,
    eps: f32,
) -> Vec<f32> {
    debug_assert_eq!(x.len(), rows * n);
    debug_assert_eq!(weight.len(), n);
    debug_assert!(bias.is_empty() || bias.len() == n);
    let use_bias = !bias.is_empty();
    let mut output = vec![0.0f32; rows * n];
    for r in 0..rows {
        let start = r * n;
        let end = (r + 1) * n;
        let Some(row) = x.get(start..end) else {
            continue;
        };
        // WHY: n is bounded by hidden_size (max 4096); precision loss at 2^24 is unreachable.
        #[expect(clippy::cast_precision_loss, reason = "n < 2^24 for any realistic hidden_size")]
        let n_f32 = n as f32;
        let mean: f32 = row.iter().sum::<f32>() / n_f32;
        let var: f32 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n_f32;
        let inv = (var + eps).sqrt().recip();
        let Some(y_row) = output.get_mut(start..end) else {
            continue;
        };
        for (i, ((dst, &xv), &wv)) in y_row
            .iter_mut()
            .zip(row.iter())
            .zip(weight.iter())
            .enumerate()
        {
            let normalized = (xv - mean) * inv * wv;
            *dst = if use_bias {
                normalized + bias.get(i).copied().unwrap_or(0.0)
            } else {
                normalized
            };
        }
    }
    output
}

/// GELU activation (exact, non-approximate).
///
///
///
/// erf via Horner-form rational approximation (A&S 7.1.26); max error < 1.5e-7.
#[must_use]
pub fn gelu(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            let t = 1.0_f32 / (1.0 + 0.327_591_1 * v.abs());
            let poly = t
                * (0.254_829_592
                    + t * (-0.284_496_736
                        + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
            let erf_abs = 1.0 - poly * (-(v * v)).exp();
            let erf_v = if v >= 0.0 { erf_abs } else { -erf_abs };
            0.5 * v * (1.0 + erf_v)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// GeGLU MLP
// ---------------------------------------------------------------------------

/// GeGLU MLP weights — ModernBERT convention.
///
/// HF weight names: `mlp.Wi.weight` (gate+up fused, `[2*intermediate, hidden]`),
/// `mlp.Wo.weight` (down, `[hidden, intermediate]`). Optional biases per
/// `mlp_bias` config flag.
#[derive(Debug, Clone)]
pub struct GeGluMlpWeights {
    /// `mlp.Wi.weight` — `[2 * intermediate, hidden]` (gate slice + up slice).
    pub wi: Vec<f32>,
    /// `mlp.Wo.weight` — `[hidden, intermediate]`.
    pub wo: Vec<f32>,
    /// `mlp.Wi.bias` — `[2 * intermediate]` or empty.
    pub bi: Vec<f32>,
    /// `mlp.Wo.bias` — `[hidden]` or empty.
    pub bo: Vec<f32>,
}

/// GeGLU MLP block used by ModernBERT.
///
/// `y = Wo(gelu(Wi_gate(x)) * Wi_up(x))`
#[derive(Debug, Clone)]
pub struct GeGluMlp {
    /// Hidden dimension.
    pub hidden: usize,
    /// Intermediate dimension (`Wi` output width is `2 * intermediate`).
    pub intermediate: usize,
    /// Weights.
    pub weights: GeGluMlpWeights,
}

impl GeGluMlp {
    /// Construct with shape-checking.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] on weight-size / config disagreement.
    pub fn new(hidden: usize, intermediate: usize, weights: GeGluMlpWeights) -> Result<Self> {
        let wi_expected = 2 * intermediate * hidden;
        if weights.wi.len() != wi_expected {
            return Err(Error::Shape(format!(
                "Wi: expected {wi_expected} elements, got {}",
                weights.wi.len()
            )));
        }
        if weights.wo.len() != hidden * intermediate {
            return Err(Error::Shape(format!(
                "Wo: expected {} elements, got {}",
                hidden * intermediate,
                weights.wo.len()
            )));
        }
        if !weights.bi.is_empty() && weights.bi.len() != 2 * intermediate {
            return Err(Error::Shape(format!(
                "Wi bias: expected {} or 0, got {}",
                2 * intermediate,
                weights.bi.len()
            )));
        }
        if !weights.bo.is_empty() && weights.bo.len() != hidden {
            return Err(Error::Shape(format!(
                "Wo bias: expected {hidden} or 0, got {}",
                weights.bo.len()
            )));
        }
        Ok(Self {
            hidden,
            intermediate,
            weights,
        })
    }

    /// Forward pass over a `[seq, hidden]` input.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] when `x.len()` is not a multiple of `hidden`.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        if !x.len().is_multiple_of(self.hidden) {
            return Err(Error::Shape(format!(
                "geglu.forward: x.len()={} not multiple of hidden={}",
                x.len(),
                self.hidden
            )));
        }
        let seq = x.len() / self.hidden;
        // wi_out = x @ Wi^T -> [seq, 2*intermediate]
        let wi_bias = if self.weights.bi.is_empty() {
            None
        } else {
            Some(self.weights.bi.as_slice())
        };
        let wi_out = cpu_f32::linear_t(
            x,
            &self.weights.wi,
            wi_bias,
            seq,
            2 * self.intermediate,
            self.hidden,
        );
        // Split into gate and up halves: each [seq, intermediate]
        let half = self.intermediate;
        let mut gate = Vec::with_capacity(seq * half);
        let mut up = Vec::with_capacity(seq * half);
        for s in 0..seq {
            let row_start = s * 2 * half;
            gate.extend_from_slice(wi_out.get(row_start..row_start + half).unwrap_or_default());
            up.extend_from_slice(
                wi_out
                    .get(row_start + half..row_start + 2 * half)
                    .unwrap_or_default(),
            );
        }
        // gelu(gate) * up
        let gate_act = gelu(&gate);
        let prod = cpu_f32::hadamard(&gate_act, &up);
        // wo_out = prod @ Wo^T -> [seq, hidden]
        let wo_bias = if self.weights.bo.is_empty() {
            None
        } else {
            Some(self.weights.bo.as_slice())
        };
        let out = cpu_f32::linear_t(
            &prod,
            &self.weights.wo,
            wo_bias,
            seq,
            self.hidden,
            self.intermediate,
        );
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// ModernBERT attention
// ---------------------------------------------------------------------------

/// ModernBERT attention configuration.
#[derive(Debug, Clone, Copy)]
pub struct ModernBertAttentionConfig {
    /// Hidden size.
    pub hidden: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Per-head dimension (`hidden / n_heads`).
    pub head_dim: usize,
    /// Local attention window (half-window on each side).
    pub local_window: usize,
    /// Whether this layer uses global (full bidirectional) attention.
    pub is_global: bool,
    /// Whether QKV projections have bias.
    pub attention_bias: bool,
}

/// Per-layer ModernBERT attention weights.
#[derive(Debug, Clone)]
pub struct ModernBertAttentionWeights {
    /// `attn.Wqkv.weight` — `[3*hidden, hidden]` or per-head Q+K+V fused.
    pub wqkv: Vec<f32>,
    /// `attn.Wqkv.bias` — `[3*hidden]` or empty when `attention_bias=false`.
    pub bqkv: Vec<f32>,
    /// `attn.Wo.weight` — `[hidden, hidden]`.
    pub wo: Vec<f32>,
    /// `attn.Wo.bias` — `[hidden]` or empty.
    pub bo: Vec<f32>,
}

/// ModernBERT attention block (local sliding-window or global bidirectional).
#[derive(Debug, Clone)]
pub struct ModernBertAttention {
    /// Configuration.
    pub cfg: ModernBertAttentionConfig,
    /// Weights.
    pub weights: ModernBertAttentionWeights,
}

impl ModernBertAttention {
    /// Construct with shape-checking.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] on weight-size mismatch.
    pub fn new(
        cfg: ModernBertAttentionConfig,
        weights: ModernBertAttentionWeights,
    ) -> Result<Self> {
        let h = cfg.hidden;
        let w3h = 3 * h;
        if weights.wqkv.len() != w3h * h {
            return Err(Error::Shape(format!(
                "Wqkv: expected {}, got {}",
                w3h * h,
                weights.wqkv.len()
            )));
        }
        if cfg.attention_bias && weights.bqkv.len() != w3h {
            return Err(Error::Shape(format!(
                "bqkv: expected {w3h}, got {}",
                weights.bqkv.len()
            )));
        }
        if weights.wo.len() != h * h {
            return Err(Error::Shape(format!(
                "Wo: expected {}, got {}",
                h * h,
                weights.wo.len()
            )));
        }
        if cfg.n_heads == 0 || h != cfg.n_heads * cfg.head_dim {
            return Err(Error::Shape(format!(
                "hidden={h} != n_heads={} * head_dim={}",
                cfg.n_heads, cfg.head_dim
            )));
        }
        Ok(Self { cfg, weights })
    }

    /// Forward pass.
    ///
    /// - `x`: `[seq, hidden]`
    /// - `rope`: per-token cos/sin slice — caller gathers from the correct table.
    /// - `mask`: `[seq]` attention mask (1=valid, 0=padding).
    ///
    /// Returns `[seq, hidden]`.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] on size disagreement.
    pub fn forward(
        &self,
        x: &[f32],
        rope: &RopeTable,
        positions: &[usize],
        mask: &[u8],
    ) -> Result<Vec<f32>> {
        let seq = positions.len();
        if x.len() != seq * self.cfg.hidden {
            return Err(Error::Shape(format!(
                "attention.forward: x.len()={} != seq*hidden={}*{}",
                x.len(),
                seq,
                self.cfg.hidden
            )));
        }
        let h = self.cfg.hidden;
        let n_h = self.cfg.n_heads;
        let d = self.cfg.head_dim;
        // QKV projection: [seq, 3*h]
        let bqkv = if self.cfg.attention_bias {
            Some(self.weights.bqkv.as_slice())
        } else {
            None
        };
        let qkv = cpu_f32::linear_t(x, &self.weights.wqkv, bqkv, seq, 3 * h, h);
        // Split into Q, K, V — each [seq, h]
        let mut q = Vec::with_capacity(seq * h);
        let mut k = Vec::with_capacity(seq * h);
        let mut v = Vec::with_capacity(seq * h);
        for s in 0..seq {
            let start = s * 3 * h;
            q.extend_from_slice(qkv.get(start..start + h).unwrap_or_default());
            k.extend_from_slice(qkv.get(start + h..start + 2 * h).unwrap_or_default());
            v.extend_from_slice(qkv.get(start + 2 * h..start + 3 * h).unwrap_or_default());
        }
        // Reshape to [seq, n_h, d] for RoPE; apply per-head
        // RoPE is applied per-head, halves-rotation convention.
        // Reshape Q/K to [n_h, seq, d] for per-head processing.
        let q_hsd = transpose_seq_heads(&q, seq, n_h, d);
        let k_hsd = transpose_seq_heads(&k, seq, n_h, d);
        let v_hsd = transpose_seq_heads(&v, seq, n_h, d);
        // Gather RoPE for each position
        let (cos_rows, sin_rows) = rope.gather(positions);
        // Apply RoPE per head to Q and K
        let mut q_rope = q_hsd.clone();
        let mut k_rope = k_hsd.clone();
        for h_idx in 0..n_h {
            let start = h_idx * seq * d;
            let end = (h_idx + 1) * seq * d;
            if let Some(q_head) = q_rope.get_mut(start..end) {
                cpu_f32::rope_halves_in_place(q_head, &cos_rows, &sin_rows, seq, d);
            }
            if let Some(k_head) = k_rope.get_mut(start..end) {
                cpu_f32::rope_halves_in_place(k_head, &cos_rows, &sin_rows, seq, d);
            }
        }
        // Compute attention scores [n_h, seq, seq]
        // WHY: d is head_dim (64 or 128); precision loss at 2^24 is unreachable.
        #[expect(clippy::cast_precision_loss, reason = "head_dim <= 128, well within f32 exact range")]
        let scale = (d as f32).sqrt().recip();
        let mut scores = vec![0.0f32; n_h * seq * seq];
        for h_idx in 0..n_h {
            let q_head = &q_rope[h_idx * seq * d..(h_idx + 1) * seq * d];
            let k_head = &k_rope[h_idx * seq * d..(h_idx + 1) * seq * d];
            // [seq, seq] = [seq, d] @ [d, seq]
            let sc = cpu_f32::linear_t(q_head, k_head, None, seq, seq, d);
            let dst = &mut scores[h_idx * seq * seq..(h_idx + 1) * seq * seq];
            for (s, v) in dst.iter_mut().zip(sc.iter()) {
                *s = v * scale;
            }
        }
        // Apply attention mask (padding and optionally local window)
        for h_idx in 0..n_h {
            let head = &mut scores[h_idx * seq * seq..(h_idx + 1) * seq * seq];
            apply_attention_mask(head, mask, seq, self.cfg.local_window, self.cfg.is_global);
        }
        // Softmax per row per head
        for h_idx in 0..n_h {
            let head = &mut scores[h_idx * seq * seq..(h_idx + 1) * seq * seq];
            let sm = cpu_f32::softmax_last_dim(head, seq, seq);
            head.copy_from_slice(&sm);
        }
        // Attention output: [n_h, seq, d] = softmax @ V
        let mut attn = vec![0.0f32; n_h * seq * d];
        for h_idx in 0..n_h {
            let sh = &scores[h_idx * seq * seq..(h_idx + 1) * seq * seq];
            let vh = &v_hsd[h_idx * seq * d..(h_idx + 1) * seq * d];
            let ah = cpu_f32::linear(sh, vh, None, seq, d, seq);
            let dst = &mut attn[h_idx * seq * d..(h_idx + 1) * seq * d];
            dst.copy_from_slice(&ah);
        }
        // Merge heads: [n_h, seq, d] -> [seq, hidden]
        let merged = merge_heads(&attn, n_h, seq, d);
        // Output projection
        let bo = if self.weights.bo.is_empty() {
            None
        } else {
            Some(self.weights.bo.as_slice())
        };
        let out = cpu_f32::linear_t(&merged, &self.weights.wo, bo, seq, h, h);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Attention mask helpers
// ---------------------------------------------------------------------------

/// Apply padding + optional local-window mask to scores `[seq, seq]`.
///
/// Padding: set column `j` to `-inf` wherever `mask[j] == 0`.
/// Local window: for non-global layers, also mask positions `j` where
/// `|i - j| > local_window`.
fn apply_attention_mask(
    scores: &mut [f32],
    mask: &[u8],
    seq: usize,
    local_window: usize,
    is_global: bool,
) {
    debug_assert_eq!(scores.len(), seq * seq);
    for i in 0..seq {
        for j in 0..seq {
            let Some(slot) = scores.get_mut(i * seq + j) else {
                continue;
            };
            let padding_ok = mask.get(j).copied().unwrap_or(0) != 0;
            let window_ok = if is_global {
                true
            } else {
                let diff = if i >= j { i - j } else { j - i };
                diff <= local_window
            };
            if !padding_ok || !window_ok {
                *slot = f32::NEG_INFINITY;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Head transpose helpers (same as in attention.rs)
// ---------------------------------------------------------------------------

/// `[seq, heads, head_dim] -> [heads, seq, head_dim]`.
fn transpose_seq_heads(x: &[f32], seq: usize, heads: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq * heads * d];
    for s in 0..seq {
        for h in 0..heads {
            let src = &x[(s * heads + h) * d..(s * heads + h + 1) * d];
            let dst = &mut out[(h * seq + s) * d..(h * seq + s + 1) * d];
            dst.copy_from_slice(src);
        }
    }
    out
}

/// `[heads, seq, head_dim] -> [seq, heads * head_dim]`.
fn merge_heads(attn: &[f32], heads: usize, seq: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq * heads * d];
    for h in 0..heads {
        for s in 0..seq {
            let src = &attn[(h * seq + s) * d..(h * seq + s + 1) * d];
            let dst = &mut out[(s * heads + h) * d..(s * heads + h + 1) * d];
            dst.copy_from_slice(src);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests intentionally unwrap successful fixture calls to keep failure locations precise"
)]
mod tests {
    use super::*;

    #[test]
    fn layer_norm_unit_weights_zero_bias() {
        // With weight=1, bias=0, LN should normalize to zero mean unit variance.
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let w = [1.0_f32; 4];
        let b: [f32; 0] = [];
        let y = layer_norm_f32(&x, &w, &b, 1, 4, 1e-5);
        let mean: f32 = y.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean should be near 0, got {mean}");
        let var: f32 = y.iter().map(|&v| v * v).sum::<f32>() / 4.0;
        assert!(
            (var - 1.0).abs() < 1e-4,
            "variance should be near 1, got {var}"
        );
    }

    #[test]
    fn layer_norm_with_bias() {
        let x = [0.0_f32; 4];
        let w = [2.0_f32; 4];
        let b = [1.0_f32; 4];
        let y = layer_norm_f32(&x, &w, &b, 1, 4, 1e-5);
        // All zeros: mean=0, var=0, so after LN: 0 * w + b = b = 1
        for &v in &y {
            assert!((v - 1.0).abs() < 1e-4, "expected 1.0 got {v}");
        }
    }

    #[test]
    fn gelu_zero_is_zero() {
        let y = gelu(&[0.0_f32]);
        assert!(y[0].abs() < 1e-7);
    }

    #[test]
    fn gelu_positive_is_positive() {
        let y = gelu(&[1.0_f32]);
        assert!(y[0] > 0.0 && y[0] < 1.0, "gelu(1) ≈ 0.841, got {}", y[0]);
        assert!((y[0] - 0.8413_f32).abs() < 1e-3);
    }

    #[test]
    fn geglu_mlp_zero_input_zero_output() {
        let hidden = 4;
        let intermediate = 8;
        let wi = vec![0.0f32; 2 * intermediate * hidden];
        let wo = vec![0.0f32; hidden * intermediate];
        let weights = GeGluMlpWeights {
            wi,
            wo,
            bi: vec![],
            bo: vec![],
        };
        let mlp = GeGluMlp::new(hidden, intermediate, weights).unwrap();
        let x = vec![1.0f32; hidden];
        let out = mlp.forward(&x).unwrap();
        assert_eq!(out.len(), hidden);
        assert!(out.iter().all(|&v| v.abs() < 1e-7));
    }

    #[test]
    fn geglu_mlp_identity_gate_pass_through() {
        // Wi_gate = 0 (GELU(0) = 0 → kills output regardless of Wi_up)
        let hidden = 2;
        let intermediate = 2;
        let wi = vec![0.0f32; 2 * intermediate * hidden];
        let wo = vec![1.0f32; hidden * intermediate];
        let weights = GeGluMlpWeights {
            wi,
            wo,
            bi: vec![],
            bo: vec![],
        };
        let mlp = GeGluMlp::new(hidden, intermediate, weights).unwrap();
        let x = vec![1.0f32; hidden];
        let out = mlp.forward(&x).unwrap();
        assert!(
            out.iter().all(|&v| v.abs() < 1e-7),
            "gate=0 must zero output"
        );
    }

    #[test]
    fn attention_forward_output_shape() {
        let hidden = 8;
        let n_heads = 2;
        let head_dim = 4;
        let seq = 3;
        let cfg = ModernBertAttentionConfig {
            hidden,
            n_heads,
            head_dim,
            local_window: 128,
            is_global: true,
            attention_bias: false,
        };
        let wqkv = vec![0.0f32; 3 * hidden * hidden];
        let wo = vec![0.0f32; hidden * hidden];
        let weights = ModernBertAttentionWeights {
            wqkv,
            bqkv: vec![],
            wo,
            bo: vec![],
        };
        let attn = ModernBertAttention::new(cfg, weights).unwrap();
        let x = vec![1.0f32; seq * hidden];
        let rope = RopeTable::new(128, head_dim, 10_000.0);
        let positions: Vec<usize> = (0..seq).collect();
        let mask = vec![1u8; seq];
        let out = attn.forward(&x, &rope, &positions, &mask).unwrap();
        assert_eq!(out.len(), seq * hidden);
    }
}
