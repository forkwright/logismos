//! Phase-3 CPU reference kernels at fp32 precision.
//!
//! Stella's parity budget is 1e-3 per dim; the final dense head upcasts
//! to fp32 anyway. Rather than round-trip through fp16 on CPU and
//! re-accumulate noise, Phase 3 lands an fp32-throughout reference path.
//! The existing `kernels::*::cpu::*_fp16_ref` functions remain untouched
//! and continue to back the Phase-1 GPU-parity tests.
//!
//! Contracts:
//!
//! - All tensors row-major, contiguous.
//! - All ops are single-threaded and branch-free in their hot loops.
//! - Numerics: sum reductions use `f32` pairwise-ish (iter::sum); softmax
//!   subtracts the per-row max before `exp`; rms_norm accumulates in f32.
//!
//! Shapes follow PyTorch conventions. Where a matmul signature mentions
//! `(m, n, k)` it means `A: [m, k] @ B: [k, n] -> C: [m, n]`.

use std::f32;

use num_traits::ToPrimitive;

fn usize_to_f32(value: usize) -> f32 {
    value.to_f32().unwrap_or(f32::INFINITY)
}

fn usize_to_f64(value: usize) -> f64 {
    value.to_f64().unwrap_or(f64::INFINITY)
}

fn usize_to_isize(value: usize) -> isize {
    isize::try_from(value).unwrap_or(isize::MAX)
}

/// Embedding lookup: `out[b, s, :] = weight[ids[b, s], :]`.
///
/// - `weight`: `[vocab, hidden]`
/// - `ids`: `[b * s]` flat
/// - returns: `[b * s * hidden]`
#[must_use]
pub fn embed_lookup(weight: &[f32], hidden: usize, vocab: usize, ids: &[u32]) -> Vec<f32> {
    let mut out = vec![0.0f32; ids.len() * hidden];
    for (i, &id) in ids.iter().enumerate() {
        let Ok(id) = usize::try_from(id) else {
            continue;
        };
        debug_assert!(id < vocab);
        let src_start = id * hidden;
        let src_end = (id + 1) * hidden;
        let dst_start = i * hidden;
        let dst_end = (i + 1) * hidden;
        if let (Some(src), Some(dst)) = (
            weight.get(src_start..src_end),
            out.get_mut(dst_start..dst_end),
        ) {
            dst.copy_from_slice(src);
        }
    }
    out
}

/// RMSNorm per row.
///
/// `y[r, :] = weight * x[r, :] / sqrt(mean(x[r, :]^2) + eps)`.
///
/// - `x`: `[rows, n]`
/// - `weight`: `[n]`
/// - returns: `[rows, n]`
#[must_use]
pub fn rms_norm(x: &[f32], weight: &[f32], rows: usize, n: usize, eps: f32) -> Vec<f32> {
    debug_assert_eq!(x.len(), rows * n);
    debug_assert_eq!(weight.len(), n);
    let mut y = vec![0.0f32; rows * n];
    for r in 0..rows {
        let row_start = r * n;
        let row_end = (r + 1) * n;
        let Some(row) = x.get(row_start..row_end) else {
            continue;
        };
        let mut sum_sq = 0.0f32;
        for &v in row {
            sum_sq += v * v;
        }
        let inv = ((sum_sq / usize_to_f32(n)) + eps).sqrt().recip();
        if let Some(y_row) = y.get_mut(row_start..row_end) {
            for ((dst, &xv), &wv) in y_row.iter_mut().zip(row.iter()).zip(weight.iter()) {
                *dst = xv * inv * wv;
            }
        }
    }
    y
}

/// `C = A @ B` plus optional bias, all fp32.
///
/// `a`: `[m, k]`, `b`: `[k, n]`, `bias`: `[n]` or empty. Output `[m, n]`.
///
/// Parallelised across rows of the output via rayon. For small `m` the
/// fallback is still a row-at-a-time loop; rayon's scheduler folds small
/// batches onto a single worker.
#[must_use]
pub fn linear(
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    if let Some(bv) = bias {
        debug_assert_eq!(bv.len(), n);
    }
    let mut c = vec![0.0f32; m * n];
    // SAFETY: strides / sizes match the declared shapes; sgemm only
    // writes into `c`; `a` and `b` live beyond the call.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            usize_to_isize(k),
            1,
            b.as_ptr(),
            usize_to_isize(n),
            1,
            0.0,
            c.as_mut_ptr(),
            usize_to_isize(n),
            1,
        );
    }
    if let Some(bv) = bias {
        for row in c.chunks_mut(n) {
            for (slot, &bias_value) in row.iter_mut().zip(bv.iter()) {
                *slot += bias_value;
            }
        }
    }
    c
}

/// `C = A @ B^T`, no bias. HF linear layers store weight as `[out, in]`
/// so `y = x @ W^T + b` is the canonical shape. This routine fuses the
/// transpose into the inner loop; much cheaper than a materialised transpose.
///
/// `a`: `[m, k]`, `b`: `[n, k]` (the untransposed weight). Output `[m, n]`.
#[must_use]
pub fn linear_t(
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    if let Some(bv) = bias {
        debug_assert_eq!(bv.len(), n);
    }

    // C = A @ B^T  where A: [m, k] row-major, B: [n, k] row-major (the
    // HF linear-weight layout). matrixmultiply::sgemm wants C = A @ B
    // with arbitrary strides, so we express B^T as B with swapped row
    // and column strides: B^T[p, j] = B[j, p], meaning a single source
    // stride (rsb, csb) -> (1, k) encodes B as [k, n] with row stride 1
    // and column stride k. This avoids materialising a transpose.
    let mut c = vec![0.0f32; m * n];
    // SAFETY: the buffer lengths / strides match the declared shapes;
    // matrixmultiply::sgemm only writes into `c`, and all input slices
    // are valid for the read extents declared by the row/col strides.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            usize_to_isize(k), // row stride of A (row-major)
            1,                 // col stride of A
            b.as_ptr(),
            1,                 // row stride of B^T (= col stride of B)
            usize_to_isize(k), // col stride of B^T (= row stride of B)
            0.0,
            c.as_mut_ptr(),
            usize_to_isize(n), // row stride of C
            1,                 // col stride of C
        );
    }
    if let Some(bv) = bias {
        // Bias add is cheap; keep it serial to avoid thread contention
        // with the outer batch-parallel loop in `encode_batch`.
        for row in c.chunks_mut(n) {
            for (slot, &bias_value) in row.iter_mut().zip(bv.iter()) {
                *slot += bias_value;
            }
        }
    }
    c
}

/// SiLU / swish: `y = x * sigmoid(x)`.
#[must_use]
pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

/// Elementwise product `c = a * b`.
#[must_use]
pub fn hadamard(a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Row-wise softmax along the last axis (fp32 throughout).
///
/// `x`: `[rows, n]`. Returns `[rows, n]`.
#[must_use]
pub fn softmax_last_dim(x: &[f32], rows: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), rows * n);
    let mut y = vec![0.0f32; rows * n];
    for r in 0..rows {
        let row_start = r * n;
        let row_end = (r + 1) * n;
        let Some(row) = x.get(row_start..row_end) else {
            continue;
        };
        let mut m = f32::NEG_INFINITY;
        for &v in row {
            if v > m {
                m = v;
            }
        }
        let mut denom = 0.0f32;
        for (j, &v) in row.iter().enumerate() {
            let e = (v - m).exp();
            if let Some(slot) = y.get_mut(row_start + j) {
                *slot = e;
            }
            denom += e;
        }
        let inv = denom.recip();
        for j in 0..n {
            if let Some(slot) = y.get_mut(row_start + j) {
                *slot *= inv;
            }
        }
    }
    y
}

/// Apply an additive mask in place. `scores`: `[rows, n]`; `mask`: `[rows_mask, n]`
/// where `rows` is a whole multiple of `rows_mask` (mask broadcasts across the
/// heads dimension). A zero in `mask` zeroes the corresponding `scores` entry
/// by adding `-inf`.
///
/// No-op when `n == 0`, `rows == 0`, or `mask` is empty — there is no work to
/// do and no scores to mask.
pub fn mask_additive_in_place(scores: &mut [f32], mask: &[u8], rows: usize, n: usize) {
    if n == 0 || rows == 0 || mask.is_empty() {
        return;
    }
    debug_assert_eq!(scores.len(), rows * n);
    let mask_rows = mask.len() / n;
    debug_assert_eq!(mask.len(), mask_rows * n);
    if mask_rows == 0 {
        return;
    }
    debug_assert!(rows.is_multiple_of(mask_rows));
    let repeat = rows / mask_rows;
    for mr in 0..mask_rows {
        let mask_start = mr * n;
        let mask_end = (mr + 1) * n;
        let Some(mrow) = mask.get(mask_start..mask_end) else {
            continue;
        };
        for rep in 0..repeat {
            let r = mr * repeat + rep;
            let score_start = r * n;
            let score_end = (r + 1) * n;
            let Some(srow) = scores.get_mut(score_start..score_end) else {
                continue;
            };
            for (score, &mask_value) in srow.iter_mut().zip(mrow.iter()) {
                if mask_value == 0 {
                    *score = f32::NEG_INFINITY;
                }
            }
        }
    }
}

/// Apply RoPE to a tensor shaped `[rows, head_dim]` using the HF-Qwen2
/// halves-rotation convention:
///
/// For each position `pos` (row `rows / heads` ∋ implicit), split the row
/// into two halves of length `head_dim/2`:
/// ```text
/// y[i]         = x[i]         * cos[i] - x[i + d/2] * sin[i]
/// y[i + d/2]   = x[i]         * sin[i] + x[i + d/2] * cos[i]    (i < d/2)
/// ```
/// `cos` and `sin` are `[seq_len, head_dim/2]`; the caller passes the row's
/// position via `pos_ids`, one per output row, so the same table works for
/// both `[B, S, H, D]` collapsed to `[B*S*H, D]` and `[B, H, S, D]` collapsed
/// to `[B*H*S, D]` — the position index per row is the only thing the
/// routine needs.
pub fn rope_halves_in_place(x: &mut [f32], cos: &[f32], sin: &[f32], rows: usize, head_dim: usize) {
    debug_assert_eq!(x.len(), rows * head_dim);
    debug_assert!(head_dim.is_multiple_of(2));
    let half = head_dim / 2;
    debug_assert_eq!(cos.len() % half, 0);
    debug_assert_eq!(sin.len(), cos.len());
    // Note: `rows` encodes (batch*heads, seq) — callers supply a flat cos/sin
    // laid out as `[seq, half]` and iterate rows with pos=(row % seq). For a
    // naive single-sequence encoder forward, the caller must pre-gather the
    // per-row `cos, sin` slice. Here we accept a flat `(rows, half)` cos/sin
    // so the caller explicitly materialises the per-row tables. That is done
    // in `transformers::StellaAttention::forward`.
    debug_assert_eq!(cos.len(), rows * half);
    for r in 0..rows {
        let row_start = r * head_dim;
        let row_end = (r + 1) * head_dim;
        let cos_start = r * half;
        let cos_end = (r + 1) * half;
        let Some(row) = x.get_mut(row_start..row_end) else {
            continue;
        };
        let Some(cr) = cos.get(cos_start..cos_end) else {
            continue;
        };
        let Some(sr) = sin.get(cos_start..cos_end) else {
            continue;
        };
        for i in 0..half {
            let Some(&x0) = row.get(i) else {
                continue;
            };
            let Some(&x1) = row.get(i + half) else {
                continue;
            };
            let Some((&cos_value, &sin_value)) = cr.get(i).zip(sr.get(i)) else {
                continue;
            };
            if let Some(slot) = row.get_mut(i) {
                *slot = x0 * cos_value - x1 * sin_value;
            }
            if let Some(slot) = row.get_mut(i + half) {
                *slot = x0 * sin_value + x1 * cos_value;
            }
        }
    }
}

/// Mean pool a `[seq, hidden]` tensor along `seq` using a `[seq]` attention
/// mask. Denominator is clamped to `max(1, sum(mask))` to survive all-pad
/// rows (non-parity CI edge case).
#[must_use]
pub fn mean_pool_masked(h: &[f32], mask: &[u8], seq: usize, hidden: usize) -> Vec<f32> {
    debug_assert_eq!(h.len(), seq * hidden);
    debug_assert_eq!(mask.len(), seq);
    let mut out = vec![0.0f32; hidden];
    let mut den = 0.0f32;
    for s in 0..seq {
        if mask.get(s).copied().unwrap_or(0) == 0 {
            continue;
        }
        den += 1.0;
        let row_start = s * hidden;
        let row_end = (s + 1) * hidden;
        let Some(row) = h.get(row_start..row_end) else {
            continue;
        };
        for (slot, &value) in out.iter_mut().zip(row.iter()) {
            *slot += value;
        }
    }
    let inv = if den > 0.0 { den.recip() } else { 1.0 };
    for v in &mut out {
        *v *= inv;
    }
    out
}

/// L2-normalise a `[hidden]` vector in place. Denominator is clamped to
/// `1e-12` to avoid NaN on zero input (safety net; real Stella outputs are
/// never zero).
pub fn l2_normalize_in_place(v: &mut [f32]) {
    let mut sq = 0.0f32;
    for &x in v.iter() {
        sq += x * x;
    }
    let inv = sq.sqrt().max(1e-12).recip();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Build a Qwen2-style `(seq, head_dim/2)` cos+sin table. Returns
/// `(cos, sin)` each of length `seq * head_dim / 2`, fp32.
///
/// `inv_freq` uses base `theta`, evaluated in f64 to curb accumulation error
/// as recommended by the Phase-3 PLAN §12.1.
#[must_use]
pub fn build_rope_table_f32(seq: usize, head_dim: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    debug_assert!(head_dim.is_multiple_of(2));
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; seq * half];
    let mut sin = vec![0.0f32; seq * half];
    for pos in 0..seq {
        for i in 0..half {
            // inv_freq[i] = 1 / theta^(2i / head_dim)
            let exp = (2.0 * usize_to_f64(i)) / usize_to_f64(head_dim);
            let inv_freq = theta.powf(-exp);
            let angle = usize_to_f64(pos) * inv_freq;
            if let Some(slot) = cos.get_mut(pos * half + i) {
                *slot = angle.cos().to_f32().unwrap_or(0.0);
            }
            if let Some(slot) = sin.get_mut(pos * half + i) {
                *slot = angle.sin().to_f32().unwrap_or(0.0);
            }
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_matches_manual() {
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let w = [1.0_f32; 4];
        let y = rms_norm(&x, &w, 1, 4, 1e-6);
        let inv = 1.0_f32 / f32::sqrt(((1.0 + 4.0 + 9.0 + 16.0) / 4.0) + 1e-6);
        let want = [1.0 * inv, 2.0 * inv, 3.0 * inv, 4.0 * inv];
        for (a, b) in y.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-6, "a={a} b={b}");
        }
    }

    #[test]
    fn linear_t_matches_naive() {
        // x: [1, 3], w: [2, 3] → y: [1, 2]
        let x = [1.0_f32, 2.0, 3.0];
        let w = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 1.0];
        let b = [10.0_f32, 20.0];
        let y = linear_t(&x, &w, Some(&b), 1, 2, 3);
        assert!((y[0] - (1.0 + 10.0)).abs() < 1e-6);
        assert!((y[1] - (2.0 + 3.0 + 20.0)).abs() < 1e-6);
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let x = [0.0_f32, 1.0, 2.0, -1.0, 0.0, 1.0];
        let y = softmax_last_dim(&x, 2, 3);
        for r in 0..2 {
            let s: f32 = y[r * 3..(r + 1) * 3].iter().sum();
            assert!((s - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_zero_pos_identity_halves() {
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let (cos, sin) = build_rope_table_f32(1, 4, 1_000_000.0);
        rope_halves_in_place(&mut x, &cos, &sin, 1, 4);
        assert_eq!(x, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rope_rotation_is_inverse_after_pi() {
        // After a full 2π rotation the vector returns; use theta=1 so angle
        // grows fast. Direct check: rotate forward, rotate backward → identity.
        let mut x = vec![1.0_f32, 2.0];
        let cos = vec![0.5_f32.sqrt()];
        let sin = vec![0.5_f32.sqrt()];
        let orig = x.clone();
        rope_halves_in_place(&mut x, &cos, &sin, 1, 2);
        // Inverse rotation: swap sign of sin.
        let inv_sin = vec![-0.5_f32.sqrt()];
        rope_halves_in_place(&mut x, &cos, &inv_sin, 1, 2);
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn mean_pool_respects_mask() {
        let h = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = [1_u8, 0, 1];
        let pooled = mean_pool_masked(&h, &mask, 3, 2);
        assert_eq!(pooled, vec![3.0, 4.0]);
    }

    #[test]
    fn l2_normalize_projects_to_unit() {
        let mut v = vec![3.0_f32, 4.0];
        l2_normalize_in_place(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn mask_additive_broadcasts_across_heads() {
        let mut s = vec![1.0_f32; 12]; // 4 rows, 3 cols
        let mask = [1_u8, 0, 1, 1, 1, 1]; // 2 mask rows
        mask_additive_in_place(&mut s, &mask, 4, 3);
        // row 0 + 1 see mask row 0 [1,0,1]; row 2+3 see mask row 1 [1,1,1]
        assert!(s[0].is_finite());
        assert!(s[1].is_infinite() && s[1].is_sign_negative());
        assert!(s[2].is_finite());
        assert!(s[4].is_infinite() && s[4].is_sign_negative());
        assert!(s[6..].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mask_additive_zero_n_does_not_panic() {
        // n == 0: mask.len() / n would divide by zero pre-fix.
        let mut s: Vec<f32> = vec![];
        let mask: [u8; 0] = [];
        mask_additive_in_place(&mut s, &mask, 4, 0);
    }

    #[test]
    fn mask_additive_empty_mask_does_not_panic() {
        // rows > 0 with an empty mask: mask_rows == 0, so rows / mask_rows
        // would divide by zero pre-fix.
        let mut s = vec![1.0_f32; 12]; // 4 rows, 3 cols
        let mask: [u8; 0] = [];
        mask_additive_in_place(&mut s, &mask, 4, 3);
        // No-op: scores are untouched since there is no mask to apply.
        assert!(s.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn silu_matches_manual() {
        let y = silu(&[0.0, 1.0, -1.0]);
        assert!((y[0]).abs() < 1e-6);
        assert!((y[1] - 1.0 / (1.0 + (-1.0_f32).exp())).abs() < 1e-6);
        assert!((y[2] - (-1.0) / (1.0 + 1.0_f32.exp())).abs() < 1e-6);
    }
}
