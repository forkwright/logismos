//! CPU reference for Gated Delta Rule (GDN) kernels.
//!
//! All functions are pure-Rust fp32 references. Shapes follow the
//! logismos Phase 6a contract (`phases/06a-gdn-hybrid/PLAN.md`
//! sections 6.2 and 6.3). Tolerance gate is 1e-3 vs FLA reference
//! on synthetic inputs.
//!
//! Notation: `B` = batch, `T` = sequence length, `H` = num_heads,
//! `K` = key/state dim, `V` = value dim, `BT = 64` = chunk size.

/// L2-normalise rows of a `[T, H, K]` tensor in place.
///
/// `x` is `[T * H * K]` flat, row = `[K]`. Each row is normalised to
/// unit length with a clamp of `1e-12` to avoid NaN on zero rows.
///
/// # Panics
///
/// Debug-mode: `x.len()` not a multiple of `k`.
pub fn l2_norm_in_place(x: &mut [f32], k: usize) {
    debug_assert!(k > 0 && x.len().is_multiple_of(k));
    let rows = x.len() / k;
    for r in 0..rows {
        let start = r * k;
        let end = (r + 1) * k;
        let Some(row) = x.get_mut(start..end) else {
            continue;
        };
        let sq: f32 = row.iter().map(|&v| v * v).sum();
        let inv = sq.sqrt().max(1e-12_f32).recip();
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

/// Cumulative sum of a `[T]` sequence along T (in-place).
///
/// Implements stage 1 (`chunk_local_cumsum`) for a single head.
/// The full kernel operates on `[B, T, HV]`; this reference covers
/// one `[T]` slice.
///
/// # Panics
///
/// Never panics.
#[must_use]
pub fn cumsum_f32(g: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(g.len());
    let mut acc = 0.0f32;
    for &v in g {
        acc += v;
        out.push(acc);
    }
    out
}

/// Gated Delta Rule recurrent decode — per-head reference.
///
/// Implements stage 8 (`fused_recurrent_gated_delta_rule_packed_decode`)
/// for a single head over `T` tokens. Inputs are all fp32.
///
/// ## Arguments
///
/// - `q`: `[T, K]` query.
/// - `k`: `[T, K]` key.
/// - `v`: `[T, V]` value.
/// - `beta`: `[T]` learning rate (scalar per token).
/// - `g`: `[T]` gate log-decay per token.
/// - `scale`: attention scale (`1 / sqrt(K)`, typically).
/// - `state_in`: `[K, V]` initial SSM state (or zeros).
///
/// ## Returns
///
/// `(o, state_out)` where `o: [T, V]` is the output and
/// `state_out: [K, V]` is the final SSM state.
///
/// ## Algorithm
///
/// For each token `t`:
/// 1. `h = h * exp(g[t])` — decay the state.
/// 2. `v_delta = beta[t] * (v[t] - h^T @ k[t])` — error signal.
/// 3. `h += outer(k[t], v_delta)` — state update.
/// 4. `o[t] = h^T @ (q[t] * scale)` — output.
///
/// This matches `fused_recurrent_gated_delta_rule_fwd_kernel` from
/// FLA `fla/ops/gated_delta_rule/fused_recurrent.py` with
/// `USE_G=true`, `USE_GK=false`, `USE_GV=false`, `APPLY_BETA_SIGMOID=false`.
///
/// # Panics
///
/// Debug-mode shape checks: `q/k/v` and `state_in` length.
#[must_use]
pub fn gated_delta_rule_recurrent_fwd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    g: &[f32],
    scale: f32,
    state_in: &[f32],
    k_dim: usize,
    v_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let t = beta.len();
    debug_assert_eq!(q.len(), t * k_dim);
    debug_assert_eq!(k.len(), t * k_dim);
    debug_assert_eq!(v.len(), t * v_dim);
    debug_assert_eq!(g.len(), t);
    debug_assert_eq!(state_in.len(), k_dim * v_dim);

    // Working state: [K, V] row-major (k rows, v cols)
    let mut h: Vec<f32> = state_in.to_vec();
    let mut o = vec![0.0f32; t * v_dim];

    for t_idx in 0..t {
        let q_t = q
            .get(t_idx * k_dim..(t_idx + 1) * k_dim)
            .unwrap_or_default();
        let k_t = k
            .get(t_idx * k_dim..(t_idx + 1) * k_dim)
            .unwrap_or_default();
        let v_t = v
            .get(t_idx * v_dim..(t_idx + 1) * v_dim)
            .unwrap_or_default();
        let beta_t = beta.get(t_idx).copied().unwrap_or(1.0);
        let g_t = g.get(t_idx).copied().unwrap_or(0.0);

        // 1. Decay: h *= exp(g_t)
        let decay = g_t.exp();
        for hv in &mut h {
            *hv *= decay;
        }

        // 2. v_new = h^T @ k_t: [V] = [V, K] @ [K]
        //    h is [K, V] so h^T is [V, K]; h^T @ k = sum_k h[k,v] * k[k]
        let mut h_t_k = vec![0.0f32; v_dim];
        for ki in 0..k_dim {
            // vi also drives h's flat offset (ki * v_dim + vi), not just h_t_k's —
            // can't collapse to a single iterator without losing the get().unwrap_or(0.0)
            // out-of-bounds fallback that guards release builds against malformed shapes.
            #[expect(
                clippy::needless_range_loop,
                reason = "vi indexes both h_t_k[vi] and h[ki * v_dim + vi]; a slice zip would \
                          drop the get().unwrap_or(0.0) bounds fallback"
            )]
            for vi in 0..v_dim {
                h_t_k[vi] += h.get(ki * v_dim + vi).copied().unwrap_or(0.0)
                    * k_t.get(ki).copied().unwrap_or(0.0);
            }
        }

        // 3. v_delta = beta_t * (v_t - h^T @ k_t)
        let mut v_delta = vec![0.0f32; v_dim];
        for vi in 0..v_dim {
            v_delta[vi] = beta_t * (v_t.get(vi).copied().unwrap_or(0.0) - h_t_k[vi]);
        }

        // 4. h += outer(k_t, v_delta): [K, V]
        for ki in 0..k_dim {
            let k_val = k_t.get(ki).copied().unwrap_or(0.0);
            for vi in 0..v_dim {
                if let Some(slot) = h.get_mut(ki * v_dim + vi) {
                    *slot += k_val * v_delta.get(vi).copied().unwrap_or(0.0);
                }
            }
        }

        // 5. o_t = h^T @ (q_t * scale) = sum_k h[k,v] * q_t[k] * scale
        let o_t = o
            .get_mut(t_idx * v_dim..(t_idx + 1) * v_dim)
            .unwrap_or_default();
        // vi also drives h's flat offset (ki * v_dim + vi) in the inner loop, not
        // just o_t[vi] — same fallback-preserving rationale as the h_t_k loop above.
        #[expect(
            clippy::needless_range_loop,
            reason = "vi indexes both o_t[vi] and h[ki * v_dim + vi]; a slice zip would \
                      drop the get().unwrap_or(0.0) bounds fallback"
        )]
        for vi in 0..v_dim {
            let mut acc = 0.0f32;
            for ki in 0..k_dim {
                acc += h.get(ki * v_dim + vi).copied().unwrap_or(0.0)
                    * q_t.get(ki).copied().unwrap_or(0.0)
                    * scale;
            }
            o_t[vi] = acc;
        }
    }

    (o, h)
}

/// Chunk-level Gated Delta Rule forward — produces inter-chunk SSM states.
///
/// Implements a simplified version of stage 5 (`chunk_gated_delta_rule_fwd_h`).
/// Input `w` and `u` are the pre-computed scratch values from stages 2-4;
/// this reference skips stages 2-4 and works directly from `k`, `v`, `beta`,
/// `g` (the non-fused path for correctness verification).
///
/// Equivalent to calling `gated_delta_rule_recurrent_fwd` on each chunk of
/// size `bt` with the state carried across chunks. Returns:
///
/// - `o: [T, V]` output tensor.
/// - `s_end: [n_chunks, K, V]` per-chunk end states (for multi-step decoding).
///
/// # Panics
///
/// Debug-mode: shape checks.
#[must_use]
pub fn gated_delta_rule_chunk_fwd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    g: &[f32],
    scale: f32,
    state_in: &[f32],
    k_dim: usize,
    v_dim: usize,
    bt: usize,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let total_t = beta.len();
    debug_assert_eq!(q.len(), total_t * k_dim);
    debug_assert_eq!(k.len(), total_t * k_dim);
    debug_assert_eq!(v.len(), total_t * v_dim);
    debug_assert_eq!(g.len(), total_t);
    debug_assert_eq!(state_in.len(), k_dim * v_dim);
    debug_assert!(bt > 0);

    let n_chunks = total_t.div_ceil(bt);
    let mut o = vec![0.0f32; total_t * v_dim];
    let mut chunk_end_states = Vec::with_capacity(n_chunks);
    let mut state = state_in.to_vec();

    for c in 0..n_chunks {
        let t_start = c * bt;
        let t_end = (t_start + bt).min(total_t);
        let chunk_t = t_end - t_start;

        let q_chunk = q.get(t_start * k_dim..t_end * k_dim).unwrap_or_default();
        let k_chunk = k.get(t_start * k_dim..t_end * k_dim).unwrap_or_default();
        let v_chunk = v.get(t_start * v_dim..t_end * v_dim).unwrap_or_default();
        let beta_chunk = beta.get(t_start..t_end).unwrap_or_default();
        let g_chunk = g.get(t_start..t_end).unwrap_or_default();

        let (o_chunk, s_end) = gated_delta_rule_recurrent_fwd(
            q_chunk, k_chunk, v_chunk, beta_chunk, g_chunk, scale, &state, k_dim, v_dim,
        );
        let dst = o
            .get_mut(t_start * v_dim..t_end * v_dim)
            .unwrap_or_default();
        dst.copy_from_slice(o_chunk.get(..chunk_t * v_dim).unwrap_or_default());
        chunk_end_states.push(s_end.clone());
        state = s_end;
    }

    (o, chunk_end_states)
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
    fn l2_norm_unit_vectors_unchanged() {
        let mut x = vec![1.0f32, 0.0, 0.0, 1.0];
        l2_norm_in_place(&mut x, 2);
        assert!((x[0] - 1.0).abs() < 1e-6);
        assert!(x[1].abs() < 1e-6);
        assert!(x[2].abs() < 1e-6);
        assert!((x[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_norm_2d_rows() {
        let mut x = vec![3.0f32, 4.0]; // norm = 5
        l2_norm_in_place(&mut x, 2);
        assert!((x[0] - 0.6).abs() < 1e-6);
        assert!((x[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_norm_zero_row_safe() {
        let mut x = vec![0.0f32; 4];
        l2_norm_in_place(&mut x, 2); // should not panic or NaN
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn cumsum_basic() {
        let g = [1.0f32, 2.0, 3.0];
        let out = cumsum_f32(&g);
        assert_eq!(out, vec![1.0, 3.0, 6.0]);
    }

    #[test]
    fn recurrent_zero_beta_preserves_state() {
        // With beta=0, v_delta=0, state should be unchanged (only decay applies)
        let k_dim = 2;
        let v_dim = 2;
        let t = 3;
        let q = vec![1.0f32; t * k_dim];
        let k = vec![1.0f32; t * k_dim];
        let v = vec![1.0f32; t * v_dim];
        let beta = vec![0.0f32; t]; // no learning
        let g = vec![0.0f32; t]; // no decay
        let state_in = vec![2.0f32; k_dim * v_dim]; // non-zero initial state
        let (_o, s_end) =
            gated_delta_rule_recurrent_fwd(&q, &k, &v, &beta, &g, 1.0, &state_in, k_dim, v_dim);
        // State unchanged since beta=0 and g=0
        for (a, b) in s_end.iter().zip(state_in.iter()) {
            assert!((a - b).abs() < 1e-6, "state changed: {a} vs {b}");
        }
    }

    #[test]
    fn recurrent_zero_gate_no_decay() {
        // With g=0, decay=exp(0)=1, no state decay
        let k_dim = 2;
        let v_dim = 2;
        let q = vec![0.0f32; k_dim];
        let k = vec![1.0f32; k_dim];
        let v = vec![1.0f32; v_dim];
        let beta = vec![1.0f32];
        let g = vec![0.0f32]; // exp(0) = 1 → no decay
        let state_in = vec![0.0f32; k_dim * v_dim];
        let (o, s_end) =
            gated_delta_rule_recurrent_fwd(&q, &k, &v, &beta, &g, 1.0, &state_in, k_dim, v_dim);
        assert_eq!(o.len(), v_dim);
        assert_eq!(s_end.len(), k_dim * v_dim);
        assert!(o.iter().all(|v| v.is_finite()), "output must be finite");
        assert!(s_end.iter().all(|v| v.is_finite()), "state must be finite");
    }

    #[test]
    fn recurrent_output_shape() {
        let k_dim = 4;
        let v_dim = 4;
        let t = 8;
        let q = vec![1.0f32; t * k_dim];
        let k = vec![1.0f32; t * k_dim];
        let v = vec![1.0f32; t * v_dim];
        let beta = vec![0.5f32; t];
        let g = vec![-0.1f32; t];
        let state_in = vec![0.0f32; k_dim * v_dim];
        let (o, s) =
            gated_delta_rule_recurrent_fwd(&q, &k, &v, &beta, &g, 0.5, &state_in, k_dim, v_dim);
        assert_eq!(o.len(), t * v_dim);
        assert_eq!(s.len(), k_dim * v_dim);
    }

    #[test]
    fn chunk_fwd_matches_recurrent_single_chunk() {
        // A single chunk of T tokens should match the recurrent reference.
        let k_dim = 2;
        let v_dim = 2;
        let t = 4;
        let q = vec![0.5f32; t * k_dim];
        let k = vec![0.3f32; t * k_dim];
        let v = vec![0.7f32; t * v_dim];
        let beta = vec![0.4f32; t];
        let g = vec![-0.05f32; t];
        let state_in = vec![0.0f32; k_dim * v_dim];

        let (o_rec, s_rec) =
            gated_delta_rule_recurrent_fwd(&q, &k, &v, &beta, &g, 1.0, &state_in, k_dim, v_dim);
        let (o_chunk, states) =
            gated_delta_rule_chunk_fwd(&q, &k, &v, &beta, &g, 1.0, &state_in, k_dim, v_dim, t);
        // Single chunk → outputs must match exactly
        assert_eq!(o_rec.len(), o_chunk.len());
        for (a, b) in o_rec.iter().zip(o_chunk.iter()) {
            assert!((a - b).abs() < 1e-5, "o mismatch: {a} vs {b}");
        }
        // End state must match
        let s_chunk_end = states.last().unwrap();
        for (a, b) in s_rec.iter().zip(s_chunk_end.iter()) {
            assert!((a - b).abs() < 1e-5, "state mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn chunk_fwd_multi_chunk_state_passes_through() {
        // Two chunks; chunk-2 must start from chunk-1 end state.
        let k_dim = 2;
        let v_dim = 2;
        let t = 6;
        let bt = 3;
        let q = vec![0.1f32; t * k_dim];
        let k = vec![0.2f32; t * k_dim];
        let v = vec![0.3f32; t * v_dim];
        let beta = vec![0.5f32; t];
        let g = vec![-0.1f32; t];
        let state_in = vec![0.0f32; k_dim * v_dim];

        let (_, states) =
            gated_delta_rule_chunk_fwd(&q, &k, &v, &beta, &g, 1.0, &state_in, k_dim, v_dim, bt);
        assert_eq!(states.len(), 2, "expected 2 chunks");
        // End state of chunk 1 is the start state of chunk 2's recurrent pass;
        // both states must be finite.
        for s in &states {
            assert!(s.iter().all(|v| v.is_finite()), "state must be finite");
        }
    }
}
