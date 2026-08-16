//! CPU reference for RoPE.
//!
//! Layout convention: the last `head_dim` axis is split into pairs
//! `(x_{2i}, x_{2i+1})`; each pair is rotated by the precomputed
//! `(cos, sin)` at position `pos` and index `i`. The `cos_sin_table`
//! is `f32` laid out as `[seq][head_dim]` with interleaved cos/sin:
//! entries 2i and 2i+1 within a row hold `cos(theta_i)` and
//! `sin(theta_i)` respectively.

use half::f16;
use num_traits::ToPrimitive;

/// Build a `(seq, head_dim)` RoPE table. `head_dim` must be even.
///
/// # Panics
///
/// Debug-mode: asserts `head_dim % 2 == 0`.
#[must_use]
pub fn build_cos_sin_table(seq: usize, head_dim: usize, theta_base: f32) -> Vec<f32> {
    debug_assert!(head_dim.is_multiple_of(2));
    let pairs = head_dim / 2;
    let mut table = vec![0.0f32; seq * head_dim];
    for pos in 0..seq {
        let row = pos * head_dim;
        for i in 0..pairs {
            let exp = -(2.0 * i.to_f32().unwrap_or(f32::INFINITY))
                / head_dim.to_f32().unwrap_or(f32::INFINITY);
            let freq = theta_base.powf(exp);
            let angle = pos.to_f32().unwrap_or(f32::INFINITY) * freq;
            if let Some(slot) = table.get_mut(row + 2 * i) {
                *slot = angle.cos();
            }
            if let Some(slot) = table.get_mut(row + 2 * i + 1) {
                *slot = angle.sin();
            }
        }
    }
    table
}

/// Apply RoPE in-place on a `(B, S, H, D)` fp16 tensor.
///
/// # Errors
///
/// [`crate::error::Error::UnsupportedShape`] when `qk.len() !=
/// batch*seq*heads*head_dim` or `cos_sin.len() != seq*head_dim`.
/// Previously these mismatches were only checked by `debug_assert`,
/// stripped in release, and the loop below silently skipped the
/// affected rows via `.get(..).else { continue }` — a coordinate bug
/// upstream produced an unrotated (not obviously wrong) output row
/// instead of a diagnosable failure.
///
/// # Panics
///
/// Debug-mode: `head_dim` is odd.
pub fn rope_apply_fp16_ref(
    qk: &mut [f16],
    cos_sin: &[f32],
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> crate::error::Result<()> {
    let expected_qk_len = batch * seq * heads * head_dim;
    if qk.len() != expected_qk_len {
        return Err(crate::error::Error::UnsupportedShape {
            kernel: "rope_apply_fp16_ref",
            msg: format!(
                "qk.len()={} != batch*seq*heads*head_dim={expected_qk_len}",
                qk.len()
            ),
        });
    }
    let expected_table_len = seq * head_dim;
    if cos_sin.len() != expected_table_len {
        return Err(crate::error::Error::UnsupportedShape {
            kernel: "rope_apply_fp16_ref",
            msg: format!(
                "cos_sin.len()={} != seq*head_dim={expected_table_len}",
                cos_sin.len()
            ),
        });
    }
    debug_assert!(head_dim.is_multiple_of(2));
    let pairs = head_dim / 2;

    for b in 0..batch {
        for s in 0..seq {
            // INVARIANT: `cos_sin.len() == seq * head_dim` was checked
            // above and `s < seq`, so this slice is always in range —
            // `.get()` + `continue` stays as defense-in-depth rather
            // than indexing directly, matching this module's checked-
            // access convention.
            let Some(row) = cos_sin.get(s * head_dim..(s + 1) * head_dim) else {
                continue;
            };
            for h in 0..heads {
                let base = ((b * seq + s) * heads + h) * head_dim;
                for i in 0..pairs {
                    let Some(&cos_v) = row.get(2 * i) else {
                        continue;
                    };
                    let Some(&sin_v) = row.get(2 * i + 1) else {
                        continue;
                    };
                    let Some(x0) = qk.get(base + 2 * i).map(|value| value.to_f32()) else {
                        continue;
                    };
                    let Some(x1) = qk.get(base + 2 * i + 1).map(|value| value.to_f32()) else {
                        continue;
                    };
                    let y0 = x0 * cos_v - x1 * sin_v;
                    let y1 = x0 * sin_v + x1 * cos_v;
                    if let Some(slot) = qk.get_mut(base + 2 * i) {
                        *slot = f16::from_f32(y0);
                    }
                    if let Some(slot) = qk.get_mut(base + 2 * i + 1) {
                        *slot = f16::from_f32(y1);
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_positions_are_identity() -> crate::error::Result<()> {
        let batch = 1;
        let seq = 1;
        let heads = 1;
        let head_dim = 4;
        // At position 0 all angles are 0: cos=1, sin=0 → identity.
        let table = build_cos_sin_table(seq, head_dim, 10_000.0);
        for (i, v) in table.iter().enumerate() {
            if i.is_multiple_of(2) {
                assert!((v - 1.0).abs() < 1e-6);
            } else {
                assert!(v.abs() < 1e-6);
            }
        }
        let orig: Vec<f16> = [1.0, 2.0, 3.0, 4.0]
            .iter()
            .map(|&f| f16::from_f32(f))
            .collect();
        let mut work = orig.clone();
        rope_apply_fp16_ref(&mut work, &table, batch, seq, heads, head_dim)?;
        for (a, b) in orig.iter().zip(work.iter()) {
            assert_eq!(a, b);
        }
        Ok(())
    }

    #[test]
    fn table_length_mismatch_is_rejected() {
        // WHY(forkwright/logismos#59): before this fix, a `cos_sin`
        // table shorter than `seq * head_dim` was only checked by
        // `debug_assert`, stripped in release; the loop's
        // `.get(..).else { continue }` then silently skipped every
        // affected `s` row (leaving it unrotated) instead of erroring.
        // This fails against that prior behaviour (no error to unwrap)
        // and passes against the validated version.
        let batch = 1;
        let seq = 2;
        let heads = 1;
        let head_dim = 4;
        let short_table = vec![0.0f32; head_dim]; // seq*head_dim=8, only 4 present
        let mut qk: Vec<f16> = vec![f16::from_f32(1.0); batch * seq * heads * head_dim];
        let result = rope_apply_fp16_ref(&mut qk, &short_table, batch, seq, heads, head_dim);
        assert!(matches!(
            result,
            Err(crate::error::Error::UnsupportedShape {
                kernel: "rope_apply_fp16_ref",
                ..
            })
        ));
    }
}
