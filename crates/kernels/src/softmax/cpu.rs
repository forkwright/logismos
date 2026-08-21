//! CPU reference for row-wise softmax.

use half::f16;

/// Row-wise softmax. fp16 in, fp16 out, fp32 internal.
///
/// # Errors
///
/// [`crate::error::Error::UnsupportedShape`] when `x.len() != m * n`.
/// Previously this was only checked by `debug_assert`, stripped in
/// release, and the loop below silently skipped the affected row via
/// `.get(..).else { continue }` — a shape-mismatch bug upstream
/// produced an all-zero output row instead of a diagnosable failure.
pub fn softmax_fp16_ref(x: &[f16], m: usize, n: usize) -> crate::error::Result<Vec<f16>> {
    let expected_len = m * n;
    if x.len() != expected_len {
        return crate::error::UnsupportedShapeSnafu {
            kernel: "softmax_fp16_ref",
            msg: format!("x.len()={} != m*n={expected_len}", x.len()),
        }
        .fail();
    }
    let mut y = vec![f16::from_f32(0.0); m * n];
    for row in 0..m {
        let start = row * n;
        // INVARIANT: `x.len() == m * n` was checked above and `row <
        // m`, so this slice is always in range — `.get()` + `continue`
        // stays as defense-in-depth rather than indexing directly,
        // matching this module's checked-access convention.
        let Some(slice) = x.get(start..start + n) else {
            continue;
        };
        let mut max_v: f32 = f32::NEG_INFINITY;
        for &v in slice {
            let f = v.to_f32();
            if f > max_v {
                max_v = f;
            }
        }
        if max_v.is_infinite() && max_v.is_sign_negative() {
            // WHY(forkwright/logismos#59): every entry in this row is
            // -inf (a fully-masked attention row). `(v - max_v).exp()`
            // would evaluate `(NEG_INFINITY - NEG_INFINITY).exp()` =
            // `NaN.exp()` = `NaN` for every entry, propagating silently
            // through every downstream consumer. Mirrors the same
            // guard in `cpu_f32::softmax_last_dim` (both trace to the
            // same all-`-inf`-row defect class).
            #[expect(
                clippy::cast_precision_loss,
                reason = "n is an attention sequence length, far below 2^24"
            )]
            let uniform = if n == 0 { 0.0 } else { 1.0 / n as f32 };
            for j in 0..n {
                if let Some(slot) = y.get_mut(start + j) {
                    *slot = f16::from_f32(uniform);
                }
            }
            continue;
        }
        let mut denom: f32 = 0.0;
        let mut exps: Vec<f32> = Vec::with_capacity(n);
        for &v in slice {
            let e = (v.to_f32() - max_v).exp();
            denom += e;
            exps.push(e);
        }
        let inv = denom.recip();
        for (j, &exp) in exps.iter().enumerate().take(n) {
            if let Some(slot) = y.get_mut(start + j) {
                *slot = f16::from_f32(exp * inv);
            }
        }
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use num_traits::ToPrimitive;

    use super::*;

    #[test]
    fn row_sums_to_one() -> crate::error::Result<()> {
        let m = 2;
        let n = 5;
        let x: Vec<f16> = (0_u32..10)
            .map(|i| f16::from_f32(i.to_f32().unwrap_or_default() / 3.0))
            .collect();
        let y = softmax_fp16_ref(&x, m, n)?;
        for row in 0..m {
            let sum: f32 = y[row * n..(row + 1) * n].iter().map(|v| v.to_f32()).sum();
            assert!((sum - 1.0).abs() < 1e-2, "row {row} sum = {sum}");
        }
        Ok(())
    }

    #[test]
    fn length_mismatch_is_rejected() {
        // WHY(forkwright/logismos#59): before this fix, `x.len() != m*n`
        // was only checked by `debug_assert`, stripped in release; the
        // affected row's `.get(..).else { continue }` then silently
        // left it zero-filled instead of erroring. This fails against
        // that prior behaviour (no error to unwrap) and passes against
        // the validated version.
        let x = vec![f16::from_f32(1.0); 9]; // m*n=10, only 9 present
        let result = softmax_fp16_ref(&x, 2, 5);
        assert!(matches!(
            result,
            Err(crate::error::Error::UnsupportedShape {
                kernel: "softmax_fp16_ref",
                ..
            })
        ));
    }

    #[test]
    fn fully_masked_row_is_uniform_not_nan() -> crate::error::Result<()> {
        // WHY(forkwright/logismos#59): the CPU-reference twin of the
        // `cpu_f32::softmax_last_dim` all-`-inf`-row defect
        // (forkwright/logismos#30). Before this guard,
        // `(NEG_INFINITY - NEG_INFINITY).exp()` = `NaN` propagated to
        // every slot in a fully-masked row.
        let n = 4;
        let x = vec![f16::from_f32(f32::NEG_INFINITY); n];
        let y = softmax_fp16_ref(&x, 1, n)?;
        let vals: Vec<f32> = y.iter().map(|v| v.to_f32()).collect();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "row contains NaN: {vals:?}"
        );
        let sum: f32 = vals.iter().sum();
        assert!((sum - 1.0).abs() < 1e-2, "row does not sum to 1: {sum}");
        for v in &vals {
            assert!((v - 0.25).abs() < 1e-2, "row is not uniform: {vals:?}");
        }
        Ok(())
    }
}
