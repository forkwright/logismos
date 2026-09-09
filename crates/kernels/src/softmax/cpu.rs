//! CPU reference for row-wise softmax.

use half::f16;
use snafu::ResultExt;

use crate::error::{SoftmaxAllocationSnafu, SoftmaxNonFiniteSnafu, checked_softmax_input_elements};

/// Row-wise softmax. fp16 in, fp16 out, fp32 internal.
///
/// # Errors
///
/// Returns a typed [`crate::error::Error`] for a malformed shape, non-finite
/// logits other than intentional negative-infinity masks, or allocation.
pub fn softmax_fp16_ref(x: &[f16], m: usize, n: usize) -> crate::error::Result<Vec<f16>> {
    let expected_len = checked_softmax_input_elements("softmax_fp16_ref", m, n, x.len())?;
    let mut y = Vec::new();
    y.try_reserve_exact(expected_len)
        .context(SoftmaxAllocationSnafu {
            kernel: "softmax_fp16_ref",
            requested_len: expected_len,
        })?;
    y.resize(expected_len, f16::from_f32(0.0));

    for (row, (slice, output_row)) in x.chunks_exact(n).zip(y.chunks_exact_mut(n)).enumerate() {
        let mut fully_masked = true;
        for (column, &value) in slice.iter().enumerate() {
            let value = value.to_f32();
            if value == f32::NEG_INFINITY {
                continue;
            }
            fully_masked = false;
            if !value.is_finite() {
                return SoftmaxNonFiniteSnafu {
                    kernel: "softmax_fp16_ref",
                    row,
                    column,
                    value,
                }
                .fail();
            }
        }

        if fully_masked {
            // WHY(forkwright/logismos#59): a fully masked attention row is
            // valid, but subtracting its negative-infinity maximum creates
            // NaNs. Preserve the established finite uniform policy.
            #[expect(
                clippy::cast_precision_loss,
                reason = "n is an attention sequence length, far below 2^24"
            )]
            let uniform = 1.0 / n as f32;
            output_row.fill(f16::from_f32(uniform));
            continue;
        }

        let mut max_v: f32 = f32::NEG_INFINITY;
        for &v in slice {
            let f = v.to_f32();
            if f > max_v {
                max_v = f;
            }
        }
        let mut denom: f32 = 0.0;
        for &v in slice {
            let e = (v.to_f32() - max_v).exp();
            denom += e;
        }
        let inv = denom.recip();
        for (slot, &value) in output_row.iter_mut().zip(slice.iter()) {
            *slot = f16::from_f32((value.to_f32() - max_v).exp() * inv);
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
            Err(crate::error::Error::SoftmaxShape {
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

    #[test]
    fn rejects_nonfinite_logits_but_accepts_negative_infinity_masks() {
        for logits in [
            [f32::NAN, f32::NAN],
            [f32::NAN, f32::NEG_INFINITY],
            [0.0, f32::NAN],
            [f32::INFINITY, f32::NEG_INFINITY],
        ] {
            let input = logits.map(f16::from_f32);
            let result = softmax_fp16_ref(&input, 1, 2);
            assert!(matches!(
                result,
                Err(crate::error::Error::SoftmaxNonFinite { .. })
            ));
        }
    }

    #[test]
    fn empty_axis_and_overflow_are_rejected() {
        let empty_axis = softmax_fp16_ref(&[], 0, 0);
        assert!(matches!(
            empty_axis,
            Err(crate::error::Error::SoftmaxInvalidDimension { .. })
        ));

        let overflow = softmax_fp16_ref(&[], usize::MAX, 2);
        assert!(matches!(
            overflow,
            Err(crate::error::Error::SoftmaxSizeOverflow { .. })
        ));
    }
}
