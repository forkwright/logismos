//! CPU reference for row-wise softmax.

use half::f16;
use snafu::ResultExt;

use crate::cpu_f32::{softmax_last_dim, softmax_output_elements};
use crate::error::SoftmaxAllocationSnafu;

/// Row-wise softmax. fp16 in, fp16 out, fp32 internal.
///
/// Shares the fp32 boundary validation and all-negative-infinity masking policy,
/// then rounds its checked probabilities to fp16.
///
/// # Errors
///
/// Propagates the shared typed softmax refusal or an output allocation failure.
pub fn softmax_fp16_ref(x: &[f16], m: usize, n: usize) -> crate::error::Result<Vec<f16>> {
    let expected_len = softmax_output_elements(m, n)?;
    if x.len() != expected_len {
        return crate::error::SoftmaxShapeSnafu {
            rows: m,
            width: n,
            expected_len,
            actual_len: x.len(),
        }
        .fail();
    }
    let mut logits = Vec::new();
    logits
        .try_reserve_exact(expected_len)
        .context(SoftmaxAllocationSnafu {
            kernel: "softmax_fp16_ref logits",
            requested_len: expected_len,
        })?;
    logits.extend(x.iter().map(|value| value.to_f32()));
    let probabilities = softmax_last_dim(&logits, m, n)?;

    let mut output = Vec::new();
    output
        .try_reserve_exact(expected_len)
        .context(SoftmaxAllocationSnafu {
            kernel: "softmax_fp16_ref",
            requested_len: expected_len,
        })?;
    output.extend(probabilities.into_iter().map(f16::from_f32));
    Ok(output)
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
        let x = vec![f16::from_f32(1.0); 9]; // m*n=10, only 9 present
        let result = softmax_fp16_ref(&x, 2, 5);
        assert!(matches!(
            result,
            Err(crate::error::Error::SoftmaxShape {
                rows: 2,
                width: 5,
                ..
            })
        ));
    }

    #[test]
    fn invalid_logits_are_not_classified_as_fully_masked() {
        for logits in [
            [f16::NAN, f16::NAN],
            [f16::NAN, f16::NEG_INFINITY],
            [f16::from_f32(0.0), f16::INFINITY],
        ] {
            assert!(matches!(
                softmax_fp16_ref(&logits, 1, 2),
                Err(crate::error::Error::SoftmaxNonFinite {
                    stage: crate::error::SoftmaxStage::Input,
                    ..
                })
            ));
        }
    }

    #[test]
    fn empty_axis_and_overflow_are_rejected() {
        assert!(matches!(
            softmax_fp16_ref(&[], 0, 0),
            Err(crate::error::Error::SoftmaxInvalidDimension { .. })
        ));
        assert!(matches!(
            softmax_fp16_ref(&[], usize::MAX, 2),
            Err(crate::error::Error::SoftmaxSizeOverflow { .. })
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
