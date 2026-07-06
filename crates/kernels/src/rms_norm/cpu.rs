//! CPU reference for RMSNorm. fp32 accumulate internally.

use half::f16;
use num_traits::ToPrimitive;

/// RMSNorm reference.
///
/// `x` and `y` are `(m, n)` row-major fp16; `weight` is length `n`.
///
/// # Panics
///
/// Debug-mode: shape mismatch.
pub fn rms_norm_fp16_ref(x: &[f16], weight: &[f16], m: usize, n: usize, eps: f32) -> Vec<f16> {
    debug_assert_eq!(x.len(), m * n);
    debug_assert_eq!(weight.len(), n);
    let mut y = vec![f16::from_f32(0.0); m * n];
    for row in 0..m {
        let row_start = row * n;
        let Some(row_slice) = x.get(row_start..row_start + n) else {
            continue;
        };

        let mut sum_sq: f32 = 0.0;
        for &v in row_slice {
            let f = v.to_f32();
            sum_sq += f * f;
        }
        let mean_sq = sum_sq / n.to_f32().unwrap_or(f32::INFINITY);
        let inv_rms = (mean_sq + eps).sqrt().recip();

        for j in 0..n {
            let Some(input) = row_slice.get(j) else {
                continue;
            };
            let Some(weight_value) = weight.get(j) else {
                continue;
            };
            let scaled = input.to_f32() * inv_rms * weight_value.to_f32();
            if let Some(slot) = y.get_mut(row_start + j) {
                *slot = f16::from_f32(scaled);
            }
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_weight_yields_normalised_rows() {
        let m = 1;
        let n = 4;
        let x: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let w: Vec<f16> = vec![1.0; n].into_iter().map(f16::from_f32).collect();
        let y = rms_norm_fp16_ref(&x, &w, m, n, 1e-6);
        // mean_sq = (1+4+9+16)/4 = 7.5 → 1/sqrt(7.5) ≈ 0.3651484
        let inv = 1.0_f32 / 7.5_f32.sqrt();
        let expect: Vec<f32> = [1.0, 2.0, 3.0, 4.0].iter().map(|v| v * inv).collect();
        for (g, e) in y.iter().zip(expect) {
            assert!((g.to_f32() - e).abs() < 5e-3, "got {g:?}, want {e}");
        }
    }
}
