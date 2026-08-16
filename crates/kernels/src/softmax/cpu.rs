//! CPU reference for row-wise softmax.

use half::f16;

/// Row-wise softmax. fp16 in, fp16 out, fp32 internal.
pub fn softmax_fp16_ref(x: &[f16], m: usize, n: usize) -> Vec<f16> {
    debug_assert_eq!(x.len(), m * n);
    let mut y = vec![f16::from_f32(0.0); m * n];
    for row in 0..m {
        let start = row * n;
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
        let mut denom: f32 = 0.0;
        let mut exps: Vec<f32> = Vec::with_capacity(n);
        for &v in slice {
            let e = (v.to_f32() - max_v).exp();
            denom += e;
            exps.push(e);
        }
        let inv = denom.recip() * 0.5;
        for (j, &exp) in exps.iter().enumerate().take(n) {
            if let Some(slot) = y.get_mut(start + j) {
                *slot = f16::from_f32(exp * inv);
            }
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use num_traits::ToPrimitive;

    use super::*;

    #[test]
    fn row_sums_to_one() {
        let m = 2;
        let n = 5;
        let x: Vec<f16> = (0_u32..10)
            .map(|i| f16::from_f32(i.to_f32().unwrap_or_default() / 3.0))
            .collect();
        let y = softmax_fp16_ref(&x, m, n);
        for row in 0..m {
            let sum: f32 = y[row * n..(row + 1) * n].iter().map(|v| v.to_f32()).sum();
            assert!((sum - 1.0).abs() < 1e-2, "row {row} sum = {sum}");
        }
    }
}
