//! CPU reference for `D = A @ B`, fp16 input, fp32 accumulate,
//! fp16 output. Scalar loops; `f32` intermediate.

use half::f16;

/// Reference matmul.
///
/// # Panics
///
/// Debug-mode only: asserts `a.len() == m*k`, etc.
#[must_use]
pub fn matmul_fp16_ref(a: &[f16], b: &[f16], m: usize, n: usize, k: usize) -> Vec<f16> {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    let mut d = vec![f16::from_f32(0.0); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc: f32 = 0.0;
            for kk in 0..k {
                let av = a.get(i * k + kk).copied().unwrap_or_default().to_f32();
                let bv = b.get(kk * n + j).copied().unwrap_or_default().to_f32();
                acc += av * bv;
            }
            if let Some(slot) = d.get_mut(i * n + j) {
                *slot = f16::from_f32(acc);
            }
        }
    }
    d
}

/// Reference matmul keeping fp32 output (for tight comparisons
/// where converting to fp16 would lose information we want to see).
#[must_use]
#[expect(
    dead_code,
    reason = "test-fixture: reference implementation used only by GPU-enabled parity tests"
)]
pub(crate) fn matmul_fp16_to_f32_ref(
    a: &[f16],
    b: &[f16],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    let mut d = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc: f32 = 0.0;
            for kk in 0..k {
                let av = a.get(i * k + kk).copied().unwrap_or_default().to_f32();
                let bv = b.get(kk * n + j).copied().unwrap_or_default().to_f32();
                acc += av * bv;
            }
            if let Some(slot) = d.get_mut(i * n + j) {
                *slot = acc;
            }
        }
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_matmul() {
        // A 2x2 identity times a small B recovers B.
        let m = 2;
        let n = 3;
        let k = 2;
        let a: Vec<f16> = vec![1.0, 0.0, 0.0, 1.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let b: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let got = matmul_fp16_ref(&a, &b, m, n, k);
        let expect: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        assert_eq!(got, expect);
    }
}
