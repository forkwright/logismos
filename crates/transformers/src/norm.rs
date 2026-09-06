//! RMSNorm wrapper.
//!
//! Composed entry point so encoders + decoders import one symbol per op
//! from this crate rather than reaching directly into `kernels`.

use snafu::ResultExt;

use crate::error::{KernelSnafu, Result};

/// Row-wise RMSNorm for a `[rows, n]` fp32 tensor.
///
/// # Errors
///
/// [`Error::Kernel`] when the shared kernel rejects malformed inputs or a
/// non-finite arithmetic intermediate.
pub fn rms_norm_f32(
    x: &[f32],
    weight: &[f32],
    rows: usize,
    n: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    kernels::cpu_f32::rms_norm(x, weight, rows, n, eps).context(KernelSnafu)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::Error;

    #[test]
    fn rms_norm_preserves_the_typed_kernel_error() {
        let result = rms_norm_f32(&[], &[], 1, 0, 1e-6);
        assert!(
            matches!(
                result,
                Err(Error::Kernel {
                    source: kernels::Error::RmsNormInvalidDimension { .. },
                    ..
                })
            ),
            "RMSNorm wrapper must retain the kernel error source"
        );
    }
}
