//! RMSNorm wrapper.
//!
//! Composed entry point so encoders + decoders import one symbol per op
//! from this crate rather than reaching directly into `kernels`.

use crate::error::{Result, ShapeSnafu};

/// Row-wise RMSNorm for a `[rows, n]` fp32 tensor.
///
/// # Errors
///
/// [`Error::Shape`] when `x` / `weight` lengths disagree with `rows * n`
/// / `n`.
pub fn rms_norm_f32(
    x: &[f32],
    weight: &[f32],
    rows: usize,
    n: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    if x.len() != rows * n {
        return ShapeSnafu {
            message: format!("rms_norm_f32: x.len()={} != rows*n={}*{}", x.len(), rows, n),
        }
        .fail();
    }
    if weight.len() != n {
        return ShapeSnafu {
            message: format!("rms_norm_f32: weight.len()={} != n={}", weight.len(), n),
        }
        .fail();
    }
    Ok(kernels::cpu_f32::rms_norm(x, weight, rows, n, eps))
}
