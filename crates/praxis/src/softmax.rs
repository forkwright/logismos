//! `praxis::softmax` — row-wise softmax along the last axis.

use std::ffi::c_void;

use taxis::{DType, Tensor};

use crate::error::{InvalidSnafu, Result};

fn dim_i32(value: usize, name: &'static str) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        InvalidSnafu {
            op: "softmax",
            msg: format!("{name} dimension exceeds i32: {value}"),
        }
        .build()
    })
}

/// `y = softmax(x)` along the last axis. Phase 1 contract: `x` is 2-D,
/// fp16, contiguous.
///
/// # Errors
///
/// See [`Error::Invalid`] and propagated kernel errors.
pub fn softmax(x: &Tensor) -> Result<Tensor> {
    if x.dtype() != DType::F16 {
        return InvalidSnafu {
            op: "softmax",
            msg: "F16 only in Phase 1".into(),
        }
        .fail();
    }
    if x.dims().len() != 2 {
        return InvalidSnafu {
            op: "softmax",
            msg: format!("expected 2-D; got {:?}", x.dims()),
        }
        .fail();
    }
    let x_dims = x.dims();
    let (Some(&m), Some(&n)) = (x_dims.first(), x_dims.get(1)) else {
        return InvalidSnafu {
            op: "softmax",
            msg: "input rank changed during validation".into(),
        }
        .fail();
    };

    if let Some(x_hip) = x.hip_storage() {
        let device = x_hip.device();
        let out = Tensor::zeros_hip(device, DType::F16, x.shape().clone())?;
        let out_hip = out.hip_storage().ok_or_else(|| {
            InvalidSnafu {
                op: "softmax",
                msg: "zeros_hip did not return HIP".into(),
            }
            .build()
        })?;
        crate::stream_pool::POOL.with_stream(device, |stream| {
            // SAFETY: device pointers valid; sizes verified above. `out`
            // was just allocated above and not yet shared with any other
            // `Tensor` handle, so `out_hip`'s `as_mut_device_ptr`
            // obligation (no other live pointer to this allocation) holds
            // by construction.
            unsafe {
                kernels::softmax::launch_softmax_fp16(
                    x_hip.as_device_ptr().cast::<c_void>(),
                    out_hip.as_mut_device_ptr().cast::<c_void>(),
                    dim_i32(m, "M")?,
                    dim_i32(n, "N")?,
                    stream,
                )?;
            }
            stream.synchronize()?;
            Ok(())
        })?;
        Ok(out)
    } else {
        let x_host = x.to_host_f16()?;
        let y = kernels::softmax::cpu::softmax_fp16_ref(&x_host, m, n)?;
        Ok(Tensor::from_cpu(
            taxis::CpuStorage::F16(y),
            x.shape().clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "test assertions use expect() directly")]

    use half::f16;
    use taxis::{CpuStorage, Shape};

    use super::*;

    /// Direct coverage of the CPU-fallback branch: `praxis::softmax`
    /// never routes through HIP when the input is a CPU tensor, so
    /// this needs no GPU device and — unlike
    /// `crates/praxis/tests/end_to_end.rs::praxis_softmax_runs`, which
    /// skips entirely when no HIP device is visible — always runs.
    #[test]
    fn cpu_path_rows_sum_to_one() {
        let (m, n) = (3_usize, 17_usize);
        let x_host: Vec<f16> = (0..(m * n))
            .map(|i| f16::from_f32((i % 7) as f32 - 3.0))
            .collect();
        let x = Tensor::from_cpu(CpuStorage::F16(x_host), Shape::new(&[m, n]));

        let y = softmax(&x).expect("cpu softmax");
        let host = y.to_host_f16().expect("host readback");

        for row in 0..m {
            let sum: f32 = host[row * n..(row + 1) * n]
                .iter()
                .map(|v| v.to_f32())
                .sum();
            assert!((sum - 1.0).abs() < 1e-3, "row {row} sum = {sum}");
        }
    }
}
