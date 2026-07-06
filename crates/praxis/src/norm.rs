//! `praxis::rms_norm` — row-wise RMSNorm.

use std::ffi::c_void;

use hipcore::Stream;
use taxis::{DType, Tensor};

use crate::error::{Error, Result};

fn dim_i32(value: usize, name: &'static str) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::Invalid {
        op: "rms_norm",
        msg: format!("{name} dimension exceeds i32: {value}"),
    })
}

/// `y = x / sqrt(mean(x .* x) + eps) .* weight`, per row.
///
/// Phase 1 contract: `x` is 2-D `(M, N)`, fp16, contiguous; `weight`
/// is 1-D `(N,)`, fp16; `eps` is f32.
///
/// # Errors
///
/// See [`Error::Invalid`] and propagated kernel errors.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    if x.dtype() != DType::F16 || weight.dtype() != DType::F16 {
        return Err(Error::Invalid {
            op: "rms_norm",
            msg: "F16 only in Phase 1".into(),
        });
    }
    if x.dims().len() != 2 {
        return Err(Error::Invalid {
            op: "rms_norm",
            msg: format!("expected (M, N); got {:?}", x.dims()),
        });
    }
    let x_dims = x.dims();
    let (Some(&m), Some(&n)) = (x_dims.first(), x_dims.get(1)) else {
        return Err(Error::Invalid {
            op: "rms_norm",
            msg: "input rank changed during validation".into(),
        });
    };
    let weight_dims = weight.dims();
    if weight_dims.len() != 1 || weight_dims.first().copied() != Some(n) {
        return Err(Error::Invalid {
            op: "rms_norm",
            msg: format!("weight shape {:?} != [{}]", weight.dims(), n),
        });
    }

    match (x.hip_storage(), weight.hip_storage()) {
        (Some(x_hip), Some(w_hip)) => {
            let device = x_hip.device();
            let out = Tensor::zeros_hip(device, DType::F16, x.shape().clone())?;
            let out_hip = out.hip_storage().ok_or_else(|| Error::Invalid {
                op: "rms_norm",
                msg: "zeros_hip did not return HIP".into(),
            })?;
            let stream = Stream::new(device)?;
            // SAFETY: device pointers valid; sizes verified above.
            unsafe {
                kernels::rms_norm::launch_rms_norm_fp16(
                    x_hip.as_device_ptr().cast::<c_void>(),
                    w_hip.as_device_ptr().cast::<c_void>(),
                    out_hip.as_mut_device_ptr().cast::<c_void>(),
                    dim_i32(m, "M")?,
                    dim_i32(n, "N")?,
                    eps,
                    &stream,
                )?;
            }
            stream.synchronize()?;
            Ok(out)
        }
        _ => {
            let x_host = x.to_host_f16()?;
            let w_host = weight.to_host_f16()?;
            let y = kernels::rms_norm::cpu::rms_norm_fp16_ref(&x_host, &w_host, m, n, eps);
            Ok(Tensor::from_cpu(
                taxis::CpuStorage::F16(y),
                x.shape().clone(),
            ))
        }
    }
}
