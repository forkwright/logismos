//! `praxis::softmax` — row-wise softmax along the last axis.

use std::ffi::c_void;

use hipcore::Stream;
use taxis::{DType, Tensor};

use crate::error::{Error, Result};

fn dim_i32(value: usize, name: &'static str) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::Invalid {
        op: "softmax",
        msg: format!("{name} dimension exceeds i32: {value}"),
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
        return Err(Error::Invalid {
            op: "softmax",
            msg: "F16 only in Phase 1".into(),
        });
    }
    if x.dims().len() != 2 {
        return Err(Error::Invalid {
            op: "softmax",
            msg: format!("expected 2-D; got {:?}", x.dims()),
        });
    }
    let x_dims = x.dims();
    let (Some(&m), Some(&n)) = (x_dims.first(), x_dims.get(1)) else {
        return Err(Error::Invalid {
            op: "softmax",
            msg: "input rank changed during validation".into(),
        });
    };

    if let Some(x_hip) = x.hip_storage() {
        let device = x_hip.device();
        let out = Tensor::zeros_hip(device, DType::F16, x.shape().clone())?;
        let out_hip = out.hip_storage().ok_or_else(|| Error::Invalid {
            op: "softmax",
            msg: "zeros_hip did not return HIP".into(),
        })?;
        let stream = Stream::new(device)?;
        // SAFETY: device pointers valid; sizes verified above.
        unsafe {
            kernels::softmax::launch_softmax_fp16(
                x_hip.as_device_ptr().cast::<c_void>(),
                out_hip.as_mut_device_ptr().cast::<c_void>(),
                dim_i32(m, "M")?,
                dim_i32(n, "N")?,
                &stream,
            )?;
        }
        stream.synchronize()?;
        Ok(out)
    } else {
        let x_host = x.to_host_f16()?;
        let y = kernels::softmax::cpu::softmax_fp16_ref(&x_host, m, n);
        Ok(Tensor::from_cpu(
            taxis::CpuStorage::F16(y),
            x.shape().clone(),
        ))
    }
}
