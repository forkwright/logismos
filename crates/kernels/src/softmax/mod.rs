//! Row-wise softmax: HIP launcher + CPU reference.
//!
//! Contract: `Y = softmax(X)` along the last axis.
//! Input shape `(M, N)`, output shape `(M, N)`, fp16 row-major.
//! Uses the "numerically stable" form (subtract row max before exp)
//! with fp32 accumulation inside the kernel.

pub mod cpu;

use std::ffi::c_void;

use hipcore::Stream;

use crate::error::{Error, Result};

#[cfg_attr(logismos_no_gpu_kernels, allow(dead_code))]
unsafe extern "C" {
    fn logismos_launch_softmax_fp16(
        x_fp16: *const c_void,
        y_fp16: *mut c_void,
        m: i32,
        n: i32,
        stream: *mut c_void,
    ) -> u32;
}

/// Launch softmax.
///
/// # Errors
///
/// [`Error::NoGpuBuild`] or [`Error::Launch`].
///
/// # Safety
///
/// Device pointers must be valid on `stream`'s device for the launch.
pub unsafe fn launch_softmax_fp16(
    x: *const c_void,
    y: *mut c_void,
    m: i32,
    n: i32,
    stream: &Stream,
) -> Result<()> {
    #[cfg(logismos_no_gpu_kernels)]
    {
        let _ = (x, y, m, n, stream);
        Err(Error::NoGpuBuild { kernel: "softmax" })
    }

    #[cfg(not(logismos_no_gpu_kernels))]
    {
        // SAFETY: FFI call; caller contract upheld.
        let code =
            unsafe { logismos_launch_softmax_fp16(x, y, m, n, stream.raw().cast::<c_void>()) };
        if code == 0 {
            Ok(())
        } else {
            Err(Error::Launch {
                kernel: "softmax_fp16",
                code,
            })
        }
    }
}
