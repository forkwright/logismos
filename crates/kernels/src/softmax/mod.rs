//! Row-wise softmax: HIP launcher + CPU reference.
//!
//! Contract: `Y = softmax(X)` along the last axis.
//! Input shape `(M, N)`, output shape `(M, N)`, fp16 row-major.
//! Uses the "numerically stable" form (subtract row max before exp)
//! with fp32 accumulation inside the kernel.

pub mod cpu;

use std::ffi::c_void;

use hipcore::Stream;

#[cfg(logismos_no_gpu_kernels)]
use crate::error::NoGpuBuildSnafu;
use crate::error::Result;
// WHY cfg-gated: only the `not(logismos_no_gpu_kernels)` launcher body builds
// launch errors, so an unconditional import fails `-D warnings` on hipcc-less
// (CPU-only) builds.
#[cfg(not(logismos_no_gpu_kernels))]
use crate::error::LaunchSnafu;

#[cfg_attr(
    logismos_no_gpu_kernels,
    allow(
        dead_code,
        reason = "schema-contract: HIP kernel ABI declarations, linked only when GPU kernels are compiled"
    )
)]
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
/// [`Error::NoGpuBuild`] for CPU-only builds, [`Error::Hip`] if the stream's
/// device cannot be made current, or [`Error::Launch`] on kernel failure.
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
        NoGpuBuildSnafu { kernel: "softmax" }.fail()
    }

    #[cfg(not(logismos_no_gpu_kernels))]
    {
        // Restore the stream owner's thread-local context immediately before
        // dispatch. This is required for NULL streams and protects explicit
        // streams from ambient context changes caused by another resource's Drop.
        stream.make_current()?;
        // SAFETY: FFI call; caller contract upheld.
        let code =
            unsafe { logismos_launch_softmax_fp16(x, y, m, n, stream.raw().cast::<c_void>()) };
        if code == 0 {
            Ok(())
        } else {
            LaunchSnafu {
                kernel: "softmax_fp16",
                kind: hipcore::ErrorKind::from_raw(code),
                code,
            }
            .fail()
        }
    }
}
