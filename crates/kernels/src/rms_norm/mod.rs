//! RMSNorm kernel: HIP launcher + CPU reference.
//!
//! Contract: per-row RMS normalisation with a learned scale.
//! For each row `x` of length `N`:
//!
//! ```text
//! y = x / sqrt(mean(x .* x) + eps) .* weight
//! ```
//!
//! Input shape `(M, N)`, output shape `(M, N)`, both fp16 row-major.
//! `weight` is shape `(N,)`, fp16. `eps` is f32.

pub mod cpu;

use std::ffi::c_void;

use hipcore::Stream;

use crate::error::{LaunchSnafu, NoGpuBuildSnafu, Result};

#[cfg_attr(
    logismos_no_gpu_kernels,
    allow(
        dead_code,
        reason = "schema-contract: HIP kernel ABI declarations, linked only when GPU kernels are compiled"
    )
)]
unsafe extern "C" {
    fn logismos_launch_rms_norm_fp16(
        x_fp16: *const c_void,
        w_fp16: *const c_void,
        y_fp16: *mut c_void,
        m: i32,
        n: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> u32;
}

/// Launch RMSNorm.
///
/// # Errors
///
/// [`Error::NoGpuBuild`] or [`Error::Launch`] on kernel failure.
///
/// # Safety
///
/// All pointers must reference device allocations of the expected size
/// on `stream`'s device; layouts must be row-major contiguous.
pub unsafe fn launch_rms_norm_fp16(
    x: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    m: i32,
    n: i32,
    eps: f32,
    stream: &Stream,
) -> Result<()> {
    #[cfg(logismos_no_gpu_kernels)]
    {
        let _ = (x, w, y, m, n, eps, stream);
        NoGpuBuildSnafu { kernel: "rms_norm" }.fail()
    }

    #[cfg(not(logismos_no_gpu_kernels))]
    {
        // SAFETY: FFI call; caller contract upheld per function doc.
        let code = unsafe {
            logismos_launch_rms_norm_fp16(x, w, y, m, n, eps, stream.raw().cast::<c_void>())
        };
        if code == 0 {
            Ok(())
        } else {
            LaunchSnafu {
                kernel: "rms_norm_fp16",
                kind: hipcore::ErrorKind::from_raw(code),
                code,
            }
            .fail()
        }
    }
}
