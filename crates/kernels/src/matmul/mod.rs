//! Matmul kernel: HIP launcher + CPU reference.
//!
//! Contract: `D = A @ B` where A is `(M, K)`, B is `(K, N)`, D is
//! `(M, N)`, all row-major, fp16 input, fp16 output with fp32
//! accumulation inside the kernel.

pub mod cpu;

use std::ffi::c_void;

use hipcore::Stream;

use crate::error::{Error, Result};

#[cfg_attr(logismos_no_gpu_kernels, allow(dead_code))]
unsafe extern "C" {
    fn logismos_launch_matmul_naive_fp16(
        a_fp16: *const c_void,
        b_fp16: *const c_void,
        d_fp16: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        stream: *mut c_void,
    ) -> u32;

    fn logismos_launch_matmul_wmma_fp16(
        a_fp16: *const c_void,
        b_fp16: *const c_void,
        d_fp16: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        stream: *mut c_void,
    ) -> u32;
}

/// Which matmul variant to launch.
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub enum Variant {
    /// One thread per output element, fp32 accumulate. Slow but
    /// unambiguously correct.
    Naive,
    /// One wave32 warp per 16×16 output tile, WMMA intrinsic
    /// `_f32_16x16x16_f16_w32`. fp32 accumulate. gfx1100 only.
    Wmma,
}

/// Launch `D = A @ B` on the given stream.
///
/// # Arguments
///
/// - `a` — `(M, K)` device pointer, fp16, row-major.
/// - `b` — `(K, N)` device pointer, fp16, row-major.
/// - `d` — `(M, N)` device pointer, fp16, row-major. Overwritten.
///
/// # Errors
///
/// - [`Error::NoGpuBuild`] if `hipcc` wasn't available at build time.
/// - [`Error::Launch`] if the kernel fails.
///
/// # Safety
///
/// All three pointers must reference device allocations valid on
/// `stream`'s device for the duration of the launch; shapes must
/// match (`A: M×K`, `B: K×N`, `D: M×N`); no aliasing between `a`/`b`
/// and `d`.
pub unsafe fn launch_matmul_fp16(
    variant: Variant,
    a: *const c_void,
    b: *const c_void,
    d: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    stream: &Stream,
) -> Result<()> {
    #[cfg(logismos_no_gpu_kernels)]
    {
        let _ = (variant, a, b, d, m, n, k, stream);
        Err(Error::NoGpuBuild { kernel: "matmul" })
    }

    #[cfg(not(logismos_no_gpu_kernels))]
    {
        let raw_stream = stream.raw().cast::<c_void>();
        let code = match variant {
            // SAFETY: FFI call; all pointers valid per function
            // contract (caller upholds the safety notes above).
            Variant::Naive => unsafe {
                logismos_launch_matmul_naive_fp16(a, b, d, m, n, k, raw_stream)
            },
            // SAFETY: same.
            Variant::Wmma => unsafe {
                logismos_launch_matmul_wmma_fp16(a, b, d, m, n, k, raw_stream)
            },
        };
        if code == 0 {
            Ok(())
        } else {
            Err(Error::Launch {
                kernel: match variant {
                    Variant::Naive => "matmul_naive_fp16",
                    Variant::Wmma => "matmul_wmma_fp16",
                },
                code,
            })
        }
    }
}
