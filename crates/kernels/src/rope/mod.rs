//! RoPE (Rotary Position Embedding) — HIP launcher + CPU reference.
//!
//! Standard Llama / Qwen RoPE: the query and key tensors are rotated
//! pairwise along `head_dim`, with angle `pos_base * (theta_base ^
//! (-2i / head_dim))`.
//!
//! Input shape `(batch, seq, heads, head_dim)`, row-major, fp16.
//! Output has the same shape and dtype. `cos_sin` is a precomputed
//! `(seq, head_dim)` fp32 table laid out as interleaved cos/sin per
//! pair of elements along the last axis.
//!
//! Phase 1 ships the CPU reference + an HIP kernel; `praxis::rope_apply`
//! composes them.

pub mod cpu;

use std::ffi::c_void;

use hipcore::Stream;

use crate::error::{Error, Result};

#[cfg_attr(
    logismos_no_gpu_kernels,
    allow(
        dead_code,
        reason = "schema-contract: HIP kernel ABI declarations, linked only when GPU kernels are compiled"
    )
)]
unsafe extern "C" {
    fn logismos_launch_rope_fp16(
        qk_fp16: *mut c_void,
        cos_sin_f32: *const c_void,
        batch: i32,
        seq: i32,
        heads: i32,
        head_dim: i32,
        stream: *mut c_void,
    ) -> u32;
}

/// In-place rotary embedding on a `(B, S, H, D)` fp16 tensor.
///
/// # Errors
///
/// [`Error::NoGpuBuild`] or [`Error::Launch`].
///
/// # Safety
///
/// `qk` and `cos_sin` must point at device allocations of correct
/// sizes on `stream`'s device; `qk` is treated as writable.
pub unsafe fn launch_rope_fp16_in_place(
    qk: *mut c_void,
    cos_sin: *const c_void,
    batch: i32,
    seq: i32,
    heads: i32,
    head_dim: i32,
    stream: &Stream,
) -> Result<()> {
    #[cfg(logismos_no_gpu_kernels)]
    {
        let _ = (qk, cos_sin, batch, seq, heads, head_dim, stream);
        Err(Error::NoGpuBuild { kernel: "rope" })
    }

    #[cfg(not(logismos_no_gpu_kernels))]
    {
        // SAFETY: FFI call; caller upholds pointer validity.
        let code = unsafe {
            logismos_launch_rope_fp16(
                qk,
                cos_sin,
                batch,
                seq,
                heads,
                head_dim,
                stream.raw().cast::<c_void>(),
            )
        };
        if code == 0 {
            Ok(())
        } else {
            Err(Error::Launch {
                kernel: "rope_fp16",
                code,
            })
        }
    }
}
