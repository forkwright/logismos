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

/// Rejects a rope shape whose composite device-memory index the
/// kernel cannot address without its 32-bit intermediates overflowing.
///
/// WHY here and not only in `praxis::rope::dim_i32`: that check
/// confirms each of `batch`/`seq`/`heads`/`head_dim` individually
/// fits `i32`, but `rope.hip`'s composite write index
/// `((b*seq+s)*heads+h)*head_dim` overflows a 32-bit product at
/// long-context shapes even when every individual dimension is well
/// within `i32` range (forkwright/logismos#33) — a defect in the
/// composite, not in any one operand, so no per-operand check can
/// catch it. `logismos_launch_rope_fp16` above is declared without
/// `pub` inside this module's private `extern "C"` block, so
/// [`launch_rope_fp16_in_place`] is the only Rust path able to reach
/// it; checking here means no future caller can route around this by
/// skipping `praxis`.
///
/// The bound: `qk` holds exactly `batch*seq*heads*head_dim` fp16
/// elements, and the kernel's highest write index is
/// `batch*seq*heads*head_dim - 1` (derived from the composite base
/// plus the largest `2*pair_idx+1` offset, `head_dim - 1`, for even
/// `head_dim`). Rejecting whenever the raw product exceeds
/// `i32::MAX` is a one-element-conservative superset of that exact
/// bound and is cheaper to state and verify.
fn check_rope_shape(batch: i32, seq: i32, heads: i32, head_dim: i32) -> Result<()> {
    for (value, name) in [
        (batch, "batch"),
        (seq, "seq"),
        (heads, "heads"),
        (head_dim, "head_dim"),
    ] {
        if value <= 0 {
            return Err(Error::UnsupportedShape {
                kernel: "rope_fp16",
                msg: format!("{name} must be positive, got {value}"),
            });
        }
    }
    // INVARIANT: all four operands are positive `i32` here, so this
    // `i64` product is bounded by `i32::MAX as i64` to the 4th power —
    // far under `i64::MAX` — and cannot itself overflow.
    let total = i64::from(batch) * i64::from(seq) * i64::from(heads) * i64::from(head_dim);
    if total > i64::from(i32::MAX) {
        let max = i32::MAX;
        return Err(Error::UnsupportedShape {
            kernel: "rope_fp16",
            msg: format!(
                "batch*seq*heads*head_dim = {total} exceeds i32::MAX ({max}); \
                 rope.hip's composite base index cannot address this shape"
            ),
        });
    }
    Ok(())
}

/// In-place rotary embedding on a `(B, S, H, D)` fp16 tensor.
///
/// # Errors
///
/// [`Error::UnsupportedShape`] if the shape's composite index would
/// overflow the kernel's 32-bit-derived arithmetic; [`Error::NoGpuBuild`]
/// or [`Error::Launch`] otherwise.
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
    check_rope_shape(batch, seq, heads, head_dim)?;

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
                kind: hipcore::ErrorKind::from_raw(code),
                code,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "test assertions use expect() directly")]

    use super::*;

    // WHY host-side only: `check_rope_shape` is pure `i32`/`i64`
    // arithmetic with no device access, so it needs neither `hipcc` (HIP
    // kernel compile) nor a physical HIP device to run. That is
    // deliberate — the device write this guards
    // (`rope.hip`'s `qk[base + ...]`) cannot be exercised on any box in
    // this fleet, metis or CI (forkwright/logismos#95): no box carries a
    // physical AMD GPU, and CI's `libamdhip64-dev` package supplies
    // headers/link only. This test is what stands in for that
    // unreachable coverage — it pins the exact boundary the launcher
    // enforces before a kernel launch is even attempted, and it calls
    // `check_rope_shape` itself rather than a re-derivation of its
    // arithmetic.

    #[test]
    fn check_rope_shape_rejects_overflowing_product() {
        // batch=1, seq=131_073, heads=128, head_dim=128: total =
        // 2_147_500_032, which is 16_385 past i32::MAX (2_147_483_647).
        // Exactly the dimensions forkwright/logismos#33 cites as the
        // attacker-reachable case: a GGUF file can declare this `seq`
        // legitimately.
        let err =
            check_rope_shape(1, 131_073, 128, 128).expect_err("overflowing shape must be rejected");
        assert!(
            matches!(
                &err,
                Error::UnsupportedShape { kernel: "rope_fp16", msg }
                    if msg.contains("2147500032") && msg.contains("i32::MAX")
            ),
            "expected UnsupportedShape citing the overflowing product, got {err:?}"
        );
    }

    #[test]
    fn check_rope_shape_accepts_in_range_shape() {
        // The exact shape `op_parity.rs::rope_parity` launches — this
        // check must never reject a shape the kernel can actually
        // address, or the guard becomes a false-positive outage.
        assert!(check_rope_shape(2, 64, 8, 128).is_ok());
    }

    #[test]
    fn check_rope_shape_rejects_nonpositive_dims() {
        assert!(check_rope_shape(0, 64, 8, 128).is_err());
        assert!(check_rope_shape(1, -1, 8, 128).is_err());
        assert!(check_rope_shape(1, 64, 0, 128).is_err());
        assert!(check_rope_shape(1, 64, 8, -128).is_err());
    }
}
