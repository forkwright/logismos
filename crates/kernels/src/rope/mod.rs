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

use crate::error::{Error, LaunchSnafu, NoGpuBuildSnafu, Result, UnsupportedShapeSnafu};

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
            return UnsupportedShapeSnafu {
                kernel: "rope_fp16",
                msg: format!("{name} must be positive, got {value}"),
            }
            .fail();
        }
    }
    // INVARIANT: all four operands are positive `i32` here, but their
    // widened `i64` product is NOT guaranteed to fit `i64`:
    // `(i32::MAX as i64)^4` ≈ 2.13e37 vastly exceeds `i64::MAX` ≈
    // 9.22e18, so a chained `i64::from(a) * i64::from(b) * ...`
    // multiplication can itself overflow and silently wrap to a
    // negative value — one that a bare `total > i32::MAX` comparison
    // then fails to reject (forkwright/logismos#103 review: a prior
    // version of this guard did exactly that and admitted
    // `check_rope_shape(i32::MAX, i32::MAX, 2, 2)`, whose true product
    // ≈1.845e19 wrapped to -17179869180). `checked_mul` folded across
    // every factor turns that overflow into `None`, rejected below
    // explicitly rather than compared as a number.
    let total = [batch, seq, heads, head_dim]
        .into_iter()
        .try_fold(1i64, |acc, value| acc.checked_mul(i64::from(value)));
    let Some(total) = total else {
        return UnsupportedShapeSnafu {
            kernel: "rope_fp16",
            msg: format!(
                "batch*seq*heads*head_dim overflows i64 (batch={batch}, seq={seq}, \
                 heads={heads}, head_dim={head_dim}); rope.hip's composite base \
                 index cannot address this shape"
            ),
        }
        .fail();
    };
    if total > i64::from(i32::MAX) {
        let max = i32::MAX;
        return UnsupportedShapeSnafu {
            kernel: "rope_fp16",
            msg: format!(
                "batch*seq*heads*head_dim = {total} exceeds i32::MAX ({max}); \
                 rope.hip's composite base index cannot address this shape"
            ),
        }
        .fail();
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
        NoGpuBuildSnafu { kernel: "rope" }.fail()
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
            LaunchSnafu {
                kernel: "rope_fp16",
                kind: hipcore::ErrorKind::from_raw(code),
                code,
            }
            .fail()
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
                Error::UnsupportedShape { kernel: "rope_fp16", msg, .. }
                    if msg.contains("2147500032") && msg.contains("i32::MAX")
            ),
            "expected UnsupportedShape citing the overflowing product, got {err:?}"
        );
    }

    #[test]
    fn check_rope_shape_rejects_i64_wraparound_product() {
        // batch=seq=i32::MAX exercises the case the boundary test
        // above does not: a chained `i64` multiply of all four factors
        // that itself overflows i64 (true product ≈1.845e19 vs
        // i64::MAX ≈9.22e18), wrapping to a negative number. Before
        // the `checked_mul` fold, a bare `total > i32::MAX` comparison
        // was false for any negative `total`, so this exact input
        // returned `Ok(())` and admitted a shape that overflows
        // rope.hip's composite index (forkwright/logismos#103 review).
        let err = check_rope_shape(i32::MAX, i32::MAX, 2, 2)
            .expect_err("i64-overflowing product must be rejected, not wrap to a false accept");
        assert!(
            matches!(
                &err,
                Error::UnsupportedShape { kernel: "rope_fp16", msg, .. }
                    if msg.contains("overflows i64")
            ),
            "expected UnsupportedShape citing the i64 overflow, got {err:?}"
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
