//! Matmul kernel: HIP launcher + CPU reference.
//!
//! Contract: `D = A @ B` where A is `(M, K)`, B is `(K, N)`, D is
//! `(M, N)`, all row-major, fp16 input, fp16 output with fp32
//! accumulation inside the kernel.

pub mod cpu;

use std::ffi::c_void;

use hipcore::Stream;

#[cfg(logismos_no_gpu_kernels)]
use crate::error::NoGpuBuildSnafu;
use crate::error::{Result, UnsupportedShapeSnafu};
// WHY cfg-gated: only the `not(logismos_no_gpu_kernels)` launcher body builds
// launch errors, so an unconditional import fails `-D warnings` on hipcc-less
// (CPU-only) builds.
#[cfg(not(logismos_no_gpu_kernels))]
use crate::error::LaunchSnafu;
// WHY imported without a code reference: the `# Errors` sections below link to
// `Error` variants by intra-doc path, which rustdoc resolves only against items
// in scope. Split from the group above so the expectation covers this import
// alone -- a later genuinely-unused import in the group still fails the gate.
#[expect(
    unused_imports,
    reason = "resolves intra-doc links in this module's `# Errors` sections"
)]
use crate::error::Error;

#[cfg_attr(
    logismos_no_gpu_kernels,
    allow(
        dead_code,
        reason = "schema-contract: HIP kernel ABI declarations, linked only when GPU kernels are compiled"
    )
)]
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

/// Rejects an `M`/`N`/`K` combination whose element-index products the
/// naive/WMMA kernels cannot address without their 32-bit-derived
/// intermediates overflowing.
///
/// WHY here and not only in `praxis::matmul::dim_i32`: that check
/// confirms each of `M`/`N`/`K` individually fits `i32`, but
/// `matmul_naive.hip`'s element-index products `row*K+i` and `i*N+col`
/// overflow a 32-bit product for large dimensions even when `M`, `N`,
/// and `K` are each individually small (forkwright/logismos#32) — a
/// defect in the composite index, not in any one operand, so a
/// per-operand check cannot catch it. Both
/// `logismos_launch_matmul_naive_fp16` and
/// `logismos_launch_matmul_wmma_fp16` above are declared without `pub`
/// inside this module's private `extern "C"` block, so
/// [`launch_matmul_fp16`] is the only Rust path able to reach either —
/// checking here, once, covers `Variant::Wmma` too (its tile-indexed
/// reads/store share the identical `M*K`/`K*N`/`M*N` bound), without a
/// second check per variant and without a future caller able to route
/// around it by skipping `praxis`.
fn check_matmul_shape(m: i32, n: i32, k: i32) -> Result<()> {
    for (value, name) in [(m, "M"), (n, "N"), (k, "K")] {
        if value <= 0 {
            return UnsupportedShapeSnafu {
                kernel: "matmul_fp16",
                msg: format!("{name} must be positive, got {value}"),
            }
            .fail();
        }
    }
    // INVARIANT: all three operands are positive `i32` here, so each
    // `i64` product is bounded by `i32::MAX as i64` squared — far
    // under `i64::MAX` — and cannot itself overflow.
    let (m64, n64, k64) = (i64::from(m), i64::from(n), i64::from(k));
    for (label, product) in [("M*K", m64 * k64), ("K*N", k64 * n64), ("M*N", m64 * n64)] {
        if product > i64::from(i32::MAX) {
            let max = i32::MAX;
            return UnsupportedShapeSnafu {
                kernel: "matmul_fp16",
                msg: format!(
                    "{label} = {product} exceeds i32::MAX ({max}); matmul_naive.hip's \
                     element-index products cannot address this shape"
                ),
            }
            .fail();
        }
    }
    Ok(())
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
/// - [`Error::UnsupportedShape`] if `M`/`N`/`K` would drive an
///   element-index product past `i32::MAX` inside the kernel.
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
    check_matmul_shape(m, n, k)?;

    #[cfg(logismos_no_gpu_kernels)]
    {
        let _ = (variant, a, b, d, m, n, k, stream);
        NoGpuBuildSnafu { kernel: "matmul" }.fail()
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
            LaunchSnafu {
                kernel: match variant {
                    Variant::Naive => "matmul_naive_fp16",
                    Variant::Wmma => "matmul_wmma_fp16",
                },
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
    // WHY not via `use super::*`: the parent's `Error` import is declared
    // `#[expect(unused_imports)]` for intra-doc links; resolving these
    // assertions through this dedicated import keeps that expectation
    // fulfilled in test builds.
    use crate::error::Error;

    // WHY host-side only: `check_matmul_shape` is pure `i32`/`i64`
    // arithmetic with no device access, so it needs neither `hipcc` (HIP
    // kernel compile) nor a physical HIP device to run. That is
    // deliberate — the device read this guards
    // (`matmul_naive.hip`'s `a[row*k+i]` / `b[i*n+col]`) cannot be
    // exercised on any box in this fleet, metis or CI
    // (forkwright/logismos#95): no box carries a physical AMD GPU, and
    // CI's `libamdhip64-dev` package supplies headers/link only. This
    // test is what stands in for that unreachable coverage — it pins
    // the exact boundary the launcher enforces before a kernel launch
    // is even attempted, and it calls `check_matmul_shape` itself
    // rather than a re-derivation of its arithmetic.

    #[test]
    fn check_matmul_shape_rejects_overflowing_product() {
        // M=32_768, N=1, K=65_538: M*K = 2_147_549_184, which is 65_537
        // past i32::MAX (2_147_483_647). Exactly the dimensions
        // forkwright/logismos#32 cites (row=32_767 zero-indexed => M=32_768,
        // K=65_538) as the attacker-reachable case: a GGUF file can
        // declare matrix dimensions this large legitimately.
        let err =
            check_matmul_shape(32_768, 1, 65_538).expect_err("overflowing shape must be rejected");
        assert!(
            matches!(
                &err,
                Error::UnsupportedShape { kernel: "matmul_fp16", msg, .. }
                    if msg.contains("2147549184") && msg.contains("i32::MAX")
            ),
            "expected UnsupportedShape citing the overflowing product, got {err:?}"
        );
    }

    #[test]
    fn check_matmul_shape_accepts_in_range_shapes() {
        // The exact shapes `matmul_parity.rs` launches — this check
        // must never reject a shape the kernels can actually address,
        // or the guard becomes a false-positive outage. Covers both
        // the naive-dispatch and WMMA-dispatch parity cases; the check
        // is variant-agnostic (see the WHY above `check_matmul_shape`).
        assert!(check_matmul_shape(32, 48, 64).is_ok());
        assert!(check_matmul_shape(256, 256, 256).is_ok());
    }

    #[test]
    fn check_matmul_shape_rejects_nonpositive_dims() {
        assert!(check_matmul_shape(0, 48, 64).is_err());
        assert!(check_matmul_shape(32, -48, 64).is_err());
        assert!(check_matmul_shape(32, 48, 0).is_err());
    }
}
