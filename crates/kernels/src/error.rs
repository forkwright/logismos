//! `kernels` error surface.

/// Result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors surfaced by the kernel launchers and CPU references.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// Propagated HIP failure.
    #[error(transparent)]
    Hip(#[from] hipcore::Error),

    /// Propagated tensor failure.
    #[error(transparent)]
    Taxis(#[from] taxis::Error),

    /// Kernel launch failed — HIP reported a non-success status after
    /// kernel submission.
    #[error("kernel {kernel}: launch failed: {kind:?} (code {code})")]
    Launch {
        /// Symbolic kernel name.
        kernel: &'static str,
        /// Classified HIP error kind for `code`, so a launch failure
        /// is diagnosable without cross-referencing the HIP headers.
        kind: hipcore::ErrorKind,
        /// Raw HIP error code.
        code: u32,
    },

    /// Tensor shape is not supported by this kernel.
    #[error("kernel {kernel}: unsupported shape: {msg}")]
    UnsupportedShape {
        /// Symbolic kernel name.
        kernel: &'static str,
        /// Description.
        msg: String,
    },

    /// Build was produced without the HIP kernel archive (e.g. `hipcc`
    /// was absent). CPU references still work; GPU paths return this.
    #[error("kernel {kernel}: no-GPU build (set HIPCC or install ROCm to enable)")]
    NoGpuBuild {
        /// Symbolic kernel name.
        kernel: &'static str,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_error_carries_symbolic_kind() {
        // WHY(forkwright/logismos#59): `Error::Launch` used to carry
        // only the raw `u32` HIP code, so diagnosing a launch failure
        // meant manually cross-referencing the HIP headers. `kind` is
        // derived from the same code via `hipcore::ErrorKind::from_raw`
        // (the mapping this finding says already existed) and its
        // `Display` now names the failure instead of just the number.
        let err = Error::Launch {
            kernel: "matmul_naive_fp16",
            kind: hipcore::ErrorKind::from_raw(2), // hipErrorOutOfMemory
            code: 2,
        };
        assert_eq!(
            hipcore::ErrorKind::from_raw(2),
            hipcore::ErrorKind::OutOfMemory
        );
        let rendered = err.to_string();
        assert!(
            rendered.contains("OutOfMemory"),
            "Display must name the classified kind, got: {rendered}"
        );
        assert!(
            rendered.contains('2'),
            "Display must still carry the raw code, got: {rendered}"
        );
    }
}
