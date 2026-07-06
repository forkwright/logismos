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
    #[error("kernel {kernel}: launch failed (code {code})")]
    Launch {
        /// Symbolic kernel name.
        kernel: &'static str,
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
