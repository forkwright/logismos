//! `praxis` error surface — lifts the underlying crates.

/// Result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors surfaced by `praxis` free functions.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// Propagated HIP error.
    #[error(transparent)]
    Hip(#[from] hipcore::Error),

    /// Propagated tensor error.
    #[error(transparent)]
    Taxis(#[from] taxis::Error),

    /// Propagated kernel error.
    #[error(transparent)]
    Kernel(#[from] kernels::Error),

    /// Invalid shape / dtype combination for the requested op.
    #[error("praxis {op}: {msg}")]
    Invalid {
        /// Op name.
        op: &'static str,
        /// Description.
        msg: String,
    },
}
