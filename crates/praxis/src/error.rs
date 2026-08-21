//! `praxis` error surface — lifts the underlying crates.

use snafu::Snafu;

/// Result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors surfaced by `praxis` free functions.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum Error {
    /// Propagated HIP error.
    #[snafu(transparent)]
    Hip {
        /// Source HIP error.
        source: hipcore::Error,
    },

    /// Propagated tensor error.
    #[snafu(transparent)]
    Taxis {
        /// Source tensor error.
        source: taxis::Error,
    },

    /// Propagated kernel error.
    #[snafu(transparent)]
    Kernel {
        /// Source kernel error.
        source: kernels::Error,
    },

    /// Invalid shape / dtype combination for the requested op.
    #[snafu(display("praxis {op}: {msg}"))]
    Invalid {
        /// Op name.
        op: &'static str,
        /// Description.
        msg: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}
