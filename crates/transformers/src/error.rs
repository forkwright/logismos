//! Error type for `transformers`.

use snafu::Snafu;

/// Transformer-block errors.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum Error {
    /// A shape contract was violated at block construction or forward.
    #[snafu(display("shape: {message}"))]
    Shape {
        /// Free-form description.
        message: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Downstream kernel failure (unused in Phase 3; reserved for Phase 6).
    #[snafu(display("kernel: {message}"))]
    Kernel {
        /// Free-form description.
        message: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}

/// Crate-local result alias.
pub type Result<T> = std::result::Result<T, Error>;
