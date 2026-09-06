//! Error types for the `tokenize` crate.

use snafu::Snafu;

/// Result alias used throughout `tokenize`.
pub type Result<T> = core::result::Result<T, Error>;

/// Tokenizer-surface errors.
///
/// The upstream crate uses `Box<dyn std::error::Error + Send + Sync>`
/// which we decline to re-export: it would leak the upstream-specific
/// error hierarchy across the facade. Instead, the upstream's
/// `Display` is captured as a string.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum Error {
    /// A byte-limit configuration was zero and therefore cannot admit a tokenizer.
    #[snafu(display("tokenizer byte limit must be non-zero, got {maximum}"))]
    InvalidByteLimit {
        /// Invalid requested maximum.
        maximum: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Upstream `tokenizers` failure (loading, encode, decode).
    #[snafu(display("tokenizers upstream: {message}"))]
    Upstream {
        /// Upstream error rendered without exposing its type hierarchy.
        message: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Tokenizer bytes exceeded the caller-supplied parsing limit.
    #[snafu(display("tokenizer bytes have length {actual}, exceeding limit {limit}"))]
    ByteLimitExceeded {
        /// Maximum permitted byte length.
        limit: usize,
        /// Actual supplied byte length.
        actual: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Tokenizer bytes did not have the expected exact length.
    #[snafu(display("tokenizer bytes have length {actual}, expected {expected}"))]
    ByteLengthMismatch {
        /// Required byte length from the explicit tokenizer identity.
        expected: usize,
        /// Actual supplied byte length.
        actual: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Tokenizer bytes did not have the expected SHA-256 digest.
    #[snafu(display("tokenizer bytes do not match the required SHA-256 digest"))]
    DigestMismatch {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}
