//! Error types for the `tokenize` crate.

/// Result alias used throughout `tokenize`.
pub type Result<T> = core::result::Result<T, Error>;

/// Tokenizer-surface errors.
///
/// The upstream crate uses `Box<dyn std::error::Error + Send + Sync>`
/// which we decline to re-export: it would leak the upstream-specific
/// error hierarchy across the facade. Instead, the upstream's
/// `Display` is captured as a string.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// Upstream `tokenizers` failure (loading, encode, decode).
    #[error("tokenizers upstream: {0}")]
    Upstream(String),

    /// Free-form error.
    #[error("tokenize: {0}")]
    Msg(String),
}
