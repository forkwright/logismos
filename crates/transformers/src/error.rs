//! Error type for `transformers`.

/// Transformer-block errors.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A shape contract was violated at block construction or forward.
    #[error("shape: {0}")]
    Shape(String),
    /// Downstream kernel failure (unused in Phase 3; reserved for Phase 6).
    #[error("kernel: {0}")]
    Kernel(String),
}

/// Crate-local result alias.
pub type Result<T> = std::result::Result<T, Error>;
