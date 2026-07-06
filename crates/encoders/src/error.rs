//! Error type for `encoders`.

/// Encoder crate errors.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Structural violation (shapes, layer count, etc.).
    #[error("shape: {0}")]
    Shape(String),
    /// A weight name referenced in the Stella map was not found in the
    /// archive, or an archive tensor was not consumed.
    #[error("weight layout: {0}")]
    Layout(String),
    /// Downstream loader error.
    #[error("loader: {0}")]
    Loader(#[from] loader::Error),
    /// Downstream transformer-block error.
    #[error("transformers: {0}")]
    Transformers(String),
}

impl From<transformers::Error> for Error {
    fn from(e: transformers::Error) -> Self {
        Self::Transformers(e.to_string())
    }
}

/// Crate-local result alias.
pub type Result<T> = std::result::Result<T, Error>;
