//! Error type for `embed`.

/// Embed crate errors.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Model directory lookup failed.
    #[error("io: {0}")]
    Io(String),
    /// Encoder crate bubbled an error.
    #[error("encoders: {0}")]
    Encoders(#[from] encoders::Error),
    /// Loader crate bubbled an error.
    #[error("loader: {0}")]
    Loader(#[from] loader::Error),
    /// Tokenizer crate bubbled an error.
    #[error("tokenize: {0}")]
    Tokenize(#[from] tokenize::Error),
    /// Caller asked for a dim the model does not support.
    #[error("unsupported dim {0}")]
    UnsupportedDim(usize),
    /// Input token length exceeds configured max.
    #[error("input too long: got {got}, limit {limit}")]
    InputTooLong {
        /// Actual token count.
        got: usize,
        /// Configured maximum.
        limit: usize,
    },
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

/// Crate-local result alias.
pub type Result<T> = std::result::Result<T, Error>;
