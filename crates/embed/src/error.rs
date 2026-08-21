//! Error type for `embed`.

use snafu::Snafu;

/// Embed crate errors.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum Error {
    /// Model directory lookup failed.
    #[snafu(display("io: {message}"))]
    Io {
        /// Free-form description.
        message: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Encoder crate bubbled an error.
    #[snafu(transparent)]
    Encoders {
        /// Source encoder error.
        source: encoders::Error,
    },
    /// Loader crate bubbled an error.
    #[snafu(transparent)]
    Loader {
        /// Source loader error.
        source: loader::Error,
    },
    /// Tokenizer crate bubbled an error.
    #[snafu(transparent)]
    Tokenize {
        /// Source tokenizer error.
        source: tokenize::Error,
    },
    /// Caller asked for a dim the model does not support.
    #[snafu(display("unsupported dim {dim}"))]
    UnsupportedDim {
        /// The unsupported dimension requested.
        dim: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Input token length exceeds configured max.
    #[snafu(display("input too long: got {got}, limit {limit}"))]
    InputTooLong {
        /// Actual token count.
        got: usize,
        /// Configured maximum.
        limit: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        IoSnafu {
            message: e.to_string(),
        }
        .build()
    }
}

/// Crate-local result alias.
pub type Result<T> = std::result::Result<T, Error>;
