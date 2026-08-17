//! Error type for `encoders`.

use snafu::Snafu;

/// Encoder crate errors.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum Error {
    /// Structural violation (shapes, layer count, etc.).
    #[snafu(display("shape: {message}"))]
    Shape {
        /// Free-form description.
        message: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// A weight name referenced in the Stella map was not found in the
    /// archive, or an archive tensor was not consumed.
    #[snafu(display("weight layout: {message}"))]
    Layout {
        /// Free-form description.
        message: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Downstream loader error.
    #[snafu(transparent)]
    Loader {
        /// Source loader error.
        source: loader::Error,
    },
    /// Downstream transformer-block error.
    #[snafu(display("transformers: {message}"))]
    Transformers {
        /// Stringified downstream error.
        message: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}

impl From<transformers::Error> for Error {
    fn from(e: transformers::Error) -> Self {
        TransformersSnafu {
            message: e.to_string(),
        }
        .build()
    }
}

/// Crate-local result alias.
pub type Result<T> = std::result::Result<T, Error>;
