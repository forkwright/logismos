//! Error surface for the `decode` crate.

use snafu::Snafu;

/// Result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// Decode-surface errors.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum Error {
    /// Logits vector is empty or malformed.
    #[snafu(display("decode: bad logits: {message}"))]
    BadLogits {
        /// Free-form description.
        message: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Sampler parameter out of range.
    #[snafu(display("decode: bad parameter `{name}`: {msg}"))]
    BadParam {
        /// Parameter name.
        name: &'static str,
        /// Free-form description.
        msg: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Free-form error.
    #[snafu(display("decode: {message}"))]
    Msg {
        /// Free-form description.
        message: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}
