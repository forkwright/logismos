//! Error surface for the `decode` crate.

/// Result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// Decode-surface errors.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// Logits vector is empty or malformed.
    #[error("decode: bad logits: {0}")]
    BadLogits(String),

    /// Sampler parameter out of range.
    #[error("decode: bad parameter `{name}`: {msg}")]
    BadParam {
        /// Parameter name.
        name: &'static str,
        /// Free-form description.
        msg: String,
    },

    /// Free-form error.
    #[error("decode: {0}")]
    Msg(String),
}
