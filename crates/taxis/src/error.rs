//! Error surface for `taxis`.

/// Result alias used throughout `taxis`.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors surfaced by tensor construction, reshape, and storage ops.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// Underlying HIP failure (allocation, copy, etc.).
    #[error(transparent)]
    Hip(#[from] hipcore::Error),

    /// Incompatible dtype for the requested operation.
    #[error("dtype mismatch: expected {expected:?}, got {got:?} (in {op})")]
    DTypeMismatch {
        /// Operation name.
        op: &'static str,
        /// Required dtype.
        expected: crate::DType,
        /// Supplied dtype.
        got: crate::DType,
    },

    /// Shape mismatch between operands or between a shape and a slice.
    #[error("shape mismatch in {op}: {msg}")]
    ShapeMismatch {
        /// Operation name.
        op: &'static str,
        /// Free-form description of the mismatch.
        msg: String,
    },

    /// Broadcasting rule could not be applied.
    #[error("broadcast failed: {lhs:?} vs {rhs:?}")]
    BroadcastFailed {
        /// Left-hand dims.
        lhs: Vec<usize>,
        /// Right-hand dims.
        rhs: Vec<usize>,
    },

    /// Layout is not contiguous where a contiguous layout was required.
    #[error("non-contiguous layout in {op}: {msg}")]
    NotContiguous {
        /// Operation name.
        op: &'static str,
        /// Free-form description.
        msg: String,
    },

    /// Tensor is on the wrong device or storage kind.
    #[error("wrong storage kind in {op}: {msg}")]
    WrongStorage {
        /// Operation name.
        op: &'static str,
        /// Free-form description.
        msg: String,
    },

    /// Generic validation failure.
    #[error("taxis: {0}")]
    Msg(String),
}
