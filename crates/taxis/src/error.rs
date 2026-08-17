//! Error surface for `taxis`.

use snafu::Snafu;

/// Result alias used throughout `taxis`.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors surfaced by tensor construction, reshape, and storage ops.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum Error {
    /// Underlying HIP failure (allocation, copy, etc.).
    #[snafu(transparent)]
    Hip {
        /// Source HIP error.
        source: hipcore::Error,
    },

    /// Incompatible dtype for the requested operation.
    #[snafu(display("dtype mismatch: expected {expected:?}, got {got:?} (in {op})"))]
    DTypeMismatch {
        /// Operation name.
        op: &'static str,
        /// Required dtype.
        expected: crate::DType,
        /// Supplied dtype.
        got: crate::DType,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Shape mismatch between operands or between a shape and a slice.
    #[snafu(display("shape mismatch in {op}: {msg}"))]
    ShapeMismatch {
        /// Operation name.
        op: &'static str,
        /// Free-form description of the mismatch.
        msg: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Broadcasting rule could not be applied.
    #[snafu(display("broadcast failed: {lhs:?} vs {rhs:?}"))]
    BroadcastFailed {
        /// Left-hand dims.
        lhs: Vec<usize>,
        /// Right-hand dims.
        rhs: Vec<usize>,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Layout is not contiguous where a contiguous layout was required.
    #[snafu(display("non-contiguous layout in {op}: {msg}"))]
    NotContiguous {
        /// Operation name.
        op: &'static str,
        /// Free-form description.
        msg: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Tensor is on the wrong device or storage kind.
    #[snafu(display("wrong storage kind in {op}: {msg}"))]
    WrongStorage {
        /// Operation name.
        op: &'static str,
        /// Free-form description.
        msg: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Generic validation failure.
    #[snafu(display("taxis: {message}"))]
    Msg {
        /// Free-form description.
        message: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}
