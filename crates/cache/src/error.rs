//! Cache error surface.

use snafu::Snafu;

/// Result alias used throughout `cache`.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors surfaced by KV-cache operations.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum Error {
    /// Underlying tensor-layer failure.
    #[snafu(transparent)]
    Taxis {
        /// Source tensor error.
        source: taxis::Error,
    },

    /// Layer index out of range for this cache.
    #[snafu(display("cache: layer {layer_idx} out of bounds (num_layers={num_layers})"))]
    LayerOutOfRange {
        /// Requested layer index.
        layer_idx: usize,
        /// Cache's declared layer count.
        num_layers: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Appending `n_new` tokens would exceed `max_seq_len`.
    #[snafu(display(
        "cache: layer {layer_idx} overflow — have {current}, adding {n_new} \
         exceeds max_seq_len={max_seq_len}"
    ))]
    LenOverflow {
        /// Layer that overflowed.
        layer_idx: usize,
        /// Current written length.
        current: usize,
        /// Tokens the caller asked to append.
        n_new: usize,
        /// Cache's declared `max_seq_len`.
        max_seq_len: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Read request exceeds the layer's written length.
    #[snafu(display("cache: layer {layer_idx} read {requested} > written {current}"))]
    ReadBeyondWritten {
        /// Layer index.
        layer_idx: usize,
        /// Requested read length.
        requested: usize,
        /// Current written length.
        current: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Dtype of the supplied tensor does not match the cache.
    #[snafu(display("cache: dtype mismatch — cache={cache:?}, supplied={supplied:?}"))]
    DTypeMismatch {
        /// Cache dtype.
        cache: taxis::DType,
        /// Supplied tensor dtype.
        supplied: taxis::DType,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Supplied tensor shape is incompatible with the cache's layout.
    #[snafu(display("cache: shape mismatch — {msg}"))]
    ShapeMismatch {
        /// Free-form description.
        msg: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Tensor storage this cache cannot marshal to bytes: a non-CPU-backed
    /// tensor, or a `CpuStorage` variant this crate does not yet decode.
    /// Distinct from [`Error::ShapeMismatch`] — the dimensions may be
    /// perfectly valid; the storage *representation* is what this code
    /// path cannot handle.
    #[snafu(display("cache: unsupported storage — {msg}"))]
    UnsupportedStorage {
        /// Free-form description.
        msg: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Free-form error.
    #[snafu(display("cache: {message}"))]
    Msg {
        /// Free-form description.
        message: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}
