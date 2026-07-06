//! Cache error surface.

/// Result alias used throughout `cache`.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors surfaced by KV-cache operations.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// Underlying tensor-layer failure.
    #[error(transparent)]
    Taxis(#[from] taxis::Error),

    /// Layer index out of range for this cache.
    #[error("cache: layer {layer_idx} out of bounds (num_layers={num_layers})")]
    LayerOutOfRange {
        /// Requested layer index.
        layer_idx: usize,
        /// Cache's declared layer count.
        num_layers: usize,
    },

    /// Appending `n_new` tokens would exceed `max_seq_len`.
    #[error(
        "cache: layer {layer_idx} overflow — have {current}, adding {n_new} \
         exceeds max_seq_len={max_seq_len}"
    )]
    LenOverflow {
        /// Layer that overflowed.
        layer_idx: usize,
        /// Current written length.
        current: usize,
        /// Tokens the caller asked to append.
        n_new: usize,
        /// Cache's declared `max_seq_len`.
        max_seq_len: usize,
    },

    /// Read request exceeds the layer's written length.
    #[error("cache: layer {layer_idx} read {requested} > written {current}")]
    ReadBeyondWritten {
        /// Layer index.
        layer_idx: usize,
        /// Requested read length.
        requested: usize,
        /// Current written length.
        current: usize,
    },

    /// Dtype of the supplied tensor does not match the cache.
    #[error("cache: dtype mismatch — cache={cache:?}, supplied={supplied:?}")]
    DTypeMismatch {
        /// Cache dtype.
        cache: taxis::DType,
        /// Supplied tensor dtype.
        supplied: taxis::DType,
    },

    /// Supplied tensor shape is incompatible with the cache's layout.
    #[error("cache: shape mismatch — {msg}")]
    ShapeMismatch {
        /// Free-form description.
        msg: String,
    },

    /// Free-form error.
    #[error("cache: {0}")]
    Msg(String),
}
