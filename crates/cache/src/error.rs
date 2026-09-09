//! Cache error surface.

use snafu::Snafu;

/// Result alias used throughout `cache`.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors surfaced by KV-cache operations.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum Error {
    /// Underlying HIP ownership or synchronization failure for the optional native cache.
    #[cfg(feature = "gpu")]
    #[snafu(transparent)]
    Hip {
        /// Source HIP failure.
        source: hipcore::Error,
    },

    /// Underlying cache-agnostic native copy or append kernel failure.
    #[cfg(feature = "gpu")]
    #[snafu(transparent)]
    Kernel {
        /// Source kernel failure.
        source: kernels::Error,
    },

    /// Underlying tensor-layer failure.
    #[cfg(feature = "flat")]
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
    #[cfg(feature = "flat")]
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

    /// Paged-KV metadata allocation failed before the pool was mutated.
    #[snafu(display("cache: could not reserve paged-KV {target} metadata"))]
    PagedAllocation {
        /// Allocation purpose.
        target: &'static str,
        /// Allocation failure retained for callers.
        source: std::collections::TryReserveError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Paged-KV geometry has a zero dimension.
    #[snafu(display("cache: paged-KV {field} must be nonzero"))]
    PagedZeroDimension {
        /// Invalid geometry field.
        field: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Checked paged-KV accounting overflowed.
    #[snafu(display("cache: paged-KV {operation} overflowed"))]
    PagedArithmetic {
        /// Checked arithmetic operation.
        operation: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// An append contains no rows.
    #[snafu(display("cache: paged-KV append must contain at least one token"))]
    PagedEmptyAppend {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// An append exceeds the pool's admitted context.
    #[snafu(display(
        "cache: paged-KV append of {append_tokens} tokens at {committed_tokens} exceeds context {max_context}"
    ))]
    PagedContextOverflow {
        /// Committed token count.
        committed_tokens: usize,
        /// Requested append count.
        append_tokens: usize,
        /// Pool context bound.
        max_context: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A fully preallocated pool cannot stage the requested page changes.
    #[snafu(display(
        "cache: paged-KV requires {required_bundles} free bundles, only {available_bundles} remain"
    ))]
    PagedCapacity {
        /// Bundles needed by this transaction.
        required_bundles: usize,
        /// Currently free bundles.
        available_bundles: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Paged-KV layer index is outside the all-layer pool.
    #[snafu(display("cache: paged-KV layer {layer} is outside {layers} layers"))]
    PagedLayerOutOfRange {
        /// Requested layer.
        layer: usize,
        /// Pool layer count.
        layers: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A key or value row has the wrong width.
    #[snafu(display(
        "cache: paged-KV {kind} row for layer {layer} has width {actual}, expected {expected}"
    ))]
    PagedRowWidth {
        /// K/V row kind.
        kind: &'static str,
        /// Layer receiving the row.
        layer: usize,
        /// Observed element count.
        actual: usize,
        /// Required row width.
        expected: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A transaction token index is outside its requested append.
    #[snafu(display(
        "cache: paged-KV transaction token {token} is outside append length {append_tokens}"
    ))]
    PagedAppendTokenOutOfRange {
        /// Requested transaction-relative token.
        token: usize,
        /// Transaction append length.
        append_tokens: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A layer row was not written in contiguous token order.
    #[snafu(display(
        "cache: paged-KV layer {layer} expected transaction token {expected}, got {actual}"
    ))]
    PagedWriteOrder {
        /// Layer receiving the row.
        layer: usize,
        /// Required next transaction-relative token.
        expected: usize,
        /// Supplied transaction-relative token.
        actual: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A view attempted to read a token that has not been committed or staged for that layer.
    #[snafu(display("cache: paged-KV token {token} is outside visible length {visible_tokens}"))]
    PagedReadBeyondVisible {
        /// Requested absolute token.
        token: usize,
        /// Rows safely visible to the caller.
        visible_tokens: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Commit found a layer that did not receive every requested row.
    #[snafu(display(
        "cache: paged-KV layer {layer} wrote {written_tokens} of {append_tokens} transaction rows"
    ))]
    PagedIncompleteAppend {
        /// Incomplete layer.
        layer: usize,
        /// Rows received for that layer.
        written_tokens: usize,
        /// Rows required for every layer.
        append_tokens: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Internal checked layout did not match its preallocated backing.
    #[snafu(display("cache: paged-KV {operation} exceeded its validated layout"))]
    PagedLayout {
        /// Failed internal layout operation.
        operation: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A submitted native cache operation made completion uncertain.
    #[cfg(feature = "gpu")]
    #[snafu(display("cache: native paged-KV backing is poisoned after submitted device work"))]
    PagedNativePoisoned {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A native append has already been prepared and awaits completion proof.
    #[cfg(any(feature = "gpu", test))]
    #[snafu(display("cache: native paged-KV commit is already prepared"))]
    PagedNativeCommitPrepared {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Native publication was requested without a prepared all-layer append.
    #[cfg(any(feature = "gpu", test))]
    #[snafu(display("cache: native paged-KV commit was not prepared"))]
    PagedNativeCommitNotPrepared {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A stream does not belong to this native cache pool's process-local device.
    #[cfg(feature = "gpu")]
    #[snafu(display(
        "cache: native paged-KV stream device {actual} does not match pool device {expected}"
    ))]
    PagedNativeDeviceMismatch {
        /// Process-local ordinal of the pool's device.
        expected: std::ffi::c_int,
        /// Process-local ordinal of the supplied stream's device.
        actual: std::ffi::c_int,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}
