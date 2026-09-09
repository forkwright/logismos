//! Error types for the `loader` crate.

use std::num::NonZeroU64;
use std::path::PathBuf;

use snafu::Snafu;

use crate::gguf::Sha256Digest;

/// Result alias used throughout `loader`.
pub type Result<T> = core::result::Result<T, Error>;

/// Loader-surface errors.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum Error {
    /// Underlying checked tensor geometry or construction failure.
    #[cfg(feature = "tensor")]
    #[snafu(transparent)]
    Taxis {
        /// Source tensor error.
        source: taxis::Error,
    },

    /// Filesystem / mmap failure.
    #[snafu(display("io error: {source}"), context(false))]
    Io {
        /// Underlying IO error.
        source: std::io::Error,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Upstream safetensors parser failure.
    #[cfg(feature = "tensor")]
    #[snafu(display("safetensors: {message}"))]
    Safetensors {
        /// Stringified upstream error.
        message: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// GGUF parser failure with context.
    #[snafu(display("gguf parse error at offset {offset}: {msg}"))]
    Gguf {
        /// Byte offset where the parse failed.
        offset: u64,
        /// Free-form description.
        msg: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// GGUF declared a GGML storage type this loader does not recognize.
    #[snafu(display("unknown ggml type id {type_id} at descriptor offset {offset}"))]
    UnknownGgmlType {
        /// Numeric GGML storage type id from the tensor descriptor.
        type_id: u32,
        /// Byte offset immediately after that id in the GGUF descriptor stream.
        offset: u64,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Requested tensor does not exist in the archive.
    #[snafu(display("tensor `{name}` not found in archive"))]
    TensorNotFound {
        /// Missing tensor name.
        name: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Archive tensor shape disagrees with the declared dtype × element
    /// count.
    #[cfg(feature = "tensor")]
    #[snafu(display(
        "tensor `{name}` shape mismatch: dtype={dtype:?} elem_count={elem_count} \
         expected {expected_bytes}B, got {actual_bytes}B"
    ))]
    ShapeMismatch {
        /// Tensor name.
        name: String,
        /// Archive dtype.
        dtype: taxis::DType,
        /// Declared element count.
        elem_count: usize,
        /// Expected byte count.
        expected_bytes: usize,
        /// Actual byte count in the archive.
        actual_bytes: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Dtype not supported by the Phase-2 loader.
    #[cfg(feature = "tensor")]
    #[snafu(display("tensor `{name}` has unsupported dtype {dtype:?}"))]
    UnsupportedDType {
        /// Tensor name.
        name: String,
        /// Unsupported dtype.
        dtype: taxis::DType,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// `Archive::open` could not dispatch on file extension.
    #[cfg(feature = "tensor")]
    #[snafu(display("unknown archive format at {}", path.display()))]
    UnknownFormat {
        /// Offending path.
        path: PathBuf,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// An opened file's length no longer matches the length that bounded an
    /// mmap or owned inspection stream — most likely a concurrent re-save or
    /// truncation of an artifact the loader has open.
    #[snafu(display(
        "{} changed size during inspection ({expected_len}B -> {actual_len}B); \
         refusing a stale observation",
        path.display()
    ))]
    MmapStale {
        /// The file whose length changed.
        path: PathBuf,
        /// Length observed when the operation began.
        expected_len: u64,
        /// Length observed on the just-completed handle re-stat.
        actual_len: u64,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The serialized artifact cannot fit in the caller-authorized backing.
    #[snafu(display(
        "artifact has {serialized_bytes} serialized bytes, exceeding the caller limit of {limit}"
    ))]
    ArtifactExceedsByteLimit {
        /// Serialized byte length obtained from the opened input file.
        serialized_bytes: u64,
        /// Explicit maximum permitted serialized backing length.
        limit: NonZeroU64,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The supplied input is not a regular file.
    #[snafu(display("verified artifact input {} is not a regular file", path.display()))]
    ArtifactInputNotRegular {
        /// Path supplied to the verified-artifact loader.
        path: PathBuf,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Reserving the verified serialized backing failed.
    #[snafu(display(
        "unable to reserve {serialized_bytes} bytes for the verified artifact backing: {source}"
    ))]
    ArtifactBackingAllocation {
        /// Requested serialized backing length.
        serialized_bytes: u64,
        /// Allocator failure retained for typed error inspection.
        source: std::collections::TryReserveError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The copied serialized bytes do not equal the required artifact identity.
    #[snafu(display("verified artifact digest mismatch: expected {expected}, observed {actual}"))]
    ArtifactDigestMismatch {
        /// Required SHA-256 identity supplied by the caller.
        expected: Sha256Digest,
        /// SHA-256 computed from the privately owned backing bytes.
        actual: Sha256Digest,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Free-form error; prefer a typed variant when adding a new
    /// failure mode.
    #[snafu(display("loader: {message}"))]
    Msg {
        /// Free-form description.
        message: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}

#[cfg(feature = "tensor")]
impl From<::safetensors::SafeTensorError> for Error {
    fn from(value: ::safetensors::SafeTensorError) -> Self {
        SafetensorsSnafu {
            message: value.to_string(),
        }
        .build()
    }
}
