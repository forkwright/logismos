//! Error types for the `loader` crate.

use std::path::PathBuf;

use snafu::Snafu;

/// Result alias used throughout `loader`.
pub type Result<T> = core::result::Result<T, Error>;

/// Loader-surface errors.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum Error {
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
    #[snafu(display("unknown archive format at {}", path.display()))]
    UnknownFormat {
        /// Offending path.
        path: PathBuf,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The mapped file's length no longer matches what was observed
    /// when the mapping was created — most likely a concurrent re-save
    /// or truncation of a weights file the loader has open.
    #[snafu(display(
        "{} changed size since it was mapped ({expected_len}B -> {actual_len}B); \
         refusing a stale mapping",
        path.display()
    ))]
    MmapStale {
        /// The file whose length changed.
        path: PathBuf,
        /// Length observed when the mapping was created.
        expected_len: u64,
        /// Length observed on the just-completed re-stat.
        actual_len: u64,
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

impl From<::safetensors::SafeTensorError> for Error {
    fn from(value: ::safetensors::SafeTensorError) -> Self {
        SafetensorsSnafu {
            message: value.to_string(),
        }
        .build()
    }
}
