//! Error types for the `loader` crate.

use std::path::PathBuf;

/// Result alias used throughout `loader`.
pub type Result<T> = core::result::Result<T, Error>;

/// Loader-surface errors.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// Filesystem / mmap failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Upstream safetensors parser failure.
    #[error("safetensors: {0}")]
    Safetensors(String),

    /// GGUF parser failure with context.
    #[error("gguf parse error at offset {offset}: {msg}")]
    Gguf {
        /// Byte offset where the parse failed.
        offset: u64,
        /// Free-form description.
        msg: String,
    },

    /// Requested tensor does not exist in the archive.
    #[error("tensor `{name}` not found in archive")]
    TensorNotFound {
        /// Missing tensor name.
        name: String,
    },

    /// Archive tensor shape disagrees with the declared dtype × element
    /// count.
    #[error(
        "tensor `{name}` shape mismatch: dtype={dtype:?} elem_count={elem_count} \
         expected {expected_bytes}B, got {actual_bytes}B"
    )]
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
    },

    /// Dtype not supported by the Phase-2 loader.
    #[error("tensor `{name}` has unsupported dtype {dtype:?}")]
    UnsupportedDType {
        /// Tensor name.
        name: String,
        /// Unsupported dtype.
        dtype: taxis::DType,
    },

    /// `Archive::open` could not dispatch on file extension.
    #[error("unknown archive format at {}", path.display())]
    UnknownFormat {
        /// Offending path.
        path: PathBuf,
    },

    /// Free-form error; prefer a typed variant when adding a new
    /// failure mode.
    #[error("loader: {0}")]
    Msg(String),
}

impl From<::safetensors::SafeTensorError> for Error {
    fn from(value: ::safetensors::SafeTensorError) -> Self {
        Self::Safetensors(value.to_string())
    }
}
