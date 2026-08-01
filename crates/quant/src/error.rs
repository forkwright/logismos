//! Errors surfaced by `quant` preflight utilities.

use core::fmt;

use crate::scheme::TurboQuantScheme;

/// Crate-local result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors surfaced by `quant` preflight utilities.
#[derive(Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// Caller supplied a quantized index outside the scheme's codebook range.
    IndexOutOfRange {
        /// Quantization scheme being packed.
        scheme: TurboQuantScheme,
        /// Index position in the 32-value block.
        position: usize,
        /// Supplied index value.
        value: u8,
        /// Maximum allowed index for the scheme.
        max: u8,
    },

    /// Caller supplied a value count that is not one 128-value head chunk.
    InvalidHeadDim {
        /// Supplied scalar count.
        got: usize,
        /// Required scalar count.
        expected: usize,
    },

    /// Caller supplied a block count that is not one 128-value head chunk.
    InvalidBlockCount {
        /// Quantization scheme being decoded.
        scheme: TurboQuantScheme,
        /// Supplied block count.
        got: usize,
        /// Required block count.
        expected: usize,
    },

    /// The requested FWHT encode/decode path has not landed yet.
    Unsupported {
        /// Operation name.
        operation: &'static str,
        /// Reason the operation is not available.
        reason: &'static str,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IndexOutOfRange {
                scheme,
                position,
                value,
                max,
            } => write!(
                f,
                "{scheme}: index at position {position} has value {value}, max {max}"
            ),
            Self::InvalidHeadDim { got, expected } => {
                write!(f, "turboquant: head_dim must be {expected}, got {got}")
            }
            Self::InvalidBlockCount {
                scheme,
                got,
                expected,
            } => write!(
                f,
                "{scheme}: block count must be {expected} per head, got {got}"
            ),
            Self::Unsupported { operation, reason } => {
                write!(f, "{operation} unsupported: {reason}")
            }
        }
    }
}

impl std::error::Error for Error {}
