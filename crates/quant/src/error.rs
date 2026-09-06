//! Errors surfaced by `quant` preflight and block-decoding utilities.

use crate::scheme::TurboQuantScheme;
use snafu::Snafu;

/// Crate-local result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors surfaced by `quant` preflight and block-decoding utilities.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum Error {
    /// Caller supplied a quantized index outside the scheme's codebook range.
    #[snafu(display("{scheme}: index at position {position} has value {value}, max {max}"))]
    IndexOutOfRange {
        /// Quantization scheme being packed.
        scheme: TurboQuantScheme,
        /// Index position in the 32-value block.
        position: usize,
        /// Supplied index value.
        value: u8,
        /// Maximum allowed index for the scheme.
        max: u8,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Caller supplied a value count that is not one 128-value head chunk.
    #[snafu(display("turboquant: head_dim must be {expected}, got {got}"))]
    InvalidHeadDim {
        /// Supplied scalar count.
        got: usize,
        /// Required scalar count.
        expected: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Caller supplied a block count that is not one 128-value head chunk.
    #[snafu(display("{scheme}: block count must be {expected} per head, got {got}"))]
    InvalidBlockCount {
        /// Quantization scheme being decoded.
        scheme: TurboQuantScheme,
        /// Supplied block count.
        got: usize,
        /// Required block count.
        expected: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The requested FWHT encode/decode path has not landed yet.
    #[snafu(display("{operation} unsupported: {reason}"))]
    Unsupported {
        /// Operation name.
        operation: &'static str,
        /// Reason the operation is not available.
        reason: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A `Q8_0` block has the wrong byte length.
    #[snafu(display("q8_0 block must be {expected} bytes, got {actual}"))]
    InvalidQ8BlockLength {
        /// Supplied byte length.
        actual: usize,
        /// Required byte length.
        expected: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A `Q8_0` block encodes a non-finite fp16 scale.
    #[snafu(display("q8_0 block scale has non-finite fp16 bits 0x{bits:04x}"))]
    NonFiniteQ8Scale {
        /// Little-endian fp16 scale bits as a host integer.
        bits: u16,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}
