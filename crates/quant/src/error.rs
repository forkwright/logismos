//! Errors surfaced by `quant` preflight and block-decoding utilities.

use core::fmt;

use crate::scheme::TurboQuantScheme;
use snafu::Snafu;

/// Crate-local result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// The non-finite arithmetic phase in a `Q8_0` row dot product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Q8ArithmeticStage {
    /// A decoded weight multiplied by its activation was non-finite.
    Product,
    /// Adding a finite product to the running row total was non-finite.
    Accumulation,
}

impl fmt::Display for Q8ArithmeticStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Product => formatter.write_str("product"),
            Self::Accumulation => formatter.write_str("accumulation"),
        }
    }
}

/// Errors surfaced by `quant` preflight and block-decoding utilities.
#[derive(Debug, PartialEq, Eq, Snafu)]
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

    /// A `Q8_0` row-dot operation received no activation values.
    #[snafu(display("q8_0 row dot requires at least one activation value"))]
    EmptyQ8RowInput {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A `Q8_0` row-dot activation length cannot form whole `Q8_0` blocks.
    #[snafu(display(
        "q8_0 row dot activation length {actual} is not a multiple of block width {block_elements}"
    ))]
    InvalidQ8RowInputLength {
        /// Supplied activation count.
        actual: usize,
        /// Required values in each `Q8_0` block.
        block_elements: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Computing the serialized `Q8_0` row length overflowed `usize`.
    #[snafu(display(
        "q8_0 row dot serialized length overflows: {block_count} blocks of {block_bytes} bytes"
    ))]
    Q8RowByteLengthOverflow {
        /// Number of `Q8_0` blocks implied by the activation length.
        block_count: usize,
        /// Bytes in each serialized `Q8_0` block.
        block_bytes: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Serialized `Q8_0` bytes did not exactly cover the activation row.
    #[snafu(display("q8_0 row dot needs {expected} serialized bytes, got {actual}"))]
    Q8RowByteLengthMismatch {
        /// Supplied serialized byte count.
        actual: usize,
        /// Byte count implied by the activation row.
        expected: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A `Q8_0` row-dot activation is not finite.
    #[snafu(display("q8_0 row dot activation at index {index} is not finite"))]
    NonFiniteQ8Activation {
        /// Flat activation index within the row.
        index: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A `Q8_0` row-dot intermediate is not finite.
    #[snafu(display(
        "q8_0 row dot {stage} is not finite at block {block_index}, lane {lane_index}"
    ))]
    NonFiniteQ8Arithmetic {
        /// Arithmetic stage that overflowed or became non-finite.
        stage: Q8ArithmeticStage,
        /// Zero-based `Q8_0` block index.
        block_index: usize,
        /// Zero-based value index inside the `Q8_0` block.
        lane_index: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}
