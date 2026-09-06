//! Errors surfaced by `quant` preflight and block-decoding utilities.

use core::fmt;

use crate::RowFormat;
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

/// The non-finite arithmetic phase in a serialized row dot product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RowArithmeticStage {
    /// A decoded weight multiplied by its activation was non-finite.
    Product,
    /// Adding a finite product to the running row total was non-finite.
    Accumulation,
}

impl fmt::Display for RowArithmeticStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Product => formatter.write_str("product"),
            Self::Accumulation => formatter.write_str("accumulation"),
        }
    }
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

    /// A row-dot operation received no activation values.
    #[snafu(display("{format} row dot requires at least one activation value"))]
    EmptyRowInput {
        /// Executable format selected for the row.
        format: RowFormat,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A row width cannot form whole blocks for its selected format.
    #[snafu(display(
        "{format} row dot activation length {actual} is not a multiple of block width {block_elements}"
    ))]
    InvalidRowInputLength {
        /// Executable format selected for the row.
        format: RowFormat,
        /// Supplied activation count.
        actual: usize,
        /// Values represented by one serialized block.
        block_elements: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Computing a serialized row length overflowed `usize`.
    #[snafu(display(
        "{format} row serialized length overflows: {block_count} blocks of {block_bytes} bytes"
    ))]
    RowByteLengthOverflow {
        /// Executable format selected for the row.
        format: RowFormat,
        /// Number of serialized blocks implied by the logical width.
        block_count: usize,
        /// Serialized bytes in one block.
        block_bytes: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Serialized bytes did not exactly cover the requested activation row.
    #[snafu(display("{format} row dot needs {expected} serialized bytes, got {actual}"))]
    RowByteLengthMismatch {
        /// Executable format selected for the row.
        format: RowFormat,
        /// Supplied serialized byte count.
        actual: usize,
        /// Byte count implied by the activation row.
        expected: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A row-dot activation is NaN or infinite.
    #[snafu(display("{format} row dot activation at index {index} is not finite"))]
    NonFiniteRowActivation {
        /// Executable format selected for the row.
        format: RowFormat,
        /// Flat activation index in the row.
        index: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A product or running total became NaN or infinite.
    #[snafu(display(
        "{format} row dot {stage} is not finite at block {block_index}, lane {lane_index}"
    ))]
    NonFiniteRowArithmetic {
        /// Executable format selected for the row.
        format: RowFormat,
        /// Arithmetic phase that became non-finite.
        stage: RowArithmeticStage,
        /// Zero-based serialized block index.
        block_index: usize,
        /// Zero-based lane index inside the block.
        lane_index: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A K-quant block has the wrong serialized length.
    #[snafu(display("{format} block must be {expected} bytes, got {actual}"))]
    InvalidKBlockLength {
        /// K-quant format being parsed.
        format: RowFormat,
        /// Supplied byte length.
        actual: usize,
        /// Required byte length.
        expected: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A K-quant fp16 scale field is NaN or infinite.
    #[snafu(display("{format} block {field} has non-finite fp16 bits 0x{bits:04x}"))]
    NonFiniteKScale {
        /// K-quant format being parsed.
        format: RowFormat,
        /// Named fp16 field containing the invalid value.
        field: &'static str,
        /// Invalid fp16 bits in host byte order.
        bits: u16,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// F32 serialized bytes cannot form whole scalar values.
    #[snafu(display("f32 row has {actual} bytes, which is not a whole number of f32 values"))]
    InvalidF32RowLength {
        /// Supplied byte length.
        actual: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// An encoded f32 weight is NaN or infinite.
    #[snafu(display("f32 row value at index {index} is not finite"))]
    NonFiniteF32Weight {
        /// Flat f32 value index in the serialized row.
        index: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}
