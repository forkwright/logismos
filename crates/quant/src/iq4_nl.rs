//! Original CPU-only GGML `IQ4_NL` block decoding.

use half::f16;

use crate::error::InvalidIq4BlockLengthSnafu;
use crate::iq4::{RECONSTRUCTION_VALUES, SCALE_BYTES, finite_scale};
use crate::row::{self, Geometry};
use crate::{Result, RowFormat};

/// Values represented by one `IQ4_NL` block.
pub const IQ4_NL_VALUES_PER_BLOCK: usize = 32;
/// Bytes storing the fp16 block scale.
pub const IQ4_NL_SCALE_BYTES: usize = SCALE_BYTES;
/// Bytes storing two four-bit reconstruction indices per byte.
pub const IQ4_NL_QUANT_BYTES: usize = IQ4_NL_VALUES_PER_BLOCK / 2;
/// Exact serialized width of an `IQ4_NL` block.
pub const IQ4_NL_BLOCK_BYTES: usize = IQ4_NL_SCALE_BYTES + IQ4_NL_QUANT_BYTES;

const GEOMETRY: Geometry = Geometry {
    format: RowFormat::IQ4NL,
    bytes_per_block: IQ4_NL_BLOCK_BYTES,
};

/// One validated GGML `IQ4_NL` block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Iq4NlBlock {
    bytes: [u8; IQ4_NL_BLOCK_BYTES],
}

impl Iq4NlBlock {
    /// Parse exactly one `IQ4_NL` block from serialized bytes.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] for a malformed block length or non-finite
    /// fp16 scale field.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != IQ4_NL_BLOCK_BYTES {
            return InvalidIq4BlockLengthSnafu {
                format: RowFormat::IQ4NL,
                actual: bytes.len(),
                expected: IQ4_NL_BLOCK_BYTES,
            }
            .fail();
        }
        let mut stored = [0; IQ4_NL_BLOCK_BYTES];
        stored.copy_from_slice(bytes);
        let _ = finite_scale(RowFormat::IQ4NL, [stored[0], stored[1]])?;
        Ok(Self { bytes: stored })
    }

    /// Decode this block into 32 f32 values.
    #[must_use]
    pub fn decode_f32(&self) -> [f32; IQ4_NL_VALUES_PER_BLOCK] {
        let scale = f16::from_bits(u16::from_le_bytes([self.bytes[0], self.bytes[1]])).to_f32();
        let mut decoded = [0.0; IQ4_NL_VALUES_PER_BLOCK];
        for (lane, packed) in self.bytes[IQ4_NL_SCALE_BYTES..].iter().enumerate() {
            decoded[lane] = scale * f32::from(RECONSTRUCTION_VALUES[usize::from(*packed & 0x0f)]);
            decoded[lane + IQ4_NL_QUANT_BYTES] =
                scale * f32::from(RECONSTRUCTION_VALUES[usize::from(*packed >> 4)]);
        }
        decoded
    }
}

/// Derive the checked serialized length of one complete `IQ4_NL` row.
///
/// # Errors
///
/// Returns [`crate::Error`] when `value_count` is empty, not block-aligned,
/// or overflows its serialized representation.
pub fn row_byte_len(value_count: usize) -> Result<usize> {
    row::byte_len::<IQ4_NL_VALUES_PER_BLOCK>(GEOMETRY, value_count)
}

/// Compute a sequential f32 dot product for one serialized `IQ4_NL` row.
///
/// # Errors
///
/// Returns [`crate::Error`] for invalid geometry, encoded data, non-finite
/// inputs, products, or running accumulator.
pub fn row_dot_f32(serialized_row: &[u8], activations: &[f32]) -> Result<f32> {
    row::dot(GEOMETRY, serialized_row, activations, |block| {
        Ok(Iq4NlBlock::parse(block)?.decode_f32())
    })
}

pub(crate) fn row_decode_f32(serialized_row: &[u8], value_count: usize) -> Result<Vec<f32>> {
    row::decode(GEOMETRY, serialized_row, value_count, |block| {
        Ok(Iq4NlBlock::parse(block)?.decode_f32())
    })
}
