//! Original CPU-only GGML `IQ4_XS` block decoding.

use half::f16;

use crate::error::InvalidIq4BlockLengthSnafu;
use crate::iq4::{RECONSTRUCTION_VALUES, SCALE_BYTES, finite_scale};
use crate::row::{self, Geometry};
use crate::{Result, RowFormat};

/// Values represented by one `IQ4_XS` block.
pub const IQ4_XS_VALUES_PER_BLOCK: usize = 256;
/// Bytes storing the fp16 block scale.
pub const IQ4_XS_SCALE_BYTES: usize = SCALE_BYTES;
/// Bytes storing the low four bits of the eight group scales.
pub const IQ4_XS_SCALE_LOW_BYTES: usize = 4;
/// Bytes storing the high two bits of the eight group scales.
pub const IQ4_XS_SCALE_HIGH_BYTES: usize = 2;
/// Bytes storing two four-bit reconstruction indices per byte.
pub const IQ4_XS_QUANT_BYTES: usize = IQ4_XS_VALUES_PER_BLOCK / 2;
/// Exact serialized width of an `IQ4_XS` block.
pub const IQ4_XS_BLOCK_BYTES: usize =
    IQ4_XS_SCALE_BYTES + IQ4_XS_SCALE_LOW_BYTES + IQ4_XS_SCALE_HIGH_BYTES + IQ4_XS_QUANT_BYTES;

const GROUP_VALUES: usize = 32;
const GROUP_COUNT: usize = IQ4_XS_VALUES_PER_BLOCK / GROUP_VALUES;
const QUANT_OFFSET: usize = IQ4_XS_SCALE_BYTES + IQ4_XS_SCALE_LOW_BYTES + IQ4_XS_SCALE_HIGH_BYTES;
const GEOMETRY: Geometry = Geometry {
    format: RowFormat::IQ4XS,
    bytes_per_block: IQ4_XS_BLOCK_BYTES,
};

/// One validated GGML `IQ4_XS` block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Iq4XsBlock {
    bytes: [u8; IQ4_XS_BLOCK_BYTES],
}

impl Iq4XsBlock {
    /// Parse exactly one `IQ4_XS` block from serialized bytes.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] for a malformed block length or non-finite
    /// fp16 scale field.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != IQ4_XS_BLOCK_BYTES {
            return InvalidIq4BlockLengthSnafu {
                format: RowFormat::IQ4XS,
                actual: bytes.len(),
                expected: IQ4_XS_BLOCK_BYTES,
            }
            .fail();
        }
        let mut stored = [0; IQ4_XS_BLOCK_BYTES];
        stored.copy_from_slice(bytes);
        let _ = finite_scale(RowFormat::IQ4XS, [stored[0], stored[1]])?;
        Ok(Self { bytes: stored })
    }

    /// Decode this block into 256 f32 values.
    #[must_use]
    pub fn decode_f32(&self) -> [f32; IQ4_XS_VALUES_PER_BLOCK] {
        let block_scale =
            f16::from_bits(u16::from_le_bytes([self.bytes[0], self.bytes[1]])).to_f32();
        let scale_low =
            &self.bytes[IQ4_XS_SCALE_BYTES..IQ4_XS_SCALE_BYTES + IQ4_XS_SCALE_LOW_BYTES];
        let scale_high = u16::from_le_bytes([
            self.bytes[IQ4_XS_SCALE_BYTES + IQ4_XS_SCALE_LOW_BYTES],
            self.bytes[IQ4_XS_SCALE_BYTES + IQ4_XS_SCALE_LOW_BYTES + 1],
        ]);
        let quantized = &self.bytes[QUANT_OFFSET..];
        let mut decoded = [0.0; IQ4_XS_VALUES_PER_BLOCK];
        for group in 0..GROUP_COUNT {
            let low_bits = if group.is_multiple_of(2) {
                scale_low[group / 2] & 0x0f
            } else {
                scale_low[group / 2] >> 4
            };
            let high_bits = ((scale_high >> (group * 2)) & 0x03) as u8;
            let group_scale = i16::from(low_bits | (high_bits << 4)) - 32;
            let value_scale = block_scale * f32::from(group_scale);
            let group_start = group * GROUP_VALUES;
            for lane in 0..(GROUP_VALUES / 2) {
                let packed = quantized[group * (GROUP_VALUES / 2) + lane];
                decoded[group_start + lane] =
                    value_scale * f32::from(RECONSTRUCTION_VALUES[usize::from(packed & 0x0f)]);
                decoded[group_start + GROUP_VALUES / 2 + lane] =
                    value_scale * f32::from(RECONSTRUCTION_VALUES[usize::from(packed >> 4)]);
            }
        }
        decoded
    }
}

/// Derive the checked serialized length of one complete `IQ4_XS` row.
///
/// # Errors
///
/// Returns [`crate::Error`] when `value_count` is empty, not block-aligned,
/// or overflows its serialized representation.
pub fn row_byte_len(value_count: usize) -> Result<usize> {
    row::byte_len::<IQ4_XS_VALUES_PER_BLOCK>(GEOMETRY, value_count)
}

/// Compute a sequential f32 dot product for one serialized `IQ4_XS` row.
///
/// # Errors
///
/// Returns [`crate::Error`] for invalid geometry, encoded data, non-finite
/// inputs, products, or running accumulator.
pub fn row_dot_f32(serialized_row: &[u8], activations: &[f32]) -> Result<f32> {
    row::dot(GEOMETRY, serialized_row, activations, |block| {
        Ok(Iq4XsBlock::parse(block)?.decode_f32())
    })
}

pub(crate) fn row_decode_f32(serialized_row: &[u8], value_count: usize) -> Result<Vec<f32>> {
    row::decode(GEOMETRY, serialized_row, value_count, |block| {
        Ok(Iq4XsBlock::parse(block)?.decode_f32())
    })
}
