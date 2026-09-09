//! Original CPU-only GGML `Q6_K` block decoding.

use crate::error::InvalidKBlockLengthSnafu;
use crate::k::{K_VALUES_PER_BLOCK, finite_half};
use crate::row::{self, Geometry};
use crate::{Result, RowFormat};

/// Values represented by one `Q6_K` block.
pub const Q6_K_VALUES_PER_BLOCK: usize = K_VALUES_PER_BLOCK;
/// Bytes storing the low four-bit value planes.
pub const Q6_K_LOW_BITS_BYTES: usize = Q6_K_VALUES_PER_BLOCK / 2;
/// Bytes storing the high two-bit value planes.
pub const Q6_K_HIGH_BITS_BYTES: usize = Q6_K_VALUES_PER_BLOCK / 4;
/// Bytes storing one signed scale for every 16 values.
pub const Q6_K_SCALE_BYTES: usize = Q6_K_VALUES_PER_BLOCK / 16;
/// Bytes storing the fp16 super-scale.
pub const Q6_K_SUPER_SCALE_BYTES: usize = 2;
/// Exact serialized width of a `Q6_K` block.
pub const Q6_K_BLOCK_BYTES: usize =
    Q6_K_LOW_BITS_BYTES + Q6_K_HIGH_BITS_BYTES + Q6_K_SCALE_BYTES + Q6_K_SUPER_SCALE_BYTES;

/// Values represented by one `Q6_K` packed quarter.
pub const Q6_K_VALUES_PER_QUARTER: usize = 32;
/// Packed `Q6_K` quarters represented by one half block.
pub const Q6_K_QUARTERS_PER_HALF_BLOCK: usize = 4;
const Q6_K_LOW_BITS_OFFSET: usize = 0;
/// Offset of the high two-bit value planes.
pub const Q6_K_HIGH_BITS_OFFSET: usize = Q6_K_LOW_BITS_OFFSET + Q6_K_LOW_BITS_BYTES;
/// Offset of the signed scale bytes.
pub const Q6_K_SCALE_OFFSET: usize = Q6_K_HIGH_BITS_OFFSET + Q6_K_HIGH_BITS_BYTES;
/// Offset of the fp16 super-scale.
pub const Q6_K_SUPER_SCALE_OFFSET: usize = Q6_K_SCALE_OFFSET + Q6_K_SCALE_BYTES;
const GEOMETRY: Geometry = Geometry {
    format: RowFormat::Q6K,
    bytes_per_block: Q6_K_BLOCK_BYTES,
};

/// One validated GGML `Q6_K` block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Q6KBlock {
    bytes: [u8; Q6_K_BLOCK_BYTES],
}

impl Q6KBlock {
    /// Parse exactly one `Q6_K` block from serialized bytes.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] for a malformed block length or non-finite
    /// fp16 super-scale field.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != Q6_K_BLOCK_BYTES {
            return InvalidKBlockLengthSnafu {
                format: RowFormat::Q6K,
                actual: bytes.len(),
                expected: Q6_K_BLOCK_BYTES,
            }
            .fail();
        }
        let mut stored = [0; Q6_K_BLOCK_BYTES];
        stored.copy_from_slice(bytes);
        let _ = finite_half(
            RowFormat::Q6K,
            "scale",
            [
                stored[Q6_K_SUPER_SCALE_OFFSET],
                stored[Q6_K_SUPER_SCALE_OFFSET + 1],
            ],
        )?;
        Ok(Self { bytes: stored })
    }

    /// Decode this block into 256 f32 values.
    #[must_use]
    pub fn decode_f32(&self) -> [f32; Q6_K_VALUES_PER_BLOCK] {
        let low = &self.bytes[Q6_K_LOW_BITS_OFFSET..Q6_K_HIGH_BITS_OFFSET];
        let high = &self.bytes[Q6_K_HIGH_BITS_OFFSET..Q6_K_SCALE_OFFSET];
        let scales = &self.bytes[Q6_K_SCALE_OFFSET..Q6_K_SUPER_SCALE_OFFSET];
        let super_scale = half::f16::from_bits(u16::from_le_bytes([
            self.bytes[Q6_K_SUPER_SCALE_OFFSET],
            self.bytes[Q6_K_SUPER_SCALE_OFFSET + 1],
        ]))
        .to_f32();
        let mut decoded = [0.0; Q6_K_VALUES_PER_BLOCK];
        for half_block in 0..2 {
            let low_base = half_block * 64;
            let high_base = half_block * 32;
            let scale_base = half_block * 8;
            for lane in 0..Q6_K_VALUES_PER_QUARTER {
                let packed_high = high[high_base + lane];
                for quarter in 0..Q6_K_QUARTERS_PER_HALF_BLOCK {
                    let low_byte = low[low_base + (quarter % 2) * Q6_K_VALUES_PER_QUARTER + lane];
                    let lower = if quarter < 2 {
                        low_byte & 0x0f
                    } else {
                        low_byte >> 4
                    };
                    let upper = (packed_high >> (quarter * 2)) & 0x03;
                    let quantized = i16::from((upper << 4) | lower) - 32;
                    let scale_index = scale_base + quarter * 2 + lane / 16;
                    decoded[half_block * 128 + quarter * Q6_K_VALUES_PER_QUARTER + lane] =
                        super_scale
                            * f32::from(i8::from_le_bytes([scales[scale_index]]))
                            * f32::from(quantized);
                }
            }
        }
        decoded
    }
}

/// Derive the checked serialized length of one complete `Q6_K` row.
///
/// # Errors
///
/// Returns [`crate::Error`] when `value_count` is empty, not block-aligned,
/// or overflows its serialized representation.
pub fn row_byte_len(value_count: usize) -> Result<usize> {
    row::byte_len::<Q6_K_VALUES_PER_BLOCK>(GEOMETRY, value_count)
}
/// Compute a sequential f32 dot product for one serialized `Q6_K` row.
///
/// # Errors
///
/// Returns [`crate::Error`] for invalid geometry, encoded data, non-finite
/// inputs, products, or running accumulator.
pub fn row_dot_f32(serialized_row: &[u8], activations: &[f32]) -> Result<f32> {
    row::dot(GEOMETRY, serialized_row, activations, |block| {
        Ok(Q6KBlock::parse(block)?.decode_f32())
    })
}

pub(crate) fn row_decode_f32(serialized_row: &[u8], value_count: usize) -> Result<Vec<f32>> {
    row::decode(GEOMETRY, serialized_row, value_count, |block| {
        Ok(Q6KBlock::parse(block)?.decode_f32())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    #[test]
    fn decodes_independently_packed_two_bit_planes() -> Result<()> {
        let mut bytes = [0; Q6_K_BLOCK_BYTES];
        bytes[..Q6_K_LOW_BITS_BYTES].fill(0x10);
        bytes[Q6_K_LOW_BITS_BYTES..Q6_K_LOW_BITS_BYTES + Q6_K_HIGH_BITS_BYTES].fill(0xe4);
        let scales_start = Q6_K_LOW_BITS_BYTES + Q6_K_HIGH_BITS_BYTES;
        bytes[scales_start..scales_start + Q6_K_SCALE_BYTES].fill(1);
        bytes[scales_start + Q6_K_SCALE_BYTES..].copy_from_slice(&0x3c00_u16.to_le_bytes());
        let decoded = Q6KBlock::parse(&bytes)?.decode_f32();
        assert_eq!(
            decoded[0].to_bits(),
            (-32.0_f32).to_bits(),
            "first two-bit plane must produce signed -32"
        );
        assert_eq!(
            decoded[32].to_bits(),
            (-16.0_f32).to_bits(),
            "second two-bit plane must select its scale group"
        );
        assert_eq!(
            decoded[64].to_bits(),
            1.0_f32.to_bits(),
            "third two-bit plane must use the high nibble"
        );
        assert_eq!(
            decoded[96].to_bits(),
            17.0_f32.to_bits(),
            "fourth two-bit plane must retain both packed fields"
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_length_and_nonfinite_scale() {
        assert!(
            matches!(
                Q6KBlock::parse(&[0; Q6_K_BLOCK_BYTES - 1]),
                Err(Error::InvalidKBlockLength {
                    format: RowFormat::Q6K,
                    ..
                })
            ),
            "Q6_K parser must reject incomplete blocks"
        );
        let mut bytes = [0; Q6_K_BLOCK_BYTES];
        bytes[Q6_K_BLOCK_BYTES - 2..].copy_from_slice(&0x7e00_u16.to_le_bytes());
        assert!(
            matches!(
                Q6KBlock::parse(&bytes),
                Err(Error::NonFiniteKScale {
                    format: RowFormat::Q6K,
                    ..
                })
            ),
            "Q6_K parser must reject NaN super scales"
        );
    }
}
