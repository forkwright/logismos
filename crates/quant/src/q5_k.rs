//! Original CPU-only GGML `Q5_K` block decoding.

use crate::error::InvalidKBlockLengthSnafu;
use crate::k::{K_GROUP_VALUES, K_SCALE_BYTES, K_VALUES_PER_BLOCK, finite_half, scale_min};
use crate::row::{self, Geometry};
use crate::{Result, RowFormat};

/// Values represented by one `Q5_K` block.
pub const Q5_K_VALUES_PER_BLOCK: usize = K_VALUES_PER_BLOCK;
/// Bytes storing the fp16 super-scale and fp16 minimum scale.
pub const Q5_K_PREFIX_BYTES: usize = 4;
/// Bytes storing packed six-bit scale/minimum pairs.
pub const Q5_K_SCALE_BYTES: usize = K_SCALE_BYTES;
/// Bytes storing fifth-bit planes.
pub const Q5_K_HIGH_BITS_BYTES: usize = Q5_K_VALUES_PER_BLOCK / 8;
/// Bytes storing low four-bit values.
pub const Q5_K_QUANT_BYTES: usize = Q5_K_VALUES_PER_BLOCK / 2;
/// Exact serialized width of a `Q5_K` block.
pub const Q5_K_BLOCK_BYTES: usize =
    Q5_K_PREFIX_BYTES + Q5_K_SCALE_BYTES + Q5_K_HIGH_BITS_BYTES + Q5_K_QUANT_BYTES;

const GEOMETRY: Geometry = Geometry {
    format: RowFormat::Q5K,
    bytes_per_block: Q5_K_BLOCK_BYTES,
};

/// One validated GGML `Q5_K` block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Q5KBlock {
    bytes: [u8; Q5_K_BLOCK_BYTES],
}

impl Q5KBlock {
    /// Parse exactly one `Q5_K` block from serialized bytes.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] for a malformed block length or non-finite
    /// fp16 super-scale field.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != Q5_K_BLOCK_BYTES {
            return InvalidKBlockLengthSnafu {
                format: RowFormat::Q5K,
                actual: bytes.len(),
                expected: Q5_K_BLOCK_BYTES,
            }
            .fail();
        }
        let mut stored = [0; Q5_K_BLOCK_BYTES];
        stored.copy_from_slice(bytes);
        let _ = finite_half(RowFormat::Q5K, "scale", [stored[0], stored[1]])?;
        let _ = finite_half(RowFormat::Q5K, "minimum scale", [stored[2], stored[3]])?;
        Ok(Self { bytes: stored })
    }

    /// Decode this block into 256 f32 values.
    #[must_use]
    pub fn decode_f32(&self) -> [f32; Q5_K_VALUES_PER_BLOCK] {
        let super_scale =
            half::f16::from_bits(u16::from_le_bytes([self.bytes[0], self.bytes[1]])).to_f32();
        let super_minimum =
            half::f16::from_bits(u16::from_le_bytes([self.bytes[2], self.bytes[3]])).to_f32();
        let mut scales = [0; K_SCALE_BYTES];
        scales.copy_from_slice(&self.bytes[Q5_K_PREFIX_BYTES..Q5_K_PREFIX_BYTES + K_SCALE_BYTES]);
        let high_bits_start = Q5_K_PREFIX_BYTES + K_SCALE_BYTES;
        let high_bits = &self.bytes[high_bits_start..high_bits_start + Q5_K_HIGH_BITS_BYTES];
        let quantized = &self.bytes[high_bits_start + Q5_K_HIGH_BITS_BYTES..];
        let mut decoded = [0.0; Q5_K_VALUES_PER_BLOCK];
        for pair in 0..(Q5_K_VALUES_PER_BLOCK / (K_GROUP_VALUES * 2)) {
            let (scale_low, minimum_low) = scale_min(&scales, pair * 2);
            let (scale_high, minimum_high) = scale_min(&scales, pair * 2 + 1);
            let low_scale = super_scale * f32::from(scale_low);
            let low_minimum = super_minimum * f32::from(minimum_low);
            let high_scale = super_scale * f32::from(scale_high);
            let high_minimum = super_minimum * f32::from(minimum_high);
            let low_bit = 1_u8 << (pair * 2);
            let high_bit = low_bit << 1;
            for lane in 0..K_GROUP_VALUES {
                let packed = quantized[pair * K_GROUP_VALUES + lane];
                let fifth = high_bits[lane];
                let low = (packed & 0x0f) + if fifth & low_bit == 0 { 0 } else { 16 };
                let high = (packed >> 4) + if fifth & high_bit == 0 { 0 } else { 16 };
                decoded[pair * K_GROUP_VALUES * 2 + lane] =
                    low_scale * f32::from(low) - low_minimum;
                decoded[pair * K_GROUP_VALUES * 2 + K_GROUP_VALUES + lane] =
                    high_scale * f32::from(high) - high_minimum;
            }
        }
        decoded
    }
}

/// Derive the checked serialized length of one complete `Q5_K` row.
///
/// # Errors
///
/// Returns [`crate::Error`] when `value_count` is empty, not block-aligned,
/// or overflows its serialized representation.
pub fn row_byte_len(value_count: usize) -> Result<usize> {
    row::byte_len::<Q5_K_VALUES_PER_BLOCK>(GEOMETRY, value_count)
}
/// Compute a sequential f32 dot product for one serialized `Q5_K` row.
///
/// # Errors
///
/// Returns [`crate::Error`] for invalid geometry, encoded data, non-finite
/// inputs, products, or running accumulator.
pub fn row_dot_f32(serialized_row: &[u8], activations: &[f32]) -> Result<f32> {
    row::dot(GEOMETRY, serialized_row, activations, |block| {
        Ok(Q5KBlock::parse(block)?.decode_f32())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    #[test]
    fn decodes_independently_packed_fifth_bits() -> Result<()> {
        let mut bytes = [0; Q5_K_BLOCK_BYTES];
        bytes[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        bytes[2..4].copy_from_slice(&0x3c00_u16.to_le_bytes());
        bytes[4..8].fill(1);
        bytes[8..12].fill(1);
        bytes[12..16].fill(0x11);
        bytes[16..48].fill(0x03);
        bytes[48..].fill(0x21);
        let decoded = Q5KBlock::parse(&bytes)?.decode_f32();
        assert!(
            decoded[..K_GROUP_VALUES]
                .iter()
                .all(|value| value.to_bits() == 16.0_f32.to_bits()),
            "low fifth-bit plane must extend the first 32-value group"
        );
        assert!(
            decoded[K_GROUP_VALUES..K_GROUP_VALUES * 2]
                .iter()
                .all(|value| value.to_bits() == 17.0_f32.to_bits()),
            "high fifth-bit plane must extend the second 32-value group"
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_length_and_nonfinite_scale() {
        assert!(
            matches!(
                Q5KBlock::parse(&[0; Q5_K_BLOCK_BYTES - 1]),
                Err(Error::InvalidKBlockLength {
                    format: RowFormat::Q5K,
                    ..
                })
            ),
            "Q5_K parser must reject incomplete blocks"
        );
        let mut bytes = [0; Q5_K_BLOCK_BYTES];
        bytes[2..4].copy_from_slice(&0xfc00_u16.to_le_bytes());
        assert!(
            matches!(
                Q5KBlock::parse(&bytes),
                Err(Error::NonFiniteKScale {
                    format: RowFormat::Q5K,
                    ..
                })
            ),
            "Q5_K parser must reject infinite minimum scales"
        );
    }
}
