//! Original CPU-only GGML `Q4_K` block decoding.

use crate::error::InvalidKBlockLengthSnafu;
use crate::k::{K_GROUP_VALUES, K_SCALE_BYTES, K_VALUES_PER_BLOCK, finite_half, scale_min};
use crate::row::{self, Geometry};
use crate::{Result, RowFormat};

/// Values represented by one `Q4_K` block.
pub const Q4_K_VALUES_PER_BLOCK: usize = K_VALUES_PER_BLOCK;
/// Bytes storing the fp16 super-scale and fp16 minimum scale.
pub const Q4_K_PREFIX_BYTES: usize = 4;
/// Bytes storing the packed six-bit scale/minimum pairs.
pub const Q4_K_SCALE_BYTES: usize = K_SCALE_BYTES;
/// Bytes storing the two four-bit quantized values per byte.
pub const Q4_K_QUANT_BYTES: usize = Q4_K_VALUES_PER_BLOCK / 2;
/// Exact serialized width of a `Q4_K` block.
pub const Q4_K_BLOCK_BYTES: usize = Q4_K_PREFIX_BYTES + Q4_K_SCALE_BYTES + Q4_K_QUANT_BYTES;

const GEOMETRY: Geometry = Geometry {
    format: RowFormat::Q4K,
    bytes_per_block: Q4_K_BLOCK_BYTES,
};

/// One validated GGML `Q4_K` block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Q4KBlock {
    bytes: [u8; Q4_K_BLOCK_BYTES],
}

impl Q4KBlock {
    /// Parse exactly one `Q4_K` block from serialized bytes.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] for a malformed block length or non-finite
    /// fp16 super-scale field.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != Q4_K_BLOCK_BYTES {
            return InvalidKBlockLengthSnafu {
                format: RowFormat::Q4K,
                actual: bytes.len(),
                expected: Q4_K_BLOCK_BYTES,
            }
            .fail();
        }
        let mut stored = [0; Q4_K_BLOCK_BYTES];
        stored.copy_from_slice(bytes);
        let _ = finite_half(RowFormat::Q4K, "scale", [stored[0], stored[1]])?;
        let _ = finite_half(RowFormat::Q4K, "minimum scale", [stored[2], stored[3]])?;
        Ok(Self { bytes: stored })
    }

    /// Decode this block into 256 f32 values.
    #[must_use]
    pub fn decode_f32(&self) -> [f32; Q4_K_VALUES_PER_BLOCK] {
        let super_scale =
            half::f16::from_bits(u16::from_le_bytes([self.bytes[0], self.bytes[1]])).to_f32();
        let super_minimum =
            half::f16::from_bits(u16::from_le_bytes([self.bytes[2], self.bytes[3]])).to_f32();
        let mut scales = [0; K_SCALE_BYTES];
        scales.copy_from_slice(&self.bytes[Q4_K_PREFIX_BYTES..Q4_K_PREFIX_BYTES + K_SCALE_BYTES]);
        let quantized = &self.bytes[Q4_K_PREFIX_BYTES + K_SCALE_BYTES..];
        let mut decoded = [0.0; Q4_K_VALUES_PER_BLOCK];
        for pair in 0..(Q4_K_VALUES_PER_BLOCK / (K_GROUP_VALUES * 2)) {
            let (scale_low, minimum_low) = scale_min(&scales, pair * 2);
            let (scale_high, minimum_high) = scale_min(&scales, pair * 2 + 1);
            let low_scale = super_scale * f32::from(scale_low);
            let low_minimum = super_minimum * f32::from(minimum_low);
            let high_scale = super_scale * f32::from(scale_high);
            let high_minimum = super_minimum * f32::from(minimum_high);
            for lane in 0..K_GROUP_VALUES {
                let packed = quantized[pair * K_GROUP_VALUES + lane];
                decoded[pair * K_GROUP_VALUES * 2 + lane] =
                    low_scale * f32::from(packed & 0x0f) - low_minimum;
                decoded[pair * K_GROUP_VALUES * 2 + K_GROUP_VALUES + lane] =
                    high_scale * f32::from(packed >> 4) - high_minimum;
            }
        }
        decoded
    }
}

/// Derive the checked serialized length of one complete `Q4_K` row.
///
/// # Errors
///
/// Returns [`crate::Error`] when `value_count` is empty, not block-aligned,
/// or overflows its serialized representation.
pub fn row_byte_len(value_count: usize) -> Result<usize> {
    row::byte_len::<Q4_K_VALUES_PER_BLOCK>(GEOMETRY, value_count)
}

/// Compute a sequential f32 dot product for one serialized `Q4_K` row.
///
/// # Errors
///
/// Returns [`crate::Error`] for invalid geometry, encoded data, non-finite
/// inputs, products, or running accumulator.
pub fn row_dot_f32(serialized_row: &[u8], activations: &[f32]) -> Result<f32> {
    row::dot(GEOMETRY, serialized_row, activations, |block| {
        Ok(Q4KBlock::parse(block)?.decode_f32())
    })
}

pub(crate) fn row_decode_f32(serialized_row: &[u8], value_count: usize) -> Result<Vec<f32>> {
    row::decode(GEOMETRY, serialized_row, value_count, |block| {
        Ok(Q4KBlock::parse(block)?.decode_f32())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    #[test]
    fn decodes_independently_packed_nibbles_and_scale_pairs() -> Result<()> {
        let mut bytes = [0; Q4_K_BLOCK_BYTES];
        bytes[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        bytes[2..4].copy_from_slice(&0x3c00_u16.to_le_bytes());
        bytes[4..8].fill(1);
        bytes[8..12].fill(1);
        bytes[12..16].fill(0x11);
        bytes[16..].fill(0x21);
        let decoded = Q4KBlock::parse(&bytes)?.decode_f32();
        for group in decoded.chunks_exact(K_GROUP_VALUES * 2) {
            assert!(
                group[..K_GROUP_VALUES]
                    .iter()
                    .all(|value| value.to_bits() == 0.0_f32.to_bits()),
                "low nibbles must use their paired affine minimum"
            );
            assert!(
                group[K_GROUP_VALUES..]
                    .iter()
                    .all(|value| value.to_bits() == 1.0_f32.to_bits()),
                "high nibbles must use their paired affine minimum"
            );
        }
        Ok(())
    }

    #[test]
    fn rejects_invalid_length_and_nonfinite_scale() {
        assert!(
            matches!(
                Q4KBlock::parse(&[0; Q4_K_BLOCK_BYTES - 1]),
                Err(Error::InvalidKBlockLength {
                    format: RowFormat::Q4K,
                    ..
                })
            ),
            "Q4_K parser must reject incomplete blocks"
        );
        let mut bytes = [0; Q4_K_BLOCK_BYTES];
        bytes[..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
        assert!(
            matches!(
                Q4KBlock::parse(&bytes),
                Err(Error::NonFiniteKScale {
                    format: RowFormat::Q4K,
                    ..
                })
            ),
            "Q4_K parser must reject infinite scales"
        );
    }
}
