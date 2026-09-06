//! Original CPU-only GGML `Q8_0` block decoding.

use half::f16;

use crate::Result;
use crate::error::{InvalidQ8BlockLengthSnafu, NonFiniteQ8ScaleSnafu};

/// Number of signed quantized values in one `Q8_0` block.
pub const Q8_0_VALUES_PER_BLOCK: usize = 32;

/// Stored byte width of the little-endian fp16 `Q8_0` scale.
pub const Q8_0_SCALE_BYTES: usize = 2;

/// Stored byte width of the direct signed `Q8_0` value payload.
pub const Q8_0_VALUE_BYTES: usize = Q8_0_VALUES_PER_BLOCK;

/// Total byte width of one `Q8_0` block.
pub const Q8_0_BLOCK_BYTES: usize = Q8_0_SCALE_BYTES + Q8_0_VALUE_BYTES;

/// One validated GGML `Q8_0` block.
///
/// The block retains a little-endian fp16 scale and 32 direct signed-i8 values.
/// It accepts finite scales including zero, negative, and subnormal values;
/// NaN and infinities are rejected at parse time.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Q8_0Block {
    scale_f16_le: [u8; Q8_0_SCALE_BYTES],
    values: [u8; Q8_0_VALUE_BYTES],
}

impl Q8_0Block {
    /// Parse exactly one `Q8_0` block from its serialized bytes.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::InvalidQ8BlockLength`] when `bytes` is not one
    /// block, or [`crate::Error::NonFiniteQ8Scale`] when its fp16 scale is NaN
    /// or infinite.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != Q8_0_BLOCK_BYTES {
            return InvalidQ8BlockLengthSnafu {
                actual: bytes.len(),
                expected: Q8_0_BLOCK_BYTES,
            }
            .fail();
        }

        let (scale_f16_le, values) = bytes.split_at(Q8_0_SCALE_BYTES);
        let mut stored_scale = [0; Q8_0_SCALE_BYTES];
        stored_scale.copy_from_slice(scale_f16_le);
        let scale_bits = u16::from_le_bytes(stored_scale);
        if !f16::from_bits(scale_bits).is_finite() {
            return NonFiniteQ8ScaleSnafu { bits: scale_bits }.fail();
        }

        let mut stored_values = [0; Q8_0_VALUE_BYTES];
        stored_values.copy_from_slice(values);
        Ok(Self {
            scale_f16_le: stored_scale,
            values: stored_values,
        })
    }

    /// Decode this block into 32 f32 values.
    #[must_use]
    pub fn decode_f32(&self) -> [f32; Q8_0_VALUES_PER_BLOCK] {
        let scale = f16::from_bits(u16::from_le_bytes(self.scale_f16_le)).to_f32();
        let mut decoded = [0.0; Q8_0_VALUES_PER_BLOCK];
        for (output, stored_value) in decoded.iter_mut().zip(self.values) {
            *output = scale * f32::from(i8::from_le_bytes([stored_value]));
        }
        decoded
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    #[test]
    fn geometry_is_derived_from_scale_and_value_counts() {
        assert_eq!(Q8_0_VALUES_PER_BLOCK, 32, "GGML Q8_0 stores 32 values");
        assert_eq!(Q8_0_SCALE_BYTES, 2, "GGML Q8_0 stores one fp16 scale");
        assert_eq!(
            Q8_0_VALUE_BYTES, Q8_0_VALUES_PER_BLOCK,
            "Q8_0 stores one byte for every signed value"
        );
        assert_eq!(
            Q8_0_BLOCK_BYTES,
            Q8_0_SCALE_BYTES + Q8_0_VALUE_BYTES,
            "Q8_0 block width derives from its two serialized fields"
        );
    }

    #[test]
    fn decodes_hand_encoded_signed_extrema() -> Result<()> {
        let values = [
            -128, -1, 0, 1, 127, -64, -32, -16, -8, -4, -2, 2, 4, 8, 16, 32, 64, -127, -3, 3, -5,
            5, -7, 7, -9, 9, -11, 11, -13, 13, -15, 15,
        ];
        let bytes = block_bytes(0x3800, values);
        let decoded = Q8_0Block::parse(&bytes)?.decode_f32();
        let expected: [f32; Q8_0_VALUES_PER_BLOCK] = [
            -64.0, -0.5, 0.0, 0.5, 63.5, -32.0, -16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0,
            16.0, 32.0, -63.5, -1.5, 1.5, -2.5, 2.5, -3.5, 3.5, -4.5, 4.5, -5.5, 5.5, -6.5, 6.5,
            -7.5, 7.5,
        ];
        for (actual, expected) in decoded.iter().zip(expected) {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "direct i8 values must retain their signs exactly"
            );
        }
        Ok(())
    }

    #[test]
    fn matches_independent_scale_and_signed_byte_oracle() -> Result<()> {
        let values = [
            -128, -120, -112, -104, -96, -88, -80, -72, -64, -56, -48, -40, -32, -24, -16, -8, 0,
            8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 127,
        ];
        let bytes = block_bytes(0x3555, values);
        let decoded = Q8_0Block::parse(&bytes)?.decode_f32();
        let scale = oracle_f16_to_f32(0x3555);

        for ((actual, expected_signed), expected_byte) in
            decoded.iter().zip(values).zip(bytes[2..].iter())
        {
            let expected = scale * f32::from(expected_signed);
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "decoded value must match the format oracle exactly"
            );
            assert_eq!(
                i8::from_le_bytes([*expected_byte]),
                expected_signed,
                "payload byte must decode as a direct signed i8"
            );
        }
        Ok(())
    }

    #[test]
    fn accepts_finite_zero_negative_and_subnormal_scales() -> Result<()> {
        for scale_bits in [0x0000, 0x8000, 0x8001, 0x0001, 0xbc00] {
            let decoded = Q8_0Block::parse(&block_bytes(scale_bits, [1; Q8_0_VALUES_PER_BLOCK]))?
                .decode_f32();
            assert!(
                decoded.iter().all(|value| value.is_finite()),
                "finite fp16 scale 0x{scale_bits:04x} must decode to finite f32 values"
            );
        }
        Ok(())
    }

    #[test]
    fn rejects_nonfinite_scales() {
        for scale_bits in [0x7c00, 0xfc00, 0x7e01] {
            let error = Q8_0Block::parse(&block_bytes(scale_bits, [0; Q8_0_VALUES_PER_BLOCK]));
            assert!(
                matches!(error, Err(Error::NonFiniteQ8Scale { bits, .. }) if bits == scale_bits),
                "non-finite fp16 scale 0x{scale_bits:04x} must be rejected"
            );
        }
    }

    #[test]
    fn rejects_wrong_lengths_without_mutating_input() {
        for length in [0, Q8_0_BLOCK_BYTES - 1, Q8_0_BLOCK_BYTES + 1] {
            let bytes = vec![0x5a; length];
            let original = bytes.clone();
            let error = Q8_0Block::parse(&bytes);
            assert!(
                matches!(error, Err(Error::InvalidQ8BlockLength { actual, expected, .. }) if actual == length && expected == Q8_0_BLOCK_BYTES),
                "length {length} must be rejected with the exact Q8_0 geometry"
            );
            assert_eq!(bytes, original, "parsing must not mutate caller bytes");
        }
    }

    fn block_bytes(scale_bits: u16, values: [i8; Q8_0_VALUES_PER_BLOCK]) -> [u8; Q8_0_BLOCK_BYTES] {
        let mut bytes = [0; Q8_0_BLOCK_BYTES];
        let (stored_scale, stored_values) = bytes.split_at_mut(Q8_0_SCALE_BYTES);
        stored_scale.copy_from_slice(&scale_bits.to_le_bytes());
        for (stored_value, value) in stored_values.iter_mut().zip(values) {
            *stored_value = value.to_le_bytes()[0];
        }
        bytes
    }

    fn oracle_f16_to_f32(bits: u16) -> f32 {
        let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
        let exponent = (bits >> 10) & 0x001f;
        let fraction = bits & 0x03ff;
        if exponent == 0 {
            return sign * f32::from(fraction) * 2.0_f32.powi(-24);
        }
        sign * (1.0 + f32::from(fraction) / 1024.0) * 2.0_f32.powi(i32::from(exponent) - 15)
    }
}
