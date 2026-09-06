//! Original CPU-only GGML `Q8_0` block decoding.

use half::f16;

use crate::Result;
use crate::error::{
    EmptyQ8RowInputSnafu, InvalidQ8BlockLengthSnafu, InvalidQ8RowInputLengthSnafu,
    NonFiniteQ8ActivationSnafu, NonFiniteQ8ArithmeticSnafu, NonFiniteQ8ScaleSnafu,
    Q8RowByteLengthMismatchSnafu, Q8RowByteLengthOverflowSnafu,
};

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

/// Compute one serialized `Q8_0` row dotted with its f32 activation row.
///
/// The activation count establishes the row width: it must be positive and a
/// multiple of [`Q8_0_VALUES_PER_BLOCK`]. `serialized_row` must contain the
/// exact checked number of `Q8_0` blocks for that width. The operation reparses
/// each block through [`Q8_0Block::parse`], uses a single stack-resident
/// decoded block, and returns no output on a refusal.
///
/// WHY: a later artifact-bound tensor capability can lend one validated `Q8_0`
/// row without materializing a full f32 matrix; this keeps GGML block semantics
/// in one CPU-only authority rather than duplicating a dequantization loop in
/// model code.
///
/// # Errors
///
/// Returns [`crate::Error`] when the row geometry is invalid, a `Q8_0` scale or
/// activation is non-finite, or a product or running accumulator is non-finite.
pub fn row_dot_f32(serialized_row: &[u8], activations: &[f32]) -> Result<f32> {
    let expected_bytes = checked_row_byte_len(activations.len())?;
    if serialized_row.len() != expected_bytes {
        return Q8RowByteLengthMismatchSnafu {
            actual: serialized_row.len(),
            expected: expected_bytes,
        }
        .fail();
    }

    let mut accumulator = 0.0_f32;
    for (block_index, (serialized_block, activation_block)) in serialized_row
        .chunks_exact(Q8_0_BLOCK_BYTES)
        .zip(activations.chunks_exact(Q8_0_VALUES_PER_BLOCK))
        .enumerate()
    {
        let decoded_block = Q8_0Block::parse(serialized_block)?.decode_f32();
        for (lane_index, (weight, activation)) in
            decoded_block.iter().zip(activation_block).enumerate()
        {
            if !activation.is_finite() {
                return NonFiniteQ8ActivationSnafu {
                    index: block_index * Q8_0_VALUES_PER_BLOCK + lane_index,
                }
                .fail();
            }
            let product = *weight * *activation;
            if !product.is_finite() {
                return NonFiniteQ8ArithmeticSnafu {
                    stage: "product",
                    block_index,
                    lane_index,
                }
                .fail();
            }
            accumulator += product;
            if !accumulator.is_finite() {
                return NonFiniteQ8ArithmeticSnafu {
                    stage: "accumulation",
                    block_index,
                    lane_index,
                }
                .fail();
            }
        }
    }
    Ok(accumulator)
}

fn checked_row_byte_len(activation_len: usize) -> Result<usize> {
    if activation_len == 0 {
        return EmptyQ8RowInputSnafu.fail();
    }
    if !activation_len.is_multiple_of(Q8_0_VALUES_PER_BLOCK) {
        return InvalidQ8RowInputLengthSnafu {
            actual: activation_len,
            block_elements: Q8_0_VALUES_PER_BLOCK,
        }
        .fail();
    }
    let block_count = activation_len / Q8_0_VALUES_PER_BLOCK;
    block_count.checked_mul(Q8_0_BLOCK_BYTES).ok_or_else(|| {
        Q8RowByteLengthOverflowSnafu {
            block_count,
            block_bytes: Q8_0_BLOCK_BYTES,
        }
        .build()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    const ACCUMULATION_ACTIVATION_DIVISOR: f32 = 16_000_000.0_f32;

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

    #[test]
    fn row_dot_matches_independent_two_block_oracle() -> Result<()> {
        let first_values = [
            -128, -1, 0, 1, 127, -64, -32, -16, -8, -4, -2, 2, 4, 8, 16, 32, 64, -127, -3, 3, -5,
            5, -7, 7, -9, 9, -11, 11, -13, 13, -15, 15,
        ];
        let second_values = [
            127, -128, 1, -1, 0, 64, 32, 16, 8, 4, 2, -2, -4, -8, -16, -32, -64, 126, 3, -3, 5, -5,
            7, -7, 9, -9, 11, -11, 13, -13, 15, -15,
        ];
        let mut serialized = block_bytes(0x3555, first_values).to_vec();
        serialized.extend(block_bytes(0xbc00, second_values));
        let mut activations = [0.25_f32; Q8_0_VALUES_PER_BLOCK * 2];
        activations[Q8_0_VALUES_PER_BLOCK..].fill(-0.5_f32);

        let actual = row_dot_f32(&serialized, &activations)?;
        let expected = oracle_row_dot(&serialized, &activations);

        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "Q8_0 row dot must match the independent fp16/signed-byte oracle"
        );
        Ok(())
    }

    #[test]
    fn row_dot_accepts_negative_and_subnormal_scales() -> Result<()> {
        for scale_bits in [0x8001, 0x0001, 0xbc00] {
            let serialized = block_bytes(scale_bits, [1; Q8_0_VALUES_PER_BLOCK]);
            let activations = [1.0_f32; Q8_0_VALUES_PER_BLOCK];

            let actual = row_dot_f32(&serialized, &activations)?;
            let expected = oracle_row_dot(&serialized, &activations);
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "finite fp16 scale 0x{scale_bits:04x} must retain row-dot semantics"
            );
        }
        Ok(())
    }

    #[test]
    fn row_dot_rejects_geometry_without_mutating_inputs() {
        let serialized = block_bytes(0x3c00, [1; Q8_0_VALUES_PER_BLOCK]).to_vec();
        let original_serialized = serialized.clone();
        let empty: [f32; 0] = [];
        assert!(matches!(
            row_dot_f32(&serialized, &empty),
            Err(Error::EmptyQ8RowInput { .. })
        ));

        let unaligned_activations = [1.0_f32; Q8_0_VALUES_PER_BLOCK + 1];
        assert!(matches!(
            row_dot_f32(&serialized, &unaligned_activations),
            Err(Error::InvalidQ8RowInputLength { actual, .. })
                if actual == unaligned_activations.len()
        ));

        let short_serialized = &serialized[..Q8_0_BLOCK_BYTES - 1];
        let full_activations = [1.0_f32; Q8_0_VALUES_PER_BLOCK];
        assert!(matches!(
            row_dot_f32(short_serialized, &full_activations),
            Err(Error::Q8RowByteLengthMismatch { actual, expected, .. })
                if actual == short_serialized.len() && expected == serialized.len()
        ));
        assert_eq!(
            serialized, original_serialized,
            "row-dot refusal must not mutate bytes"
        );
    }

    #[test]
    fn row_dot_rejects_nonfinite_scale_and_activation() {
        let nonfinite_scale = block_bytes(0x7c00, [1; Q8_0_VALUES_PER_BLOCK]);
        let finite_activations = [1.0_f32; Q8_0_VALUES_PER_BLOCK];
        assert!(matches!(
            row_dot_f32(&nonfinite_scale, &finite_activations),
            Err(Error::NonFiniteQ8Scale { bits: 0x7c00, .. })
        ));

        let finite_scale = block_bytes(0x3c00, [1; Q8_0_VALUES_PER_BLOCK]);
        let mut nonfinite_activations = [1.0_f32; Q8_0_VALUES_PER_BLOCK];
        let nonfinite_index = 5;
        nonfinite_activations[nonfinite_index] = f32::NAN;
        assert!(matches!(
            row_dot_f32(&finite_scale, &nonfinite_activations),
            Err(Error::NonFiniteQ8Activation { index, .. }) if index == nonfinite_index
        ));
    }

    #[test]
    fn row_dot_rejects_nonfinite_product_and_accumulator() {
        let maximum_scale = block_bytes(0x7bff, [127; Q8_0_VALUES_PER_BLOCK]);
        let maximum_activations = [f32::MAX; Q8_0_VALUES_PER_BLOCK];
        assert!(matches!(
            row_dot_f32(&maximum_scale, &maximum_activations),
            Err(Error::NonFiniteQ8Arithmetic {
                stage: "product",
                block_index: 0,
                lane_index: 0,
                ..
            })
        ));

        let mut serialized = block_bytes(0x7bff, [0; Q8_0_VALUES_PER_BLOCK]).to_vec();
        serialized[Q8_0_SCALE_BYTES] = 127_u8;
        let mut second_block = block_bytes(0x7bff, [0; Q8_0_VALUES_PER_BLOCK]);
        second_block[Q8_0_SCALE_BYTES] = 127_u8;
        serialized.extend(second_block);
        let mut activations = [0.0_f32; Q8_0_VALUES_PER_BLOCK * 2];
        let large_finite_activation = f32::MAX / ACCUMULATION_ACTIVATION_DIVISOR;
        activations[0] = large_finite_activation;
        activations[Q8_0_VALUES_PER_BLOCK] = large_finite_activation;
        assert!(matches!(
            row_dot_f32(&serialized, &activations),
            Err(Error::NonFiniteQ8Arithmetic {
                stage: "accumulation",
                block_index: 1,
                lane_index: 0,
                ..
            })
        ));
    }

    #[test]
    fn row_dot_rejects_unrepresentable_serialized_length() {
        let max_multiple = usize::MAX - (usize::MAX % Q8_0_VALUES_PER_BLOCK);
        assert!(matches!(
            checked_row_byte_len(max_multiple),
            Err(Error::Q8RowByteLengthOverflow { .. })
        ));
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

    fn oracle_row_dot(serialized: &[u8], activations: &[f32]) -> f32 {
        const I8_SIGN_BIT: u8 = 0x80;
        const I8_MODULUS: i16 = 256;

        let mut accumulator = 0.0_f32;
        for block_index in 0..serialized.len() / Q8_0_BLOCK_BYTES {
            let block_start = block_index * Q8_0_BLOCK_BYTES;
            let scale_bytes = [serialized[block_start], serialized[block_start + 1]];
            let scale = oracle_f16_to_f32(u16::from_le_bytes(scale_bytes));
            for lane_index in 0..Q8_0_VALUES_PER_BLOCK {
                let stored_value = serialized[block_start + Q8_0_SCALE_BYTES + lane_index];
                let signed_value = if stored_value & I8_SIGN_BIT == 0 {
                    i16::from(stored_value)
                } else {
                    i16::from(stored_value) - I8_MODULUS
                };
                let activation_index = block_index * Q8_0_VALUES_PER_BLOCK + lane_index;
                accumulator += scale * f32::from(signed_value) * activations[activation_index];
            }
        }
        accumulator
    }
}
