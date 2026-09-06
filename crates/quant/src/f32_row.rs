//! Checked zero-copy little-endian f32 row access and arithmetic.

use crate::error::{InvalidF32RowLengthSnafu, NonFiniteF32WeightSnafu};
use crate::row::{self, Geometry};
use crate::{Result, RowFormat};

/// Bytes in one serialized little-endian f32 value.
pub const F32_ROW_VALUE_BYTES: usize = 4;

const GEOMETRY: Geometry = Geometry {
    format: RowFormat::F32,
    bytes_per_block: F32_ROW_VALUE_BYTES,
};

/// A validated, zero-copy view of finite little-endian f32 values.
#[derive(Clone, Copy, Debug)]
pub struct F32Row<'bytes> {
    bytes: &'bytes [u8],
}

impl<'bytes> F32Row<'bytes> {
    /// Validate serialized little-endian f32 values without allocating.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] if the byte length is not divisible by four
    /// or any encoded value is NaN or infinite.
    pub fn parse(bytes: &'bytes [u8]) -> Result<Self> {
        if !bytes.len().is_multiple_of(F32_ROW_VALUE_BYTES) {
            return InvalidF32RowLengthSnafu {
                actual: bytes.len(),
            }
            .fail();
        }
        for (index, encoded) in bytes.chunks_exact(F32_ROW_VALUE_BYTES).enumerate() {
            let value = decode_one(encoded);
            if !value.is_finite() {
                return NonFiniteF32WeightSnafu { index }.fail();
            }
        }
        Ok(Self { bytes })
    }

    /// Return the number of validated f32 values in this row.
    #[must_use]
    pub fn len(self) -> usize {
        self.bytes.len() / F32_ROW_VALUE_BYTES
    }

    /// Return whether this row contains no values.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.bytes.is_empty()
    }

    /// Read one already-validated f32 value by index.
    #[must_use]
    pub fn value(self, index: usize) -> Option<f32> {
        self.bytes
            .chunks_exact(F32_ROW_VALUE_BYTES)
            .nth(index)
            .map(decode_one)
    }
}

/// Derive the checked serialized length of one complete f32 row.
///
/// # Errors
///
/// Returns [`crate::Error`] when `value_count` is zero or byte-length
/// multiplication overflows `usize`.
pub fn row_byte_len(value_count: usize) -> Result<usize> {
    row::byte_len::<1>(GEOMETRY, value_count)
}
/// Compute a sequential f32 dot product for one serialized f32 row.
///
/// # Errors
///
/// Returns [`crate::Error`] for invalid geometry, non-finite serialized
/// weights or activations, and non-finite products or running accumulator.
pub fn row_dot_f32(serialized_row: &[u8], activations: &[f32]) -> Result<f32> {
    F32Row::parse(serialized_row)?;
    row::dot(GEOMETRY, serialized_row, activations, |encoded| {
        Ok([decode_one(encoded)])
    })
}

pub(crate) fn row_decode_f32(serialized_row: &[u8], value_count: usize) -> Result<Vec<f32>> {
    F32Row::parse(serialized_row)?;
    row::decode(GEOMETRY, serialized_row, value_count, |encoded| {
        Ok([decode_one(encoded)])
    })
}

fn decode_one(encoded: &[u8]) -> f32 {
    let mut bytes = [0; F32_ROW_VALUE_BYTES];
    bytes.copy_from_slice(encoded);
    f32::from_bits(u32::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Error, RowFormat, row_dot_f32 as dispatch_row_dot_f32};

    #[test]
    fn reads_finite_values_without_materializing_a_vector() -> Result<()> {
        let mut bytes = Vec::new();
        for value in [-1.5_f32, 0.0, 2.25] {
            bytes.extend(value.to_le_bytes());
        }
        let row = F32Row::parse(&bytes)?;
        assert_eq!(
            row.len(),
            3,
            "f32 row length must derive from serialized values"
        );
        assert_eq!(
            row.value(2),
            Some(2.25),
            "f32 row reads must preserve little-endian bits"
        );
        assert_eq!(
            row.value(3),
            None,
            "f32 row reads beyond the checked range must not panic"
        );
        Ok(())
    }

    #[test]
    fn rejects_partial_and_nonfinite_values() {
        assert!(
            matches!(
                F32Row::parse(&[0; 3]),
                Err(Error::InvalidF32RowLength { .. })
            ),
            "partial f32 values must be refused"
        );
        assert!(
            matches!(
                F32Row::parse(&f32::NAN.to_le_bytes()),
                Err(Error::NonFiniteF32Weight { index: 0, .. })
            ),
            "non-finite f32 weights must be refused"
        );
    }

    #[test]
    fn dispatcher_dots_finite_little_endian_weights_sequentially() -> Result<()> {
        let mut bytes = Vec::new();
        for weight in [1.5_f32, -2.0, 0.25] {
            bytes.extend(weight.to_le_bytes());
        }
        let result = dispatch_row_dot_f32(RowFormat::F32, &bytes, &[2.0, 3.0, 4.0])?;
        assert_eq!(
            result.to_bits(),
            (-2.0_f32).to_bits(),
            "the dispatcher must preserve the left-to-right f32 row-dot contract"
        );
        Ok(())
    }

    #[test]
    fn dispatcher_refuses_nonfinite_serialized_weight_before_arithmetic() {
        let mut bytes = Vec::new();
        bytes.extend(1.0_f32.to_le_bytes());
        bytes.extend(f32::INFINITY.to_le_bytes());
        assert!(
            matches!(
                dispatch_row_dot_f32(RowFormat::F32, &bytes, &[1.0, 1.0]),
                Err(Error::NonFiniteF32Weight { index: 1, .. })
            ),
            "F32 dispatcher must report the flat index of an infinite encoded weight"
        );
    }
}
