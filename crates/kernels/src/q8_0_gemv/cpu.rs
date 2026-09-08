//! Checked CPU reference for raw Q8_0 GEMV.

use snafu::ResultExt;

use crate::Result;
use crate::error::Q8GemvAllocationSnafu;
use crate::q8_0_gemv::Q8GemvShape;

/// Compute `matrix[rows, width] * activations[width]` from raw Q8_0 rows.
///
/// Each output row delegates to [`quant::q8_0::row_dot_f32`], preserving the
/// format owner's finite checks and sequential scale, product, and
/// accumulation semantics.
///
/// # Errors
///
/// Returns [`crate::Error`] when the checked Q8_0 shape, serialized row,
/// activations, arithmetic, or output allocation is invalid.
pub fn q8_0_gemv_f32(
    matrix_q8_0: &[u8],
    activations: &[f32],
    shape: Q8GemvShape,
) -> Result<Vec<f32>> {
    if matrix_q8_0.len() != shape.matrix_bytes() || activations.len() != shape.width() {
        return crate::error::UnsupportedShapeSnafu {
            kernel: "q8_0_gemv_f32",
            msg: "buffer extents differ from the checked Q8_0 GEMV shape".to_string(),
        }
        .fail();
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(shape.rows())
        .context(Q8GemvAllocationSnafu {
            requested_len: shape.rows(),
        })?;
    for row in matrix_q8_0.chunks_exact(shape.row_bytes()) {
        output.push(quant::q8_0::row_dot_f32(row, activations).context(crate::error::QuantSnafu)?);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::q8_0_gemv_f32;

    const VALUES_PER_BLOCK: usize = quant::q8_0::Q8_0_VALUES_PER_BLOCK;
    const SCALE_BYTES: usize = quant::q8_0::Q8_0_SCALE_BYTES;

    #[test]
    fn q8_0_gemv_matches_independent_f64_packed_matrix_oracle() -> std::result::Result<(), String> {
        let first = signed_ramp(-64, 3);
        let second = signed_ramp(47, -2);
        let third = signed_ramp(-13, 5);
        let fourth = signed_ramp(91, -4);
        let mut matrix = Vec::new();
        matrix.extend(block(0x3555, first));
        matrix.extend(block(0xbc00, second));
        matrix.extend(block(0x3c00, third));
        matrix.extend(block(0x3800, fourth));
        let activations: Vec<f32> = (0..VALUES_PER_BLOCK * 2)
            .map(|index| (index as f32 - 19.0) / 11.0)
            .collect();

        let shape =
            super::Q8GemvShape::new(2, VALUES_PER_BLOCK * 2, matrix.len(), activations.len(), 2)
                .map_err(|error| error.to_string())?;
        let actual =
            q8_0_gemv_f32(&matrix, &activations, shape).map_err(|error| error.to_string())?;
        let expected = oracle_matrix(&matrix, 2, VALUES_PER_BLOCK * 2, &activations)?;
        assert_close(&actual, &expected, "asymmetric multi-block Q8_0 GEMV");
        Ok(())
    }

    #[test]
    fn q8_0_gemv_preserves_finite_zero_negative_and_subnormal_scales()
    -> std::result::Result<(), String> {
        let rows = 3;
        let mut matrix = Vec::new();
        for (bits, value) in [(0x0000, 127), (0xbc00, 3), (0x0001, 1)] {
            matrix.extend(block(bits, [value; VALUES_PER_BLOCK]));
        }
        let activations = [0.25_f32; VALUES_PER_BLOCK];
        let shape = super::Q8GemvShape::new(
            rows,
            VALUES_PER_BLOCK,
            matrix.len(),
            activations.len(),
            rows,
        )
        .map_err(|error| error.to_string())?;
        let actual =
            q8_0_gemv_f32(&matrix, &activations, shape).map_err(|error| error.to_string())?;
        let expected = oracle_matrix(&matrix, rows, VALUES_PER_BLOCK, &activations)?;
        assert_close(&actual, &expected, "finite special Q8_0 scales");
        if actual[0] != 0.0 || actual[1] >= 0.0 || actual[2] <= 0.0 {
            return Err("Q8_0 GEMV lost zero, negative, or subnormal scale signs".to_string());
        }
        Ok(())
    }

    #[test]
    fn q8_0_gemv_preserves_negative_i8_minimum_and_negative_zero_scale()
    -> std::result::Result<(), String> {
        let mut values = [0_i8; VALUES_PER_BLOCK];
        values[0] = i8::MIN;
        values[1] = -1;
        values[2] = 127;
        let matrix = block(0x8000, values);
        let activations = [1.0_f32; VALUES_PER_BLOCK];
        let shape =
            super::Q8GemvShape::new(1, VALUES_PER_BLOCK, matrix.len(), activations.len(), 1)
                .map_err(|error| error.to_string())?;
        let actual =
            q8_0_gemv_f32(&matrix, &activations, shape).map_err(|error| error.to_string())?;
        if actual != [0.0] {
            return Err(format!(
                "negative-zero Q8_0 scale should remain a finite zero, got {actual:?}"
            ));
        }
        Ok(())
    }

    #[test]
    fn q8_0_gemv_propagates_quant_finite_refusals() -> std::result::Result<(), String> {
        let matrix = block(0x7bff, [127; VALUES_PER_BLOCK]);
        let shape = super::Q8GemvShape::new(1, VALUES_PER_BLOCK, matrix.len(), VALUES_PER_BLOCK, 1)
            .map_err(|error| error.to_string())?;

        assert!(matches!(
            q8_0_gemv_f32(&matrix, &[f32::NAN; VALUES_PER_BLOCK], shape),
            Err(crate::Error::Quant {
                source: quant::Error::NonFiniteRowActivation { .. }
            })
        ));
        assert!(matches!(
            q8_0_gemv_f32(&matrix, &[f32::MAX; VALUES_PER_BLOCK], shape),
            Err(crate::Error::Quant {
                source: quant::Error::NonFiniteRowArithmetic {
                    stage: quant::RowArithmeticStage::Product,
                    ..
                }
            })
        ));

        let mut accumulation_activations = [0.0_f32; VALUES_PER_BLOCK];
        accumulation_activations[..2].fill(2.1e31_f32);
        assert!(matches!(
            q8_0_gemv_f32(&matrix, &accumulation_activations, shape),
            Err(crate::Error::Quant {
                source: quant::Error::NonFiniteRowArithmetic {
                    stage: quant::RowArithmeticStage::Accumulation,
                    ..
                }
            })
        ));
        Ok(())
    }

    fn block(scale_bits: u16, values: [i8; VALUES_PER_BLOCK]) -> Vec<u8> {
        let mut bytes = scale_bits.to_le_bytes().to_vec();
        bytes.extend(values.into_iter().map(|value| value as u8));
        bytes
    }

    fn signed_ramp(start: i8, step: i8) -> [i8; VALUES_PER_BLOCK] {
        std::array::from_fn(|index| start.wrapping_add(step.wrapping_mul(index as i8)))
    }

    fn oracle_matrix(
        matrix: &[u8],
        rows: usize,
        width: usize,
        activations: &[f32],
    ) -> std::result::Result<Vec<f64>, String> {
        let row_bytes = quant::q8_0::row_byte_len(width).map_err(|error| error.to_string())?;
        let mut output = Vec::with_capacity(rows);
        for row in matrix.chunks_exact(row_bytes) {
            let mut total = 0.0_f64;
            for (block_index, block_bytes) in
                row.chunks_exact(SCALE_BYTES + VALUES_PER_BLOCK).enumerate()
            {
                let bits = u16::from_le_bytes([block_bytes[0], block_bytes[1]]);
                let scale = oracle_f16(bits)?;
                for lane in 0..VALUES_PER_BLOCK {
                    let value = i8::from_le_bytes([block_bytes[SCALE_BYTES + lane]]);
                    let activation = activations[block_index * VALUES_PER_BLOCK + lane];
                    total += scale * f64::from(value) * f64::from(activation);
                }
            }
            output.push(total);
        }
        Ok(output)
    }

    fn oracle_f16(bits: u16) -> std::result::Result<f64, String> {
        let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
        let exponent = i32::from((bits >> 10) & 0x1f);
        let fraction = u32::from(bits & 0x03ff);
        match exponent {
            0 => Ok(sign * f64::from(fraction) * 2_f64.powi(-24)),
            31 => Err("oracle fixture used a non-finite fp16 scale".to_string()),
            _ => Ok(sign * (1.0 + f64::from(fraction) / 1024.0) * 2_f64.powi(exponent - 15)),
        }
    }

    fn assert_close(actual: &[f32], expected: &[f64], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 1e-4_f64.max(expected.abs() * 1e-6);
            assert!(
                (f64::from(*actual) - expected).abs() <= tolerance,
                "{label} row {index}: got {actual}, expected {expected}"
            );
        }
    }
}
