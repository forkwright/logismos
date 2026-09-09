//! Checked CPU reference for raw serialized-row GEMV.

use snafu::ResultExt;

use crate::Result;
use crate::error::{QuantSnafu, UnsupportedShapeSnafu};
use crate::row_gemv::{KERNEL, RowGemvShape, reserve_output};

/// Compute `matrix[rows, width] * activations[width]` from serialized rows.
///
/// Each row delegates byte geometry, decoding, finite checks, and serial f32
/// arithmetic to the authoritative `quant` format owner.
///
/// # Errors
///
/// Returns [`crate::Error`] when checked extents differ from `shape`, or when
/// a serialized row, activation, product, or accumulation is invalid.
pub fn row_gemv_f32(matrix: &[u8], activations: &[f32], shape: RowGemvShape) -> Result<Vec<f32>> {
    if matrix.len() != shape.matrix_bytes() || activations.len() != shape.width() {
        return UnsupportedShapeSnafu {
            kernel: KERNEL,
            msg: "buffer extents differ from the checked serialized-row GEMV shape".to_string(),
        }
        .fail();
    }
    let mut output = reserve_output(shape.rows())?;
    for row in matrix.chunks_exact(shape.row_bytes()) {
        output.push(quant::row_dot_f32(shape.format(), row, activations).context(QuantSnafu)?);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::row_gemv_f32;
    use crate::row_gemv::RowGemvShape;

    #[test]
    fn every_executable_format_matches_an_independent_f64_serial_oracle()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        for format in formats() {
            let fixture = fixture(format)?;
            let shape = RowGemvShape::new(
                format,
                fixture.rows,
                fixture.width,
                fixture.matrix.len(),
                fixture.activations.len(),
                fixture.rows,
            )?;
            let actual = row_gemv_f32(&fixture.matrix, &fixture.activations, shape)?;
            let expected = oracle_matrix(
                format,
                &fixture.matrix,
                fixture.rows,
                fixture.width,
                &fixture.activations,
            )?;
            assert_close(&actual, &expected, format.to_string().as_str());
            assert_eq!(actual[0], 0.0_f32, "{format} zero row");
        }
        Ok(())
    }

    #[test]
    fn q8_finite_zero_negative_and_subnormal_scales_remain_cpu_reference_behavior()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let values = [1_i8; 32];
        let mut matrix = block_q8(0x0000, values);
        matrix.extend(block_q8(0xbc00, values));
        matrix.extend(block_q8(0x0001, values));
        let activations = vec![0.25_f32; 32];
        let shape = RowGemvShape::new(quant::RowFormat::Q8_0, 3, 32, matrix.len(), 32, 3)?;
        let actual = row_gemv_f32(&matrix, &activations, shape)?;
        assert!(actual[0] == 0.0 && actual[1] < 0.0 && actual[2] > 0.0);
        Ok(())
    }

    #[test]
    fn q8_i8_minimum_and_negative_zero_scale_remain_cpu_reference_behavior()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let mut values = [0_i8; 32];
        values[0] = i8::MIN;
        values[1] = -1;
        values[2] = 127;
        let matrix = block_q8(0x8000, values);
        let shape = RowGemvShape::new(quant::RowFormat::Q8_0, 1, 32, matrix.len(), 32, 1)?;
        let actual = row_gemv_f32(&matrix, &[1.0_f32; 32], shape)?;
        assert_eq!(actual, [0.0]);
        Ok(())
    }

    #[test]
    fn q8_nonfinite_input_product_and_accumulation_refusals_remain_visible()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let matrix = block_q8(0x7bff, [127_i8; 32]);
        let shape = RowGemvShape::new(quant::RowFormat::Q8_0, 1, 32, matrix.len(), 32, 1)?;
        assert!(matches!(
            row_gemv_f32(&matrix, &[f32::NAN; 32], shape),
            Err(crate::Error::Quant {
                source: quant::Error::NonFiniteRowActivation { .. },
                ..
            })
        ));
        assert!(matches!(
            row_gemv_f32(&matrix, &[f32::MAX; 32], shape),
            Err(crate::Error::Quant {
                source: quant::Error::NonFiniteRowArithmetic {
                    stage: quant::error::RowArithmeticStage::Product,
                    ..
                },
                ..
            })
        ));
        let mut accumulation = [0.0_f32; 32];
        accumulation[..2].fill(2.1e31_f32);
        assert!(matches!(
            row_gemv_f32(&matrix, &accumulation, shape),
            Err(crate::Error::Quant {
                source: quant::Error::NonFiniteRowArithmetic {
                    stage: quant::error::RowArithmeticStage::Accumulation,
                    ..
                },
                ..
            })
        ));
        Ok(())
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "requires an explicitly reserved HIP device; absent devices are a failure"]
    fn reserved_device_all_formats_match_cpu_reference_with_tail_rows()
    -> core::result::Result<(), String> {
        use hipcore::{Device, DeviceBuffer, Stream};

        let device = Device::new(0).map_err(|error| format!("open reserved device 0: {error}"))?;
        let stream = Stream::new(&device).map_err(|error| format!("create stream: {error}"))?;
        for format in formats() {
            let mut fixture = fixture(format).map_err(|error| error.to_string())?;
            let row_bytes = quant::row_byte_len(format, fixture.width)
                .map_err(|error| format!("derive {format} row bytes: {error}"))?;
            let tail = fixture.matrix[..row_bytes].to_vec();
            fixture.matrix.extend(tail);
            let rows = fixture.rows + 1;
            let shape = RowGemvShape::new(
                format,
                rows,
                fixture.width,
                fixture.matrix.len(),
                fixture.activations.len(),
                rows,
            )
            .map_err(|error| format!("validate {format} shape: {error}"))?;
            let expected = row_gemv_f32(&fixture.matrix, &fixture.activations, shape)
                .map_err(|error| format!("CPU {format} reference: {error}"))?;
            let matrix = DeviceBuffer::<u8>::from_host(&device, &fixture.matrix)
                .map_err(|error| format!("upload {format} matrix: {error}"))?;
            let activations = DeviceBuffer::<f32>::from_host(&device, &fixture.activations)
                .map_err(|error| format!("upload {format} activations: {error}"))?;
            let output = DeviceBuffer::<f32>::alloc(&device, rows)
                .map_err(|error| format!("allocate {format} output: {error}"))?;
            // SAFETY: distinct allocations match the checked exact extents and
            // remain alive and exclusively owned through synchronization.
            unsafe {
                crate::row_gemv::launch_row_gemv_f32(
                    shape,
                    matrix.as_device_ptr(),
                    matrix.len(),
                    activations.as_device_ptr(),
                    activations.len(),
                    output.as_device_ptr(),
                    output.len(),
                    &stream,
                )
            }
            .map_err(|error| format!("launch {format}: {error}"))?;
            stream
                .synchronize()
                .map_err(|error| format!("synchronize {format}: {error}"))?;
            let mut actual = vec![0.0_f32; rows];
            output
                .copy_to_host(&mut actual)
                .map_err(|error| format!("read {format} output: {error}"))?;
            assert_gpu_close(&actual, &expected, format.to_string().as_str())?;
        }
        Ok(())
    }

    fn formats() -> [quant::RowFormat; 7] {
        [
            quant::RowFormat::F32,
            quant::RowFormat::Q8_0,
            quant::RowFormat::Q4K,
            quant::RowFormat::Q5K,
            quant::RowFormat::Q6K,
            quant::RowFormat::IQ4NL,
            quant::RowFormat::IQ4XS,
        ]
    }

    struct Fixture {
        rows: usize,
        width: usize,
        matrix: Vec<u8>,
        activations: Vec<f32>,
    }

    fn fixture(
        format: quant::RowFormat,
    ) -> core::result::Result<Fixture, Box<dyn std::error::Error>> {
        let rows = 2;
        let (width, first, second) = match format {
            quant::RowFormat::F32 => (
                4,
                [0.0_f32; 4]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
                [-1.0_f32, 2.5, -0.75, 0.0]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
            ),
            quant::RowFormat::Q8_0 => (
                64,
                [
                    block_q8(0x0000, ramp_i8(-17, 3)),
                    block_q8(0x0000, ramp_i8(39, -2)),
                ]
                .concat(),
                [
                    block_q8(0x3800, ramp_i8(7, 5)),
                    block_q8(0x3c00, ramp_i8(-41, 4)),
                ]
                .concat(),
            ),
            quant::RowFormat::Q4K => (
                512,
                [
                    block_q4(0x0000, 0x0000, 0x51),
                    block_q4(0x0000, 0x0000, 0xa7),
                ]
                .concat(),
                [
                    block_q4(0xbc00, 0x3c00, 0x2e),
                    block_q4(0x3c00, 0x3800, 0xd4),
                ]
                .concat(),
            ),
            quant::RowFormat::Q5K => (
                512,
                [
                    block_q5(0x0000, 0x0000, 0x03, 0x51),
                    block_q5(0x0000, 0x0000, 0x54, 0xa7),
                ]
                .concat(),
                [
                    block_q5(0xbc00, 0x3c00, 0x9a, 0x2e),
                    block_q5(0x3c00, 0x3800, 0xc3, 0xd4),
                ]
                .concat(),
            ),
            quant::RowFormat::Q6K => (
                512,
                [
                    block_q6(0xe4, 0x10, 3, 0x0000),
                    block_q6(0x1b, 0xa5, -2, 0x0000),
                ]
                .concat(),
                [
                    block_q6(0x6c, 0x3e, 5, 0xbc00),
                    block_q6(0x93, 0xc1, -4, 0x3c00),
                ]
                .concat(),
            ),
            quant::RowFormat::IQ4NL => (
                64,
                [block_iq4_nl(0x0000, 0x51), block_iq4_nl(0x0000, 0xa7)].concat(),
                [block_iq4_nl(0x3800, 0x2e), block_iq4_nl(0x3c00, 0xd4)].concat(),
            ),
            quant::RowFormat::IQ4XS => (
                512,
                [
                    block_iq4_xs(0x0000, 0x1b, 0x51),
                    block_iq4_xs(0x0000, 0xe4, 0xa7),
                ]
                .concat(),
                [
                    block_iq4_xs(0x3800, 0x6c, 0x2e),
                    block_iq4_xs(0x3c00, 0x93, 0xd4),
                ]
                .concat(),
            ),
            _ => return Err("unknown executable row format".into()),
        };
        let mut matrix = first;
        matrix.extend(second);
        let activations = (0..width)
            .map(|index| (index as f32 - 91.0) / 29.0)
            .collect();
        Ok(Fixture {
            rows,
            width,
            matrix,
            activations,
        })
    }

    fn ramp_i8(start: i8, step: i8) -> [i8; 32] {
        std::array::from_fn(|index| start.wrapping_add(step.wrapping_mul(index as i8)))
    }

    fn block_q8(scale: u16, values: [i8; 32]) -> Vec<u8> {
        let mut block = scale.to_le_bytes().to_vec();
        block.extend(values.map(|value| value as u8));
        block
    }

    fn block_q4(scale: u16, minimum: u16, pattern: u8) -> Vec<u8> {
        let mut block = Vec::new();
        block.extend(scale.to_le_bytes());
        block.extend(minimum.to_le_bytes());
        block.extend((0..12).map(|index| pattern.wrapping_add(index as u8 * 19)));
        block.extend((0..128).map(|index| pattern.wrapping_add(index as u8 * 7)));
        block
    }

    fn block_q5(scale: u16, minimum: u16, high: u8, pattern: u8) -> Vec<u8> {
        let mut block = block_q4(scale, minimum, pattern);
        block.splice(
            16..16,
            (0..32).map(|index| high.rotate_left(index as u32 % 8)),
        );
        block
    }

    fn block_q6(high: u8, low: u8, scale: i8, super_scale: u16) -> Vec<u8> {
        let mut block = vec![low; 128];
        block.extend((0..64).map(|index| high.rotate_left(index as u32 % 8)));
        block.extend((0..16).map(|index| scale.wrapping_add(index as i8) as u8));
        block.extend(super_scale.to_le_bytes());
        block
    }

    fn block_iq4_nl(scale: u16, pattern: u8) -> Vec<u8> {
        let mut block = scale.to_le_bytes().to_vec();
        block.extend((0..16).map(|index| pattern.wrapping_add(index as u8 * 11)));
        block
    }

    fn block_iq4_xs(scale: u16, high: u8, pattern: u8) -> Vec<u8> {
        let mut block = scale.to_le_bytes().to_vec();
        block.extend([high, high.rotate_left(3)]);
        block.extend((0..4).map(|index| pattern.wrapping_add(index as u8 * 17)));
        block.extend((0..128).map(|index| pattern.wrapping_add(index as u8 * 13)));
        block
    }

    fn oracle_matrix(
        format: quant::RowFormat,
        matrix: &[u8],
        rows: usize,
        width: usize,
        activations: &[f32],
    ) -> core::result::Result<Vec<f64>, Box<dyn std::error::Error>> {
        let row_bytes = quant::row_byte_len(format, width)?;
        let mut results = Vec::new();
        for row in matrix.chunks_exact(row_bytes).take(rows) {
            let mut total = 0.0_f64;
            for index in 0..width {
                total += oracle_weight(format, row, index)? * f64::from(activations[index]);
            }
            results.push(total);
        }
        Ok(results)
    }

    fn oracle_weight(
        format: quant::RowFormat,
        row: &[u8],
        index: usize,
    ) -> core::result::Result<f64, Box<dyn std::error::Error>> {
        match format {
            quant::RowFormat::F32 => {
                let offset = index * 4;
                Ok(f64::from(f32::from_le_bytes(
                    row[offset..offset + 4].try_into()?,
                )))
            }
            quant::RowFormat::Q8_0 => {
                let block = &row[index / 32 * 34..];
                Ok(f16(block)? * f64::from(i8::from_le_bytes([block[2 + index % 32]])))
            }
            quant::RowFormat::Q4K => oracle_q4(row, index),
            quant::RowFormat::Q5K => oracle_q5(row, index),
            quant::RowFormat::Q6K => oracle_q6(row, index),
            quant::RowFormat::IQ4NL => oracle_iq4_nl(row, index),
            quant::RowFormat::IQ4XS => oracle_iq4_xs(row, index),
            _ => Err("unknown executable row format".into()),
        }
    }

    fn q_scale_min(scales: &[u8], group: usize) -> (u8, u8) {
        if group < 4 {
            (scales[group] & 0x3f, scales[group + 4] & 0x3f)
        } else {
            (
                (scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4),
                (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
            )
        }
    }

    fn oracle_q4(
        row: &[u8],
        index: usize,
    ) -> core::result::Result<f64, Box<dyn std::error::Error>> {
        let block = &row[index / 256 * 144..];
        let group = index % 256 / 32;
        let lane = index % 32;
        let (scale, minimum) = q_scale_min(&block[4..16], group);
        let packed = block[16 + group / 2 * 32 + lane];
        let quant = if group.is_multiple_of(2) {
            packed & 0x0f
        } else {
            packed >> 4
        };
        Ok(f16(block)? * f64::from(scale) * f64::from(quant)
            - f16(&block[2..])? * f64::from(minimum))
    }

    fn oracle_q5(
        row: &[u8],
        index: usize,
    ) -> core::result::Result<f64, Box<dyn std::error::Error>> {
        let block = &row[index / 256 * 176..];
        let group = index % 256 / 32;
        let lane = index % 32;
        let (scale, minimum) = q_scale_min(&block[4..16], group);
        let packed = block[48 + group / 2 * 32 + lane];
        let fifth = if block[16 + lane] & (1 << group) == 0 {
            0
        } else {
            16
        };
        let quant = (if group.is_multiple_of(2) {
            packed & 0x0f
        } else {
            packed >> 4
        }) + fifth;
        Ok(f16(block)? * f64::from(scale) * f64::from(quant)
            - f16(&block[2..])? * f64::from(minimum))
    }

    fn oracle_q6(
        row: &[u8],
        index: usize,
    ) -> core::result::Result<f64, Box<dyn std::error::Error>> {
        let block = &row[index / 256 * 210..];
        let local = index % 256;
        let half = local / 128;
        let quarter = local % 128 / 32;
        let lane = local % 32;
        let low = block[half * 64 + (quarter % 2) * 32 + lane];
        let lower = if quarter < 2 { low & 0x0f } else { low >> 4 };
        let upper = (block[128 + half * 32 + lane] >> (quarter * 2)) & 0x03;
        let quant = i16::from((upper << 4) | lower) - 32;
        let scale = i8::from_le_bytes([block[192 + half * 8 + quarter * 2 + lane / 16]]);
        Ok(f16(&block[208..])? * f64::from(scale) * f64::from(quant))
    }

    fn oracle_iq4_nl(
        row: &[u8],
        index: usize,
    ) -> core::result::Result<f64, Box<dyn std::error::Error>> {
        let block = &row[index / 32 * 18..];
        let lane = index % 32;
        let packed = block[2 + lane % 16];
        let code = if lane < 16 {
            packed & 0x0f
        } else {
            packed >> 4
        };
        Ok(f16(block)? * f64::from(iq4_value(code)))
    }

    fn oracle_iq4_xs(
        row: &[u8],
        index: usize,
    ) -> core::result::Result<f64, Box<dyn std::error::Error>> {
        let block = &row[index / 256 * 136..];
        let local = index % 256;
        let group = local / 32;
        let lane = local % 32;
        let low = if group.is_multiple_of(2) {
            block[4 + group / 2] & 0x0f
        } else {
            block[4 + group / 2] >> 4
        };
        let high = (u16::from_le_bytes([block[2], block[3]]) >> (group * 2)) & 0x03;
        let group_scale = i16::from(low | ((high as u8) << 4)) - 32;
        let packed = block[8 + group * 16 + lane % 16];
        let code = if lane < 16 {
            packed & 0x0f
        } else {
            packed >> 4
        };
        Ok(f16(block)? * f64::from(group_scale) * f64::from(iq4_value(code)))
    }

    fn iq4_value(index: u8) -> i8 {
        [
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        ][usize::from(index)]
    }

    fn f16(bytes: &[u8]) -> core::result::Result<f64, Box<dyn std::error::Error>> {
        let bits = u16::from_le_bytes(bytes[..2].try_into()?);
        let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
        let exponent = i32::from((bits >> 10) & 0x1f);
        let fraction = u32::from(bits & 0x03ff);
        match exponent {
            0 => Ok(sign * f64::from(fraction) * 2_f64.powi(-24)),
            31 => Err("synthetic fixture contains non-finite fp16".into()),
            _ => Ok(sign * (1.0 + f64::from(fraction) / 1024.0) * 2_f64.powi(exponent - 15)),
        }
    }

    fn assert_close(actual: &[f32], expected: &[f64], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} result length");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 1e-3_f64.max(expected.abs() * 1e-5);
            assert!(
                (f64::from(*actual) - expected).abs() <= tolerance,
                "{label} row {index}: got {actual}, expected {expected}"
            );
        }
    }

    #[cfg(feature = "gpu")]
    fn assert_gpu_close(
        actual: &[f32],
        expected: &[f32],
        label: &str,
    ) -> core::result::Result<(), String> {
        if actual.len() != expected.len() {
            return Err(format!("{label} device result length differs"));
        }
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 1e-3_f32.max(expected.abs() * 1e-3);
            if (*actual - *expected).abs() > tolerance {
                return Err(format!(
                    "{label} device row {index}: got {actual}, expected {expected}"
                ));
            }
        }
        Ok(())
    }
}
