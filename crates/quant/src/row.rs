//! Shared checked, sequential serialized-row arithmetic.

use crate::error::{
    EmptyRowInputSnafu, InvalidRowInputLengthSnafu, NonFiniteRowActivationSnafu,
    NonFiniteRowArithmeticSnafu, RowArithmeticStage, RowByteLengthMismatchSnafu,
    RowByteLengthOverflowSnafu,
};
use crate::{Result, RowFormat};

#[derive(Clone, Copy)]
pub(crate) struct Geometry {
    pub(crate) format: RowFormat,
    pub(crate) bytes_per_block: usize,
}

pub(crate) fn byte_len<const VALUES: usize>(
    geometry: Geometry,
    value_count: usize,
) -> Result<usize> {
    if value_count == 0 {
        return EmptyRowInputSnafu {
            format: geometry.format,
        }
        .fail();
    }
    if !value_count.is_multiple_of(VALUES) {
        return InvalidRowInputLengthSnafu {
            format: geometry.format,
            actual: value_count,
            block_elements: VALUES,
        }
        .fail();
    }
    let block_count = value_count / VALUES;
    block_count
        .checked_mul(geometry.bytes_per_block)
        .ok_or_else(|| {
            RowByteLengthOverflowSnafu {
                format: geometry.format,
                block_count,
                block_bytes: geometry.bytes_per_block,
            }
            .build()
        })
}

pub(crate) fn dot<const VALUES: usize, Decode>(
    geometry: Geometry,
    serialized_row: &[u8],
    activations: &[f32],
    mut decode: Decode,
) -> Result<f32>
where
    Decode: FnMut(&[u8]) -> Result<[f32; VALUES]>,
{
    let expected_bytes = byte_len::<VALUES>(geometry, activations.len())?;
    if serialized_row.len() != expected_bytes {
        return RowByteLengthMismatchSnafu {
            format: geometry.format,
            actual: serialized_row.len(),
            expected: expected_bytes,
        }
        .fail();
    }

    let mut accumulator = 0.0_f32;
    for (block_index, (serialized_block, activation_block)) in serialized_row
        .chunks_exact(geometry.bytes_per_block)
        .zip(activations.chunks_exact(VALUES))
        .enumerate()
    {
        let decoded = decode(serialized_block)?;
        for (lane_index, (weight, activation)) in decoded.iter().zip(activation_block).enumerate() {
            if !activation.is_finite() {
                return NonFiniteRowActivationSnafu {
                    format: geometry.format,
                    index: block_index * VALUES + lane_index,
                }
                .fail();
            }
            let product = *weight * *activation;
            if !product.is_finite() {
                return NonFiniteRowArithmeticSnafu {
                    format: geometry.format,
                    stage: RowArithmeticStage::Product,
                    block_index,
                    lane_index,
                }
                .fail();
            }
            accumulator += product;
            if !accumulator.is_finite() {
                return NonFiniteRowArithmeticSnafu {
                    format: geometry.format,
                    stage: RowArithmeticStage::Accumulation,
                    block_index,
                    lane_index,
                }
                .fail();
            }
        }
    }
    Ok(accumulator)
}
