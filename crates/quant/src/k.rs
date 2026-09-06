//! Shared `K`-quant block fields and scale-pair decoding.

use half::f16;

use crate::error::NonFiniteKScaleSnafu;
use crate::{Result, RowFormat};

pub(crate) const K_VALUES_PER_BLOCK: usize = 256;
pub(crate) const K_SCALE_BYTES: usize = 12;
pub(crate) const K_GROUP_VALUES: usize = 32;

pub(crate) fn finite_half(format: RowFormat, field: &'static str, bytes: [u8; 2]) -> Result<f32> {
    let bits = u16::from_le_bytes(bytes);
    let value = f16::from_bits(bits).to_f32();
    if !value.is_finite() {
        return NonFiniteKScaleSnafu {
            format,
            field,
            bits,
        }
        .fail();
    }
    Ok(value)
}

pub(crate) fn scale_min(scales: &[u8; K_SCALE_BYTES], group: usize) -> (u8, u8) {
    const LOW_SIX_BITS: u8 = 0x3f;
    const LOW_FOUR_BITS: u8 = 0x0f;
    const HIGH_TWO_SHIFT: u8 = 6;
    const HIGH_NIBBLE_SHIFT: u8 = 4;
    const FIRST_DIRECT_GROUPS: usize = 4;

    if group < FIRST_DIRECT_GROUPS {
        return (
            scales[group] & LOW_SIX_BITS,
            scales[group + FIRST_DIRECT_GROUPS] & LOW_SIX_BITS,
        );
    }

    let packed = scales[group + FIRST_DIRECT_GROUPS];
    let scale_high = (scales[group - FIRST_DIRECT_GROUPS] >> HIGH_TWO_SHIFT) << HIGH_NIBBLE_SHIFT;
    let minimum_high = (scales[group] >> HIGH_TWO_SHIFT) << HIGH_NIBBLE_SHIFT;
    (
        (packed & LOW_FOUR_BITS) | scale_high,
        (packed >> HIGH_NIBBLE_SHIFT) | minimum_high,
    )
}
