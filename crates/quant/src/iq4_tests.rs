//! Independent CPU witnesses for the two executable IQ4 layouts.

use crate::error::RowArithmeticStage;
use crate::{
    Error, IQ4_NL_BLOCK_BYTES, IQ4_NL_VALUES_PER_BLOCK, IQ4_XS_BLOCK_BYTES,
    IQ4_XS_VALUES_PER_BLOCK, Iq4NlBlock, Iq4XsBlock, RowFormat, row_byte_len, row_decode_f32,
    row_dot_f32,
};

const HALF_CODEPOINTS: [f32; 16] = [
    -63.5, -52.0, -41.5, -32.5, -24.5, -17.5, -11.0, -5.0, 0.5, 6.5, 12.5, 19.0, 26.5, 34.5, 44.5,
    56.5,
];
const XS_GROUP_SCALES: [f32; 8] = [-16.0, -7.5, -0.5, 0.0, 0.5, 3.5, 8.0, 15.5];
const PACKED_CODEPOINTS: [u8; 16] = [
    0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b, 0x3c, 0x2d, 0x1e, 0x0f,
];

#[test]
fn iq4_nl_decodes_every_codepoint_from_both_nibble_planes() -> crate::Result<()> {
    let bytes = nl_codepoint_block(0x3800);
    let decoded = Iq4NlBlock::parse(&bytes)?.decode_f32();
    assert_eq!(
        row_decode_f32(RowFormat::IQ4NL, &bytes, IQ4_NL_VALUES_PER_BLOCK)?,
        decoded
    );

    for lane in 0..HALF_CODEPOINTS.len() {
        assert_eq!(
            decoded[lane].to_bits(),
            HALF_CODEPOINTS[lane].to_bits(),
            "low IQ4_NL nibble must decode codepoint {lane}"
        );
        assert_eq!(
            decoded[lane + HALF_CODEPOINTS.len()].to_bits(),
            HALF_CODEPOINTS[HALF_CODEPOINTS.len() - 1 - lane].to_bits(),
            "high IQ4_NL nibble must decode the independently ordered codepoint {lane}"
        );
    }
    Ok(())
}

#[test]
fn iq4_xs_decodes_all_group_scales_and_high_bit_planes() -> crate::Result<()> {
    let bytes = xs_scale_witness(0x3800);
    let decoded = Iq4XsBlock::parse(&bytes)?.decode_f32();
    assert_eq!(
        row_decode_f32(RowFormat::IQ4XS, &bytes, IQ4_XS_VALUES_PER_BLOCK)?,
        decoded
    );

    for (group, expected) in XS_GROUP_SCALES.iter().enumerate() {
        let group_start = group * 32;
        assert!(
            decoded[group_start..group_start + 32]
                .iter()
                .all(|actual| actual.to_bits() == expected.to_bits()),
            "IQ4_XS group {group} must retain its low and high packed scale bits"
        );
    }
    Ok(())
}

#[test]
fn iq4_rows_keep_block_order_across_multiple_blocks() -> crate::Result<()> {
    let mut nl_serialized = nl_codepoint_block(0x3800).to_vec();
    nl_serialized.extend(nl_codepoint_block(0x3c00));
    let nl_actual = row_dot_f32(
        RowFormat::IQ4NL,
        &nl_serialized,
        &[1.0; IQ4_NL_VALUES_PER_BLOCK * 2],
    )?;
    assert_eq!(nl_actual.to_bits(), (-282.0_f32).to_bits());

    let mut xs_serialized = xs_codepoint_blocks(0x3800).to_vec();
    xs_serialized.extend(xs_codepoint_blocks(0x3c00));
    let xs_actual = row_dot_f32(
        RowFormat::IQ4XS,
        &xs_serialized,
        &[1.0; IQ4_XS_VALUES_PER_BLOCK * 2],
    )?;
    assert_eq!(xs_actual.to_bits(), 55_272.0_f32.to_bits());
    Ok(())
}

#[test]
fn iq4_rows_refuse_malformed_and_misaligned_inputs() {
    assert!(
        matches!(
            Iq4NlBlock::parse(&[0; IQ4_NL_BLOCK_BYTES - 1]),
            Err(Error::InvalidIq4BlockLength {
                format: RowFormat::IQ4NL,
                ..
            })
        ),
        "IQ4_NL parser must reject incomplete blocks"
    );

    let nl = nl_codepoint_block(0x3800);
    assert!(
        matches!(
            row_dot_f32(RowFormat::IQ4NL, &nl, &[]),
            Err(Error::EmptyRowInput {
                format: RowFormat::IQ4NL,
                ..
            })
        ),
        "IQ4_NL rows must not accept empty activation vectors"
    );
    assert!(
        matches!(
            row_dot_f32(RowFormat::IQ4NL, &nl, &[1.0; IQ4_NL_VALUES_PER_BLOCK - 1]),
            Err(Error::InvalidRowInputLength {
                format: RowFormat::IQ4NL,
                ..
            })
        ),
        "IQ4_NL rows must require complete blocks"
    );
    assert!(
        matches!(
            row_decode_f32(
                RowFormat::IQ4NL,
                &nl[..IQ4_NL_BLOCK_BYTES - 1],
                IQ4_NL_VALUES_PER_BLOCK
            ),
            Err(Error::RowByteLengthMismatch {
                format: RowFormat::IQ4NL,
                ..
            })
        ),
        "IQ4_NL rows must require exact serialized extents"
    );
}

#[test]
fn iq4_rows_refuse_nonfinite_scales_activations_and_products() {
    let mut nonfinite_xs = xs_scale_witness(0x3800);
    nonfinite_xs[..2].copy_from_slice(&0x7e00_u16.to_le_bytes());
    assert!(
        matches!(
            row_decode_f32(RowFormat::IQ4XS, &nonfinite_xs, IQ4_XS_VALUES_PER_BLOCK),
            Err(Error::NonFiniteIq4Scale {
                format: RowFormat::IQ4XS,
                ..
            })
        ),
        "IQ4_XS parser must reject a NaN block scale"
    );

    let mut nonfinite_activations = [1.0; IQ4_XS_VALUES_PER_BLOCK];
    nonfinite_activations[129] = f32::INFINITY;
    assert!(
        matches!(
            row_dot_f32(
                RowFormat::IQ4XS,
                &xs_scale_witness(0x3800),
                &nonfinite_activations
            ),
            Err(Error::NonFiniteRowActivation {
                format: RowFormat::IQ4XS,
                index: 129,
                ..
            })
        ),
        "IQ4_XS rows must preserve flat non-finite activation indexes"
    );

    let mut maximum_scale = nl_codepoint_block(0x7bff);
    maximum_scale[2] = 0;
    assert!(
        matches!(
            row_dot_f32(
                RowFormat::IQ4NL,
                &maximum_scale,
                &[f32::MAX; IQ4_NL_VALUES_PER_BLOCK]
            ),
            Err(Error::NonFiniteRowArithmetic {
                format: RowFormat::IQ4NL,
                stage: RowArithmeticStage::Product,
                ..
            })
        ),
        "IQ4_NL rows must refuse a non-finite product"
    );
    assert_eq!(
        row_byte_len(RowFormat::IQ4XS, IQ4_XS_VALUES_PER_BLOCK),
        Ok(IQ4_XS_BLOCK_BYTES),
        "IQ4_XS byte length must use its one canonical geometry"
    );
}

fn nl_codepoint_block(scale_bits: u16) -> [u8; IQ4_NL_BLOCK_BYTES] {
    let mut bytes = [0; IQ4_NL_BLOCK_BYTES];
    bytes[..2].copy_from_slice(&scale_bits.to_le_bytes());
    bytes[2..].copy_from_slice(&PACKED_CODEPOINTS);
    bytes
}

fn xs_scale_witness(scale_bits: u16) -> [u8; IQ4_XS_BLOCK_BYTES] {
    let mut bytes = [0; IQ4_XS_BLOCK_BYTES];
    bytes[..2].copy_from_slice(&scale_bits.to_le_bytes());
    bytes[2..6].copy_from_slice(&[0x10, 0x0f, 0x71, 0xf0]);
    bytes[6..8].copy_from_slice(&0xfa94_u16.to_le_bytes());
    bytes[8..].fill(0x88);
    bytes
}

fn xs_codepoint_blocks(scale_bits: u16) -> [u8; IQ4_XS_BLOCK_BYTES] {
    let mut bytes = [0; IQ4_XS_BLOCK_BYTES];
    bytes[..2].copy_from_slice(&scale_bits.to_le_bytes());
    bytes[2..6].fill(0xf0);
    bytes[6..8].fill(0x00);
    for group in 0..8 {
        bytes[8 + group * 16..8 + (group + 1) * 16].copy_from_slice(&PACKED_CODEPOINTS);
    }
    bytes
}
