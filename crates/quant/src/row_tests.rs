use crate::error::RowArithmeticStage;
use crate::{
    Error, Q4_K_BLOCK_BYTES, Q4KBlock, Q5_K_BLOCK_BYTES, Q5KBlock, Q6_K_BLOCK_BYTES, Q6KBlock,
    Q8_0_BLOCK_BYTES, Q8_0_VALUES_PER_BLOCK, RowFormat, row_byte_len, row_decode_f32, row_dot_f32,
};

const VALUES_PER_BLOCK: usize = 256;
const VALUES_PER_GROUP: usize = 32;
const SCALE_BYTES: usize = 12;
const NIBBLES: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const HIGH_PAIRS: [u8; 4] = [0, 1, 2, 3];
const Q6_SCALES: [i8; 16] = [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8];

// WHY: Original synthetic conformance vector manually derived and independently
// read-checked on 2026-09-06 against pinned llama.cpp
// 6a1a922d269908a29cbd4b49c27e6a8e7fd10fae: ggml/src/ggml-common.h:323-368
// and ggml/src/ggml-quants.c:880-886,1529-1550,1731-1755,1939-1967. No upstream
// fixture, table, or implementation code is carried.
const FIXED_K_PREFIX: [u8; 4] = [0x00, 0x3c, 0x00, 0x3c];
const FIXED_K_SCALE_MIN: [u8; 12] = [
    0x01, 0x52, 0xa3, 0xf4, 0x3c, 0x6b, 0x9a, 0xc9, 0x75, 0x86, 0x97, 0xa8,
];
const FIXED_K_PAYLOAD: [u8; 16] = [
    0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b, 0x3c, 0x2d, 0x1e, 0x0f,
];
const FIXED_Q6_OTHER_PAYLOAD: [u8; 16] = [
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
];
const FIXED_Q5_HIGH_BITS: [u8; 8] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80];
const FIXED_K_SCALES: [u8; 8] = [1, 18, 35, 52, 5, 22, 39, 56];
const FIXED_K_MINIMA: [u8; 8] = [60, 43, 26, 9, 7, 24, 41, 58];

#[test]
fn q4_k_decodes_all_scale_min_fields_and_two_block_dispatcher_dot() -> crate::Result<()> {
    let (first_bytes, first_expected) = q4_fixture(0);
    let (second_bytes, second_expected) = q4_fixture(5);
    assert_decoded_q4(&first_bytes, &first_expected)?;
    assert_decoded_q4(&second_bytes, &second_expected)?;

    let expected = expected_two_block_dot(&first_expected, &second_expected);
    let mut serialized = first_bytes.to_vec();
    serialized.extend(second_bytes);
    let actual = row_dot_f32(RowFormat::Q4K, &serialized, &mixed_activations())?;
    assert_eq!(
        f64::from(actual).to_bits(),
        expected.to_bits(),
        "Q4_K dispatcher dot must equal the independent f64 oracle"
    );
    Ok(())
}

#[test]
fn fixed_q4_k_witness_decodes_all_packed_scale_min_fields() -> crate::Result<()> {
    let bytes = fixed_q4_witness();
    let decoded = Q4KBlock::parse(&bytes)?.decode_f32();
    assert_full_vector(&decoded, &fixed_q4_expected(), "fixed Q4_K witness");
    Ok(())
}

#[test]
fn q5_k_decodes_all_fifth_bit_planes_and_two_block_dispatcher_dot() -> crate::Result<()> {
    let (first_bytes, first_expected) = q5_fixture(0);
    let (second_bytes, second_expected) = q5_fixture(7);
    assert_decoded_q5(&first_bytes, &first_expected)?;
    assert_decoded_q5(&second_bytes, &second_expected)?;

    let mut changed_high_plane = first_bytes;
    changed_high_plane[16] ^= 1;
    let changed = Q5KBlock::parse(&changed_high_plane)?.decode_f32();
    assert_ne!(
        changed[0].to_bits(),
        first_expected[0].to_bits(),
        "a fifth-bit-plane error must change the first decoded Q5_K value"
    );

    let expected = expected_two_block_dot(&first_expected, &second_expected);
    let mut serialized = first_bytes.to_vec();
    serialized.extend(second_bytes);
    let actual = row_dot_f32(RowFormat::Q5K, &serialized, &mixed_activations())?;
    assert_eq!(
        f64::from(actual).to_bits(),
        expected.to_bits(),
        "Q5_K dispatcher dot must equal the independent f64 oracle"
    );
    Ok(())
}

#[test]
fn fixed_q5_k_witness_decodes_all_fifth_bit_planes() -> crate::Result<()> {
    let bytes = fixed_q5_witness();
    let decoded = Q5KBlock::parse(&bytes)?.decode_f32();
    assert_full_vector(&decoded, &fixed_q5_expected(), "fixed Q5_K witness");
    Ok(())
}

#[test]
fn q6_k_decodes_all_scale_groups_both_halves_and_two_block_dispatcher_dot() -> crate::Result<()> {
    let (first_bytes, first_expected) = q6_fixture(0);
    let (second_bytes, second_expected) = q6_fixture(3);
    assert_decoded_q6(&first_bytes, &first_expected)?;
    assert_decoded_q6(&second_bytes, &second_expected)?;

    let mut changed_high_plane = first_bytes;
    changed_high_plane[128] ^= 0x01;
    let changed = Q6KBlock::parse(&changed_high_plane)?.decode_f32();
    assert_ne!(
        changed[0].to_bits(),
        first_expected[0].to_bits(),
        "a Q6_K high-plane bit error must change the signed first value"
    );

    let expected = expected_two_block_dot(&first_expected, &second_expected);
    let mut serialized = first_bytes.to_vec();
    serialized.extend(second_bytes);
    let actual = row_dot_f32(RowFormat::Q6K, &serialized, &mixed_activations())?;
    assert_eq!(
        f64::from(actual).to_bits(),
        expected.to_bits(),
        "Q6_K dispatcher dot must equal the independent f64 oracle"
    );
    Ok(())
}

#[test]
fn fixed_q6_k_witness_decodes_both_halves_signed_scales_and_lane_splits() -> crate::Result<()> {
    let bytes = fixed_q6_witness();
    let decoded = Q6KBlock::parse(&bytes)?.decode_f32();
    assert_full_vector(&decoded, &fixed_q6_expected(), "fixed Q6_K witness");
    Ok(())
}

#[test]
fn row_decode_dispatches_all_preexisting_formats() -> crate::Result<()> {
    let f32_bytes = [
        0x00, 0x00, 0xa0, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x40,
    ];
    assert_eq!(
        row_decode_f32(RowFormat::F32, &f32_bytes, 3)?,
        vec![-1.25, 0.0, 2.25]
    );

    let mut q8_bytes = [0; Q8_0_BLOCK_BYTES];
    q8_bytes[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    q8_bytes[2..].fill(1);
    assert_eq!(
        row_decode_f32(RowFormat::Q8_0, &q8_bytes, Q8_0_VALUES_PER_BLOCK)?,
        vec![1.0; Q8_0_VALUES_PER_BLOCK]
    );

    let (q4_bytes, q4_expected) = q4_fixture(0);
    assert_eq!(
        row_decode_f32(RowFormat::Q4K, &q4_bytes, VALUES_PER_BLOCK)?,
        q4_expected
    );
    let (q5_bytes, q5_expected) = q5_fixture(0);
    assert_eq!(
        row_decode_f32(RowFormat::Q5K, &q5_bytes, VALUES_PER_BLOCK)?,
        q5_expected
    );
    let (q6_bytes, q6_expected) = q6_fixture(0);
    assert_eq!(
        row_decode_f32(RowFormat::Q6K, &q6_bytes, VALUES_PER_BLOCK)?,
        q6_expected
    );
    Ok(())
}

#[test]
fn q4_k_dispatcher_refuses_invalid_geometry_and_nonfinite_arithmetic() {
    let (bytes, _) = q4_fixture(0);
    assert!(
        matches!(
            row_dot_f32(RowFormat::Q4K, &bytes, &[]),
            Err(Error::EmptyRowInput {
                format: RowFormat::Q4K,
                ..
            })
        ),
        "Q4_K dispatcher must reject an empty activation row"
    );
    assert!(
        matches!(
            row_dot_f32(RowFormat::Q4K, &bytes, &[1.0; VALUES_PER_BLOCK - 1]),
            Err(Error::InvalidRowInputLength {
                format: RowFormat::Q4K,
                ..
            })
        ),
        "Q4_K dispatcher must require complete blocks"
    );
    assert!(
        matches!(
            row_dot_f32(
                RowFormat::Q4K,
                &bytes[..Q4_K_BLOCK_BYTES - 1],
                &[1.0; VALUES_PER_BLOCK]
            ),
            Err(Error::RowByteLengthMismatch {
                format: RowFormat::Q4K,
                ..
            })
        ),
        "Q4_K dispatcher must require exact serialized extent"
    );
    let mut nonfinite_activations = [1.0; VALUES_PER_BLOCK];
    nonfinite_activations[17] = f32::NAN;
    assert!(
        matches!(
            row_dot_f32(RowFormat::Q4K, &bytes, &nonfinite_activations),
            Err(Error::NonFiniteRowActivation {
                format: RowFormat::Q4K,
                index: 17,
                ..
            })
        ),
        "Q4_K dispatcher must retain flat activation indexes"
    );
    let maximum_multiple = usize::MAX - (usize::MAX % VALUES_PER_BLOCK);
    assert!(
        row_byte_len(RowFormat::Q4K, maximum_multiple).is_ok(),
        "Q4_K rows cannot overflow because 144 serialized bytes represent 256 values"
    );

    let mut maximum_scale = bytes;
    maximum_scale[..2].copy_from_slice(&0x7bff_u16.to_le_bytes());
    assert!(
        matches!(
            row_dot_f32(
                RowFormat::Q4K,
                &maximum_scale,
                &[f32::MAX; VALUES_PER_BLOCK]
            ),
            Err(Error::NonFiniteRowArithmetic {
                format: RowFormat::Q4K,
                stage: RowArithmeticStage::Product,
                ..
            })
        ),
        "Q4_K non-finite products must be refused before an accumulator escapes"
    );
}

fn q4_fixture(seed: usize) -> ([u8; Q4_K_BLOCK_BYTES], [f32; VALUES_PER_BLOCK]) {
    let scales = [1, 2, 3, 4, 9, 18, 35, 52];
    let minima = [5, 6, 7, 8, 10, 21, 38, 55];
    let mut bytes = [0; Q4_K_BLOCK_BYTES];
    bytes[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
    bytes[2..4].copy_from_slice(&0x3400_u16.to_le_bytes());
    pack_scale_min(&mut bytes[4..16], scales, minima);

    let mut expected = [0.0; VALUES_PER_BLOCK];
    for group in 0_usize..8 {
        for lane in 0..VALUES_PER_GROUP {
            let quantized = NIBBLES[(group + lane + seed) % NIBBLES.len()];
            expected[group * VALUES_PER_GROUP + lane] =
                0.5 * f32::from(scales[group]) * f32::from(quantized)
                    - 0.25 * f32::from(minima[group]);
            let pair = group / 2;
            let payload_index = 16 + pair * VALUES_PER_GROUP + lane;
            if group.is_multiple_of(2) {
                bytes[payload_index] |= quantized;
            } else {
                bytes[payload_index] |= quantized << 4;
            }
        }
    }
    (bytes, expected)
}

fn fixed_q4_witness() -> [u8; Q4_K_BLOCK_BYTES] {
    let mut bytes = [0; Q4_K_BLOCK_BYTES];
    bytes[..4].copy_from_slice(&FIXED_K_PREFIX);
    bytes[4..16].copy_from_slice(&FIXED_K_SCALE_MIN);
    for fragment in bytes[16..].chunks_exact_mut(FIXED_K_PAYLOAD.len()) {
        fragment.copy_from_slice(&FIXED_K_PAYLOAD);
    }
    bytes
}

fn fixed_q4_expected() -> [f32; VALUES_PER_BLOCK] {
    let mut expected = [0.0; VALUES_PER_BLOCK];
    for group in 0_usize..8 {
        for lane in 0..VALUES_PER_GROUP {
            let nibble = lane % NIBBLES.len();
            let quantized = if group.is_multiple_of(2) {
                NIBBLES[nibble]
            } else {
                NIBBLES[NIBBLES.len() - 1 - nibble]
            };
            expected[group * VALUES_PER_GROUP + lane] = f32::from(FIXED_K_SCALES[group])
                * f32::from(quantized)
                - f32::from(FIXED_K_MINIMA[group]);
        }
    }
    expected
}

fn q5_fixture(seed: usize) -> ([u8; Q5_K_BLOCK_BYTES], [f32; VALUES_PER_BLOCK]) {
    let scales = [1, 2, 3, 4, 9, 18, 35, 52];
    let minima = [5, 6, 7, 8, 10, 21, 38, 55];
    let mut bytes = [0; Q5_K_BLOCK_BYTES];
    bytes[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
    bytes[2..4].copy_from_slice(&0x3400_u16.to_le_bytes());
    pack_scale_min(&mut bytes[4..16], scales, minima);

    let mut expected = [0.0; VALUES_PER_BLOCK];
    for group in 0_usize..8 {
        for lane in 0..VALUES_PER_GROUP {
            let high = (group + lane + seed).is_multiple_of(2);
            let low = NIBBLES[(group * 3 + lane + seed) % NIBBLES.len()];
            let quantized = low + if high { 16 } else { 0 };
            expected[group * VALUES_PER_GROUP + lane] =
                0.5 * f32::from(scales[group]) * f32::from(quantized)
                    - 0.25 * f32::from(minima[group]);
            let payload_index = 48 + (group / 2) * VALUES_PER_GROUP + lane;
            if group.is_multiple_of(2) {
                bytes[payload_index] |= low;
            } else {
                bytes[payload_index] |= low << 4;
            }
            if high {
                bytes[16 + lane] |= 1_u8 << group;
            }
        }
    }
    (bytes, expected)
}

fn fixed_q5_witness() -> [u8; Q5_K_BLOCK_BYTES] {
    let mut bytes = [0; Q5_K_BLOCK_BYTES];
    bytes[..4].copy_from_slice(&FIXED_K_PREFIX);
    bytes[4..16].copy_from_slice(&FIXED_K_SCALE_MIN);
    for fragment in bytes[16..48].chunks_exact_mut(FIXED_Q5_HIGH_BITS.len()) {
        fragment.copy_from_slice(&FIXED_Q5_HIGH_BITS);
    }
    for fragment in bytes[48..].chunks_exact_mut(FIXED_K_PAYLOAD.len()) {
        fragment.copy_from_slice(&FIXED_K_PAYLOAD);
    }
    bytes
}

fn fixed_q5_expected() -> [f32; VALUES_PER_BLOCK] {
    let mut expected = [0.0; VALUES_PER_BLOCK];
    for group in 0_usize..8 {
        for lane in 0..VALUES_PER_GROUP {
            let nibble = lane % NIBBLES.len();
            let low = if group.is_multiple_of(2) {
                NIBBLES[nibble]
            } else {
                NIBBLES[NIBBLES.len() - 1 - nibble]
            };
            let quantized = low + if lane % 8 == group { 16 } else { 0 };
            expected[group * VALUES_PER_GROUP + lane] = f32::from(FIXED_K_SCALES[group])
                * f32::from(quantized)
                - f32::from(FIXED_K_MINIMA[group]);
        }
    }
    expected
}

fn q6_fixture(seed: usize) -> ([u8; Q6_K_BLOCK_BYTES], [f32; VALUES_PER_BLOCK]) {
    let mut bytes = [0; Q6_K_BLOCK_BYTES];
    bytes[208..].copy_from_slice(&0x3800_u16.to_le_bytes());
    let mut expected = [0.0; VALUES_PER_BLOCK];
    for half in 0..2 {
        for lane in 0..VALUES_PER_GROUP {
            let mut high_byte = 0_u8;
            for quarter in 0..4 {
                let group = half * 8 + quarter * 2 + lane / 16;
                let low = NIBBLES[(half + quarter + lane + seed) % NIBBLES.len()];
                let high = HIGH_PAIRS[(half + quarter + lane + seed) % HIGH_PAIRS.len()];
                let raw = low | (high << 4);
                let output_index = half * 128 + quarter * VALUES_PER_GROUP + lane;
                expected[output_index] =
                    0.5 * f32::from(Q6_SCALES[group]) * f32::from(i16::from(raw) - 32);
                let low_index = half * 64 + (quarter % 2) * VALUES_PER_GROUP + lane;
                if quarter < 2 {
                    bytes[low_index] |= low;
                } else {
                    bytes[low_index] |= low << 4;
                }
                high_byte |= high << (quarter * 2);
            }
            bytes[128 + half * VALUES_PER_GROUP + lane] = high_byte;
        }
    }
    for (index, scale) in Q6_SCALES.iter().enumerate() {
        bytes[192 + index] = scale.to_le_bytes()[0];
    }
    (bytes, expected)
}

fn fixed_q6_witness() -> [u8; Q6_K_BLOCK_BYTES] {
    let mut bytes = [0; Q6_K_BLOCK_BYTES];
    for fragment in bytes[..32].chunks_exact_mut(FIXED_K_PAYLOAD.len()) {
        fragment.copy_from_slice(&FIXED_K_PAYLOAD);
    }
    for fragment in bytes[32..96].chunks_exact_mut(FIXED_Q6_OTHER_PAYLOAD.len()) {
        fragment.copy_from_slice(&FIXED_Q6_OTHER_PAYLOAD);
    }
    for fragment in bytes[96..128].chunks_exact_mut(FIXED_K_PAYLOAD.len()) {
        fragment.copy_from_slice(&FIXED_K_PAYLOAD);
    }
    bytes[128..160].fill(0xe4);
    bytes[160..192].fill(0x1b);
    for (index, scale) in Q6_SCALES.iter().enumerate() {
        bytes[192 + index] = scale.to_le_bytes()[0];
    }
    bytes[208..].copy_from_slice(&0x3c00_u16.to_le_bytes());
    bytes
}

fn fixed_q6_expected() -> [f32; VALUES_PER_BLOCK] {
    let mut expected = [0.0; VALUES_PER_BLOCK];
    for half in 0..2 {
        for quarter in 0..4 {
            for lane in 0..VALUES_PER_GROUP {
                let remainder = i16::from(NIBBLES[lane % NIBBLES.len()]);
                let quantized = if half == 0 {
                    if quarter == 0 {
                        remainder - 32
                    } else if quarter == 1 {
                        remainder - 16
                    } else if quarter == 2 {
                        15 - remainder
                    } else {
                        remainder + 16
                    }
                } else if quarter == 0 {
                    remainder + 16
                } else if quarter == 1 {
                    remainder
                } else if quarter == 2 {
                    remainder - 16
                } else {
                    -17 - remainder
                };
                let scale_index = half * 8 + quarter * 2 + lane / 16;
                let output_index = half * 128 + quarter * VALUES_PER_GROUP + lane;
                expected[output_index] = f32::from(Q6_SCALES[scale_index]) * f32::from(quantized);
            }
        }
    }
    expected
}

fn pack_scale_min(destination: &mut [u8], scales: [u8; 8], minima: [u8; 8]) {
    assert_eq!(
        destination.len(),
        SCALE_BYTES,
        "K scale/min fixture must have twelve packed bytes"
    );
    destination[..4].copy_from_slice(&scales[..4]);
    destination[4..8].copy_from_slice(&minima[..4]);
    for group in 4..8 {
        destination[group + 4] = (scales[group] & 0x0f) | ((minima[group] & 0x0f) << 4);
        destination[group - 4] |= (scales[group] >> 4) << 6;
        destination[group] |= (minima[group] >> 4) << 6;
    }
}

fn assert_decoded_q4(
    bytes: &[u8; Q4_K_BLOCK_BYTES],
    expected: &[f32; VALUES_PER_BLOCK],
) -> crate::Result<()> {
    let decoded = Q4KBlock::parse(bytes)?.decode_f32();
    assert_full_vector(&decoded, expected, "Q4_K full vector");
    Ok(())
}

fn assert_decoded_q5(
    bytes: &[u8; Q5_K_BLOCK_BYTES],
    expected: &[f32; VALUES_PER_BLOCK],
) -> crate::Result<()> {
    let decoded = Q5KBlock::parse(bytes)?.decode_f32();
    assert_full_vector(&decoded, expected, "Q5_K full vector");
    Ok(())
}

fn assert_decoded_q6(
    bytes: &[u8; Q6_K_BLOCK_BYTES],
    expected: &[f32; VALUES_PER_BLOCK],
) -> crate::Result<()> {
    let decoded = Q6KBlock::parse(bytes)?.decode_f32();
    assert_full_vector(&decoded, expected, "Q6_K full vector");
    Ok(())
}

fn assert_full_vector(
    actual: &[f32; VALUES_PER_BLOCK],
    expected: &[f32; VALUES_PER_BLOCK],
    label: &str,
) {
    for (index, (actual_value, expected_value)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual_value.to_bits(),
            expected_value.to_bits(),
            "{label} must match its independent expected value at index {index}"
        );
    }
}

fn mixed_activations() -> [f32; VALUES_PER_BLOCK * 2] {
    let pattern = [-3.0_f32, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let mut activations = [0.0; VALUES_PER_BLOCK * 2];
    for (index, activation) in activations.iter_mut().enumerate() {
        *activation = pattern[index % pattern.len()];
    }
    activations
}

fn expected_two_block_dot(
    first: &[f32; VALUES_PER_BLOCK],
    second: &[f32; VALUES_PER_BLOCK],
) -> f64 {
    let activations = mixed_activations();
    first
        .iter()
        .chain(second)
        .zip(activations)
        .fold(0.0_f64, |sum, (weight, activation)| {
            sum + f64::from(*weight) * f64::from(activation)
        })
}
