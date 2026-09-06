use crate::error::RowArithmeticStage;
use crate::{
    Error, Q4_K_BLOCK_BYTES, Q4KBlock, Q5_K_BLOCK_BYTES, Q5KBlock, Q6_K_BLOCK_BYTES, Q6KBlock,
    RowFormat, row_byte_len, row_dot_f32,
};

const VALUES_PER_BLOCK: usize = 256;
const VALUES_PER_GROUP: usize = 32;
const SCALE_BYTES: usize = 12;
const NIBBLES: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const HIGH_PAIRS: [u8; 4] = [0, 1, 2, 3];
const Q6_SCALES: [i8; 16] = [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8];

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
    let scales = [1, 2, 3, 4, 17, 18, 19, 20];
    let minima = [5, 6, 7, 8, 33, 34, 35, 36];
    let mut bytes = [0; Q4_K_BLOCK_BYTES];
    bytes[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
    bytes[2..4].copy_from_slice(&0x3400_u16.to_le_bytes());
    pack_scale_min(&mut bytes[4..16], scales, minima);

    let mut expected = [0.0; VALUES_PER_BLOCK];
    for group in 0..8 {
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

fn q5_fixture(seed: usize) -> ([u8; Q5_K_BLOCK_BYTES], [f32; VALUES_PER_BLOCK]) {
    let scales = [1, 2, 3, 4, 17, 18, 19, 20];
    let minima = [5, 6, 7, 8, 33, 34, 35, 36];
    let mut bytes = [0; Q5_K_BLOCK_BYTES];
    bytes[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
    bytes[2..4].copy_from_slice(&0x3400_u16.to_le_bytes());
    pack_scale_min(&mut bytes[4..16], scales, minima);

    let mut expected = [0.0; VALUES_PER_BLOCK];
    for group in 0..8 {
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
