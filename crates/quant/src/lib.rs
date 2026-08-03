//! # quant
//!
//! Quantisation schemes: GPTQ, AWQ, HQQ, `TurboQuant` KV (3-bit), FP8 KV,
//! bf16 / fp16 conversions.
//!
//! The current `TurboQuant` surface is a binary-layout and packing contract.
//! FWHT encode/decode kernels land with the Phase 6 GPU implementation.
//!
//! ## Responsibility
//!
//! - Weight-only and activation-aware quantisation
//! - Dequantisation shims used by `praxis` matmul kernels
//! - KV-cache quantisation (`TurboQuant` 3-bit, FP8)
//! - `BitNet` 1.58-bit for extreme-mobile targets (thumos)
//!
//! Decisions on default dtype and primary quant scheme are closed by
//! research stream 4 before Phase 6.
#![deny(missing_docs)]

use half::f16;

pub mod error;
pub mod scheme;

pub use crate::error::{Error, Result};
pub use crate::scheme::TurboQuantScheme;

/// Number of scalar values represented by one `TurboQuant` block.
pub const TURBOQUANT_VALUES_PER_BLOCK: usize = 32;

/// Fixed head dimension supported by the `TurboQuant` FWHT contract.
pub const TURBOQUANT_HEAD_DIM: usize = 128;

/// Number of 32-value blocks in one 128-value head chunk.
pub const TURBOQUANT_BLOCKS_PER_HEAD: usize = TURBOQUANT_HEAD_DIM / TURBOQUANT_VALUES_PER_BLOCK;

/// Stored byte width of the fp16 norm prefix in every block.
pub const TURBOQUANT_NORM_BYTES: usize = 2;

/// Packed index bytes in one `turbo3_0` block.
pub const TURBO3_0_PACKED_BYTES_PER_BLOCK: usize = 12;

/// Total byte width of one `turbo3_0` block.
pub const TURBO3_0_BLOCK_BYTES: usize = TURBOQUANT_NORM_BYTES + TURBO3_0_PACKED_BYTES_PER_BLOCK;

/// Stored bytes for one 128-value head chunk encoded as `turbo3_0`.
pub const TURBO3_0_BYTES_PER_HEAD: usize = TURBO3_0_BLOCK_BYTES * TURBOQUANT_BLOCKS_PER_HEAD;

/// Packed index bytes in one `turbo4_0` block.
pub const TURBO4_0_PACKED_BYTES_PER_BLOCK: usize = 16;

/// Total byte width of one `turbo4_0` block.
pub const TURBO4_0_BLOCK_BYTES: usize = TURBOQUANT_NORM_BYTES + TURBO4_0_PACKED_BYTES_PER_BLOCK;

/// Stored bytes for one 128-value head chunk encoded as `turbo4_0`.
pub const TURBO4_0_BYTES_PER_HEAD: usize = TURBO4_0_BLOCK_BYTES * TURBOQUANT_BLOCKS_PER_HEAD;

pub(crate) const TURBO3_0_BITS_PER_INDEX: usize = 3;
const TURBO3_0_INDEX_MASK: u8 = 0x07;
pub(crate) const TURBO3_0_MAX_INDEX: u8 = 7;
pub(crate) const TURBO4_0_BITS_PER_INDEX: usize = 4;
const TURBO4_0_INDEX_MASK: u8 = 0x0f;
pub(crate) const TURBO4_0_MAX_INDEX: u8 = 15;
const ENCODE_REASON: &str = "FWHT Lloyd-Max encode parity is not implemented";
const DECODE_REASON: &str = "FWHT Lloyd-Max decode parity is not implemented";

/// Lloyd-Max Beta-distribution codebook for 3-bit `TurboQuant` (8 centroids).
/// Source: llama.cpp-turboquant convert.cu `dc_codebook_3bit[]` (MIT).
#[expect(
    clippy::excessive_precision,
    reason = "verbatim upstream f32 constants copied from llama.cpp-turboquant"
)]
pub const TURBO3_CODEBOOK: [f32; 8] = [
    -0.188_397_297_2,
    -0.118_139_905_9,
    -0.066_585_764_1,
    -0.021_604_475_1,
    0.021_604_146_1,
    0.066_585_452_0,
    0.118_139_628_1,
    0.188_397_074_8,
];

/// In-place Walsh-Hadamard transform over a 128-element f32 slice.
/// Normalized by 1/sqrt(128). CPU reference for the HIP kernel in Phase 6b GPU work.
///
/// # Panics
///
/// Panics if `data.len() != 128`.
pub fn fwht_128(data: &mut [f32; 128]) {
    // INVARIANT: data is exactly 128 elements, a power of two.
    let n = data.len();
    let mut h = 1usize;
    while h < n {
        for chunk in data.chunks_exact_mut(h * 2) {
            let (lo, hi) = chunk.split_at_mut(h);
            for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
                let x = *a;
                let y = *b;
                *a = x + y;
                *b = x - y;
            }
        }
        h *= 2;
    }
    #[expect(
        clippy::excessive_precision,
        reason = "full-precision literal for 1/sqrt(128) to match upstream kernel"
    )]
    let scale = 0.088_388_347_648_318_44_f32; // 1/sqrt(128)
    for v in &mut *data {
        *v *= scale;
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "codebook has at most 16 entries, index always fits in u8"
)]
fn nearest_centroid(value: f32, codebook: &[f32]) -> u8 {
    codebook
        .iter()
        .map(|&c| (value - c).abs())
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map_or(0, |(i, _)| i as u8)
}

/// Rust representation of the upstream `block_turbo3_0` binary layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Turbo3Block {
    norm_f16_le: [u8; TURBOQUANT_NORM_BYTES],
    packed_indices: [u8; TURBO3_0_PACKED_BYTES_PER_BLOCK],
}

impl Turbo3Block {
    /// Creates a `turbo3_0` block from the raw fp16 norm bytes and packed indices.
    #[must_use]
    pub const fn new(
        norm_f16_le: [u8; TURBOQUANT_NORM_BYTES],
        packed_indices: [u8; TURBO3_0_PACKED_BYTES_PER_BLOCK],
    ) -> Self {
        Self {
            norm_f16_le,
            packed_indices,
        }
    }

    /// Returns the stored fp16 norm bytes.
    #[must_use]
    pub const fn norm_f16_le(&self) -> [u8; TURBOQUANT_NORM_BYTES] {
        self.norm_f16_le
    }

    /// Returns the packed 3-bit index payload.
    #[must_use]
    pub const fn packed_indices(&self) -> [u8; TURBO3_0_PACKED_BYTES_PER_BLOCK] {
        self.packed_indices
    }

    /// Decomposes the block into its raw layout fields.
    #[must_use]
    pub const fn into_parts(
        self,
    ) -> (
        [u8; TURBOQUANT_NORM_BYTES],
        [u8; TURBO3_0_PACKED_BYTES_PER_BLOCK],
    ) {
        (self.norm_f16_le, self.packed_indices)
    }
}

/// Rust representation of the upstream `block_turbo4_0` binary layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Turbo4Block {
    norm_f16_le: [u8; TURBOQUANT_NORM_BYTES],
    packed_indices: [u8; TURBO4_0_PACKED_BYTES_PER_BLOCK],
}

impl Turbo4Block {
    /// Creates a `turbo4_0` block from the raw fp16 norm bytes and packed indices.
    #[must_use]
    pub const fn new(
        norm_f16_le: [u8; TURBOQUANT_NORM_BYTES],
        packed_indices: [u8; TURBO4_0_PACKED_BYTES_PER_BLOCK],
    ) -> Self {
        Self {
            norm_f16_le,
            packed_indices,
        }
    }

    /// Returns the stored fp16 norm bytes.
    #[must_use]
    pub const fn norm_f16_le(&self) -> [u8; TURBOQUANT_NORM_BYTES] {
        self.norm_f16_le
    }

    /// Returns the packed 4-bit index payload.
    #[must_use]
    pub const fn packed_indices(&self) -> [u8; TURBO4_0_PACKED_BYTES_PER_BLOCK] {
        self.packed_indices
    }

    /// Decomposes the block into its raw layout fields.
    #[must_use]
    pub const fn into_parts(
        self,
    ) -> (
        [u8; TURBOQUANT_NORM_BYTES],
        [u8; TURBO4_0_PACKED_BYTES_PER_BLOCK],
    ) {
        (self.norm_f16_le, self.packed_indices)
    }
}

/// Packs 32 `turbo3_0` codebook indices into the upstream 12-byte bitstream.
///
/// # Errors
///
/// Returns [`Error::IndexOutOfRange`] when an index is greater than 7.
pub fn pack_turbo3_indices(
    indices: &[u8; TURBOQUANT_VALUES_PER_BLOCK],
) -> Result<[u8; TURBO3_0_PACKED_BYTES_PER_BLOCK]> {
    let scheme = TurboQuantScheme::Turbo3_0;
    let mut packed_bits = 0_u128;

    for (position, value) in indices.iter().copied().enumerate() {
        check_index(scheme, position, value)?;

        let bit_offset = position * TURBO3_0_BITS_PER_INDEX;
        packed_bits |= u128::from(value) << bit_offset;
    }

    let mut packed = [0; TURBO3_0_PACKED_BYTES_PER_BLOCK];
    let bytes = packed_bits.to_le_bytes();
    for (packed_byte, source_byte) in packed.iter_mut().zip(bytes) {
        *packed_byte = source_byte;
    }

    Ok(packed)
}

/// Unpacks a `turbo3_0` 12-byte payload into 32 codebook indices.
#[must_use]
pub fn unpack_turbo3_indices(
    packed: &[u8; TURBO3_0_PACKED_BYTES_PER_BLOCK],
) -> [u8; TURBOQUANT_VALUES_PER_BLOCK] {
    let mut bytes = [0; 16];
    for (target_byte, source_byte) in bytes.iter_mut().zip(packed.iter().copied()) {
        *target_byte = source_byte;
    }

    let packed_bits = u128::from_le_bytes(bytes);
    let mut indices = [0; TURBOQUANT_VALUES_PER_BLOCK];

    for (position, index) in indices.iter_mut().enumerate() {
        let bit_offset = position * TURBO3_0_BITS_PER_INDEX;
        *index = narrow_masked_index(packed_bits >> bit_offset, TURBO3_0_INDEX_MASK);
    }

    indices
}

/// Packs 32 `turbo4_0` codebook indices into the upstream 16-byte nibble layout.
///
/// # Errors
///
/// Returns [`Error::IndexOutOfRange`] when an index is greater than 15.
pub fn pack_turbo4_indices(
    indices: &[u8; TURBOQUANT_VALUES_PER_BLOCK],
) -> Result<[u8; TURBO4_0_PACKED_BYTES_PER_BLOCK]> {
    let scheme = TurboQuantScheme::Turbo4_0;
    let mut packed_bits = 0_u128;

    for (position, value) in indices.iter().copied().enumerate() {
        check_index(scheme, position, value)?;
        let bit_offset = position * TURBO4_0_BITS_PER_INDEX;
        packed_bits |= u128::from(value) << bit_offset;
    }

    Ok(packed_bits.to_le_bytes())
}

/// Unpacks a `turbo4_0` 16-byte payload into 32 codebook indices.
#[must_use]
pub fn unpack_turbo4_indices(
    packed: &[u8; TURBO4_0_PACKED_BYTES_PER_BLOCK],
) -> [u8; TURBOQUANT_VALUES_PER_BLOCK] {
    let packed_bits = u128::from_le_bytes(*packed);
    let mut indices = [0; TURBOQUANT_VALUES_PER_BLOCK];

    for (position, index) in indices.iter_mut().enumerate() {
        let bit_offset = position * TURBO4_0_BITS_PER_INDEX;
        *index = narrow_masked_index(packed_bits >> bit_offset, TURBO4_0_INDEX_MASK);
    }

    indices
}

/// Encodes a 128-element `f32` slice into four [`Turbo3Block`]s.
///
/// Algorithm:
/// 1. Compute L2 norm of the 128 values.
/// 2. Normalize, apply FWHT, and scale by `1/sqrt(128)`.
/// 3. Quantise each coefficient to the nearest 3-bit centroid.
/// 4. Pack indices into the block layout.
///
/// The norm is stored redundantly in each block's `norm_f16_le` field.
///
/// # Errors
///
/// Returns [`Error::InvalidHeadDim`] unless `values.len()` is 128.
pub fn encode_turbo3_0_head(values: &[f32]) -> Result<[Turbo3Block; TURBOQUANT_BLOCKS_PER_HEAD]> {
    check_head_dim(values.len())?;

    let mut sum_sq = 0.0f32;
    for &v in values {
        sum_sq += v * v;
    }
    let norm = sum_sq.sqrt();
    let norm_bytes = f16::from_f32(norm).to_bits().to_le_bytes();

    let mut transformed = [0.0f32; TURBOQUANT_HEAD_DIM];
    if norm > 1e-12 {
        for (t, &v) in transformed.iter_mut().zip(values.iter()) {
            *t = v / norm;
        }
    }

    fwht_128(&mut transformed);

    let mut blocks = [Turbo3Block::new(
        [0; TURBOQUANT_NORM_BYTES],
        [0; TURBO3_0_PACKED_BYTES_PER_BLOCK],
    ); TURBOQUANT_BLOCKS_PER_HEAD];
    for (bi, block) in blocks.iter_mut().enumerate() {
        let mut indices = [0u8; TURBOQUANT_VALUES_PER_BLOCK];
        for (i, index) in indices.iter_mut().enumerate() {
            let idx = bi * TURBOQUANT_VALUES_PER_BLOCK + i;
            *index = nearest_centroid(
                transformed.get(idx).copied().unwrap_or(0.0),
                &TURBO3_CODEBOOK,
            );
        }
        let packed = pack_turbo3_indices(&indices)?;
        *block = Turbo3Block::new(norm_bytes, packed);
    }

    Ok(blocks)
}

/// Preflights a `turbo4_0` encode request for one 128-value head chunk.
///
/// # Errors
///
/// Returns [`Error::InvalidHeadDim`] unless `values.len()` is 128. Returns
/// [`Error::Unsupported`] for valid input until the FWHT/codebook path lands.
pub fn encode_turbo4_0_head(values: &[f32]) -> Result<[Turbo4Block; TURBOQUANT_BLOCKS_PER_HEAD]> {
    check_head_dim(values.len())?;
    Err(Error::Unsupported {
        operation: "turbo4_0 encode",
        reason: ENCODE_REASON,
    })
}

/// Decodes four [`Turbo3Block`]s into a 128-element `f32` vector.
///
/// Algorithm:
/// 1. Unpack indices and look up 3-bit centroids.
/// 2. Apply inverse FWHT (same butterfly, same normalisation).
/// 3. Scale by the stored norm.
///
/// # Errors
///
/// Returns [`Error::InvalidBlockCount`] unless `blocks.len()` is 4.
pub fn decode_turbo3_0_head(blocks: &[Turbo3Block]) -> Result<[f32; TURBOQUANT_HEAD_DIM]> {
    check_block_count(TurboQuantScheme::Turbo3_0, blocks.len())?;

    let mut transformed = [0.0f32; TURBOQUANT_HEAD_DIM];
    for (bi, block) in blocks.iter().enumerate() {
        let indices = unpack_turbo3_indices(&block.packed_indices());
        for (i, &index) in indices.iter().enumerate() {
            let idx = bi * TURBOQUANT_VALUES_PER_BLOCK + i;
            if let Some(slot) = transformed.get_mut(idx) {
                *slot = TURBO3_CODEBOOK
                    .get(usize::from(index))
                    .copied()
                    .unwrap_or(0.0);
            }
        }
    }

    fwht_128(&mut transformed);

    let Some(first_block) = blocks.first().copied() else {
        return Err(Error::InvalidBlockCount {
            scheme: TurboQuantScheme::Turbo3_0,
            got: blocks.len(),
            expected: TURBOQUANT_BLOCKS_PER_HEAD,
        });
    };
    let norm_bits = u16::from_le_bytes(first_block.norm_f16_le());
    let norm = f16::from_bits(norm_bits).to_f32();

    for v in &mut transformed {
        *v *= norm;
    }

    Ok(transformed)
}

/// Preflights a `turbo4_0` decode request for one 128-value head chunk.
///
/// # Errors
///
/// Returns [`Error::InvalidBlockCount`] unless `blocks.len()` is 4. Returns
/// [`Error::Unsupported`] for valid input until the FWHT/codebook path lands.
pub fn decode_turbo4_0_head(blocks: &[Turbo4Block]) -> Result<[f32; TURBOQUANT_HEAD_DIM]> {
    check_block_count(TurboQuantScheme::Turbo4_0, blocks.len())?;
    Err(Error::Unsupported {
        operation: "turbo4_0 decode",
        reason: DECODE_REASON,
    })
}

fn check_index(scheme: TurboQuantScheme, position: usize, value: u8) -> Result<()> {
    let max = scheme.max_index();
    if value > max {
        return Err(Error::IndexOutOfRange {
            scheme,
            position,
            value,
            max,
        });
    }

    Ok(())
}

fn check_head_dim(got: usize) -> Result<()> {
    if got != TURBOQUANT_HEAD_DIM {
        return Err(Error::InvalidHeadDim {
            got,
            expected: TURBOQUANT_HEAD_DIM,
        });
    }

    Ok(())
}

fn check_block_count(scheme: TurboQuantScheme, got: usize) -> Result<()> {
    if got != TURBOQUANT_BLOCKS_PER_HEAD {
        return Err(Error::InvalidBlockCount {
            scheme,
            got,
            expected: TURBOQUANT_BLOCKS_PER_HEAD,
        });
    }

    Ok(())
}

#[expect(
    clippy::manual_unwrap_or_default,
    reason = "the codebook mask guarantees the value fits in u8"
)]
fn narrow_masked_index(value: u128, mask: u8) -> u8 {
    match u8::try_from(value & u128::from(mask)) {
        Ok(index) => index,
        Err(_) => 0,
    }
}

#[cfg(test)]
const CRATE_NAME: &str = "quant";

#[cfg(test)]
mod tests {
    #![expect(
        clippy::cast_precision_loss,
        clippy::expect_used,
        reason = "test fixtures use small deterministic index->f32 conversions \
                  (indices < 128, well within f32's exact range) and expect() \
                  as the assertion mechanism"
    )]

    use core::mem::{align_of, size_of};

    use super::*;

    const NORM_BYTES: [u8; TURBOQUANT_NORM_BYTES] = [0x34, 0x12];

    #[test]
    fn crate_identity_matches_role() {
        assert_eq!(env!("CARGO_PKG_NAME"), CRATE_NAME);
    }

    #[test]
    fn turbo3_block_layout_matches_upstream_contract() {
        assert_eq!(size_of::<Turbo3Block>(), TURBO3_0_BLOCK_BYTES);
        assert_eq!(align_of::<Turbo3Block>(), 1);
        assert_eq!(TURBO3_0_BLOCK_BYTES, 14);
        assert_eq!(TURBO3_0_BYTES_PER_HEAD, 56);
    }

    #[test]
    fn turbo4_block_layout_matches_upstream_contract() {
        assert_eq!(size_of::<Turbo4Block>(), TURBO4_0_BLOCK_BYTES);
        assert_eq!(align_of::<Turbo4Block>(), 1);
        assert_eq!(TURBO4_0_BLOCK_BYTES, 18);
        assert_eq!(TURBO4_0_BYTES_PER_HEAD, 72);
    }

    #[test]
    fn turbo3_block_preserves_raw_parts() {
        let packed = [0xa5; TURBO3_0_PACKED_BYTES_PER_BLOCK];
        let block = Turbo3Block::new(NORM_BYTES, packed);

        assert_eq!(block.norm_f16_le(), NORM_BYTES);
        assert_eq!(block.packed_indices(), packed);
        assert_eq!(block.into_parts(), (NORM_BYTES, packed));
    }

    #[test]
    fn turbo4_block_preserves_raw_parts() {
        let packed = [0x5a; TURBO4_0_PACKED_BYTES_PER_BLOCK];
        let block = Turbo4Block::new(NORM_BYTES, packed);

        assert_eq!(block.norm_f16_le(), NORM_BYTES);
        assert_eq!(block.packed_indices(), packed);
        assert_eq!(block.into_parts(), (NORM_BYTES, packed));
    }

    #[test]
    fn turbo3_pack_matches_golden_pattern() -> Result<()> {
        let indices =
            cyclic_indices::<{ TURBOQUANT_VALUES_PER_BLOCK }, { TURBO3_0_MAX_INDEX + 1 }>();
        let expected = [
            0x88, 0xc6, 0xfa, 0x88, 0xc6, 0xfa, 0x88, 0xc6, 0xfa, 0x88, 0xc6, 0xfa,
        ];

        assert_eq!(pack_turbo3_indices(&indices)?, expected);
        assert_eq!(unpack_turbo3_indices(&expected), indices);

        Ok(())
    }

    #[test]
    fn turbo4_pack_matches_golden_pattern() -> Result<()> {
        let indices =
            cyclic_indices::<{ TURBOQUANT_VALUES_PER_BLOCK }, { TURBO4_0_MAX_INDEX + 1 }>();
        let expected = [
            0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe, 0x10, 0x32, 0x54, 0x76, 0x98, 0xba,
            0xdc, 0xfe,
        ];

        assert_eq!(pack_turbo4_indices(&indices)?, expected);
        assert_eq!(unpack_turbo4_indices(&expected), indices);

        Ok(())
    }

    #[test]
    fn turbo3_pack_rejects_out_of_range_index() {
        let mut indices = [0; TURBOQUANT_VALUES_PER_BLOCK];
        indices[5] = TURBO3_0_MAX_INDEX + 1;

        assert_eq!(
            pack_turbo3_indices(&indices),
            Err(Error::IndexOutOfRange {
                scheme: TurboQuantScheme::Turbo3_0,
                position: 5,
                value: 8,
                max: TURBO3_0_MAX_INDEX,
            })
        );
    }

    #[test]
    fn turbo4_pack_rejects_out_of_range_index() {
        let mut indices = [0; TURBOQUANT_VALUES_PER_BLOCK];
        indices[9] = TURBO4_0_MAX_INDEX + 1;

        assert_eq!(
            pack_turbo4_indices(&indices),
            Err(Error::IndexOutOfRange {
                scheme: TurboQuantScheme::Turbo4_0,
                position: 9,
                value: 16,
                max: TURBO4_0_MAX_INDEX,
            })
        );
    }

    #[test]
    fn encode_preflight_rejects_wrong_head_dim() {
        let err = encode_turbo3_0_head(&[]).err();

        assert_eq!(
            err,
            Some(Error::InvalidHeadDim {
                got: 0,
                expected: TURBOQUANT_HEAD_DIM,
            })
        );
    }

    #[test]
    fn encode_preflight_reports_precise_unsupported_path() {
        let values = [0.0; TURBOQUANT_HEAD_DIM];
        let err = encode_turbo4_0_head(&values).err();

        assert_eq!(
            err,
            Some(Error::Unsupported {
                operation: "turbo4_0 encode",
                reason: ENCODE_REASON,
            })
        );
    }

    #[test]
    fn decode_preflight_rejects_wrong_block_count() {
        let err = decode_turbo3_0_head(&[]).err();

        assert_eq!(
            err,
            Some(Error::InvalidBlockCount {
                scheme: TurboQuantScheme::Turbo3_0,
                got: 0,
                expected: TURBOQUANT_BLOCKS_PER_HEAD,
            })
        );
    }

    #[test]
    fn decode_preflight_reports_precise_unsupported_path() {
        let block = Turbo4Block::new(
            [0; TURBOQUANT_NORM_BYTES],
            [0; TURBO4_0_PACKED_BYTES_PER_BLOCK],
        );
        let blocks = [block; TURBOQUANT_BLOCKS_PER_HEAD];
        let err = decode_turbo4_0_head(&blocks).err();

        assert_eq!(
            err,
            Some(Error::Unsupported {
                operation: "turbo4_0 decode",
                reason: DECODE_REASON,
            })
        );
    }

    #[test]
    fn fwht_128_identity() {
        let mut data: [f32; 128] = core::array::from_fn(|i| (i as f32).sin() * 0.5);
        let original = data;
        fwht_128(&mut data);
        fwht_128(&mut data);
        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "FWHT^2 identity violated: {a} vs {b}");
        }
    }

    #[test]
    fn fwht_128_orthonormal() {
        let data: [f32; 128] = core::array::from_fn(|i| (i as f32).cos() * 0.3);
        let original_norm = data.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mut transformed = data;
        fwht_128(&mut transformed);
        let transformed_norm = transformed.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (original_norm - transformed_norm).abs() < 1e-5,
            "norm not preserved: {original_norm} vs {transformed_norm}"
        );
    }

    #[test]
    fn encode_decode_turbo3_roundtrip() {
        // WHY: use small-amplitude sinusoidal data so that post-FWHT coefficients
        // mostly fall inside the Lloyd-Max codebook range (~±0.19).
        let src: [f32; 128] = core::array::from_fn(|i| {
            ((i as f32) * 0.1).sin() * 0.1 + ((i as f32) * 0.03).cos() * 0.1
        });
        let blocks = encode_turbo3_0_head(&src).expect("encode should succeed");
        let decoded = decode_turbo3_0_head(&blocks).expect("decode should succeed");
        let mse = src
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / 128.0;
        assert!(mse < 0.05, "MSE too large: {mse}");
    }

    #[test]
    fn encode_rejects_wrong_head_dim() {
        let err = encode_turbo3_0_head(&[]);
        assert_eq!(
            err,
            Err(Error::InvalidHeadDim {
                got: 0,
                expected: TURBOQUANT_HEAD_DIM,
            })
        );
    }

    fn cyclic_indices<const N: usize, const PERIOD: u8>() -> [u8; N] {
        let mut indices = [0; N];
        let mut index = 0;
        let mut value = 0;

        while index < N {
            indices[index] = value;
            value += 1;
            if value == PERIOD {
                value = 0;
            }
            index += 1;
        }

        indices
    }
}
