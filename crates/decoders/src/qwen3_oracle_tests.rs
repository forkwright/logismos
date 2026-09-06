//! Independent f64 whole-model witness for bounded Qwen3 embedding execution.
//!
//! The composition contract is pinned to `ggml-org/llama.cpp` revision
//! `6a1a922d269908a29cbd4b49c27e6a8e7fd10fae`,
//! `src/models/qwen3.cpp:18-45,76-155`. This test owns its raw F32/Q8 decode
//! and numerical oracle and calls no production math, quantization, or `RoPE` helper.

use std::num::NonZeroU64;

use loader::gguf::{ArtifactByteLimit, Sha256Digest, VerifiedArtifact};
use test_fixtures::{RawGguf, RawMetadata, RawMetadataValue, RawTensor, serialize_raw_gguf};

use crate::Qwen3Weights;

type TestResult<T> = std::result::Result<T, String>;

const HIDDEN: usize = 32;
const QUERY_HEADS: usize = 8;
const KEY_VALUE_HEADS: usize = 2;
const HEAD_DIMENSION: usize = 8;
const QUERY_WIDTH: usize = QUERY_HEADS * HEAD_DIMENSION;
const KEY_VALUE_WIDTH: usize = KEY_VALUE_HEADS * HEAD_DIMENSION;
const FEED_FORWARD: usize = 96;
const BLOCKS: usize = 2;
const VOCABULARY: usize = 7;
const CONTEXT: usize = 5;
const EPSILON: f32 = 0.001;
const ROPE_BASE: f32 = 10_000.0;
const TOKEN_IDS: [u32; 3] = [1, 5, 2];
const F32_FORMAT: u32 = 0;
const Q8_0_FORMAT: u32 = 8;
const Q8_BLOCK_VALUES: usize = 32;
const Q8_BLOCK_BYTES: usize = 34;
const ABSOLUTE_TOLERANCE: f64 = 1.0e-3;
const RELATIVE_TOLERANCE: f64 = 2.0e-4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Storage {
    F32,
    Q8,
}

impl Storage {
    const fn tag(self) -> u32 {
        match self {
            Self::F32 => F32_FORMAT,
            Self::Q8 => Q8_0_FORMAT,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Fault {
    None,
    NonCausal,
    ModuloGqa,
    AdjacentRope,
    MissingQkNorm,
    ColumnMajorMatrices,
    SwappedFfnBranches,
    MissingAttentionResidual,
    MissingFfnResidual,
    MissingFinalNorm,
    FirstTokenPool,
}

impl Fault {
    const fn uses_causal_attention(self) -> bool {
        !matches!(self, Self::NonCausal)
    }

    const fn uses_modulo_gqa(self) -> bool {
        matches!(self, Self::ModuloGqa)
    }

    const fn uses_adjacent_rope(self) -> bool {
        matches!(self, Self::AdjacentRope)
    }

    const fn applies_qk_norm(self) -> bool {
        !matches!(self, Self::MissingQkNorm)
    }

    const fn uses_input_major_matrices(self) -> bool {
        matches!(self, Self::ColumnMajorMatrices)
    }

    const fn swaps_ffn_branches(self) -> bool {
        matches!(self, Self::SwappedFfnBranches)
    }

    const fn applies_attention_residual(self) -> bool {
        !matches!(self, Self::MissingAttentionResidual)
    }

    const fn applies_ffn_residual(self) -> bool {
        !matches!(self, Self::MissingFfnResidual)
    }

    const fn applies_final_norm(self) -> bool {
        !matches!(self, Self::MissingFinalNorm)
    }

    const fn pools_first_token(self) -> bool {
        matches!(self, Self::FirstTokenPool)
    }
}

#[derive(Debug)]
struct Matrix {
    input: usize,
    output: usize,
    values: Vec<f64>,
}

#[test]
fn mixed_q8_f32_execution_matches_independent_f64_model_and_falsifiers() -> TestResult<()> {
    let fixture = qwen3_fixture()?;
    let artifact = verify_fixture(&fixture)?;
    let weights = Qwen3Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;
    if weights.hidden_width() != HIDDEN {
        return Err("Qwen3 weights did not preserve the asymmetric hidden width".to_string());
    }
    let execution = weights
        .execution(CONTEXT)
        .map_err(|error| error.to_string())?;
    let actual = execution
        .last_hidden(&TOKEN_IDS)
        .map_err(|error| error.to_string())?;
    let expected = oracle_last_hidden(&fixture, &TOKEN_IDS, Fault::None)?;
    assert_f32_matches_f64(&actual, &expected, "whole-model result")?;

    for (name, fault) in [
        ("non-causal attention", Fault::NonCausal),
        ("modulo GQA mapping", Fault::ModuloGqa),
        ("adjacent-pair RoPE", Fault::AdjacentRope),
        ("missing Q/K RMS normalization", Fault::MissingQkNorm),
        (
            "input-major matrix interpretation",
            Fault::ColumnMajorMatrices,
        ),
        ("swapped SwiGLU gate/up roles", Fault::SwappedFfnBranches),
        (
            "missing attention residual",
            Fault::MissingAttentionResidual,
        ),
        ("missing FFN residual", Fault::MissingFfnResidual),
        ("missing final RMS normalization", Fault::MissingFinalNorm),
        (
            "first-token rather than last-token pooling",
            Fault::FirstTokenPool,
        ),
    ] {
        let incorrect = oracle_last_hidden(&fixture, &TOKEN_IDS, fault)?;
        assert_discriminated(&expected, &incorrect, name)?;
    }
    Ok(())
}

#[test]
fn late_invalid_token_refusal_leaves_stateless_execution_retryable() -> TestResult<()> {
    let fixture = qwen3_fixture()?;
    let artifact = verify_fixture(&fixture)?;
    let weights = Qwen3Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;
    let execution = weights
        .execution(CONTEXT)
        .map_err(|error| error.to_string())?;
    let mut invalid = TOKEN_IDS.to_vec();
    invalid.push(u32::MAX);
    if execution.last_hidden(&invalid).is_ok() {
        return Err("a late out-of-vocabulary token unexpectedly executed".to_string());
    }
    let retry = execution
        .last_hidden(&TOKEN_IDS)
        .map_err(|error| error.to_string())?;
    let fresh = weights
        .execution(CONTEXT)
        .map_err(|error| error.to_string())?
        .last_hidden(&TOKEN_IDS)
        .map_err(|error| error.to_string())?;
    if retry != fresh {
        return Err("late refusal changed a stateless executor's subsequent result".to_string());
    }
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "the raw fixture names every role and deliberately alternates F32/Q8 storage"
)]
fn qwen3_fixture() -> TestResult<RawGguf> {
    let mut tensors = vec![
        matrix_tensor("token_embd.weight", HIDDEN, VOCABULARY, Storage::Q8, 3)?,
        norm_tensor("output_norm.weight", HIDDEN, 97)?,
    ];
    for block in 0..BLOCKS {
        let base = u32::try_from(block)
            .map_err(|error| error.to_string())?
            .checked_mul(41)
            .and_then(|value| value.checked_add(11))
            .ok_or("fixture seed overflow")?;
        tensors.extend([
            norm_tensor(&block_name(block, "attn_norm.weight"), HIDDEN, base)?,
            norm_tensor(
                &block_name(block, "attn_q_norm.weight"),
                HEAD_DIMENSION,
                base + 1,
            )?,
            norm_tensor(
                &block_name(block, "attn_k_norm.weight"),
                HEAD_DIMENSION,
                base + 2,
            )?,
            norm_tensor(&block_name(block, "ffn_norm.weight"), HIDDEN, base + 3)?,
        ]);
        let formats = if block == 0 {
            [
                Storage::Q8,
                Storage::F32,
                Storage::Q8,
                Storage::F32,
                Storage::Q8,
                Storage::F32,
                Storage::Q8,
            ]
        } else {
            [
                Storage::F32,
                Storage::Q8,
                Storage::F32,
                Storage::Q8,
                Storage::F32,
                Storage::Q8,
                Storage::F32,
            ]
        };
        for (offset, (role, input, output, storage)) in [
            ("attn_q.weight", HIDDEN, QUERY_WIDTH, formats[0]),
            ("attn_k.weight", HIDDEN, KEY_VALUE_WIDTH, formats[1]),
            ("attn_v.weight", HIDDEN, KEY_VALUE_WIDTH, formats[2]),
            ("attn_output.weight", QUERY_WIDTH, HIDDEN, formats[3]),
            ("ffn_gate.weight", HIDDEN, FEED_FORWARD, formats[4]),
            ("ffn_up.weight", HIDDEN, FEED_FORWARD, formats[5]),
            ("ffn_down.weight", FEED_FORWARD, HIDDEN, formats[6]),
        ]
        .into_iter()
        .enumerate()
        {
            let seed = base
                .checked_add(u32::try_from(offset).map_err(|error| error.to_string())? + 7)
                .ok_or("matrix fixture seed overflow")?;
            tensors.push(matrix_tensor(
                &block_name(block, role),
                input,
                output,
                storage,
                seed,
            )?);
        }
    }
    Ok(RawGguf {
        metadata: vec![
            metadata_string("general.architecture", "qwen3"),
            metadata_u32("qwen3.block_count", BLOCKS)?,
            metadata_u32("qwen3.context_length", CONTEXT)?,
            metadata_u32("qwen3.embedding_length", HIDDEN)?,
            metadata_u32("qwen3.feed_forward_length", FEED_FORWARD)?,
            metadata_u32("qwen3.attention.head_count", QUERY_HEADS)?,
            metadata_u32("qwen3.attention.head_count_kv", KEY_VALUE_HEADS)?,
            metadata_u32("qwen3.attention.key_length", HEAD_DIMENSION)?,
            metadata_u32("qwen3.attention.value_length", HEAD_DIMENSION)?,
            metadata_f32("qwen3.attention.layer_norm_rms_epsilon", EPSILON),
            metadata_u32("qwen3.rope.dimension_count", HEAD_DIMENSION)?,
            metadata_f32("qwen3.rope.freq_base", ROPE_BASE),
            metadata_u32("qwen3.pooling_type", 3)?,
            RawMetadata {
                key: "tokenizer.ggml.tokens".to_string(),
                value: RawMetadataValue::StringArray(
                    (0..VOCABULARY)
                        .map(|index| format!("token-{index}"))
                        .collect(),
                ),
            },
        ],
        tensors,
    })
}

fn block_name(block: usize, role: &str) -> String {
    format!("blk.{block}.{role}")
}

fn metadata_string(key: &str, value: &str) -> RawMetadata {
    RawMetadata {
        key: key.to_string(),
        value: RawMetadataValue::String(value.to_string()),
    }
}

fn metadata_u32(key: &str, value: usize) -> TestResult<RawMetadata> {
    Ok(RawMetadata {
        key: key.to_string(),
        value: RawMetadataValue::U32(u32::try_from(value).map_err(|error| error.to_string())?),
    })
}

fn metadata_f32(key: &str, value: f32) -> RawMetadata {
    RawMetadata {
        key: key.to_string(),
        value: RawMetadataValue::F32(value),
    }
}

fn norm_tensor(name: &str, width: usize, seed: u32) -> TestResult<RawTensor> {
    let mut payload = Vec::new();
    for index in 0..width {
        let code = signed_code(seed, index, index)?;
        let value = 1.0_f32 + f32::from(code) * 0.007_812_5;
        payload.extend_from_slice(&value.to_le_bytes());
    }
    Ok(RawTensor {
        name: name.to_string(),
        dims: vec![u64_from_usize(width)?],
        format: F32_FORMAT,
        payload,
    })
}

fn matrix_tensor(
    name: &str,
    input: usize,
    output: usize,
    storage: Storage,
    seed: u32,
) -> TestResult<RawTensor> {
    let payload = match storage {
        Storage::F32 => f32_matrix_bytes(input, output, seed)?,
        Storage::Q8 => q8_matrix_bytes(input, output, seed)?,
    };
    Ok(RawTensor {
        name: name.to_string(),
        dims: vec![u64_from_usize(input)?, u64_from_usize(output)?],
        format: storage.tag(),
        payload,
    })
}

fn f32_matrix_bytes(input: usize, output: usize, seed: u32) -> TestResult<Vec<u8>> {
    let mut payload = Vec::new();
    for row in 0..output {
        for column in 0..input {
            let code = signed_code(seed, row, column)?;
            let value = f32::from(code) * 0.007_812_5;
            payload.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(payload)
}

fn q8_matrix_bytes(input: usize, output: usize, seed: u32) -> TestResult<Vec<u8>> {
    if !input.is_multiple_of(Q8_BLOCK_VALUES) {
        return Err("Q8 fixture input width must contain complete 32-value blocks".to_string());
    }
    let mut payload = Vec::new();
    for row in 0..output {
        for block in 0..input / Q8_BLOCK_VALUES {
            let row_u32 = u32::try_from(row).map_err(|error| error.to_string())?;
            let block_u32 = u32::try_from(block).map_err(|error| error.to_string())?;
            let selector = seed
                .checked_add(row_u32)
                .and_then(|value| value.checked_add(block_u32))
                .ok_or("Q8 scale selector overflow")?;
            let scale_bits = match selector % 3 {
                0 => 0x1c00_u16,
                1 => 0x2000_u16,
                _ => 0x2400_u16,
            };
            payload.extend_from_slice(&scale_bits.to_le_bytes());
            for lane in 0..Q8_BLOCK_VALUES {
                let column = block
                    .checked_mul(Q8_BLOCK_VALUES)
                    .and_then(|value| value.checked_add(lane))
                    .ok_or("Q8 fixture column overflow")?;
                payload.extend_from_slice(&signed_code(seed, row, column)?.to_le_bytes());
            }
        }
    }
    Ok(payload)
}

fn signed_code(seed: u32, row: usize, column: usize) -> TestResult<i8> {
    let row = u32::try_from(row).map_err(|error| error.to_string())?;
    let column = u32::try_from(column).map_err(|error| error.to_string())?;
    let mixed = seed
        .wrapping_mul(17)
        .wrapping_add(row.wrapping_mul(13))
        .wrapping_add(column.wrapping_mul(7))
        .wrapping_add(row.wrapping_mul(column).wrapping_mul(3));
    let code = i16::try_from(mixed % 23).map_err(|error| error.to_string())? - 11;
    i8::try_from(if code == 0 { 5 } else { code }).map_err(|error| error.to_string())
}

fn u64_from_usize(value: usize) -> TestResult<u64> {
    u64::try_from(value).map_err(|error| error.to_string())
}

fn verify_fixture(raw: &RawGguf) -> TestResult<VerifiedArtifact> {
    let serialized = serialize_raw_gguf(raw).map_err(|error| error.to_string())?;
    let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
    let path = directory.path().join("qwen3-independent-oracle.gguf");
    std::fs::write(&path, &serialized.bytes).map_err(|error| error.to_string())?;
    let limit = NonZeroU64::new(
        serialized
            .byte_len
            .checked_add(1)
            .ok_or("fixture byte limit overflow")?,
    )
    .ok_or("fixture byte limit cannot be zero")?;
    VerifiedArtifact::load(
        &path,
        Sha256Digest::from_bytes(serialized.sha256),
        ArtifactByteLimit::new(limit),
    )
    .map_err(|error| error.to_string())
}

fn oracle_last_hidden(raw: &RawGguf, token_ids: &[u32], fault: Fault) -> TestResult<Vec<f64>> {
    let embedding = decode_matrix(raw, "token_embd.weight")?;
    let mut hidden = Vec::new();
    for token_id in token_ids {
        let token = usize::try_from(*token_id).map_err(|error| error.to_string())?;
        hidden.push(matrix_row(&embedding, token)?.to_vec());
    }
    for block in 0..BLOCKS {
        hidden = oracle_block(raw, block, &hidden, fault)?;
    }
    let normalized = if fault.applies_final_norm() {
        let weights = decode_vector(raw, "output_norm.weight", HIDDEN)?;
        hidden
            .iter()
            .map(|row| rms_norm(row, &weights))
            .collect::<TestResult<Vec<_>>>()?
    } else {
        hidden
    };
    let selected = if fault.pools_first_token() {
        0
    } else {
        normalized
            .len()
            .checked_sub(1)
            .ok_or("oracle cannot pool an empty sequence")?
    };
    normalized
        .get(selected)
        .cloned()
        .ok_or_else(|| "oracle pooled row is outside the token sequence".to_string())
}

fn oracle_block(
    raw: &RawGguf,
    block: usize,
    hidden: &[Vec<f64>],
    fault: Fault,
) -> TestResult<Vec<Vec<f64>>> {
    let attention_norm = decode_vector(raw, &block_name(block, "attn_norm.weight"), HIDDEN)?;
    let query_norm = decode_vector(
        raw,
        &block_name(block, "attn_q_norm.weight"),
        HEAD_DIMENSION,
    )?;
    let key_norm = decode_vector(
        raw,
        &block_name(block, "attn_k_norm.weight"),
        HEAD_DIMENSION,
    )?;
    let query_matrix = decode_matrix(raw, &block_name(block, "attn_q.weight"))?;
    let key_matrix = decode_matrix(raw, &block_name(block, "attn_k.weight"))?;
    let value_matrix = decode_matrix(raw, &block_name(block, "attn_v.weight"))?;
    let output_matrix = decode_matrix(raw, &block_name(block, "attn_output.weight"))?;

    let mut queries = Vec::new();
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for (position, row) in hidden.iter().enumerate() {
        let normalized = rms_norm(row, &attention_norm)?;
        let mut query = project(&query_matrix, &normalized, fault)?;
        let mut key = project(&key_matrix, &normalized, fault)?;
        let value = project(&value_matrix, &normalized, fault)?;
        if fault.applies_qk_norm() {
            normalize_heads(&mut query, QUERY_HEADS, &query_norm)?;
            normalize_heads(&mut key, KEY_VALUE_HEADS, &key_norm)?;
        }
        apply_rope(&mut query, QUERY_HEADS, position, fault)?;
        apply_rope(&mut key, KEY_VALUE_HEADS, position, fault)?;
        queries.push(query);
        keys.push(key);
        values.push(value);
    }

    let mut post_attention = Vec::new();
    for token in 0..hidden.len() {
        let merged = oracle_attention(token, &queries, &keys, &values, fault)?;
        let projected = project(&output_matrix, &merged, fault)?;
        post_attention.push(if fault.applies_attention_residual() {
            add_vectors(
                hidden
                    .get(token)
                    .ok_or("attention residual token row is missing")?,
                &projected,
            )?
        } else {
            projected
        });
    }

    let ffn_norm = decode_vector(raw, &block_name(block, "ffn_norm.weight"), HIDDEN)?;
    let gate = decode_matrix(raw, &block_name(block, "ffn_gate.weight"))?;
    let up = decode_matrix(raw, &block_name(block, "ffn_up.weight"))?;
    let down = decode_matrix(raw, &block_name(block, "ffn_down.weight"))?;
    let mut output = Vec::new();
    for row in &post_attention {
        let normalized = rms_norm(row, &ffn_norm)?;
        let gate_values = project(&gate, &normalized, fault)?;
        let up_values = project(&up, &normalized, fault)?;
        let (activated, linear) = if fault.swaps_ffn_branches() {
            (up_values, gate_values)
        } else {
            (gate_values, up_values)
        };
        let fused = activated
            .iter()
            .zip(&linear)
            .map(|(gate_value, up_value)| silu(*gate_value) * up_value)
            .collect::<Vec<_>>();
        let projected = project(&down, &fused, fault)?;
        output.push(if fault.applies_ffn_residual() {
            add_vectors(row, &projected)?
        } else {
            projected
        });
    }
    finite_rows(&output, "oracle block output")?;
    Ok(output)
}

fn oracle_attention(
    token: usize,
    queries: &[Vec<f64>],
    keys: &[Vec<f64>],
    values: &[Vec<f64>],
    fault: Fault,
) -> TestResult<Vec<f64>> {
    let query = queries.get(token).ok_or("oracle query token is missing")?;
    let token_count = if fault.uses_causal_attention() {
        token + 1
    } else {
        keys.len()
    };
    let group = QUERY_HEADS / KEY_VALUE_HEADS;
    let scale = f64_from_usize(HEAD_DIMENSION)?.sqrt().recip();
    let mut merged = Vec::new();
    for query_head in 0..QUERY_HEADS {
        let key_value_head = if fault.uses_modulo_gqa() {
            query_head % KEY_VALUE_HEADS
        } else {
            query_head / group
        };
        let query_row = head_row(query, query_head)?;
        let mut scores = Vec::new();
        for key in keys.iter().take(token_count) {
            scores.push(dot(query_row, head_row(key, key_value_head)?)? * scale);
        }
        let maximum = scores
            .iter()
            .copied()
            .reduce(f64::max)
            .ok_or("oracle attention has no scores")?;
        let exponentials = scores
            .iter()
            .map(|score| (*score - maximum).exp())
            .collect::<Vec<_>>();
        let total = exponentials.iter().sum::<f64>();
        if !total.is_finite() || total <= 0.0 {
            return Err("oracle attention normalization is not finite positive".to_string());
        }
        for lane in 0..HEAD_DIMENSION {
            let mut output = 0.0;
            for (source_token, exponential) in exponentials.iter().enumerate() {
                let value = values
                    .get(source_token)
                    .ok_or("oracle attention value token is missing")?;
                let lane_value = head_row(value, key_value_head)?
                    .get(lane)
                    .copied()
                    .ok_or("oracle attention value lane is missing")?;
                output += exponential / total * lane_value;
            }
            if !output.is_finite() {
                return Err("oracle attention output became non-finite".to_string());
            }
            merged.push(output);
        }
    }
    Ok(merged)
}

fn normalize_heads(values: &mut [f64], heads: usize, weights: &[f64]) -> TestResult<()> {
    if values.len() != heads * HEAD_DIMENSION {
        return Err("oracle head normalization received the wrong width".to_string());
    }
    for head in values.chunks_exact_mut(HEAD_DIMENSION) {
        let normalized = rms_norm(head, weights)?;
        head.copy_from_slice(&normalized);
    }
    Ok(())
}

fn apply_rope(values: &mut [f64], heads: usize, position: usize, fault: Fault) -> TestResult<()> {
    if values.len() != heads * HEAD_DIMENSION {
        return Err("oracle RoPE received the wrong head geometry".to_string());
    }
    let position = f64_from_usize(position)?;
    let head_dimension = f64_from_usize(HEAD_DIMENSION)?;
    for head in values.chunks_exact_mut(HEAD_DIMENSION) {
        for pair in 0..HEAD_DIMENSION / 2 {
            let pair_value = f64_from_usize(pair)?;
            let angle = position / f64::from(ROPE_BASE).powf(2.0 * pair_value / head_dimension);
            let (left, right) = if fault.uses_adjacent_rope() {
                (pair * 2, pair * 2 + 1)
            } else {
                (pair, pair + HEAD_DIMENSION / 2)
            };
            let left_value = *head.get(left).ok_or("oracle RoPE left lane is missing")?;
            let right_value = *head.get(right).ok_or("oracle RoPE right lane is missing")?;
            *head
                .get_mut(left)
                .ok_or("oracle RoPE mutable left lane is missing")? =
                left_value * angle.cos() - right_value * angle.sin();
            *head
                .get_mut(right)
                .ok_or("oracle RoPE mutable right lane is missing")? =
                left_value * angle.sin() + right_value * angle.cos();
        }
    }
    Ok(())
}

fn rms_norm(values: &[f64], weights: &[f64]) -> TestResult<Vec<f64>> {
    if values.len() != weights.len() || values.is_empty() {
        return Err("oracle RMS normalization geometry is inconsistent".to_string());
    }
    let width = f64_from_usize(values.len())?;
    let mean_square = values.iter().map(|value| value * value).sum::<f64>() / width;
    let scale = (mean_square + f64::from(EPSILON)).sqrt().recip();
    let output = values
        .iter()
        .zip(weights)
        .map(|(value, weight)| value * scale * weight)
        .collect::<Vec<_>>();
    finite(&output, "oracle RMS normalization")?;
    Ok(output)
}

fn silu(value: f64) -> f64 {
    value / (1.0 + (-value).exp())
}

fn add_vectors(left: &[f64], right: &[f64]) -> TestResult<Vec<f64>> {
    if left.len() != right.len() {
        return Err("oracle residual geometry is inconsistent".to_string());
    }
    let output = left
        .iter()
        .zip(right)
        .map(|(left, right)| left + right)
        .collect::<Vec<_>>();
    finite(&output, "oracle residual")?;
    Ok(output)
}

fn dot(left: &[f64], right: &[f64]) -> TestResult<f64> {
    if left.len() != right.len() {
        return Err("oracle dot-product geometry is inconsistent".to_string());
    }
    let value = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    value
        .is_finite()
        .then_some(value)
        .ok_or_else(|| "oracle dot product became non-finite".to_string())
}

fn project(matrix: &Matrix, input: &[f64], fault: Fault) -> TestResult<Vec<f64>> {
    if input.len() != matrix.input {
        return Err("oracle projection input width is inconsistent".to_string());
    }
    let mut output = Vec::new();
    for row in 0..matrix.output {
        let mut value = 0.0;
        for (column, activation) in input.iter().enumerate() {
            let index = if fault.uses_input_major_matrices() {
                column
                    .checked_mul(matrix.output)
                    .and_then(|value| value.checked_add(row))
            } else {
                row.checked_mul(matrix.input)
                    .and_then(|value| value.checked_add(column))
            }
            .ok_or("oracle matrix index overflow")?;
            let weight = matrix
                .values
                .get(index)
                .copied()
                .ok_or("oracle matrix weight is missing")?;
            value += weight * activation;
        }
        if !value.is_finite() {
            return Err("oracle projection became non-finite".to_string());
        }
        output.push(value);
    }
    Ok(output)
}

fn matrix_row(matrix: &Matrix, row: usize) -> TestResult<&[f64]> {
    let start = row
        .checked_mul(matrix.input)
        .ok_or("oracle matrix row offset overflow")?;
    let end = start
        .checked_add(matrix.input)
        .ok_or("oracle matrix row end overflow")?;
    matrix
        .values
        .get(start..end)
        .ok_or_else(|| "oracle matrix row is outside the payload".to_string())
}

fn head_row(values: &[f64], head: usize) -> TestResult<&[f64]> {
    let start = head
        .checked_mul(HEAD_DIMENSION)
        .ok_or("oracle head offset overflow")?;
    let end = start
        .checked_add(HEAD_DIMENSION)
        .ok_or("oracle head end overflow")?;
    values
        .get(start..end)
        .ok_or_else(|| "oracle head row is outside the projected values".to_string())
}

fn decode_vector(raw: &RawGguf, name: &str, width: usize) -> TestResult<Vec<f64>> {
    let tensor = raw_tensor(raw, name)?;
    if tensor.format != F32_FORMAT || tensor.dims != [u64_from_usize(width)?] {
        return Err(format!(
            "oracle vector `{name}` has unexpected geometry or storage"
        ));
    }
    let values = decode_f32_values(&tensor.payload)?;
    if values.len() != width {
        return Err(format!("oracle vector `{name}` has an incomplete payload"));
    }
    Ok(values)
}

fn decode_matrix(raw: &RawGguf, name: &str) -> TestResult<Matrix> {
    let tensor = raw_tensor(raw, name)?;
    let [input, output] = tensor.dims.as_slice() else {
        return Err(format!("oracle matrix `{name}` is not rank two"));
    };
    let input = usize::try_from(*input).map_err(|error| error.to_string())?;
    let output = usize::try_from(*output).map_err(|error| error.to_string())?;
    let values = match tensor.format {
        F32_FORMAT => decode_f32_values(&tensor.payload)?,
        Q8_0_FORMAT => decode_q8_values(&tensor.payload, input, output)?,
        other => {
            return Err(format!(
                "oracle matrix `{name}` uses unsupported format {other}"
            ));
        }
    };
    let expected = input
        .checked_mul(output)
        .ok_or("oracle matrix element count overflow")?;
    if values.len() != expected {
        return Err(format!(
            "oracle matrix `{name}` decoded {} values, expected {expected}",
            values.len()
        ));
    }
    finite(&values, "oracle matrix decode")?;
    Ok(Matrix {
        input,
        output,
        values,
    })
}

fn raw_tensor<'fixture>(raw: &'fixture RawGguf, name: &str) -> TestResult<&'fixture RawTensor> {
    let mut matching = raw.tensors.iter().filter(|tensor| tensor.name == name);
    let tensor = matching
        .next()
        .ok_or_else(|| format!("oracle fixture is missing tensor `{name}`"))?;
    if matching.next().is_some() {
        return Err(format!("oracle fixture duplicates tensor `{name}`"));
    }
    Ok(tensor)
}

fn decode_f32_values(payload: &[u8]) -> TestResult<Vec<f64>> {
    if !payload.len().is_multiple_of(4) {
        return Err("oracle F32 payload has trailing bytes".to_string());
    }
    payload
        .chunks_exact(4)
        .map(|bytes| {
            let bytes: [u8; 4] = bytes
                .try_into()
                .map_err(|_| "oracle F32 chunk is not four bytes".to_string())?;
            let value = f32::from_le_bytes(bytes);
            value
                .is_finite()
                .then_some(f64::from(value))
                .ok_or_else(|| "oracle F32 payload contains a non-finite value".to_string())
        })
        .collect()
}

fn decode_q8_values(payload: &[u8], input: usize, output: usize) -> TestResult<Vec<f64>> {
    if !input.is_multiple_of(Q8_BLOCK_VALUES) {
        return Err("oracle Q8 input width is not block aligned".to_string());
    }
    let blocks_per_row = input / Q8_BLOCK_VALUES;
    let expected_blocks = blocks_per_row
        .checked_mul(output)
        .ok_or("oracle Q8 block count overflow")?;
    let expected_bytes = expected_blocks
        .checked_mul(Q8_BLOCK_BYTES)
        .ok_or("oracle Q8 byte count overflow")?;
    if payload.len() != expected_bytes {
        return Err("oracle Q8 payload byte length is inconsistent".to_string());
    }
    let mut values = Vec::new();
    for block in payload.chunks_exact(Q8_BLOCK_BYTES) {
        let scale_bytes: [u8; 2] = block
            .get(..2)
            .ok_or("oracle Q8 scale bytes are missing")?
            .try_into()
            .map_err(|_| "oracle Q8 scale is not two bytes".to_string())?;
        let scale = decode_binary16(u16::from_le_bytes(scale_bytes))?;
        for byte in block
            .get(2..)
            .ok_or("oracle Q8 signed values are missing")?
        {
            let signed = i8::from_le_bytes([*byte]);
            values.push(scale * f64::from(signed));
        }
    }
    Ok(values)
}

fn decode_binary16(bits: u16) -> TestResult<f64> {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;
    let magnitude = match exponent {
        0 => f64::from(fraction) * 2.0_f64.powi(-24),
        0x1f => return Err("oracle Q8 scale is non-finite".to_string()),
        _ => (1.0 + f64::from(fraction) / 1024.0) * 2.0_f64.powi(i32::from(exponent) - 15),
    };
    Ok(sign * magnitude)
}

fn finite(values: &[f64], stage: &str) -> TestResult<()> {
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!("{stage} is non-finite at index {index}: {value}"));
    }
    Ok(())
}

fn finite_rows(rows: &[Vec<f64>], stage: &str) -> TestResult<()> {
    for (row, values) in rows.iter().enumerate() {
        finite(values, &format!("{stage} row {row}"))?;
    }
    Ok(())
}

fn f64_from_usize(value: usize) -> TestResult<f64> {
    let exact = u32::try_from(value).map_err(|error| error.to_string())?;
    Ok(f64::from(exact))
}

fn tolerance(left: f64, right: f64) -> f64 {
    ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * left.abs().max(right.abs())
}

fn assert_f32_matches_f64(actual: &[f32], expected: &[f64], label: &str) -> TestResult<()> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label} width differs: actual {}, expected {}",
            actual.len(),
            expected.len()
        ));
    }
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let actual = f64::from(*actual);
        if !actual.is_finite() || !expected.is_finite() {
            return Err(format!(
                "{label} is non-finite at {index}: actual {actual}, expected {expected}"
            ));
        }
        let delta = (actual - expected).abs();
        let allowed = tolerance(actual, *expected);
        if delta > allowed {
            return Err(format!(
                "{label} differs at {index}: actual {actual}, expected {expected}, delta {delta}, tolerance {allowed}"
            ));
        }
    }
    Ok(())
}

fn assert_discriminated(expected: &[f64], incorrect: &[f64], label: &str) -> TestResult<()> {
    if expected.len() != incorrect.len() || expected.is_empty() {
        return Err(format!(
            "{label} falsifier has inconsistent output geometry"
        ));
    }
    let mut largest_ratio = 0.0_f64;
    for (correct, wrong) in expected.iter().zip(incorrect) {
        if !correct.is_finite() || !wrong.is_finite() {
            return Err(format!("{label} falsifier produced a non-finite value"));
        }
        let threshold = 2.0 * tolerance(*correct, *wrong);
        if threshold > 0.0 {
            largest_ratio = largest_ratio.max((correct - wrong).abs() / threshold);
        }
    }
    if largest_ratio <= 1.0 {
        return Err(format!(
            "{label} was not distinguished beyond twice the production envelope; largest ratio {largest_ratio}"
        ));
    }
    Ok(())
}
