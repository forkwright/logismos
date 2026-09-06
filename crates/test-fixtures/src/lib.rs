//! Deterministic, dev-only GGUF fixtures shared by native execution tests.

#![deny(unsafe_op_in_unsafe_fn)]

use sha2::{Digest, Sha256};
use snafu::Snafu;

const ALIGNMENT: u64 = 32;
const HIDDEN: u64 = 3;
const FEED_FORWARD: u64 = 5;
const HEADS: u64 = 2;
const KEY_VALUE_HEADS: u64 = 1;
const HEAD_WIDTH: u64 = 2;
const INNER: u64 = 8;
const STATE: u64 = 2;
const TIME_STEP_RANK: u64 = 4;
const GROUP_COUNT: u64 = 2;
const CONV_KERNEL: u64 = 2;
const MAIN_BLOCKS: u64 = 4;
const CONTEXT: u32 = 8;
const GGML_TYPE_F32: u32 = 0;

struct Location;

impl Location {
    #[track_caller]
    fn caller() -> snafu::Location {
        std::panic::Location::caller()
    }
}

/// Errors while constructing a bounded synthetic fixture.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum FixtureError {
    /// A supplied token id does not name the configured vocabulary.
    #[snafu(display("{field} token id {token_id} is outside vocabulary length {vocabulary}"))]
    InvalidTokenId {
        /// Config field that contained the invalid id.
        field: &'static str,
        /// Caller-provided token id.
        token_id: u32,
        /// Number of configured tokens.
        vocabulary: u32,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// A descriptor count or serialized extent cannot fit its GGUF field.
    #[snafu(display("fixture {context} cannot be represented"))]
    NotRepresentable {
        /// What could not be represented.
        context: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Checked layout arithmetic overflowed.
    #[snafu(display("fixture {context} overflowed"))]
    Overflow {
        /// Arithmetic operation that overflowed.
        context: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// The fixture factory's output tensor is structurally inconsistent.
    #[snafu(display("fixture output tensor is malformed: {reason}"))]
    InvalidOutput {
        /// Structural invariant that was violated.
        reason: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}

/// Exact token and template controls for one synthetic Qwen3.5 GGUF.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35FixtureConfig {
    /// Vocabulary in exact token-id order.
    pub tokens: Vec<String>,
    /// Beginning-of-sequence token id.
    pub bos_token_id: u32,
    /// End-of-sequence token id.
    pub eos_token_id: u32,
    /// Whether the tokenizer contract adds BOS.
    pub add_bos: bool,
    /// Whether the tokenizer contract adds EOS.
    pub add_eos: bool,
    /// Chat template metadata value.
    pub chat_template: String,
    /// Output row made uniquely maximal by the fixture.
    pub greedy_token_id: u32,
}

impl Default for Qwen35FixtureConfig {
    fn default() -> Self {
        Self {
            tokens: vec![
                "[UNK]".to_string(),
                "<bos>".to_string(),
                "<eos>".to_string(),
                "hello".to_string(),
                "assistant".to_string(),
            ],
            bos_token_id: 1,
            eos_token_id: 2,
            add_bos: true,
            add_eos: false,
            chat_template: "{{ messages }}".to_string(),
            greedy_token_id: 4,
        }
    }
}

/// Serialized synthetic GGUF plus its independently computed identity facts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SyntheticGguf {
    /// Complete GGUF v3 bytes.
    pub bytes: Vec<u8>,
    /// SHA-256 of [`Self::bytes`].
    pub sha256: [u8; 32],
    /// Exact byte length of [`Self::bytes`].
    pub byte_len: u64,
}

/// Raw GGUF metadata value for intentionally varied synthetic artifacts.
#[derive(Clone, Debug, PartialEq)]
pub enum RawMetadataValue {
    /// Unsigned 32-bit scalar.
    U32(u32),
    /// Single-precision scalar.
    F32(f32),
    /// Boolean scalar.
    Bool(bool),
    /// UTF-8 string.
    String(String),
    /// UTF-8 string array.
    StringArray(Vec<String>),
    /// Signed 32-bit array.
    I32Array(Vec<i32>),
}

/// One raw metadata descriptor.
#[derive(Clone, Debug, PartialEq)]
pub struct RawMetadata {
    /// GGUF key.
    pub key: String,
    /// Serialized value.
    pub value: RawMetadataValue,
}

/// One raw tensor descriptor and its complete serialized payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawTensor {
    /// GGUF tensor name.
    pub name: String,
    /// Logical GGUF dimensions.
    pub dims: Vec<u64>,
    /// GGML format tag, for example `0` for F32 or a quantized type tag.
    pub format: u32,
    /// Exact serialized bytes in tensor-offset order.
    pub payload: Vec<u8>,
}

/// Mutable synthetic GGUF source before one canonical serialization pass.
#[derive(Clone, Debug, PartialEq)]
pub struct RawGguf {
    /// Metadata descriptors in serialized order.
    pub metadata: Vec<RawMetadata>,
    /// Tensor descriptors in serialized order.
    pub tensors: Vec<RawTensor>,
}

/// Serialize mutable synthetic GGUF descriptors without decoding payloads.
///
/// This deliberately accepts arbitrary format tags and payload bytes: refusal
/// tests can model malformed tensor data without a second serializer.
///
/// # Errors
///
/// Returns an error when a descriptor count or aligned extent cannot be
/// represented by GGUF's on-disk fields.
pub fn serialize_raw_gguf(raw: &RawGguf) -> Result<SyntheticGguf, FixtureError> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&u64_from_usize(raw.tensors.len(), "tensor count")?.to_le_bytes());
    bytes.extend_from_slice(&u64_from_usize(raw.metadata.len(), "metadata count")?.to_le_bytes());
    for entry in &raw.metadata {
        append_raw_metadata(&mut bytes, entry)?;
    }

    let mut offsets = Vec::with_capacity(raw.tensors.len());
    let mut offset = 0_u64;
    for tensor in &raw.tensors {
        offset = align(offset)?;
        offsets.push(offset);
        offset = offset
            .checked_add(u64_from_usize(
                tensor.payload.len(),
                "tensor payload length",
            )?)
            .ok_or_else(|| FixtureError::Overflow {
                context: "tensor offset",
                location: Location::caller(),
            })?;
    }
    for (tensor, offset) in raw.tensors.iter().zip(offsets) {
        string(&mut bytes, &tensor.name)?;
        bytes.extend_from_slice(
            &u32_from_usize(tensor.dims.len(), "tensor dimension count")?.to_le_bytes(),
        );
        for dimension in &tensor.dims {
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
        bytes.extend_from_slice(&tensor.format.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
    }
    pad(&mut bytes)?;
    for tensor in &raw.tensors {
        pad(&mut bytes)?;
        bytes.extend_from_slice(&tensor.payload);
    }
    let byte_len = u64_from_usize(bytes.len(), "serialized byte length")?;
    Ok(SyntheticGguf {
        sha256: Sha256::digest(&bytes).into(),
        byte_len,
        bytes,
    })
}

/// Build a structurally valid, all-F32 Qwen3.5 hybrid GGUF for test use.
///
/// # Errors
///
/// Returns an error when token ids do not name the supplied vocabulary or a
/// checked serialized size cannot be represented.
pub fn build_qwen35_fixture(config: &Qwen35FixtureConfig) -> Result<SyntheticGguf, FixtureError> {
    serialize_raw_gguf(&raw_qwen35_fixture(config)?)
}

/// Build mutable descriptors for the standard synthetic Qwen3.5 hybrid GGUF.
///
/// Consumers may alter metadata, shapes, format tags, or payload bytes before
/// passing the result to [`serialize_raw_gguf`] for negative-path tests.
///
/// # Errors
///
/// Returns an error when token ids do not name the supplied vocabulary or a
/// required F32 payload size cannot be represented.
pub fn raw_qwen35_fixture(config: &Qwen35FixtureConfig) -> Result<RawGguf, FixtureError> {
    validate_config(config)?;
    let vocabulary = u64_from_usize(config.tokens.len(), "vocabulary length")?;
    let mut raw = RawGguf {
        metadata: qwen35_metadata(config)?,
        tensors: Vec::new(),
    };

    push_f32_tensor(
        &mut raw.tensors,
        "token_embd.weight",
        vec![HIDDEN, vocabulary],
        1.0,
    )?;
    push_f32_tensor(&mut raw.tensors, "output_norm.weight", vec![HIDDEN], 1.0)?;
    push_f32_tensor(
        &mut raw.tensors,
        "output.weight",
        vec![HIDDEN, vocabulary],
        0.0,
    )?;
    set_output_row(
        &mut raw.tensors,
        usize_from_u32(config.greedy_token_id, "greedy token id")?,
    )?;
    for block in 0..MAIN_BLOCKS {
        if block + 1 == MAIN_BLOCKS {
            full_block(&mut raw.tensors, block)?;
        } else {
            recurrent_block(&mut raw.tensors, block)?;
        }
    }
    Ok(raw)
}

fn qwen35_metadata(config: &Qwen35FixtureConfig) -> Result<Vec<RawMetadata>, FixtureError> {
    Ok(vec![
        metadata_string("general.architecture", "qwen35"),
        metadata_u32(
            "qwen35.block_count",
            u32_from_u64(MAIN_BLOCKS, "block count")?,
        ),
        metadata_u32("qwen35.nextn_predict_layers", 0),
        metadata_u32(
            "qwen35.full_attention_interval",
            u32_from_u64(MAIN_BLOCKS, "attention interval")?,
        ),
        metadata_u32(
            "qwen35.embedding_length",
            u32_from_u64(HIDDEN, "hidden size")?,
        ),
        metadata_u32(
            "qwen35.feed_forward_length",
            u32_from_u64(FEED_FORWARD, "feed-forward size")?,
        ),
        metadata_u32(
            "qwen35.attention.head_count",
            u32_from_u64(HEADS, "head count")?,
        ),
        metadata_u32(
            "qwen35.attention.head_count_kv",
            u32_from_u64(KEY_VALUE_HEADS, "KV head count")?,
        ),
        metadata_u32(
            "qwen35.attention.key_length",
            u32_from_u64(HEAD_WIDTH, "key width")?,
        ),
        metadata_u32(
            "qwen35.attention.value_length",
            u32_from_u64(HEAD_WIDTH, "value width")?,
        ),
        metadata_u32(
            "qwen35.ssm.conv_kernel",
            u32_from_u64(CONV_KERNEL, "kernel")?,
        ),
        metadata_u32("qwen35.ssm.inner_size", u32_from_u64(INNER, "inner size")?),
        metadata_u32("qwen35.ssm.state_size", u32_from_u64(STATE, "state size")?),
        metadata_u32(
            "qwen35.ssm.time_step_rank",
            u32_from_u64(TIME_STEP_RANK, "time-step rank")?,
        ),
        metadata_u32(
            "qwen35.ssm.group_count",
            u32_from_u64(GROUP_COUNT, "group count")?,
        ),
        metadata_f32("qwen35.attention.layer_norm_rms_epsilon", 0.001),
        RawMetadata {
            key: "tokenizer.ggml.tokens".to_string(),
            value: RawMetadataValue::StringArray(config.tokens.clone()),
        },
        metadata_u32("tokenizer.ggml.bos_token_id", config.bos_token_id),
        metadata_u32("tokenizer.ggml.eos_token_id", config.eos_token_id),
        metadata_bool("tokenizer.ggml.add_bos_token", config.add_bos),
        metadata_bool("tokenizer.ggml.add_eos_token", config.add_eos),
        metadata_string("tokenizer.chat_template", &config.chat_template),
        metadata_u32("general.alignment", u32_from_u64(ALIGNMENT, "alignment")?),
        metadata_u32("qwen35.context_length", CONTEXT),
        RawMetadata {
            key: "qwen35.rope.dimension_sections".to_string(),
            value: RawMetadataValue::I32Array(vec![1, 0, 0, 0]),
        },
        metadata_f32("qwen35.rope.freq_base", 10_000.0),
        metadata_string("qwen35.rope.scaling.type", "none"),
    ])
}

fn validate_config(config: &Qwen35FixtureConfig) -> Result<(), FixtureError> {
    let vocabulary = u32_from_usize(config.tokens.len(), "vocabulary length")?;
    for (field, token_id) in [
        ("bos_token_id", config.bos_token_id),
        ("eos_token_id", config.eos_token_id),
        ("greedy_token_id", config.greedy_token_id),
    ] {
        if token_id >= vocabulary {
            return Err(FixtureError::InvalidTokenId {
                field,
                token_id,
                vocabulary,
                location: Location::caller(),
            });
        }
    }
    Ok(())
}

fn metadata_u32(key: &str, value: u32) -> RawMetadata {
    RawMetadata {
        key: key.to_string(),
        value: RawMetadataValue::U32(value),
    }
}

fn metadata_f32(key: &str, value: f32) -> RawMetadata {
    RawMetadata {
        key: key.to_string(),
        value: RawMetadataValue::F32(value),
    }
}

fn metadata_bool(key: &str, value: bool) -> RawMetadata {
    RawMetadata {
        key: key.to_string(),
        value: RawMetadataValue::Bool(value),
    }
}

fn metadata_string(key: &str, value: &str) -> RawMetadata {
    RawMetadata {
        key: key.to_string(),
        value: RawMetadataValue::String(value.to_string()),
    }
}

fn push_f32_tensor(
    tensors: &mut Vec<RawTensor>,
    name: &str,
    dims: Vec<u64>,
    value: f32,
) -> Result<(), FixtureError> {
    tensors.push(RawTensor {
        name: name.to_string(),
        payload: f32_payload(&dims, value)?,
        dims,
        format: GGML_TYPE_F32,
    });
    Ok(())
}

fn f32_payload(dims: &[u64], value: f32) -> Result<Vec<u8>, FixtureError> {
    let values = dims.iter().try_fold(1_u64, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| FixtureError::Overflow {
                context: "F32 tensor element count",
                location: Location::caller(),
            })
    })?;
    let bytes = values
        .checked_mul(4)
        .ok_or_else(|| FixtureError::Overflow {
            context: "F32 tensor byte count",
            location: Location::caller(),
        })?;
    let byte_len = usize_from_u64(bytes, "F32 tensor byte count")?;
    let mut payload = Vec::with_capacity(byte_len);
    for _ in 0..values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    Ok(payload)
}

fn set_output_row(tensors: &mut [RawTensor], row: usize) -> Result<(), FixtureError> {
    let output = tensors
        .iter_mut()
        .find(|tensor| tensor.name == "output.weight")
        .ok_or_else(|| FixtureError::InvalidOutput {
            reason: "output.weight is absent",
            location: Location::caller(),
        })?;
    if output.format != GGML_TYPE_F32 || output.dims.len() != 2 || output.dims[0] != HIDDEN {
        return Err(FixtureError::InvalidOutput {
            reason: "output.weight is not an F32 hidden-by-vocabulary matrix",
            location: Location::caller(),
        });
    }
    let rows = usize_from_u64(output.dims[1], "output vocabulary")?;
    if row >= rows {
        return Err(FixtureError::InvalidOutput {
            reason: "chosen output row is absent",
            location: Location::caller(),
        });
    }
    let width = usize_from_u64(HIDDEN, "output hidden width")?;
    let start = row
        .checked_mul(width)
        .ok_or_else(|| FixtureError::Overflow {
            context: "output row start",
            location: Location::caller(),
        })?;
    let end = start
        .checked_add(width)
        .ok_or_else(|| FixtureError::Overflow {
            context: "output row end",
            location: Location::caller(),
        })?;
    let byte_start = start.checked_mul(4).ok_or_else(|| FixtureError::Overflow {
        context: "output row byte start",
        location: Location::caller(),
    })?;
    let byte_end = end.checked_mul(4).ok_or_else(|| FixtureError::Overflow {
        context: "output row byte end",
        location: Location::caller(),
    })?;
    let selected = output
        .payload
        .get_mut(byte_start..byte_end)
        .ok_or_else(|| FixtureError::InvalidOutput {
            reason: "output payload is shorter than its dimensions",
            location: Location::caller(),
        })?;
    for cell in selected.chunks_exact_mut(4) {
        cell.copy_from_slice(&1.0_f32.to_le_bytes());
    }
    Ok(())
}

fn name(block: u64, role: &str) -> String {
    format!("blk.{block}.{role}")
}

fn ffn(tensors: &mut Vec<RawTensor>, block: u64) -> Result<(), FixtureError> {
    push_f32_tensor(
        tensors,
        &name(block, "ffn_down.weight"),
        vec![FEED_FORWARD, HIDDEN],
        0.0,
    )?;
    push_f32_tensor(
        tensors,
        &name(block, "ffn_gate.weight"),
        vec![HIDDEN, FEED_FORWARD],
        0.0,
    )?;
    push_f32_tensor(
        tensors,
        &name(block, "ffn_up.weight"),
        vec![HIDDEN, FEED_FORWARD],
        0.0,
    )
}

fn recurrent_block(tensors: &mut Vec<RawTensor>, block: u64) -> Result<(), FixtureError> {
    push_f32_tensor(
        tensors,
        &name(block, "attn_gate.weight"),
        vec![HIDDEN, INNER],
        0.0,
    )?;
    push_f32_tensor(tensors, &name(block, "attn_norm.weight"), vec![HIDDEN], 1.0)?;
    push_f32_tensor(
        tensors,
        &name(block, "attn_qkv.weight"),
        vec![HIDDEN, INNER * 2],
        0.0,
    )?;
    ffn(tensors, block)?;
    push_f32_tensor(
        tensors,
        &name(block, "post_attention_norm.weight"),
        vec![HIDDEN],
        1.0,
    )?;
    push_f32_tensor(tensors, &name(block, "ssm_a"), vec![TIME_STEP_RANK], -1.0)?;
    push_f32_tensor(
        tensors,
        &name(block, "ssm_alpha.weight"),
        vec![HIDDEN, TIME_STEP_RANK],
        0.0,
    )?;
    push_f32_tensor(
        tensors,
        &name(block, "ssm_beta.weight"),
        vec![HIDDEN, TIME_STEP_RANK],
        0.0,
    )?;
    push_f32_tensor(
        tensors,
        &name(block, "ssm_conv1d.weight"),
        vec![CONV_KERNEL, INNER * 2],
        0.0,
    )?;
    push_f32_tensor(
        tensors,
        &name(block, "ssm_dt.bias"),
        vec![TIME_STEP_RANK],
        0.0,
    )?;
    push_f32_tensor(tensors, &name(block, "ssm_norm.weight"), vec![STATE], 1.0)?;
    push_f32_tensor(
        tensors,
        &name(block, "ssm_out.weight"),
        vec![INNER, HIDDEN],
        0.0,
    )
}

fn full_block(tensors: &mut Vec<RawTensor>, block: u64) -> Result<(), FixtureError> {
    push_f32_tensor(
        tensors,
        &name(block, "attn_k.weight"),
        vec![HIDDEN, HEAD_WIDTH],
        0.0,
    )?;
    push_f32_tensor(
        tensors,
        &name(block, "attn_k_norm.weight"),
        vec![HEAD_WIDTH],
        1.0,
    )?;
    push_f32_tensor(tensors, &name(block, "attn_norm.weight"), vec![HIDDEN], 1.0)?;
    push_f32_tensor(
        tensors,
        &name(block, "attn_output.weight"),
        vec![HEADS * HEAD_WIDTH, HIDDEN],
        0.0,
    )?;
    push_f32_tensor(
        tensors,
        &name(block, "attn_q.weight"),
        vec![HIDDEN, HEADS * HEAD_WIDTH * 2],
        0.0,
    )?;
    push_f32_tensor(
        tensors,
        &name(block, "attn_q_norm.weight"),
        vec![HEAD_WIDTH],
        1.0,
    )?;
    push_f32_tensor(
        tensors,
        &name(block, "attn_v.weight"),
        vec![HIDDEN, HEAD_WIDTH],
        0.0,
    )?;
    ffn(tensors, block)?;
    push_f32_tensor(
        tensors,
        &name(block, "post_attention_norm.weight"),
        vec![HIDDEN],
        1.0,
    )
}

fn append_raw_metadata(bytes: &mut Vec<u8>, entry: &RawMetadata) -> Result<(), FixtureError> {
    string(bytes, &entry.key)?;
    match &entry.value {
        RawMetadataValue::U32(value) => {
            bytes.extend_from_slice(&4_u32.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        RawMetadataValue::F32(value) => {
            bytes.extend_from_slice(&6_u32.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        RawMetadataValue::Bool(value) => {
            bytes.extend_from_slice(&7_u32.to_le_bytes());
            bytes.push(u8::from(*value));
        }
        RawMetadataValue::String(value) => {
            bytes.extend_from_slice(&8_u32.to_le_bytes());
            string(bytes, value)?;
        }
        RawMetadataValue::StringArray(values) => {
            bytes.extend_from_slice(&9_u32.to_le_bytes());
            bytes.extend_from_slice(&8_u32.to_le_bytes());
            bytes.extend_from_slice(
                &u64_from_usize(values.len(), "string array length")?.to_le_bytes(),
            );
            for value in values {
                string(bytes, value)?;
            }
        }
        RawMetadataValue::I32Array(values) => {
            bytes.extend_from_slice(&9_u32.to_le_bytes());
            bytes.extend_from_slice(&5_u32.to_le_bytes());
            bytes.extend_from_slice(
                &u64_from_usize(values.len(), "I32 array length")?.to_le_bytes(),
            );
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    Ok(())
}

fn string(bytes: &mut Vec<u8>, value: &str) -> Result<(), FixtureError> {
    bytes.extend_from_slice(&u64_from_usize(value.len(), "string length")?.to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
}

fn align(value: u64) -> Result<u64, FixtureError> {
    value
        .checked_add(ALIGNMENT - 1)
        .map(|value| value / ALIGNMENT * ALIGNMENT)
        .ok_or_else(|| FixtureError::Overflow {
            context: "GGUF alignment",
            location: Location::caller(),
        })
}

fn pad(bytes: &mut Vec<u8>) -> Result<(), FixtureError> {
    let length = u64_from_usize(bytes.len(), "serialized byte length")?;
    let padded = align(length)?;
    let padding = padded
        .checked_sub(length)
        .ok_or_else(|| FixtureError::Overflow {
            context: "GGUF padding",
            location: Location::caller(),
        })?;
    bytes.extend(std::iter::repeat_n(
        0_u8,
        usize_from_u64(padding, "GGUF padding length")?,
    ));
    Ok(())
}

fn u64_from_usize(value: usize, context: &'static str) -> Result<u64, FixtureError> {
    u64::try_from(value).map_err(|_| FixtureError::NotRepresentable {
        context,
        location: Location::caller(),
    })
}
fn u32_from_usize(value: usize, context: &'static str) -> Result<u32, FixtureError> {
    u32::try_from(value).map_err(|_| FixtureError::NotRepresentable {
        context,
        location: Location::caller(),
    })
}
fn u32_from_u64(value: u64, context: &'static str) -> Result<u32, FixtureError> {
    u32::try_from(value).map_err(|_| FixtureError::NotRepresentable {
        context,
        location: Location::caller(),
    })
}
fn usize_from_u64(value: u64, context: &'static str) -> Result<usize, FixtureError> {
    usize::try_from(value).map_err(|_| FixtureError::NotRepresentable {
        context,
        location: Location::caller(),
    })
}
fn usize_from_u32(value: u32, context: &'static str) -> Result<usize, FixtureError> {
    usize::try_from(value).map_err(|_| FixtureError::NotRepresentable {
        context,
        location: Location::caller(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        FixtureError, Qwen35FixtureConfig, RawGguf, RawMetadata, RawMetadataValue, RawTensor,
        build_qwen35_fixture, raw_qwen35_fixture, serialize_raw_gguf,
    };

    #[test]
    fn qwen_fixture_identity_is_deterministic() -> Result<(), Box<dyn std::error::Error>> {
        let config = Qwen35FixtureConfig::default();
        let first = build_qwen35_fixture(&config)?;
        let second = build_qwen35_fixture(&config)?;
        assert_eq!(first, second);
        assert_eq!(first.byte_len, u64::try_from(first.bytes.len())?);
        assert_eq!(first.bytes.get(..8), Some(&b"GGUF\x03\0\0\0"[..]));
        Ok(())
    }

    #[test]
    fn qwen_fixture_rejects_out_of_range_token_ids() -> Result<(), Box<dyn std::error::Error>> {
        let mut config = Qwen35FixtureConfig::default();
        config.greedy_token_id = u32::try_from(config.tokens.len())?;
        let error = raw_qwen35_fixture(&config)
            .err()
            .ok_or_else(|| std::io::Error::other("fixture unexpectedly accepted bad token id"))?;
        assert!(matches!(
            error,
            FixtureError::InvalidTokenId {
                field: "greedy_token_id",
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn raw_serializer_preserves_descriptor_layout_and_payload()
    -> Result<(), Box<dyn std::error::Error>> {
        let raw = RawGguf {
            metadata: vec![RawMetadata {
                key: "fixture.answer".to_string(),
                value: RawMetadataValue::U32(42),
            }],
            tensors: vec![
                RawTensor {
                    name: "first".to_string(),
                    dims: vec![3],
                    format: 24,
                    payload: vec![1, 2, 3],
                },
                RawTensor {
                    name: "second".to_string(),
                    dims: vec![1],
                    format: 16,
                    payload: vec![4, 5],
                },
            ],
        };
        let fixture = serialize_raw_gguf(&raw)?;
        assert_eq!(fixture.bytes.get(..4), Some(&b"GGUF"[..]));
        assert_eq!(u64_at(&fixture.bytes, 8)?, 2);
        assert_eq!(u64_at(&fixture.bytes, 83)?, 0);
        assert_eq!(u64_at(&fixture.bytes, 121)?, 32);
        assert_eq!(fixture.bytes.get(160..163), Some(&[1, 2, 3][..]));
        assert_eq!(fixture.bytes.get(192..194), Some(&[4, 5][..]));
        Ok(())
    }

    fn u64_at(bytes: &[u8], offset: usize) -> Result<u64, Box<dyn std::error::Error>> {
        let end = offset
            .checked_add(8)
            .ok_or_else(|| std::io::Error::other("test offset overflowed"))?;
        let value: [u8; 8] = bytes
            .get(offset..end)
            .ok_or_else(|| std::io::Error::other("test layout ended early"))?
            .try_into()?;
        Ok(u64::from_le_bytes(value))
    }
}
