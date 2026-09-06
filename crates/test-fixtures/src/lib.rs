//! Deterministic, dev-only GGUF fixtures shared by native execution tests.

use sha2::{Digest, Sha256};

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

/// Build a structurally valid, all-F32 Qwen3.5 hybrid GGUF for test use.
///
/// # Errors
///
/// Returns an error when token ids do not name the supplied vocabulary or a
/// checked serialized size cannot be represented.
pub fn build_qwen35_fixture(config: &Qwen35FixtureConfig) -> Result<SyntheticGguf, String> {
    validate_config(config)?;
    let vocabulary = u64::try_from(config.tokens.len()).map_err(|error| error.to_string())?;
    let mut metadata = vec![
        Meta::String("general.architecture", "qwen35".to_string()),
        Meta::U32("qwen35.block_count", 4),
        Meta::U32("qwen35.nextn_predict_layers", 0),
        Meta::U32("qwen35.full_attention_interval", 4),
        Meta::U32("qwen35.embedding_length", 3),
        Meta::U32("qwen35.feed_forward_length", 5),
        Meta::U32("qwen35.attention.head_count", 2),
        Meta::U32("qwen35.attention.head_count_kv", 1),
        Meta::U32("qwen35.attention.key_length", 2),
        Meta::U32("qwen35.attention.value_length", 2),
        Meta::U32("qwen35.ssm.conv_kernel", 2),
        Meta::U32("qwen35.ssm.inner_size", 8),
        Meta::U32("qwen35.ssm.state_size", 2),
        Meta::U32("qwen35.ssm.time_step_rank", 4),
        Meta::U32("qwen35.ssm.group_count", 2),
        Meta::F32("qwen35.attention.layer_norm_rms_epsilon", 0.001),
        Meta::StringArray("tokenizer.ggml.tokens", config.tokens.clone()),
        Meta::U32("tokenizer.ggml.bos_token_id", config.bos_token_id),
        Meta::U32("tokenizer.ggml.eos_token_id", config.eos_token_id),
        Meta::Bool("tokenizer.ggml.add_bos_token", config.add_bos),
        Meta::Bool("tokenizer.ggml.add_eos_token", config.add_eos),
        Meta::String("tokenizer.chat_template", config.chat_template.clone()),
        Meta::U32("general.alignment", 32),
        Meta::U32("qwen35.context_length", 8),
        Meta::I32Array("qwen35.rope.dimension_sections", vec![1, 0, 0, 0]),
        Meta::F32("qwen35.rope.freq_base", 10_000.0),
        Meta::String("qwen35.rope.scaling.type", "none".to_string()),
    ];
    metadata.shrink_to_fit();
    let mut tensors = Vec::new();
    tensor(
        &mut tensors,
        "token_embd.weight",
        vec![HIDDEN, vocabulary],
        1.0,
    );
    tensor(&mut tensors, "output_norm.weight", vec![HIDDEN], 1.0);
    tensor(&mut tensors, "output.weight", vec![HIDDEN, vocabulary], 0.0);
    set_output_row(
        &mut tensors,
        usize::try_from(config.greedy_token_id).map_err(|error| error.to_string())?,
    )?;
    for block in 0..MAIN_BLOCKS {
        if block == 3 {
            full_block(&mut tensors, block);
        } else {
            recurrent_block(&mut tensors, block);
        }
    }
    let bytes = serialize(&metadata, &tensors)?;
    let byte_len = u64::try_from(bytes.len()).map_err(|error| error.to_string())?;
    Ok(SyntheticGguf {
        sha256: Sha256::digest(&bytes).into(),
        byte_len,
        bytes,
    })
}

fn validate_config(config: &Qwen35FixtureConfig) -> Result<(), String> {
    let length = u32::try_from(config.tokens.len()).map_err(|error| error.to_string())?;
    for (field, value) in [
        ("bos_token_id", config.bos_token_id),
        ("eos_token_id", config.eos_token_id),
        ("greedy_token_id", config.greedy_token_id),
    ] {
        if value >= length {
            return Err(format!("{field} must name one configured token"));
        }
    }
    Ok(())
}

#[derive(Clone)]
enum Meta {
    U32(&'static str, u32),
    F32(&'static str, f32),
    Bool(&'static str, bool),
    String(&'static str, String),
    StringArray(&'static str, Vec<String>),
    I32Array(&'static str, Vec<i32>),
}
impl Meta {
    fn key(&self) -> &'static str {
        match self {
            Self::U32(k, _)
            | Self::F32(k, _)
            | Self::Bool(k, _)
            | Self::String(k, _)
            | Self::StringArray(k, _)
            | Self::I32Array(k, _) => k,
        }
    }
}
struct Tensor {
    name: String,
    dims: Vec<u64>,
    values: Vec<f32>,
}
fn tensor(tensors: &mut Vec<Tensor>, name: &str, dims: Vec<u64>, value: f32) {
    let count = dims.iter().product::<u64>() as usize;
    tensors.push(Tensor {
        name: name.to_string(),
        dims,
        values: vec![value; count],
    });
}
fn set_output_row(tensors: &mut [Tensor], row: usize) -> Result<(), String> {
    let tensor = tensors
        .iter_mut()
        .find(|tensor| tensor.name == "output.weight")
        .ok_or_else(|| "missing output".to_string())?;
    let start = row * 3;
    tensor.values[start..start + 3].copy_from_slice(&[1.0, 1.0, 1.0]);
    Ok(())
}
fn name(block: u64, role: &str) -> String {
    format!("blk.{block}.{role}")
}
fn ffn(t: &mut Vec<Tensor>, b: u64) {
    tensor(
        t,
        &name(b, "ffn_down.weight"),
        vec![FEED_FORWARD, HIDDEN],
        0.0,
    );
    tensor(
        t,
        &name(b, "ffn_gate.weight"),
        vec![HIDDEN, FEED_FORWARD],
        0.0,
    );
    tensor(
        t,
        &name(b, "ffn_up.weight"),
        vec![HIDDEN, FEED_FORWARD],
        0.0,
    );
}
fn recurrent_block(t: &mut Vec<Tensor>, b: u64) {
    tensor(t, &name(b, "attn_gate.weight"), vec![HIDDEN, INNER], 0.0);
    tensor(t, &name(b, "attn_norm.weight"), vec![HIDDEN], 1.0);
    tensor(t, &name(b, "attn_qkv.weight"), vec![HIDDEN, 16], 0.0);
    ffn(t, b);
    tensor(t, &name(b, "post_attention_norm.weight"), vec![HIDDEN], 1.0);
    tensor(t, &name(b, "ssm_a"), vec![TIME_STEP_RANK], -1.0);
    tensor(
        t,
        &name(b, "ssm_alpha.weight"),
        vec![HIDDEN, TIME_STEP_RANK],
        0.0,
    );
    tensor(
        t,
        &name(b, "ssm_beta.weight"),
        vec![HIDDEN, TIME_STEP_RANK],
        0.0,
    );
    tensor(t, &name(b, "ssm_conv1d.weight"), vec![CONV_KERNEL, 16], 0.0);
    tensor(t, &name(b, "ssm_dt.bias"), vec![TIME_STEP_RANK], 0.0);
    tensor(t, &name(b, "ssm_norm.weight"), vec![STATE], 1.0);
    tensor(t, &name(b, "ssm_out.weight"), vec![INNER, HIDDEN], 0.0);
}
fn full_block(t: &mut Vec<Tensor>, b: u64) {
    tensor(t, &name(b, "attn_k.weight"), vec![HIDDEN, HEAD_WIDTH], 0.0);
    tensor(t, &name(b, "attn_k_norm.weight"), vec![HEAD_WIDTH], 1.0);
    tensor(t, &name(b, "attn_norm.weight"), vec![HIDDEN], 1.0);
    tensor(t, &name(b, "attn_output.weight"), vec![4, HIDDEN], 0.0);
    tensor(t, &name(b, "attn_q.weight"), vec![HIDDEN, 8], 0.0);
    tensor(t, &name(b, "attn_q_norm.weight"), vec![HEAD_WIDTH], 1.0);
    tensor(t, &name(b, "attn_v.weight"), vec![HIDDEN, HEAD_WIDTH], 0.0);
    ffn(t, b);
    tensor(t, &name(b, "post_attention_norm.weight"), vec![HIDDEN], 1.0);
}

fn serialize(metadata: &[Meta], tensors: &[Tensor]) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for meta in metadata {
        string(&mut bytes, meta.key())?;
        match meta {
            Meta::U32(_, v) => {
                bytes.extend_from_slice(&4_u32.to_le_bytes());
                bytes.extend_from_slice(&v.to_le_bytes())
            }
            Meta::F32(_, v) => {
                bytes.extend_from_slice(&6_u32.to_le_bytes());
                bytes.extend_from_slice(&v.to_le_bytes())
            }
            Meta::Bool(_, v) => {
                bytes.extend_from_slice(&7_u32.to_le_bytes());
                bytes.push(u8::from(*v))
            }
            Meta::String(_, v) => {
                bytes.extend_from_slice(&8_u32.to_le_bytes());
                string(&mut bytes, v)?
            }
            Meta::StringArray(_, values) => {
                bytes.extend_from_slice(&9_u32.to_le_bytes());
                bytes.extend_from_slice(&8_u32.to_le_bytes());
                bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
                for value in values {
                    string(&mut bytes, value)?
                }
            }
            Meta::I32Array(_, values) => {
                bytes.extend_from_slice(&9_u32.to_le_bytes());
                bytes.extend_from_slice(&5_u32.to_le_bytes());
                bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes())
                }
            }
        }
    }
    let mut offsets = Vec::new();
    let mut offset = 0_u64;
    for tensor in tensors {
        offset = align(offset)?;
        offsets.push(offset);
        offset = offset
            .checked_add(
                (tensor.values.len() as u64)
                    .checked_mul(4)
                    .ok_or_else(|| "tensor bytes overflowed".to_string())?,
            )
            .ok_or_else(|| "tensor offset overflowed".to_string())?;
    }
    for (tensor, offset) in tensors.iter().zip(offsets) {
        string(&mut bytes, &tensor.name)?;
        bytes.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
        for dim in &tensor.dims {
            bytes.extend_from_slice(&dim.to_le_bytes())
        }
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes())
    }
    pad(&mut bytes)?;
    for tensor in tensors {
        pad(&mut bytes)?;
        for value in &tensor.values {
            bytes.extend_from_slice(&value.to_le_bytes())
        }
    }
    Ok(bytes)
}
fn string(bytes: &mut Vec<u8>, value: &str) -> Result<(), String> {
    let length = u64::try_from(value.len()).map_err(|error| error.to_string())?;
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
}
fn align(value: u64) -> Result<u64, String> {
    value
        .checked_add(ALIGNMENT - 1)
        .map(|value| value / ALIGNMENT * ALIGNMENT)
        .ok_or_else(|| "GGUF alignment overflowed".to_string())
}
fn pad(bytes: &mut Vec<u8>) -> Result<(), String> {
    let length = u64::try_from(bytes.len()).map_err(|error| error.to_string())?;
    let padded = align(length)?;
    bytes.extend(std::iter::repeat_n(
        0_u8,
        usize::try_from(padded - length).map_err(|error| error.to_string())?,
    ));
    Ok(())
}
