//! Bounded native CPU Qwen3 embedding execution over one verified GGUF payload.

use std::collections::{HashMap, HashSet};

use loader::gguf::{GgmlType, MetaValue, MetaValueType, VerifiedArtifact};
use num_traits::ToPrimitive;
use snafu::ResultExt;

use crate::Result;
use crate::error::{
    Qwen3AllocationSnafu, Qwen3ArithmeticSnafu, Qwen3CpuSnafu, Qwen3ExecutionSnafu,
    Qwen3MetadataSnafu, Qwen3TensorSnafu,
};
use crate::matrix::CheckedMatrix;

const ARCHITECTURE: &str = "general.architecture";
const BLOCK_COUNT: &str = "qwen3.block_count";
const CONTEXT_LENGTH: &str = "qwen3.context_length";
const HIDDEN: &str = "qwen3.embedding_length";
const FEED_FORWARD: &str = "qwen3.feed_forward_length";
const HEADS: &str = "qwen3.attention.head_count";
const KV_HEADS: &str = "qwen3.attention.head_count_kv";
const KEY_LENGTH: &str = "qwen3.attention.key_length";
const VALUE_LENGTH: &str = "qwen3.attention.value_length";
const RMS_EPSILON: &str = "qwen3.attention.layer_norm_rms_epsilon";
const CAUSAL: &str = "qwen3.attention.causal";
const ROPE_DIMENSION: &str = "qwen3.rope.dimension_count";
const ROPE_BASE: &str = "qwen3.rope.freq_base";
const ROPE_SCALING_TYPE: &str = "qwen3.rope.scaling.type";
const ROPE_SCALING_FACTOR: &str = "qwen3.rope.scaling.factor";
const POOLING_TYPE: &str = "qwen3.pooling_type";

const TOKEN_EMBEDDING: &str = "token_embd.weight";
const OUTPUT_NORM: &str = "output_norm.weight";
const LAST_POOLING_TYPE: u64 = 3;

/// One verified Qwen3 embedding payload with its checked causal geometry.
#[derive(Debug)]
pub struct Qwen3Weights<'artifact> {
    payload: &'artifact VerifiedArtifact,
    layout: Layout,
}

impl<'artifact> Qwen3Weights<'artifact> {
    /// Bind one digest-verified GGUF payload to the bounded Qwen3 embedding profile.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] when metadata, tensor roles, shapes, or the
    /// bounded no-output-head profile are not satisfied.
    pub fn try_from_verified(payload: &'artifact VerifiedArtifact) -> Result<Self> {
        let layout = Layout::from_artifact(payload)?;
        validate_inventory(payload.observation().tensor_descriptors(), layout)?;
        Ok(Self { payload, layout })
    }

    /// Return the artifact-derived hidden-vector width.
    #[must_use]
    pub const fn hidden_width(&self) -> usize {
        self.layout.hidden
    }

    /// Return the artifact-derived maximum context length.
    #[must_use]
    pub const fn max_context(&self) -> usize {
        self.layout.context
    }

    /// Create one stateless bounded CPU embedding executor.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] when `max_context` is zero or exceeds the
    /// verified artifact's declared context length.
    pub fn execution(&self, max_context: usize) -> Result<Qwen3Execution<'_, 'artifact>> {
        if max_context == 0 || max_context > self.layout.context {
            return Qwen3ExecutionSnafu {
                requested: max_context,
                rule: "max context must be nonzero and no greater than the artifact context",
            }
            .fail();
        }
        Ok(Qwen3Execution {
            weights: self,
            max_context,
        })
    }
}

/// Stateless Qwen3 causal embedding execution with an explicit caller context bound.
#[derive(Debug)]
pub struct Qwen3Execution<'weights, 'artifact> {
    weights: &'weights Qwen3Weights<'artifact>,
    max_context: usize,
}

impl Qwen3Execution<'_, '_> {
    /// Execute token IDs through every causal block and return the final RMS-normalized last row.
    ///
    /// The input contains no padding: every supplied ID is a valid token row
    /// and the final supplied ID selects the returned hidden vector. Pooling,
    /// prompts, tokenizer policy, and output L2 normalization remain outside
    /// this family execution boundary.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] without exposing partial hidden states when an
    /// input, artifact row, allocation, or finite arithmetic check fails.
    pub fn last_hidden(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() || token_ids.len() > self.max_context {
            return Qwen3ExecutionSnafu {
                requested: token_ids.len(),
                rule: "token IDs must be nonempty and fit the caller context bound",
            }
            .fail();
        }
        let embedding = CheckedMatrix::from_payload(self.weights.payload, TOKEN_EMBEDDING)?;
        let mut hidden = reserve(
            "token hidden rows",
            product(token_ids.len(), self.weights.layout.hidden)?,
        )?;
        for token_id in token_ids {
            let token = usize::try_from(*token_id).map_err(|_| {
                Qwen3ExecutionSnafu {
                    requested: usize::MAX,
                    rule: "token ID must fit usize",
                }
                .build()
            })?;
            hidden.extend(embedding.decode_row(token)?);
        }
        for block in 0..self.weights.layout.blocks {
            self.run_block(block, &mut hidden)?;
        }
        let final_norm = read_f32_vector(
            self.weights.payload,
            OUTPUT_NORM,
            self.weights.layout.hidden,
        )?;
        let normalized = kernels::cpu_f32::rms_norm(
            &hidden,
            &final_norm,
            token_ids.len(),
            self.weights.layout.hidden,
            self.weights.layout.epsilon,
        )
        .context(Qwen3CpuSnafu)?;
        let start = product(token_ids.len() - 1, self.weights.layout.hidden)?;
        let result = normalized.get(start..).ok_or_else(|| {
            Qwen3ExecutionSnafu {
                requested: start,
                rule: "final hidden row must fit normalized token rows",
            }
            .build()
        })?;
        finite(result, "final RMS norm")?;
        let mut output = reserve("final hidden row", result.len())?;
        output.extend_from_slice(result);
        Ok(output)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the checked causal-attention and FFN order is one source-defined transformer block"
    )]
    fn run_block(&self, block: usize, hidden: &mut [f32]) -> Result<()> {
        let layout = self.weights.layout;
        let tokens = hidden.len() / layout.hidden;
        let attn_norm = read_f32_vector(
            self.weights.payload,
            &block_name(block, "attn_norm.weight"),
            layout.hidden,
        )?;
        let q_norm = read_f32_vector(
            self.weights.payload,
            &block_name(block, "attn_q_norm.weight"),
            layout.head_dim,
        )?;
        let k_norm = read_f32_vector(
            self.weights.payload,
            &block_name(block, "attn_k_norm.weight"),
            layout.head_dim,
        )?;
        let q =
            CheckedMatrix::from_payload(self.weights.payload, &block_name(block, "attn_q.weight"))?;
        let k =
            CheckedMatrix::from_payload(self.weights.payload, &block_name(block, "attn_k.weight"))?;
        let v =
            CheckedMatrix::from_payload(self.weights.payload, &block_name(block, "attn_v.weight"))?;
        let output = CheckedMatrix::from_payload(
            self.weights.payload,
            &block_name(block, "attn_output.weight"),
        )?;
        let key_cache_len = product(tokens, layout.kv_width)?;
        let mut keys = reserve("causal key cache", key_cache_len)?;
        let mut values = reserve("causal value cache", key_cache_len)?;
        keys.resize(key_cache_len, 0.0);
        values.resize(key_cache_len, 0.0);
        let mut attention = reserve("attention residual", hidden.len())?;
        for token in 0..tokens {
            let row = row(hidden, token, layout.hidden)?;
            let normalized =
                kernels::cpu_f32::rms_norm(row, &attn_norm, 1, layout.hidden, layout.epsilon)
                    .context(Qwen3CpuSnafu)?;
            let mut query = q.project(&normalized)?;
            let mut key = k.project(&normalized)?;
            let value = v.project(&normalized)?;
            query = kernels::cpu_f32::rms_norm(
                &query,
                &q_norm,
                layout.heads,
                layout.head_dim,
                layout.epsilon,
            )
            .context(Qwen3CpuSnafu)?;
            key = kernels::cpu_f32::rms_norm(
                &key,
                &k_norm,
                layout.kv_heads,
                layout.head_dim,
                layout.epsilon,
            )
            .context(Qwen3CpuSnafu)?;
            apply_neox_rope(&mut query, token, layout)?;
            apply_neox_rope(&mut key, token, layout)?;
            let cache_start = product(token, layout.kv_width)?;
            copy_into(&mut keys, cache_start, &key, "key cache")?;
            copy_into(&mut values, cache_start, &value, "value cache")?;
            let merged = causal_attention(&query, &keys, &values, token + 1, layout)?;
            attention.extend(output.project(&merged)?);
        }
        add_in_place(hidden, &attention, "attention residual")?;
        let ffn_norm = read_f32_vector(
            self.weights.payload,
            &block_name(block, "ffn_norm.weight"),
            layout.hidden,
        )?;
        let gate = CheckedMatrix::from_payload(
            self.weights.payload,
            &block_name(block, "ffn_gate.weight"),
        )?;
        let up =
            CheckedMatrix::from_payload(self.weights.payload, &block_name(block, "ffn_up.weight"))?;
        let down = CheckedMatrix::from_payload(
            self.weights.payload,
            &block_name(block, "ffn_down.weight"),
        )?;
        let mut ffn = reserve("FFN residual", hidden.len())?;
        for token in 0..tokens {
            let normalized = kernels::cpu_f32::rms_norm(
                row(hidden, token, layout.hidden)?,
                &ffn_norm,
                1,
                layout.hidden,
                layout.epsilon,
            )
            .context(Qwen3CpuSnafu)?;
            let activated =
                kernels::cpu_f32::try_silu(&gate.project(&normalized)?).context(Qwen3CpuSnafu)?;
            let up = up.project(&normalized)?;
            let fused = multiply(&activated, &up, "SwiGLU")?;
            ffn.extend(down.project(&fused)?);
        }
        add_in_place(hidden, &ffn, "FFN residual")
    }
}

#[derive(Clone, Copy, Debug)]
struct Layout {
    blocks: usize,
    context: usize,
    hidden: usize,
    feed_forward: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    kv_width: usize,
    q_width: usize,
    vocabulary: usize,
    epsilon: f32,
    rope_base: f64,
}

impl Layout {
    fn from_artifact(payload: &VerifiedArtifact) -> Result<Self> {
        let metadata = payload.observation().metadata();
        require_string(metadata, ARCHITECTURE, "qwen3")?;
        require_u32(metadata, POOLING_TYPE, "last-token pooling type")?
            .eq(&LAST_POOLING_TYPE)
            .then_some(())
            .ok_or_else(|| {
                Qwen3MetadataSnafu {
                    key: POOLING_TYPE,
                    rule: "must select last-token pooling type 3",
                }
                .build()
            })?;
        if matches!(metadata.get(CAUSAL), Some(MetaValue::Bool(false))) {
            return Qwen3MetadataSnafu {
                key: CAUSAL,
                rule: "explicit false is outside the causal embedding profile",
            }
            .fail();
        }
        if metadata
            .get(CAUSAL)
            .is_some_and(|value| !matches!(value, MetaValue::Bool(_)))
        {
            return Qwen3MetadataSnafu {
                key: CAUSAL,
                rule: "must be boolean when present",
            }
            .fail();
        }
        require_neutral_rope_scaling(metadata)?;
        let blocks = positive(metadata, BLOCK_COUNT)?;
        let context = positive(metadata, CONTEXT_LENGTH)?;
        let hidden = positive(metadata, HIDDEN)?;
        let feed_forward = positive(metadata, FEED_FORWARD)?;
        let heads = positive(metadata, HEADS)?;
        let kv_heads = positive(metadata, KV_HEADS)?;
        let head_dim = positive(metadata, KEY_LENGTH)?;
        let value_dim = positive(metadata, VALUE_LENGTH)?;
        let rope_dim = positive(metadata, ROPE_DIMENSION)?;
        let epsilon = require_f32(metadata, RMS_EPSILON)?;
        let rope_base = f64::from(require_f32(metadata, ROPE_BASE)?);
        if value_dim != head_dim
            || rope_dim != head_dim
            || !head_dim.is_multiple_of(2)
            || !heads.is_multiple_of(kv_heads)
        {
            return Qwen3MetadataSnafu {
                key: HEADS,
                rule: "requires divisible Q/KV heads plus equal even key/value/full rotary widths",
            }
            .fail();
        }
        if !epsilon.is_finite() || epsilon <= 0.0 || !rope_base.is_finite() || rope_base <= 0.0 {
            return Qwen3MetadataSnafu {
                key: RMS_EPSILON,
                rule: "epsilon and RoPE base must be finite positive values",
            }
            .fail();
        }
        let q_width = product(heads, head_dim)?;
        let kv_width = product(kv_heads, head_dim)?;
        let vocabulary = vocabulary(metadata)?;
        Ok(Self {
            blocks,
            context,
            hidden,
            feed_forward,
            heads,
            kv_heads,
            head_dim,
            kv_width,
            q_width,
            vocabulary,
            epsilon,
            rope_base,
        })
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "one role inventory defines the complete bounded Qwen3 embedding tensor contract"
)]
fn validate_inventory(tensors: &[loader::gguf::TensorDescriptor], layout: Layout) -> Result<()> {
    let expected_count = layout
        .blocks
        .checked_mul(11)
        .and_then(|count| count.checked_add(2))
        .ok_or_else(|| {
            Qwen3ExecutionSnafu {
                requested: layout.blocks,
                rule: "tensor inventory count overflowed",
            }
            .build()
        })?;
    if tensors.len() != expected_count {
        return Qwen3TensorSnafu {
            name: "inventory".to_string(),
            rule: "must have exactly the metadata-derived bounded role count",
        }
        .fail();
    }
    let mut expected = HashMap::with_capacity(expected_count);
    expected.insert(
        TOKEN_EMBEDDING.to_string(),
        vec![u64_from(layout.hidden)?, u64_from(layout.vocabulary)?],
    );
    expected.insert(OUTPUT_NORM.to_string(), vec![u64_from(layout.hidden)?]);
    for block in 0..layout.blocks {
        let prefix = format!("blk.{block}.");
        for (role, shape) in [
            ("attn_norm.weight", vec![u64_from(layout.hidden)?]),
            ("attn_q_norm.weight", vec![u64_from(layout.head_dim)?]),
            ("attn_k_norm.weight", vec![u64_from(layout.head_dim)?]),
            ("ffn_norm.weight", vec![u64_from(layout.hidden)?]),
            (
                "attn_q.weight",
                vec![u64_from(layout.hidden)?, u64_from(layout.q_width)?],
            ),
            (
                "attn_k.weight",
                vec![u64_from(layout.hidden)?, u64_from(layout.kv_width)?],
            ),
            (
                "attn_v.weight",
                vec![u64_from(layout.hidden)?, u64_from(layout.kv_width)?],
            ),
            (
                "attn_output.weight",
                vec![u64_from(layout.q_width)?, u64_from(layout.hidden)?],
            ),
            (
                "ffn_gate.weight",
                vec![u64_from(layout.hidden)?, u64_from(layout.feed_forward)?],
            ),
            (
                "ffn_up.weight",
                vec![u64_from(layout.hidden)?, u64_from(layout.feed_forward)?],
            ),
            (
                "ffn_down.weight",
                vec![u64_from(layout.feed_forward)?, u64_from(layout.hidden)?],
            ),
        ] {
            expected.insert(format!("{prefix}{role}"), shape);
        }
    }
    let mut found = HashSet::new();
    for tensor in tensors {
        let Some(shape) = expected.get(&tensor.name) else {
            return Qwen3TensorSnafu {
                name: tensor.name.clone(),
                rule: "is outside the bounded embedding role inventory",
            }
            .fail();
        };
        if !found.insert(&tensor.name) {
            return Qwen3TensorSnafu {
                name: tensor.name.clone(),
                rule: "must not occur more than once",
            }
            .fail();
        }
        if tensor.dims != *shape {
            return Qwen3TensorSnafu {
                name: tensor.name.clone(),
                rule: "shape must derive from checked architecture metadata",
            }
            .fail();
        }
        let vector = tensor.name.ends_with("norm.weight") || tensor.name == OUTPUT_NORM;
        if vector && tensor.ggml_type != GgmlType::F32 {
            return Qwen3TensorSnafu {
                name: tensor.name.clone(),
                rule: "normalization vectors must use F32",
            }
            .fail();
        }
    }
    for name in expected.keys() {
        if !found.contains(name) {
            return Qwen3TensorSnafu {
                name: name.clone(),
                rule: "is required by the bounded embedding profile",
            }
            .fail();
        }
    }
    Ok(())
}

fn causal_attention(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    tokens: usize,
    layout: Layout,
) -> Result<Vec<f32>> {
    let mut output = reserve("causal attention output", layout.q_width)?;
    let group = layout.heads / layout.kv_heads;
    let head_dim = layout.head_dim.to_f32().ok_or_else(|| {
        Qwen3ExecutionSnafu {
            requested: layout.head_dim,
            rule: "head dimension must convert to f32 for attention scaling",
        }
        .build()
    })?;
    let scale = head_dim.sqrt().recip();
    for head in 0..layout.heads {
        let q = row(query, head, layout.head_dim)?;
        let kv_head = head / group;
        let mut scores = reserve("causal attention scores", tokens)?;
        for token in 0..tokens {
            let key = row(row(keys, token, layout.kv_width)?, kv_head, layout.head_dim)?;
            let score = q
                .iter()
                .zip(key)
                .map(|(left, right)| left * right)
                .sum::<f32>()
                * scale;
            finite_one(score, "attention score", token)?;
            scores.push(score);
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exponents = reserve("causal attention exponentials", tokens)?;
        for score in &scores {
            let exponent = (*score - max).exp();
            finite_one(exponent, "attention exponential", exponents.len())?;
            exponents.push(exponent);
        }
        let total = exponents.iter().sum::<f32>();
        finite_one(total, "attention normalization", head)?;
        for lane in 0..layout.head_dim {
            let mut value = 0.0;
            for (token, exponent) in exponents.iter().enumerate() {
                let v = row(
                    row(values, token, layout.kv_width)?,
                    kv_head,
                    layout.head_dim,
                )?[lane];
                value += exponent / total * v;
            }
            finite_one(value, "attention value", head * layout.head_dim + lane)?;
            output.push(value);
        }
    }
    Ok(output)
}

fn apply_neox_rope(values: &mut [f32], position: usize, layout: Layout) -> Result<()> {
    let position = position.to_f64().ok_or_else(|| {
        Qwen3ExecutionSnafu {
            requested: position,
            rule: "RoPE position must convert to f64",
        }
        .build()
    })?;
    let head_dim = layout.head_dim.to_f64().ok_or_else(|| {
        Qwen3ExecutionSnafu {
            requested: layout.head_dim,
            rule: "RoPE head dimension must convert to f64",
        }
        .build()
    })?;
    for head in values.chunks_exact_mut(layout.head_dim) {
        for pair_index in 0..layout.head_dim / 2 {
            let pair = pair_index.to_f64().ok_or_else(|| {
                Qwen3ExecutionSnafu {
                    requested: layout.head_dim,
                    rule: "RoPE pair index must convert to f64",
                }
                .build()
            })?;
            let angle = position / layout.rope_base.powf((2.0 * pair) / head_dim);
            let cosine = angle.cos().to_f32().ok_or_else(|| {
                Qwen3ArithmeticSnafu {
                    stage: "NeoX RoPE cosine",
                    index: pair_index,
                }
                .build()
            })?;
            let sine = angle.sin().to_f32().ok_or_else(|| {
                Qwen3ArithmeticSnafu {
                    stage: "NeoX RoPE sine",
                    index: pair_index,
                }
                .build()
            })?;
            let right = pair_index + layout.head_dim / 2;
            let (left_value, right_value) = (head[pair_index], head[right]);
            head[pair_index] = left_value * cosine - right_value * sine;
            head[right] = left_value * sine + right_value * cosine;
        }
    }
    finite(values, "NeoX RoPE")
}

fn read_f32_vector(payload: &VerifiedArtifact, name: &str, width: usize) -> Result<Vec<f32>> {
    let tensor = payload.tensor(name).map_err(|_| {
        Qwen3TensorSnafu {
            name: name.to_string(),
            rule: "must be present in the verified payload",
        }
        .build()
    })?;
    if tensor.ggml_type() != GgmlType::F32 || tensor.dims() != [u64_from(width)?] {
        return Qwen3TensorSnafu {
            name: name.to_string(),
            rule: "must be an F32 vector with its metadata-derived width",
        }
        .fail();
    }
    quant::row_decode_f32(quant::RowFormat::F32, tensor.bytes(), width).map_err(|_| {
        Qwen3TensorSnafu {
            name: name.to_string(),
            rule: "must contain one finite complete F32 row",
        }
        .build()
    })
}

fn require_string(
    metadata: &HashMap<String, MetaValue>,
    key: &'static str,
    expected: &'static str,
) -> Result<()> {
    match metadata.get(key) {
        Some(MetaValue::String(value)) if value == expected => Ok(()),
        _ => Qwen3MetadataSnafu {
            key,
            rule: "must be the exact bounded profile value",
        }
        .fail(),
    }
}
fn require_u32(
    metadata: &HashMap<String, MetaValue>,
    key: &'static str,
    rule: &'static str,
) -> Result<u64> {
    match metadata.get(key) {
        Some(MetaValue::U32(value)) => Ok(u64::from(*value)),
        _ => Qwen3MetadataSnafu { key, rule }.fail(),
    }
}
fn require_f32(metadata: &HashMap<String, MetaValue>, key: &'static str) -> Result<f32> {
    match metadata.get(key) {
        Some(MetaValue::F32(value)) => Ok(*value),
        _ => Qwen3MetadataSnafu {
            key,
            rule: "must be F32",
        }
        .fail(),
    }
}
fn positive(metadata: &HashMap<String, MetaValue>, key: &'static str) -> Result<usize> {
    let value = require_u32(metadata, key, "must be U32")?;
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            Qwen3MetadataSnafu {
                key,
                rule: "must be a positive usize",
            }
            .build()
        })
}
fn vocabulary(metadata: &HashMap<String, MetaValue>) -> Result<usize> {
    match metadata.get("tokenizer.ggml.tokens") {
        Some(MetaValue::Array(values)) if values.element_type() == MetaValueType::String => {
            NonZero::from(values.values().len()).ok_or_else(|| {
                Qwen3MetadataSnafu {
                    key: "tokenizer.ggml.tokens",
                    rule: "must be a nonempty string array",
                }
                .build()
            })
        }
        _ => Qwen3MetadataSnafu {
            key: "tokenizer.ggml.tokens",
            rule: "must be a string array",
        }
        .fail(),
    }
}
fn require_neutral_rope_scaling(metadata: &HashMap<String, MetaValue>) -> Result<()> {
    if let Some(value) = metadata.get(ROPE_SCALING_TYPE)
        && !matches!(value, MetaValue::String(value) if value == "linear")
    {
        return Qwen3MetadataSnafu {
            key: ROPE_SCALING_TYPE,
            rule: "must be absent or exact neutral linear scaling",
        }
        .fail();
    }
    if let Some(value) = metadata.get(ROPE_SCALING_FACTOR)
        && !matches!(value, MetaValue::F32(value) if value.to_bits() == 1.0_f32.to_bits())
    {
        return Qwen3MetadataSnafu {
            key: ROPE_SCALING_FACTOR,
            rule: "must be absent or exact neutral factor 1",
        }
        .fail();
    }
    Ok(())
}
fn block_name(block: usize, role: &str) -> String {
    format!("blk.{block}.{role}")
}
fn product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right).ok_or_else(|| {
        Qwen3ExecutionSnafu {
            requested: left,
            rule: "execution geometry multiplication overflowed",
        }
        .build()
    })
}
fn u64_from(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| {
        Qwen3ExecutionSnafu {
            requested: value,
            rule: "execution geometry exceeds u64",
        }
        .build()
    })
}
fn reserve(target: &'static str, length: usize) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .context(Qwen3AllocationSnafu { target, length })?;
    Ok(values)
}
fn row(values: &[f32], index: usize, width: usize) -> Result<&[f32]> {
    let start = product(index, width)?;
    values.get(start..start + width).ok_or_else(|| {
        Qwen3ExecutionSnafu {
            requested: index,
            rule: "row must fit its checked buffer",
        }
        .build()
    })
}
fn copy_into(
    destination: &mut [f32],
    start: usize,
    source: &[f32],
    target: &'static str,
) -> Result<()> {
    let end = start.checked_add(source.len()).ok_or_else(|| {
        Qwen3ExecutionSnafu {
            requested: start,
            rule: "copy range overflowed",
        }
        .build()
    })?;
    let slot = destination.get_mut(start..end).ok_or_else(|| {
        Qwen3ExecutionSnafu {
            requested: start,
            rule: "copy range must fit its checked buffer",
        }
        .build()
    })?;
    slot.copy_from_slice(source);
    finite(slot, target)
}
fn multiply(left: &[f32], right: &[f32], stage: &'static str) -> Result<Vec<f32>> {
    if left.len() != right.len() {
        return Qwen3ExecutionSnafu {
            requested: left.len(),
            rule: "elementwise operands must have equal lengths",
        }
        .fail();
    }
    let mut output = reserve(stage, left.len())?;
    for (index, (left, right)) in left.iter().zip(right).enumerate() {
        let value = left * right;
        finite_one(value, stage, index)?;
        output.push(value);
    }
    Ok(output)
}
fn add_in_place(destination: &mut [f32], source: &[f32], stage: &'static str) -> Result<()> {
    if destination.len() != source.len() {
        return Qwen3ExecutionSnafu {
            requested: destination.len(),
            rule: "residual operands must have equal lengths",
        }
        .fail();
    }
    for (index, (left, right)) in destination.iter_mut().zip(source).enumerate() {
        *left += right;
        finite_one(*left, stage, index)?;
    }
    Ok(())
}
fn finite(values: &[f32], stage: &'static str) -> Result<()> {
    for (index, value) in values.iter().enumerate() {
        finite_one(*value, stage, index)?;
    }
    Ok(())
}
fn finite_one(value: f32, stage: &'static str, index: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Qwen3ArithmeticSnafu { stage, index }.fail()
    }
}

struct NonZero;
impl NonZero {
    fn from(value: usize) -> Option<usize> {
        (value != 0).then_some(value)
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use loader::gguf::{ArtifactByteLimit, Sha256Digest, VerifiedArtifact};
    use test_fixtures::{RawGguf, RawMetadata, RawMetadataValue, RawTensor, serialize_raw_gguf};

    use super::*;

    const TEST_HIDDEN: u64 = 3;
    const TEST_HEADS: u64 = 2;
    const TEST_KV_HEADS: u64 = 1;
    const TEST_HEAD_DIM: u64 = 2;
    const TEST_FEED_FORWARD: u64 = 4;
    const TEST_VOCABULARY: u64 = 4;
    const TEST_BLOCKS: u64 = 1;
    const TEST_CONTEXT: u32 = 4;

    #[test]
    fn executes_an_asymmetric_causal_fixture_to_a_final_hidden_row()
    -> std::result::Result<(), String> {
        let raw = fixture()?;
        let artifact = verify(&raw)?;
        let weights =
            Qwen3Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;
        let execution = weights
            .execution(usize::try_from(TEST_CONTEXT).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        let one = execution
            .last_hidden(&[0])
            .map_err(|error| error.to_string())?;
        let two = execution
            .last_hidden(&[0, 1])
            .map_err(|error| error.to_string())?;
        if one.len() != usize::try_from(TEST_HIDDEN).map_err(|error| error.to_string())? {
            return Err("one-token result did not retain the hidden width".to_string());
        }
        if !two.iter().all(|value| value.is_finite()) {
            return Err("final hidden row was not finite".to_string());
        }
        if one == two {
            return Err("causal fixture did not distinguish its final token row".to_string());
        }
        Ok(())
    }

    #[test]
    fn rejects_an_output_head_outside_the_embedding_profile() -> std::result::Result<(), String> {
        let mut raw = fixture()?;
        raw.tensors.push(tensor(
            "output.weight",
            vec![TEST_HIDDEN, TEST_VOCABULARY],
            0.5,
        )?);
        let serialized = serialize_raw_gguf(&raw).map_err(|error| error.to_string())?;
        let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
        let path = directory.path().join("qwen3-extra-output.gguf");
        std::fs::write(&path, serialized.bytes).map_err(|error| error.to_string())?;
        let limit = NonZeroU64::new(serialized.byte_len + 1).ok_or("invalid fixture limit")?;
        let artifact = VerifiedArtifact::load(
            &path,
            Sha256Digest::from_bytes(serialized.sha256),
            ArtifactByteLimit::new(limit),
        )
        .map_err(|error| error.to_string())?;
        if Qwen3Weights::try_from_verified(&artifact).is_ok() {
            return Err("bounded embedding profile accepted output.weight".to_string());
        }
        Ok(())
    }

    #[test]
    fn refuses_a_general_pooling_key_without_the_qwen3_contract_key()
    -> std::result::Result<(), String> {
        let mut raw = fixture()?;
        raw.metadata.retain(|entry| entry.key != POOLING_TYPE);
        raw.metadata.push(metadata_u32(
            "general.pooling_type",
            u32::try_from(LAST_POOLING_TYPE).map_err(|error| error.to_string())?,
        ));
        let artifact = verify(&raw)?;
        if Qwen3Weights::try_from_verified(&artifact).is_ok() {
            return Err(
                "general pooling metadata substituted for qwen3 pooling metadata".to_string(),
            );
        }
        Ok(())
    }

    fn verify(raw: &RawGguf) -> std::result::Result<VerifiedArtifact, String> {
        let serialized = serialize_raw_gguf(raw).map_err(|error| error.to_string())?;
        let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
        let path = directory.path().join("qwen3.gguf");
        std::fs::write(&path, &serialized.bytes).map_err(|error| error.to_string())?;
        let limit = NonZeroU64::new(serialized.byte_len + 1).ok_or("invalid fixture limit")?;
        VerifiedArtifact::load(
            &path,
            Sha256Digest::from_bytes(serialized.sha256),
            ArtifactByteLimit::new(limit),
        )
        .map_err(|error| error.to_string())
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the fixture names every tensor in the one bounded Qwen3 profile"
    )]
    fn fixture() -> std::result::Result<RawGguf, String> {
        let mut tensors = vec![
            tensor(
                "token_embd.weight",
                vec![TEST_HIDDEN, TEST_VOCABULARY],
                0.125,
            )?,
            tensor("output_norm.weight", vec![TEST_HIDDEN], 1.0)?,
        ];
        for (role, dimensions, seed) in [
            ("attn_norm.weight", vec![TEST_HIDDEN], 1.0),
            ("attn_q_norm.weight", vec![TEST_HEAD_DIM], 1.0),
            ("attn_k_norm.weight", vec![TEST_HEAD_DIM], 1.0),
            ("ffn_norm.weight", vec![TEST_HIDDEN], 1.0),
            (
                "attn_q.weight",
                vec![TEST_HIDDEN, TEST_HEADS * TEST_HEAD_DIM],
                0.0625,
            ),
            (
                "attn_k.weight",
                vec![TEST_HIDDEN, TEST_KV_HEADS * TEST_HEAD_DIM],
                0.09375,
            ),
            (
                "attn_v.weight",
                vec![TEST_HIDDEN, TEST_KV_HEADS * TEST_HEAD_DIM],
                0.125,
            ),
            (
                "attn_output.weight",
                vec![TEST_HEADS * TEST_HEAD_DIM, TEST_HIDDEN],
                0.15625,
            ),
            (
                "ffn_gate.weight",
                vec![TEST_HIDDEN, TEST_FEED_FORWARD],
                0.1875,
            ),
            (
                "ffn_up.weight",
                vec![TEST_HIDDEN, TEST_FEED_FORWARD],
                0.21875,
            ),
            (
                "ffn_down.weight",
                vec![TEST_FEED_FORWARD, TEST_HIDDEN],
                0.25,
            ),
        ] {
            tensors.push(tensor(&format!("blk.0.{role}"), dimensions, seed)?);
        }
        Ok(RawGguf {
            metadata: vec![
                metadata_string(ARCHITECTURE, "qwen3"),
                metadata_u32(
                    BLOCK_COUNT,
                    u32::try_from(TEST_BLOCKS).map_err(|error| error.to_string())?,
                ),
                metadata_u32(CONTEXT_LENGTH, TEST_CONTEXT),
                metadata_u32(
                    HIDDEN,
                    u32::try_from(TEST_HIDDEN).map_err(|error| error.to_string())?,
                ),
                metadata_u32(
                    FEED_FORWARD,
                    u32::try_from(TEST_FEED_FORWARD).map_err(|error| error.to_string())?,
                ),
                metadata_u32(
                    HEADS,
                    u32::try_from(TEST_HEADS).map_err(|error| error.to_string())?,
                ),
                metadata_u32(
                    KV_HEADS,
                    u32::try_from(TEST_KV_HEADS).map_err(|error| error.to_string())?,
                ),
                metadata_u32(
                    KEY_LENGTH,
                    u32::try_from(TEST_HEAD_DIM).map_err(|error| error.to_string())?,
                ),
                metadata_u32(
                    VALUE_LENGTH,
                    u32::try_from(TEST_HEAD_DIM).map_err(|error| error.to_string())?,
                ),
                metadata_f32(RMS_EPSILON, 0.001),
                metadata_u32(
                    ROPE_DIMENSION,
                    u32::try_from(TEST_HEAD_DIM).map_err(|error| error.to_string())?,
                ),
                metadata_f32(ROPE_BASE, 10_000.0),
                metadata_u32(
                    POOLING_TYPE,
                    u32::try_from(LAST_POOLING_TYPE).map_err(|error| error.to_string())?,
                ),
                RawMetadata {
                    key: "tokenizer.ggml.tokens".to_string(),
                    value: RawMetadataValue::StringArray(vec![
                        "alice".to_string(),
                        "bob".to_string(),
                        "acme".to_string(),
                        "corp".to_string(),
                    ]),
                },
            ],
            tensors,
        })
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
    fn metadata_string(key: &str, value: &str) -> RawMetadata {
        RawMetadata {
            key: key.to_string(),
            value: RawMetadataValue::String(value.to_string()),
        }
    }
    fn tensor(name: &str, dims: Vec<u64>, seed: f32) -> std::result::Result<RawTensor, String> {
        let count = dims.iter().try_fold(1_usize, |count, dimension| {
            count
                .checked_mul(usize::try_from(*dimension).map_err(|error| error.to_string())?)
                .ok_or("fixture element count overflow".to_string())
        })?;
        let mut payload = Vec::with_capacity(count * 4);
        for index in 0..count {
            let index = index
                .to_f32()
                .ok_or("fixture element index cannot convert to f32")?;
            payload.extend_from_slice(&(seed + index * 0.007_812_5).to_le_bytes());
        }
        Ok(RawTensor {
            name: name.to_string(),
            dims,
            format: 0,
            payload,
        })
    }
}
