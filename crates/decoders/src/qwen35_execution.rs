//! Bounded CPU token-to-logits execution for a verified Qwen3.5 payload.

use loader::gguf::{GgmlType, MetaValue, MetaValueType};
use num_traits::ToPrimitive;
use quant::f32_row::F32Row;
use snafu::ResultExt;

use crate::error::{
    ArithmeticOverflowSnafu, ExecutionArithmeticSnafu, ExecutionContextSnafu, ExecutionTokenSnafu,
    MetadataRelationSnafu, MetadataTypeSnafu, MissingMetadataSnafu, PayloadTensorSnafu,
    ProjectionBytesSnafu, ProjectionDtypeSnafu, ProjectionRowSnafu, RecurrentRmsNormSnafu,
    TensorShapeSnafu,
};
use crate::qwen35::recurrent_layernorm_rms_epsilon;
use crate::{Qwen35RecurrentExecution, Qwen35Weights, Result};

const CONTEXT_LENGTH_KEY: &str = "qwen35.context_length";
const ROPE_SECTIONS_KEY: &str = "qwen35.rope.dimension_sections";
const ROPE_FREQ_BASE_KEY: &str = "qwen35.rope.freq_base";
const ROPE_SCALING_TYPE_KEY: &str = "qwen35.rope.scaling.type";
const ROPE_SCALING_FACTOR_KEY: &str = "qwen35.rope.scaling.factor";
const ROPE_SCALING_ATTENTION_FACTOR_KEY: &str = "qwen35.rope.scaling.attn_factor";
const ROPE_LEGACY_LINEAR_SCALE_KEY: &str = "qwen35.rope.scale_linear";
const ROPE_DIMENSION_COUNT_KEY: &str = "qwen35.rope.dimension_count";
const ATTENTION_SCALE_KEY: &str = "qwen35.attention.scale";
const ATTENTION_CAUSAL_KEY: &str = "qwen35.attention.causal";
const TOKEN_EMBEDDING: &str = "token_embd.weight";
const OUTPUT_NORM: &str = "output_norm.weight";
const OUTPUT: &str = "output.weight";

/// Stateful, bounded CPU text execution bound to one verified payload.
///
/// Each successful [`Self::step`] returns one vocabulary-logit row per input
/// token. The method stages recurrent and full-attention history in a clone,
/// committing it only after the whole call, including final logits, succeeds.
#[derive(Debug)]
pub struct Qwen35Execution<'weights, 'artifact> {
    weights: &'weights Qwen35Weights<'artifact>,
    layout: Layout,
    layers: Vec<LayerState<'weights, 'artifact>>,
    position: usize,
}

#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "direct recurrent storage avoids an additional infallible per-layer heap allocation; the checked Vec allocation bounds every main block"
)]
enum LayerState<'weights, 'artifact> {
    Recurrent(Qwen35RecurrentExecution<'weights, 'artifact>),
    Full(FullAttentionState),
}

impl<'weights, 'artifact> Qwen35Execution<'weights, 'artifact> {
    pub(crate) fn try_from_weights(
        weights: &'weights Qwen35Weights<'artifact>,
        max_context: usize,
    ) -> Result<Self> {
        let layout = Layout::from_metadata(weights, max_context)?;
        let block_count = layout.main_blocks;
        let mut layers = reserve_slots("main-block execution slots", block_count)?;
        for block in 0..block_count {
            if layout.is_full(block) {
                layers.push(LayerState::Full(FullAttentionState::new(&layout)?));
            } else {
                layers.push(LayerState::Recurrent(weights.recurrent_execution(
                    u64::try_from(block).map_err(|_| {
                        ArithmeticOverflowSnafu {
                            context: "recurrent execution block index",
                        }
                        .build()
                    })?,
                )?));
            }
        }
        Ok(Self {
            weights,
            layout,
            layers,
            position: 0,
        })
    }

    /// Execute complete token ids and return token-major vocabulary logits.
    ///
    /// The session owns only state derived from its verified payload; callers
    /// cannot inject KV or recurrent state.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] without committing state if a token, payload
    /// projection, checked allocation, or finite CPU operation is invalid.
    pub fn step(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return ExecutionContextSnafu {
                requested: 0_usize,
                rule: "token batches must be non-empty",
            }
            .fail();
        }
        let requested = self.position.checked_add(token_ids.len()).ok_or_else(|| {
            ArithmeticOverflowSnafu {
                context: "execution position plus token count",
            }
            .build()
        })?;
        if requested > self.layout.max_context {
            return ExecutionContextSnafu {
                requested,
                rule: "must not exceed the caller-bounded context",
            }
            .fail();
        }
        let mut staged = self.stage()?;
        let logits = staged.step_staged(token_ids)?;
        *self = staged;
        Ok(logits)
    }

    fn stage(&self) -> Result<Self> {
        let mut layers = reserve_slots("transaction main-block slots", self.layers.len())?;
        for layer in &self.layers {
            layers.push(match layer {
                LayerState::Recurrent(execution) => {
                    LayerState::Recurrent(execution.try_clone_for_transaction()?)
                }
                LayerState::Full(state) => {
                    LayerState::Full(state.try_clone_for_transaction(&self.layout)?)
                }
            });
        }
        Ok(Self {
            weights: self.weights,
            layout: self.layout,
            layers,
            position: self.position,
        })
    }

    fn step_staged(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let total = token_ids
            .len()
            .checked_mul(self.layout.vocabulary)
            .ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "token logits allocation",
                }
                .build()
            })?;
        let mut logits = reserve("token logits", total)?;
        for token_id in token_ids {
            let mut hidden = self.embed(*token_id)?;
            for block in 0..self.layout.main_blocks {
                let weights = self.weights;
                let layout = self.layout;
                let position = self.position;
                let layer = self.layers.get_mut(block).ok_or_else(|| {
                    ExecutionContextSnafu {
                        requested: block,
                        rule: "main-block state must exist for every artifact main block",
                    }
                    .build()
                })?;
                let attention = match layer {
                    LayerState::Recurrent(execution) => execution.step(&hidden)?,
                    LayerState::Full(state) => {
                        full_attention(weights, layout, position, block, &hidden, state)?
                    }
                };
                add_in_place(&mut hidden, &attention, "attention residual")?;
                let post_norm = read_f32(
                    self.weights,
                    &block_name(block, "post_attention_norm.weight"),
                    &[self.layout.hidden_u64],
                )?;
                let normalized = kernels::cpu_f32::rms_norm(
                    &hidden,
                    &post_norm,
                    1,
                    self.layout.hidden,
                    self.layout.epsilon,
                )
                .context(RecurrentRmsNormSnafu)?;
                let ffn = self.ffn(block, &normalized)?;
                add_in_place(&mut hidden, &ffn, "FFN residual")?;
            }
            let output_norm = read_f32(self.weights, OUTPUT_NORM, &[self.layout.hidden_u64])?;
            let normalized = kernels::cpu_f32::rms_norm(
                &hidden,
                &output_norm,
                1,
                self.layout.hidden,
                self.layout.epsilon,
            )
            .context(RecurrentRmsNormSnafu)?;
            logits.extend(self.weights.project(OUTPUT, &normalized)?);
            self.position = self.position.checked_add(1).ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "execution position increment",
                }
                .build()
            })?;
        }
        Ok(logits)
    }

    fn embed(&self, token_id: u32) -> Result<Vec<f32>> {
        let token = usize::try_from(token_id).map_err(|_| {
            ExecutionTokenSnafu {
                token_id,
                vocabulary: self.layout.vocabulary,
            }
            .build()
        })?;
        if token >= self.layout.vocabulary {
            return ExecutionTokenSnafu {
                token_id,
                vocabulary: self.layout.vocabulary,
            }
            .fail();
        }
        let embedding = self.weights.decode_row(TOKEN_EMBEDDING, token)?;
        if embedding.len() != self.layout.hidden {
            return ExecutionContextSnafu {
                requested: embedding.len(),
                rule: "token embedding row must match artifact-derived hidden width",
            }
            .fail();
        }
        finite(&embedding, "token embedding")?;
        Ok(embedding)
    }

    fn ffn(&self, block: usize, input: &[f32]) -> Result<Vec<f32>> {
        let gate = self
            .weights
            .project(&block_name(block, "ffn_gate.weight"), input)?;
        let up = self
            .weights
            .project(&block_name(block, "ffn_up.weight"), input)?;
        let activated = kernels::cpu_f32::silu(&gate);
        let mut fused = reserve("SwiGLU activation", self.layout.feed_forward)?;
        for (index, (left, right)) in activated.iter().zip(up.iter()).enumerate() {
            let value = left * right;
            finite_one(value, "SwiGLU", index)?;
            fused.push(value);
        }
        self.weights
            .project(&block_name(block, "ffn_down.weight"), &fused)
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "the pinned full-attention operation order is one bounded transactional unit"
)]
fn full_attention(
    weights: &Qwen35Weights<'_>,
    layout: Layout,
    position: usize,
    block: usize,
    input: &[f32],
    state: &mut FullAttentionState,
) -> Result<Vec<f32>> {
    let norm = read_f32(
        weights,
        &block_name(block, "attn_norm.weight"),
        &[layout.hidden_u64],
    )?;
    let normalized = kernels::cpu_f32::rms_norm(input, &norm, 1, layout.hidden, layout.epsilon)
        .context(RecurrentRmsNormSnafu)?;
    let q_gate = weights.project(&block_name(block, "attn_q.weight"), &normalized)?;
    let key = weights.project(&block_name(block, "attn_k.weight"), &normalized)?;
    let value = weights.project(&block_name(block, "attn_v.weight"), &normalized)?;
    let q_norm = read_f32(
        weights,
        &block_name(block, "attn_q_norm.weight"),
        &[layout.key_u64],
    )?;
    let k_norm = read_f32(
        weights,
        &block_name(block, "attn_k_norm.weight"),
        &[layout.key_u64],
    )?;
    let mut query = reserve("full-attention query", layout.query_width)?;
    let mut gate = reserve("full-attention gate", layout.query_width)?;
    for head in 0..layout.heads {
        let start = head
            .checked_mul(layout.key.checked_mul(2).ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "Q/gate head width",
                }
                .build()
            })?)
            .ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "Q/gate head offset",
                }
                .build()
            })?;
        let middle = start.checked_add(layout.key).ok_or_else(|| {
            ArithmeticOverflowSnafu {
                context: "Q/gate split",
            }
            .build()
        })?;
        let end = middle.checked_add(layout.key).ok_or_else(|| {
            ArithmeticOverflowSnafu {
                context: "Q/gate end",
            }
            .build()
        })?;
        query.extend(q_gate.get(start..middle).ok_or_else(|| {
            ExecutionContextSnafu {
                requested: start,
                rule: "Q/gate projection must be interleaved per head",
            }
            .build()
        })?);
        gate.extend(q_gate.get(middle..end).ok_or_else(|| {
            ExecutionContextSnafu {
                requested: middle,
                rule: "Q/gate projection must be interleaved per head",
            }
            .build()
        })?);
    }
    query = kernels::cpu_f32::rms_norm(&query, &q_norm, layout.heads, layout.key, layout.epsilon)
        .context(RecurrentRmsNormSnafu)?;
    let mut key =
        kernels::cpu_f32::rms_norm(&key, &k_norm, layout.kv_heads, layout.key, layout.epsilon)
            .context(RecurrentRmsNormSnafu)?;
    apply_text_mrope(layout, position, &mut query)?;
    apply_text_mrope(layout, position, &mut key)?;
    state.push(&key, &value, &layout)?;
    let mut merged = reserve("full-attention merged output", layout.query_width)?;
    for head in 0..layout.heads {
        let kv_head = head / layout.gqa_group;
        let query_start = head.checked_mul(layout.key).ok_or_else(|| {
            ArithmeticOverflowSnafu {
                context: "attention query offset",
            }
            .build()
        })?;
        let query = query
            .get(query_start..query_start + layout.key)
            .ok_or_else(|| {
                ExecutionContextSnafu {
                    requested: query_start,
                    rule: "query head range must fit",
                }
                .build()
            })?;
        let attended = state.attend(query, kv_head, &layout)?;
        let gate_row = gate
            .get(query_start..query_start + layout.key)
            .ok_or_else(|| {
                ExecutionContextSnafu {
                    requested: query_start,
                    rule: "attention gate range must fit",
                }
                .build()
            })?;
        for (index, (output, gate_value)) in attended.iter().zip(gate_row.iter()).enumerate() {
            let sigmoid = 1.0 / (1.0 + (-gate_value).exp());
            let value = output * sigmoid;
            finite_one(value, "full attention gate", index)?;
            merged.push(value);
        }
    }
    weights.project(&block_name(block, "attn_output.weight"), &merged)
}

fn apply_text_mrope(layout: Layout, position: usize, values: &mut [f32]) -> Result<()> {
    // The Qwen3.5 text position contract is component-major `[p,p,p,0]`.
    // The validated sections select which rotary pairs use each component.
    let position = f64::from(i32::try_from(position).map_err(|_| {
        ArithmeticOverflowSnafu {
            context: "text MRoPE position",
        }
        .build()
    })?);
    for row in values.chunks_exact_mut(layout.key) {
        for pair in 0..(layout.n_rot / 2) {
            let left_index = pair;
            let pair_f64 = pair.to_f64().ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "text MRoPE pair",
                }
                .build()
            })?;
            let key_f64 = layout.n_rot.to_f64().ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "text MRoPE rotary width",
                }
                .build()
            })?;
            let exponent = (2.0 * pair_f64) / key_f64;
            let axis = layout.rope_axis(pair)?;
            let text_position = if axis == 3 { 0.0 } else { position };
            let angle = text_position / layout.rope_base.powf(exponent);
            let cos = angle.cos().to_f32().ok_or_else(|| {
                ExecutionArithmeticSnafu {
                    stage: "text MRoPE cosine",
                    index: pair,
                }
                .build()
            })?;
            let sin = angle.sin().to_f32().ok_or_else(|| {
                ExecutionArithmeticSnafu {
                    stage: "text MRoPE sine",
                    index: pair,
                }
                .build()
            })?;
            let right_index = left_index.checked_add(layout.n_rot / 2).ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "RoPE half-split pair end",
                }
                .build()
            })?;
            let left = *row.get(left_index).ok_or_else(|| {
                ExecutionContextSnafu {
                    requested: left_index,
                    rule: "RoPE even coordinate must fit",
                }
                .build()
            })?;
            let right = *row.get(right_index).ok_or_else(|| {
                ExecutionContextSnafu {
                    requested: right_index,
                    rule: "RoPE odd coordinate must fit",
                }
                .build()
            })?;
            row[left_index] = left * cos - right * sin;
            row[right_index] = left * sin + right * cos;
        }
    }
    finite(values, "text MRoPE")
}

#[derive(Debug)]
struct FullAttentionState {
    keys: Vec<f32>,
    values: Vec<f32>,
    tokens: usize,
}

impl FullAttentionState {
    fn new(layout: &Layout) -> Result<Self> {
        let key_capacity = layout
            .max_context
            .checked_mul(layout.kv_width)
            .ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "KV key capacity",
                }
                .build()
            })?;
        let value_capacity = layout
            .max_context
            .checked_mul(layout.kv_width)
            .ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "KV value capacity",
                }
                .build()
            })?;
        Ok(Self {
            keys: reserve("KV keys", key_capacity)?,
            values: reserve("KV values", value_capacity)?,
            tokens: 0,
        })
    }

    fn push(&mut self, key: &[f32], value: &[f32], layout: &Layout) -> Result<()> {
        if self.tokens >= layout.max_context {
            return ExecutionContextSnafu {
                requested: self.tokens + 1,
                rule: "KV state must not exceed caller-bounded context",
            }
            .fail();
        }
        if key.len() != layout.kv_width || value.len() != layout.kv_width {
            return ExecutionContextSnafu {
                requested: key.len(),
                rule: "full-attention KV projections must match artifact-derived width",
            }
            .fail();
        }
        self.keys.extend_from_slice(key);
        self.values.extend_from_slice(value);
        self.tokens += 1;
        Ok(())
    }

    fn try_clone_for_transaction(&self, layout: &Layout) -> Result<Self> {
        let capacity = layout
            .max_context
            .checked_mul(layout.kv_width)
            .ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "transaction KV capacity",
                }
                .build()
            })?;
        let mut keys = reserve("transaction KV keys", capacity)?;
        keys.extend_from_slice(&self.keys);
        let mut values = reserve("transaction KV values", capacity)?;
        values.extend_from_slice(&self.values);
        Ok(Self {
            keys,
            values,
            tokens: self.tokens,
        })
    }

    fn attend(&self, query: &[f32], kv_head: usize, layout: &Layout) -> Result<Vec<f32>> {
        let mut scores = reserve("attention scores", self.tokens)?;
        let scale = layout
            .key
            .to_f32()
            .ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "attention key width",
                }
                .build()
            })?
            .sqrt()
            .recip();
        for token in 0..self.tokens {
            let start = token
                .checked_mul(layout.kv_width)
                .and_then(|offset| offset.checked_add(kv_head * layout.key))
                .ok_or_else(|| {
                    ArithmeticOverflowSnafu {
                        context: "KV key offset",
                    }
                    .build()
                })?;
            let key = self.keys.get(start..start + layout.key).ok_or_else(|| {
                ExecutionContextSnafu {
                    requested: start,
                    rule: "KV key range must fit retained state",
                }
                .build()
            })?;
            let score = query.iter().zip(key).map(|(a, b)| a * b).sum::<f32>() * scale;
            finite_one(score, "attention score", token)?;
            scores.push(score);
        }
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let normalizer = scores
            .iter()
            .map(|score| (*score - maximum).exp())
            .sum::<f32>();
        finite_one(normalizer, "attention softmax normalizer", 0)?;
        let mut output = reserve("attention head output", layout.key)?;
        output.resize(layout.key, 0.0);
        for (token, score) in scores.iter().enumerate() {
            let probability = (*score - maximum).exp() / normalizer;
            let start = token
                .checked_mul(layout.kv_width)
                .and_then(|offset| offset.checked_add(kv_head * layout.key))
                .ok_or_else(|| {
                    ArithmeticOverflowSnafu {
                        context: "KV value offset",
                    }
                    .build()
                })?;
            let value = self.values.get(start..start + layout.key).ok_or_else(|| {
                ExecutionContextSnafu {
                    requested: start,
                    rule: "KV value range must fit retained state",
                }
                .build()
            })?;
            for (index, (destination, source)) in output.iter_mut().zip(value).enumerate() {
                *destination += probability * source;
                finite_one(*destination, "attention value", index)?;
            }
        }
        Ok(output)
    }
}

#[derive(Debug, Clone, Copy)]
struct Layout {
    hidden: usize,
    hidden_u64: u64,
    feed_forward: usize,
    heads: usize,
    kv_heads: usize,
    key: usize,
    n_rot: usize,
    key_u64: u64,
    kv_width: usize,
    query_width: usize,
    gqa_group: usize,
    vocabulary: usize,
    main_blocks: usize,
    full_interval: usize,
    max_context: usize,
    epsilon: f32,
    rope_base: f64,
    rope_sections: [usize; 4],
}

impl Layout {
    #[expect(
        clippy::too_many_lines,
        reason = "execution-only metadata is admitted at one artifact-bound boundary"
    )]
    fn from_metadata(weights: &Qwen35Weights<'_>, max_context: usize) -> Result<Self> {
        if max_context == 0 {
            return ExecutionContextSnafu {
                requested: max_context,
                rule: "caller context must be non-zero",
            }
            .fail();
        }
        let metadata = weights.payload().observation().metadata();
        let context = u32_meta(metadata, CONTEXT_LENGTH_KEY)? as usize;
        if max_context > context {
            return ExecutionContextSnafu {
                requested: max_context,
                rule: "caller context must not exceed artifact context_length",
            }
            .fail();
        }
        if max_context > i32::MAX as usize + 1 {
            return ExecutionContextSnafu {
                requested: max_context,
                rule: "caller context must fit the source signed text-position domain",
            }
            .fail();
        }
        let dimensions = weights.execution_dimensions();
        let hidden_u64 = dimensions.hidden;
        let hidden = usize::try_from(hidden_u64).map_err(|_| {
            ArithmeticOverflowSnafu {
                context: "execution hidden width",
            }
            .build()
        })?;
        let feed_forward = usize::try_from(dimensions.feed_forward).map_err(|_| {
            ArithmeticOverflowSnafu {
                context: "execution feed-forward width",
            }
            .build()
        })?;
        let heads = usize::try_from(dimensions.heads).map_err(|_| {
            ArithmeticOverflowSnafu {
                context: "execution head count",
            }
            .build()
        })?;
        let kv_heads = usize::try_from(dimensions.key_value_heads).map_err(|_| {
            ArithmeticOverflowSnafu {
                context: "execution key-value head count",
            }
            .build()
        })?;
        let key_u64 = dimensions.key_width;
        let key = usize::try_from(key_u64).map_err(|_| {
            ArithmeticOverflowSnafu {
                context: "execution key width",
            }
            .build()
        })?;
        let n_rot = optional_u32_meta(metadata, ROPE_DIMENSION_COUNT_KEY)?
            .map(usize::try_from)
            .transpose()
            .map_err(|_| {
                ArithmeticOverflowSnafu {
                    context: "execution rotary width",
                }
                .build()
            })?
            .unwrap_or(key);
        if n_rot == 0 || n_rot > key || !n_rot.is_multiple_of(2) {
            return MetadataRelationSnafu {
                key: ROPE_DIMENSION_COUNT_KEY,
                rule: "must be nonzero, even, and no wider than key width",
            }
            .fail();
        }
        if let Some(scale) = optional_f32_meta(metadata, ATTENTION_SCALE_KEY)?
            && (!scale.is_finite() || scale != 0.0)
        {
            return MetadataRelationSnafu {
                key: ATTENTION_SCALE_KEY,
                rule: "only the source default 1/sqrt(key width) attention scale is implemented",
            }
            .fail();
        }
        if let Some(causal) = optional_bool_meta(metadata, ATTENTION_CAUSAL_KEY)?
            && !causal
        {
            return MetadataRelationSnafu {
                key: ATTENTION_CAUSAL_KEY,
                rule: "only causal full attention is implemented",
            }
            .fail();
        }
        let vocabulary = usize::try_from(dimensions.vocabulary).map_err(|_| {
            ArithmeticOverflowSnafu {
                context: "execution vocabulary",
            }
            .build()
        })?;
        let main_blocks = usize::try_from(dimensions.main_block_count).map_err(|_| {
            ArithmeticOverflowSnafu {
                context: "execution main block count",
            }
            .build()
        })?;
        let full_interval = usize::try_from(dimensions.full_attention_interval).map_err(|_| {
            ArithmeticOverflowSnafu {
                context: "execution full-attention interval",
            }
            .build()
        })?;
        if full_interval == 0
            || heads == 0
            || kv_heads == 0
            || key == 0
            || !heads.is_multiple_of(kv_heads)
        {
            return MetadataRelationSnafu {
                key: "qwen35.attention.head_count",
                rule: "execution requires nonzero divisible heads and a nonzero key width",
            }
            .fail();
        }
        let sections = i32_array(metadata, ROPE_SECTIONS_KEY)?;
        if sections.iter().any(|section| *section < 0) {
            return MetadataRelationSnafu {
                key: ROPE_SECTIONS_KEY,
                rule: "must not contain negative sections",
            }
            .fail();
        }
        let section_pairs = sections.iter().try_fold(0usize, |sum, item| {
            sum.checked_add(usize::try_from(*item).map_err(|_| {
                ArithmeticOverflowSnafu {
                    context: "MRoPE section",
                }
                .build()
            })?)
            .ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "MRoPE section pairs",
                }
                .build()
            })
        })?;
        if sections.len() != 4
            || section_pairs > key
            || sections[..3].iter().all(|section| *section == 0)
        {
            return MetadataRelationSnafu {
                key: ROPE_SECTIONS_KEY,
                rule: "must contain four sections no wider than the key width, with one text axis",
            }
            .fail();
        }
        let rope_base = optional_f32_meta(metadata, ROPE_FREQ_BASE_KEY)?.unwrap_or(10_000.0);
        if !rope_base.is_finite() || rope_base <= 0.0 {
            return MetadataRelationSnafu {
                key: ROPE_FREQ_BASE_KEY,
                rule: "must be finite and positive for execution",
            }
            .fail();
        }
        let scaling = optional_string_meta(metadata, ROPE_SCALING_TYPE_KEY)?.unwrap_or("linear");
        if !matches!(scaling, "linear" | "none") {
            return MetadataRelationSnafu {
                key: ROPE_SCALING_TYPE_KEY,
                rule: "only source-defined linear or none RoPE scaling is implemented",
            }
            .fail();
        }
        let configured_factor = optional_f32_meta(metadata, ROPE_SCALING_FACTOR_KEY)?;
        let raw_factor = match configured_factor {
            Some(factor) => factor,
            None => optional_f32_meta(metadata, ROPE_LEGACY_LINEAR_SCALE_KEY)?.unwrap_or(0.0),
        };
        let linear_factor = if matches!(raw_factor.to_bits(), 0 | 0x8000_0000) {
            1.0
        } else {
            raw_factor
        };
        // Source `none` explicitly resets frequency scaling after parsing the
        // metadata factor; a nonunit stored factor is therefore ineffective.
        let factor = if scaling == "none" {
            1.0
        } else {
            linear_factor
        };
        let attention_factor =
            optional_f32_meta(metadata, ROPE_SCALING_ATTENTION_FACTOR_KEY)?.unwrap_or(1.0);
        if !factor.is_finite()
            || factor <= 0.0
            || !attention_factor.is_finite()
            || attention_factor <= 0.0
        {
            return MetadataRelationSnafu {
                key: ROPE_SCALING_FACTOR_KEY,
                rule: "present scaling factors must be finite and positive",
            }
            .fail();
        }
        if factor.to_bits() != 1.0_f32.to_bits() || attention_factor.to_bits() != 1.0_f32.to_bits()
        {
            return MetadataRelationSnafu {
                key: ROPE_SCALING_TYPE_KEY,
                rule: "only source-defined effective unscaled RoPE is implemented",
            }
            .fail();
        }
        let kv_width = kv_heads.checked_mul(key).ok_or_else(|| {
            ArithmeticOverflowSnafu {
                context: "execution KV width",
            }
            .build()
        })?;
        let rope_sections = [
            usize::try_from(sections[0]).map_err(|_| {
                ArithmeticOverflowSnafu {
                    context: "MRoPE section",
                }
                .build()
            })?,
            usize::try_from(sections[1]).map_err(|_| {
                ArithmeticOverflowSnafu {
                    context: "MRoPE section",
                }
                .build()
            })?,
            usize::try_from(sections[2]).map_err(|_| {
                ArithmeticOverflowSnafu {
                    context: "MRoPE section",
                }
                .build()
            })?,
            usize::try_from(sections[3]).map_err(|_| {
                ArithmeticOverflowSnafu {
                    context: "MRoPE section",
                }
                .build()
            })?,
        ];
        Ok(Self {
            hidden,
            hidden_u64,
            feed_forward,
            heads,
            kv_heads,
            key,
            n_rot,
            key_u64,
            kv_width,
            query_width: heads.checked_mul(key).ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "execution query width",
                }
                .build()
            })?,
            gqa_group: heads / kv_heads,
            vocabulary,
            main_blocks,
            full_interval,
            max_context,
            epsilon: recurrent_layernorm_rms_epsilon(metadata)?,
            rope_base: f64::from(rope_base),
            rope_sections,
        })
    }
    fn is_full(self, block: usize) -> bool {
        (block + 1).is_multiple_of(self.full_interval)
    }
    fn rope_axis(self, pair: usize) -> Result<usize> {
        let total = self.rope_sections.iter().try_fold(0usize, |sum, section| {
            sum.checked_add(*section).ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "MRoPE section total",
                }
                .build()
            })
        })?;
        let sector = pair % total;
        if sector % 3 == 1 && sector < 3 * self.rope_sections[1] {
            return Ok(1);
        }
        if sector % 3 == 2 && sector < 3 * self.rope_sections[2] {
            return Ok(2);
        }
        if sector.is_multiple_of(3) && sector < 3 * self.rope_sections[0] {
            return Ok(0);
        }
        Ok(3)
    }
}

fn read_f32(weights: &Qwen35Weights<'_>, name: &str, expected_dims: &[u64]) -> Result<Vec<f32>> {
    let tensor = weights.payload().tensor(name).context(PayloadTensorSnafu {
        name: name.to_string(),
    })?;
    if tensor.ggml_type() != GgmlType::F32 {
        return ProjectionDtypeSnafu {
            name: tensor.name().to_string(),
            actual: tensor.ggml_type(),
        }
        .fail();
    }
    if tensor.dims() != expected_dims {
        return TensorShapeSnafu {
            name: tensor.name().to_string(),
            expected: expected_dims.to_vec(),
            actual: tensor.dims().to_vec(),
        }
        .fail();
    }
    let values = expected_dims.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(usize::try_from(*dimension).map_err(|_| {
                ArithmeticOverflowSnafu {
                    context: "F32 tensor dimension",
                }
                .build()
            })?)
            .ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "F32 tensor values",
                }
                .build()
            })
    })?;
    let row = F32Row::parse(tensor.bytes()).with_context(|_| ProjectionRowSnafu {
        name: tensor.name().to_string(),
        row: 0_usize,
    })?;
    if row.len() != values {
        return ProjectionBytesSnafu {
            name: tensor.name().to_string(),
            expected: values.checked_mul(4).ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "F32 tensor bytes",
                }
                .build()
            })?,
            actual: tensor.bytes().len(),
        }
        .fail();
    }
    let mut output = reserve("F32 parameter", values)?;
    for index in 0..values {
        output.push(row.value(index).ok_or_else(|| {
            ProjectionBytesSnafu {
                name: tensor.name().to_string(),
                expected: values * 4,
                actual: tensor.bytes().len(),
            }
            .build()
        })?);
    }
    finite(&output, "F32 parameter")?;
    Ok(output)
}

fn u32_meta(
    metadata: &std::collections::HashMap<String, MetaValue>,
    key: &'static str,
) -> Result<u32> {
    match metadata.get(key) {
        None => MissingMetadataSnafu { key }.fail(),
        Some(MetaValue::U32(value)) => Ok(*value),
        Some(value) => MetadataTypeSnafu {
            key,
            expected: "u32",
            actual: value.value_type(),
        }
        .fail(),
    }
}
fn optional_f32_meta(
    metadata: &std::collections::HashMap<String, MetaValue>,
    key: &'static str,
) -> Result<Option<f32>> {
    match metadata.get(key) {
        None => Ok(None),
        Some(MetaValue::F32(value)) => Ok(Some(*value)),
        Some(value) => MetadataTypeSnafu {
            key,
            expected: "f32",
            actual: value.value_type(),
        }
        .fail(),
    }
}
fn optional_u32_meta(
    metadata: &std::collections::HashMap<String, MetaValue>,
    key: &'static str,
) -> Result<Option<u32>> {
    match metadata.get(key) {
        None => Ok(None),
        Some(MetaValue::U32(value)) => Ok(Some(*value)),
        Some(value) => MetadataTypeSnafu {
            key,
            expected: "u32",
            actual: value.value_type(),
        }
        .fail(),
    }
}
fn optional_bool_meta(
    metadata: &std::collections::HashMap<String, MetaValue>,
    key: &'static str,
) -> Result<Option<bool>> {
    match metadata.get(key) {
        None => Ok(None),
        Some(MetaValue::Bool(value)) => Ok(Some(*value)),
        Some(value) => MetadataTypeSnafu {
            key,
            expected: "bool",
            actual: value.value_type(),
        }
        .fail(),
    }
}
fn optional_string_meta<'a>(
    metadata: &'a std::collections::HashMap<String, MetaValue>,
    key: &'static str,
) -> Result<Option<&'a str>> {
    match metadata.get(key) {
        None => Ok(None),
        Some(MetaValue::String(value)) => Ok(Some(value)),
        Some(value) => MetadataTypeSnafu {
            key,
            expected: "string",
            actual: value.value_type(),
        }
        .fail(),
    }
}
fn i32_array(
    metadata: &std::collections::HashMap<String, MetaValue>,
    key: &'static str,
) -> Result<Vec<i32>> {
    let Some(MetaValue::Array(array)) = metadata.get(key) else {
        return match metadata.get(key) {
            None => MissingMetadataSnafu { key }.fail(),
            Some(value) => MetadataTypeSnafu {
                key,
                expected: "array<i32>",
                actual: value.value_type(),
            }
            .fail(),
        };
    };
    if array.element_type() != MetaValueType::I32 {
        return MetadataRelationSnafu {
            key,
            rule: "must declare i32 array elements",
        }
        .fail();
    }
    array
        .values()
        .iter()
        .map(|value| match value {
            MetaValue::I32(number) => Ok(*number),
            _ => MetadataRelationSnafu {
                key,
                rule: "must contain only i32 values",
            }
            .fail(),
        })
        .collect()
}
fn block_name(block: usize, role: &str) -> String {
    format!("blk.{block}.{role}")
}
fn reserve(target: &'static str, length: usize) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| ArithmeticOverflowSnafu { context: target }.build())?;
    Ok(values)
}
fn clone_values(target: &'static str, source: &[f32]) -> Result<Vec<f32>> {
    let mut values = reserve(target, source.len())?;
    values.extend_from_slice(source);
    Ok(values)
}
fn reserve_slots<T>(target: &'static str, length: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| ArithmeticOverflowSnafu { context: target }.build())?;
    Ok(values)
}
fn finite(values: &[f32], stage: &'static str) -> Result<()> {
    for (index, value) in values.iter().copied().enumerate() {
        finite_one(value, stage, index)?;
    }
    Ok(())
}
fn finite_one(value: f32, stage: &'static str, index: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        ExecutionArithmeticSnafu { stage, index }.fail()
    }
}
fn add_in_place(destination: &mut [f32], source: &[f32], stage: &'static str) -> Result<()> {
    if destination.len() != source.len() {
        return ExecutionContextSnafu {
            requested: source.len(),
            rule: "residual operands must have the same hidden width",
        }
        .fail();
    }
    for (index, (left, right)) in destination.iter_mut().zip(source).enumerate() {
        *left += right;
        finite_one(*left, stage, index)?;
    }
    Ok(())
}
