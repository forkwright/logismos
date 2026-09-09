//! Bounded CPU token-to-logits execution for a verified Qwen3.5 payload.

use cache::{PagedAppend, PagedKvGeometry, PagedKvPlan, PagedKvPool};
use loader::gguf::{GgmlType, MetaValue, MetaValueType};
use quant::f32_row::F32Row;
use snafu::ResultExt;

use crate::error::{
    ArithmeticOverflowSnafu, ExecutionAllocationPlanSnafu, ExecutionAllocationSnafu,
    ExecutionArithmeticSnafu, ExecutionContextSnafu, ExecutionCpuSnafu,
    ExecutionPagedDecodePlanSnafu, ExecutionPagedDecodeSnafu, ExecutionPagedKvSnafu,
    ExecutionTokenSnafu, MetadataRelationSnafu, MetadataTypeSnafu, MissingMetadataSnafu,
    PayloadTensorSnafu, ProjectionBytesSnafu, ProjectionDtypeSnafu, ProjectionRowSnafu,
    RecurrentRmsNormSnafu, TensorShapeSnafu,
};
use crate::qwen35::recurrent_layernorm_rms_epsilon;
use crate::qwen35_mrope::{TextMrope, text_mrope_coefficient};
use crate::qwen35_requirements::Qwen35CpuRequirements;
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
pub(crate) const TOKEN_EMBEDDING: &str = "token_embd.weight";
pub(crate) const OUTPUT_NORM: &str = "output_norm.weight";
pub(crate) const OUTPUT: &str = "output.weight";

/// Select which token-logit rows a bounded execution retains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Qwen35LogitSelection {
    /// Retain one vocabulary-logit row for every supplied token.
    AllTokens,
    /// Execute every supplied token but retain only the final vocabulary-logit row.
    LastToken,
}

/// Opaque artifact-bound construction plan for one CPU execution session.
#[derive(Debug)]
pub struct Qwen35ExecutionPlan {
    weights: Qwen35Weights,
    layout: Layout,
    max_step_tokens: usize,
    selection: Qwen35LogitSelection,
    paged_kv_plan: Option<PagedKvPlan>,
    requirements: Qwen35CpuRequirements,
}

/// Stateful, bounded CPU text execution bound to one verified payload.
///
/// Each successful [`Self::step`] executes every input token and returns rows
/// according to the plan's [`Qwen35LogitSelection`]. The method stages recurrent
/// state while borrowing a private paged-KV append transaction; both publish only
/// after the whole call, including selected final logits, succeeds.
#[derive(Debug)]
pub struct Qwen35Execution {
    weights: Qwen35Weights,
    layout: Layout,
    max_step_tokens: usize,
    selection: Qwen35LogitSelection,
    layers: Vec<LayerState>,
    paged_kv_pool: Option<PagedKvPool>,
    position: usize,
}

#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "direct recurrent storage avoids an additional infallible per-layer heap allocation; the checked Vec allocation bounds every main block"
)]
enum LayerState {
    Recurrent(Qwen35RecurrentExecution),
    Full(usize),
}

impl Qwen35Execution {
    pub(crate) fn try_from_weights(weights: &Qwen35Weights, max_context: usize) -> Result<Self> {
        Qwen35ExecutionPlan::try_from_weights(
            weights,
            max_context,
            max_context,
            Qwen35LogitSelection::AllTokens,
        )?
        .execution()
    }
}

impl Qwen35ExecutionPlan {
    pub(crate) fn try_from_weights(
        weights: &Qwen35Weights,
        max_context: usize,
        max_step_tokens: usize,
        selection: Qwen35LogitSelection,
    ) -> Result<Self> {
        let layout = Layout::from_metadata(weights, max_context)?;
        if max_step_tokens == 0 || max_step_tokens > max_context {
            return ExecutionContextSnafu {
                requested: max_step_tokens,
                rule: "maximum step tokens must be nonzero and no greater than caller context",
            }
            .fail();
        }
        let paged_kv_plan = paged_kv_plan(layout)?;
        let requirements = Qwen35CpuRequirements::try_from_plan(
            weights,
            layout,
            max_step_tokens,
            selection,
            paged_kv_plan,
        )?;
        Ok(Self {
            weights: weights.clone(),
            layout,
            max_step_tokens,
            selection,
            paged_kv_plan,
            requirements,
        })
    }

    /// Construct the session described by this checked plan.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] if one plan-derived retained allocation or
    /// artifact-bound recurrent state cannot be constructed.
    pub fn execution(self) -> Result<Qwen35Execution> {
        let Self {
            weights,
            layout,
            max_step_tokens,
            selection,
            paged_kv_plan,
            requirements: _,
        } = self;
        let block_count = layout.main_blocks;
        let mut layers = reserve("main-block execution slots", block_count)?;
        let mut full_layer = 0;
        for block in 0..block_count {
            if layout.is_full(block) {
                layers.push(LayerState::Full(full_layer));
                full_layer = full_layer.checked_add(1).ok_or_else(|| {
                    ArithmeticOverflowSnafu {
                        context: "full-attention layer index",
                    }
                    .build()
                })?;
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
        let paged_kv_pool = paged_kv_plan
            .map(PagedKvPool::new)
            .transpose()
            .context(ExecutionPagedKvSnafu)?;
        Ok(Qwen35Execution {
            weights,
            layout,
            max_step_tokens,
            selection,
            layers,
            paged_kv_pool,
            position: 0,
        })
    }

    /// Return the precomputed logical CPU allocation envelope for this plan.
    ///
    /// WHY: all fallible size arithmetic runs during plan admission, so reading
    /// an admitted plan's requirements cannot introduce a new failure.
    #[must_use]
    pub const fn cpu_requirements(&self) -> Qwen35CpuRequirements {
        self.requirements
    }
}

impl Qwen35Execution {
    /// Execute complete token ids and return token-major vocabulary logits.
    ///
    /// The session owns only state derived from its verified payload; callers
    /// cannot inject KV or recurrent state. [`Qwen35LogitSelection::AllTokens`]
    /// returns one row per input token; [`Qwen35LogitSelection::LastToken`]
    /// still executes and commits every token but returns only the final row.
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
        if token_ids.len() > self.max_step_tokens || requested > self.layout.max_context {
            return ExecutionContextSnafu {
                requested,
                rule: "must not exceed the plan's step or caller-bounded context",
            }
            .fail();
        }
        let mut staged = self.stage()?;
        let mut append = self
            .paged_kv_pool
            .as_mut()
            .map(|pool| {
                pool.begin_append(token_ids.len())
                    .context(ExecutionPagedKvSnafu)
            })
            .transpose()?;
        let logits = staged.step_staged(token_ids, append.as_mut())?;
        if let Some(append) = append {
            append.commit().context(ExecutionPagedKvSnafu)?;
        }
        self.layers = staged.layers;
        self.position = staged.position;
        Ok(logits)
    }

    fn stage(&self) -> Result<StagedExecution> {
        let mut layers = reserve("transaction main-block slots", self.layers.len())?;
        for layer in &self.layers {
            layers.push(match layer {
                LayerState::Recurrent(execution) => {
                    LayerState::Recurrent(execution.try_clone_for_transaction()?)
                }
                LayerState::Full(layer) => LayerState::Full(*layer),
            });
        }
        Ok(StagedExecution {
            weights: self.weights.clone(),
            layout: self.layout,
            selection: self.selection,
            layers,
            position: self.position,
        })
    }
}

struct StagedExecution {
    weights: Qwen35Weights,
    layout: Layout,
    selection: Qwen35LogitSelection,
    layers: Vec<LayerState>,
    position: usize,
}

impl StagedExecution {
    fn step_staged(
        &mut self,
        token_ids: &[u32],
        mut append: Option<&mut PagedAppend<'_>>,
    ) -> Result<Vec<f32>> {
        let total = returned_logits_elements(self.layout, token_ids.len(), self.selection)?;
        let mut logits = reserve("token logits", total)?;
        for (token_index, token_id) in token_ids.iter().enumerate() {
            let mut hidden = self.embed(*token_id)?;
            for block in 0..self.layout.main_blocks {
                let weights = &self.weights;
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
                    LayerState::Full(full_layer) => full_attention(
                        weights,
                        layout,
                        FullAttentionStep {
                            position,
                            block,
                            full_layer: *full_layer,
                            append_token: token_index,
                        },
                        &hidden,
                        append.as_deref_mut().ok_or_else(|| {
                            ExecutionContextSnafu {
                                requested: block,
                                rule: "full-attention execution must own a paged KV transaction",
                            }
                            .build()
                        })?,
                    )?,
                };
                self.finish_layer(block, &mut hidden, &attention)?;
            }
            let lm_head = LmHeadWorkspaceAllocations::try_from_layout(self.layout)?;
            let output_norm = read_f32(
                self.weights,
                OUTPUT_NORM,
                &[self.layout.hidden_u64],
                lm_head.output_norm,
            )?;
            let normalized = kernels::cpu_f32::rms_norm(
                &hidden,
                &output_norm,
                1,
                self.layout.hidden,
                self.layout.epsilon,
            )
            .context(RecurrentRmsNormSnafu)?;
            ensure_execution_plan(
                "LM-head normalized hidden",
                normalized.len(),
                lm_head.normalized_hidden,
            )?;
            if matches!(self.selection, Qwen35LogitSelection::AllTokens)
                || token_index + 1 == token_ids.len()
            {
                logits.extend(project_checked(
                    self.weights,
                    OUTPUT,
                    &normalized,
                    lm_head.vocabulary_projection,
                )?);
            }
            self.position = self.position.checked_add(1).ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "execution position increment",
                }
                .build()
            })?;
        }
        Ok(logits)
    }

    fn finish_layer(&self, block: usize, hidden: &mut [f32], attention: &[f32]) -> Result<()> {
        let finish = LayerFinishWorkspaceAllocations::try_from_layout(self.layout)?;
        ensure_execution_plan(
            "layer attention output",
            attention.len(),
            finish.attention_output,
        )?;
        add_in_place(hidden, attention, "attention residual")?;
        let post_norm = read_f32(
            self.weights,
            &block_name(block, "post_attention_norm.weight"),
            &[self.layout.hidden_u64],
            finish.post_attention_norm,
        )?;
        let normalized = kernels::cpu_f32::rms_norm(
            hidden,
            &post_norm,
            1,
            self.layout.hidden,
            self.layout.epsilon,
        )
        .context(RecurrentRmsNormSnafu)?;
        ensure_execution_plan(
            "post-attention normalized hidden",
            normalized.len(),
            finish.normalized_hidden,
        )?;
        let ffn = self.ffn(block, &normalized, finish.feed_forward)?;
        add_in_place(hidden, &ffn, "FFN residual")
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
        let planned = Qwen35Weights::decoded_row_elements(self.layout.hidden);
        if embedding.len() != planned {
            return ExecutionContextSnafu {
                requested: embedding.len(),
                rule: "token embedding row must match artifact-derived hidden width",
            }
            .fail();
        }
        finite(&embedding, "token embedding")?;
        Ok(embedding)
    }

    fn ffn(
        &self,
        block: usize,
        input: &[f32],
        allocations: FeedForwardWorkspaceAllocations,
    ) -> Result<Vec<f32>> {
        let gate = project_checked(
            self.weights,
            &block_name(block, "ffn_gate.weight"),
            input,
            allocations.gate_projection,
        )?;
        let up = project_checked(
            self.weights,
            &block_name(block, "ffn_up.weight"),
            input,
            allocations.up_projection,
        )?;
        let activated = kernels::cpu_f32::try_silu(&gate).context(ExecutionCpuSnafu)?;
        ensure_execution_plan(
            "SwiGLU activated gate",
            activated.len(),
            allocations.activated_gate,
        )?;
        let mut fused = reserve("SwiGLU activation", allocations.fused_activation)?;
        for (index, (left, right)) in activated.iter().zip(up.iter()).enumerate() {
            let value = left * right;
            finite_one(value, "SwiGLU", index)?;
            fused.push(value);
        }
        project_checked(
            self.weights,
            &block_name(block, "ffn_down.weight"),
            &fused,
            allocations.down_projection,
        )
    }
}

fn paged_kv_plan(layout: Layout) -> Result<Option<PagedKvPlan>> {
    if layout.full_layer_count() == 0 {
        return Ok(None);
    }
    PagedKvPlan::select(PagedKvGeometry {
        layers: layout.full_layer_count(),
        row_width: layout.kv_width,
        max_context: layout.max_context,
    })
    .map(Some)
    .context(ExecutionPagedKvSnafu)
}

#[derive(Clone, Copy)]
struct FullAttentionStep {
    position: usize,
    block: usize,
    full_layer: usize,
    append_token: usize,
}

#[expect(
    clippy::too_many_lines,
    reason = "the pinned full-attention operation order is one bounded transactional unit"
)]
fn full_attention(
    weights: &Qwen35Weights,
    layout: Layout,
    step: FullAttentionStep,
    input: &[f32],
    append: &mut PagedAppend<'_>,
) -> Result<Vec<f32>> {
    let FullAttentionStep {
        position,
        block,
        full_layer,
        append_token,
    } = step;
    let attention_tokens = position.checked_add(1).ok_or_else(|| {
        ArithmeticOverflowSnafu {
            context: "full-attention token count",
        }
        .build()
    })?;
    let allocations = FullAttentionWorkspaceAllocations::try_from_layout(layout, attention_tokens)?;
    let norm = read_f32(
        weights,
        &block_name(block, "attn_norm.weight"),
        &[layout.hidden_u64],
        allocations.attention_norm,
    )?;
    let normalized = kernels::cpu_f32::rms_norm(input, &norm, 1, layout.hidden, layout.epsilon)
        .context(RecurrentRmsNormSnafu)?;
    ensure_execution_plan(
        "full-attention normalized input",
        normalized.len(),
        allocations.normalized_input,
    )?;
    let q_gate = project_checked(
        weights,
        &block_name(block, "attn_q.weight"),
        &normalized,
        allocations.query_gate_projection,
    )?;
    let key = project_checked(
        weights,
        &block_name(block, "attn_k.weight"),
        &normalized,
        allocations.key_projection,
    )?;
    let value = project_checked(
        weights,
        &block_name(block, "attn_v.weight"),
        &normalized,
        allocations.value_projection,
    )?;
    let q_norm = read_f32(
        weights,
        &block_name(block, "attn_q_norm.weight"),
        &[layout.key_u64],
        allocations.query_norm,
    )?;
    let k_norm = read_f32(
        weights,
        &block_name(block, "attn_k_norm.weight"),
        &[layout.key_u64],
        allocations.key_norm,
    )?;
    let mut query = reserve("full-attention query", allocations.split_query)?;
    let mut gate = reserve("full-attention gate", allocations.split_gate)?;
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
        query.extend_from_slice(q_gate.get(start..middle).ok_or_else(|| {
            ExecutionContextSnafu {
                requested: start,
                rule: "Q/gate projection must be interleaved per head",
            }
            .build()
        })?);
        gate.extend_from_slice(q_gate.get(middle..end).ok_or_else(|| {
            ExecutionContextSnafu {
                requested: middle,
                rule: "Q/gate projection must be interleaved per head",
            }
            .build()
        })?);
    }
    query = kernels::cpu_f32::rms_norm(&query, &q_norm, layout.heads, layout.key, layout.epsilon)
        .context(RecurrentRmsNormSnafu)?;
    ensure_execution_plan(
        "full-attention normalized query",
        query.len(),
        allocations.normalized_query,
    )?;
    let mut key =
        kernels::cpu_f32::rms_norm(&key, &k_norm, layout.kv_heads, layout.key, layout.epsilon)
            .context(RecurrentRmsNormSnafu)?;
    ensure_execution_plan(
        "full-attention normalized key",
        key.len(),
        allocations.normalized_key,
    )?;
    apply_text_mrope(layout, position, &mut query)?;
    apply_text_mrope(layout, position, &mut key)?;
    append
        .write_layer_row(full_layer, append_token, &key, &value)
        .context(ExecutionPagedKvSnafu)?;
    let kv = append.layer_kv(full_layer).context(ExecutionPagedKvSnafu)?;
    if kv.tokens() != attention_tokens {
        return ExecutionContextSnafu {
            requested: kv.tokens(),
            rule: "paged KV transaction rows must match token-serial full-attention state",
        }
        .fail();
    }
    let mut merged = reserve("full-attention merged output", allocations.merged_output)?;
    for head in 0..layout.heads {
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
        let attended = paged_decode(allocations.paged_decode, head, query, &kv)?;
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
    project_checked(
        weights,
        &block_name(block, "attn_output.weight"),
        &merged,
        allocations.output_projection,
    )
}

fn apply_text_mrope(layout: Layout, position: usize, values: &mut [f32]) -> Result<()> {
    for row in values.chunks_exact_mut(layout.key) {
        for pair in 0..(layout.n_rot / 2) {
            let left_index = pair;
            let (cos, sin) = text_mrope_coefficient(layout.text_mrope(), position, pair)?;
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

#[derive(Debug, Clone, Copy)]
struct FullAttentionWorkspaceAllocations {
    attention_norm: usize,
    normalized_input: usize,
    query_gate_projection: usize,
    key_projection: usize,
    value_projection: usize,
    query_norm: usize,
    key_norm: usize,
    split_query: usize,
    split_gate: usize,
    normalized_query: usize,
    normalized_key: usize,
    merged_output: usize,
    paged_decode: kernels::PagedDecodePlan,
    output_projection: usize,
    workspace_elements: usize,
}

impl FullAttentionWorkspaceAllocations {
    fn try_from_layout(layout: Layout, attention_tokens: usize) -> Result<Self> {
        let attention_norm = f32_tensor_elements(&[layout.hidden_u64])?;
        let normalized_input = kernels::cpu_f32::rms_norm_output_elements(1, layout.hidden)
            .context(RecurrentRmsNormSnafu)?;
        let query_gate_projection = Qwen35Weights::projection_output_elements(
            layout.query_width.checked_mul(2).ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "full-attention Q/gate projection",
                }
                .build()
            })?,
        );
        let key_projection = Qwen35Weights::projection_output_elements(layout.kv_width);
        let value_projection = Qwen35Weights::projection_output_elements(layout.kv_width);
        let query_norm = f32_tensor_elements(&[layout.key_u64])?;
        let key_norm = f32_tensor_elements(&[layout.key_u64])?;
        let split_query = layout.query_width;
        let split_gate = layout.query_width;
        let normalized_query = kernels::cpu_f32::rms_norm_output_elements(layout.heads, layout.key)
            .context(RecurrentRmsNormSnafu)?;
        let normalized_key =
            kernels::cpu_f32::rms_norm_output_elements(layout.kv_heads, layout.key)
                .context(RecurrentRmsNormSnafu)?;
        let merged_output = layout.query_width;
        let paged_decode = kernels::PagedDecodePlan::try_from_dimensions(
            attention_tokens,
            layout.heads,
            layout.kv_heads,
            layout.key,
        )
        .context(ExecutionPagedDecodePlanSnafu)?;
        let output_projection = Qwen35Weights::projection_output_elements(layout.hidden);
        let core = sum_elements(
            &[
                attention_norm,
                normalized_input,
                query_gate_projection,
                key_projection,
                value_projection,
                query_norm,
                key_norm,
                split_query,
                split_gate,
                normalized_query,
                normalized_key,
                merged_output,
            ],
            "full-attention core workspace",
        )?;
        let attention_phase = sum_elements(
            &[core, paged_decode.workspace_elements()],
            "full-attention score phase",
        )?;
        let output_phase = checked_add(
            core,
            output_projection,
            "full-attention output projection phase",
        )?;
        Ok(Self {
            attention_norm,
            normalized_input,
            query_gate_projection,
            key_projection,
            value_projection,
            query_norm,
            key_norm,
            split_query,
            split_gate,
            normalized_query,
            normalized_key,
            merged_output,
            paged_decode,
            output_projection,
            workspace_elements: attention_phase.max(output_phase),
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct FeedForwardWorkspaceAllocations {
    gate_projection: usize,
    up_projection: usize,
    activated_gate: usize,
    fused_activation: usize,
    down_projection: usize,
}

#[derive(Debug, Clone, Copy)]
struct LayerFinishWorkspaceAllocations {
    attention_output: usize,
    post_attention_norm: usize,
    normalized_hidden: usize,
    feed_forward: FeedForwardWorkspaceAllocations,
}

impl LayerFinishWorkspaceAllocations {
    fn try_from_layout(layout: Layout) -> Result<Self> {
        Ok(Self {
            attention_output: Qwen35Weights::projection_output_elements(layout.hidden),
            post_attention_norm: f32_tensor_elements(&[layout.hidden_u64])?,
            normalized_hidden: kernels::cpu_f32::rms_norm_output_elements(1, layout.hidden)
                .context(RecurrentRmsNormSnafu)?,
            feed_forward: FeedForwardWorkspaceAllocations::from_layout(layout),
        })
    }

    fn total_elements(self) -> Result<usize> {
        sum_elements(
            &[
                self.attention_output,
                self.post_attention_norm,
                self.normalized_hidden,
                self.feed_forward.total_elements()?,
            ],
            "post-attention feed-forward phase",
        )
    }
}

impl FeedForwardWorkspaceAllocations {
    fn from_layout(layout: Layout) -> Self {
        Self {
            gate_projection: Qwen35Weights::projection_output_elements(layout.feed_forward),
            up_projection: Qwen35Weights::projection_output_elements(layout.feed_forward),
            activated_gate: kernels::cpu_f32::unary_output_elements(layout.feed_forward),
            fused_activation: layout.feed_forward,
            down_projection: Qwen35Weights::projection_output_elements(layout.hidden),
        }
    }

    fn total_elements(self) -> Result<usize> {
        sum_elements(
            &[
                self.gate_projection,
                self.up_projection,
                self.activated_gate,
                self.fused_activation,
                self.down_projection,
            ],
            "feed-forward workspace",
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct LmHeadWorkspaceAllocations {
    output_norm: usize,
    normalized_hidden: usize,
    vocabulary_projection: usize,
}

impl LmHeadWorkspaceAllocations {
    fn try_from_layout(layout: Layout) -> Result<Self> {
        Ok(Self {
            output_norm: f32_tensor_elements(&[layout.hidden_u64])?,
            normalized_hidden: kernels::cpu_f32::rms_norm_output_elements(1, layout.hidden)
                .context(RecurrentRmsNormSnafu)?,
            vocabulary_projection: Qwen35Weights::projection_output_elements(layout.vocabulary),
        })
    }

    fn total_elements(self) -> Result<usize> {
        sum_elements(
            &[
                self.output_norm,
                self.normalized_hidden,
                self.vocabulary_projection,
            ],
            "LM-head workspace",
        )
    }
}

pub(crate) const fn full_attention_retained_elements(plan: PagedKvPlan) -> usize {
    plan.requested_f32_elements()
}

pub(crate) fn full_attention_workspace_elements(
    layout: Layout,
    attention_tokens: usize,
) -> Result<usize> {
    Ok(
        FullAttentionWorkspaceAllocations::try_from_layout(layout, attention_tokens)?
            .workspace_elements,
    )
}

pub(crate) fn layer_finish_workspace_elements(layout: Layout) -> Result<usize> {
    LayerFinishWorkspaceAllocations::try_from_layout(layout)?.total_elements()
}

pub(crate) const fn embedding_workspace_elements(layout: Layout) -> usize {
    Qwen35Weights::decoded_row_elements(layout.hidden)
}

pub(crate) fn lm_head_workspace_elements(layout: Layout) -> Result<usize> {
    LmHeadWorkspaceAllocations::try_from_layout(layout)?.total_elements()
}

pub(crate) fn returned_logits_elements(
    layout: Layout,
    token_count: usize,
    selection: Qwen35LogitSelection,
) -> Result<usize> {
    let rows = match selection {
        Qwen35LogitSelection::AllTokens => token_count,
        Qwen35LogitSelection::LastToken => 1,
    };
    rows.checked_mul(layout.vocabulary).ok_or_else(|| {
        ArithmeticOverflowSnafu {
            context: "returned logits allocation",
        }
        .build()
    })
}

fn paged_decode(
    plan: kernels::PagedDecodePlan,
    query_head: usize,
    query: &[f32],
    kv: &cache::PagedLayerKv<'_>,
) -> Result<Vec<f32>> {
    kernels::paged_decode_cpu(
        plan,
        query_head,
        query,
        |token| kv.key_row(token),
        |token| kv.value_row(token),
    )
    .context(ExecutionPagedDecodeSnafu)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Layout {
    pub(crate) hidden: usize,
    hidden_u64: u64,
    pub(crate) feed_forward: usize,
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) key: usize,
    n_rot: usize,
    key_u64: u64,
    pub(crate) kv_width: usize,
    pub(crate) query_width: usize,
    vocabulary: usize,
    main_blocks: usize,
    full_interval: usize,
    max_context: usize,
    pub(crate) epsilon: f32,
    rope_base: f64,
    rope_sections: [usize; 4],
}

impl Layout {
    #[cfg(feature = "gpu")]
    pub(crate) const fn vocabulary(self) -> usize {
        self.vocabulary
    }

    #[cfg(feature = "gpu")]
    pub(crate) const fn main_block_count(self) -> usize {
        self.main_blocks
    }

    #[cfg(feature = "gpu")]
    pub(crate) const fn hidden_dimension(self) -> u64 {
        self.hidden_u64
    }

    pub(crate) const fn max_context(self) -> usize {
        self.max_context
    }

    pub(crate) const fn epsilon(self) -> f32 {
        self.epsilon
    }

    pub(crate) const fn full_layer_count(self) -> usize {
        self.main_blocks / self.full_interval
    }

    pub(crate) fn recurrent_layer_count(self) -> Result<usize> {
        self.main_blocks
            .checked_sub(self.full_layer_count())
            .ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "recurrent execution layer count",
                }
                .build()
            })
    }

    #[expect(
        clippy::too_many_lines,
        reason = "execution-only metadata is admitted at one artifact-bound boundary"
    )]
    pub(crate) fn from_metadata(weights: &Qwen35Weights, max_context: usize) -> Result<Self> {
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
        let sections = rope_sections(metadata, ROPE_SECTIONS_KEY)?;
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
        if section_pairs > key || sections[..3].iter().all(|section| *section == 0) {
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
            vocabulary,
            main_blocks,
            full_interval,
            max_context,
            epsilon: recurrent_layernorm_rms_epsilon(metadata)?,
            rope_base: f64::from(rope_base),
            rope_sections,
        })
    }
    pub(crate) fn is_full(self, block: usize) -> bool {
        (block + 1).is_multiple_of(self.full_interval)
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn is_admitted_full_block(self, block: usize) -> bool {
        block < self.main_blocks && self.is_full(block)
    }
    pub(crate) const fn text_mrope(self) -> TextMrope {
        TextMrope::new(self.n_rot, self.rope_base, self.rope_sections)
    }
}

pub(crate) fn read_f32(
    weights: &Qwen35Weights,
    name: &str,
    expected_dims: &[u64],
    planned_values: usize,
) -> Result<Vec<f32>> {
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
    let values = f32_tensor_elements(expected_dims)?;
    ensure_execution_plan("F32 parameter", values, planned_values)?;
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
    let mut output = reserve("F32 parameter", planned_values)?;
    for index in 0..planned_values {
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
fn rope_sections(
    metadata: &std::collections::HashMap<String, MetaValue>,
    key: &'static str,
) -> Result<[i32; 4]> {
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
    if array.values().len() != 4 {
        return MetadataRelationSnafu {
            key,
            rule: "must contain exactly four sections",
        }
        .fail();
    }
    let mut sections = [0; 4];
    for (section, value) in sections.iter_mut().zip(array.values()) {
        match value {
            MetaValue::I32(number) => *section = *number,
            _ => {
                return MetadataRelationSnafu {
                    key,
                    rule: "must contain only i32 values",
                }
                .fail();
            }
        }
    }
    Ok(sections)
}
pub(crate) fn block_name(block: usize, role: &str) -> String {
    format!("blk.{block}.{role}")
}
fn reserve<T>(target: &'static str, length: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .context(ExecutionAllocationSnafu { target, length })?;
    Ok(values)
}

fn f32_tensor_elements(dimensions: &[u64]) -> Result<usize> {
    dimensions.iter().try_fold(1_usize, |elements, dimension| {
        let dimension = usize::try_from(*dimension).map_err(|_| {
            ArithmeticOverflowSnafu {
                context: "F32 tensor dimension",
            }
            .build()
        })?;
        elements.checked_mul(dimension).ok_or_else(|| {
            ArithmeticOverflowSnafu {
                context: "F32 tensor values",
            }
            .build()
        })
    })
}

fn project_checked(
    weights: &Qwen35Weights,
    name: &str,
    input: &[f32],
    planned_values: usize,
) -> Result<Vec<f32>> {
    let output = weights.project(name, input)?;
    ensure_execution_plan("matrix projection", output.len(), planned_values)?;
    Ok(output)
}

fn ensure_execution_plan(target: &'static str, derived: usize, planned: usize) -> Result<()> {
    if derived != planned {
        return ExecutionAllocationPlanSnafu {
            target,
            planned,
            derived,
        }
        .fail();
    }
    Ok(())
}

fn checked_add(left: usize, right: usize, context: &'static str) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| ArithmeticOverflowSnafu { context }.build())
}

fn sum_elements(elements: &[usize], context: &'static str) -> Result<usize> {
    elements
        .iter()
        .copied()
        .try_fold(0_usize, |sum, value| checked_add(sum, value, context))
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

#[cfg(test)]
mod qwen35_execution_oracle_tests;

#[cfg(test)]
mod qwen35_execution_requirements_tests;
