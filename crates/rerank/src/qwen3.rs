//! Bounded native CPU Qwen3 cross-encoder reranking.

use loader::gguf::{MetaValue, MetaValueType, VerifiedArtifact};
use num_traits::ToPrimitive;
use serde::Serialize;
use snafu::ResultExt;
use templates::{BoundedTemplate, TemplateLimits};
use tokenize::VerifiedTokenizer;

use decoders::{Qwen3CpuRequirements, Qwen3RankWeights};

use crate::batch::{Predictions, RerankBatch};
use crate::error::{
    EmptyBatchSnafu, EmptyDocumentSnafu, EmptyQuerySnafu, Qwen3BatchTooLargeSnafu,
    Qwen3DecoderSnafu, Qwen3InputByteLengthOverflowSnafu, Qwen3InputBytesTooLongSnafu,
    Qwen3InputTokensTooLongSnafu, Qwen3LimitsSnafu, Qwen3MetadataSnafu, Qwen3NonFiniteScoreSnafu,
    Qwen3RequirementsOverflowSnafu, Qwen3TemplateSnafu, Qwen3TokenizerSnafu, Result,
};
use crate::reranker::Reranker;

const TOKENS: &str = "tokenizer.ggml.tokens";
const CHAT_TEMPLATE: &str = "tokenizer.chat_template";
const ADD_BOS: &str = "tokenizer.ggml.add_bos_token";
const ADD_EOS: &str = "tokenizer.ggml.add_eos_token";
const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const SCORE_OUTPUT_ELEMENTS: usize = 1;

/// Explicit CPU work limits for one Qwen3 reranker.
///
/// These limits bound accepted input and retained renderer output, not every
/// tokenizer or template temporary allocation in the process. Requests that
/// exceed them are refused rather than truncated.
#[derive(Clone, Copy, Debug)]
pub struct Qwen3RerankerLimits {
    /// Maximum checked UTF-8 bytes across instruction, query, and document.
    pub max_pair_bytes: usize,
    /// Maximum rendered token IDs per pair.
    pub max_tokens: usize,
    /// Maximum pairs accepted by one prediction request.
    pub max_batch_items: usize,
    /// Validated independent limits for artifact-owned template rendering.
    pub template: TemplateLimits,
}

/// Logical CPU `f32` payload requirements for the Qwen3 rerank adapter.
///
/// This report composes one artifact-bound rank execution with the sequential
/// scalar result rows owned by the native reranker. It excludes B-tree nodes,
/// `Vec` headers and allocator capacity, tokenizer/template data and
/// intermediates, process memory, physical residency, and GPU memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen3RerankerCpuRequirements {
    decoder: Qwen3CpuRequirements,
    max_batch_items: usize,
    returned_output_bytes: u64,
    logical_f32_upper_bound_bytes: u64,
}

impl Qwen3RerankerCpuRequirements {
    /// Return the one-item artifact-bound rank decoder requirement report.
    #[must_use]
    pub const fn decoder_cpu_requirements(self) -> Qwen3CpuRequirements {
        self.decoder
    }

    /// Return the trusted maximum number of sequential score rows.
    #[must_use]
    pub const fn max_batch_items(self) -> usize {
        self.max_batch_items
    }

    /// Return caller-retained scalar-score `f32` payload bytes.
    #[must_use]
    pub const fn returned_output_bytes(self) -> u64 {
        self.returned_output_bytes
    }

    /// Return the sequential-batch logical `f32` payload upper bound.
    #[must_use]
    pub const fn logical_f32_upper_bound_bytes(self) -> u64 {
        self.logical_f32_upper_bound_bytes
    }
}

/// Artifact-bound native CPU Qwen3 reranker with one signed relevance score per pair.
pub struct Qwen3Reranker<'artifact> {
    weights: Qwen3RankWeights<'artifact>,
    tokenizer: VerifiedTokenizer,
    template: BoundedTemplate<'artifact>,
    instruction: String,
    max_pair_bytes: usize,
    max_tokens: usize,
    max_batch_items: usize,
}

impl<'artifact> Qwen3Reranker<'artifact> {
    /// Construct one bounded CPU-only Qwen3 reranker from verified artifacts and trusted setup.
    ///
    /// # Errors
    ///
    /// Returns a typed error when setup limits, artifact metadata, tokenizer
    /// vocabulary/special tokens, template compilation, or rank weights fail
    /// their bounded Qwen3 contract.
    pub fn from_verified_cpu(
        artifact: &'artifact VerifiedArtifact,
        tokenizer: VerifiedTokenizer,
        limits: Qwen3RerankerLimits,
        instruction: String,
    ) -> Result<Self> {
        validate_limits(limits)?;
        if instruction.trim().is_empty() {
            return Qwen3LimitsSnafu {
                rule: "trusted rerank instruction must not be blank",
            }
            .fail();
        }
        tokenizer
            .verify_unpadded_untruncated()
            .context(Qwen3TokenizerSnafu)?;
        let metadata = artifact.observation().metadata();
        let vocabulary = vocabulary(metadata)?;
        tokenizer
            .verify_exact_vocabulary(
                vocabulary.len(),
                vocabulary.iter().filter_map(|value| match value {
                    MetaValue::String(spelling) => Some(spelling.as_str()),
                    _ => None,
                }),
            )
            .context(Qwen3TokenizerSnafu)?;
        require_no_automatic_specials(metadata, ADD_BOS)?;
        require_no_automatic_specials(metadata, ADD_EOS)?;
        verify_special(&tokenizer, vocabulary.len(), IM_START)?;
        verify_special(&tokenizer, vocabulary.len(), IM_END)?;
        let Some(MetaValue::String(template_source)) = metadata.get(CHAT_TEMPLATE) else {
            return Qwen3MetadataSnafu { key: CHAT_TEMPLATE }.fail();
        };
        let template =
            BoundedTemplate::new(template_source, limits.template).context(Qwen3TemplateSnafu)?;
        let weights = Qwen3RankWeights::try_from_verified(artifact).context(Qwen3DecoderSnafu)?;
        if limits.max_tokens > weights.max_context() {
            return Qwen3LimitsSnafu {
                rule: "setup token limit exceeds artifact context",
            }
            .fail();
        }
        Ok(Self {
            weights,
            tokenizer,
            template,
            instruction,
            max_pair_bytes: limits.max_pair_bytes,
            max_tokens: limits.max_tokens,
            max_batch_items: limits.max_batch_items,
        })
    }

    /// Return logical CPU `f32` payload requirements at this reranker's trusted setup maxima.
    ///
    /// The embedded decoder report carries immutable serialized backing
    /// separately. This wrapper counts only requested scalar-score payloads;
    /// it is not an allocator, process, physical-residency, or GPU budget.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] when the admitted rank decoder report or the
    /// checked sequential result composition cannot be represented.
    pub fn cpu_requirements(&self) -> Result<Qwen3RerankerCpuRequirements> {
        let decoder = self
            .weights
            .execution(self.max_tokens)
            .context(Qwen3DecoderSnafu)?
            .cpu_requirements()
            .context(Qwen3DecoderSnafu)?;
        let one_score_bytes = score_output_bytes()?;
        let returned_output_bytes = multiply_bytes(
            one_score_bytes,
            self.max_batch_items,
            "rerank batch returned scores",
        )?;
        let prior_output_bytes = multiply_bytes(
            one_score_bytes,
            self.max_batch_items - 1,
            "rerank prior batch scores",
        )?;
        let in_flight_bytes = prior_output_bytes
            .checked_add(decoder.workspace_upper_bound_bytes())
            .ok_or_else(|| {
                Qwen3RequirementsOverflowSnafu {
                    target: "rerank sequential in-flight payload",
                }
                .build()
            })?;
        Ok(Qwen3RerankerCpuRequirements {
            decoder,
            max_batch_items: self.max_batch_items,
            returned_output_bytes,
            logical_f32_upper_bound_bytes: returned_output_bytes.max(in_flight_bytes),
        })
    }

    fn score_item(&self, index: usize, query: &str, document: &str) -> Result<f32> {
        let input_bytes = checked_input_bytes(&self.instruction, query, document)?;
        if input_bytes > self.max_pair_bytes {
            return Qwen3InputBytesTooLongSnafu {
                index,
                actual: input_bytes,
                limit: self.max_pair_bytes,
            }
            .fail();
        }
        let rendered = self
            .template
            .render(RenderContext::new(&self.instruction, query, document))
            .context(Qwen3TemplateSnafu)?;
        let token_ids = self
            .tokenizer
            .tokenizer()
            .encode(&rendered, false)
            .context(Qwen3TokenizerSnafu)?;
        if token_ids.is_empty() || token_ids.len() > self.max_tokens {
            return Qwen3InputTokensTooLongSnafu {
                index,
                actual: token_ids.len(),
                limit: self.max_tokens,
            }
            .fail();
        }
        let logits = self
            .weights
            .execution(self.max_tokens)
            .context(Qwen3DecoderSnafu)?
            .last_logits(&token_ids)
            .context(Qwen3DecoderSnafu)?;
        signed_relevance_score(index, logits)
    }
}

impl Reranker for Qwen3Reranker<'_> {
    fn predict(&self, batch: RerankBatch) -> Result<Predictions> {
        validate_batch(&batch, self.max_batch_items)?;
        let mut predictions = Predictions::new();
        for (index, item) in batch.items.iter().enumerate() {
            let score = self.score_item(index, &item.query, &item.document)?;
            let mut row = score_output()?;
            row.push(score);
            predictions.insert(index, row);
        }
        Ok(predictions)
    }
}

#[derive(Serialize)]
struct RenderContext<'text> {
    messages: [RenderMessage<'text>; 3],
}

impl<'text> RenderContext<'text> {
    const fn new(instruction: &'text str, query: &'text str, document: &'text str) -> Self {
        Self {
            messages: [
                RenderMessage {
                    role: RenderRole::System,
                    content: instruction,
                },
                RenderMessage {
                    role: RenderRole::Query,
                    content: query,
                },
                RenderMessage {
                    role: RenderRole::Document,
                    content: document,
                },
            ],
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "lowercase")]
enum RenderRole {
    System,
    Query,
    Document,
}

#[derive(Serialize)]
struct RenderMessage<'text> {
    role: RenderRole,
    content: &'text str,
}

fn validate_limits(limits: Qwen3RerankerLimits) -> Result<()> {
    if [
        limits.max_pair_bytes,
        limits.max_tokens,
        limits.max_batch_items,
    ]
    .contains(&0)
    {
        return Qwen3LimitsSnafu {
            rule: "pair, token, and batch limits must be nonzero",
        }
        .fail();
    }
    Ok(())
}

fn vocabulary(metadata: &std::collections::HashMap<String, MetaValue>) -> Result<&[MetaValue]> {
    let Some(MetaValue::Array(values)) = metadata.get(TOKENS) else {
        return Qwen3MetadataSnafu { key: TOKENS }.fail();
    };
    if values.element_type() != MetaValueType::String {
        return Qwen3MetadataSnafu { key: TOKENS }.fail();
    }
    Ok(values.values())
}

fn require_no_automatic_specials(
    metadata: &std::collections::HashMap<String, MetaValue>,
    key: &'static str,
) -> Result<()> {
    match metadata.get(key) {
        None | Some(MetaValue::Bool(false)) => Ok(()),
        _ => Qwen3MetadataSnafu { key }.fail(),
    }
}

fn verify_special(
    tokenizer: &VerifiedTokenizer,
    vocabulary_size: usize,
    spelling: &str,
) -> Result<()> {
    let id = tokenizer
        .tokenizer()
        .token_to_id(spelling)
        .ok_or_else(|| Qwen3MetadataSnafu { key: TOKENS }.build())?;
    if tokenizer.tokenizer().id_to_token(id).as_deref() != Some(spelling) {
        return Qwen3MetadataSnafu { key: TOKENS }.fail();
    }
    tokenizer
        .verify_declared_special_id(vocabulary_size, id)
        .context(Qwen3TokenizerSnafu)
}

fn checked_input_bytes(instruction: &str, query: &str, document: &str) -> Result<usize> {
    instruction
        .len()
        .checked_add(query.len())
        .and_then(|total| total.checked_add(document.len()))
        .ok_or_else(|| Qwen3InputByteLengthOverflowSnafu.build())
}

fn signed_relevance_score(index: usize, logits: [f32; 2]) -> Result<f32> {
    (f64::from(logits[0]) - f64::from(logits[1]))
        .to_f32()
        .filter(|score| score.is_finite())
        .ok_or_else(|| Qwen3NonFiniteScoreSnafu { index }.build())
}

fn score_output() -> Result<Vec<f32>> {
    let mut scores = Vec::new();
    scores.try_reserve_exact(SCORE_OUTPUT_ELEMENTS).context(
        crate::error::Qwen3AllocationSnafu {
            target: "rerank scalar score row",
        },
    )?;
    Ok(scores)
}

fn score_output_bytes() -> Result<u64> {
    let elements = u64::try_from(SCORE_OUTPUT_ELEMENTS).map_err(|_| {
        Qwen3RequirementsOverflowSnafu {
            target: "rerank scalar score elements",
        }
        .build()
    })?;
    let bytes = u64::try_from(std::mem::size_of::<f32>()).map_err(|_| {
        Qwen3RequirementsOverflowSnafu {
            target: "rerank scalar f32 bytes",
        }
        .build()
    })?;
    elements.checked_mul(bytes).ok_or_else(|| {
        Qwen3RequirementsOverflowSnafu {
            target: "rerank scalar score payload",
        }
        .build()
    })
}

fn multiply_bytes(bytes: u64, count: usize, target: &'static str) -> Result<u64> {
    let count =
        u64::try_from(count).map_err(|_| Qwen3RequirementsOverflowSnafu { target }.build())?;
    bytes
        .checked_mul(count)
        .ok_or_else(|| Qwen3RequirementsOverflowSnafu { target }.build())
}

fn validate_batch(batch: &RerankBatch, limit: usize) -> Result<()> {
    if batch.items.is_empty() {
        return EmptyBatchSnafu.fail();
    }
    if batch.items.len() > limit {
        return Qwen3BatchTooLargeSnafu {
            actual: batch.items.len(),
            limit,
        }
        .fail();
    }
    for (index, item) in batch.items.iter().enumerate() {
        if item.query.trim().is_empty() {
            return EmptyQuerySnafu { index }.fail();
        }
        if item.document.trim().is_empty() {
            return EmptyDocumentSnafu { index }.fail();
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "qwen3/tests.rs"]
mod tests;
