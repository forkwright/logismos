//! Bounded native CPU Qwen3 rank-head execution over a verified GGUF payload.

use loader::gguf::{MetaValue, MetaValueType, VerifiedArtifact};

use crate::Result;
use crate::error::{Qwen3ExecutionSnafu, Qwen3MetadataSnafu};
use crate::matrix::CheckedMatrix;
use crate::qwen3::{Qwen3BodyWeights, Qwen3Profile, RANK_HEAD, RANK_HEAD_ROLE};

const CLASSIFIER_OUTPUT_LABELS: &str = "qwen3.classifier.output_labels";
const RANK_LABELS: [&str; 2] = ["yes", "no"];

/// One verified Qwen3 rank payload with its checked causal body and two-row head.
#[derive(Debug)]
pub struct Qwen3RankWeights<'artifact> {
    body: Qwen3BodyWeights<'artifact>,
}

impl<'artifact> Qwen3RankWeights<'artifact> {
    /// Bind a verified GGUF payload to the bounded Qwen3 rank profile.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] when the causal body, rank metadata, or the
    /// exact two-label classifier head does not satisfy the bounded profile.
    pub fn try_from_verified(payload: &'artifact VerifiedArtifact) -> Result<Self> {
        require_rank_labels(payload)?;
        Ok(Self {
            body: Qwen3BodyWeights::try_from_verified(
                payload,
                Qwen3Profile::Rank,
                &[RANK_HEAD_ROLE],
            )?,
        })
    }

    /// Return the artifact-derived hidden-vector width.
    #[must_use]
    pub const fn hidden_width(&self) -> usize {
        self.body.hidden_width()
    }

    /// Return the artifact-derived maximum context length.
    #[must_use]
    pub const fn max_context(&self) -> usize {
        self.body.max_context()
    }

    /// Create one stateless bounded CPU rank executor.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] when `max_context` is zero or exceeds the
    /// verified artifact's declared context length.
    pub fn execution(&self, max_context: usize) -> Result<Qwen3RankExecution<'_, 'artifact>> {
        Ok(Qwen3RankExecution {
            weights: self,
            body_execution: self.body.execution(max_context)?,
        })
    }
}

/// Stateless Qwen3 causal rank execution with an explicit caller context bound.
#[derive(Debug)]
pub struct Qwen3RankExecution<'weights, 'artifact> {
    weights: &'weights Qwen3RankWeights<'artifact>,
    body_execution: crate::qwen3::Qwen3Execution<'weights, 'artifact>,
}

impl Qwen3RankExecution<'_, '_> {
    /// Return raw final-token classifier logits in the verified `[yes, no]` order.
    ///
    /// The signed relevance-score reduction belongs to the reranker adapter;
    /// this decoder boundary only executes the admitted two-row head.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] without exposing partial hidden state or a
    /// partial classifier result when body execution or projection fails.
    pub fn last_logits(&self, token_ids: &[u32]) -> Result<[f32; 2]> {
        let hidden = self.body_execution.last_hidden(token_ids)?;
        let head = CheckedMatrix::from_payload(self.weights.body.payload(), RANK_HEAD)?;
        let logits: [f32; 2] = head
            .project(&hidden)?
            .try_into()
            .map_err(|values: Vec<f32>| {
                Qwen3ExecutionSnafu {
                    requested: values.len(),
                    rule: "the admitted rank classifier head must project exactly two logits",
                }
                .build()
            })?;
        for (index, logit) in logits.iter().enumerate() {
            if !logit.is_finite() {
                return crate::error::Qwen3ArithmeticSnafu {
                    stage: "rank classifier projection",
                    index,
                }
                .fail();
            }
        }
        Ok(logits)
    }
}

fn require_rank_labels(payload: &VerifiedArtifact) -> Result<()> {
    let metadata = payload.observation().metadata();
    let Some(MetaValue::Array(labels)) = metadata.get(CLASSIFIER_OUTPUT_LABELS) else {
        return Qwen3MetadataSnafu {
            key: CLASSIFIER_OUTPUT_LABELS,
            rule: "must be the exact string array [yes, no]",
        }
        .fail();
    };
    if labels.element_type() != MetaValueType::String
        || labels.values().len() != RANK_LABELS.len()
        || !labels
            .values()
            .iter()
            .zip(RANK_LABELS)
            .all(|(label, expected)| matches!(label, MetaValue::String(value) if value == expected))
    {
        return Qwen3MetadataSnafu {
            key: CLASSIFIER_OUTPUT_LABELS,
            rule: "must be the exact string array [yes, no] in source label order",
        }
        .fail();
    }
    Ok(())
}
