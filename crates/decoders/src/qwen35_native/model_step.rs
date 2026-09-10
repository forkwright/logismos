//! Action-free checked geometry for one native main-model chunk.

use kernels::PackedPrefillPlan;
use snafu::ResultExt;

use super::model_plan::DeviceModelPlan;
use crate::Result;
use crate::error::{
    ArithmeticOverflowSnafu, ExecutionAllocationSnafu, ExecutionPagedPrefillPlanSnafu,
    NativeKernelSnafu, NativeSessionStateSnafu,
};

/// Complete action-free geometry admitted before a native model allocates,
/// reserves K/V, or submits a chunk.
#[derive(Debug)]
pub(super) struct ModelChunkPlan {
    /// The sole operation-local authority for this B=1 chunk's row count and offset.
    pub(super) packed: PackedPrefillPlan,
    /// Exact selected serialized embedding rows in chronological chunk order.
    pub(super) embeddings: Vec<kernels::row_gemv::RowDecodePlan>,
    /// Committed absolute position supplied by the packed sequence descriptor.
    pub(super) position: usize,
    /// Position published only with every other transaction result.
    pub(super) next_position: usize,
    /// Full-attention chunk geometry when this model owns native paged K/V.
    pub(super) attention: Option<kernels::attention::NativePagedPrefillPlan>,
}

impl ModelChunkPlan {
    /// Bind a nonempty one-sequence token chunk to this session's committed position.
    ///
    /// This performs no device allocation, host-to-device copy, cache reservation,
    /// stream submission, or mutable model-state transition. It validates every
    /// token row before any caller can begin those effects.
    pub(super) fn from_model(
        plan: &DeviceModelPlan,
        position: usize,
        tokens: &[u32],
    ) -> Result<Self> {
        let packed =
            PackedPrefillPlan::new(&[tokens.len()], &[position], plan.layout.max_context())
                .context(NativeKernelSnafu)?;
        Self::from_packed_sequence(plan, tokens, &packed, 0)
    }

    pub(super) fn from_packed_sequence(
        plan: &DeviceModelPlan,
        tokens: &[u32],
        packed: &PackedPrefillPlan,
        sequence: usize,
    ) -> Result<Self> {
        let token_count = packed.sequence_length(sequence).ok_or_else(|| {
            NativeSessionStateSnafu {
                rule: "native packed model sequence must exist",
            }
            .build()
        })?;
        let position = packed.committed_offset(sequence).ok_or_else(|| {
            NativeSessionStateSnafu {
                rule: "native packed model sequence must retain its committed position",
            }
            .build()
        })?;
        if token_count != tokens.len() {
            return NativeSessionStateSnafu {
                rule: "native packed model sequence length must match token ids",
            }
            .fail();
        }
        if token_count > plan.max_chunk_tokens {
            return NativeSessionStateSnafu {
                rule: "native model chunk must remain within its admitted capacity",
            }
            .fail();
        }
        let next_position = position.checked_add(token_count).ok_or_else(|| {
            ArithmeticOverflowSnafu {
                context: "native model chunk next position",
            }
            .build()
        })?;
        if next_position > plan.layout.max_context() {
            return NativeSessionStateSnafu {
                rule: "native packed model sequence must fit its exact session context",
            }
            .fail();
        }
        let mut embeddings = Vec::new();
        embeddings
            .try_reserve_exact(token_count)
            .context(ExecutionAllocationSnafu {
                target: "native model chunk embedding rows",
                length: token_count,
            })?;
        for &token in tokens {
            let row = usize::try_from(token).map_err(|_| {
                ArithmeticOverflowSnafu {
                    context: "native embedding token row",
                }
                .build()
            })?;
            embeddings.push(
                kernels::row_gemv::RowDecodePlan::try_from_shape(plan.embedding.shape, row)
                    .context(NativeKernelSnafu)?,
            );
        }
        let native_packed =
            PackedPrefillPlan::new(&[token_count], &[position], plan.layout.max_context())
                .context(NativeKernelSnafu)?;
        let attention = plan
            .kv
            .map(|kv| {
                let logical = kernels::attention::PagedPrefillPlan::try_from_packed_prefill(
                    &native_packed,
                    plan.layout.heads,
                    plan.layout.kv_heads,
                    plan.layout.key,
                )
                .context(ExecutionPagedPrefillPlanSnafu)?;
                kernels::attention::NativePagedPrefillPlan::try_from_paged_prefill(
                    logical,
                    kv.layout().page_tokens(),
                    kv.layout().physical_pages(),
                )
                .context(ExecutionPagedPrefillPlanSnafu)
            })
            .transpose()?;

        Ok(Self {
            packed: native_packed,
            embeddings,
            position,
            next_position,
            attention,
        })
    }

    #[must_use]
    pub(super) const fn token_count(&self) -> usize {
        self.packed.total_tokens()
    }
}

#[cfg(test)]
mod tests {
    use super::ModelChunkPlan;
    use crate::Qwen35Weights;
    use crate::qwen35::tests::{canonical_hybrid_fixture_with_context, verify_fixture};
    use crate::qwen35_native::model_plan::DeviceModelPlan;

    const CONTEXT: usize = 16;
    const CAPACITY: usize = 3;
    const PAGE_TOKENS: kernels::attention::NativePageTokens =
        kernels::attention::NativePageTokens::B8;

    fn model_plan() -> core::result::Result<DeviceModelPlan, String> {
        let artifact = verify_fixture(&canonical_hybrid_fixture_with_context(CONTEXT)?)?;
        let weights =
            Qwen35Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;
        DeviceModelPlan::from_weights_prefill(&weights, CONTEXT, CAPACITY, PAGE_TOKENS)
            .map_err(|error| error.to_string())
    }

    #[test]
    fn model_chunk_plan_refuses_any_out_of_vocabulary_embedding_row_before_effects()
    -> core::result::Result<(), String> {
        let plan = model_plan()?;
        let invalid = u32::try_from(plan.layout.vocabulary()).map_err(|error| error.to_string())?;

        assert!(ModelChunkPlan::from_model(&plan, 0, &[0, 1, invalid]).is_err());
        let retry =
            ModelChunkPlan::from_model(&plan, 0, &[0, 1]).map_err(|error| error.to_string())?;
        assert_eq!(
            retry.packed.committed_offset(0),
            Some(0),
            "an invalid final token must not advance the action-free chunk position"
        );
        Ok(())
    }

    #[test]
    fn model_chunk_plan_refuses_empty_over_capacity_and_context_overflow_action_free()
    -> core::result::Result<(), String> {
        let plan = model_plan()?;

        assert!(ModelChunkPlan::from_model(&plan, 0, &[]).is_err());
        assert!(ModelChunkPlan::from_model(&plan, 0, &[0; CAPACITY + 1]).is_err());
        assert!(ModelChunkPlan::from_model(&plan, CONTEXT - 1, &[0, 0]).is_err());
        Ok(())
    }

    #[test]
    fn model_chunk_plan_keeps_one_position_and_derives_page_crossing_geometry()
    -> core::result::Result<(), String> {
        let plan = model_plan()?;
        let chunk =
            ModelChunkPlan::from_model(&plan, 7, &[0, 1, 2]).map_err(|error| error.to_string())?;
        let attention = chunk
            .attention
            .ok_or("canonical model needs paged attention")?;

        assert_eq!(chunk.packed.sequence_count(), 1);
        assert_eq!(chunk.packed.committed_offset(0), Some(7));
        assert_eq!(chunk.next_position, 10);
        assert_eq!(chunk.embeddings.len(), 3);
        assert_eq!(attention.tokens(), 3);
        assert_eq!(attention.visible_tokens(), 10);
        Ok(())
    }
}
