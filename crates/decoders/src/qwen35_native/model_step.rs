//! Action-free checked geometry for one native main-model token step.

use snafu::ResultExt;

use super::model_plan::DeviceModelPlan;
use crate::Result;
use crate::error::{
    ArithmeticOverflowSnafu, ExecutionPagedDecodePlanSnafu, NativeKernelSnafu,
    NativeSessionStateSnafu,
};

/// Complete action-free geometry admitted before a native model allocates or submits.
#[derive(Debug, Clone, Copy)]
pub(super) struct ModelTokenPlan {
    /// Exact selected serialized embedding row.
    pub(super) embedding: kernels::row_gemv::RowDecodePlan,
    /// Visible prefix length, committed only after successful model-wide publication.
    pub(super) next_position: usize,
    /// Full-attention decode geometry when this model owns native paged KV.
    pub(super) attention: Option<kernels::attention::NativePagedDecodePlan>,
}

impl ModelTokenPlan {
    /// Bind one token and committed position to an existing model descriptor.
    ///
    /// This performs no device allocation, host-to-device copy, stream
    /// submission, cache reservation, or mutable model-state transition.
    pub(super) fn from_model(plan: &DeviceModelPlan, position: usize, token: u32) -> Result<Self> {
        let next_position = position.checked_add(1).ok_or_else(|| {
            ArithmeticOverflowSnafu {
                context: "native model next position",
            }
            .build()
        })?;
        if next_position > plan.layout.max_context() {
            return NativeSessionStateSnafu {
                rule: "native model context must remain within its plan",
            }
            .fail();
        }
        let row = usize::try_from(token).map_err(|_| {
            ArithmeticOverflowSnafu {
                context: "native embedding token row",
            }
            .build()
        })?;
        let embedding = kernels::row_gemv::RowDecodePlan::try_from_shape(plan.embedding.shape, row)
            .context(NativeKernelSnafu)?;
        let attention = plan
            .kv
            .map(|kv| {
                let logical = kernels::PagedDecodePlan::try_from_dimensions(
                    next_position,
                    plan.layout.heads,
                    plan.layout.kv_heads,
                    plan.layout.key,
                )
                .context(ExecutionPagedDecodePlanSnafu)?;
                kernels::attention::NativePagedDecodePlan::try_from_paged_decode(
                    logical,
                    kv.layout().page_tokens(),
                    kv.layout().physical_pages(),
                )
                .context(ExecutionPagedDecodePlanSnafu)
            })
            .transpose()?;

        Ok(Self {
            embedding,
            next_position,
            attention,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::ModelTokenPlan;
    use crate::Qwen35Weights;
    use crate::qwen35::tests::{canonical_hybrid_fixture_with_context, verify_fixture};
    use crate::qwen35_native::model_plan::DeviceModelPlan;

    const CONTEXT: usize = 16;
    const PAGE_TOKENS: kernels::attention::NativePageTokens =
        kernels::attention::NativePageTokens::B8;

    fn model_plan() -> core::result::Result<DeviceModelPlan, String> {
        let artifact = verify_fixture(&canonical_hybrid_fixture_with_context(CONTEXT)?)?;
        let weights =
            Qwen35Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;
        DeviceModelPlan::from_weights(&weights, CONTEXT, PAGE_TOKENS)
            .map_err(|error| error.to_string())
    }

    #[test]
    fn model_token_plan_refuses_an_out_of_vocabulary_embedding_row()
    -> core::result::Result<(), String> {
        let plan = model_plan()?;
        let invalid = u32::try_from(plan.layout.vocabulary()).map_err(|error| error.to_string())?;

        assert!(ModelTokenPlan::from_model(&plan, 0, invalid).is_err());
        Ok(())
    }

    #[test]
    fn model_token_plan_keeps_last_admissible_context_action_free()
    -> core::result::Result<(), String> {
        let plan = model_plan()?;
        let token =
            ModelTokenPlan::from_model(&plan, CONTEXT - 1, 0).map_err(|error| error.to_string())?;

        assert_eq!(token.next_position, CONTEXT);
        assert!(ModelTokenPlan::from_model(&plan, CONTEXT, 0).is_err());
        Ok(())
    }

    #[test]
    fn model_token_plan_refuses_position_overflow_before_geometry()
    -> core::result::Result<(), String> {
        let plan = model_plan()?;

        assert!(ModelTokenPlan::from_model(&plan, usize::MAX, 0).is_err());
        Ok(())
    }

    #[test]
    fn model_token_plan_derives_b8_boundary_decode_geometry() -> core::result::Result<(), String> {
        let plan = model_plan()?;
        let at_eight =
            ModelTokenPlan::from_model(&plan, 7, 0).map_err(|error| error.to_string())?;
        let at_nine = ModelTokenPlan::from_model(&plan, 8, 0).map_err(|error| error.to_string())?;
        let eight_attention = at_eight
            .attention
            .ok_or("canonical model needs paged attention")?;
        let nine_attention = at_nine
            .attention
            .ok_or("canonical model needs paged attention")?;

        assert_eq!(eight_attention.logical().visible_tokens(), 8);
        assert_eq!(eight_attention.page_table_entries(), 1);
        assert_eq!(nine_attention.logical().visible_tokens(), 9);
        assert_eq!(nine_attention.page_table_entries(), 2);
        Ok(())
    }
}
