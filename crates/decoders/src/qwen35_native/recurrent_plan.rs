//! Checked native recurrent-block allocation and verified-weight descriptors.

use snafu::ResultExt;

use super::plan::{F32Parameter, ProjectionWeight, elements_bytes, f32_parameter, projection, sum};
use crate::Qwen35Weights;
use crate::Result;
use crate::error::{
    ArithmeticOverflowSnafu, NativeKernelSnafu, NativeSessionStateSnafu, RecurrentConvolutionSnafu,
    RecurrentGdnSnafu,
};
use crate::qwen35::recurrent_layernorm_rms_epsilon;
use crate::qwen35_recurrent::{ExecutionLayout, RecurrentTensorRole, recurrent_tensor_name};
use kernels::PackedPrefillPlan;

/// One verified matrix descriptor used by a native recurrent block.
#[derive(Debug)]
pub(super) struct RecurrentProjectionWeights {
    pub(super) qkv: ProjectionWeight,
    pub(super) gate: ProjectionWeight,
    pub(super) alpha: ProjectionWeight,
    pub(super) beta: ProjectionWeight,
    pub(super) output: ProjectionWeight,
}

/// One verified F32 descriptor used by a native recurrent block.
#[derive(Debug)]
pub(super) struct RecurrentF32Parameters {
    pub(super) attention_norm: F32Parameter,
    pub(super) a: F32Parameter,
    pub(super) dt: F32Parameter,
    pub(super) convolution: F32Parameter,
    pub(super) output_norm: F32Parameter,
}

/// Checked native recurrent operation geometry and named workspace extents.
#[derive(Debug, Clone, Copy)]
pub(super) struct RecurrentWorkspacePlan {
    pub(super) input_norm: kernels::decoder_ops::RmsNormF32Plan,
    pub(super) convolution_silu: kernels::decoder_ops::ElementwiseF32Plan,
    pub(super) qk_l2: kernels::decoder_ops::RecurrentQkL2F32Plan,
    pub(super) value_layout: kernels::decoder_ops::RecurrentValueLayoutF32Plan,
    pub(super) scalars: kernels::decoder_ops::RecurrentScalarsF32Plan,
    pub(super) output_norm: kernels::decoder_ops::RmsNormF32Plan,
    pub(super) output_silu_product: kernels::decoder_ops::ElementwiseF32Plan,
    pub(super) normalized_hidden: usize,
    pub(super) qkv: usize,
    pub(super) z: usize,
    pub(super) alpha: usize,
    pub(super) beta_projection: usize,
    pub(super) raw_convolution: usize,
    pub(super) activated_convolution: usize,
    pub(super) tiled_query: usize,
    pub(super) tiled_key: usize,
    pub(super) value_head_major: usize,
    pub(super) beta: usize,
    pub(super) log_decay: usize,
    pub(super) recurrence_output: usize,
    pub(super) token_major_recurrence_output: usize,
    pub(super) normalized_output: usize,
    pub(super) gated_output: usize,
    pub(super) projected_attention: usize,
    elements: usize,
}

/// Existing checked kernel plans that jointly define one recurrent workspace.
struct RecurrentOperationPlans {
    input_norm: kernels::decoder_ops::RmsNormF32Plan,
    convolution: kernels::CausalConvAllocationPlan,
    recurrence: kernels::MultiHeadRecurrentAllocationPlan,
    convolution_silu: kernels::decoder_ops::ElementwiseF32Plan,
    qk_l2: kernels::decoder_ops::RecurrentQkL2F32Plan,
    value_layout: kernels::decoder_ops::RecurrentValueLayoutF32Plan,
    scalars: kernels::decoder_ops::RecurrentScalarsF32Plan,
    output_norm: kernels::decoder_ops::RmsNormF32Plan,
    output_silu_product: kernels::decoder_ops::ElementwiseF32Plan,
}

impl RecurrentWorkspacePlan {
    /// Return the exact simultaneously allocated recurrent workspace extent.
    #[must_use]
    pub(super) const fn elements(self) -> usize {
        self.elements
    }
}

/// Artifact-bound native recurrent-block plan without device ownership.
///
/// The plan derives its recurrent relations from the CPU owner's checked
/// `ExecutionLayout`; it does not reinterpret CPU logical-allocation reports as
/// a device demand. A future model resource owner supplies active and staged
/// state buffers separately and owns their atomic publication.
#[derive(Debug)]
pub(super) struct DeviceRecurrentPlan {
    pub(super) layout: ExecutionLayout,
    pub(super) matrices: RecurrentProjectionWeights,
    pub(super) parameters: RecurrentF32Parameters,
    pub(super) workspace: RecurrentWorkspacePlan,
    pub(super) convolution: kernels::CausalConvAllocationPlan,
    pub(super) recurrence: kernels::MultiHeadRecurrentAllocationPlan,
    weight_bytes: usize,
}

/// Checked active recurrent operation geometry with no weight ownership.
#[derive(Debug, Clone, Copy)]
pub(super) struct ActiveRecurrentPlan {
    pub(super) layout: ExecutionLayout,
    pub(super) workspace: RecurrentWorkspacePlan,
    pub(super) convolution: kernels::CausalConvAllocationPlan,
    pub(super) recurrence: kernels::MultiHeadRecurrentAllocationPlan,
}

impl DeviceRecurrentPlan {
    /// Bind one single-sequence packed recurrent chunk to native descriptors.
    ///
    /// The packed descriptor remains borrowed at this boundary: its existing
    /// context admission is consumed as geometry, never copied into model
    /// position, grant, or publication state.
    ///
    pub(super) fn from_packed_prefill(
        weights: &Qwen35Weights,
        block: usize,
        packed: &PackedPrefillPlan,
    ) -> Result<Self> {
        if packed.sequence_count() != 1 {
            return NativeSessionStateSnafu {
                rule: "native recurrent packed chunk requires exactly one sequence",
            }
            .fail();
        }
        Self::from_token_count(weights, block, packed.total_tokens())
    }

    /// # Errors
    ///
    /// Returns an error when `token_count` cannot form checked convolution or
    /// recurrence geometry under this retained artifact binding.
    pub(super) fn active(&self, token_count: usize) -> Result<ActiveRecurrentPlan> {
        let convolution = kernels::CausalConvAllocationPlan::try_from_dimensions(
            token_count,
            self.layout.convolution_width(),
            self.layout.convolution_kernel(),
        )
        .context(RecurrentConvolutionSnafu)?;
        let recurrence = kernels::MultiHeadRecurrentAllocationPlan::try_from_dimensions(
            token_count,
            self.layout.value_head_count(),
            self.layout.value_head_count(),
            self.layout.key_dim(),
            self.layout.value_dim(),
        )
        .context(RecurrentGdnSnafu)?;
        Ok(ActiveRecurrentPlan {
            layout: self.layout,
            workspace: RecurrentWorkspacePlan::from_layout(
                self.layout,
                token_count,
                convolution,
                recurrence,
            )?,
            convolution,
            recurrence,
        })
    }

    fn from_token_count(weights: &Qwen35Weights, block: usize, token_count: usize) -> Result<Self> {
        let block_index = u64::try_from(block).map_err(|_| {
            ArithmeticOverflowSnafu {
                context: "native recurrent block index",
            }
            .build()
        })?;
        let epsilon = recurrent_layernorm_rms_epsilon(weights.payload().observation().metadata())?;
        let layout = ExecutionLayout::try_from_profile(weights.recurrent_layout(), epsilon)?;
        layout.validate_recurrent_block(block_index)?;

        let convolution = kernels::CausalConvAllocationPlan::try_from_dimensions(
            token_count,
            layout.convolution_width(),
            layout.convolution_kernel(),
        )
        .context(RecurrentConvolutionSnafu)?;
        let recurrence = kernels::MultiHeadRecurrentAllocationPlan::try_from_dimensions(
            token_count,
            layout.value_head_count(),
            layout.value_head_count(),
            layout.key_dim(),
            layout.value_dim(),
        )
        .context(RecurrentGdnSnafu)?;
        let workspace =
            RecurrentWorkspacePlan::from_layout(layout, token_count, convolution, recurrence)?;
        let matrices = RecurrentProjectionWeights::from_weights(weights, block_index)?;
        let parameters =
            RecurrentF32Parameters::from_weights(weights, block_index, layout, convolution)?;
        let weight_bytes = sum(
            &[
                matrices.qkv.serialized_bytes,
                matrices.gate.serialized_bytes,
                matrices.alpha.serialized_bytes,
                matrices.beta.serialized_bytes,
                matrices.output.serialized_bytes,
                elements_bytes(
                    layout.hidden(),
                    core::mem::size_of::<f32>(),
                    "native recurrent attention norm bytes",
                )?,
                elements_bytes(
                    layout.value_head_count(),
                    core::mem::size_of::<f32>(),
                    "native recurrent A bytes",
                )?,
                elements_bytes(
                    layout.value_head_count(),
                    core::mem::size_of::<f32>(),
                    "native recurrent dt bytes",
                )?,
                elements_bytes(
                    convolution.weight_elements(),
                    core::mem::size_of::<f32>(),
                    "native recurrent convolution bytes",
                )?,
                elements_bytes(
                    layout.value_dim(),
                    core::mem::size_of::<f32>(),
                    "native recurrent output norm bytes",
                )?,
            ],
            "native recurrent weight bytes",
        )?;

        Ok(Self {
            layout,
            matrices,
            parameters,
            workspace,
            convolution,
            recurrence,
            weight_bytes,
        })
    }

    /// Return the checked immutable-weight upload extent for this block.
    #[must_use]
    pub(super) const fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    /// Return the checked active-or-staged convolution-history extent.
    #[must_use]
    pub(super) const fn convolution_history_elements(&self) -> usize {
        self.convolution.history_elements()
    }

    /// Return the checked active-or-staged GDN-state extent.
    #[must_use]
    pub(super) const fn recurrent_state_elements(&self) -> usize {
        self.recurrence.state_elements()
    }
}

impl RecurrentProjectionWeights {
    fn from_weights(weights: &Qwen35Weights, block: u64) -> Result<Self> {
        Ok(Self {
            qkv: projection(
                weights,
                recurrent_tensor_name(block, RecurrentTensorRole::AttentionQkv),
            )?,
            gate: projection(
                weights,
                recurrent_tensor_name(block, RecurrentTensorRole::AttentionGate),
            )?,
            alpha: projection(
                weights,
                recurrent_tensor_name(block, RecurrentTensorRole::SsmAlpha),
            )?,
            beta: projection(
                weights,
                recurrent_tensor_name(block, RecurrentTensorRole::SsmBeta),
            )?,
            output: projection(
                weights,
                recurrent_tensor_name(block, RecurrentTensorRole::SsmOutput),
            )?,
        })
    }
}

impl RecurrentF32Parameters {
    fn from_weights(
        weights: &Qwen35Weights,
        block: u64,
        layout: ExecutionLayout,
        convolution: kernels::CausalConvAllocationPlan,
    ) -> Result<Self> {
        Ok(Self {
            attention_norm: f32_parameter(
                weights,
                recurrent_tensor_name(block, RecurrentTensorRole::AttentionNorm),
                vec![layout.hidden_u64()],
                layout.hidden(),
            )?,
            a: f32_parameter(
                weights,
                recurrent_tensor_name(block, RecurrentTensorRole::SsmA),
                vec![layout.value_head_count_u64()],
                layout.value_head_count(),
            )?,
            dt: f32_parameter(
                weights,
                recurrent_tensor_name(block, RecurrentTensorRole::SsmDt),
                vec![layout.value_head_count_u64()],
                layout.value_head_count(),
            )?,
            convolution: f32_parameter(
                weights,
                recurrent_tensor_name(block, RecurrentTensorRole::SsmConvolution),
                vec![
                    layout.convolution_kernel_u64(),
                    layout.convolution_width_u64(),
                ],
                convolution.weight_elements(),
            )?,
            output_norm: f32_parameter(
                weights,
                recurrent_tensor_name(block, RecurrentTensorRole::SsmNorm),
                vec![layout.value_dim_u64()],
                layout.value_dim(),
            )?,
        })
    }
}

impl RecurrentWorkspacePlan {
    fn from_layout(
        layout: ExecutionLayout,
        token_count: usize,
        convolution: kernels::CausalConvAllocationPlan,
        recurrence: kernels::MultiHeadRecurrentAllocationPlan,
    ) -> Result<Self> {
        let input_norm = kernels::decoder_ops::RmsNormF32Plan::try_from_dimensions(
            token_count,
            layout.hidden(),
            layout.epsilon(),
        )
        .context(NativeKernelSnafu)?;
        let convolution_silu = kernels::decoder_ops::ElementwiseF32Plan::try_from_elements(
            convolution.output_elements(),
        )
        .context(NativeKernelSnafu)?;
        let qk_l2 = kernels::decoder_ops::RecurrentQkL2F32Plan::try_from_token_rows(
            token_count,
            layout.convolution_width(),
            layout.key_head_count(),
            layout.value_head_count(),
            layout.key_dim(),
            layout.epsilon(),
        )
        .context(NativeKernelSnafu)?;
        let value_layout = kernels::decoder_ops::RecurrentValueLayoutF32Plan::try_from_qk_plan(
            qk_l2,
            layout.value_dim(),
        )
        .context(NativeKernelSnafu)?;
        let scalars = kernels::decoder_ops::RecurrentScalarsF32Plan::try_from_token_rows(
            token_count,
            layout.value_head_count(),
        )
        .context(NativeKernelSnafu)?;
        let output_norm = kernels::decoder_ops::RmsNormF32Plan::try_from_dimensions(
            token_count
                .checked_mul(layout.value_head_count())
                .ok_or_else(|| {
                    ArithmeticOverflowSnafu {
                        context: "native recurrent output-normalization rows",
                    }
                    .build()
                })?,
            layout.value_dim(),
            layout.epsilon(),
        )
        .context(NativeKernelSnafu)?;
        let output_silu_product =
            kernels::decoder_ops::ElementwiseF32Plan::try_from_elements(output_norm.elements())
                .context(NativeKernelSnafu)?;

        Self::from_operations(&RecurrentOperationPlans {
            input_norm,
            convolution,
            recurrence,
            convolution_silu,
            qk_l2,
            value_layout,
            scalars,
            output_norm,
            output_silu_product,
        })
    }

    fn from_operations(operations: &RecurrentOperationPlans) -> Result<Self> {
        validate_operations(operations)?;
        let normalized_hidden = operations.input_norm.elements();
        let qkv = operations.convolution.output_elements();
        let z = operations.recurrence.output_elements();
        let alpha = operations.scalars.elements();
        let beta_projection = operations.scalars.elements();
        let raw_convolution = operations.convolution.output_elements();
        let activated_convolution = operations.convolution_silu.elements();
        let tiled_query = operations.qk_l2.output_elements();
        let tiled_key = operations.qk_l2.output_elements();
        let value_head_major = operations.recurrence.output_elements();
        let beta = operations.scalars.elements();
        let log_decay = operations.scalars.elements();
        let recurrence_output = operations.recurrence.output_elements();
        let token_major_recurrence_output = operations.recurrence.output_elements();
        let normalized_output = operations.output_norm.elements();
        let gated_output = operations.output_silu_product.elements();
        let projected_attention = operations.input_norm.elements();
        let elements = sum(
            &[
                normalized_hidden,
                qkv,
                z,
                alpha,
                beta_projection,
                raw_convolution,
                activated_convolution,
                tiled_query,
                tiled_key,
                value_head_major,
                beta,
                log_decay,
                recurrence_output,
                token_major_recurrence_output,
                normalized_output,
                gated_output,
                projected_attention,
            ],
            "native recurrent workspace elements",
        )?;

        Ok(Self {
            input_norm: operations.input_norm,
            convolution_silu: operations.convolution_silu,
            qk_l2: operations.qk_l2,
            value_layout: operations.value_layout,
            scalars: operations.scalars,
            output_norm: operations.output_norm,
            output_silu_product: operations.output_silu_product,
            normalized_hidden,
            qkv,
            z,
            alpha,
            beta_projection,
            raw_convolution,
            activated_convolution,
            tiled_query,
            tiled_key,
            value_head_major,
            beta,
            log_decay,
            recurrence_output,
            token_major_recurrence_output,
            normalized_output,
            gated_output,
            projected_attention,
            elements,
        })
    }
}

fn validate_operations(operations: &RecurrentOperationPlans) -> Result<()> {
    if operations.qk_l2.output_elements() != operations.recurrence.query_and_key_elements()
        || operations.value_layout.elements() != operations.recurrence.output_elements()
        || operations.scalars.elements() != operations.recurrence.scalar_elements()
        || operations.output_norm.elements() != operations.recurrence.output_elements()
        || operations.qk_l2.convolved_elements() != operations.convolution_silu.elements()
        || operations.value_layout.convolved_elements() != operations.convolution_silu.elements()
    {
        return NativeSessionStateSnafu {
            rule: "native recurrent kernel plans must share the CPU-derived equal-head layout",
        }
        .fail();
    }
    if operations.value_layout.value_offset() != operations.qk_l2.qk_prefix_elements() {
        return NativeSessionStateSnafu {
            rule: "native recurrent V gather must begin after every row's Q/K prefix",
        }
        .fail();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{DeviceRecurrentPlan, RecurrentOperationPlans, RecurrentWorkspacePlan, sum};
    use crate::Qwen35Weights;
    use crate::qwen35::tests::{canonical_hybrid_fixture, verify_fixture};
    use kernels::PackedPrefillPlan;

    #[test]
    fn verified_recurrent_plan_reuses_the_cpu_layout_and_tensor_roles()
    -> core::result::Result<(), String> {
        let artifact = verify_fixture(&canonical_hybrid_fixture()?)?;
        let weights =
            Qwen35Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;
        let packed = PackedPrefillPlan::new(&[1], &[0], 1).map_err(|error| error.to_string())?;
        let plan = DeviceRecurrentPlan::from_packed_prefill(&weights, 0, &packed)
            .map_err(|error| error.to_string())?;

        assert_eq!(plan.workspace.qkv, plan.convolution.output_elements());
        assert_eq!(
            plan.workspace.value_layout.elements(),
            plan.recurrence.output_elements()
        );
        assert_eq!(
            plan.workspace.value_layout.value_offset(),
            plan.workspace.qk_l2.qk_prefix_elements(),
            "V must begin after each activated-convolution row's Q/K prefix"
        );
        assert_eq!(plan.matrices.qkv.name, "blk.0.attn_qkv.weight");
        assert_eq!(plan.matrices.output.name, "blk.0.ssm_out.weight");
        assert_eq!(plan.parameters.convolution.name, "blk.0.ssm_conv1d.weight");
        assert!(plan.weight_bytes() > 0);
        assert!(plan.workspace.elements() > 0);
        Ok(())
    }

    #[test]
    fn recurrent_plan_refuses_a_full_attention_or_missing_main_block()
    -> core::result::Result<(), String> {
        let artifact = verify_fixture(&canonical_hybrid_fixture()?)?;
        let weights =
            Qwen35Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;

        let packed = PackedPrefillPlan::new(&[1], &[0], 1).map_err(|error| error.to_string())?;
        assert!(
            DeviceRecurrentPlan::from_packed_prefill(&weights, 3, &packed).is_err(),
            "a cadence full-attention block cannot use recurrent native geometry"
        );
        assert!(
            DeviceRecurrentPlan::from_packed_prefill(&weights, 4, &packed).is_err(),
            "a block outside the main-block domain cannot use recurrent native geometry"
        );
        Ok(())
    }

    #[test]
    fn recurrent_workspace_sum_refuses_overflow() {
        assert!(
            sum(&[usize::MAX, 1], "native recurrent workspace elements").is_err(),
            "native workspace demand must reject overflow before device allocation"
        );
    }

    #[test]
    fn packed_native_recurrent_plan_derives_every_token_bearing_owner()
    -> core::result::Result<(), String> {
        let artifact = verify_fixture(&canonical_hybrid_fixture()?)?;
        let weights =
            Qwen35Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;

        for token_count in [1_usize, 2, 3] {
            let packed = PackedPrefillPlan::new(&[token_count], &[0], token_count)
                .map_err(|error| error.to_string())?;
            let plan = DeviceRecurrentPlan::from_packed_prefill(&weights, 0, &packed)
                .map_err(|error| error.to_string())?;
            let workspace = plan.workspace;

            assert_eq!(plan.convolution.token_count(), token_count);
            assert_eq!(plan.recurrence.token_count(), token_count);
            assert_eq!(
                workspace.normalized_hidden,
                token_count * plan.layout.hidden()
            );
            assert_eq!(workspace.qkv, plan.convolution.output_elements());
            assert_eq!(
                workspace.raw_convolution,
                plan.convolution.output_elements()
            );
            assert_eq!(
                workspace.activated_convolution,
                plan.convolution.output_elements()
            );
            assert_eq!(
                workspace.value_head_major,
                plan.recurrence.output_elements()
            );
            assert_eq!(
                workspace.recurrence_output,
                plan.recurrence.output_elements()
            );
            assert_eq!(
                workspace.token_major_recurrence_output,
                plan.recurrence.output_elements()
            );
            assert_eq!(workspace.beta, plan.recurrence.scalar_elements());
            assert_eq!(workspace.log_decay, plan.recurrence.scalar_elements());
            assert_eq!(
                workspace.projected_attention,
                token_count * plan.layout.hidden()
            );
            assert_eq!(
                workspace.elements(),
                sum(
                    &[
                        workspace.normalized_hidden,
                        workspace.qkv,
                        workspace.z,
                        workspace.alpha,
                        workspace.beta_projection,
                        workspace.raw_convolution,
                        workspace.activated_convolution,
                        workspace.tiled_query,
                        workspace.tiled_key,
                        workspace.value_head_major,
                        workspace.beta,
                        workspace.log_decay,
                        workspace.recurrence_output,
                        workspace.token_major_recurrence_output,
                        workspace.normalized_output,
                        workspace.gated_output,
                        workspace.projected_attention,
                    ],
                    "native recurrent test workspace elements",
                )
                .map_err(|error| error.to_string())?,
                "the demand must enumerate every concurrent workspace owner"
            );
        }
        Ok(())
    }

    #[test]
    fn packed_native_recurrent_plan_refuses_multi_sequence_before_device_ownership()
    -> core::result::Result<(), String> {
        let artifact = verify_fixture(&canonical_hybrid_fixture()?)?;
        let weights =
            Qwen35Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;
        let packed =
            PackedPrefillPlan::new(&[1, 1], &[0, 0], 1).map_err(|error| error.to_string())?;

        assert!(
            DeviceRecurrentPlan::from_packed_prefill(&weights, 0, &packed).is_err(),
            "the B=1 native owner must refuse before any device allocation or submission"
        );
        Ok(())
    }

    #[test]
    fn packed_context_refusal_precedes_native_chunk_planning() {
        assert!(
            PackedPrefillPlan::new(&[2], &[1], 2).is_err(),
            "the borrowed packed descriptor remains the sole context authority"
        );
    }

    #[test]
    fn recurrent_workspace_keeps_unequal_key_and_value_widths_distinct()
    -> core::result::Result<(), String> {
        let convolution = kernels::CausalConvAllocationPlan::try_from_dimensions(1, 14, 1)
            .map_err(|error| error.to_string())?;
        let recurrence =
            kernels::MultiHeadRecurrentAllocationPlan::try_from_dimensions(1, 2, 2, 3, 4)
                .map_err(|error| error.to_string())?;
        let input_norm = kernels::decoder_ops::RmsNormF32Plan::try_from_dimensions(1, 7, 0.5)
            .map_err(|error| error.to_string())?;
        let convolution_silu = kernels::decoder_ops::ElementwiseF32Plan::try_from_elements(14)
            .map_err(|error| error.to_string())?;
        let qk_l2 =
            kernels::decoder_ops::RecurrentQkL2F32Plan::try_from_dimensions(14, 1, 2, 3, 0.5)
                .map_err(|error| error.to_string())?;
        let value_layout =
            kernels::decoder_ops::RecurrentValueLayoutF32Plan::try_from_qk_plan(qk_l2, 4)
                .map_err(|error| error.to_string())?;
        let scalars = kernels::decoder_ops::RecurrentScalarsF32Plan::try_from_value_heads(2)
            .map_err(|error| error.to_string())?;
        let output_norm = kernels::decoder_ops::RmsNormF32Plan::try_from_dimensions(2, 4, 0.5)
            .map_err(|error| error.to_string())?;
        let output_silu_product =
            kernels::decoder_ops::ElementwiseF32Plan::try_from_elements(output_norm.elements())
                .map_err(|error| error.to_string())?;

        let workspace = RecurrentWorkspacePlan::from_operations(&RecurrentOperationPlans {
            input_norm,
            convolution,
            recurrence,
            convolution_silu,
            qk_l2,
            value_layout,
            scalars,
            output_norm,
            output_silu_product,
        })
        .map_err(|error| error.to_string())?;

        assert_eq!(workspace.tiled_query, 6);
        assert_eq!(workspace.recurrence_output, 8);
        assert_eq!(workspace.value_layout.elements(), 8);
        Ok(())
    }
}
