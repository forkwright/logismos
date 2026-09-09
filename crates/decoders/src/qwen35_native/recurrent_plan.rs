//! Checked native recurrent-block allocation and verified-weight descriptors.

use core::mem::size_of;

use kernels;
use snafu::ResultExt;

use super::plan::{F32Parameter, ProjectionWeight, f32_parameter, projection};
use crate::Qwen35Weights;
use crate::Result;
use crate::error::{
    ArithmeticOverflowSnafu, NativeKernelSnafu, NativeSessionStateSnafu, RecurrentConvolutionSnafu,
    RecurrentGdnSnafu,
};
use crate::qwen35::recurrent_layernorm_rms_epsilon;
use crate::qwen35_recurrent::{ExecutionLayout, RecurrentTensorRole, recurrent_tensor_name};

/// One verified matrix descriptor used by a native recurrent block.
#[derive(Debug)]
pub(crate) struct RecurrentProjectionWeights {
    pub(crate) qkv: ProjectionWeight,
    pub(crate) gate: ProjectionWeight,
    pub(crate) alpha: ProjectionWeight,
    pub(crate) beta: ProjectionWeight,
    pub(crate) output: ProjectionWeight,
}

/// One verified F32 descriptor used by a native recurrent block.
#[derive(Debug)]
pub(crate) struct RecurrentF32Parameters {
    pub(crate) attention_norm: F32Parameter,
    pub(crate) a: F32Parameter,
    pub(crate) dt: F32Parameter,
    pub(crate) convolution: F32Parameter,
    pub(crate) output_norm: F32Parameter,
}

/// Checked native recurrent operation geometry and named workspace extents.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RecurrentWorkspacePlan {
    pub(crate) input_norm: kernels::decoder_ops::RmsNormF32Plan,
    pub(crate) convolution_silu: kernels::decoder_ops::ElementwiseF32Plan,
    pub(crate) qk_l2: kernels::decoder_ops::RecurrentQkL2F32Plan,
    pub(crate) scalars: kernels::decoder_ops::RecurrentScalarsF32Plan,
    pub(crate) output_norm: kernels::decoder_ops::RmsNormF32Plan,
    pub(crate) output_silu_product: kernels::decoder_ops::ElementwiseF32Plan,
    pub(crate) normalized_hidden: usize,
    pub(crate) qkv: usize,
    pub(crate) z: usize,
    pub(crate) alpha: usize,
    pub(crate) beta_projection: usize,
    pub(crate) raw_convolution: usize,
    pub(crate) activated_convolution: usize,
    pub(crate) tiled_query: usize,
    pub(crate) tiled_key: usize,
    pub(crate) beta: usize,
    pub(crate) log_decay: usize,
    pub(crate) recurrence_output: usize,
    pub(crate) normalized_output: usize,
    pub(crate) gated_output: usize,
    pub(crate) projected_attention: usize,
    pub(crate) value_tail_offset: usize,
    pub(crate) value_tail_elements: usize,
    elements: usize,
}

impl RecurrentWorkspacePlan {
    /// Return the exact simultaneously allocated recurrent workspace extent.
    #[must_use]
    pub(crate) const fn elements(self) -> usize {
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
pub(crate) struct DeviceRecurrentPlan {
    pub(crate) block: usize,
    pub(crate) layout: ExecutionLayout,
    pub(crate) matrices: RecurrentProjectionWeights,
    pub(crate) parameters: RecurrentF32Parameters,
    pub(crate) workspace: RecurrentWorkspacePlan,
    pub(crate) convolution: kernels::CausalConvAllocationPlan,
    pub(crate) recurrence: kernels::MultiHeadRecurrentAllocationPlan,
    weight_bytes: usize,
}

impl DeviceRecurrentPlan {
    /// Bind one admitted recurrent main block to native operation descriptors.
    ///
    /// No device allocation, upload, submission, state mutation, or cache
    /// publication occurs here.
    pub(crate) fn from_weights(weights: &Qwen35Weights<'_>, block: usize) -> Result<Self> {
        let block_index = u64::try_from(block).map_err(|_| ArithmeticOverflowSnafu {
            context: "native recurrent block index",
        })?;
        let epsilon = recurrent_layernorm_rms_epsilon(weights.payload().observation().metadata())?;
        let layout = ExecutionLayout::try_from_profile(weights.recurrent_layout(), epsilon)?;
        layout.validate_recurrent_block(block_index)?;

        let convolution = kernels::CausalConvAllocationPlan::try_from_dimensions(
            1,
            layout.convolution_width(),
            layout.convolution_kernel(),
        )
        .context(RecurrentConvolutionSnafu)?;
        let recurrence = kernels::MultiHeadRecurrentAllocationPlan::try_from_dimensions(
            1,
            layout.value_head_count(),
            layout.value_head_count(),
            layout.key_dim(),
            layout.value_dim(),
        )
        .context(RecurrentGdnSnafu)?;
        let workspace = RecurrentWorkspacePlan::from_layout(layout, convolution, recurrence)?;
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
                elements_bytes(layout.hidden(), "native recurrent attention norm bytes")?,
                elements_bytes(layout.value_head_count(), "native recurrent A bytes")?,
                elements_bytes(layout.value_head_count(), "native recurrent dt bytes")?,
                elements_bytes(
                    convolution.weight_elements(),
                    "native recurrent convolution bytes",
                )?,
                elements_bytes(layout.value_dim(), "native recurrent output norm bytes")?,
            ],
            "native recurrent weight bytes",
        )?;

        Ok(Self {
            block,
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
    pub(crate) const fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    /// Return the checked active-or-staged convolution-history extent.
    #[must_use]
    pub(crate) const fn convolution_history_elements(&self) -> usize {
        self.convolution.history_elements()
    }

    /// Return the checked active-or-staged GDN-state extent.
    #[must_use]
    pub(crate) const fn recurrent_state_elements(&self) -> usize {
        self.recurrence.state_elements()
    }
}

impl RecurrentProjectionWeights {
    fn from_weights(weights: &Qwen35Weights<'_>, block: u64) -> Result<Self> {
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
        weights: &Qwen35Weights<'_>,
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
        convolution: kernels::CausalConvAllocationPlan,
        recurrence: kernels::MultiHeadRecurrentAllocationPlan,
    ) -> Result<Self> {
        let input_norm = kernels::decoder_ops::RmsNormF32Plan::try_from_dimensions(
            1,
            layout.hidden(),
            layout.epsilon(),
        )
        .context(NativeKernelSnafu)?;
        let convolution_silu = kernels::decoder_ops::ElementwiseF32Plan::try_from_elements(
            convolution.output_elements(),
        )
        .context(NativeKernelSnafu)?;
        let qk_l2 = kernels::decoder_ops::RecurrentQkL2F32Plan::try_from_dimensions(
            convolution_silu.elements(),
            layout.key_head_count(),
            layout.value_head_count(),
            layout.key_dim(),
            layout.epsilon(),
        )
        .context(NativeKernelSnafu)?;
        let scalars = kernels::decoder_ops::RecurrentScalarsF32Plan::try_from_value_heads(
            layout.value_head_count(),
        )
        .context(NativeKernelSnafu)?;
        let output_norm = kernels::decoder_ops::RmsNormF32Plan::try_from_dimensions(
            layout.value_head_count(),
            layout.value_dim(),
            layout.epsilon(),
        )
        .context(NativeKernelSnafu)?;
        let output_silu_product =
            kernels::decoder_ops::ElementwiseF32Plan::try_from_elements(output_norm.elements())
                .context(NativeKernelSnafu)?;

        if qk_l2.output_elements() != recurrence.query_and_key_elements()
            || qk_l2.output_elements() != recurrence.output_elements()
            || scalars.value_heads() != recurrence.scalar_elements()
            || output_norm.elements() != recurrence.output_elements()
        {
            return NativeSessionStateSnafu {
                rule: "native recurrent kernel plans must share the CPU-derived equal-head layout",
            }
            .fail();
        }

        let value_tail_offset = qk_l2.source_elements().checked_mul(2).ok_or_else(|| {
            ArithmeticOverflowSnafu {
                context: "native recurrent activated-convolution V offset",
            }
            .build()
        })?;
        let value_tail_end = value_tail_offset
            .checked_add(recurrence.output_elements())
            .ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "native recurrent activated-convolution V end",
                }
                .build()
            })?;
        if value_tail_end != convolution_silu.elements() {
            return NativeSessionStateSnafu {
                rule: "native recurrent V must be the exact activated-convolution tail",
            }
            .fail();
        }

        let normalized_hidden = input_norm.elements();
        let qkv = convolution.output_elements();
        let z = recurrence.output_elements();
        let alpha = scalars.value_heads();
        let beta_projection = scalars.value_heads();
        let raw_convolution = convolution.output_elements();
        let activated_convolution = convolution_silu.elements();
        let tiled_query = qk_l2.output_elements();
        let tiled_key = qk_l2.output_elements();
        let beta = scalars.value_heads();
        let log_decay = scalars.value_heads();
        let recurrence_output = recurrence.output_elements();
        let normalized_output = output_norm.elements();
        let gated_output = output_silu_product.elements();
        let projected_attention = layout.hidden();
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
                beta,
                log_decay,
                recurrence_output,
                normalized_output,
                gated_output,
                projected_attention,
            ],
            "native recurrent workspace elements",
        )?;

        Ok(Self {
            input_norm,
            convolution_silu,
            qk_l2,
            scalars,
            output_norm,
            output_silu_product,
            normalized_hidden,
            qkv,
            z,
            alpha,
            beta_projection,
            raw_convolution,
            activated_convolution,
            tiled_query,
            tiled_key,
            beta,
            log_decay,
            recurrence_output,
            normalized_output,
            gated_output,
            projected_attention,
            value_tail_offset,
            value_tail_elements: recurrence.output_elements(),
            elements,
        })
    }
}

fn elements_bytes(elements: usize, context: &'static str) -> Result<usize> {
    elements
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| ArithmeticOverflowSnafu { context }.build())
}

fn sum(values: &[usize], context: &'static str) -> Result<usize> {
    values.iter().try_fold(0_usize, |total, value| {
        total
            .checked_add(*value)
            .ok_or_else(|| ArithmeticOverflowSnafu { context }.build())
    })
}

#[cfg(test)]
mod tests {
    use super::{DeviceRecurrentPlan, sum};
    use crate::Qwen35Weights;
    use crate::qwen35::tests::{canonical_hybrid_fixture, verify_fixture};

    #[test]
    fn verified_recurrent_plan_reuses_the_cpu_layout_and_tensor_roles()
    -> core::result::Result<(), String> {
        let artifact = verify_fixture(&canonical_hybrid_fixture()?)?;
        let weights =
            Qwen35Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;
        let plan =
            DeviceRecurrentPlan::from_weights(&weights, 0).map_err(|error| error.to_string())?;

        assert_eq!(plan.workspace.qkv, plan.convolution.output_elements());
        assert_eq!(
            plan.workspace.value_tail_elements,
            plan.recurrence.output_elements()
        );
        assert_eq!(
            plan.workspace.value_tail_offset + plan.workspace.value_tail_elements,
            plan.workspace.activated_convolution,
            "V must remain an activated-convolution tail view"
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

        assert!(
            DeviceRecurrentPlan::from_weights(&weights, 3).is_err(),
            "a cadence full-attention block cannot use recurrent native geometry"
        );
        assert!(
            DeviceRecurrentPlan::from_weights(&weights, 4).is_err(),
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
}
