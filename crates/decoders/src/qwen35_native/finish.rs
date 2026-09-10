//! Shared native residual, feed-forward, and final-residual layer finish.

use core::mem::size_of;

use hipcore::{DeviceBuffer, Stream};
use snafu::ResultExt;

use super::custody::{NativeBufferSink, NativeBuildResult, NativeBuildScope};
use super::dispatch::{
    launch_residual_view, launch_rms_norm_view, launch_silu_mul_view,
};
use super::resources::NativeBufferView;
use super::plan::{
    F32Parameter, ProjectionWeight, dimension, elements_bytes, f32_parameter, projection, sum,
};
use super::weights::{NativeMatrix, f32_parameter_buffer};
use crate::error::NativeKernelSnafu;
use crate::qwen35_execution::{Layout, block_name};
use crate::{Qwen35Weights, Result};

#[derive(Debug, Clone)]
pub(super) struct LayerFinishPlan {
    pub(super) post_attention_norm: kernels::decoder_ops::RmsNormF32Plan,
    pub(super) ffn: kernels::decoder_ops::ElementwiseF32Plan,
    pub(super) residual: kernels::decoder_ops::ElementwiseF32Plan,
    pub(super) weights: LayerFinishWeightPlan,
    pub(super) workspace: LayerFinishWorkspacePlan,
    pub(super) demand: LayerFinishDeviceDemand,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LayerFinishDeviceDemand {
    pub(super) weights: usize,
    pub(super) scratch: usize,
}

#[derive(Debug, Clone)]
pub(super) struct LayerFinishWeightPlan {
    pub(super) post_attention_norm: F32Parameter,
    pub(super) ffn_gate: ProjectionWeight,
    pub(super) ffn_up: ProjectionWeight,
    pub(super) ffn_down: ProjectionWeight,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LayerFinishWorkspacePlan {
    pub(super) attention_residual: usize,
    pub(super) post_norm: usize,
    pub(super) ffn_gate: usize,
    pub(super) ffn_up: usize,
    pub(super) ffn_product: usize,
    pub(super) ffn_down: usize,
}

pub(super) struct LayerFinishWeights {
    pub(super) post_attention_norm: DeviceBuffer<f32>,
    pub(super) ffn_gate: NativeMatrix,
    pub(super) ffn_up: NativeMatrix,
    pub(super) ffn_down: NativeMatrix,
}

pub(super) struct LayerFinishWorkspace {
    pub(super) attention_residual: DeviceBuffer<f32>,
    pub(super) post_norm: DeviceBuffer<f32>,
    pub(super) ffn_gate: DeviceBuffer<f32>,
    pub(super) ffn_up: DeviceBuffer<f32>,
    pub(super) ffn_product: DeviceBuffer<f32>,
    pub(super) ffn_down: DeviceBuffer<f32>,
}

/// Exact active scratch views borrowed from a capacity-owned finish workspace.
pub(super) struct LayerFinishWorkspaceViews<'resources> {
    attention_residual: NativeBufferView<'resources, f32>,
    post_norm: NativeBufferView<'resources, f32>,
    ffn_gate: NativeBufferView<'resources, f32>,
    ffn_up: NativeBufferView<'resources, f32>,
    ffn_product: NativeBufferView<'resources, f32>,
    ffn_down: NativeBufferView<'resources, f32>,
}

/// Borrowed finish operands with no KV publication or resource-lifecycle authority.
pub(super) struct DeferredLayerFinish<'resources> {
    pub(super) input: NativeBufferView<'resources, f32>,
    pub(super) attention_projection: NativeBufferView<'resources, f32>,
    pub(super) output: NativeBufferView<'resources, f32>,
    pub(super) plan: &'resources LayerFinishPlan,
    pub(super) weights: &'resources LayerFinishWeights,
    pub(super) workspace: LayerFinishWorkspaceViews<'resources>,
    pub(super) stream: &'resources Stream,
    pub(super) numerical_status: &'resources kernels::numerical_status::NativeNumericalStatus,
}

impl LayerFinishPlan {
    pub(super) fn from_weights(
        weights: &Qwen35Weights,
        layout: Layout,
        block: usize,
    ) -> Result<Self> {
        Self::from_weights_rows(weights, layout, block, 1)
    }

    pub(super) fn from_weights_rows(
        weights: &Qwen35Weights,
        layout: Layout,
        block: usize,
        token_count: usize,
    ) -> Result<Self> {
        let weights = LayerFinishWeightPlan {
            post_attention_norm: f32_parameter(
                weights,
                block_name(block, "post_attention_norm.weight"),
                vec![dimension(
                    layout.hidden,
                    "native post-attention norm width",
                )?],
                layout.hidden,
            )?,
            ffn_gate: projection(weights, block_name(block, "ffn_gate.weight"))?,
            ffn_up: projection(weights, block_name(block, "ffn_up.weight"))?,
            ffn_down: projection(weights, block_name(block, "ffn_down.weight"))?,
        };
        Self::from_bound_weights(layout, weights, token_count)
    }

    pub(super) fn active(&self, layout: Layout, token_count: usize) -> Result<Self> {
        Self::from_bound_weights(layout, self.weights.clone(), token_count)
    }

    fn from_bound_weights(
        layout: Layout,
        weights: LayerFinishWeightPlan,
        token_count: usize,
    ) -> Result<Self> {
        let post_attention_norm = kernels::decoder_ops::RmsNormF32Plan::try_from_dimensions(
            token_count,
            layout.hidden,
            layout.epsilon(),
        )
        .context(NativeKernelSnafu)?;
        let ffn = kernels::decoder_ops::ElementwiseF32Plan::try_from_elements(
            token_count
                .checked_mul(layout.feed_forward)
                .ok_or_else(|| {
                    crate::error::ArithmeticOverflowSnafu {
                        context: "native layer-finish token count * feed-forward width",
                    }
                    .build()
                })?,
        )
        .context(NativeKernelSnafu)?;
        let residual = kernels::decoder_ops::ElementwiseF32Plan::try_from_elements(
            post_attention_norm.elements(),
        )
        .context(NativeKernelSnafu)?;
        let workspace = LayerFinishWorkspacePlan {
            attention_residual: residual.elements(),
            post_norm: post_attention_norm.elements(),
            ffn_gate: ffn.elements(),
            ffn_up: ffn.elements(),
            ffn_product: ffn.elements(),
            ffn_down: residual.elements(),
        };
        let demand = LayerFinishDeviceDemand {
            weights: weights.bytes()?,
            scratch: workspace.bytes()?,
        };
        Ok(Self {
            post_attention_norm,
            ffn,
            residual,
            weights,
            workspace,
            demand,
        })
    }
}

impl LayerFinishWeightPlan {
    fn bytes(&self) -> Result<usize> {
        let parameter_bytes = elements_bytes(
            self.post_attention_norm.elements,
            size_of::<f32>(),
            "native layer-finish parameter bytes",
        )?;
        sum(
            &[
                self.ffn_gate.serialized_bytes,
                self.ffn_up.serialized_bytes,
                self.ffn_down.serialized_bytes,
                parameter_bytes,
            ],
            "native layer-finish weight bytes",
        )
    }
}

impl LayerFinishWorkspacePlan {
    fn elements(self) -> Result<usize> {
        sum(
            &[
                self.attention_residual,
                self.post_norm,
                self.ffn_gate,
                self.ffn_up,
                self.ffn_product,
                self.ffn_down,
            ],
            "native layer-finish scratch elements",
        )
    }

    fn bytes(self) -> Result<usize> {
        elements_bytes(
            self.elements()?,
            size_of::<f32>(),
            "native layer-finish scratch bytes",
        )
    }
}

impl LayerFinishWeights {
    pub(super) fn upload(
        weights: &Qwen35Weights,
        plan: &LayerFinishPlan,
        device: &hipcore::Device,
        scope: &NativeBuildScope,
    ) -> NativeBuildResult<Self> {
        let post_attention_norm = scope.guard(
            f32_parameter_buffer(weights, &plan.weights.post_attention_norm, device, scope)?,
            |buffer, sink| sink.push_f32(buffer),
        );
        let ffn_gate = scope.guard(
            NativeMatrix::upload(weights, &plan.weights.ffn_gate, device, scope)?,
            NativeMatrix::into_buffer_sink,
        );
        let ffn_up = scope.guard(
            NativeMatrix::upload(weights, &plan.weights.ffn_up, device, scope)?,
            NativeMatrix::into_buffer_sink,
        );
        let ffn_down = scope.guard(
            NativeMatrix::upload(weights, &plan.weights.ffn_down, device, scope)?,
            NativeMatrix::into_buffer_sink,
        );
        Ok(Self {
            post_attention_norm: post_attention_norm.commit(),
            ffn_gate: ffn_gate.commit(),
            ffn_up: ffn_up.commit(),
            ffn_down: ffn_down.commit(),
        })
    }

    pub(super) fn into_buffer_sink(self, sink: &mut impl NativeBufferSink) {
        sink.push_f32(self.post_attention_norm);
        self.ffn_gate.into_buffer_sink(sink);
        self.ffn_up.into_buffer_sink(sink);
        self.ffn_down.into_buffer_sink(sink);
    }
}

impl LayerFinishWorkspace {
    pub(super) fn new(
        plan: LayerFinishWorkspacePlan,
        device: &hipcore::Device,
        scope: &NativeBuildScope,
    ) -> NativeBuildResult<Self> {
        macro_rules! buffer {
            ($field:ident) => {
                scope.allocate_f32(device, plan.$field)?
            };
        }
        let attention_residual = buffer!(attention_residual);
        let post_norm = buffer!(post_norm);
        let ffn_gate = buffer!(ffn_gate);
        let ffn_up = buffer!(ffn_up);
        let ffn_product = buffer!(ffn_product);
        let ffn_down = buffer!(ffn_down);
        Ok(Self {
            attention_residual: attention_residual.commit(),
            post_norm: post_norm.commit(),
            ffn_gate: ffn_gate.commit(),
            ffn_up: ffn_up.commit(),
            ffn_product: ffn_product.commit(),
            ffn_down: ffn_down.commit(),
        })
    }

    pub(super) fn into_buffer_sink(self, sink: &mut impl NativeBufferSink) {
        sink.push_f32(self.attention_residual);
        sink.push_f32(self.post_norm);
        sink.push_f32(self.ffn_gate);
        sink.push_f32(self.ffn_up);
        sink.push_f32(self.ffn_product);
        sink.push_f32(self.ffn_down);
    }

    pub(super) fn active(
        &self,
        plan: LayerFinishWorkspacePlan,
    ) -> Result<LayerFinishWorkspaceViews<'_>> {
        Ok(LayerFinishWorkspaceViews {
            attention_residual: NativeBufferView::prefix(
                &self.attention_residual,
                plan.attention_residual,
            )?,
            post_norm: NativeBufferView::prefix(&self.post_norm, plan.post_norm)?,
            ffn_gate: NativeBufferView::prefix(&self.ffn_gate, plan.ffn_gate)?,
            ffn_up: NativeBufferView::prefix(&self.ffn_up, plan.ffn_up)?,
            ffn_product: NativeBufferView::prefix(&self.ffn_product, plan.ffn_product)?,
            ffn_down: NativeBufferView::prefix(&self.ffn_down, plan.ffn_down)?,
        })
    }
}

impl DeferredLayerFinish<'_> {
    /// Submit the shared residual and feed-forward tail without synchronizing.
    ///
    /// # Safety
    ///
    /// All borrowed device buffers and the sticky status must be exact,
    /// non-overlapping spans on the stream device through completion. Checked
    /// launches classify explicit operands and results before publication.
    pub(super) unsafe fn submit(&self) -> Result<()> {
        // SAFETY: the caller retains the exact checked input, attention
        // projection, output, weights, and scratch spans through completion.
        unsafe {
            launch_residual_view(
                self.plan.residual,
                self.input,
                self.attention_projection,
                self.workspace.attention_residual,
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the caller retains the exact checked operands and output.
        unsafe {
            launch_rms_norm_view(
                self.plan.post_attention_norm,
                self.workspace.attention_residual,
                NativeBufferView::prefix(
                    &self.weights.post_attention_norm,
                    self.weights.post_attention_norm.len(),
                )?,
                self.workspace.post_norm,
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the checked matrix descriptor and exact owned spans remain
        // live on the ordered stream through completion.
        unsafe {
            self.weights.ffn_gate.launch_rows_view(
                self.workspace.post_norm,
                self.workspace.ffn_gate,
                self.plan.post_attention_norm.rows(),
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the checked matrix descriptor and exact owned spans remain
        // live on the ordered stream through completion.
        unsafe {
            self.weights.ffn_up.launch_rows_view(
                self.workspace.post_norm,
                self.workspace.ffn_up,
                self.plan.post_attention_norm.rows(),
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the caller retains the exact checked operands and output.
        unsafe {
            launch_silu_mul_view(
                self.plan.ffn,
                self.workspace.ffn_gate,
                self.workspace.ffn_up,
                self.workspace.ffn_product,
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the checked matrix descriptor and exact owned spans remain
        // live on the ordered stream through completion.
        unsafe {
            self.weights.ffn_down.launch_rows_view(
                self.workspace.ffn_product,
                self.workspace.ffn_down,
                self.plan.post_attention_norm.rows(),
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the caller retains the exact input and final output spans
        // through completion.
        unsafe {
            launch_residual_view(
                self.plan.residual,
                self.workspace.attention_residual,
                self.workspace.ffn_down,
                self.output,
                self.stream,
                self.numerical_status,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LayerFinishPlan;
    use crate::Qwen35Weights;
    use crate::qwen35::tests::{canonical_hybrid_fixture, verify_fixture};
    use crate::qwen35_execution::Layout;

    #[test]
    fn finish_rows_derive_t1_t2_t3_workspace_extents_from_the_shared_plans()
    -> core::result::Result<(), String> {
        let artifact = verify_fixture(&canonical_hybrid_fixture()?)?;
        let weights =
            Qwen35Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;
        let layout = Layout::from_metadata(&weights, 4).map_err(|error| error.to_string())?;
        let t1 = LayerFinishPlan::from_weights(&weights, layout, 3)
            .map_err(|error| error.to_string())?;
        assert!(
            LayerFinishPlan::from_weights_rows(&weights, layout, 3, 0).is_err(),
            "zero-token finish geometry must be refused before allocation"
        );
        for token_count in 1..=3 {
            let plan = LayerFinishPlan::from_weights_rows(&weights, layout, 3, token_count)
                .map_err(|error| error.to_string())?;
            assert_eq!(
                plan.post_attention_norm.rows(),
                token_count,
                "RMSNorm rows are the sole token-count owner"
            );
            assert_eq!(
                plan.residual.elements(),
                token_count * layout.hidden,
                "residual extent must derive from RMSNorm rows and hidden width"
            );
            assert_eq!(
                plan.ffn.elements(),
                token_count * layout.feed_forward,
                "FFN extent must derive from the requested token count"
            );
            assert_eq!(
                plan.workspace.attention_residual,
                plan.residual.elements(),
                "attention residual scratch must match residual geometry"
            );
            assert_eq!(
                plan.workspace.post_norm,
                plan.post_attention_norm.elements(),
                "post-norm scratch must use its checked RMSNorm extent"
            );
            assert_eq!(
                plan.workspace.ffn_down,
                plan.residual.elements(),
                "down projection scratch must match residual geometry"
            );
            assert_eq!(
                plan.workspace.bytes().map_err(|error| error.to_string())?,
                plan.demand.scratch,
                "scratch demand must be derived from the named workspace owner"
            );
            if token_count == 1 {
                assert_eq!(
                    plan.workspace.attention_residual, t1.workspace.attention_residual,
                    "T1 row constructor must preserve residual scratch"
                );
                assert_eq!(
                    plan.workspace.ffn_gate, t1.workspace.ffn_gate,
                    "T1 row constructor must preserve FFN scratch"
                );
                assert_eq!(
                    plan.demand.scratch, t1.demand.scratch,
                    "T1 row constructor must preserve scratch demand"
                );
            }
        }
        Ok(())
    }
}
