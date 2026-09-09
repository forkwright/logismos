//! Shared native residual, feed-forward, and final-residual layer finish.

use core::mem::size_of;

use hipcore::{DeviceBuffer, Stream};
use snafu::ResultExt;

use super::custody::NativeBufferSink;
use super::dispatch::{launch_residual, launch_rms_norm, launch_silu_mul};
use super::plan::{
    F32Parameter, ProjectionWeight, dimension, elements_bytes, f32_parameter, projection, sum,
};
use super::weights::{NativeMatrix, f32_parameter_buffer};
use crate::error::NativeKernelSnafu;
use crate::qwen35_execution::{Layout, block_name};
use crate::{Qwen35Weights, Result};

#[derive(Debug)]
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

#[derive(Debug)]
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

/// Borrowed finish operands with no KV publication or resource-lifecycle authority.
pub(super) struct DeferredLayerFinish<'resources> {
    pub(super) input: &'resources DeviceBuffer<f32>,
    pub(super) attention_projection: &'resources DeviceBuffer<f32>,
    pub(super) output: &'resources DeviceBuffer<f32>,
    pub(super) plan: &'resources LayerFinishPlan,
    pub(super) weights: &'resources LayerFinishWeights,
    pub(super) workspace: &'resources LayerFinishWorkspace,
    pub(super) stream: &'resources Stream,
    pub(super) numerical_status: &'resources kernels::numerical_status::NativeNumericalStatus,
}

impl LayerFinishPlan {
    pub(super) fn from_weights(
        weights: &Qwen35Weights,
        layout: Layout,
        block: usize,
    ) -> Result<Self> {
        let post_attention_norm = kernels::decoder_ops::RmsNormF32Plan::try_from_dimensions(
            1,
            layout.hidden,
            layout.epsilon(),
        )
        .context(NativeKernelSnafu)?;
        let ffn = kernels::decoder_ops::ElementwiseF32Plan::try_from_elements(layout.feed_forward)
            .context(NativeKernelSnafu)?;
        let residual = kernels::decoder_ops::ElementwiseF32Plan::try_from_elements(layout.hidden)
            .context(NativeKernelSnafu)?;
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
        let workspace = LayerFinishWorkspacePlan {
            attention_residual: residual.elements(),
            post_norm: post_attention_norm.elements(),
            ffn_gate: ffn.elements(),
            ffn_up: ffn.elements(),
            ffn_product: ffn.elements(),
            ffn_down: layout.hidden,
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
    ) -> Result<Self> {
        Ok(Self {
            post_attention_norm: f32_parameter_buffer(
                weights,
                &plan.weights.post_attention_norm,
                device,
            )?,
            ffn_gate: NativeMatrix::upload(weights, &plan.weights.ffn_gate, device)?,
            ffn_up: NativeMatrix::upload(weights, &plan.weights.ffn_up, device)?,
            ffn_down: NativeMatrix::upload(weights, &plan.weights.ffn_down, device)?,
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
    pub(super) fn new(plan: LayerFinishWorkspacePlan, device: &hipcore::Device) -> Result<Self> {
        macro_rules! buffer {
            ($field:ident) => {
                DeviceBuffer::alloc(device, plan.$field).context(crate::error::NativeDeviceSnafu)?
            };
        }
        Ok(Self {
            attention_residual: buffer!(attention_residual),
            post_norm: buffer!(post_norm),
            ffn_gate: buffer!(ffn_gate),
            ffn_up: buffer!(ffn_up),
            ffn_product: buffer!(ffn_product),
            ffn_down: buffer!(ffn_down),
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
            launch_residual(
                self.plan.residual,
                self.input,
                self.attention_projection,
                &self.workspace.attention_residual,
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the caller retains the exact checked operands and output.
        unsafe {
            launch_rms_norm(
                self.plan.post_attention_norm,
                &self.workspace.attention_residual,
                &self.weights.post_attention_norm,
                &self.workspace.post_norm,
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the checked matrix descriptor and exact owned spans remain
        // live on the ordered stream through completion.
        unsafe {
            self.weights.ffn_gate.launch(
                &self.workspace.post_norm,
                &self.workspace.ffn_gate,
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the checked matrix descriptor and exact owned spans remain
        // live on the ordered stream through completion.
        unsafe {
            self.weights.ffn_up.launch(
                &self.workspace.post_norm,
                &self.workspace.ffn_up,
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the caller retains the exact checked operands and output.
        unsafe {
            launch_silu_mul(
                self.plan.ffn,
                &self.workspace.ffn_gate,
                &self.workspace.ffn_up,
                &self.workspace.ffn_product,
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the checked matrix descriptor and exact owned spans remain
        // live on the ordered stream through completion.
        unsafe {
            self.weights.ffn_down.launch(
                &self.workspace.ffn_product,
                &self.workspace.ffn_down,
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the caller retains the exact input and final output spans
        // through completion.
        unsafe {
            launch_residual(
                self.plan.residual,
                &self.workspace.attention_residual,
                &self.workspace.ffn_down,
                self.output,
                self.stream,
                self.numerical_status,
            )
        }
    }
}
