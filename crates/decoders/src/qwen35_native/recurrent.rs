//! Deferred native recurrent-block resources and launch ordering.

use core::ptr::{null, null_mut};

use hipcore::{Device, DeviceBuffer, Stream};
use snafu::ResultExt;

use super::dispatch::{launch_rms_norm, launch_silu_mul};
use super::finish::{
    DeferredLayerFinish, LayerFinishPlan, LayerFinishWeights, LayerFinishWorkspace,
};
use super::recurrent_plan::{DeviceRecurrentPlan, RecurrentWorkspacePlan};
use super::weights::{NativeMatrix, f32_parameter_buffer};
use crate::error::{NativeDeviceSnafu, NativeKernelSnafu};
use crate::{Qwen35Weights, Result};

/// Owned uploaded weights for one checked native recurrent main block.
pub(super) struct NativeRecurrentWeights {
    pub(super) qkv: NativeMatrix,
    pub(super) gate: NativeMatrix,
    pub(super) alpha: NativeMatrix,
    pub(super) beta: NativeMatrix,
    pub(super) output: NativeMatrix,
    pub(super) attention_norm: DeviceBuffer<f32>,
    pub(super) a: DeviceBuffer<f32>,
    pub(super) dt: DeviceBuffer<f32>,
    pub(super) convolution: DeviceBuffer<f32>,
    pub(super) output_norm: DeviceBuffer<f32>,
}

/// Exact named scratch buffers for one checked native recurrent block.
pub(super) struct NativeRecurrentWorkspace {
    pub(super) normalized_hidden: DeviceBuffer<f32>,
    pub(super) qkv: DeviceBuffer<f32>,
    pub(super) z: DeviceBuffer<f32>,
    pub(super) alpha: DeviceBuffer<f32>,
    pub(super) beta_projection: DeviceBuffer<f32>,
    pub(super) raw_convolution: DeviceBuffer<f32>,
    pub(super) activated_convolution: DeviceBuffer<f32>,
    pub(super) tiled_query: DeviceBuffer<f32>,
    pub(super) tiled_key: DeviceBuffer<f32>,
    pub(super) beta: DeviceBuffer<f32>,
    pub(super) log_decay: DeviceBuffer<f32>,
    pub(super) recurrence_output: DeviceBuffer<f32>,
    pub(super) normalized_output: DeviceBuffer<f32>,
    pub(super) gated_output: DeviceBuffer<f32>,
    pub(super) projected_attention: DeviceBuffer<f32>,
}

/// Per-layer committed and staged recurrent state.
///
/// A width-one convolution has no history footprint, so its two history
/// buffers are absent rather than represented by invalid zero-length spans.
pub(super) struct NativeRecurrentState {
    committed_convolution_history: Option<DeviceBuffer<f32>>,
    staged_convolution_history: Option<DeviceBuffer<f32>>,
    committed_recurrent_state: DeviceBuffer<f32>,
    staged_recurrent_state: DeviceBuffer<f32>,
}

/// Borrowed recurrent launch inputs with no completion or publication authority.
pub(super) struct DeferredRecurrent<'resources> {
    pub(super) plan: &'resources DeviceRecurrentPlan,
    pub(super) weights: &'resources NativeRecurrentWeights,
    pub(super) workspace: &'resources NativeRecurrentWorkspace,
    pub(super) state: &'resources NativeRecurrentState,
    pub(super) input: &'resources DeviceBuffer<f32>,
    pub(super) output: &'resources DeviceBuffer<f32>,
    pub(super) finish_plan: &'resources LayerFinishPlan,
    pub(super) finish_weights: &'resources LayerFinishWeights,
    pub(super) finish_workspace: &'resources LayerFinishWorkspace,
    pub(super) stream: &'resources Stream,
    pub(super) numerical_status: &'resources kernels::numerical_status::NativeNumericalStatus,
}

impl NativeRecurrentWeights {
    /// Upload exactly the verified descriptors bound by `plan`.
    pub(super) fn upload(
        weights: &Qwen35Weights,
        plan: &DeviceRecurrentPlan,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            qkv: NativeMatrix::upload(weights, &plan.matrices.qkv, device)?,
            gate: NativeMatrix::upload(weights, &plan.matrices.gate, device)?,
            alpha: NativeMatrix::upload(weights, &plan.matrices.alpha, device)?,
            beta: NativeMatrix::upload(weights, &plan.matrices.beta, device)?,
            output: NativeMatrix::upload(weights, &plan.matrices.output, device)?,
            attention_norm: f32_parameter_buffer(weights, &plan.parameters.attention_norm, device)?,
            a: f32_parameter_buffer(weights, &plan.parameters.a, device)?,
            dt: f32_parameter_buffer(weights, &plan.parameters.dt, device)?,
            convolution: f32_parameter_buffer(weights, &plan.parameters.convolution, device)?,
            output_norm: f32_parameter_buffer(weights, &plan.parameters.output_norm, device)?,
        })
    }
}

impl NativeRecurrentWorkspace {
    /// Allocate the exact scratch spans named by the recurrent plan.
    pub(super) fn new(plan: &RecurrentWorkspacePlan, device: &Device) -> Result<Self> {
        macro_rules! buffer {
            ($field:ident) => {
                DeviceBuffer::alloc(device, plan.$field).context(NativeDeviceSnafu)?
            };
        }
        Ok(Self {
            normalized_hidden: buffer!(normalized_hidden),
            qkv: buffer!(qkv),
            z: buffer!(z),
            alpha: buffer!(alpha),
            beta_projection: buffer!(beta_projection),
            raw_convolution: buffer!(raw_convolution),
            activated_convolution: buffer!(activated_convolution),
            tiled_query: buffer!(tiled_query),
            tiled_key: buffer!(tiled_key),
            beta: buffer!(beta),
            log_decay: buffer!(log_decay),
            recurrence_output: buffer!(recurrence_output),
            normalized_output: buffer!(normalized_output),
            gated_output: buffer!(gated_output),
            projected_attention: buffer!(projected_attention),
        })
    }
}

impl NativeRecurrentState {
    /// Allocate zero-initialized committed and staged state for one layer.
    pub(super) fn new(plan: &DeviceRecurrentPlan, device: &Device) -> Result<Self> {
        let history_elements = plan.convolution_history_elements();
        let committed_convolution_history = if history_elements == 0 {
            None
        } else {
            Some(zeroed_buffer(device, history_elements)?)
        };
        let staged_convolution_history = if history_elements == 0 {
            None
        } else {
            Some(zeroed_buffer(device, history_elements)?)
        };
        let recurrent_elements = plan.recurrent_state_elements();
        Ok(Self {
            committed_convolution_history,
            staged_convolution_history,
            committed_recurrent_state: zeroed_buffer(device, recurrent_elements)?,
            staged_recurrent_state: zeroed_buffer(device, recurrent_elements)?,
        })
    }

    /// Publish the already-complete staged state without allocation or I/O.
    ///
    /// The model-wide owner invokes this only after its one shared stream has
    /// proved completion for every recurrent and full-attention layer.
    pub(super) fn publish_completed(&mut self) {
        core::mem::swap(
            &mut self.committed_convolution_history,
            &mut self.staged_convolution_history,
        );
        core::mem::swap(
            &mut self.committed_recurrent_state,
            &mut self.staged_recurrent_state,
        );
    }
}

impl DeferredRecurrent<'_> {
    /// Submit one recurrent block without synchronization or state publication.
    ///
    /// # Safety
    ///
    /// All borrowed buffers must be exact non-overlapping spans on the stream
    /// device and remain live through stream completion. The sticky status
    /// classifies explicit operands and results. The caller retains the
    /// complete model resource bundle on submission failure and publishes
    /// staged recurrent state only after model-wide completion is established.
    pub(super) unsafe fn submit(&self) -> Result<()> {
        // SAFETY: submit's contract retains every checked projection span.
        unsafe { self.submit_projections() }?;
        // SAFETY: the staged convolution, activation, and Q/K spans remain live.
        unsafe { self.submit_convolution_arrangement() }?;
        // SAFETY: scalar controls and staged recurrence state remain live.
        unsafe { self.submit_scalars_and_recurrence() }?;
        // SAFETY: output, common finish, and all scratch spans remain live.
        unsafe { self.submit_output_and_finish() }
    }

    unsafe fn submit_projections(&self) -> Result<()> {
        // SAFETY: submit's contract retains the exact input, norm, and output
        // spans, and the checked plan fixes their geometry.
        unsafe {
            launch_rms_norm(
                self.plan.workspace.input_norm,
                self.input,
                &self.weights.attention_norm,
                &self.workspace.normalized_hidden,
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: each verified matrix binds its exact distinct input/output
        // span; all buffers remain owned by the deferred model resource.
        unsafe {
            self.weights.qkv.launch(
                &self.workspace.normalized_hidden,
                &self.workspace.qkv,
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: each remaining projection has its own exact output span.
        unsafe {
            self.weights.gate.launch(
                &self.workspace.normalized_hidden,
                &self.workspace.z,
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: alpha and beta outputs are distinct exact workspace spans.
        unsafe {
            self.weights.alpha.launch(
                &self.workspace.normalized_hidden,
                &self.workspace.alpha,
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: beta projection output remains distinct through completion.
        unsafe {
            self.weights.beta.launch(
                &self.workspace.normalized_hidden,
                &self.workspace.beta_projection,
                self.stream,
                self.numerical_status,
            )
        }
    }

    unsafe fn submit_convolution_arrangement(&self) -> Result<()> {
        // SAFETY: checked causal geometry and separate staged history preserve
        // the immutable committed state through completion.
        unsafe { self.submit_convolution() }?;
        // SAFETY: raw and activated convolution spans are distinct exact buffers.
        unsafe {
            kernels::decoder_ops::silu_checked(
                self.plan.workspace.convolution_silu,
                self.workspace.raw_convolution.as_device_ptr().cast_const(),
                self.workspace.raw_convolution.len(),
                self.workspace.activated_convolution.as_device_ptr(),
                self.workspace.activated_convolution.len(),
                self.stream,
                self.numerical_status,
            )
        }
        .context(NativeKernelSnafu)?;
        // SAFETY: the plan owns the grouped source and distinct tiled Q/K outputs.
        unsafe {
            kernels::decoder_ops::launch_recurrent_qk_l2_f32_checked(
                self.plan.workspace.qk_l2,
                self.workspace
                    .activated_convolution
                    .as_device_ptr()
                    .cast_const(),
                self.workspace.activated_convolution.len(),
                self.workspace.tiled_query.as_device_ptr(),
                self.workspace.tiled_query.len(),
                self.workspace.tiled_key.as_device_ptr(),
                self.workspace.tiled_key.len(),
                self.stream,
                self.numerical_status,
            )
        }
        .context(NativeKernelSnafu)
    }

    unsafe fn submit_scalars_and_recurrence(&self) -> Result<()> {
        // SAFETY: all scalar inputs are immutable exact spans and both outputs
        // are distinct writable buffers owned by this workspace.
        unsafe {
            kernels::decoder_ops::launch_recurrent_scalars_f32_checked(
                self.plan.workspace.scalars,
                self.workspace.alpha.as_device_ptr().cast_const(),
                self.workspace.alpha.len(),
                self.weights.dt.as_device_ptr().cast_const(),
                self.weights.dt.len(),
                self.weights.a.as_device_ptr().cast_const(),
                self.weights.a.len(),
                self.workspace.beta_projection.as_device_ptr().cast_const(),
                self.workspace.beta_projection.len(),
                self.workspace.beta.as_device_ptr(),
                self.workspace.beta.len(),
                self.workspace.log_decay.as_device_ptr(),
                self.workspace.log_decay.len(),
                self.stream,
                self.numerical_status,
            )
        }
        .context(NativeKernelSnafu)?;
        // SAFETY: staged GDN state and output are distinct from committed state
        // and all prior-operation buffers; V is the plan-checked tail view.
        unsafe { self.submit_recurrence() }
    }

    unsafe fn submit_output_and_finish(&self) -> Result<()> {
        // SAFETY: each output head is an exact separate RMSNorm row.
        unsafe {
            launch_rms_norm(
                self.plan.workspace.output_norm,
                &self.workspace.recurrence_output,
                &self.weights.output_norm,
                &self.workspace.normalized_output,
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: normalized recurrence output and Z are immutable distinct
        // inputs; gated_output is a separate owned exact span.
        unsafe {
            launch_silu_mul(
                self.plan.workspace.output_silu_product,
                &self.workspace.z,
                &self.workspace.normalized_output,
                &self.workspace.gated_output,
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the checked output projection and its spans remain live.
        unsafe {
            self.weights.output.launch(
                &self.workspace.gated_output,
                &self.workspace.projected_attention,
                self.stream,
                self.numerical_status,
            )
        }?;
        let finish = DeferredLayerFinish {
            input: self.input,
            attention_projection: &self.workspace.projected_attention,
            output: self.output,
            plan: self.finish_plan,
            weights: self.finish_weights,
            workspace: self.finish_workspace,
            stream: self.stream,
            numerical_status: self.numerical_status,
        };
        // SAFETY: submit's contract retains all common-finish spans and their
        // shared status allocation through stream completion.
        unsafe { finish.submit() }
    }

    unsafe fn submit_convolution(&self) -> Result<()> {
        let history_in: *const f32 = self
            .state
            .committed_convolution_history
            .as_ref()
            .map_or(null(), |buffer| buffer.as_device_ptr().cast_const());
        let history_out: *mut f32 = self
            .state
            .staged_convolution_history
            .as_ref()
            .map_or(null_mut(), DeviceBuffer::as_device_ptr);
        // SAFETY: the checked plan gives zero-length absent history for W=1,
        // otherwise the two state buffers are exact distinct device spans.
        unsafe {
            kernels::causal_conv::launch_causal_conv_step_f32_checked(
                self.plan.convolution,
                self.workspace.qkv.as_device_ptr().cast_const(),
                self.workspace.qkv.len(),
                self.weights.convolution.as_device_ptr().cast_const(),
                self.weights.convolution.len(),
                history_in,
                self.plan.convolution_history_elements(),
                history_out,
                self.plan.convolution_history_elements(),
                self.workspace.raw_convolution.as_device_ptr(),
                self.workspace.raw_convolution.len(),
                self.stream,
                self.numerical_status,
            )
        }
        .context(NativeKernelSnafu)
    }

    unsafe fn submit_recurrence(&self) -> Result<()> {
        let value = self
            .workspace
            .activated_convolution
            .as_device_ptr()
            .wrapping_add(self.plan.workspace.value_tail_offset)
            .cast_const();
        // SAFETY: the plan derives the V tail from the activated-convolution
        // extent, and committed/staged GDN state plus output are distinct.
        unsafe {
            kernels::gdn::launch_multi_head_recurrent_step_f32_checked(
                self.plan.recurrence,
                self.workspace.tiled_query.as_device_ptr().cast_const(),
                self.workspace.tiled_query.len(),
                self.workspace.tiled_key.as_device_ptr().cast_const(),
                self.workspace.tiled_key.len(),
                value,
                self.plan.workspace.value_tail_elements,
                self.workspace.beta.as_device_ptr().cast_const(),
                self.workspace.beta.len(),
                self.workspace.log_decay.as_device_ptr().cast_const(),
                self.workspace.log_decay.len(),
                self.plan.layout.gdn_scale(),
                self.state
                    .committed_recurrent_state
                    .as_device_ptr()
                    .cast_const(),
                self.state.committed_recurrent_state.len(),
                self.state.staged_recurrent_state.as_device_ptr(),
                self.state.staged_recurrent_state.len(),
                self.workspace.recurrence_output.as_device_ptr(),
                self.workspace.recurrence_output.len(),
                self.stream,
                self.numerical_status,
            )
        }
        .context(NativeKernelSnafu)
    }
}

fn zeroed_buffer(device: &Device, elements: usize) -> Result<DeviceBuffer<f32>> {
    let mut buffer = DeviceBuffer::alloc(device, elements).context(NativeDeviceSnafu)?;
    buffer.zero_fill().context(NativeDeviceSnafu)?;
    Ok(buffer)
}
