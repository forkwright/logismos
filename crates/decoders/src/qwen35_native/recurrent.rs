//! Deferred native recurrent-block resources and launch ordering.

use core::ptr::{null, null_mut};

use hipcore::{Device, DeviceBuffer, Stream};
use snafu::ResultExt;

use super::custody::{
    NativeBufferSink, NativeBuildGuard, NativeBuildResult, NativeBuildScope, NativeBuildSource,
};
use super::dispatch::{launch_rms_norm_view, launch_silu_mul_view};
use super::finish::{
    ActiveLayerFinishPlan, DeferredLayerFinish, LayerFinishWeights, LayerFinishWorkspace,
};
use super::recurrent_plan::{ActiveRecurrentPlan, DeviceRecurrentPlan, RecurrentWorkspacePlan};
use super::resources::NativeBufferView;
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
    pub(super) value_head_major: DeviceBuffer<f32>,
    pub(super) beta: DeviceBuffer<f32>,
    pub(super) log_decay: DeviceBuffer<f32>,
    pub(super) recurrence_output: DeviceBuffer<f32>,
    pub(super) token_major_recurrence_output: DeviceBuffer<f32>,
    pub(super) normalized_output: DeviceBuffer<f32>,
    pub(super) gated_output: DeviceBuffer<f32>,
    pub(super) projected_attention: DeviceBuffer<f32>,
}

/// Exact active recurrent scratch borrowed from capacity-owned workspace.
pub(super) struct NativeRecurrentWorkspaceViews<'resources> {
    normalized_hidden: NativeBufferView<'resources, f32>,
    qkv: NativeBufferView<'resources, f32>,
    z: NativeBufferView<'resources, f32>,
    alpha: NativeBufferView<'resources, f32>,
    beta_projection: NativeBufferView<'resources, f32>,
    raw_convolution: NativeBufferView<'resources, f32>,
    activated_convolution: NativeBufferView<'resources, f32>,
    tiled_query: NativeBufferView<'resources, f32>,
    tiled_key: NativeBufferView<'resources, f32>,
    value_head_major: NativeBufferView<'resources, f32>,
    beta: NativeBufferView<'resources, f32>,
    log_decay: NativeBufferView<'resources, f32>,
    recurrence_output: NativeBufferView<'resources, f32>,
    token_major_recurrence_output: NativeBufferView<'resources, f32>,
    normalized_output: NativeBufferView<'resources, f32>,
    gated_output: NativeBufferView<'resources, f32>,
    projected_attention: NativeBufferView<'resources, f32>,
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
    pub(super) plan: &'resources ActiveRecurrentPlan,
    pub(super) weights: &'resources NativeRecurrentWeights,
    pub(super) workspace: NativeRecurrentWorkspaceViews<'resources>,
    pub(super) state: &'resources NativeRecurrentState,
    pub(super) input: NativeBufferView<'resources, f32>,
    pub(super) output: NativeBufferView<'resources, f32>,
    pub(super) finish_plan: &'resources ActiveLayerFinishPlan,
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
        scope: &NativeBuildScope,
    ) -> NativeBuildResult<Self> {
        let qkv = matrix_guard(weights, &plan.matrices.qkv, device, scope)?;
        let gate = matrix_guard(weights, &plan.matrices.gate, device, scope)?;
        let alpha = matrix_guard(weights, &plan.matrices.alpha, device, scope)?;
        let beta = matrix_guard(weights, &plan.matrices.beta, device, scope)?;
        let output = matrix_guard(weights, &plan.matrices.output, device, scope)?;
        let attention_norm =
            parameter_guard(weights, &plan.parameters.attention_norm, device, scope)?;
        let a = parameter_guard(weights, &plan.parameters.a, device, scope)?;
        let dt = parameter_guard(weights, &plan.parameters.dt, device, scope)?;
        let convolution = parameter_guard(weights, &plan.parameters.convolution, device, scope)?;
        let output_norm = parameter_guard(weights, &plan.parameters.output_norm, device, scope)?;
        Ok(Self {
            qkv: qkv.commit(),
            gate: gate.commit(),
            alpha: alpha.commit(),
            beta: beta.commit(),
            output: output.commit(),
            attention_norm: attention_norm.commit(),
            a: a.commit(),
            dt: dt.commit(),
            convolution: convolution.commit(),
            output_norm: output_norm.commit(),
        })
    }

    pub(super) fn into_buffer_sink(self, sink: &mut impl NativeBufferSink) {
        self.qkv.into_buffer_sink(sink);
        self.gate.into_buffer_sink(sink);
        self.alpha.into_buffer_sink(sink);
        self.beta.into_buffer_sink(sink);
        self.output.into_buffer_sink(sink);
        sink.push_f32(self.attention_norm);
        sink.push_f32(self.a);
        sink.push_f32(self.dt);
        sink.push_f32(self.convolution);
        sink.push_f32(self.output_norm);
    }
}

impl NativeRecurrentWorkspace {
    /// Allocate the exact scratch spans named by the recurrent plan.
    pub(super) fn new(
        plan: &RecurrentWorkspacePlan,
        device: &Device,
        scope: &NativeBuildScope,
    ) -> NativeBuildResult<Self> {
        macro_rules! buffer {
            ($field:ident) => {
                scope.allocate_f32(device, plan.$field)?
            };
        }
        let normalized_hidden = buffer!(normalized_hidden);
        let qkv = buffer!(qkv);
        let z = buffer!(z);
        let alpha = buffer!(alpha);
        let beta_projection = buffer!(beta_projection);
        let raw_convolution = buffer!(raw_convolution);
        let activated_convolution = buffer!(activated_convolution);
        let tiled_query = buffer!(tiled_query);
        let tiled_key = buffer!(tiled_key);
        let value_head_major = buffer!(value_head_major);
        let beta = buffer!(beta);
        let log_decay = buffer!(log_decay);
        let recurrence_output = buffer!(recurrence_output);
        let token_major_recurrence_output = buffer!(token_major_recurrence_output);
        let normalized_output = buffer!(normalized_output);
        let gated_output = buffer!(gated_output);
        let projected_attention = buffer!(projected_attention);
        Ok(Self {
            normalized_hidden: normalized_hidden.commit(),
            qkv: qkv.commit(),
            z: z.commit(),
            alpha: alpha.commit(),
            beta_projection: beta_projection.commit(),
            raw_convolution: raw_convolution.commit(),
            activated_convolution: activated_convolution.commit(),
            tiled_query: tiled_query.commit(),
            tiled_key: tiled_key.commit(),
            value_head_major: value_head_major.commit(),
            beta: beta.commit(),
            log_decay: log_decay.commit(),
            recurrence_output: recurrence_output.commit(),
            token_major_recurrence_output: token_major_recurrence_output.commit(),
            normalized_output: normalized_output.commit(),
            gated_output: gated_output.commit(),
            projected_attention: projected_attention.commit(),
        })
    }

    pub(super) fn into_buffer_sink(self, sink: &mut impl NativeBufferSink) {
        sink.push_f32(self.normalized_hidden);
        sink.push_f32(self.qkv);
        sink.push_f32(self.z);
        sink.push_f32(self.alpha);
        sink.push_f32(self.beta_projection);
        sink.push_f32(self.raw_convolution);
        sink.push_f32(self.activated_convolution);
        sink.push_f32(self.tiled_query);
        sink.push_f32(self.tiled_key);
        sink.push_f32(self.value_head_major);
        sink.push_f32(self.beta);
        sink.push_f32(self.log_decay);
        sink.push_f32(self.recurrence_output);
        sink.push_f32(self.token_major_recurrence_output);
        sink.push_f32(self.normalized_output);
        sink.push_f32(self.gated_output);
        sink.push_f32(self.projected_attention);
    }

    /// # Errors
    ///
    /// Returns an error when any active scratch extent exceeds this capacity owner.
    pub(super) fn active(
        &self,
        plan: RecurrentWorkspacePlan,
    ) -> Result<NativeRecurrentWorkspaceViews<'_>> {
        Ok(NativeRecurrentWorkspaceViews {
            normalized_hidden: NativeBufferView::prefix(
                &self.normalized_hidden,
                plan.normalized_hidden,
            )?,
            qkv: NativeBufferView::prefix(&self.qkv, plan.qkv)?,
            z: NativeBufferView::prefix(&self.z, plan.z)?,
            alpha: NativeBufferView::prefix(&self.alpha, plan.alpha)?,
            beta_projection: NativeBufferView::prefix(&self.beta_projection, plan.beta_projection)?,
            raw_convolution: NativeBufferView::prefix(&self.raw_convolution, plan.raw_convolution)?,
            activated_convolution: NativeBufferView::prefix(
                &self.activated_convolution,
                plan.activated_convolution,
            )?,
            tiled_query: NativeBufferView::prefix(&self.tiled_query, plan.tiled_query)?,
            tiled_key: NativeBufferView::prefix(&self.tiled_key, plan.tiled_key)?,
            value_head_major: NativeBufferView::prefix(
                &self.value_head_major,
                plan.value_head_major,
            )?,
            beta: NativeBufferView::prefix(&self.beta, plan.beta)?,
            log_decay: NativeBufferView::prefix(&self.log_decay, plan.log_decay)?,
            recurrence_output: NativeBufferView::prefix(
                &self.recurrence_output,
                plan.recurrence_output,
            )?,
            token_major_recurrence_output: NativeBufferView::prefix(
                &self.token_major_recurrence_output,
                plan.token_major_recurrence_output,
            )?,
            normalized_output: NativeBufferView::prefix(
                &self.normalized_output,
                plan.normalized_output,
            )?,
            gated_output: NativeBufferView::prefix(&self.gated_output, plan.gated_output)?,
            projected_attention: NativeBufferView::prefix(
                &self.projected_attention,
                plan.projected_attention,
            )?,
        })
    }
}

impl NativeRecurrentState {
    /// Allocate zero-initialized committed and staged state for one layer.
    pub(super) fn new(
        plan: &DeviceRecurrentPlan,
        device: &Device,
        scope: &NativeBuildScope,
    ) -> NativeBuildResult<Self> {
        let history_elements = plan.convolution_history_elements();
        let committed_convolution_history = if history_elements == 0 {
            None
        } else {
            Some(zeroed_buffer(device, history_elements, scope)?)
        };
        let staged_convolution_history = if history_elements == 0 {
            None
        } else {
            Some(zeroed_buffer(device, history_elements, scope)?)
        };
        let recurrent_elements = plan.recurrent_state_elements();
        let committed_recurrent_state = zeroed_buffer(device, recurrent_elements, scope)?;
        let staged_recurrent_state = zeroed_buffer(device, recurrent_elements, scope)?;
        Ok(Self {
            committed_convolution_history: committed_convolution_history
                .map(NativeBuildGuard::commit),
            staged_convolution_history: staged_convolution_history.map(NativeBuildGuard::commit),
            committed_recurrent_state: committed_recurrent_state.commit(),
            staged_recurrent_state: staged_recurrent_state.commit(),
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

    pub(super) fn into_buffer_sink(self, sink: &mut impl NativeBufferSink) {
        if let Some(history) = self.committed_convolution_history {
            sink.push_f32(history);
        }
        if let Some(history) = self.staged_convolution_history {
            sink.push_f32(history);
        }
        sink.push_f32(self.committed_recurrent_state);
        sink.push_f32(self.staged_recurrent_state);
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
            launch_rms_norm_view(
                self.plan.workspace.input_norm,
                self.input,
                NativeBufferView::prefix(
                    &self.weights.attention_norm,
                    self.weights.attention_norm.len(),
                )?,
                self.workspace.normalized_hidden,
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: each verified matrix binds its exact distinct input/output
        // span; all buffers remain owned by the deferred model resource.
        unsafe {
            self.weights.qkv.launch_rows_view(
                self.workspace.normalized_hidden,
                self.workspace.qkv,
                self.plan.recurrence.token_count(),
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: each remaining projection has its own exact output span.
        unsafe {
            self.weights.gate.launch_rows_view(
                self.workspace.normalized_hidden,
                self.workspace.z,
                self.plan.recurrence.token_count(),
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: alpha and beta outputs are distinct exact workspace spans.
        unsafe {
            self.weights.alpha.launch_rows_view(
                self.workspace.normalized_hidden,
                self.workspace.alpha,
                self.plan.recurrence.token_count(),
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: beta projection output remains distinct through completion.
        unsafe {
            self.weights.beta.launch_rows_view(
                self.workspace.normalized_hidden,
                self.workspace.beta_projection,
                self.plan.recurrence.token_count(),
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
                self.workspace.raw_convolution.as_const_ptr(),
                self.workspace.raw_convolution.len(),
                self.workspace.activated_convolution.as_mut_ptr(),
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
                self.workspace.activated_convolution.as_const_ptr(),
                self.workspace.activated_convolution.len(),
                self.workspace.tiled_query.as_mut_ptr(),
                self.workspace.tiled_query.len(),
                self.workspace.tiled_key.as_mut_ptr(),
                self.workspace.tiled_key.len(),
                self.stream,
                self.numerical_status,
            )
        }
        .context(NativeKernelSnafu)?;
        // SAFETY: the value layout gathers every token's checked convolution
        // tail into distinct head-major storage retained through completion.
        unsafe {
            kernels::decoder_ops::launch_recurrent_values_to_head_major_f32_checked(
                self.plan.workspace.value_layout,
                self.workspace.activated_convolution.as_const_ptr(),
                self.workspace.activated_convolution.len(),
                self.workspace.value_head_major.as_mut_ptr(),
                self.workspace.value_head_major.len(),
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
                self.workspace.alpha.as_const_ptr(),
                self.workspace.alpha.len(),
                self.weights.dt.as_device_ptr().cast_const(),
                self.weights.dt.len(),
                self.weights.a.as_device_ptr().cast_const(),
                self.weights.a.len(),
                self.workspace.beta_projection.as_const_ptr(),
                self.workspace.beta_projection.len(),
                self.workspace.beta.as_mut_ptr(),
                self.workspace.beta.len(),
                self.workspace.log_decay.as_mut_ptr(),
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
        // SAFETY: the compact GDN output and token-major destination are exact
        // disjoint workspace spans retained by this deferred submission.
        unsafe {
            kernels::decoder_ops::launch_recurrent_values_to_token_major_f32_checked(
                self.plan.workspace.value_layout,
                self.workspace.recurrence_output.as_const_ptr(),
                self.workspace.recurrence_output.len(),
                self.workspace.token_major_recurrence_output.as_mut_ptr(),
                self.workspace.token_major_recurrence_output.len(),
                self.stream,
                self.numerical_status,
            )
        }
        .context(NativeKernelSnafu)?;
        // SAFETY: each output head is an exact separate RMSNorm row.
        unsafe {
            launch_rms_norm_view(
                self.plan.workspace.output_norm,
                self.workspace.token_major_recurrence_output,
                NativeBufferView::prefix(
                    &self.weights.output_norm,
                    self.weights.output_norm.len(),
                )?,
                self.workspace.normalized_output,
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: normalized recurrence output and Z are immutable distinct
        // inputs; gated_output is a separate owned exact span.
        unsafe {
            launch_silu_mul_view(
                self.plan.workspace.output_silu_product,
                self.workspace.z,
                self.workspace.normalized_output,
                self.workspace.gated_output,
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the checked output projection and its spans remain live.
        unsafe {
            self.weights.output.launch_rows_view(
                self.workspace.gated_output,
                self.workspace.projected_attention,
                self.plan.recurrence.token_count(),
                self.stream,
                self.numerical_status,
            )
        }?;
        let finish = DeferredLayerFinish {
            input: self.input,
            attention_projection: self.workspace.projected_attention,
            output: self.output,
            plan: self.finish_plan,
            weights: self.finish_weights,
            workspace: self.finish_workspace.active(self.finish_plan.workspace)?,
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
            kernels::causal_conv::launch_causal_conv_fwd_f32_checked(
                self.plan.convolution,
                self.workspace.qkv.as_const_ptr(),
                self.workspace.qkv.len(),
                self.weights.convolution.as_device_ptr().cast_const(),
                self.weights.convolution.len(),
                history_in,
                self.plan.convolution.history_elements(),
                history_out,
                self.plan.convolution.history_elements(),
                self.workspace.raw_convolution.as_mut_ptr(),
                self.workspace.raw_convolution.len(),
                self.stream,
                self.numerical_status,
            )
        }
        .context(NativeKernelSnafu)
    }

    unsafe fn submit_recurrence(&self) -> Result<()> {
        // SAFETY: the plan derives compact V from each activated-convolution
        // row, and committed/staged GDN state plus output are distinct.
        unsafe {
            kernels::gdn::launch_multi_head_recurrent_fwd_f32_checked(
                self.plan.recurrence,
                self.workspace.tiled_query.as_const_ptr(),
                self.workspace.tiled_query.len(),
                self.workspace.tiled_key.as_const_ptr(),
                self.workspace.tiled_key.len(),
                self.workspace.value_head_major.as_const_ptr(),
                self.workspace.value_head_major.len(),
                self.workspace.beta.as_const_ptr(),
                self.workspace.beta.len(),
                self.workspace.log_decay.as_const_ptr(),
                self.workspace.log_decay.len(),
                self.plan.layout.gdn_scale(),
                self.state
                    .committed_recurrent_state
                    .as_device_ptr()
                    .cast_const(),
                self.state.committed_recurrent_state.len(),
                self.state.staged_recurrent_state.as_device_ptr(),
                self.state.staged_recurrent_state.len(),
                self.workspace.recurrence_output.as_mut_ptr(),
                self.workspace.recurrence_output.len(),
                self.stream,
                self.numerical_status,
            )
        }
        .context(NativeKernelSnafu)
    }
}

fn matrix_guard(
    weights: &Qwen35Weights,
    plan: &super::plan::ProjectionWeight,
    device: &Device,
    scope: &NativeBuildScope,
) -> NativeBuildResult<NativeBuildGuard<NativeMatrix>> {
    let matrix = NativeMatrix::upload(weights, plan, device, scope)?;
    Ok(scope.guard(matrix, NativeMatrix::into_buffer_sink))
}

fn parameter_guard(
    weights: &Qwen35Weights,
    plan: &super::plan::F32Parameter,
    device: &Device,
    scope: &NativeBuildScope,
) -> NativeBuildResult<NativeBuildGuard<DeviceBuffer<f32>>> {
    let buffer = f32_parameter_buffer(weights, plan, device, scope)?;
    Ok(scope.guard(buffer, |buffer, sink| sink.push_f32(buffer)))
}

fn zeroed_buffer(
    device: &Device,
    elements: usize,
    scope: &NativeBuildScope,
) -> NativeBuildResult<NativeBuildGuard<DeviceBuffer<f32>>> {
    let mut buffer = scope.allocate_f32(device, elements)?;
    buffer
        .zero_fill()
        .context(NativeDeviceSnafu)
        .map_err(NativeBuildSource::decoder)?;
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use hipcore::{Device, Stream};

    use super::{
        DeferredRecurrent, NativeRecurrentState, NativeRecurrentWeights, NativeRecurrentWorkspace,
    };
    use crate::Qwen35Weights;
    use crate::qwen35::tests::{
        CanonicalHybridOracle, assert_f32_matches_f64, canonical_hybrid_fixture, verify_fixture,
    };
    use crate::qwen35_execution::Layout;
    use crate::qwen35_native::custody::{NativeBufferSink, NativeBuildScope};
    use crate::qwen35_native::finish::{LayerFinishPlan, LayerFinishWeights, LayerFinishWorkspace};
    use crate::qwen35_native::model_resources::build_numerical_status;
    use crate::qwen35_native::recurrent_plan::DeviceRecurrentPlan;
    use crate::qwen35_native::resources::NativeBufferView;
    use kernels::PackedPrefillPlan;

    const RECURRENT_BLOCK: usize = 0;
    const RECURRENT_BLOCK_LAYER: u64 = 0;
    const TOKEN_COUNT: usize = 3;

    struct RecurrentWitnessResources {
        stream: Stream,
        numerical_status: kernels::numerical_status::NativeNumericalStatus,
        native_weights: NativeRecurrentWeights,
        recurrent_workspace: NativeRecurrentWorkspace,
        state: NativeRecurrentState,
        finish_weights: LayerFinishWeights,
        finish_workspace: LayerFinishWorkspace,
        input: hipcore::DeviceBuffer<f32>,
        output: hipcore::DeviceBuffer<f32>,
    }

    impl RecurrentWitnessResources {
        /// Retain every submitted owner when completion cannot be established.
        fn retain_unconfirmed(self) {
            core::mem::forget(self);
        }

        fn build(
            device: &Device,
            weights: &Qwen35Weights,
            plan: &DeviceRecurrentPlan,
            finish: &LayerFinishPlan,
            hidden: &[f32],
        ) -> core::result::Result<Self, String> {
            let stream =
                Stream::new_tracked(device).map_err(|error| format!("create stream: {error}"))?;
            let scope = NativeBuildScope::new();
            let numerical_status = scope.guard(
                build_numerical_status(device, &scope).map_err(|error| error.to_string())?,
                |status, sink| sink.push_u32(status.into_buffer()),
            );
            let native_weights = scope.guard(
                NativeRecurrentWeights::upload(weights, plan, device, &scope)
                    .map_err(|error| error.to_string())?,
                NativeRecurrentWeights::into_buffer_sink,
            );
            let recurrent_workspace = scope.guard(
                NativeRecurrentWorkspace::new(&plan.workspace, device, &scope)
                    .map_err(|error| error.to_string())?,
                NativeRecurrentWorkspace::into_buffer_sink,
            );
            let state = scope.guard(
                NativeRecurrentState::new(plan, device, &scope)
                    .map_err(|error| error.to_string())?,
                NativeRecurrentState::into_buffer_sink,
            );
            let finish_weights = scope.guard(
                LayerFinishWeights::upload(weights, finish, device, &scope)
                    .map_err(|error| error.to_string())?,
                LayerFinishWeights::into_buffer_sink,
            );
            let finish_workspace = scope.guard(
                LayerFinishWorkspace::new(finish.workspace, device, &scope)
                    .map_err(|error| error.to_string())?,
                LayerFinishWorkspace::into_buffer_sink,
            );
            let mut input = scope
                .allocate_f32(device, hidden.len())
                .map_err(|error| error.to_string())?;
            input
                .copy_from_host(hidden)
                .map_err(|error| format!("upload recurrent rows: {error}"))?;
            let output = scope
                .allocate_f32(device, hidden.len())
                .map_err(|error| error.to_string())?;

            Ok(Self {
                stream,
                numerical_status: numerical_status.commit(),
                native_weights: native_weights.commit(),
                recurrent_workspace: recurrent_workspace.commit(),
                state: state.commit(),
                finish_weights: finish_weights.commit(),
                finish_workspace: finish_workspace.commit(),
                input: input.commit(),
                output: output.commit(),
            })
        }

        fn assert_matches(
            &self,
            expected_hidden: &[f64],
            expected_history: &[f64],
            expected_state: &[f64],
        ) -> core::result::Result<(), String> {
            let actual_hidden = copy_to_host(&self.output, "native recurrent hidden")?;
            let actual_history = match &self.state.staged_convolution_history {
                Some(history) => copy_to_host(history, "native recurrent staged history")?,
                None => Vec::new(),
            };
            let actual_state = copy_to_host(
                &self.state.staged_recurrent_state,
                "native recurrent staged GDN state",
            )?;
            assert_f32_matches_f64(
                &actual_hidden,
                expected_hidden,
                "native recurrent complete rows",
            )?;
            assert_f32_matches_f64(
                &actual_history,
                expected_history,
                "native recurrent staged history",
            )?;
            assert_f32_matches_f64(
                &actual_state,
                expected_state,
                "native recurrent staged GDN state",
            )
        }
    }

    #[test]
    #[ignore = "requires an operator-reserved visible gfx1100 device; source tests do not qualify hardware"]
    fn reserved_device_native_recurrent_rows_match_cpu_block_and_staged_state()
    -> core::result::Result<(), String> {
        let fixture = canonical_hybrid_fixture()?;
        let artifact = verify_fixture(&fixture)?;
        let weights =
            Qwen35Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;
        let packed = PackedPrefillPlan::new(&[TOKEN_COUNT], &[0], TOKEN_COUNT)
            .map_err(|error| error.to_string())?;
        let plan = DeviceRecurrentPlan::from_packed_prefill(&weights, RECURRENT_BLOCK, &packed)
            .map_err(|error| error.to_string())?;
        let hidden = recurrent_witness_input(TOKEN_COUNT, plan.layout.hidden())?;
        let layout =
            Layout::from_metadata(&weights, TOKEN_COUNT).map_err(|error| error.to_string())?;
        let finish =
            LayerFinishPlan::from_weights_rows(&weights, layout, RECURRENT_BLOCK, TOKEN_COUNT)
                .map_err(|error| error.to_string())?;
        let active_plan = plan
            .active(TOKEN_COUNT)
            .map_err(|error| error.to_string())?;
        let active_finish = finish
            .active(layout, TOKEN_COUNT)
            .map_err(|error| error.to_string())?;
        let (expected_hidden, expected_history, expected_state) =
            cpu_recurrent_expectations(&weights, &fixture, &hidden, &packed)?;

        let device = Device::new(0).map_err(|error| format!("open reserved device: {error}"))?;
        let resources =
            RecurrentWitnessResources::build(&device, &weights, &plan, &finish, &hidden)?;
        let deferred = DeferredRecurrent {
            plan: &active_plan,
            weights: &resources.native_weights,
            workspace: resources
                .recurrent_workspace
                .active(active_plan.workspace)
                .map_err(|error| error.to_string())?,
            state: &resources.state,
            input: NativeBufferView::prefix(&resources.input, hidden.len())
                .map_err(|error| error.to_string())?,
            output: NativeBufferView::prefix(&resources.output, hidden.len())
                .map_err(|error| error.to_string())?,
            finish_plan: &active_finish,
            finish_weights: &resources.finish_weights,
            finish_workspace: &resources.finish_workspace,
            stream: &resources.stream,
            numerical_status: &resources.numerical_status,
        };

        // SAFETY: this ignored witness owns every exact checked input, output,
        // state, workspace, weight, status, and stream allocation through its
        // completion check on the operator-reserved device.
        let submitted = unsafe { deferred.submit() };
        if let Err(error) = submitted {
            resources.retain_unconfirmed();
            return Err(error.to_string());
        }
        if let Err(error) = resources.stream.synchronize() {
            resources.retain_unconfirmed();
            return Err(format!("synchronize recurrent rows: {error}"));
        }
        resources
            .numerical_status
            .read_after_synchronization()
            .map_err(|error| error.to_string())?;

        resources.assert_matches(&expected_hidden, &expected_history, &expected_state)
    }

    fn recurrent_witness_input(
        token_count: usize,
        hidden: usize,
    ) -> core::result::Result<Vec<f32>, String> {
        const VALUES: [f32; 3] = [0.25, -0.5, 0.75];
        let elements = token_count
            .checked_mul(hidden)
            .ok_or("recurrent witness input extent overflow")?;
        Ok((0..elements)
            .map(|index| VALUES[index % VALUES.len()])
            .collect())
    }

    fn cpu_recurrent_expectations(
        weights: &Qwen35Weights,
        fixture: &crate::qwen35::tests::Fixture,
        hidden: &[f32],
        packed: &PackedPrefillPlan,
    ) -> core::result::Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
        let mut recurrent = weights
            .recurrent_execution(RECURRENT_BLOCK_LAYER)
            .map_err(|error| error.to_string())?;
        recurrent
            .step_packed_at(hidden, packed)
            .map_err(|error| error.to_string())?;
        let (history, state) = recurrent.transaction_state_for_test();
        let mut oracle = CanonicalHybridOracle::from_fixture(fixture)?;
        let row_width = hidden.len() / TOKEN_COUNT;
        let mut output = Vec::new();
        for row in hidden.chunks_exact(row_width) {
            let row = row.iter().copied().map(f64::from).collect::<Vec<_>>();
            output.extend(oracle.recurrent_block_step(RECURRENT_BLOCK, &row)?);
        }
        Ok((
            output,
            history.iter().copied().map(f64::from).collect(),
            state.iter().copied().map(f64::from).collect(),
        ))
    }

    fn copy_to_host(
        buffer: &hipcore::DeviceBuffer<f32>,
        label: &str,
    ) -> core::result::Result<Vec<f32>, String> {
        let mut values = vec![0.0_f32; buffer.len()];
        buffer
            .copy_to_host(&mut values)
            .map_err(|error| format!("read {label}: {error}"))?;
        Ok(values)
    }
}
