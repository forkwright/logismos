//! One-token native full-attention launch ordering.

use cache::NativePagedAppend;
use hipcore::{DeviceBuffer, Stream};
use snafu::ResultExt;

use super::CompletionResource;
use super::finish::{
    DeferredLayerFinish, LayerFinishPlan, LayerFinishWeights, LayerFinishWorkspace,
};
use super::plan::WorkspacePlan;
use super::resources::{DeviceResources, FullAttentionStep, NativeWorkspace};
use super::weights::{NativeMatrix, NativeWeights};
use crate::Result;
use crate::error::{
    NativeDeviceSnafu, NativeKernelSnafu, NativePagedKvSnafu, NativeSessionStateSnafu,
};

impl CompletionResource for DeviceResources {
    type Error = crate::Error;

    fn synchronize(&mut self) -> Result<()> {
        self.stream.synchronize().context(NativeDeviceSnafu)
    }

    fn validate_after_synchronization(&mut self) -> Result<()> {
        self.numerical_status
            .read_after_synchronization()
            .context(NativeKernelSnafu)
    }
}

/// Borrowed native full-attention launch inputs with no cache-publication authority.
pub(super) struct DeferredFullAttention<'resources> {
    pub(super) weights: &'resources NativeWeights,
    pub(super) workspace: &'resources NativeWorkspace,
    pub(super) finish_plan: &'resources LayerFinishPlan,
    pub(super) finish_weights: &'resources LayerFinishWeights,
    pub(super) finish_workspace: &'resources LayerFinishWorkspace,
    pub(super) plan: WorkspacePlan,
    pub(super) step: FullAttentionStep<'resources>,
    pub(super) stream: &'resources Stream,
    pub(super) numerical_status: &'resources kernels::numerical_status::NativeNumericalStatus,
    pub(super) full_layer: usize,
}

impl DeviceResources {
    /// Submit the complete one-token native full-attention block.
    ///
    /// The caller marks its owning resource guard submitted before invoking
    /// this method and retains the complete bundle until synchronization proves
    /// completion. This method only parks the native KV reservation; it does
    /// not synchronize or publish host logical state.
    ///
    /// # Safety
    ///
    /// The prepared input and all owned resources, including the sticky status
    /// allocation, must remain exclusively owned on this stream's device until
    /// the caller has established completion or retained them after uncertainty.
    /// Checked launches classify their explicit operands and results; this does
    /// not prove math-library internals or device floating-point mode.
    pub(crate) unsafe fn submit_step(&mut self) -> Result<()> {
        let step = self.step.as_ref().ok_or_else(|| {
            NativeSessionStateSnafu {
                rule: "native submission requires a prepared one-token step",
            }
            .build()
        })?;
        let stream = &self.stream;
        // SAFETY: this session owns the cache, stream, and exact one-token
        // append row buffers; all remain live until guard completion.
        let mut append = unsafe { self.kv.begin_append(1, stream) }.context(NativePagedKvSnafu)?;
        let deferred = DeferredFullAttention {
            weights: &self.weights,
            workspace: &self.workspace,
            finish_plan: &self.plan.finish,
            finish_weights: &self.weights.finish,
            finish_workspace: &self.finish_workspace,
            plan: self.plan.workspace,
            step: step.full_attention(),
            stream,
            numerical_status: &self.numerical_status,
            full_layer: 0,
        };
        // SAFETY: submit_step's contract retains the bundle, and this private
        // session supplies the one checked cache append through completion.
        unsafe { deferred.submit(&mut append) }?;
        append.prepare_commit().context(NativePagedKvSnafu)
    }
}

impl DeferredFullAttention<'_> {
    /// Submit one checked full-attention chain without synchronizing or publishing KV.
    ///
    /// # Safety
    ///
    /// The borrowed buffers, append, and sticky status must remain exclusively
    /// owned on this stream through completion. Checked launches classify their
    /// explicit operands and results before the owner publishes cache state.
    pub(super) unsafe fn submit(&self, append: &mut NativePagedAppend<'_>) -> Result<()> {
        // SAFETY: submit's contract retains the exact checked pre-attention
        // buffers and weights through completion.
        unsafe { self.project_input() }?;
        // SAFETY: submit's contract retains the checked Q/K and controls.
        unsafe { self.normalize_and_rotate() }?;
        // SAFETY: submit's contract retains the append and opaque cache spans.
        unsafe { self.append_and_attend(append) }?;
        // SAFETY: submit's contract retains the checked attention projection spans.
        unsafe { self.project_attention() }?;
        let finish = DeferredLayerFinish {
            input: self.step.input,
            attention_projection: &self.workspace.output_projection,
            output: self.step.output,
            plan: self.finish_plan,
            weights: self.finish_weights,
            workspace: self.finish_workspace,
            stream: self.stream,
            numerical_status: self.numerical_status,
        };
        // SAFETY: submit's contract retains the exact input, attention
        // projection, output, finish weights, and finish scratch through completion.
        unsafe { finish.submit() }
    }

    unsafe fn project_input(&self) -> Result<()> {
        let weights = self.weights;
        let workspace = self.workspace;
        let stream = self.stream;
        // SAFETY: this method's caller retains the exact checked spans and
        // sticky status allocation through completion.
        unsafe {
            launch_rms_norm(
                self.plan.hidden_norm,
                self.step.input,
                &weights.input_norm,
                &workspace.hidden,
                stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the checked matrix descriptor and distinct spans remain live.
        unsafe {
            weights.q_gate.launch(
                &workspace.hidden,
                &workspace.q_gate,
                stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the checked matrix descriptor and distinct spans remain live.
        unsafe {
            weights.key.launch(
                &workspace.hidden,
                &workspace.key,
                stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the checked matrix descriptor and distinct spans remain live.
        unsafe {
            weights.value.launch(
                &workspace.hidden,
                &workspace.value,
                stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the checked split geometry and exact spans remain live.
        unsafe {
            launch_split(
                self.plan.split,
                &workspace.q_gate,
                &workspace.query,
                &workspace.gate,
                stream,
                self.numerical_status,
            )
        }
    }

    unsafe fn normalize_and_rotate(&self) -> Result<()> {
        let weights = self.weights;
        let workspace = self.workspace;
        let stream = self.stream;
        // SAFETY: this method's caller retains the exact checked spans and
        // sticky status allocation through completion.
        unsafe {
            launch_rms_norm(
                self.plan.query_norm,
                &workspace.query,
                &weights.query_norm,
                &workspace.normalized_query,
                stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the exact checked K-normalization spans remain live.
        unsafe {
            launch_rms_norm(
                self.plan.key_norm,
                &workspace.key,
                &weights.key_norm,
                &workspace.normalized_key,
                stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the controls and rotated Q span are distinct exact buffers.
        unsafe {
            launch_rotary(
                self.plan.query_rotary,
                &workspace.normalized_query,
                self.step.cosine,
                self.step.sine,
                stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the controls and rotated K span are distinct exact buffers.
        unsafe {
            launch_rotary(
                self.plan.key_rotary,
                &workspace.normalized_key,
                self.step.cosine,
                self.step.sine,
                stream,
                self.numerical_status,
            )
        }
    }

    unsafe fn append_and_attend(&self, append: &mut NativePagedAppend<'_>) -> Result<()> {
        let workspace = self.workspace;
        // SAFETY: this method's caller retains the exact native row spans and
        // append through completion on the ordered stream.
        unsafe {
            append.write_layer_row(
                self.full_layer,
                0,
                &workspace.normalized_key,
                &workspace.value,
                self.stream,
            )
        }
        .context(NativePagedKvSnafu)?;
        let layer = append
            .layer_kv(self.full_layer)
            .context(NativePagedKvSnafu)?;
        if layer.tokens() != self.step.attention.logical().visible_tokens() {
            return NativeSessionStateSnafu {
                rule: "native staged KV visibility must match the checked attention plan",
            }
            .fail();
        }
        // SAFETY: the opaque cache view retains K/V/table spans, while query
        // and output are separate owned exact spans on its stream.
        unsafe {
            layer.launch_paged_decode_checked(
                self.step.attention,
                &workspace.normalized_query,
                &workspace.attention,
                self.stream,
                self.numerical_status,
            )
        }
        .context(NativePagedKvSnafu)
    }

    unsafe fn project_attention(&self) -> Result<()> {
        let workspace = self.workspace;
        // SAFETY: this method's caller retains the checked exact elementwise
        // spans and sticky status allocation through completion.
        unsafe {
            launch_sigmoid_mul(
                self.plan.gate,
                &workspace.attention,
                &workspace.gate,
                &workspace.gated,
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the checked matrix descriptor and distinct spans remain live.
        unsafe {
            self.weights.output.launch(
                &workspace.gated,
                &workspace.output_projection,
                self.stream,
                self.numerical_status,
            )
        }
    }
}

impl NativeMatrix {
    /// Launch one verified row-major projection into an owned exact output span.
    ///
    /// # Safety
    ///
    /// `input` and `output` must be non-overlapping device spans on `stream`'s
    /// device with its sticky status allocation through completion. The checked
    /// kernel classifies explicit operands and results before publication.
    pub(super) unsafe fn launch(
        &self,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        stream: &Stream,
        numerical_status: &kernels::numerical_status::NativeNumericalStatus,
    ) -> Result<()> {
        // SAFETY: the caller upholds the native row-GEMV device-span and
        // status-lifetime contract for this verified descriptor.
        unsafe {
            kernels::row_gemv::launch_row_gemv_f32_checked(
                self.shape,
                self.bytes.as_device_ptr(),
                self.bytes.len(),
                input.as_device_ptr().cast_const(),
                input.len(),
                output.as_device_ptr(),
                output.len(),
                stream,
                numerical_status,
            )
        }
        .context(NativeKernelSnafu)
    }
}

/// # Safety
///
/// The three spans must be distinct exact buffers on `stream`'s device and
/// retain the status allocation through completion. The checked kernel records
/// explicit operand and arithmetic faults for the post-sync owner to classify.
pub(super) unsafe fn launch_rms_norm(
    plan: kernels::decoder_ops::RmsNormF32Plan,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    stream: &Stream,
    numerical_status: &kernels::numerical_status::NativeNumericalStatus,
) -> Result<()> {
    // SAFETY: caller establishes exact spans and retains the status allocation.
    unsafe {
        kernels::decoder_ops::launch_rms_norm_f32_checked(
            plan,
            input.as_device_ptr().cast_const(),
            input.len(),
            weight.as_device_ptr().cast_const(),
            weight.len(),
            output.as_device_ptr(),
            output.len(),
            stream,
            numerical_status,
        )
    }
    .context(NativeKernelSnafu)
}

/// # Safety
///
/// `values` must be exclusive, while coefficient spans remain immutable and
/// all buffers, including `numerical_status`, stay live on `stream`'s device
/// through completion.
unsafe fn launch_rotary(
    plan: kernels::decoder_ops::RotaryHalfSplitF32Plan,
    values: &DeviceBuffer<f32>,
    cosine: &DeviceBuffer<f32>,
    sine: &DeviceBuffer<f32>,
    stream: &Stream,
    numerical_status: &kernels::numerical_status::NativeNumericalStatus,
) -> Result<()> {
    // SAFETY: caller establishes exact non-aliasing spans and status lifetime.
    unsafe {
        kernels::decoder_ops::launch_rotary_half_split_f32_in_place_checked(
            plan,
            values.as_device_ptr(),
            values.len(),
            cosine.as_device_ptr().cast_const(),
            cosine.len(),
            sine.as_device_ptr().cast_const(),
            sine.len(),
            stream,
            numerical_status,
        )
    }
    .context(NativeKernelSnafu)
}

/// # Safety
///
/// The input and two output spans must be distinct exact buffers on
/// `stream`'s device and retain `numerical_status` through completion.
unsafe fn launch_split(
    plan: kernels::decoder_ops::SplitQGateF32Plan,
    input: &DeviceBuffer<f32>,
    query: &DeviceBuffer<f32>,
    gate: &DeviceBuffer<f32>,
    stream: &Stream,
    numerical_status: &kernels::numerical_status::NativeNumericalStatus,
) -> Result<()> {
    // SAFETY: caller establishes exact non-aliasing spans and status lifetime.
    unsafe {
        kernels::decoder_ops::launch_split_q_gate_f32_checked(
            plan,
            input.as_device_ptr().cast_const(),
            input.len(),
            query.as_device_ptr(),
            query.len(),
            gate.as_device_ptr(),
            gate.len(),
            stream,
            numerical_status,
        )
    }
    .context(NativeKernelSnafu)
}

/// # Safety
///
/// The two inputs and output must be distinct exact spans on `stream`'s device
/// with the shared status allocation through completion.
pub(super) unsafe fn launch_sigmoid_mul(
    plan: kernels::decoder_ops::ElementwiseF32Plan,
    value: &DeviceBuffer<f32>,
    gate: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    stream: &Stream,
    numerical_status: &kernels::numerical_status::NativeNumericalStatus,
) -> Result<()> {
    // SAFETY: caller establishes exact non-aliasing spans and status lifetime.
    unsafe {
        kernels::decoder_ops::sigmoid_mul_checked(
            plan,
            value.as_device_ptr().cast_const(),
            value.len(),
            gate.as_device_ptr().cast_const(),
            gate.len(),
            output.as_device_ptr(),
            output.len(),
            stream,
            numerical_status,
        )
    }
    .context(NativeKernelSnafu)
}

/// # Safety
///
/// The two inputs and output must be distinct exact spans on `stream`'s device
/// with the shared status allocation through completion.
pub(super) unsafe fn launch_silu_mul(
    plan: kernels::decoder_ops::ElementwiseF32Plan,
    gate: &DeviceBuffer<f32>,
    up: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    stream: &Stream,
    numerical_status: &kernels::numerical_status::NativeNumericalStatus,
) -> Result<()> {
    // SAFETY: caller establishes exact non-aliasing spans and status lifetime.
    unsafe {
        kernels::decoder_ops::silu_mul_checked(
            plan,
            gate.as_device_ptr().cast_const(),
            gate.len(),
            up.as_device_ptr().cast_const(),
            up.len(),
            output.as_device_ptr(),
            output.len(),
            stream,
            numerical_status,
        )
    }
    .context(NativeKernelSnafu)
}

/// # Safety
///
/// The two inputs and output must be distinct exact spans on `stream`'s device
/// with the shared status allocation through completion.
pub(super) unsafe fn launch_residual(
    plan: kernels::decoder_ops::ElementwiseF32Plan,
    left: &DeviceBuffer<f32>,
    right: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    stream: &Stream,
    numerical_status: &kernels::numerical_status::NativeNumericalStatus,
) -> Result<()> {
    // SAFETY: caller establishes exact non-aliasing spans and status lifetime.
    unsafe {
        kernels::decoder_ops::residual_add_checked(
            plan,
            left.as_device_ptr().cast_const(),
            left.len(),
            right.as_device_ptr().cast_const(),
            right.len(),
            output.as_device_ptr(),
            output.len(),
            stream,
            numerical_status,
        )
    }
    .context(NativeKernelSnafu)
}
