//! One-token native full-attention launch ordering.

use cache::NativePagedAppend;
use hipcore::Stream;
use snafu::ResultExt;

use super::CompletionResource;
use super::finish::{
    ActiveLayerFinishPlan, DeferredLayerFinish, LayerFinishWeights, LayerFinishWorkspace,
};
use super::plan::WorkspacePlan;
use super::resources::{
    DeviceResources, FullAttentionStep, NativeBufferView, NativeWorkspace, checked_buffer_window,
};
use super::weights::NativeWeights;
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
    pub(super) finish_plan: &'resources ActiveLayerFinishPlan,
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
        let finish_plan = self.plan.active_finish(1)?;
        // SAFETY: this session owns the cache, stream, and exact one-token
        // append row buffers; all remain live until guard completion.
        let mut append = unsafe { self.kv.begin_append(1, stream) }.context(NativePagedKvSnafu)?;
        let deferred = DeferredFullAttention {
            weights: &self.weights,
            workspace: &self.workspace,
            finish_plan: &finish_plan,
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
            input: NativeBufferView::prefix(self.step.input, self.plan.hidden)?,
            attention_projection: NativeBufferView::prefix(
                &self.workspace.output_projection,
                self.plan.output_projection,
            )?,
            output: NativeBufferView::prefix(self.step.output, self.plan.hidden)?,
            plan: self.finish_plan,
            weights: self.finish_weights,
            workspace: self.finish_workspace.active(self.finish_plan.workspace)?,
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
        let input = NativeBufferView::prefix(self.step.input, self.plan.hidden)?;
        let hidden = NativeBufferView::prefix(&workspace.hidden, self.plan.hidden)?;
        let q_gate = NativeBufferView::prefix(&workspace.q_gate, self.plan.q_gate)?;
        let key = NativeBufferView::prefix(&workspace.key, self.plan.key)?;
        let value = NativeBufferView::prefix(&workspace.value, self.plan.value)?;
        // SAFETY: this method's caller retains the exact checked spans and
        // sticky status allocation through completion.
        unsafe {
            launch_rms_norm_view(
                self.plan.hidden_norm,
                input,
                NativeBufferView::prefix(&weights.input_norm, weights.input_norm.len())?,
                hidden,
                stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the checked matrix descriptor and distinct spans remain live.
        unsafe {
            weights.q_gate.launch_rows_view(
                hidden,
                q_gate,
                self.plan.hidden_norm.rows(),
                stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the checked matrix descriptor and distinct spans remain live.
        unsafe {
            weights.key.launch_rows_view(
                hidden,
                key,
                self.plan.hidden_norm.rows(),
                stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the checked matrix descriptor and distinct spans remain live.
        unsafe {
            weights.value.launch_rows_view(
                hidden,
                value,
                self.plan.hidden_norm.rows(),
                stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the checked split geometry and exact spans remain live.
        unsafe {
            launch_split_view(
                self.plan.split,
                q_gate,
                NativeBufferView::prefix(&workspace.query, self.plan.query)?,
                NativeBufferView::prefix(&workspace.gate, self.plan.gate_values)?,
                stream,
                self.numerical_status,
            )
        }
    }

    unsafe fn normalize_and_rotate(&self) -> Result<()> {
        let weights = self.weights;
        let workspace = self.workspace;
        let stream = self.stream;
        let query = NativeBufferView::prefix(&workspace.query, self.plan.query)?;
        let normalized_query =
            NativeBufferView::prefix(&workspace.normalized_query, self.plan.normalized_query)?;
        let key = NativeBufferView::prefix(&workspace.key, self.plan.key)?;
        let normalized_key =
            NativeBufferView::prefix(&workspace.normalized_key, self.plan.normalized_key)?;
        // SAFETY: this method's caller retains the exact checked spans and
        // sticky status allocation through completion.
        unsafe {
            launch_rms_norm_view(
                self.plan.query_norm,
                query,
                NativeBufferView::prefix(&weights.query_norm, weights.query_norm.len())?,
                normalized_query,
                stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the exact checked K-normalization spans remain live.
        unsafe {
            launch_rms_norm_view(
                self.plan.key_norm,
                key,
                NativeBufferView::prefix(&weights.key_norm, weights.key_norm.len())?,
                normalized_key,
                stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the controls and rotated Q span are distinct exact buffers.
        unsafe { self.rotate_rows(normalized_query, normalized_key, stream) }
    }

    unsafe fn rotate_rows(
        &self,
        normalized_query: NativeBufferView<'_, f32>,
        normalized_key: NativeBufferView<'_, f32>,
        stream: &Stream,
    ) -> Result<()> {
        let token_count = self.plan.hidden_norm.rows();
        let query_row_elements = self.plan.query_rotary.elements();
        let key_row_elements = self.plan.key_rotary.elements();
        let coefficient_elements = self.plan.query_rotary.coefficient_elements();
        for token in 0..token_count {
            let query_offset = row_offset(token, query_row_elements, "native query rotary row")?;
            let key_offset = row_offset(token, key_row_elements, "native key rotary row")?;
            let coefficient_offset =
                row_offset(token, coefficient_elements, "native MRoPE coefficient row")?;
            checked_buffer_window(normalized_query.len(), query_offset, query_row_elements)?;
            checked_buffer_window(normalized_key.len(), key_offset, key_row_elements)?;
            let query = NativeBufferView::window(
                &self.workspace.normalized_query,
                query_offset,
                query_row_elements,
            )?;
            let key = NativeBufferView::window(
                &self.workspace.normalized_key,
                key_offset,
                key_row_elements,
            )?;
            let cosine = NativeBufferView::window(
                self.step.cosine,
                coefficient_offset,
                coefficient_elements,
            )?;
            let sine =
                NativeBufferView::window(self.step.sine, coefficient_offset, coefficient_elements)?;
            // SAFETY: each row view is a checked disjoint window and controls
            // are immutable token-major coefficients retained through completion.
            unsafe {
                launch_rotary_view(
                    self.plan.query_rotary,
                    query,
                    cosine,
                    sine,
                    stream,
                    self.numerical_status,
                )
            }?;
            // SAFETY: K is a separate checked row window with the same controls.
            unsafe {
                launch_rotary_view(
                    self.plan.key_rotary,
                    key,
                    cosine,
                    sine,
                    stream,
                    self.numerical_status,
                )
            }?;
        }
        Ok(())
    }

    unsafe fn append_and_attend(&self, append: &mut NativePagedAppend<'_>) -> Result<()> {
        let workspace = self.workspace;
        // SAFETY: the opaque append owner validates its reservation-sized
        // active prefixes inside these retained capacity allocations.
        unsafe {
            append.write_layer_rows(
                self.full_layer,
                &workspace.normalized_key,
                &workspace.value,
                self.stream,
            )
        }
        .context(NativePagedKvSnafu)?;
        let layer = append
            .layer_kv(self.full_layer)
            .context(NativePagedKvSnafu)?;
        if layer.tokens() != self.step.attention.visible_tokens() {
            return NativeSessionStateSnafu {
                rule: "native staged KV visibility must match the checked attention plan",
            }
            .fail();
        }
        // SAFETY: the opaque cache view retains K/V/table spans, while query
        // and output are separate owned exact spans on its stream.
        unsafe {
            layer.launch_paged_prefill_checked(
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
        let attention = NativeBufferView::prefix(&workspace.attention, self.plan.attention)?;
        let gate = NativeBufferView::prefix(&workspace.gate, self.plan.gate_values)?;
        let gated = NativeBufferView::prefix(&workspace.gated, self.plan.gated)?;
        let output_projection =
            NativeBufferView::prefix(&workspace.output_projection, self.plan.output_projection)?;
        // SAFETY: this method's caller retains the checked exact elementwise
        // spans and sticky status allocation through completion.
        unsafe {
            launch_sigmoid_mul_view(
                self.plan.gate,
                attention,
                gate,
                gated,
                self.stream,
                self.numerical_status,
            )
        }?;
        // SAFETY: the checked matrix descriptor and distinct spans remain live.
        unsafe {
            self.weights.output.launch_rows_view(
                gated,
                output_projection,
                self.plan.hidden_norm.rows(),
                self.stream,
                self.numerical_status,
            )
        }
    }
}

fn row_offset(row: usize, row_elements: usize, context: &'static str) -> Result<usize> {
    row.checked_mul(row_elements)
        .ok_or_else(|| crate::error::ArithmeticOverflowSnafu { context }.build())
}

/// # Safety
///
/// The three views must be distinct exact spans on `stream`'s device and
/// retain the status allocation through completion.
pub(super) unsafe fn launch_rms_norm_view(
    plan: kernels::decoder_ops::RmsNormF32Plan,
    input: NativeBufferView<'_, f32>,
    weight: NativeBufferView<'_, f32>,
    output: NativeBufferView<'_, f32>,
    stream: &Stream,
    numerical_status: &kernels::numerical_status::NativeNumericalStatus,
) -> Result<()> {
    // SAFETY: caller establishes exact non-aliasing views and status lifetime.
    unsafe {
        kernels::decoder_ops::launch_rms_norm_f32_checked(
            plan,
            input.as_const_ptr(),
            input.len(),
            weight.as_const_ptr(),
            weight.len(),
            output.as_mut_ptr(),
            output.len(),
            stream,
            numerical_status,
        )
    }
    .context(NativeKernelSnafu)
}

/// # Safety
///
/// `values` must be exclusively writable, coefficient views immutable, and
/// all views must remain live on `stream`'s device through completion.
unsafe fn launch_rotary_view(
    plan: kernels::decoder_ops::RotaryHalfSplitF32Plan,
    values: NativeBufferView<'_, f32>,
    cosine: NativeBufferView<'_, f32>,
    sine: NativeBufferView<'_, f32>,
    stream: &Stream,
    numerical_status: &kernels::numerical_status::NativeNumericalStatus,
) -> Result<()> {
    // SAFETY: caller establishes exact non-aliasing views and status lifetime.
    unsafe {
        kernels::decoder_ops::launch_rotary_half_split_f32_in_place_checked(
            plan,
            values.as_mut_ptr(),
            values.len(),
            cosine.as_const_ptr(),
            cosine.len(),
            sine.as_const_ptr(),
            sine.len(),
            stream,
            numerical_status,
        )
    }
    .context(NativeKernelSnafu)
}

/// # Safety
///
/// The input and two output views must be distinct exact spans on `stream`'s
/// device and retain `numerical_status` through completion.
unsafe fn launch_split_view(
    plan: kernels::decoder_ops::SplitQGateF32Plan,
    input: NativeBufferView<'_, f32>,
    query: NativeBufferView<'_, f32>,
    gate: NativeBufferView<'_, f32>,
    stream: &Stream,
    numerical_status: &kernels::numerical_status::NativeNumericalStatus,
) -> Result<()> {
    // SAFETY: caller establishes exact non-aliasing views and status lifetime.
    unsafe {
        kernels::decoder_ops::launch_split_q_gate_f32_checked(
            plan,
            input.as_const_ptr(),
            input.len(),
            query.as_mut_ptr(),
            query.len(),
            gate.as_mut_ptr(),
            gate.len(),
            stream,
            numerical_status,
        )
    }
    .context(NativeKernelSnafu)
}

/// # Safety
///
/// The two input and output views must be distinct exact spans on `stream`'s
/// device with the shared status allocation through completion.
pub(super) unsafe fn launch_sigmoid_mul_view(
    plan: kernels::decoder_ops::ElementwiseF32Plan,
    value: NativeBufferView<'_, f32>,
    gate: NativeBufferView<'_, f32>,
    output: NativeBufferView<'_, f32>,
    stream: &Stream,
    numerical_status: &kernels::numerical_status::NativeNumericalStatus,
) -> Result<()> {
    // SAFETY: caller establishes exact non-aliasing views and status lifetime.
    unsafe {
        kernels::decoder_ops::sigmoid_mul_checked(
            plan,
            value.as_const_ptr(),
            value.len(),
            gate.as_const_ptr(),
            gate.len(),
            output.as_mut_ptr(),
            output.len(),
            stream,
            numerical_status,
        )
    }
    .context(NativeKernelSnafu)
}

/// # Safety
///
/// The two input and output views must be distinct exact spans on `stream`'s
/// device with the shared status allocation through completion.
pub(super) unsafe fn launch_silu_mul_view(
    plan: kernels::decoder_ops::ElementwiseF32Plan,
    gate: NativeBufferView<'_, f32>,
    up: NativeBufferView<'_, f32>,
    output: NativeBufferView<'_, f32>,
    stream: &Stream,
    numerical_status: &kernels::numerical_status::NativeNumericalStatus,
) -> Result<()> {
    // SAFETY: caller establishes exact non-aliasing views and status lifetime.
    unsafe {
        kernels::decoder_ops::silu_mul_checked(
            plan,
            gate.as_const_ptr(),
            gate.len(),
            up.as_const_ptr(),
            up.len(),
            output.as_mut_ptr(),
            output.len(),
            stream,
            numerical_status,
        )
    }
    .context(NativeKernelSnafu)
}

/// # Safety
///
/// The two input and output views must be distinct exact spans on `stream`'s
/// device with the shared status allocation through completion.
pub(super) unsafe fn launch_residual_view(
    plan: kernels::decoder_ops::ElementwiseF32Plan,
    left: NativeBufferView<'_, f32>,
    right: NativeBufferView<'_, f32>,
    output: NativeBufferView<'_, f32>,
    stream: &Stream,
    numerical_status: &kernels::numerical_status::NativeNumericalStatus,
) -> Result<()> {
    // SAFETY: caller establishes exact non-aliasing views and status lifetime.
    unsafe {
        kernels::decoder_ops::residual_add_checked(
            plan,
            left.as_const_ptr(),
            left.len(),
            right.as_const_ptr(),
            right.len(),
            output.as_mut_ptr(),
            output.len(),
            stream,
            numerical_status,
        )
    }
    .context(NativeKernelSnafu)
}
