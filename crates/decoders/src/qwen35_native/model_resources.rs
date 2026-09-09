//! Owned native resources for one whole Qwen3.5 main-model step.

use cache::NativePagedKvPool;
use hipcore::{Device, DeviceBuffer, Stream};
use snafu::ResultExt;
use std::sync::Arc;

use super::CompletionResource;
use super::custody::NativeBufferSink;
use super::dispatch::{DeferredFullAttention, launch_rms_norm};
use super::finish::{LayerFinishWeights, LayerFinishWorkspace};
use super::model_plan::{DeviceModelPlan, NativeBlockPlan};
use super::model_step::ModelTokenPlan;
use super::recurrent::{
    DeferredRecurrent, NativeRecurrentState, NativeRecurrentWeights, NativeRecurrentWorkspace,
};
use super::resources::{FullAttentionStep, NativeWorkspace, native_mrope_controls};
use super::weights::{NativeMatrix, NativeWeights, f32_parameter_buffer};
use crate::error::{
    ArithmeticOverflowSnafu, ExecutionAllocationSnafu, NativeDeviceSnafu, NativeKernelSnafu,
    NativePagedKvSnafu, NativeSessionStateSnafu,
};
use crate::{Qwen35Weights, Result};

struct ModelStep {
    token: ModelTokenPlan,
    logits: DeviceBuffer<f32>,
    cosine: Option<DeviceBuffer<f32>>,
    sine: Option<DeviceBuffer<f32>>,
}

enum NativeModelLayer {
    Full(Box<NativeWeights>),
    Recurrent(Box<NativeRecurrentLayerWeights>),
}

struct NativeRecurrentLayerWeights {
    weights: NativeRecurrentWeights,
    finish: LayerFinishWeights,
}

enum NativeSessionLayer {
    Full,
    Recurrent(NativeRecurrentState),
}

/// Immutable native uploads retained by every session created from one model.
pub(super) struct NativeResidentModelResources {
    verified_weights: Qwen35Weights,
    plan: DeviceModelPlan,
    device: Device,
    embedding: NativeMatrix,
    output: NativeMatrix,
    output_norm: DeviceBuffer<f32>,
    layers: Vec<NativeModelLayer>,
}

/// One non-cloneable mutable native session retaining submitted work through completion.
pub(super) struct ModelSessionResources {
    model: Arc<NativeResidentModelResources>,
    plan: DeviceModelPlan,
    kv: Option<NativePagedKvPool>,
    stream: Stream,
    numerical_status: kernels::numerical_status::NativeNumericalStatus,
    full_workspace: Option<NativeWorkspace>,
    recurrent_workspace: Option<NativeRecurrentWorkspace>,
    finish_workspace: LayerFinishWorkspace,
    hidden_a: DeviceBuffer<f32>,
    hidden_b: DeviceBuffer<f32>,
    final_normalized: DeviceBuffer<f32>,
    layers: Vec<NativeSessionLayer>,
    step: Option<ModelStep>,
    position: usize,
}

impl NativeResidentModelResources {
    pub(super) fn new(
        weights: &Qwen35Weights,
        plan: DeviceModelPlan,
        device: &Device,
    ) -> Result<Self> {
        let _ = plan.bytes.total()?;
        let embedding = NativeMatrix::upload(weights, &plan.embedding, device)?;
        let output = NativeMatrix::upload(weights, &plan.output, device)?;
        let output_norm = f32_parameter_buffer(weights, &plan.output_norm, device)?;
        let layers = upload_resident_layers(weights, &plan, device)?;
        Ok(Self {
            verified_weights: weights.clone(),
            plan,
            device: device.clone(),
            embedding,
            output,
            output_norm,
            layers,
        })
    }

    pub(super) fn plan_session(&self, max_context: usize) -> Result<DeviceModelPlan> {
        derive_session_plan(&self.verified_weights, &self.plan, max_context)
    }

    pub(super) const fn context_ceiling(&self) -> usize {
        self.plan.layout.max_context()
    }

    pub(super) fn into_buffer_sink(self, sink: &mut impl NativeBufferSink) {
        self.embedding.into_buffer_sink(sink);
        self.output.into_buffer_sink(sink);
        sink.push_f32(self.output_norm);
        for layer in self.layers {
            layer.into_buffer_sink(sink);
        }
    }
}

impl ModelSessionResources {
    pub(super) fn new(
        model: Arc<NativeResidentModelResources>,
        plan: DeviceModelPlan,
    ) -> Result<Self> {
        let stream = Stream::new(&model.device).context(NativeDeviceSnafu)?;
        let numerical_status = kernels::numerical_status::NativeNumericalStatus::new(&model.device)
            .context(NativeKernelSnafu)?;
        let kv = plan
            .kv
            .map(|plan| NativePagedKvPool::new(plan, &model.device))
            .transpose()
            .context(NativePagedKvSnafu)?;
        let full_workspace = plan
            .full_workspace
            .as_ref()
            .map(|workspace| NativeWorkspace::new(workspace, &model.device))
            .transpose()?;
        let recurrent_workspace = plan
            .recurrent_workspace
            .as_ref()
            .map(|workspace| NativeRecurrentWorkspace::new(workspace, &model.device))
            .transpose()?;
        let finish_workspace = LayerFinishWorkspace::new(plan.finish_workspace, &model.device)?;
        let hidden_a =
            DeviceBuffer::alloc(&model.device, plan.layout.hidden).context(NativeDeviceSnafu)?;
        let hidden_b =
            DeviceBuffer::alloc(&model.device, plan.layout.hidden).context(NativeDeviceSnafu)?;
        let final_normalized = DeviceBuffer::alloc(&model.device, plan.output_rms.elements())
            .context(NativeDeviceSnafu)?;
        let layers = allocate_session_layers(&plan, &model.device)?;
        Ok(Self {
            model,
            plan,
            kv,
            stream,
            numerical_status,
            full_workspace,
            recurrent_workspace,
            finish_workspace,
            hidden_a,
            hidden_b,
            final_normalized,
            layers,
            step: None,
            position: 0,
        })
    }

    /// Transfer this session's original buffers and return its stream plus resident owner.
    ///
    /// The returned resident `Arc` must remain in the pending or quarantined
    /// teardown custody until the returned stream has a terminal completion
    /// outcome. It is not proof that the stream is quiescent or that any HIP
    /// allocation has been released.
    pub(super) fn into_buffer_sink(
        self,
        sink: &mut impl NativeBufferSink,
    ) -> (Stream, Arc<NativeResidentModelResources>) {
        let Self {
            model,
            plan,
            kv,
            stream,
            numerical_status,
            full_workspace,
            recurrent_workspace,
            finish_workspace,
            hidden_a,
            hidden_b,
            final_normalized,
            layers,
            step,
            position: _,
        } = self;
        drop(plan);
        if let Some(kv) = kv {
            let (keys, values, table) = kv.into_buffers().into_parts();
            sink.push_f32(keys);
            sink.push_f32(values);
            sink.push_u32(table);
        }
        sink.push_u32(numerical_status.into_buffer());
        if let Some(workspace) = full_workspace {
            workspace.into_buffer_sink(sink);
        }
        if let Some(workspace) = recurrent_workspace {
            workspace.into_buffer_sink(sink);
        }
        finish_workspace.into_buffer_sink(sink);
        sink.push_f32(hidden_a);
        sink.push_f32(hidden_b);
        sink.push_f32(final_normalized);
        for layer in layers {
            layer.into_buffer_sink(sink);
        }
        if let Some(step) = step {
            step.into_buffer_sink(sink);
        }
        (stream, model)
    }

    pub(super) fn prepare_step(&mut self, token: u32) -> Result<()> {
        if self.step.is_some() {
            return NativeSessionStateSnafu {
                rule: "native model step requires no pending output",
            }
            .fail();
        }
        let token = ModelTokenPlan::from_model(&self.plan, self.position, token)?;
        let (cosine, sine) = self.prepare_attention_controls(token)?;
        self.step = Some(ModelStep {
            token,
            logits: DeviceBuffer::alloc(self.stream.device(), self.model.output.shape.rows())
                .context(NativeDeviceSnafu)?,
            cosine,
            sine,
        });
        Ok(())
    }

    /// Submit one token through the complete artifact-ordered native main model.
    ///
    /// # Safety
    ///
    /// The complete owned bundle, including serialized weights, controls,
    /// staged K/V, recurrent state, input lookup output, and logits, remains
    /// exclusively owned on this ordered stream until completion is proved.
    /// Every native primitive receives this session's sticky checked-status
    /// allocation. Its post-sync read precedes all logical publication.
    #[expect(
        clippy::too_many_lines,
        reason = "one ordered native submission must retain the full model transaction and its single KV append"
    )]
    pub(super) unsafe fn submit_step(&mut self) -> Result<()> {
        let step = self.step.as_ref().ok_or_else(|| {
            NativeSessionStateSnafu {
                rule: "native model submission requires a prepared step",
            }
            .build()
        })?;
        // SAFETY: this bundle owns the exact serialized embedding matrix,
        // first hidden row, and ordered stream through completion.
        unsafe {
            kernels::row_gemv::launch_row_decode_f32_checked(
                step.token.embedding,
                self.model.embedding.bytes.as_device_ptr(),
                self.model.embedding.bytes.len(),
                self.hidden_a.as_device_ptr(),
                self.hidden_a.len(),
                &self.stream,
                &self.numerical_status,
            )
        }
        .context(NativeKernelSnafu)?;

        let mut append = self
            .kv
            .as_mut()
            .map(|pool| {
                // SAFETY: this model owns the pool and its exact ordered stream through completion.
                unsafe { pool.begin_append(1, &self.stream) }
            })
            .transpose()
            .context(NativePagedKvSnafu)?;
        let (mut input, mut output) = (&self.hidden_a, &self.hidden_b);
        let mut full_layer = 0_usize;
        for ((plan, layer), session_layer) in self
            .plan
            .layers
            .iter()
            .zip(&self.model.layers)
            .zip(&self.layers)
        {
            match (plan, layer, session_layer) {
                (
                    NativeBlockPlan::Full(plan),
                    NativeModelLayer::Full(weights),
                    NativeSessionLayer::Full,
                ) => {
                    let append = append.as_mut().ok_or_else(|| {
                        NativeSessionStateSnafu {
                            rule: "native full-attention block requires one model KV append",
                        }
                        .build()
                    })?;
                    let workspace = self.full_workspace.as_ref().ok_or_else(|| {
                        NativeSessionStateSnafu {
                            rule: "native full-attention block requires model full workspace",
                        }
                        .build()
                    })?;
                    let cosine = step.cosine.as_ref().ok_or_else(|| {
                        NativeSessionStateSnafu {
                            rule: "native full-attention block requires prepared MRoPE cosine controls",
                        }
                        .build()
                    })?;
                    let sine = step.sine.as_ref().ok_or_else(|| {
                        NativeSessionStateSnafu {
                            rule: "native full-attention block requires prepared MRoPE sine controls",
                        }
                        .build()
                    })?;
                    let attention = step.token.attention.ok_or_else(|| {
                        NativeSessionStateSnafu {
                            rule: "native full-attention block requires prepared paged attention controls",
                        }
                        .build()
                    })?;
                    let deferred = DeferredFullAttention {
                        weights,
                        workspace,
                        finish_plan: &plan.finish,
                        finish_weights: &weights.finish,
                        finish_workspace: &self.finish_workspace,
                        plan: plan.workspace,
                        step: FullAttentionStep {
                            input,
                            output,
                            cosine,
                            sine,
                            attention,
                        },
                        stream: &self.stream,
                        numerical_status: &self.numerical_status,
                        full_layer,
                    };
                    // SAFETY: the enclosing submission owns every borrowed span through completion.
                    unsafe { deferred.submit(append) }?;
                    full_layer = full_layer.checked_add(1).ok_or_else(|| {
                        ArithmeticOverflowSnafu {
                            context: "native full-attention layer cursor",
                        }
                        .build()
                    })?;
                }
                (
                    NativeBlockPlan::Recurrent { plan, finish },
                    NativeModelLayer::Recurrent(resources),
                    NativeSessionLayer::Recurrent(state),
                ) => {
                    let workspace = self.recurrent_workspace.as_ref().ok_or_else(|| {
                        NativeSessionStateSnafu {
                            rule: "native recurrent block requires model recurrent workspace",
                        }
                        .build()
                    })?;
                    let deferred = DeferredRecurrent {
                        plan,
                        weights: &resources.weights,
                        workspace,
                        state,
                        input,
                        output,
                        finish_plan: finish,
                        finish_weights: &resources.finish,
                        finish_workspace: &self.finish_workspace,
                        stream: &self.stream,
                        numerical_status: &self.numerical_status,
                    };
                    // SAFETY: the enclosing submission retains all staged recurrent state through completion.
                    unsafe { deferred.submit() }?;
                }
                _ => {
                    return NativeSessionStateSnafu {
                        rule: "native model resource roles must exactly bind planned block roles",
                    }
                    .fail();
                }
            }
            core::mem::swap(&mut input, &mut output);
        }
        // SAFETY: this model owns the exact final hidden, norm, and logits spans through completion.
        unsafe {
            launch_rms_norm(
                self.plan.output_rms,
                input,
                &self.model.output_norm,
                &self.final_normalized,
                &self.stream,
                &self.numerical_status,
            )
        }?;
        // SAFETY: the verified output matrix and exact final/logit spans remain owned through completion.
        unsafe {
            self.model.output.launch(
                &self.final_normalized,
                &step.logits,
                &self.stream,
                &self.numerical_status,
            )
        }?;
        if let Some(append) = append {
            append.prepare_commit().context(NativePagedKvSnafu)?;
        }
        Ok(())
    }

    pub(super) fn publish_completed(&mut self) -> Result<DeviceBuffer<f32>> {
        let step = self.step.take().ok_or_else(|| {
            NativeSessionStateSnafu {
                rule: "native model publication requires one prepared step",
            }
            .build()
        })?;
        let next_position = step.token.next_position;
        if let Some(kv) = self.kv.as_mut() {
            // SAFETY: the resource owner synchronized this exact stream; every fallible local check and output extraction precedes host-ledger publication.
            unsafe {
                kv.commit_prepared_after_completion()
                    .context(NativePagedKvSnafu)?;
            }
        }
        for layer in &mut self.layers {
            if let NativeSessionLayer::Recurrent(state) = layer {
                state.publish_completed();
            }
        }
        self.position = next_position;
        Ok(step.logits)
    }

    fn prepare_attention_controls(
        &self,
        token: ModelTokenPlan,
    ) -> Result<(Option<DeviceBuffer<f32>>, Option<DeviceBuffer<f32>>)> {
        if token.attention.is_none() {
            return Ok((None, None));
        }
        let Some(workspace) = self.plan.full_workspace else {
            return NativeSessionStateSnafu {
                rule: "native model paged attention requires full-attention workspace",
            }
            .fail();
        };
        let (cosine, sine) = native_mrope_controls(
            self.plan.layout.text_mrope(),
            self.position,
            workspace.query_rotary.coefficient_elements(),
        )?;
        Ok((
            Some(
                DeviceBuffer::from_host(self.stream.device(), &cosine)
                    .context(NativeDeviceSnafu)?,
            ),
            Some(DeviceBuffer::from_host(self.stream.device(), &sine).context(NativeDeviceSnafu)?),
        ))
    }
}

impl ModelStep {
    fn into_buffer_sink(self, sink: &mut impl NativeBufferSink) {
        sink.push_f32(self.logits);
        if let Some(cosine) = self.cosine {
            sink.push_f32(cosine);
        }
        if let Some(sine) = self.sine {
            sink.push_f32(sine);
        }
    }
}

impl NativeModelLayer {
    fn into_buffer_sink(self, sink: &mut impl NativeBufferSink) {
        match self {
            Self::Full(weights) => weights.into_buffer_sink(sink),
            Self::Recurrent(weights) => weights.into_buffer_sink(sink),
        }
    }
}

impl NativeRecurrentLayerWeights {
    fn into_buffer_sink(self, sink: &mut impl NativeBufferSink) {
        self.weights.into_buffer_sink(sink);
        self.finish.into_buffer_sink(sink);
    }
}

impl NativeSessionLayer {
    fn into_buffer_sink(self, sink: &mut impl NativeBufferSink) {
        match self {
            Self::Full => {}
            Self::Recurrent(state) => state.into_buffer_sink(sink),
        }
    }
}

impl CompletionResource for ModelSessionResources {
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

fn derive_session_plan(
    weights: &Qwen35Weights,
    resident: &DeviceModelPlan,
    max_context: usize,
) -> Result<DeviceModelPlan> {
    if max_context > resident.layout.max_context() {
        return NativeSessionStateSnafu {
            rule: "native session context must not exceed its resident model ceiling",
        }
        .fail();
    }
    let session = DeviceModelPlan::from_weights(weights, max_context, resident.page_tokens)?;
    if session.embedding.shape != resident.embedding.shape
        || session.output.shape != resident.output.shape
        || session.output_rms.elements() != resident.output_rms.elements()
        || !same_layer_roles(&session.layers, &resident.layers)
    {
        return NativeSessionStateSnafu {
            rule: "native session plan must retain resident uploaded-weight bindings",
        }
        .fail();
    }
    Ok(session)
}

fn same_layer_roles(session: &[NativeBlockPlan], resident: &[NativeBlockPlan]) -> bool {
    session.len() == resident.len()
        && session.iter().zip(resident).all(|(session, resident)| {
            matches!(
                (session, resident),
                (NativeBlockPlan::Full(_), NativeBlockPlan::Full(_))
                    | (
                        NativeBlockPlan::Recurrent { .. },
                        NativeBlockPlan::Recurrent { .. }
                    )
            )
        })
}

fn upload_resident_layers(
    weights: &Qwen35Weights,
    plan: &DeviceModelPlan,
    device: &Device,
) -> Result<Vec<NativeModelLayer>> {
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(plan.layers.len())
        .context(ExecutionAllocationSnafu {
            target: "native model layer resources",
            length: plan.layers.len(),
        })?;
    for block in &plan.layers {
        let layer = match block {
            NativeBlockPlan::Full(plan) => {
                NativeModelLayer::Full(Box::new(NativeWeights::upload(weights, plan, device)?))
            }
            NativeBlockPlan::Recurrent { plan, finish } => {
                NativeModelLayer::Recurrent(Box::new(NativeRecurrentLayerWeights {
                    weights: NativeRecurrentWeights::upload(weights, plan, device)?,
                    finish: LayerFinishWeights::upload(weights, finish, device)?,
                }))
            }
        };
        layers.push(layer);
    }
    if layers.len() != plan.layers.len() {
        return NativeSessionStateSnafu {
            rule: "native model resource layers must exactly cover every planned block",
        }
        .fail();
    }
    Ok(layers)
}

fn allocate_session_layers(
    plan: &DeviceModelPlan,
    device: &Device,
) -> Result<Vec<NativeSessionLayer>> {
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(plan.layers.len())
        .context(ExecutionAllocationSnafu {
            target: "native model session layers",
            length: plan.layers.len(),
        })?;
    for block in &plan.layers {
        layers.push(match block {
            NativeBlockPlan::Full(_) => NativeSessionLayer::Full,
            NativeBlockPlan::Recurrent { plan, .. } => {
                NativeSessionLayer::Recurrent(NativeRecurrentState::new(plan, device)?)
            }
        });
    }
    if layers.len() != plan.layers.len() {
        return NativeSessionStateSnafu {
            rule: "native model session layers must exactly cover every planned block",
        }
        .fail();
    }
    Ok(layers)
}

#[cfg(test)]
mod tests {
    use super::derive_session_plan;
    use crate::Qwen35Weights;
    use crate::qwen35::tests::{canonical_hybrid_fixture_with_context, verify_fixture};
    use crate::qwen35_native::model_plan::DeviceModelPlan;
    use crate::qwen35_native::model_step::ModelTokenPlan;

    const RESIDENT_CONTEXT: usize = 16;
    const SESSION_CONTEXT: usize = 4;
    const PAGE_TOKENS: kernels::attention::NativePageTokens =
        kernels::attention::NativePageTokens::B8;

    #[test]
    fn session_plan_reuses_resident_binding_at_exact_smaller_context()
    -> core::result::Result<(), String> {
        let artifact = verify_fixture(&canonical_hybrid_fixture_with_context(RESIDENT_CONTEXT)?)?;
        let weights =
            Qwen35Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;
        let resident = DeviceModelPlan::from_weights(&weights, RESIDENT_CONTEXT, PAGE_TOKENS)
            .map_err(|error| error.to_string())?;
        let session = derive_session_plan(&weights, &resident, SESSION_CONTEXT)
            .map_err(|error| error.to_string())?;

        assert_eq!(session.layout.max_context(), SESSION_CONTEXT);
        assert_eq!(session.page_tokens, resident.page_tokens);
        assert_eq!(session.layers.len(), resident.layers.len());
        assert!(
            session.bytes.key_values < resident.bytes.key_values,
            "a smaller exact session context must allocate its own smaller K/V extent"
        );
        assert!(
            ModelTokenPlan::from_model(&session, SESSION_CONTEXT, 0).is_err(),
            "the per-use plan cannot dispatch beyond its exact requested context"
        );
        assert!(
            derive_session_plan(&weights, &resident, RESIDENT_CONTEXT + 1).is_err(),
            "a session plan cannot exceed the resident model's immutable ceiling"
        );
        Ok(())
    }
}
