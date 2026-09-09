//! Owned native resources for one whole Qwen3.5 main-model step.

use cache::NativePagedKvPool;
use hipcore::{Device, DeviceBuffer, Stream};
use snafu::ResultExt;

use super::CompletionResource;
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
    Recurrent(Box<NativeRecurrentLayerResources>),
}

struct NativeRecurrentLayerResources {
    weights: NativeRecurrentWeights,
    state: NativeRecurrentState,
    finish: LayerFinishWeights,
}

/// One non-cloneable bundle retaining every native allocation through completion.
pub(super) struct ModelDeviceResources {
    plan: DeviceModelPlan,
    embedding: NativeMatrix,
    output: NativeMatrix,
    output_norm: DeviceBuffer<f32>,
    layers: Vec<NativeModelLayer>,
    kv: Option<NativePagedKvPool>,
    stream: Stream,
    numerical_status: kernels::numerical_status::NativeNumericalStatus,
    full_workspace: Option<NativeWorkspace>,
    recurrent_workspace: Option<NativeRecurrentWorkspace>,
    finish_workspace: LayerFinishWorkspace,
    hidden_a: DeviceBuffer<f32>,
    hidden_b: DeviceBuffer<f32>,
    final_normalized: DeviceBuffer<f32>,
    step: Option<ModelStep>,
    position: usize,
}

impl ModelDeviceResources {
    pub(super) fn new(
        weights: &Qwen35Weights,
        plan: DeviceModelPlan,
        device: &Device,
    ) -> Result<Self> {
        let _ = plan.bytes.total()?;
        let stream = Stream::new(device).context(NativeDeviceSnafu)?;
        let numerical_status = kernels::numerical_status::NativeNumericalStatus::new(device)
            .context(NativeKernelSnafu)?;
        let embedding = NativeMatrix::upload(weights, &plan.embedding, device)?;
        let output = NativeMatrix::upload(weights, &plan.output, device)?;
        let output_norm = f32_parameter_buffer(weights, &plan.output_norm, device)?;
        let layers = upload_layers(weights, &plan, device)?;
        let kv = plan
            .kv
            .map(|plan| NativePagedKvPool::new(plan, device))
            .transpose()
            .context(NativePagedKvSnafu)?;
        let full_workspace = plan
            .full_workspace
            .as_ref()
            .map(|workspace| NativeWorkspace::new(workspace, device))
            .transpose()?;
        let recurrent_workspace = plan
            .recurrent_workspace
            .as_ref()
            .map(|workspace| NativeRecurrentWorkspace::new(workspace, device))
            .transpose()?;
        let finish_workspace = LayerFinishWorkspace::new(plan.finish_workspace, device)?;
        let hidden_a =
            DeviceBuffer::alloc(device, plan.layout.hidden).context(NativeDeviceSnafu)?;
        let hidden_b =
            DeviceBuffer::alloc(device, plan.layout.hidden).context(NativeDeviceSnafu)?;
        let final_normalized =
            DeviceBuffer::alloc(device, plan.output_rms.elements()).context(NativeDeviceSnafu)?;
        Ok(Self {
            plan,
            embedding,
            output,
            output_norm,
            layers,
            kv,
            stream,
            numerical_status,
            full_workspace,
            recurrent_workspace,
            finish_workspace,
            hidden_a,
            hidden_b,
            final_normalized,
            step: None,
            position: 0,
        })
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
            logits: DeviceBuffer::alloc(self.stream.device(), self.output.shape.rows())
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
                self.embedding.bytes.as_device_ptr(),
                self.embedding.bytes.len(),
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
        for (plan, layer) in self.plan.layers.iter().zip(&self.layers) {
            match (plan, layer) {
                (NativeBlockPlan::Full(plan), NativeModelLayer::Full(weights)) => {
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
                        state: &resources.state,
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
                &self.output_norm,
                &self.final_normalized,
                &self.stream,
                &self.numerical_status,
            )
        }?;
        // SAFETY: the verified output matrix and exact final/logit spans remain owned through completion.
        unsafe {
            self.output.launch(
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
            if let NativeModelLayer::Recurrent(resources) = layer {
                resources.state.publish_completed();
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

impl CompletionResource for ModelDeviceResources {
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

fn upload_layers(
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
                NativeModelLayer::Recurrent(Box::new(NativeRecurrentLayerResources {
                    weights: NativeRecurrentWeights::upload(weights, plan, device)?,
                    state: NativeRecurrentState::new(plan, device)?,
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
