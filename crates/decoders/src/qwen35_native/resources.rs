//! Owned native resources before one-token submission.

use cache::NativePagedKvPool;
use hipcore::{Device, DeviceBuffer, Stream};
use snafu::ResultExt;

use super::custody::{NativeBufferSink, NativeBuildResult, NativeBuildScope, NativeBuildSource};
use super::model_resources::{StreamRetention, build_native_kv, build_numerical_status};
use super::model_session::NativeBuildFailure;
use crate::error::{
    ArithmeticOverflowSnafu, ExecutionAllocationSnafu, ExecutionPagedDecodePlanSnafu,
    NativeDeviceSnafu, NativeSessionStateSnafu,
};
use crate::qwen35_mrope::{TextMrope, text_mrope_coefficient};
use crate::qwen35_native::finish::LayerFinishWorkspace;
use crate::qwen35_native::plan::{DeviceFullAttentionPlan, WorkspacePlan};
use crate::qwen35_native::weights::NativeWeights;
use crate::{Qwen35Weights, Result};

pub(super) struct DeviceResources {
    pub(super) plan: DeviceFullAttentionPlan,
    pub(super) weights: NativeWeights,
    pub(super) kv: NativePagedKvPool,
    pub(super) stream: Stream,
    pub(super) numerical_status: kernels::numerical_status::NativeNumericalStatus,
    pub(super) workspace: NativeWorkspace,
    pub(super) finish_workspace: LayerFinishWorkspace,
    pub(super) step: Option<StepBuffers>,
    pub(super) position: usize,
}

pub(super) struct NativeWorkspace {
    pub(super) hidden: DeviceBuffer<f32>,
    pub(super) q_gate: DeviceBuffer<f32>,
    pub(super) query: DeviceBuffer<f32>,
    pub(super) gate: DeviceBuffer<f32>,
    pub(super) normalized_query: DeviceBuffer<f32>,
    pub(super) key: DeviceBuffer<f32>,
    pub(super) normalized_key: DeviceBuffer<f32>,
    pub(super) value: DeviceBuffer<f32>,
    pub(super) attention: DeviceBuffer<f32>,
    pub(super) gated: DeviceBuffer<f32>,
    pub(super) output_projection: DeviceBuffer<f32>,
}

pub(super) struct StepBuffers {
    pub(super) input: DeviceBuffer<f32>,
    pub(super) output: DeviceBuffer<f32>,
    pub(super) cosine: DeviceBuffer<f32>,
    pub(super) sine: DeviceBuffer<f32>,
    pub(super) attention: kernels::attention::NativePagedDecodePlan,
}

struct DeviceBuildFields {
    weights: NativeWeights,
    kv: NativePagedKvPool,
    numerical_status: kernels::numerical_status::NativeNumericalStatus,
    workspace: NativeWorkspace,
    finish_workspace: LayerFinishWorkspace,
}

pub(super) struct FullAttentionStep<'buffers> {
    pub(super) input: &'buffers DeviceBuffer<f32>,
    pub(super) output: &'buffers DeviceBuffer<f32>,
    pub(super) cosine: &'buffers DeviceBuffer<f32>,
    pub(super) sine: &'buffers DeviceBuffer<f32>,
    pub(super) attention: kernels::attention::NativePagedDecodePlan,
}

impl DeviceResources {
    pub(super) fn new(
        weights: &Qwen35Weights,
        plan: DeviceFullAttentionPlan,
        device: &Device,
    ) -> core::result::Result<Self, NativeBuildFailure> {
        let scope = NativeBuildScope::new();
        if let Err(error) = plan.bytes.total() {
            return Err(NativeBuildFailure::standalone(
                NativeBuildSource::decoder(error),
                scope,
                None,
                device.clone(),
            ));
        }
        let stream = match Stream::new_tracked(device) {
            Ok(stream) => StreamRetention::new(stream),
            Err(error) => {
                return Err(NativeBuildFailure::standalone(
                    NativeBuildSource::stream(error),
                    scope,
                    None,
                    device.clone(),
                ));
            }
        };
        let fields = match build_device_fields(weights, &plan, device, &scope) {
            Ok(fields) => fields,
            Err(source) => {
                return Err(NativeBuildFailure::standalone(
                    source,
                    scope,
                    Some(stream),
                    device.clone(),
                ));
            }
        };
        Ok(Self {
            plan,
            weights: fields.weights,
            kv: fields.kv,
            stream: stream.recover(),
            numerical_status: fields.numerical_status,
            workspace: fields.workspace,
            finish_workspace: fields.finish_workspace,
            step: None,
            position: 0,
        })
    }

    pub(super) fn prepare_step(&mut self, input: DeviceBuffer<f32>) -> Result<()> {
        if self.step.is_some()
            || input.device().ordinal() != self.stream.device().ordinal()
            || input.len() != self.plan.workspace.hidden
        {
            return NativeSessionStateSnafu {
                rule: "native step input must be one stream-device hidden row with no pending step",
            }
            .fail();
        }
        let visible = self.position.checked_add(1).ok_or_else(|| {
            ArithmeticOverflowSnafu {
                context: "native next position",
            }
            .build()
        })?;
        if visible > self.plan.layout.max_context() {
            return NativeSessionStateSnafu {
                rule: "native context must remain within its plan",
            }
            .fail();
        }
        let (cosine, sine) = native_mrope_controls(
            self.plan.layout.text_mrope(),
            self.position,
            self.plan.workspace.query_rotary.coefficient_elements(),
        )?;
        let logical = kernels::PagedDecodePlan::try_from_dimensions(
            visible,
            self.plan.layout.heads,
            self.plan.layout.kv_heads,
            self.plan.layout.key,
        )
        .context(ExecutionPagedDecodePlanSnafu)?;
        let attention = kernels::attention::NativePagedDecodePlan::try_from_paged_decode(
            logical,
            self.plan.kv.layout().page_tokens(),
            self.plan.kv.layout().physical_pages(),
        )
        .context(ExecutionPagedDecodePlanSnafu)?;
        self.step = Some(StepBuffers {
            input,
            output: DeviceBuffer::alloc(self.stream.device(), self.plan.workspace.hidden)
                .context(NativeDeviceSnafu)?,
            cosine: DeviceBuffer::from_host(self.stream.device(), &cosine)
                .context(NativeDeviceSnafu)?,
            sine: DeviceBuffer::from_host(self.stream.device(), &sine)
                .context(NativeDeviceSnafu)?,
            attention,
        });
        Ok(())
    }
}

fn build_device_fields(
    weights: &Qwen35Weights,
    plan: &DeviceFullAttentionPlan,
    device: &Device,
    scope: &NativeBuildScope,
) -> NativeBuildResult<DeviceBuildFields> {
    let numerical_status = scope.guard(build_numerical_status(device, scope)?, |status, sink| {
        sink.push_u32(status.into_buffer());
    });
    let owned_weights = scope.guard(
        NativeWeights::upload(weights, plan, device, scope)?,
        NativeWeights::into_buffer_sink,
    );
    let kv = scope.guard(build_native_kv(plan.kv, device, scope)?, retain_device_kv);
    let workspace = scope.guard(
        NativeWorkspace::new(&plan.workspace, device, scope)?,
        NativeWorkspace::into_buffer_sink,
    );
    let finish_workspace = scope.guard(
        LayerFinishWorkspace::new(plan.finish.workspace, device, scope)?,
        LayerFinishWorkspace::into_buffer_sink,
    );
    Ok(DeviceBuildFields {
        weights: owned_weights.commit(),
        kv: kv.commit(),
        numerical_status: numerical_status.commit(),
        workspace: workspace.commit(),
        finish_workspace: finish_workspace.commit(),
    })
}

fn retain_device_kv(pool: NativePagedKvPool, sink: &mut impl NativeBufferSink) {
    let (keys, values, table) = pool.into_buffers().into_parts();
    sink.push_f32(keys);
    sink.push_f32(values);
    sink.push_u32(table);
}

impl StepBuffers {
    pub(super) fn full_attention(&self) -> FullAttentionStep<'_> {
        FullAttentionStep {
            input: &self.input,
            output: &self.output,
            cosine: &self.cosine,
            sine: &self.sine,
            attention: self.attention,
        }
    }
}

impl NativeWorkspace {
    pub(super) fn new(
        plan: &WorkspacePlan,
        device: &Device,
        scope: &NativeBuildScope,
    ) -> NativeBuildResult<Self> {
        macro_rules! buffer {
            ($field:ident) => {
                scope.allocate_f32(device, plan.$field)?
            };
        }
        let hidden = buffer!(hidden);
        let q_gate = buffer!(q_gate);
        let query = buffer!(query);
        let gate = buffer!(gate_values);
        let normalized_query = buffer!(normalized_query);
        let key = buffer!(key);
        let normalized_key = buffer!(normalized_key);
        let value = buffer!(value);
        let attention = buffer!(attention);
        let gated = buffer!(gated);
        let output_projection = buffer!(output_projection);
        Ok(Self {
            hidden: hidden.commit(),
            q_gate: q_gate.commit(),
            query: query.commit(),
            gate: gate.commit(),
            normalized_query: normalized_query.commit(),
            key: key.commit(),
            normalized_key: normalized_key.commit(),
            value: value.commit(),
            attention: attention.commit(),
            gated: gated.commit(),
            output_projection: output_projection.commit(),
        })
    }

    pub(super) fn into_buffer_sink(self, sink: &mut impl NativeBufferSink) {
        sink.push_f32(self.hidden);
        sink.push_f32(self.q_gate);
        sink.push_f32(self.query);
        sink.push_f32(self.gate);
        sink.push_f32(self.normalized_query);
        sink.push_f32(self.key);
        sink.push_f32(self.normalized_key);
        sink.push_f32(self.value);
        sink.push_f32(self.attention);
        sink.push_f32(self.gated);
        sink.push_f32(self.output_projection);
    }
}

pub(super) fn native_mrope_controls(
    mrope: TextMrope,
    position: usize,
    pairs: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut cosine = Vec::new();
    let mut sine = Vec::new();
    cosine
        .try_reserve_exact(pairs)
        .context(ExecutionAllocationSnafu {
            target: "native cosine controls",
            length: pairs,
        })?;
    sine.try_reserve_exact(pairs)
        .context(ExecutionAllocationSnafu {
            target: "native sine controls",
            length: pairs,
        })?;
    for pair in 0..pairs {
        let (cosine_value, sine_value) = text_mrope_coefficient(mrope, position, pair)?;
        cosine.push(cosine_value);
        sine.push(sine_value);
    }
    Ok((cosine, sine))
}
