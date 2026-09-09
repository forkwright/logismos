//! Owned native resources before one-token submission.

use cache::NativePagedKvPool;
use hipcore::{Device, DeviceBuffer, Stream};
use snafu::ResultExt;

use crate::error::{
    ArithmeticOverflowSnafu, ExecutionAllocationSnafu, ExecutionPagedDecodePlanSnafu,
    NativeDeviceSnafu, NativePagedKvSnafu, NativeSessionStateSnafu,
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

pub(super) struct FullAttentionStep<'buffers> {
    pub(super) input: &'buffers DeviceBuffer<f32>,
    pub(super) output: &'buffers DeviceBuffer<f32>,
    pub(super) cosine: &'buffers DeviceBuffer<f32>,
    pub(super) sine: &'buffers DeviceBuffer<f32>,
    pub(super) attention: kernels::attention::NativePagedDecodePlan,
}

impl DeviceResources {
    pub(super) fn new(
        weights: &Qwen35Weights<'_>,
        plan: DeviceFullAttentionPlan,
        device: &Device,
    ) -> Result<Self> {
        let _ = plan.bytes.total()?;
        let stream = Stream::new(device).context(NativeDeviceSnafu)?;
        let owned_weights = NativeWeights::upload(weights, &plan, device)?;
        let kv = NativePagedKvPool::new(plan.kv, device).context(NativePagedKvSnafu)?;
        let workspace = NativeWorkspace::new(&plan.workspace, device)?;
        let finish_workspace = LayerFinishWorkspace::new(plan.finish.workspace, device)?;
        Ok(Self {
            plan,
            weights: owned_weights,
            kv,
            stream,
            workspace,
            finish_workspace,
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
    pub(super) fn new(plan: &WorkspacePlan, device: &Device) -> Result<Self> {
        macro_rules! buffer {
            ($field:ident) => {
                DeviceBuffer::alloc(device, plan.$field).context(NativeDeviceSnafu)?
            };
        }
        Ok(Self {
            hidden: buffer!(hidden),
            q_gate: buffer!(q_gate),
            query: buffer!(query),
            gate: buffer!(gate_values),
            normalized_query: buffer!(normalized_query),
            key: buffer!(key),
            normalized_key: buffer!(normalized_key),
            value: buffer!(value),
            attention: buffer!(attention),
            gated: buffer!(gated),
            output_projection: buffer!(output_projection),
        })
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
