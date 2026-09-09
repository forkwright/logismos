//! Owned native resources before one-token submission.

use cache::NativePagedKvPool;
use hipcore::{Device, DeviceBuffer, Stream};
use snafu::ResultExt;

use crate::error::{
    ArithmeticOverflowSnafu, ExecutionAllocationSnafu, ExecutionPagedDecodePlanSnafu,
    NativeDeviceSnafu, NativeKernelSnafu, NativePagedKvSnafu, NativeSessionStateSnafu,
};
use crate::qwen35_mrope::text_mrope_coefficient;
use crate::qwen35_native::plan::{DeviceFullAttentionPlan, WorkspacePlan};
use crate::qwen35_native::weights::NativeWeights;
use crate::{Qwen35Weights, Result};

pub(crate) struct DeviceResources {
    pub(crate) plan: DeviceFullAttentionPlan,
    pub(crate) weights: NativeWeights,
    pub(crate) kv: NativePagedKvPool,
    pub(crate) stream: Stream,
    pub(crate) workspace: NativeWorkspace,
    pub(crate) step: Option<StepBuffers>,
    pub(crate) position: usize,
}

pub(crate) struct NativeWorkspace {
    pub(crate) hidden: DeviceBuffer<f32>,
    pub(crate) q_gate: DeviceBuffer<f32>,
    pub(crate) query: DeviceBuffer<f32>,
    pub(crate) gate: DeviceBuffer<f32>,
    pub(crate) normalized_query: DeviceBuffer<f32>,
    pub(crate) key: DeviceBuffer<f32>,
    pub(crate) normalized_key: DeviceBuffer<f32>,
    pub(crate) value: DeviceBuffer<f32>,
    pub(crate) attention: DeviceBuffer<f32>,
    pub(crate) gated: DeviceBuffer<f32>,
    pub(crate) output_projection: DeviceBuffer<f32>,
    pub(crate) attention_residual: DeviceBuffer<f32>,
    pub(crate) post_norm: DeviceBuffer<f32>,
    pub(crate) ffn_gate: DeviceBuffer<f32>,
    pub(crate) ffn_up: DeviceBuffer<f32>,
    pub(crate) ffn_product: DeviceBuffer<f32>,
    pub(crate) ffn_down: DeviceBuffer<f32>,
}

pub(crate) struct StepBuffers {
    pub(crate) input: DeviceBuffer<f32>,
    pub(crate) output: DeviceBuffer<f32>,
    pub(crate) cosine: DeviceBuffer<f32>,
    pub(crate) sine: DeviceBuffer<f32>,
    pub(crate) attention: kernels::attention::NativePagedDecodePlan,
}

impl DeviceResources {
    pub(crate) fn new(
        weights: &Qwen35Weights<'_>,
        plan: DeviceFullAttentionPlan,
        device: &Device,
    ) -> Result<Self> {
        let _ = plan.bytes.total()?;
        let stream = Stream::new(device).context(NativeDeviceSnafu)?;
        let owned_weights = NativeWeights::upload(weights, &plan, device)?;
        let kv = NativePagedKvPool::new(plan.kv, device).context(NativePagedKvSnafu)?;
        let workspace = NativeWorkspace::new(plan.workspace, device)?;
        Ok(Self {
            plan,
            weights: owned_weights,
            kv,
            stream,
            workspace,
            step: None,
            position: 0,
        })
    }

    pub(crate) fn prepare_step(&mut self, input: DeviceBuffer<f32>) -> Result<()> {
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
        let pairs = self.plan.workspace.query_rotary.coefficient_elements();
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
            let (c, s) =
                text_mrope_coefficient(self.plan.layout.text_mrope(), self.position, pair)?;
            cosine.push(c);
            sine.push(s);
        }
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
impl NativeWorkspace {
    fn new(plan: WorkspacePlan, device: &Device) -> Result<Self> {
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
            attention_residual: buffer!(attention_residual),
            post_norm: buffer!(post_norm),
            ffn_gate: buffer!(ffn_gate),
            ffn_up: buffer!(ffn_up),
            ffn_product: buffer!(ffn_product),
            ffn_down: buffer!(ffn_down),
        })
    }
}
