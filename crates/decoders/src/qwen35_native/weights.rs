//! Owned device copies of verified full-attention weights.

use hipcore::{Device, DeviceBuffer};
use snafu::ResultExt;

use super::custody::NativeBufferSink;
use crate::error::{NativeDeviceSnafu, NativeKernelSnafu, NativeSessionStateSnafu};
use crate::qwen35_execution::read_f32;
use crate::qwen35_native::finish::LayerFinishWeights;
use crate::qwen35_native::plan::{DeviceFullAttentionPlan, F32Parameter, ProjectionWeight};
use crate::{Qwen35Weights, Result};

pub(super) struct NativeWeights {
    pub(super) q_gate: NativeMatrix,
    pub(super) key: NativeMatrix,
    pub(super) value: NativeMatrix,
    pub(super) output: NativeMatrix,
    pub(super) input_norm: DeviceBuffer<f32>,
    pub(super) query_norm: DeviceBuffer<f32>,
    pub(super) key_norm: DeviceBuffer<f32>,
    pub(super) finish: LayerFinishWeights,
}
pub(super) struct NativeMatrix {
    pub(super) shape: kernels::row_gemv::RowGemvShape,
    pub(super) bytes: DeviceBuffer<u8>,
}

impl NativeWeights {
    pub(super) fn upload(
        weights: &Qwen35Weights,
        plan: &DeviceFullAttentionPlan,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            q_gate: NativeMatrix::upload(weights, &plan.matrices.q_gate, device)?,
            key: NativeMatrix::upload(weights, &plan.matrices.key, device)?,
            value: NativeMatrix::upload(weights, &plan.matrices.value, device)?,
            output: NativeMatrix::upload(weights, &plan.matrices.output, device)?,
            input_norm: f32_parameter_buffer(weights, &plan.norms.input, device)?,
            query_norm: f32_parameter_buffer(weights, &plan.norms.query, device)?,
            key_norm: f32_parameter_buffer(weights, &plan.norms.key, device)?,
            finish: LayerFinishWeights::upload(weights, &plan.finish, device)?,
        })
    }

    pub(super) fn into_buffer_sink(self, sink: &mut impl NativeBufferSink) {
        self.q_gate.into_buffer_sink(sink);
        self.key.into_buffer_sink(sink);
        self.value.into_buffer_sink(sink);
        self.output.into_buffer_sink(sink);
        sink.push_f32(self.input_norm);
        sink.push_f32(self.query_norm);
        sink.push_f32(self.key_norm);
        self.finish.into_buffer_sink(sink);
    }
}

impl NativeMatrix {
    pub(super) fn upload(
        weights: &Qwen35Weights,
        plan: &ProjectionWeight,
        device: &Device,
    ) -> Result<Self> {
        let matrix = weights.checked_matrix(&plan.name)?;
        let shape = matrix.native_shape().context(NativeKernelSnafu)?;
        if shape != plan.shape || matrix.serialized_bytes().len() != plan.serialized_bytes {
            return NativeSessionStateSnafu {
                rule: "native matrix upload must retain its verified descriptor binding",
            }
            .fail();
        }
        Ok(Self {
            shape,
            bytes: DeviceBuffer::from_host(device, matrix.serialized_bytes())
                .context(NativeDeviceSnafu)?,
        })
    }

    pub(super) fn into_buffer_sink(self, sink: &mut impl NativeBufferSink) {
        sink.push_u8(self.bytes);
    }
}

pub(super) fn f32_parameter_buffer(
    weights: &Qwen35Weights,
    plan: &F32Parameter,
    device: &Device,
) -> Result<DeviceBuffer<f32>> {
    let values = read_f32(weights, &plan.name, &plan.dimensions, plan.elements)?;
    if values.len() != plan.elements {
        return NativeSessionStateSnafu {
            rule: "native F32 parameter upload must retain its verified descriptor binding",
        }
        .fail();
    }
    DeviceBuffer::from_host(device, &values).context(NativeDeviceSnafu)
}
