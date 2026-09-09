//! Owned device copies of verified full-attention weights.

use hipcore::{Device, DeviceBuffer};
use snafu::ResultExt;

use crate::error::{
    ArithmeticOverflowSnafu, NativeDeviceSnafu, NativeKernelSnafu, NativeSessionStateSnafu,
};
use crate::qwen35_execution::read_f32;
use crate::qwen35_native::plan::{DeviceFullAttentionPlan, ProjectionWeight, ScalarWeight};
use crate::{Qwen35Weights, Result};

pub(crate) struct NativeWeights {
    pub(crate) q_gate: NativeMatrix,
    pub(crate) key: NativeMatrix,
    pub(crate) value: NativeMatrix,
    pub(crate) output: NativeMatrix,
    pub(crate) ffn_gate: NativeMatrix,
    pub(crate) ffn_up: NativeMatrix,
    pub(crate) ffn_down: NativeMatrix,
    pub(crate) input_norm: DeviceBuffer<f32>,
    pub(crate) query_norm: DeviceBuffer<f32>,
    pub(crate) key_norm: DeviceBuffer<f32>,
    pub(crate) post_attention_norm: DeviceBuffer<f32>,
}
pub(crate) struct NativeMatrix {
    pub(crate) shape: kernels::RowGemvShape,
    pub(crate) bytes: DeviceBuffer<u8>,
}

impl NativeWeights {
    pub(crate) fn upload(
        weights: &Qwen35Weights<'_>,
        plan: &DeviceFullAttentionPlan,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            q_gate: matrix(weights, &plan.matrices.q_gate, device)?,
            key: matrix(weights, &plan.matrices.key, device)?,
            value: matrix(weights, &plan.matrices.value, device)?,
            output: matrix(weights, &plan.matrices.output, device)?,
            ffn_gate: matrix(weights, &plan.matrices.ffn_gate, device)?,
            ffn_up: matrix(weights, &plan.matrices.ffn_up, device)?,
            ffn_down: matrix(weights, &plan.matrices.ffn_down, device)?,
            input_norm: scalar(weights, &plan.scalars.input_norm, device)?,
            query_norm: scalar(weights, &plan.scalars.query_norm, device)?,
            key_norm: scalar(weights, &plan.scalars.key_norm, device)?,
            post_attention_norm: scalar(weights, &plan.scalars.post_attention_norm, device)?,
        })
    }
}

fn matrix(
    weights: &Qwen35Weights<'_>,
    plan: &ProjectionWeight,
    device: &Device,
) -> Result<NativeMatrix> {
    let matrix = weights.checked_matrix(&plan.name)?;
    let shape = matrix.native_shape().context(NativeKernelSnafu)?;
    if shape != plan.shape || matrix.serialized_bytes().len() != plan.serialized_bytes {
        return NativeSessionStateSnafu {
            rule: "native matrix upload must retain its verified descriptor binding",
        }
        .fail();
    }
    Ok(NativeMatrix {
        shape,
        bytes: DeviceBuffer::from_host(device, matrix.serialized_bytes())
            .context(NativeDeviceSnafu)?,
    })
}

fn scalar(
    weights: &Qwen35Weights<'_>,
    plan: &ScalarWeight,
    device: &Device,
) -> Result<DeviceBuffer<f32>> {
    let dimension = u64::try_from(plan.elements).map_err(|_| {
        ArithmeticOverflowSnafu {
            context: "native scalar upload width",
        }
        .build()
    })?;
    let values = read_f32(weights, &plan.name, &[dimension], plan.elements)?;
    if values.len() != plan.elements {
        return NativeSessionStateSnafu {
            rule: "native scalar upload must retain its verified descriptor binding",
        }
        .fail();
    }
    DeviceBuffer::from_host(device, &values).context(NativeDeviceSnafu)
}
