//! Owned device copies of verified full-attention weights.

use hipcore::{Device, DeviceBuffer, Stream};
use snafu::ResultExt;

use super::custody::{NativeBufferSink, NativeBuildResult, NativeBuildScope, NativeBuildSource};
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
        scope: &NativeBuildScope,
    ) -> NativeBuildResult<Self> {
        let q_gate = scope.guard(
            NativeMatrix::upload(weights, &plan.matrices.q_gate, device, scope)?,
            NativeMatrix::into_buffer_sink,
        );
        let key = scope.guard(
            NativeMatrix::upload(weights, &plan.matrices.key, device, scope)?,
            NativeMatrix::into_buffer_sink,
        );
        let value = scope.guard(
            NativeMatrix::upload(weights, &plan.matrices.value, device, scope)?,
            NativeMatrix::into_buffer_sink,
        );
        let output = scope.guard(
            NativeMatrix::upload(weights, &plan.matrices.output, device, scope)?,
            NativeMatrix::into_buffer_sink,
        );
        let input_norm = scope.guard(
            f32_parameter_buffer(weights, &plan.norms.input, device, scope)?,
            |buffer, sink| sink.push_f32(buffer),
        );
        let query_norm = scope.guard(
            f32_parameter_buffer(weights, &plan.norms.query, device, scope)?,
            |buffer, sink| sink.push_f32(buffer),
        );
        let key_norm = scope.guard(
            f32_parameter_buffer(weights, &plan.norms.key, device, scope)?,
            |buffer, sink| sink.push_f32(buffer),
        );
        let finish = scope.guard(
            LayerFinishWeights::upload(weights, &plan.finish, device, scope)?,
            LayerFinishWeights::into_buffer_sink,
        );
        Ok(Self {
            q_gate: q_gate.commit(),
            key: key.commit(),
            value: value.commit(),
            output: output.commit(),
            input_norm: input_norm.commit(),
            query_norm: query_norm.commit(),
            key_norm: key_norm.commit(),
            finish: finish.commit(),
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
        scope: &NativeBuildScope,
    ) -> NativeBuildResult<Self> {
        let matrix = weights
            .checked_matrix(&plan.name)
            .map_err(NativeBuildSource::decoder)?;
        let shape = matrix
            .native_shape()
            .context(NativeKernelSnafu)
            .map_err(NativeBuildSource::decoder)?;
        if shape != plan.shape || matrix.serialized_bytes().len() != plan.serialized_bytes {
            let error = NativeSessionStateSnafu {
                rule: "native matrix upload must retain its verified descriptor binding",
            }
            .build();
            return Err(NativeBuildSource::decoder(error));
        }
        let mut bytes = scope.allocate_u8(device, matrix.serialized_bytes().len())?;
        bytes
            .copy_from_host(matrix.serialized_bytes())
            .context(NativeDeviceSnafu)
            .map_err(NativeBuildSource::decoder)?;
        Ok(Self {
            shape,
            bytes: bytes.commit(),
        })
    }

    pub(super) fn into_buffer_sink(self, sink: &mut impl NativeBufferSink) {
        sink.push_u8(self.bytes);
    }

    /// Launch a checked projection for `token_count` dense input rows.
    ///
    /// # Safety
    ///
    /// `input` and `output` must be exact distinct spans on `stream`'s device,
    /// with the sticky status allocation retained through completion.
    pub(super) unsafe fn launch_rows(
        &self,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        token_count: usize,
        stream: &Stream,
        numerical_status: &kernels::numerical_status::NativeNumericalStatus,
    ) -> Result<()> {
        let batch = kernels::row_gemv::RowGemvBatchPlan::try_from_shape(
            self.shape,
            token_count,
            input.len(),
            output.len(),
        )
        .context(NativeKernelSnafu)?;
        // SAFETY: the caller retains the exact checked device spans and status
        // allocation through completion for the matrix shape bound above.
        unsafe {
            kernels::row_gemv::launch_row_gemv_f32_rows_checked(
                batch,
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

pub(super) fn f32_parameter_buffer(
    weights: &Qwen35Weights,
    plan: &F32Parameter,
    device: &Device,
    scope: &NativeBuildScope,
) -> NativeBuildResult<DeviceBuffer<f32>> {
    let values = read_f32(weights, &plan.name, &plan.dimensions, plan.elements)
        .map_err(NativeBuildSource::decoder)?;
    if values.len() != plan.elements {
        let error = NativeSessionStateSnafu {
            rule: "native F32 parameter upload must retain its verified descriptor binding",
        }
        .build();
        return Err(NativeBuildSource::decoder(error));
    }
    let mut buffer = scope.allocate_f32(device, values.len())?;
    buffer
        .copy_from_host(&values)
        .context(NativeDeviceSnafu)
        .map_err(NativeBuildSource::decoder)?;
    Ok(buffer.commit())
}
