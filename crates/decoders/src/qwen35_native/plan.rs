//! Checked native full-attention allocation and weight descriptors.

use core::mem::size_of;

use cache::{NativePagedKvPlan, PagedKvGeometry};
use snafu::ResultExt;

use crate::error::{
    ArithmeticOverflowSnafu, ExecutionPagedDecodePlanSnafu, NativeKernelSnafu, NativePagedKvSnafu,
    NativeSessionStateSnafu,
};
use crate::qwen35_execution::{Layout, block_name, read_f32};
use crate::qwen35_native::finish::LayerFinishPlan;
use crate::{Qwen35Weights, Result};

#[derive(Debug)]
pub(super) struct DeviceFullAttentionPlan {
    pub(super) layout: Layout,
    pub(super) matrices: AttentionProjectionWeights,
    pub(super) norms: AttentionNormalizationWeights,
    pub(super) workspace: WorkspacePlan,
    pub(super) finish: LayerFinishPlan,
    pub(super) kv: NativePagedKvPlan,
    pub(super) bytes: DeviceByteDemand,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct WorkspacePlan {
    pub(super) hidden_norm: kernels::decoder_ops::RmsNormF32Plan,
    pub(super) query_norm: kernels::decoder_ops::RmsNormF32Plan,
    pub(super) key_norm: kernels::decoder_ops::RmsNormF32Plan,
    pub(super) query_rotary: kernels::decoder_ops::RotaryHalfSplitF32Plan,
    pub(super) key_rotary: kernels::decoder_ops::RotaryHalfSplitF32Plan,
    pub(super) split: kernels::decoder_ops::SplitQGateF32Plan,
    pub(super) gate: kernels::decoder_ops::ElementwiseF32Plan,
    pub(super) hidden: usize,
    pub(super) q_gate: usize,
    pub(super) query: usize,
    pub(super) gate_values: usize,
    pub(super) normalized_query: usize,
    pub(super) key: usize,
    pub(super) normalized_key: usize,
    pub(super) value: usize,
    pub(super) attention: usize,
    pub(super) gated: usize,
    pub(super) output_projection: usize,
}
#[derive(Debug)]
pub(super) struct ProjectionWeight {
    pub(super) name: String,
    pub(super) shape: kernels::row_gemv::RowGemvShape,
    pub(super) serialized_bytes: usize,
}

#[derive(Debug)]
pub(super) struct AttentionProjectionWeights {
    pub(super) q_gate: ProjectionWeight,
    pub(super) key: ProjectionWeight,
    pub(super) value: ProjectionWeight,
    pub(super) output: ProjectionWeight,
}

#[derive(Debug)]
pub(super) struct F32Parameter {
    pub(super) name: String,
    pub(super) dimensions: Vec<u64>,
    pub(super) elements: usize,
}

#[derive(Debug)]
pub(super) struct AttentionNormalizationWeights {
    pub(super) input: F32Parameter,
    pub(super) query: F32Parameter,
    pub(super) key: F32Parameter,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(super) struct DeviceByteDemand {
    pub(super) weights: usize,
    pub(super) scratch: usize,
    pub(super) input: usize,
    pub(super) output: usize,
    pub(super) controls: usize,
    pub(super) key_values: usize,
    pub(super) table: usize,
    pub(super) numerical_status: usize,
}

impl DeviceByteDemand {
    pub(super) fn total(self) -> Result<usize> {
        sum(
            &[
                self.weights,
                self.scratch,
                self.input,
                self.output,
                self.controls,
                self.key_values,
                self.table,
                self.numerical_status,
            ],
            "native device byte total",
        )
    }
}

pub(super) fn native_decode_plan(
    layout: Layout,
    visible: usize,
    kv: NativePagedKvPlan,
) -> Result<kernels::attention::NativePagedDecodePlan> {
    let logical = kernels::PagedDecodePlan::try_from_dimensions(
        visible,
        layout.heads,
        layout.kv_heads,
        layout.key,
    )
    .context(ExecutionPagedDecodePlanSnafu)?;
    kernels::attention::NativePagedDecodePlan::try_from_paged_decode(
        logical,
        kv.layout().page_tokens(),
        kv.layout().physical_pages(),
    )
    .context(ExecutionPagedDecodePlanSnafu)
}

impl DeviceFullAttentionPlan {
    pub(super) fn from_weights(
        weights: &Qwen35Weights,
        block: usize,
        max_context: usize,
        page_tokens: kernels::attention::NativePageTokens,
    ) -> Result<Self> {
        let layout = Layout::from_metadata(weights, max_context)?;
        Self::from_layout(weights, layout, block, page_tokens)
    }

    pub(super) fn from_layout(
        weights: &Qwen35Weights,
        layout: Layout,
        block: usize,
        page_tokens: kernels::attention::NativePageTokens,
    ) -> Result<Self> {
        if !layout.is_admitted_full_block(block) {
            return NativeSessionStateSnafu {
                rule: "native plan requires an admitted full-attention main block",
            }
            .fail();
        }
        let matrices = AttentionProjectionWeights::from_weights(weights, block)?;
        let norms = AttentionNormalizationWeights::from_weights(weights, layout, block)?;
        let workspace = WorkspacePlan::from_layout(layout)?;
        let finish = LayerFinishPlan::from_weights(weights, layout, block)?;
        let kv = NativePagedKvPlan::try_from_geometry(
            PagedKvGeometry {
                layers: 1,
                row_width: layout.kv_width,
                max_context: layout.max_context(),
            },
            layout.kv_heads,
            layout.key,
            page_tokens,
        )
        .context(NativePagedKvSnafu)?;
        let attention = native_decode_plan(layout, layout.max_context(), kv)?;
        let f32_bytes = size_of::<f32>();
        let bytes = DeviceByteDemand {
            weights: sum(
                &[matrices.bytes()?, norms.bytes()?, finish.demand.weights],
                "native weight bytes",
            )?,
            scratch: sum(
                &[
                    elements_bytes(workspace.elements()?, f32_bytes, "native scratch bytes")?,
                    finish.demand.scratch,
                ],
                "native scratch bytes",
            )?,
            input: elements_bytes(layout.hidden, f32_bytes, "native input bytes")?,
            output: elements_bytes(layout.hidden, f32_bytes, "native output bytes")?,
            controls: elements_bytes(
                workspace.coefficient_elements()?,
                f32_bytes,
                "native control bytes",
            )?,
            key_values: elements_bytes(
                kv.layout()
                    .backing_elements()
                    .checked_mul(2)
                    .ok_or_else(|| {
                        ArithmeticOverflowSnafu {
                            context: "native separate K/V backing",
                        }
                        .build()
                    })?,
                f32_bytes,
                "native K/V bytes",
            )?,
            table: elements_bytes(
                attention.page_table_entries(),
                size_of::<u32>(),
                "native table bytes",
            )?,
            numerical_status: kernels::numerical_status::NativeNumericalStatus::byte_demand(),
        };
        Ok(Self {
            layout,
            matrices,
            norms,
            workspace,
            finish,
            kv,
            bytes,
        })
    }
}

impl AttentionProjectionWeights {
    fn from_weights(weights: &Qwen35Weights, block: usize) -> Result<Self> {
        Ok(Self {
            q_gate: projection(weights, block_name(block, "attn_q.weight"))?,
            key: projection(weights, block_name(block, "attn_k.weight"))?,
            value: projection(weights, block_name(block, "attn_v.weight"))?,
            output: projection(weights, block_name(block, "attn_output.weight"))?,
        })
    }

    pub(super) fn bytes(&self) -> Result<usize> {
        sum(
            &[
                self.q_gate.serialized_bytes,
                self.key.serialized_bytes,
                self.value.serialized_bytes,
                self.output.serialized_bytes,
            ],
            "native serialized projection bytes",
        )
    }
}

impl AttentionNormalizationWeights {
    fn from_weights(weights: &Qwen35Weights, layout: Layout, block: usize) -> Result<Self> {
        Ok(Self {
            input: f32_parameter(
                weights,
                block_name(block, "attn_norm.weight"),
                vec![dimension(layout.hidden, "native attention norm width")?],
                layout.hidden,
            )?,
            query: f32_parameter(
                weights,
                block_name(block, "attn_q_norm.weight"),
                vec![dimension(layout.key, "native query norm width")?],
                layout.key,
            )?,
            key: f32_parameter(
                weights,
                block_name(block, "attn_k_norm.weight"),
                vec![dimension(layout.key, "native key norm width")?],
                layout.key,
            )?,
        })
    }

    pub(super) fn bytes(&self) -> Result<usize> {
        let widths = [self.input.elements, self.query.elements, self.key.elements];
        elements_bytes(
            sum(&widths, "native scalar weight elements")?,
            size_of::<f32>(),
            "native scalar weight bytes",
        )
    }
}

pub(super) fn projection(weights: &Qwen35Weights, name: String) -> Result<ProjectionWeight> {
    let matrix = weights.checked_matrix(&name)?;
    let shape = matrix.native_shape().context(NativeKernelSnafu)?;
    Ok(ProjectionWeight {
        serialized_bytes: matrix.serialized_bytes().len(),
        name,
        shape,
    })
}

pub(super) fn dimension(elements: usize, context: &'static str) -> Result<u64> {
    u64::try_from(elements).map_err(|_| ArithmeticOverflowSnafu { context }.build())
}

pub(super) fn f32_parameter(
    weights: &Qwen35Weights,
    name: String,
    dimensions: Vec<u64>,
    elements: usize,
) -> Result<F32Parameter> {
    let values = read_f32(weights, &name, &dimensions, elements)?;
    if values.len() != elements {
        return NativeSessionStateSnafu {
            rule: "native F32 parameter descriptor must match verified tensor elements",
        }
        .fail();
    }
    Ok(F32Parameter {
        name,
        dimensions,
        elements,
    })
}

impl WorkspacePlan {
    fn from_layout(layout: Layout) -> Result<Self> {
        let hidden_norm = kernels::decoder_ops::RmsNormF32Plan::try_from_dimensions(
            1,
            layout.hidden,
            layout.epsilon(),
        )
        .context(NativeKernelSnafu)?;
        let query_norm = kernels::decoder_ops::RmsNormF32Plan::try_from_dimensions(
            layout.heads,
            layout.key,
            layout.epsilon(),
        )
        .context(NativeKernelSnafu)?;
        let key_norm = kernels::decoder_ops::RmsNormF32Plan::try_from_dimensions(
            layout.kv_heads,
            layout.key,
            layout.epsilon(),
        )
        .context(NativeKernelSnafu)?;
        let query_rotary = kernels::decoder_ops::RotaryHalfSplitF32Plan::try_from_dimensions(
            layout.heads,
            layout.key,
            layout.text_mrope().rotary_width(),
        )
        .context(NativeKernelSnafu)?;
        let key_rotary = kernels::decoder_ops::RotaryHalfSplitF32Plan::try_from_dimensions(
            layout.kv_heads,
            layout.key,
            layout.text_mrope().rotary_width(),
        )
        .context(NativeKernelSnafu)?;
        let split =
            kernels::decoder_ops::SplitQGateF32Plan::try_from_dimensions(layout.heads, layout.key)
                .context(NativeKernelSnafu)?;
        let gate = kernels::decoder_ops::ElementwiseF32Plan::try_from_elements(layout.query_width)
            .context(NativeKernelSnafu)?;
        Ok(Self {
            hidden_norm,
            query_norm,
            key_norm,
            query_rotary,
            key_rotary,
            split,
            gate,
            hidden: layout.hidden,
            q_gate: split.input_elements(),
            query: split.output_elements(),
            gate_values: split.output_elements(),
            normalized_query: query_norm.elements(),
            key: layout.kv_width,
            normalized_key: key_norm.elements(),
            value: layout.kv_width,
            attention: gate.elements(),
            gated: gate.elements(),
            output_projection: layout.hidden,
        })
    }

    pub(super) fn elements(self) -> Result<usize> {
        sum(
            &[
                self.hidden,
                self.q_gate,
                self.query,
                self.gate_values,
                self.normalized_query,
                self.key,
                self.normalized_key,
                self.value,
                self.attention,
                self.gated,
                self.output_projection,
            ],
            "native scratch elements",
        )
    }

    pub(super) fn coefficient_elements(self) -> Result<usize> {
        self.query_rotary
            .coefficient_elements()
            .checked_mul(2)
            .ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "native mRoPE coefficient controls",
                }
                .build()
            })
    }
}

pub(super) fn elements_bytes(
    elements: usize,
    bytes: usize,
    context: &'static str,
) -> Result<usize> {
    elements
        .checked_mul(bytes)
        .ok_or_else(|| ArithmeticOverflowSnafu { context }.build())
}

pub(super) fn sum(values: &[usize], context: &'static str) -> Result<usize> {
    values.iter().try_fold(0_usize, |total, value| {
        total
            .checked_add(*value)
            .ok_or_else(|| ArithmeticOverflowSnafu { context }.build())
    })
}
