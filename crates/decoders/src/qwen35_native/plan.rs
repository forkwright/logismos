//! Checked native full-attention allocation and weight descriptors.

use core::mem::size_of;

use cache::{NativePagedKvPlan, PagedKvGeometry};
use snafu::ResultExt;

use crate::error::{
    ArithmeticOverflowSnafu, NativeKernelSnafu, NativePagedKvSnafu, NativeSessionStateSnafu,
};
use crate::qwen35_execution::{Layout, block_name, read_f32};
use crate::{Qwen35Weights, Result};

#[derive(Debug)]
pub(crate) struct DeviceFullAttentionPlan {
    pub(crate) layout: Layout,
    pub(crate) block: usize,
    pub(crate) matrices: ProjectionWeights,
    pub(crate) scalars: ScalarWeights,
    pub(crate) workspace: WorkspacePlan,
    pub(crate) kv: NativePagedKvPlan,
    pub(crate) bytes: DeviceByteDemand,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct WorkspacePlan {
    pub(crate) hidden_norm: kernels::decoder_ops::RmsNormF32Plan,
    pub(crate) query_norm: kernels::decoder_ops::RmsNormF32Plan,
    pub(crate) key_norm: kernels::decoder_ops::RmsNormF32Plan,
    pub(crate) query_rotary: kernels::decoder_ops::RotaryHalfSplitF32Plan,
    pub(crate) key_rotary: kernels::decoder_ops::RotaryHalfSplitF32Plan,
    pub(crate) split: kernels::decoder_ops::SplitQGateF32Plan,
    pub(crate) gate: kernels::decoder_ops::ElementwiseF32Plan,
    pub(crate) ffn: kernels::decoder_ops::ElementwiseF32Plan,
    pub(crate) residual: kernels::decoder_ops::ElementwiseF32Plan,
    pub(crate) hidden: usize,
    pub(crate) q_gate: usize,
    pub(crate) query: usize,
    pub(crate) gate_values: usize,
    pub(crate) normalized_query: usize,
    pub(crate) key: usize,
    pub(crate) normalized_key: usize,
    pub(crate) value: usize,
    pub(crate) attention: usize,
    pub(crate) gated: usize,
    pub(crate) output_projection: usize,
    pub(crate) attention_residual: usize,
    pub(crate) post_norm: usize,
    pub(crate) ffn_gate: usize,
    pub(crate) ffn_up: usize,
    pub(crate) ffn_product: usize,
    pub(crate) ffn_down: usize,
}
#[derive(Debug)]
pub(crate) struct ProjectionWeight {
    pub(crate) name: String,
    pub(crate) shape: kernels::RowGemvShape,
    pub(crate) serialized_bytes: usize,
}

#[derive(Debug)]
pub(crate) struct ProjectionWeights {
    pub(crate) q_gate: ProjectionWeight,
    pub(crate) key: ProjectionWeight,
    pub(crate) value: ProjectionWeight,
    pub(crate) output: ProjectionWeight,
    pub(crate) ffn_gate: ProjectionWeight,
    pub(crate) ffn_up: ProjectionWeight,
    pub(crate) ffn_down: ProjectionWeight,
}

#[derive(Debug)]
pub(crate) struct ScalarWeight {
    pub(crate) name: String,
    pub(crate) elements: usize,
}

#[derive(Debug)]
pub(crate) struct ScalarWeights {
    pub(crate) input_norm: ScalarWeight,
    pub(crate) query_norm: ScalarWeight,
    pub(crate) key_norm: ScalarWeight,
    pub(crate) post_attention_norm: ScalarWeight,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DeviceByteDemand {
    pub(crate) weights: usize,
    pub(crate) scratch: usize,
    pub(crate) input: usize,
    pub(crate) output: usize,
    pub(crate) controls: usize,
    pub(crate) key_values: usize,
    pub(crate) table: usize,
}

impl DeviceByteDemand {
    pub(crate) fn total(self) -> Result<usize> {
        sum(&[self.weights, self.scratch, self.input, self.output, self.controls, self.key_values, self.table], "native device byte total")
    }
}

impl DeviceFullAttentionPlan {
    pub(crate) fn from_weights(
        weights: &Qwen35Weights<'_>,
        block: usize,
        max_context: usize,
        page_tokens: kernels::attention::NativePageTokens,
    ) -> Result<Self> {
        let layout = Layout::from_metadata(weights, max_context)?;
        if !layout.is_admitted_full_block(block) {
            return NativeSessionStateSnafu {
                rule: "native plan requires an admitted full-attention main block",
            }
            .fail();
        }
        let matrices = ProjectionWeights::from_weights(weights, block)?;
        let scalars = ScalarWeights::from_weights(weights, layout, block)?;
        let workspace = WorkspacePlan::from_layout(layout)?;
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
        let logical = kernels::PagedDecodePlan::try_from_dimensions(
            layout.max_context(),
            layout.heads,
            layout.kv_heads,
            layout.key,
        )
        .context(NativeKernelSnafu)?;
        let attention = kernels::attention::NativePagedDecodePlan::try_from_paged_decode(
            logical,
            page_tokens.get(),
            kv.layout().physical_pages(),
        )
        .context(NativeKernelSnafu)?;
        let f32_bytes = size_of::<f32>();
        let bytes = DeviceByteDemand {
            weights: sum(
                &[matrices.bytes()?, scalars.bytes()?],
                "native weight bytes",
            )?,
            scratch: elements_bytes(workspace.elements()?, f32_bytes, "native scratch bytes")?,
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
        };
        Ok(Self {
            layout,
            block,
            matrices,
            scalars,
            workspace,
            kv,
            bytes,
        })
    }
}

impl ProjectionWeights {
    fn from_weights(weights: &Qwen35Weights<'_>, block: usize) -> Result<Self> {
        Ok(Self {
            q_gate: projection(weights, block_name(block, "attn_q.weight"))?,
            key: projection(weights, block_name(block, "attn_k.weight"))?,
            value: projection(weights, block_name(block, "attn_v.weight"))?,
            output: projection(weights, block_name(block, "attn_output.weight"))?,
            ffn_gate: projection(weights, block_name(block, "ffn_gate.weight"))?,
            ffn_up: projection(weights, block_name(block, "ffn_up.weight"))?,
            ffn_down: projection(weights, block_name(block, "ffn_down.weight"))?,
        })
    }

    fn bytes(&self) -> Result<usize> {
        sum(
            &[
                self.q_gate.serialized_bytes,
                self.key.serialized_bytes,
                self.value.serialized_bytes,
                self.output.serialized_bytes,
                self.ffn_gate.serialized_bytes,
                self.ffn_up.serialized_bytes,
                self.ffn_down.serialized_bytes,
            ],
            "native serialized projection bytes",
        )
    }
}

impl ScalarWeights {
    fn from_weights(weights: &Qwen35Weights<'_>, layout: Layout, block: usize) -> Result<Self> {
        Ok(Self {
            input_norm: scalar(
                weights,
                block_name(block, "attn_norm.weight"),
                layout.hidden,
            )?,
            query_norm: scalar(weights, block_name(block, "attn_q_norm.weight"), layout.key)?,
            key_norm: scalar(weights, block_name(block, "attn_k_norm.weight"), layout.key)?,
            post_attention_norm: scalar(
                weights,
                block_name(block, "post_attention_norm.weight"),
                layout.hidden,
            )?,
        })
    }

    fn bytes(&self) -> Result<usize> {
        let widths = [
            self.input_norm.elements,
            self.query_norm.elements,
            self.key_norm.elements,
            self.post_attention_norm.elements,
        ];
        elements_bytes(
            sum(&widths, "native scalar weight elements")?,
            size_of::<f32>(),
            "native scalar weight bytes",
        )
    }
}

fn projection(weights: &Qwen35Weights<'_>, name: String) -> Result<ProjectionWeight> {
    let matrix = weights.checked_matrix(&name)?;
    let shape = matrix.native_shape().context(NativeKernelSnafu)?;
    Ok(ProjectionWeight {
        serialized_bytes: matrix.serialized_bytes().len(),
        name,
        shape,
    })
}

fn scalar(weights: &Qwen35Weights<'_>, name: String, elements: usize) -> Result<ScalarWeight> {
    let dimension = u64::try_from(elements).map_err(|_| {
        ArithmeticOverflowSnafu {
            context: "native scalar width",
        }
        .build()
    })?;
    let values = read_f32(weights, &name, &[dimension], elements)?;
    if values.len() != elements {
        return NativeSessionStateSnafu {
            rule: "native scalar descriptor must match verified tensor elements",
        }
        .fail();
    }
    Ok(ScalarWeight { name, elements })
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
        let ffn = kernels::decoder_ops::ElementwiseF32Plan::try_from_elements(layout.feed_forward)
            .context(NativeKernelSnafu)?;
        let residual = kernels::decoder_ops::ElementwiseF32Plan::try_from_elements(layout.hidden)
            .context(NativeKernelSnafu)?;
        Ok(Self {
            hidden_norm,
            query_norm,
            key_norm,
            query_rotary,
            key_rotary,
            split,
            gate,
            ffn,
            residual,
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
            attention_residual: residual.elements(),
            post_norm: hidden_norm.elements(),
            ffn_gate: ffn.elements(),
            ffn_up: ffn.elements(),
            ffn_product: ffn.elements(),
            ffn_down: layout.hidden,
        })
    }

    fn elements(self) -> Result<usize> {
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
                self.attention_residual,
                self.post_norm,
                self.ffn_gate,
                self.ffn_up,
                self.ffn_product,
                self.ffn_down,
            ],
            "native scratch elements",
        )
    }

    fn coefficient_elements(self) -> Result<usize> {
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

fn elements_bytes(elements: usize, bytes: usize, context: &'static str) -> Result<usize> {
    elements
        .checked_mul(bytes)
        .ok_or_else(|| ArithmeticOverflowSnafu { context }.build())
}

fn sum(values: &[usize], context: &'static str) -> Result<usize> {
    values.iter().try_fold(0_usize, |total, value| {
        total
            .checked_add(*value)
            .ok_or_else(|| ArithmeticOverflowSnafu { context }.build())
    })
}
