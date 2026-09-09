//! Checked native full-attention allocation and weight descriptors.

use core::mem::size_of;

use cache::{NativePagedKvPlan, PagedKvGeometry};
use snafu::ResultExt;

use crate::error::{
    ArithmeticOverflowSnafu, NativeKernelSnafu, NativePagedKvSnafu, NativeSessionStateSnafu,
};
use crate::qwen35_execution::{Layout, block_name, read_f32};
use crate::{Qwen35Weights, Result};

const PAGE_TOKENS: kernels::attention::NativePageTokens = kernels::attention::NativePageTokens::B8;

#[derive(Debug)]
pub(crate) struct DeviceFullAttentionPlan {
    pub(crate) layout: Layout,
    pub(crate) block: usize,
    pub(crate) matrices: ProjectionWeights,
    pub(crate) scalars: ScalarWeights,
    pub(crate) kv: NativePagedKvPlan,
    pub(crate) bytes: DeviceByteDemand,
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

impl DeviceFullAttentionPlan {
    pub(crate) fn from_weights(
        weights: &Qwen35Weights<'_>,
        block: usize,
        max_context: usize,
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
        let kv = NativePagedKvPlan::try_from_geometry(
            PagedKvGeometry {
                layers: 1,
                row_width: layout.kv_width,
                max_context: layout.max_context(),
            },
            layout.kv_heads,
            layout.key,
            PAGE_TOKENS,
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
            PAGE_TOKENS.get(),
            kv.layout().physical_pages(),
        )
        .context(NativeKernelSnafu)?;
        let f32_bytes = size_of::<f32>();
        let bytes = DeviceByteDemand {
            weights: sum(
                &[matrices.bytes()?, scalars.bytes()?],
                "native weight bytes",
            )?,
            scratch: elements_bytes(scratch_elements(layout)?, f32_bytes, "native scratch bytes")?,
            input: elements_bytes(layout.hidden, f32_bytes, "native input bytes")?,
            output: elements_bytes(layout.hidden, f32_bytes, "native output bytes")?,
            controls: elements_bytes(control_elements(layout)?, f32_bytes, "native control bytes")?,
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

fn scratch_elements(layout: Layout) -> Result<usize> {
    sum(
        &[
            layout.hidden.checked_mul(6).ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "native hidden scratch",
                }
                .build()
            })?,
            layout.query_width.checked_mul(5).ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "native query scratch",
                }
                .build()
            })?,
            layout.kv_width.checked_mul(3).ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "native KV scratch",
                }
                .build()
            })?,
            layout.feed_forward.checked_mul(3).ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "native FFN scratch",
                }
                .build()
            })?,
        ],
        "native scratch elements",
    )
}

fn control_elements(layout: Layout) -> Result<usize> {
    layout
        .text_mrope()
        .rotary_width()
        .checked_mul(1)
        .and_then(|width| width.checked_div(2))
        .and_then(|pairs| pairs.checked_mul(2))
        .ok_or_else(|| {
            ArithmeticOverflowSnafu {
                context: "native mRoPE coefficient controls",
            }
            .build()
        })
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
