//! Checked native main-model descriptors without device ownership.

use core::mem::size_of;

use cache::{NativePagedKvPlan, PagedKvGeometry};
use kernels::PackedPrefillPlan;
use snafu::ResultExt;

use super::finish::{LayerFinishPlan, LayerFinishWorkspacePlan};
use super::plan::{
    DeviceFullAttentionPlan, F32Parameter, ProjectionWeight, WorkspacePlan, elements_bytes,
    f32_parameter, native_decode_plan, projection, sum,
};
use super::recurrent_plan::{DeviceRecurrentPlan, RecurrentWorkspacePlan};
use crate::error::{
    ArithmeticOverflowSnafu, ExecutionAllocationSnafu, NativeKernelSnafu, NativePagedKvSnafu,
    NativeSessionStateSnafu,
};
use crate::qwen35_execution::{Layout, OUTPUT, OUTPUT_NORM, TOKEN_EMBEDDING};
use crate::{Qwen35Weights, Result};

/// One verified main block in artifact order, excluding terminal `NextN` blocks.
#[derive(Debug)]
pub(super) enum NativeBlockPlan {
    Full(Box<DeviceFullAttentionPlan>),
    Recurrent(Box<NativeRecurrentBlockPlan>),
}

/// One checked recurrent block and its common finish descriptor.
#[derive(Debug)]
pub(super) struct NativeRecurrentBlockPlan {
    pub(super) plan: DeviceRecurrentPlan,
    pub(super) finish: LayerFinishPlan,
}

/// Checked requested device bytes for one native main-model resource bundle.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(super) struct ModelDeviceByteDemand {
    pub(super) weights: usize,
    pub(super) full_workspace: usize,
    pub(super) recurrent_workspace: usize,
    pub(super) finish_workspace: usize,
    pub(super) hidden_rows: usize,
    pub(super) final_normalized: usize,
    pub(super) logits: usize,
    pub(super) mrope_controls: usize,
    pub(super) key_values: usize,
    pub(super) page_table: usize,
    pub(super) recurrent_history_active: usize,
    pub(super) recurrent_history_staged: usize,
    pub(super) recurrent_state_active: usize,
    pub(super) recurrent_state_staged: usize,
    pub(super) numerical_status: usize,
}

impl ModelDeviceByteDemand {
    pub(super) fn total(self) -> Result<usize> {
        sum(
            &[
                self.weights,
                self.full_workspace,
                self.recurrent_workspace,
                self.finish_workspace,
                self.hidden_rows,
                self.final_normalized,
                self.logits,
                self.mrope_controls,
                self.key_values,
                self.page_table,
                self.recurrent_history_active,
                self.recurrent_history_staged,
                self.recurrent_state_active,
                self.recurrent_state_staged,
                self.numerical_status,
            ],
            "native main-model device bytes",
        )
    }
}

/// Artifact-bound native main-model plan without device allocation or submission.
#[derive(Debug)]
pub(super) struct DeviceModelPlan {
    pub(super) layout: Layout,
    pub(super) page_tokens: kernels::attention::NativePageTokens,
    /// Maximum one-sequence chunk whose workspace is allocated by this plan.
    pub(super) max_chunk_tokens: usize,
    pub(super) embedding: ProjectionWeight,
    pub(super) output: ProjectionWeight,
    pub(super) output_norm: F32Parameter,
    pub(super) output_rms: kernels::decoder_ops::RmsNormF32Plan,
    pub(super) layers: Vec<NativeBlockPlan>,
    pub(super) full_workspace: Option<WorkspacePlan>,
    pub(super) recurrent_workspace: Option<RecurrentWorkspacePlan>,
    pub(super) finish_workspace: LayerFinishWorkspacePlan,
    pub(super) kv: Option<NativePagedKvPlan>,
    pub(super) bytes: ModelDeviceByteDemand,
}

struct ModelBlockPlans {
    layers: Vec<NativeBlockPlan>,
    full_workspace: Option<WorkspacePlan>,
    recurrent_workspace: Option<RecurrentWorkspacePlan>,
    finish_workspace: Option<LayerFinishWorkspacePlan>,
    finish_workspace_bytes: Option<usize>,
    weights_bytes: usize,
    recurrent_history_elements: usize,
    recurrent_state_elements: usize,
    full_layers: usize,
}

impl DeviceModelPlan {
    /// Bind one model's reusable workspace to a checked single-sequence chunk capacity.
    ///
    /// This retains the historical one-token constructor as an exact capacity-one
    /// delegate. Persistent cache and recurrent state remain context- and
    /// layer-shaped; only transient rows and controls scale with this capacity.
    pub(super) fn from_weights_prefill(
        weights: &Qwen35Weights,
        max_context: usize,
        max_chunk_tokens: usize,
        page_tokens: kernels::attention::NativePageTokens,
    ) -> Result<Self> {
        if max_chunk_tokens == 0 || max_chunk_tokens > max_context {
            return NativeSessionStateSnafu {
                rule: "native prefill chunk capacity must be positive and within context",
            }
            .fail();
        }
        let layout = Layout::from_metadata(weights, max_context)?;
        let capacity = PackedPrefillPlan::new(&[max_chunk_tokens], &[0], max_context)
            .context(NativeKernelSnafu)?;
        let embedding = projection(weights, TOKEN_EMBEDDING.to_string())?;
        let output = projection(weights, OUTPUT.to_string())?;
        verify_vocabulary_matrix(
            &embedding,
            layout,
            "native token embedding must have vocabulary rows and hidden width",
        )?;
        verify_vocabulary_matrix(
            &output,
            layout,
            "native output head must have vocabulary rows and hidden width",
        )?;
        let output_norm = f32_parameter(
            weights,
            OUTPUT_NORM.to_string(),
            vec![layout.hidden_dimension()],
            layout.hidden,
        )?;
        let output_rms = kernels::decoder_ops::RmsNormF32Plan::try_from_dimensions(
            1,
            layout.hidden,
            layout.epsilon(),
        )
        .context(NativeKernelSnafu)?;
        let blocks = ModelBlockPlans::from_weights(
            weights,
            layout,
            page_tokens,
            &capacity,
            global_weight_bytes(&embedding, &output, &output_norm)?,
        )?;
        let finish_workspace = blocks.finish_workspace()?;
        let kv = native_kv_plan(layout, blocks.full_layers, page_tokens)?;
        let bytes = ModelDeviceByteDemand::from_model(
            layout,
            max_chunk_tokens,
            output_rms,
            output.shape.rows(),
            &blocks,
            kv,
        )?;
        let _ = bytes.total()?;
        Ok(Self {
            layout,
            page_tokens,
            max_chunk_tokens,
            embedding,
            output,
            output_norm,
            output_rms,
            layers: blocks.layers,
            full_workspace: blocks.full_workspace,
            recurrent_workspace: blocks.recurrent_workspace,
            finish_workspace,
            kv,
            bytes,
        })
    }

    pub(super) fn hidden_row_elements(&self) -> Result<usize> {
        self.layout
            .hidden
            .checked_mul(self.max_chunk_tokens)
            .ok_or_else(|| {
                ArithmeticOverflowSnafu {
                    context: "native prefill hidden row elements",
                }
                .build()
            })
    }
}

impl ModelBlockPlans {
    fn from_weights(
        weights: &Qwen35Weights,
        layout: Layout,
        page_tokens: kernels::attention::NativePageTokens,
        capacity: &PackedPrefillPlan,
        weights_bytes: usize,
    ) -> Result<Self> {
        let mut plans = Self {
            layers: reserve_layers(layout.main_block_count())?,
            full_workspace: None,
            recurrent_workspace: None,
            finish_workspace: None,
            finish_workspace_bytes: None,
            weights_bytes,
            recurrent_history_elements: 0,
            recurrent_state_elements: 0,
            full_layers: 0,
        };
        for block in 0..layout.main_block_count() {
            if layout.is_full(block) {
                plans.push_full(weights, layout, block, page_tokens, capacity.total_tokens())?;
            } else {
                plans.push_recurrent(weights, layout, block, capacity)?;
            }
        }
        Ok(plans)
    }

    fn push_full(
        &mut self,
        weights: &Qwen35Weights,
        layout: Layout,
        block: usize,
        page_tokens: kernels::attention::NativePageTokens,
        token_count: usize,
    ) -> Result<()> {
        let plan = DeviceFullAttentionPlan::from_layout_rows(
            weights,
            layout,
            block,
            page_tokens,
            token_count,
        )?;
        // INVARIANT: the verified structural profile fixes main-block
        // dimensions, so one checked descriptor owns each reusable per-kind
        // workspace extent.
        self.full_workspace.get_or_insert(plan.workspace);
        self.finish_workspace.get_or_insert(plan.finish.workspace);
        self.finish_workspace_bytes
            .get_or_insert(plan.finish.demand.scratch);
        self.weights_bytes = sum(
            &[
                self.weights_bytes,
                plan.matrices.bytes()?,
                plan.norms.bytes()?,
                plan.finish.demand.weights,
            ],
            "native full-attention model weight bytes",
        )?;
        self.full_layers = self.full_layers.checked_add(1).ok_or_else(|| {
            ArithmeticOverflowSnafu {
                context: "native full-attention layer count",
            }
            .build()
        })?;
        self.layers.push(NativeBlockPlan::Full(Box::new(plan)));
        Ok(())
    }

    fn push_recurrent(
        &mut self,
        weights: &Qwen35Weights,
        layout: Layout,
        block: usize,
        capacity: &PackedPrefillPlan,
    ) -> Result<()> {
        let plan = DeviceRecurrentPlan::from_packed_prefill(weights, block, capacity)?;
        let finish =
            LayerFinishPlan::from_weights_rows(weights, layout, block, capacity.total_tokens())?;
        self.recurrent_workspace.get_or_insert(plan.workspace);
        self.finish_workspace.get_or_insert(finish.workspace);
        self.finish_workspace_bytes
            .get_or_insert(finish.demand.scratch);
        self.weights_bytes = sum(
            &[
                self.weights_bytes,
                plan.weight_bytes(),
                finish.demand.weights,
            ],
            "native recurrent model weight bytes",
        )?;
        self.recurrent_history_elements = checked_sum(
            self.recurrent_history_elements,
            plan.convolution_history_elements(),
            "native recurrent convolution history elements",
        )?;
        self.recurrent_state_elements = checked_sum(
            self.recurrent_state_elements,
            plan.recurrent_state_elements(),
            "native recurrent state elements",
        )?;
        self.layers.push(NativeBlockPlan::Recurrent(Box::new(
            NativeRecurrentBlockPlan { plan, finish },
        )));
        Ok(())
    }

    fn finish_workspace(&self) -> Result<LayerFinishWorkspacePlan> {
        self.finish_workspace.ok_or_else(|| {
            NativeSessionStateSnafu {
                rule: "native main model requires at least one main block",
            }
            .build()
        })
    }
}

impl ModelDeviceByteDemand {
    fn from_model(
        layout: Layout,
        max_chunk_tokens: usize,
        output_rms: kernels::decoder_ops::RmsNormF32Plan,
        vocabulary: usize,
        blocks: &ModelBlockPlans,
        kv: Option<NativePagedKvPlan>,
    ) -> Result<Self> {
        Ok(Self {
            weights: blocks.weights_bytes,
            full_workspace: workspace_bytes(
                blocks.full_workspace.as_ref(),
                "native full workspace bytes",
            )?,
            recurrent_workspace: recurrent_workspace_bytes(blocks.recurrent_workspace.as_ref())?,
            finish_workspace: blocks.finish_workspace_bytes.ok_or_else(|| {
                NativeSessionStateSnafu {
                    rule: "native main model requires checked layer-finish workspace bytes",
                }
                .build()
            })?,
            hidden_rows: elements_bytes(
                checked_sum(
                    hidden_row_elements(layout, max_chunk_tokens)?,
                    hidden_row_elements(layout, max_chunk_tokens)?,
                    "native hidden ping-pong rows",
                )?,
                size_of::<f32>(),
                "native hidden ping-pong bytes",
            )?,
            final_normalized: elements_bytes(
                output_rms.elements(),
                size_of::<f32>(),
                "native final normalized bytes",
            )?,
            logits: elements_bytes(vocabulary, size_of::<f32>(), "native logits bytes")?,
            mrope_controls: controls_bytes(blocks.full_workspace.as_ref())?,
            key_values: kv_bytes(kv)?,
            page_table: page_table_bytes(kv, layout)?,
            recurrent_history_active: recurrent_bytes(
                blocks.recurrent_history_elements,
                "native active recurrent convolution history bytes",
            )?,
            recurrent_history_staged: recurrent_bytes(
                blocks.recurrent_history_elements,
                "native staged recurrent convolution history bytes",
            )?,
            recurrent_state_active: recurrent_bytes(
                blocks.recurrent_state_elements,
                "native active recurrent state bytes",
            )?,
            recurrent_state_staged: recurrent_bytes(
                blocks.recurrent_state_elements,
                "native staged recurrent state bytes",
            )?,
            numerical_status: kernels::numerical_status::NativeNumericalStatus::byte_demand(),
        })
    }
}

fn reserve_layers(length: usize) -> Result<Vec<NativeBlockPlan>> {
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(length)
        .context(ExecutionAllocationSnafu {
            target: "native main-model block plans",
            length,
        })?;
    Ok(layers)
}

fn global_weight_bytes(
    embedding: &ProjectionWeight,
    output: &ProjectionWeight,
    output_norm: &F32Parameter,
) -> Result<usize> {
    sum(
        &[
            embedding.serialized_bytes,
            output.serialized_bytes,
            elements_bytes(
                output_norm.elements,
                size_of::<f32>(),
                "native output norm bytes",
            )?,
        ],
        "native global weight bytes",
    )
}

fn checked_sum(left: usize, right: usize, context: &'static str) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| ArithmeticOverflowSnafu { context }.build())
}

fn hidden_row_elements(layout: Layout, max_chunk_tokens: usize) -> Result<usize> {
    layout.hidden.checked_mul(max_chunk_tokens).ok_or_else(|| {
        ArithmeticOverflowSnafu {
            context: "native prefill hidden row elements",
        }
        .build()
    })
}

fn verify_vocabulary_matrix(
    matrix: &ProjectionWeight,
    layout: Layout,
    rule: &'static str,
) -> Result<()> {
    if matrix.shape.rows() != layout.vocabulary() || matrix.shape.width() != layout.hidden {
        return NativeSessionStateSnafu { rule }.fail();
    }
    Ok(())
}

fn workspace_bytes(plan: Option<&WorkspacePlan>, context: &'static str) -> Result<usize> {
    plan.map(|plan| plan.elements())
        .transpose()?
        .map(|elements| elements_bytes(elements, size_of::<f32>(), context))
        .transpose()
        .map(|bytes| bytes.unwrap_or(0))
}

fn recurrent_workspace_bytes(plan: Option<&RecurrentWorkspacePlan>) -> Result<usize> {
    plan.map(|plan| plan.elements())
        .map(|elements| {
            elements_bytes(
                elements,
                size_of::<f32>(),
                "native recurrent workspace bytes",
            )
        })
        .transpose()
        .map(|bytes| bytes.unwrap_or(0))
}

fn controls_bytes(plan: Option<&WorkspacePlan>) -> Result<usize> {
    plan.map(|plan| plan.coefficient_elements())
        .transpose()?
        .map(|elements| elements_bytes(elements, size_of::<f32>(), "native mRoPE control bytes"))
        .transpose()
        .map(|bytes| bytes.unwrap_or(0))
}

fn recurrent_bytes(elements: usize, context: &'static str) -> Result<usize> {
    elements_bytes(elements, size_of::<f32>(), context)
}

fn native_kv_plan(
    layout: Layout,
    full_layers: usize,
    page_tokens: kernels::attention::NativePageTokens,
) -> Result<Option<NativePagedKvPlan>> {
    if full_layers == 0 {
        return Ok(None);
    }
    NativePagedKvPlan::try_from_geometry(
        PagedKvGeometry {
            layers: full_layers,
            row_width: layout.kv_width,
            max_context: layout.max_context(),
        },
        layout.kv_heads,
        layout.key,
        page_tokens,
    )
    .map(Some)
    .context(NativePagedKvSnafu)
}

fn kv_bytes(plan: Option<NativePagedKvPlan>) -> Result<usize> {
    let Some(plan) = plan else {
        return Ok(0);
    };
    let both_backings = plan
        .layout()
        .backing_elements()
        .checked_mul(2)
        .ok_or_else(|| {
            ArithmeticOverflowSnafu {
                context: "native main-model separate K/V backing",
            }
            .build()
        })?;
    elements_bytes(
        both_backings,
        size_of::<f32>(),
        "native main-model K/V bytes",
    )
}

fn page_table_bytes(plan: Option<NativePagedKvPlan>, layout: Layout) -> Result<usize> {
    let Some(plan) = plan else {
        return Ok(0);
    };
    let native = native_decode_plan(layout, layout.max_context(), plan)?;
    elements_bytes(
        native.page_table_entries(),
        size_of::<u32>(),
        "native main-model page table bytes",
    )
}

#[cfg(test)]
mod tests {
    use super::{DeviceModelPlan, ModelDeviceByteDemand, NativeBlockPlan};
    use crate::Qwen35Weights;
    use crate::qwen35::tests::{
        canonical_hybrid_fixture, canonical_hybrid_fixture_with_nextn, verify_fixture,
    };
    use crate::qwen35_execution::{OUTPUT, OUTPUT_NORM, TOKEN_EMBEDDING};

    const CONTEXT: usize = 4;
    const PAGE_TOKENS: kernels::attention::NativePageTokens =
        kernels::attention::NativePageTokens::B8;

    fn model_plan(
        fixture: &crate::qwen35::tests::Fixture,
    ) -> std::result::Result<DeviceModelPlan, String> {
        let artifact = verify_fixture(fixture)?;
        let weights =
            Qwen35Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;
        DeviceModelPlan::from_weights_prefill(&weights, CONTEXT, 1, PAGE_TOKENS)
            .map_err(|error| error.to_string())
    }

    #[test]
    fn main_model_plan_binds_terminal_roles_and_only_main_blocks() -> std::result::Result<(), String>
    {
        let plan = model_plan(&canonical_hybrid_fixture()?)?;
        let auxiliary = model_plan(&canonical_hybrid_fixture_with_nextn(CONTEXT, None, 0.25)?)?;

        assert_eq!(plan.embedding.name, TOKEN_EMBEDDING);
        assert_eq!(plan.output.name, OUTPUT);
        assert_ne!(
            plan.embedding.name, plan.output.name,
            "native main model must retain its verified distinct output head"
        );
        assert_eq!(plan.output_norm.name, OUTPUT_NORM);
        assert_eq!(plan.layers.len(), plan.layout.main_block_count());
        assert_eq!(plan.layers.len(), 4, "terminal NextN must be excluded");
        assert!(matches!(&plan.layers[0], NativeBlockPlan::Recurrent(_)));
        assert!(matches!(&plan.layers[1], NativeBlockPlan::Recurrent(_)));
        assert!(matches!(&plan.layers[2], NativeBlockPlan::Recurrent(_)));
        assert!(matches!(&plan.layers[3], NativeBlockPlan::Full(_)));
        assert!(plan.full_workspace.is_some());
        assert!(plan.recurrent_workspace.is_some());
        assert!(plan.kv.is_some());
        let expected = ModelDeviceByteDemand {
            // 3 recurrent trunks × 660 + one full block 25_804 + global 132.
            weights: 28_492,
            // One full workspace, without the one common layer-finish workspace.
            full_workspace: 17_432,
            // 118 trunk values plus two simultaneous Hv=4, V=2 layout buffers.
            recurrent_workspace: 536,
            // One common layer-finish workspace: 24 f32 values.
            finish_workspace: 96,
            // Two H=3 rows, then one final normalized H=3 row and V=5 logits.
            hidden_rows: 24,
            final_normalized: 12,
            logits: 20,
            // Hq=2, K=2, rotary width defaults to 128: 128 f32 controls.
            mrope_controls: 512,
            // B8 selected K/V backing and its one u32 physical-page entry.
            key_values: 32_768,
            page_table: 4,
            // Three recurrent layers × 16 f32 elements, active and staged.
            recurrent_history_active: 192,
            recurrent_history_staged: 192,
            recurrent_state_active: 192,
            recurrent_state_staged: 192,
            numerical_status: kernels::numerical_status::NativeNumericalStatus::byte_demand(),
        };
        assert_eq!(
            plan.bytes, expected,
            "canonical main-model demand must use shared workspaces and all staged state"
        );
        assert_eq!(
            plan.bytes.total().map_err(|error| error.to_string())?,
            80_664_usize
                .checked_add(kernels::numerical_status::NativeNumericalStatus::byte_demand())
                .ok_or("hand-derived numerical status total overflow")?,
            "hand-derived canonical model-device allocation"
        );
        assert_eq!(
            auxiliary.bytes, plan.bytes,
            "a structurally valid terminal NextN block must not reserve main-model bytes"
        );
        assert_eq!(auxiliary.layers.len(), plan.layers.len());
        Ok(())
    }

    #[test]
    fn terminal_nextn_values_do_not_change_main_model_demand() -> std::result::Result<(), String> {
        let baseline = model_plan(&canonical_hybrid_fixture()?)?;
        let first = model_plan(&canonical_hybrid_fixture_with_nextn(CONTEXT, None, 0.25)?)?;
        let second = model_plan(&canonical_hybrid_fixture_with_nextn(CONTEXT, None, -0.75)?)?;

        assert_eq!(first.bytes, baseline.bytes);
        assert_eq!(first.bytes, second.bytes);
        assert_eq!(
            first.bytes.total().map_err(|error| error.to_string())?,
            second.bytes.total().map_err(|error| error.to_string())?,
            "terminal NextN payload changes must not allocate native main-model bytes"
        );
        assert_eq!(first.layers.len(), baseline.layers.len());
        assert_eq!(second.layers.len(), baseline.layers.len());
        Ok(())
    }

    #[test]
    fn main_model_plan_refuses_zero_context_before_any_device_allocation()
    -> std::result::Result<(), String> {
        let fixture = canonical_hybrid_fixture()?;
        let artifact = verify_fixture(&fixture)?;
        let weights =
            Qwen35Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;
        assert!(
            DeviceModelPlan::from_weights_prefill(&weights, 0, 1, PAGE_TOKENS).is_err(),
            "the model plan must refuse an empty caller context before device allocation"
        );
        Ok(())
    }

    #[test]
    fn prefill_capacity_scales_only_transient_rows_controls_and_workspaces()
    -> std::result::Result<(), String> {
        const CAPACITY: usize = 3;
        let fixture = canonical_hybrid_fixture()?;
        let artifact = verify_fixture(&fixture)?;
        let weights =
            Qwen35Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;
        let token = DeviceModelPlan::from_weights_prefill(&weights, CONTEXT, 1, PAGE_TOKENS)
            .map_err(|error| error.to_string())?;
        let chunk = DeviceModelPlan::from_weights_prefill(&weights, CONTEXT, CAPACITY, PAGE_TOKENS)
            .map_err(|error| error.to_string())?;

        assert_eq!(token.max_chunk_tokens, 1);
        assert_eq!(chunk.max_chunk_tokens, CAPACITY);
        assert_eq!(
            chunk
                .hidden_row_elements()
                .map_err(|error| error.to_string())?,
            9
        );
        assert_eq!(chunk.bytes.weights, token.bytes.weights);
        assert_eq!(chunk.bytes.key_values, token.bytes.key_values);
        assert_eq!(chunk.bytes.page_table, token.bytes.page_table);
        assert_eq!(
            chunk.bytes.recurrent_history_active,
            token.bytes.recurrent_history_active
        );
        assert_eq!(
            chunk.bytes.recurrent_history_staged,
            token.bytes.recurrent_history_staged
        );
        assert_eq!(
            chunk.bytes.recurrent_state_active,
            token.bytes.recurrent_state_active
        );
        assert_eq!(
            chunk.bytes.recurrent_state_staged,
            token.bytes.recurrent_state_staged
        );
        assert_eq!(chunk.bytes.final_normalized, token.bytes.final_normalized);
        assert_eq!(chunk.bytes.logits, token.bytes.logits);
        assert_eq!(
            chunk.bytes.full_workspace,
            token.bytes.full_workspace * CAPACITY
        );
        assert_eq!(
            chunk.bytes.recurrent_workspace,
            token.bytes.recurrent_workspace * CAPACITY
        );
        assert_eq!(
            chunk.bytes.finish_workspace,
            token.bytes.finish_workspace * CAPACITY
        );
        assert_eq!(chunk.bytes.hidden_rows, token.bytes.hidden_rows * CAPACITY);
        let token_control_elements = token
            .full_workspace
            .ok_or("canonical model must retain full-attention workspace")?
            .coefficient_elements()
            .map_err(|error| error.to_string())?;
        let chunk_control_elements = chunk
            .full_workspace
            .ok_or("canonical model must retain full-attention workspace")?
            .coefficient_elements()
            .map_err(|error| error.to_string())?;
        assert_eq!(
            chunk_control_elements,
            token_control_elements * CAPACITY,
            "the capacity workspace owner must account for one token-major cosine/sine row per admitted token"
        );
        assert_eq!(
            chunk.bytes.mrope_controls,
            elements_bytes(
                chunk_control_elements,
                size_of::<f32>(),
                "test native mRoPE controls",
            )
            .map_err(|error| error.to_string())?,
            "mRoPE byte demand must derive directly from its capacity workspace owner"
        );
        assert_eq!(
            chunk.bytes.mrope_controls,
            token.bytes.mrope_controls * CAPACITY,
            "capacity-three controls must be exactly three capacity-one control spans, never squared"
        );
        Ok(())
    }

    #[test]
    fn prefill_capacity_refuses_zero_or_context_excess_before_device_allocation()
    -> std::result::Result<(), String> {
        let fixture = canonical_hybrid_fixture()?;
        let artifact = verify_fixture(&fixture)?;
        let weights =
            Qwen35Weights::try_from_verified(&artifact).map_err(|error| error.to_string())?;

        assert!(DeviceModelPlan::from_weights_prefill(&weights, CONTEXT, 0, PAGE_TOKENS).is_err());
        assert!(
            DeviceModelPlan::from_weights_prefill(&weights, CONTEXT, CONTEXT + 1, PAGE_TOKENS)
                .is_err()
        );
        Ok(())
    }
}
