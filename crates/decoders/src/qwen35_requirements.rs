//! Executor-owned logical CPU allocation requirements for Qwen3.5.

use cache::PagedKvPlan;
use loader::gguf::ArtifactDigest;

use crate::error::ArithmeticOverflowSnafu;
use crate::qwen35::Qwen35RecurrentLayout;
use crate::qwen35_execution::{
    Layout, Qwen35LogitSelection, embedding_workspace_elements, full_attention_retained_elements,
    full_attention_workspace_elements, layer_finish_workspace_elements, lm_head_workspace_elements,
    returned_logits_elements,
};
use crate::{Qwen35RecurrentExecution, Qwen35Weights, Result};

/// Checked logical CPU backing envelope for one artifact-bound execution plan.
///
/// The `f32` fields count exact requested `Vec<f32>` capacities composed into a
/// conservative phase upper bound. They do not describe allocator capacity,
/// allocator metadata, stack or structure storage, process RSS, tokenizer or
/// template storage, physical reservations, or GPU memory. The verified
/// serialized artifact backing is reported separately and is not included in
/// the logical `f32` total.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35CpuRequirements {
    artifact_digest: ArtifactDigest,
    serialized_backing_bytes: u64,
    retained_bytes: u64,
    transaction_copy_bytes: u64,
    workspace_upper_bound_bytes: u64,
    returned_logits_bytes: u64,
    logical_f32_upper_bound_bytes: u64,
    max_context: usize,
    max_step_tokens: usize,
    selection: Qwen35LogitSelection,
}

impl Qwen35CpuRequirements {
    pub(crate) fn try_from_plan(
        weights: &Qwen35Weights,
        layout: Layout,
        max_step_tokens: usize,
        selection: Qwen35LogitSelection,
        paged_kv_plan: Option<PagedKvPlan>,
    ) -> Result<Self> {
        let elements = Qwen35RequirementElements::try_from_layout(
            layout,
            weights.recurrent_layout(),
            max_step_tokens,
            selection,
            paged_kv_plan,
        )?;
        Ok(Self {
            artifact_digest: weights.payload().observation().inspection().digest,
            serialized_backing_bytes: weights.payload().observation().inspection().file_len,
            retained_bytes: f32_bytes(elements.retained)?,
            transaction_copy_bytes: f32_bytes(elements.transaction_copy)?,
            workspace_upper_bound_bytes: f32_bytes(elements.workspace_upper_bound)?,
            returned_logits_bytes: f32_bytes(elements.returned_logits)?,
            logical_f32_upper_bound_bytes: f32_bytes(elements.logical_upper_bound)?,
            max_context: layout.max_context(),
            max_step_tokens,
            selection,
        })
    }

    /// Return the digest identifying the verified serialized bytes bound here.
    ///
    /// The digest identifies byte content; it does not authenticate a publisher
    /// or establish source provenance.
    #[must_use]
    pub const fn artifact_digest(self) -> ArtifactDigest {
        self.artifact_digest
    }

    /// Return the verified serialized GGUF backing length.
    ///
    /// This is immutable artifact backing, not decoded `f32` workspace, and is
    /// excluded from [`Self::logical_f32_upper_bound_bytes`].
    #[must_use]
    pub const fn serialized_backing_bytes(self) -> u64 {
        self.serialized_backing_bytes
    }

    /// Return retained executor `f32` backing, excluding artifact bytes.
    ///
    /// This includes decoded recurrent parameters, mutable recurrent history
    /// and state, and the selected KV pool's padded maximum-context backing
    /// plus its preallocated tail-copy spare.
    #[must_use]
    pub const fn retained_bytes(self) -> u64 {
        self.retained_bytes
    }

    /// Return separately allocated recurrent transaction-copy `f32` backing.
    ///
    /// Paged KV copy-on-write uses the already-retained spare, so it adds no
    /// allocation here. A mixed recurrent/full-attention plan therefore has
    /// less transaction-copy backing than retained backing.
    #[must_use]
    pub const fn transaction_copy_bytes(self) -> u64 {
        self.transaction_copy_bytes
    }

    /// Return the conservative upper bound for transient executor `f32` backing.
    ///
    /// This is the maximum of named execution phases, with conservative sums
    /// inside each phase. It is not a measurement of allocator capacity or RSS.
    #[must_use]
    pub const fn workspace_upper_bound_bytes(self) -> u64 {
        self.workspace_upper_bound_bytes
    }

    /// Return the selected returned-logit `f32` backing.
    #[must_use]
    pub const fn returned_logits_bytes(self) -> u64 {
        self.returned_logits_bytes
    }

    /// Return the complete executor-owned logical `f32` upper bound.
    ///
    /// This is retained backing plus separately allocated recurrent copies,
    /// transient workspace, and returned logits. The KV pool is counted once;
    /// serialized artifact backing is deliberately separate.
    #[must_use]
    pub const fn logical_f32_upper_bound_bytes(self) -> u64 {
        self.logical_f32_upper_bound_bytes
    }

    /// Return this plan's caller context bound.
    #[must_use]
    pub const fn max_context(self) -> usize {
        self.max_context
    }

    /// Return this plan's maximum accepted step-token count.
    #[must_use]
    pub const fn max_step_tokens(self) -> usize {
        self.max_step_tokens
    }

    /// Return this plan's typed returned-logit selection.
    #[must_use]
    pub const fn selection(self) -> Qwen35LogitSelection {
        self.selection
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Qwen35RequirementElements {
    pub(crate) recurrent_layer_retained: usize,
    pub(crate) full_attention_pool_retained: usize,
    pub(crate) retained: usize,
    pub(crate) transaction_copy: usize,
    pub(crate) recurrent_workspace: usize,
    pub(crate) full_attention_workspace: usize,
    pub(crate) layer_finish_workspace: usize,
    pub(crate) lm_head_workspace: usize,
    pub(crate) workspace_upper_bound: usize,
    pub(crate) returned_logits: usize,
    pub(crate) logical_upper_bound: usize,
}

impl Qwen35RequirementElements {
    #[expect(
        clippy::too_many_lines,
        reason = "one checked composition keeps the complete named f32 owner inventory auditable"
    )]
    pub(crate) fn try_from_layout(
        layout: Layout,
        recurrent_layout: Qwen35RecurrentLayout,
        max_step_tokens: usize,
        selection: Qwen35LogitSelection,
        paged_kv_plan: Option<PagedKvPlan>,
    ) -> Result<Self> {
        let full_layers = layout.full_layer_count();
        let recurrent_layers = layout.recurrent_layer_count()?;
        let recurrent_layer_retained = if recurrent_layers == 0 {
            0
        } else {
            Qwen35RecurrentExecution::retained_elements(recurrent_layout, layout.epsilon())?
        };
        let full_attention_pool_retained = match (full_layers, paged_kv_plan) {
            (0, None) => 0,
            (0, Some(_)) => {
                return ArithmeticOverflowSnafu {
                    context: "paged KV plan without full-attention layers",
                }
                .fail();
            }
            (_, Some(plan)) => full_attention_retained_elements(plan),
            (_, None) => {
                return ArithmeticOverflowSnafu {
                    context: "full-attention layers without a paged KV plan",
                }
                .fail();
            }
        };
        let retained = checked_add(
            checked_product(
                recurrent_layers,
                recurrent_layer_retained,
                "all recurrent retained allocations",
            )?,
            full_attention_pool_retained,
            "all retained executor allocations",
        )?;
        let transaction_copy = checked_product(
            recurrent_layers,
            recurrent_layer_retained,
            "staged recurrent transaction allocations",
        )?;

        // The model executor feeds one hidden row at a time through every
        // block, even when the public step accepts several token ids.
        let recurrent_workspace = if recurrent_layers == 0 {
            0
        } else {
            Qwen35RecurrentExecution::workspace_elements(recurrent_layout, layout.epsilon(), 1)?
        };
        let full_attention_workspace = if full_layers == 0 {
            0
        } else {
            full_attention_workspace_elements(layout, layout.max_context())?
        };
        let layer_finish_workspace = if layout.full_layer_count() == 0 && recurrent_layers == 0 {
            0
        } else {
            layer_finish_workspace_elements(layout)?
        };
        // The token-major hidden row remains owned by `step_staged` while an
        // attention/recurrent owner executes and while its returned attention
        // row, norms, and feed-forward owner execute. The latter live set is
        // fully represented by `layer_finish_workspace`.
        let token_hidden = embedding_workspace_elements(layout);
        let recurrent_layer_phase = if recurrent_layers == 0 {
            0
        } else {
            checked_add(
                token_hidden,
                recurrent_workspace.max(layer_finish_workspace),
                "recurrent layer workspace phase",
            )?
        };
        let full_layer_phase = if full_layers == 0 {
            0
        } else {
            checked_add(
                token_hidden,
                full_attention_workspace.max(layer_finish_workspace),
                "full-attention layer workspace phase",
            )?
        };
        let lm_head_workspace = lm_head_workspace_elements(layout)?;
        let lm_head_phase =
            checked_add(token_hidden, lm_head_workspace, "LM-head workspace phase")?;
        let workspace_upper_bound = [
            token_hidden,
            recurrent_layer_phase,
            full_layer_phase,
            lm_head_phase,
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        let returned_logits = returned_logits_elements(layout, max_step_tokens, selection)?;
        let logical_upper_bound = sum_elements(
            &[
                retained,
                transaction_copy,
                workspace_upper_bound,
                returned_logits,
            ],
            "logical f32 upper bound",
        )?;
        Ok(Self {
            recurrent_layer_retained,
            full_attention_pool_retained,
            retained,
            transaction_copy,
            recurrent_workspace,
            full_attention_workspace,
            layer_finish_workspace,
            lm_head_workspace,
            workspace_upper_bound,
            returned_logits,
            logical_upper_bound,
        })
    }
}

fn f32_bytes(elements: usize) -> Result<u64> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            ArithmeticOverflowSnafu {
                context: "logical f32 bytes",
            }
            .build()
        })?;
    u64::try_from(bytes).map_err(|_| {
        ArithmeticOverflowSnafu {
            context: "logical f32 byte representation",
        }
        .build()
    })
}

fn checked_product(left: usize, right: usize, context: &'static str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| ArithmeticOverflowSnafu { context }.build())
}

fn checked_add(left: usize, right: usize, context: &'static str) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| ArithmeticOverflowSnafu { context }.build())
}

fn sum_elements(elements: &[usize], context: &'static str) -> Result<usize> {
    elements
        .iter()
        .copied()
        .try_fold(0_usize, |sum, value| checked_add(sum, value, context))
}
