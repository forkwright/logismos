//! Atomic aggregate ownership for independent bounded CPU Qwen3.5 sessions.

use cache::{PagedAppend, PagedPreparedCommit};
use kernels::PackedPrefillPlan;
use snafu::ResultExt;

use super::{LayerState, Qwen35Execution, StagedExecution, reserve};
use crate::Result;
use crate::error::{ExecutionBatchSnafu, ExecutionCpuSnafu, ExecutionPagedKvSnafu};
use crate::qwen35_requirements::Qwen35BatchCpuRequirements;

/// Opaque atomic plan for independently owned bounded CPU Qwen3.5 sessions.
///
/// The plan borrows every execution exclusively, validates every sequence
/// before staging, and consumes itself to publish every owner together. Equal
/// artifact content is an admission relation only: it neither proves nor
/// creates allocation aliasing. Recurrent state and K/V pools remain private;
/// immutable serialized backing may already be shared or separately allocated.
#[derive(Debug)]
pub struct Qwen35BatchExecutionPlan<'execution, 'tokens> {
    executions: &'execution mut [Qwen35Execution],
    token_ids: &'tokens [&'tokens [u32]],
    packed: PackedPrefillPlan,
    requirements: Qwen35BatchCpuRequirements,
}

impl Qwen35Execution {
    /// Plan one atomic transaction across independently owned CPU sessions.
    ///
    /// Every sequence must be nonempty, fit its own step and context bounds,
    /// contain vocabulary token ids, and bind the same verified content digest
    /// and serialized length. Session-private allocation ownership remains
    /// independent even when content is equal.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] before recurrent cloning or K/V reservation if
    /// aggregate shape, individual session bounds, token ids, content identity,
    /// packed geometry, or conservative requirement arithmetic is invalid.
    pub fn plan_batch<'execution, 'tokens>(
        executions: &'execution mut [Self],
        token_ids: &'tokens [&'tokens [u32]],
    ) -> Result<Qwen35BatchExecutionPlan<'execution, 'tokens>> {
        if executions.is_empty() {
            return ExecutionBatchSnafu {
                sequence: 0_usize,
                rule: "batch must contain at least one independently owned execution",
            }
            .fail();
        }
        if executions.len() != token_ids.len() {
            return ExecutionBatchSnafu {
                sequence: token_ids.len(),
                rule: "execution and token-input sequence counts must agree",
            }
            .fail();
        }
        let reference = executions.first().ok_or_else(|| {
            ExecutionBatchSnafu {
                sequence: 0_usize,
                rule: "batch must retain its first execution receipt",
            }
            .build()
        })?;
        let reference_digest = reference.requirements.artifact_digest();
        let reference_length = reference.requirements.serialized_backing_bytes();
        let mut lengths = reserve("batch packed sequence lengths", executions.len())?;
        let mut offsets = reserve("batch packed sequence offsets", executions.len())?;
        let mut max_context = 0_usize;

        for (sequence, (execution, tokens)) in executions.iter().zip(token_ids).enumerate() {
            execution.validate_step(tokens)?;
            if execution.requirements.artifact_digest() != reference_digest
                || execution.requirements.serialized_backing_bytes() != reference_length
            {
                return ExecutionBatchSnafu {
                    sequence,
                    rule: "all executions must bind equal verified content digest and serialized length",
                }
                .fail();
            }
            lengths.push(tokens.len());
            offsets.push(execution.position);
            max_context = max_context.max(execution.layout.max_context());
        }
        let packed =
            PackedPrefillPlan::new(&lengths, &offsets, max_context).context(ExecutionCpuSnafu)?;
        let requirements = Qwen35BatchCpuRequirements::try_from_receipts(
            executions.iter().map(|execution| execution.requirements),
            &packed,
        )?;
        Ok(Qwen35BatchExecutionPlan {
            executions,
            token_ids,
            packed,
            requirements,
        })
    }
}

impl Qwen35BatchExecutionPlan<'_, '_> {
    /// Return the aggregate logical CPU backing envelope derived at admission.
    #[must_use]
    pub const fn cpu_requirements(&self) -> Qwen35BatchCpuRequirements {
        self.requirements
    }

    /// Return the checked sequence-major geometry driving this transaction.
    ///
    /// The descriptor validates input layout only; it grants neither physical
    /// capacity nor a native/vectorized execution capability.
    #[must_use]
    pub const fn packed_prefill_plan(&self) -> &PackedPrefillPlan {
        &self.packed
    }

    /// Execute and publish every independently staged session atomically.
    ///
    /// Returned vectors preserve input sequence order; each inner vector obeys
    /// its owner plan's logit-selection contract.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] without publishing any cache, recurrent layer,
    /// or position if a private staged operation, allocation, or cache
    /// preparation fails. Once every append is prepared, publication is
    /// infallible and allocation-free.
    pub fn execute(self) -> Result<Vec<Vec<f32>>> {
        let Self {
            executions,
            token_ids,
            packed,
            requirements: _,
        } = self;
        let sequence_count = packed.sequence_count();
        let mut pending = reserve("batch staged execution owners", sequence_count)?;
        stage_batch(executions, token_ids, &packed, &mut pending)?;

        let mut prepared_owners = reserve("batch prepared execution owners", sequence_count)?;
        let mut results = reserve("batch grouped logits", sequence_count)?;
        for pending in pending {
            let (prepared_owner, logits) = pending.prepare()?;
            prepared_owners.push(prepared_owner);
            results.push(logits);
        }

        for prepared_owner in prepared_owners {
            prepared_owner.publish();
        }
        Ok(results)
    }
}

fn stage_batch<'execution>(
    executions: &'execution mut [Qwen35Execution],
    token_ids: &[&[u32]],
    packed: &PackedPrefillPlan,
    pending: &mut Vec<PendingExecution<'execution>>,
) -> Result<()> {
    if executions.len() != token_ids.len() {
        return ExecutionBatchSnafu {
            sequence: token_ids.len(),
            rule: "staged execution and token-input sequences must remain aligned",
        }
        .fail();
    }
    for (sequence, (execution, tokens)) in executions.iter_mut().zip(token_ids).enumerate() {
        pending.push(PendingExecution::stage(
            execution, tokens, packed, sequence,
        )?);
    }
    Ok(())
}

pub(super) struct PendingExecution<'execution> {
    target_layers: &'execution mut Vec<LayerState>,
    target_position: &'execution mut usize,
    staged: StagedExecution,
    append: Option<PagedAppend<'execution>>,
    logits: Vec<f32>,
}

impl<'execution> PendingExecution<'execution> {
    pub(super) fn stage(
        execution: &'execution mut Qwen35Execution,
        token_ids: &[u32],
        packed: &PackedPrefillPlan,
        sequence: usize,
    ) -> Result<Self> {
        let mut staged = execution.stage()?;
        let (target_layers, target_position, paged_kv_pool) = (
            &mut execution.layers,
            &mut execution.position,
            &mut execution.paged_kv_pool,
        );
        let mut append = paged_kv_pool
            .as_mut()
            .map(|pool| {
                pool.begin_append(token_ids.len())
                    .context(ExecutionPagedKvSnafu)
            })
            .transpose()?;
        let logits = staged.step_staged(token_ids, packed, sequence, append.as_mut())?;
        Ok(Self {
            target_layers,
            target_position,
            staged,
            append,
            logits,
        })
    }

    pub(super) fn prepare(self) -> Result<(PreparedExecution<'execution>, Vec<f32>)> {
        let Self {
            target_layers,
            target_position,
            staged,
            append,
            logits,
        } = self;
        let prepared_commit = append
            .map(|append| append.prepare_commit().context(ExecutionPagedKvSnafu))
            .transpose()?;
        Ok((
            PreparedExecution {
                target_layers,
                target_position,
                staged,
                prepared_commit,
            },
            logits,
        ))
    }
}

pub(super) struct PreparedExecution<'execution> {
    target_layers: &'execution mut Vec<LayerState>,
    target_position: &'execution mut usize,
    staged: StagedExecution,
    prepared_commit: Option<PagedPreparedCommit<'execution>>,
}

impl PreparedExecution<'_> {
    pub(super) fn publish(self) {
        let Self {
            target_layers,
            target_position,
            staged,
            prepared_commit,
        } = self;
        if let Some(prepared_commit) = prepared_commit {
            prepared_commit.commit();
        }
        let StagedExecution {
            layers, position, ..
        } = staged;
        *target_layers = layers;
        *target_position = position;
    }
}
