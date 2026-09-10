//! Atomic ownership for independently mutable native model sessions.

use hipcore::DeviceBuffer;
use kernels::PackedPrefillPlan;
use snafu::ResultExt;
use std::sync::Arc;

use super::{
    Qwen35NativeExecutionBatchDeviceDemand, Qwen35NativeExecutionDeviceDemand,
    Qwen35NativeExecutionSession,
};
use crate::Result;
use crate::error::{ExecutionAllocationSnafu, NativeKernelSnafu, NativeSessionStateSnafu};
use crate::qwen35_native::PostCompletion;

use super::super::model_step::ModelChunkPlan;
use super::super::session::{begin_error, completion_error};
use super::ModelSessionResources;

/// Opaque atomic plan over independently owned native model sessions.
///
/// Admission proves every session retains the same resident `Arc` allocation,
/// while preserving each session's own context, chunk capacity, stream, cache,
/// recurrent state, and committed position. It is an unsafe blocking native
/// executor capability, not a vectorized kernel, serving, or capacity grant.
pub struct Qwen35NativeExecutionBatchPlan<'session, 'tokens> {
    sessions: &'session mut [Qwen35NativeExecutionSession],
    chunks: Vec<ModelChunkPlan>,
    packed: PackedPrefillPlan,
    demand: Qwen35NativeExecutionBatchDeviceDemand,
    token_ids: &'tokens [&'tokens [u32]],
}

impl Qwen35NativeExecutionSession {
    /// Plan one native atomic transaction across shared-resident sessions.
    ///
    /// # Errors
    ///
    /// Refuses empty or mismatched aggregates, poisoned owners, unequal resident
    /// allocation identities, exact per-session bounds or token rows, and all
    /// packed/demand arithmetic before device allocation, reservation, or submit.
    pub fn plan_batch<'session, 'tokens>(
        sessions: &'session mut [Self],
        token_ids: &'tokens [&'tokens [u32]],
    ) -> Result<Qwen35NativeExecutionBatchPlan<'session, 'tokens>> {
        let first = sessions
            .first()
            .filter(|_| sessions.len() == token_ids.len())
            .ok_or_else(|| {
                NativeSessionStateSnafu {
                    rule: "native model batch requires equal nonempty session and token inputs",
                }
                .build()
            })?;
        let first = first.owner.ready_resource().map_err(begin_error)?;
        let resident = Arc::clone(first.resident());
        let mut lengths = reserve("native model batch lengths", sessions.len())?;
        let mut offsets = reserve("native model batch offsets", sessions.len())?;
        let mut demands = reserve("native model batch demand receipts", sessions.len())?;
        let mut max_context = 0_usize;
        for (session, tokens) in sessions.iter().zip(token_ids) {
            let resource = session.owner.ready_resource().map_err(begin_error)?;
            if !Arc::ptr_eq(&resident, resource.resident()) {
                return NativeSessionStateSnafu {
                    rule: "native model batch sessions must share one resident allocation identity",
                }
                .fail();
            }
            resource.validate_prefill_tokens(tokens)?;
            lengths.push(tokens.len());
            offsets.push(resource.position());
            demands.push(Qwen35NativeExecutionDeviceDemand::from_bytes(
                resource.device_bytes(),
            )?);
            max_context = max_context.max(resource.max_context());
        }
        let packed =
            PackedPrefillPlan::new(&lengths, &offsets, max_context).context(NativeKernelSnafu)?;
        let demand = Qwen35NativeExecutionBatchDeviceDemand::try_from_demands(demands, &packed)?;
        let mut chunks = reserve("native model batch chunks", packed.sequence_count())?;
        for (sequence, (session, tokens)) in sessions.iter().zip(token_ids).enumerate() {
            let resource = session.owner.ready_resource().map_err(begin_error)?;
            chunks.push(ModelChunkPlan::from_packed_sequence(
                resource.plan_for_batch(),
                tokens,
                &packed,
                sequence,
            )?);
        }
        Ok(Self {
            sessions,
            chunks,
            packed,
            demand,
            token_ids,
        })
    }
}

impl Qwen35NativeExecutionBatchPlan<'_, '_> {
    /// Return the checked requested aggregate device-byte receipt.
    #[must_use]
    pub const fn device_demand(&self) -> Qwen35NativeExecutionBatchDeviceDemand {
        self.demand
    }

    /// Return the sole aggregate sequence-major geometry for this transaction.
    #[must_use]
    pub const fn packed_prefill_plan(&self) -> &PackedPrefillPlan {
        &self.packed
    }

    /// Execute every planned native chunk and publish every owner together.
    ///
    /// # Errors
    ///
    /// Before any publication, an allocation, submit, synchronization, status,
    /// cache-preparation, or output-preparation failure retains submitted owners
    /// as poisoned custody and leaves unsubmitted owners ready. Once every cache
    /// guard and output exists, publication is infallible and allocation-free.
    ///
    /// # Safety
    ///
    /// The caller supplies the qualified native device/numerical contract for
    /// every borrowed session. This executes B=1 arithmetic independently per
    /// sequence and does not establish vectorized, serving, or hardware parity.
    pub unsafe fn execute(self) -> Result<Vec<DeviceBuffer<f32>>> {
        let Self {
            sessions,
            chunks,
            packed: _,
            demand: _,
            token_ids,
        } = self;
        let mut pending = reserve("native model batch pending owners", sessions.len())?;
        let mut completed = reserve("native model batch completion owners", sessions.len())?;
        let mut outputs = reserve("native model batch grouped outputs", sessions.len())?;
        let mut prepared: Vec<Option<cache::NativePagedPreparedCompletion<'_>>> =
            reserve("native model batch prepared cache guards", sessions.len())?;
        for ((session, chunk), _tokens) in sessions.iter_mut().zip(chunks).zip(token_ids) {
            let mut in_flight = session.owner.begin().map_err(begin_error)?;
            if let Err(error) = in_flight
                .resource()
                .map_err(completion_error)?
                .prepare_chunk(chunk)
            {
                let requires_teardown = in_flight
                    .resource()
                    .map_err(completion_error)?
                    .requires_teardown();
                if requires_teardown {
                    in_flight.poison_known_idle();
                }
                return Err(error);
            }
            in_flight.mark_submitted();
            // SAFETY: the batch caller establishes every qualified session contract.
            unsafe {
                in_flight
                    .resource()
                    .map_err(completion_error)?
                    .submit_step()?
            };
            pending.push(in_flight);
        }
        for in_flight in pending {
            completed.push(
                in_flight
                    .complete_prepublication()
                    .map_err(completion_error)?,
            );
        }
        for completion in &mut completed {
            let output = match completion.resource().take_completed_logits() {
                Ok(output) => output,
                Err(error) => {
                    restore_outputs(&mut completed, outputs);
                    return Err(error);
                }
            };
            outputs.push(output);
        }
        for completion in &mut completed {
            let prepared_cache = match unsafe { completion.resource().prepare_cache_completion() } {
                Ok(prepared_cache) => prepared_cache,
                Err(error) => {
                    drop(prepared);
                    restore_outputs(&mut completed, outputs);
                    return Err(error);
                }
            };
            prepared.push(prepared_cache);
        }
        for prepared_cache in prepared {
            if let Some(prepared_cache) = prepared_cache {
                prepared_cache.commit();
            }
        }
        for completion in completed {
            completion.publish(ModelSessionResources::publish_after_cache);
        }
        Ok(outputs)
    }
}

fn restore_outputs(
    completed: &mut [PostCompletion<'_, ModelSessionResources>],
    outputs: Vec<DeviceBuffer<f32>>,
) {
    for (completion, output) in completed.iter_mut().zip(outputs) {
        completion.resource().restore_completed_logits(output);
    }
}

fn reserve<T>(target: &'static str, length: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .context(ExecutionAllocationSnafu { target, length })?;
    Ok(values)
}
