//! Blocking unsafe qualification session for one native Qwen3.5 main-model token.

use hipcore::{Device, DeviceBuffer};
use std::sync::Arc;

use super::ResourceOwner;
use super::model_plan::{DeviceModelPlan, ModelDeviceByteDemand};
use super::model_resources::{
    ModelSessionResources, ModelSessionTeardown, ModelSessionTeardownState,
    NativeResidentModelResources, NativeResidentTeardown,
};
use super::session::{Qwen35NativeSessionState, begin_error, completion_error, session_state};
use crate::{Qwen35Weights, Result};

/// Checked device-allocation demand for one resident native model and one session.
///
/// Categories describe requested allocation extents under the checked plan.
/// They are neither observed residency nor a capacity, performance, device-
/// qualification, or serving result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35NativeExecutionDeviceDemand {
    bytes: ModelDeviceByteDemand,
    total: usize,
}

impl Qwen35NativeExecutionDeviceDemand {
    fn from_bytes(bytes: ModelDeviceByteDemand) -> Result<Self> {
        Ok(Self {
            bytes,
            total: bytes.total()?,
        })
    }

    /// Requested immutable embedding, layer, final-normalization, and output weight bytes.
    #[must_use]
    pub const fn weight_bytes(self) -> usize {
        self.bytes.weights
    }

    /// Requested reusable full-attention workspace bytes.
    #[must_use]
    pub const fn full_workspace_bytes(self) -> usize {
        self.bytes.full_workspace
    }

    /// Requested reusable recurrent-block workspace bytes.
    #[must_use]
    pub const fn recurrent_workspace_bytes(self) -> usize {
        self.bytes.recurrent_workspace
    }

    /// Requested reusable shared layer-finish workspace bytes.
    #[must_use]
    pub const fn finish_workspace_bytes(self) -> usize {
        self.bytes.finish_workspace
    }

    /// Requested hidden ping-pong-row bytes.
    #[must_use]
    pub const fn hidden_row_bytes(self) -> usize {
        self.bytes.hidden_rows
    }

    /// Requested terminal normalized-hidden bytes.
    #[must_use]
    pub const fn final_normalized_bytes(self) -> usize {
        self.bytes.final_normalized
    }

    /// Requested output-logit bytes.
    #[must_use]
    pub const fn logits_bytes(self) -> usize {
        self.bytes.logits
    }

    /// Requested shared text-mRoPE control-buffer bytes.
    #[must_use]
    pub const fn mrope_control_bytes(self) -> usize {
        self.bytes.mrope_controls
    }

    /// Requested native paged K/V backing bytes.
    #[must_use]
    pub const fn key_value_bytes(self) -> usize {
        self.bytes.key_values
    }

    /// Requested native paged-K/V table bytes.
    #[must_use]
    pub const fn page_table_bytes(self) -> usize {
        self.bytes.page_table
    }

    /// Requested active recurrent convolution-history bytes.
    #[must_use]
    pub const fn recurrent_history_active_bytes(self) -> usize {
        self.bytes.recurrent_history_active
    }

    /// Requested staged recurrent convolution-history bytes.
    #[must_use]
    pub const fn recurrent_history_staged_bytes(self) -> usize {
        self.bytes.recurrent_history_staged
    }

    /// Requested active recurrent-state bytes.
    #[must_use]
    pub const fn recurrent_state_active_bytes(self) -> usize {
        self.bytes.recurrent_state_active
    }

    /// Requested staged recurrent-state bytes.
    #[must_use]
    pub const fn recurrent_state_staged_bytes(self) -> usize {
        self.bytes.recurrent_state_staged
    }

    /// Requested sticky native numerical-status bytes.
    ///
    /// This one session-owned word is never reset. Any checked-arithmetic
    /// status makes the owned session permanently unusable.
    #[must_use]
    pub const fn numerical_status_bytes(self) -> usize {
        self.bytes.numerical_status
    }

    /// Checked total requested device bytes for one resident model, one session,
    /// one token's controls, and one returned logits buffer.
    #[must_use]
    pub const fn total_bytes(self) -> usize {
        self.total
    }

    /// Requested immutable device bytes retained once by a resident model.
    #[must_use]
    pub const fn resident_bytes(self) -> usize {
        self.bytes.weights
    }

    /// Requested mutable device bytes retained by each native session.
    ///
    /// Construction checked the complete sum before this demand escaped, so
    /// this is the checked total less resident, one-token control, and returned
    /// output extents rather than a second mutable-category ledger.
    #[must_use]
    pub const fn session_bytes(self) -> usize {
        self.total - self.resident_bytes() - self.step_bytes() - self.output_bytes()
    }

    /// Requested one-token control-buffer bytes.
    #[must_use]
    pub const fn step_bytes(self) -> usize {
        self.bytes.mrope_controls
    }

    /// Requested returned-logit bytes for one completed token.
    #[must_use]
    pub const fn output_bytes(self) -> usize {
        self.bytes.logits
    }
}

/// Artifact-bound native plan for one Qwen3.5 main-model baseline.
///
/// This plan borrows the exact verified weights that it will upload. Its block
/// sequence contains the artifact's main blocks and terminal normalization and
/// output head; optional auxiliary `NextN` blocks are intentionally outside that
/// baseline and are not a reason to reject an otherwise supported artifact.
#[derive(Debug)]
pub struct Qwen35NativeExecutionPlan<'weights> {
    weights: &'weights Qwen35Weights,
    plan: DeviceModelPlan,
    demand: Qwen35NativeExecutionDeviceDemand,
}

impl<'weights> Qwen35NativeExecutionPlan<'weights> {
    /// Derive a checked native main-model plan from verified Qwen3.5 weights.
    ///
    /// This performs no device initialization, allocation, upload, or kernel
    /// submission.
    ///
    /// # Errors
    ///
    /// Refuses unsupported main-block metadata, matrix formats or geometry,
    /// invalid context or page geometry, and unrepresentable allocation extents.
    pub fn try_from_weights(
        weights: &'weights Qwen35Weights,
        max_context: usize,
        page_tokens: kernels::attention::NativePageTokens,
    ) -> Result<Self> {
        let plan = DeviceModelPlan::from_weights(weights, max_context, page_tokens)?;
        let demand = Qwen35NativeExecutionDeviceDemand::from_bytes(plan.bytes)?;
        Ok(Self {
            weights,
            plan,
            demand,
        })
    }

    /// Return this plan's exact checked allocation request.
    #[must_use]
    pub const fn device_demand(&self) -> Qwen35NativeExecutionDeviceDemand {
        self.demand
    }

    /// Upload this plan's immutable verified weights into one reusable native model.
    ///
    /// # Errors
    ///
    /// Returns typed upload or allocation failures without creating mutable
    /// request state or submitting inference work.
    ///
    /// # Safety
    ///
    /// `device` must be the qualified `gfx1100` device selected for the native
    /// kernels. This establishes immutable resident ownership only; it is not a
    /// physical residency, capacity, performance, or serving claim.
    pub unsafe fn into_model(self, device: &Device) -> Result<Qwen35NativeExecutionModel> {
        let resources = NativeResidentModelResources::new(self.weights, self.plan, device)?;
        Ok(Qwen35NativeExecutionModel {
            resources: Arc::new(resources),
        })
    }

    /// Upload this plan's exact verified weights and create its owned session.
    ///
    /// # Errors
    ///
    /// Returns typed stream, allocation, upload, or verified weight-binding
    /// failures. This performs no inference submission.
    ///
    /// # Safety
    ///
    /// `device` must be the qualified `gfx1100` device selected for the native
    /// kernels. Construction establishes ownership only: it is not a physical
    /// GPU qualification, capacity grant, performance claim, or serving API.
    pub unsafe fn into_session(self, device: &Device) -> Result<Qwen35NativeExecutionSession> {
        // SAFETY: the caller supplies the qualified device required to upload the immutable model.
        unsafe { self.into_model(device) }?.new_session()
    }
}

/// Shared immutable native model uploads bound to one verified execution plan.
///
/// This type declares no manual thread-safety contract. In particular, each
/// mutable [`Qwen35NativeExecutionSession`] retains its own `Stream`, whose
/// HIP wrapper is deliberately `!Sync`.
#[derive(Clone)]
pub struct Qwen35NativeExecutionModel {
    resources: Arc<NativeResidentModelResources>,
}

impl Qwen35NativeExecutionModel {
    /// Derive one exact-context session plan under this resident model's ceiling.
    ///
    /// The opaque plan retains this exact resident model, including its verified
    /// backing and device identity. It cannot select another artifact, device,
    /// page geometry, or a context above the resident ceiling. Holding it is
    /// ownership only, not an admission, physical-residency, or capacity grant;
    /// a higher layer must still authorize the requested use.
    ///
    /// # Errors
    ///
    /// Refuses invalid context, a request above the resident ceiling, or a
    /// descriptor that cannot bind the resident's immutable uploads.
    pub fn plan_session(&self, max_context: usize) -> Result<Qwen35NativeExecutionSessionPlan> {
        let plan = self.resources.plan_session(max_context)?;
        let demand = Qwen35NativeExecutionDeviceDemand::from_bytes(plan.bytes)?;
        Ok(Qwen35NativeExecutionSessionPlan {
            model: Arc::clone(&self.resources),
            plan,
            demand,
        })
    }

    /// Allocate one fresh mutable session at this model's profile ceiling.
    ///
    /// Each session owns a distinct stream, status word, cache, recurrent state,
    /// workspaces and pending token buffers. Its context bound is the exact bound
    /// admitted when this model was planned and cannot expand during dispatch.
    ///
    /// # Errors
    ///
    /// Returns typed allocation or stream-creation failures without affecting
    /// the immutable resident uploads or another session.
    pub fn new_session(&self) -> Result<Qwen35NativeExecutionSession> {
        self.plan_session(self.resources.context_ceiling())?
            .into_session()
    }

    /// Consume this model into explicit resident teardown when no use retains it.
    ///
    /// A live session or planned use keeps the immutable resident `Arc` alive,
    /// so this returns that exact model unchanged rather than treating a shared
    /// reference count as an eviction acknowledgement.
    #[must_use]
    pub fn close(self) -> Qwen35NativeExecutionModelClose {
        match Arc::try_unwrap(self.resources) {
            Ok(resources) => {
                Qwen35NativeExecutionModelClose::Teardown(Qwen35NativeExecutionModelTeardown {
                    inner: resources.into_teardown_parts().begin_release(),
                })
            }
            Err(resources) => Qwen35NativeExecutionModelClose::InUse(Self { resources }),
        }
    }
}

/// Result of requesting explicit immutable-model teardown.
#[must_use = "a shared model remains usable or teardown retains native ownership"]
pub enum Qwen35NativeExecutionModelClose {
    /// Another session or exact-context plan still retains the resident model.
    InUse(Qwen35NativeExecutionModel),
    /// The unique resident model has entered explicit HIP teardown.
    Teardown(Qwen35NativeExecutionModelTeardown),
}

/// Explicit teardown custody for a unique immutable native model.
///
/// Dropping this owner makes no HIP call and never acknowledges release.
#[must_use = "resident teardown outcomes retain explicit HIP custody"]
pub struct Qwen35NativeExecutionModelTeardown {
    inner: NativeResidentTeardown,
}

impl Qwen35NativeExecutionModelTeardown {
    /// Return the exact resident-teardown custody class.
    #[must_use]
    pub fn state(&self) -> Qwen35NativeExecutionSessionTeardownState {
        teardown_state(self.inner.state())
    }

    /// Retry HIP aggregate teardown only after its preflight-pending outcome.
    #[must_use]
    pub fn retry(self) -> Self {
        Self {
            inner: self.inner.retry_pending(),
        }
    }

    /// Deliberately retry only stream synchronization after completion was unproved.
    #[must_use]
    pub fn reconcile(self) -> Self {
        Self {
            inner: self.inner.reconcile_synchronization(),
        }
    }
}

/// Exact per-use session plan bound to one immutable resident native model.
pub struct Qwen35NativeExecutionSessionPlan {
    model: Arc<NativeResidentModelResources>,
    plan: DeviceModelPlan,
    demand: Qwen35NativeExecutionDeviceDemand,
}

impl Qwen35NativeExecutionSessionPlan {
    /// Return the exact per-use context bound admitted by this plan.
    #[must_use]
    pub const fn max_context(&self) -> usize {
        self.plan.layout.max_context()
    }

    /// Return this use's exact requested device extents.
    #[must_use]
    pub const fn device_demand(&self) -> Qwen35NativeExecutionDeviceDemand {
        self.demand
    }

    /// Allocate the fresh mutable session bound by this exact plan.
    ///
    /// # Errors
    ///
    /// Returns typed stream or allocation failures without changing the
    /// resident uploads or another planned use.
    pub fn into_session(self) -> Result<Qwen35NativeExecutionSession> {
        let resources = ModelSessionResources::new(self.model, self.plan)?;
        Ok(Qwen35NativeExecutionSession {
            owner: ResourceOwner::new(resources),
        })
    }
}

/// One owned blocking native main-model qualification session.
///
/// The session retains shared immutable verified uploads and exclusively owns
/// its stream, full/recurrent workspaces, K/V pool and table, recurrent active
/// and staged state, status word, and pending token buffers. Submitted
/// resources remain owned until completion is synchronized and publication
/// succeeds; uncertain completion retains the shared model owner with the
/// complete mutable bundle rather than releasing buffers early.
pub struct Qwen35NativeExecutionSession {
    owner: ResourceOwner<ModelSessionResources>,
}

/// Explicit teardown custody for a native main-model session.
///
/// This is a thin owner around the HIP aggregate teardown result, not a second
/// release protocol. It retains the shared immutable resident model alongside
/// HIP pending, synchronization-unconfirmed, and quarantined custody so a
/// stream cannot outlive immutable uploads it may still reference. Dropping it
/// makes no HIP call and never acknowledges release.
#[must_use = "native teardown outcomes retain explicit HIP custody"]
pub struct Qwen35NativeExecutionSessionTeardown {
    inner: ModelSessionTeardown,
}

/// Observable state of explicit native session teardown custody.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Qwen35NativeExecutionSessionTeardownState {
    /// HIP acknowledged every captured session buffer and its ordered stream.
    Released,
    /// The session stream was not eligible for aggregate HIP teardown.
    Unadmitted,
    /// Checked aggregate accounting retained admitted and unadmitted ownership.
    PartiallyAdmitted,
    /// HIP preflight retained the aggregate before a destructor call.
    Pending,
    /// Stream completion is unproved; only explicit reconciliation is sound.
    SynchronizationUnconfirmed,
    /// A destructor outcome is indeterminate and remains conservatively held.
    Quarantined,
}

impl Qwen35NativeExecutionSessionTeardown {
    /// Return the exact custody class; this is never evidence of physical eviction.
    #[must_use]
    pub fn state(&self) -> Qwen35NativeExecutionSessionTeardownState {
        teardown_state(self.inner.state())
    }

    /// Retry HIP aggregate teardown only after its preflight-pending outcome.
    ///
    /// Other outcomes retain their exact custody unchanged. In particular, a
    /// synchronization-unconfirmed result requires [`Self::reconcile`], never
    /// an ordinary destructor retry.
    #[must_use]
    pub fn retry(self) -> Self {
        Self {
            inner: self.inner.retry_pending(),
        }
    }

    /// Deliberately retry only stream synchronization after completion was unproved.
    ///
    /// This leaves every other teardown outcome unchanged and never retries a
    /// destructor after a quarantined result.
    #[must_use]
    pub fn reconcile(self) -> Self {
        Self {
            inner: self.inner.reconcile_synchronization(),
        }
    }
}

fn teardown_state(state: ModelSessionTeardownState) -> Qwen35NativeExecutionSessionTeardownState {
    match state {
        ModelSessionTeardownState::Released => Qwen35NativeExecutionSessionTeardownState::Released,
        ModelSessionTeardownState::Unadmitted => {
            Qwen35NativeExecutionSessionTeardownState::Unadmitted
        }
        ModelSessionTeardownState::PartiallyAdmitted => {
            Qwen35NativeExecutionSessionTeardownState::PartiallyAdmitted
        }
        ModelSessionTeardownState::Pending => Qwen35NativeExecutionSessionTeardownState::Pending,
        ModelSessionTeardownState::SynchronizationUnconfirmed => {
            Qwen35NativeExecutionSessionTeardownState::SynchronizationUnconfirmed
        }
        ModelSessionTeardownState::Quarantined => {
            Qwen35NativeExecutionSessionTeardownState::Quarantined
        }
    }
}

impl Qwen35NativeExecutionSession {
    /// Return the externally observable completion state.
    #[must_use]
    pub fn state(&self) -> Qwen35NativeSessionState {
        session_state(&self.owner)
    }

    /// Consume this session into explicit aggregate native teardown.
    ///
    /// Every mutable allocation is disarmed before the owned stream is
    /// admitted to HIP teardown. The result retains the resident immutable
    /// model through all non-released outcomes; it does not turn an `Arc` drop
    /// into an eviction acknowledgement.
    ///
    /// # Errors
    ///
    /// Returns only if an internal guard had already removed this session's
    /// owner. Safe callers cannot retain such a guard across this consuming
    /// operation.
    pub fn close(self) -> Result<Qwen35NativeExecutionSessionTeardown> {
        let resources = self.owner.into_resource().ok_or_else(|| {
            crate::error::NativeSessionStateSnafu {
                rule: "native session close requires its complete owned resource bundle",
            }
            .build()
        })?;
        Ok(Qwen35NativeExecutionSessionTeardown {
            inner: resources.into_teardown_parts().begin_release(),
        })
    }

    /// Execute one token through all native main blocks and return its logits.
    ///
    /// The blocking step covers embedding, every admitted full or recurrent
    /// main block, final normalization, and the distinct output head. It does
    /// not execute optional `NextN` auxiliary blocks.
    ///
    /// # Errors
    ///
    /// Preflight refusal leaves a ready session unchanged. Any failure after
    /// the first submission permanently poisons the session; its state reports
    /// whether completion is known idle or remains uncertain. Failed work never
    /// publishes staged K/V or recurrent state and returns no partial logits.
    ///
    /// # Safety
    ///
    /// The caller guarantees a qualified `gfx1100` device. Checked launches
    /// classify explicit operands and arithmetic results in one owned sticky
    /// status word before KV, recurrent state, position, or logits publish.
    /// This depends on the qualified compiler, math implementation, and device
    /// preserving the checked denorm contract. It does not establish hardware
    /// parity, performance, physical-GPU qualification, capacity, artifact
    /// quality, or serving safety.
    pub unsafe fn step(&mut self, token: u32) -> Result<DeviceBuffer<f32>> {
        let mut in_flight = self.owner.begin().map_err(begin_error)?;
        let (preparation, requires_teardown) = {
            let resources = in_flight.resource().map_err(completion_error)?;
            let preparation = resources.prepare_step(token);
            (preparation, resources.requires_teardown())
        };
        if let Err(error) = preparation {
            if requires_teardown {
                in_flight.poison_known_idle();
            }
            return Err(error);
        }
        in_flight.mark_submitted();
        {
            let resources = in_flight.resource().map_err(completion_error)?;
            // SAFETY: the caller establishes device qualification; checked
            // launchers retain and classify their explicit numerical inputs.
            unsafe { resources.submit_step()? };
        }
        in_flight
            .complete(publish_completed_step)
            .map_err(completion_error)
    }
}

fn publish_completed_step(resources: &mut ModelSessionResources) -> Result<DeviceBuffer<f32>> {
    resources.publish_completed()
}

#[cfg(test)]
mod tests {
    use super::Qwen35NativeExecutionDeviceDemand;
    use crate::qwen35_native::model_plan::ModelDeviceByteDemand;

    #[test]
    fn demand_partitions_one_resident_model_from_one_session_and_token() -> Result<(), String> {
        let bytes = ModelDeviceByteDemand {
            weights: 11,
            full_workspace: 13,
            recurrent_workspace: 17,
            finish_workspace: 19,
            hidden_rows: 23,
            final_normalized: 29,
            logits: 31,
            mrope_controls: 37,
            key_values: 41,
            page_table: 43,
            recurrent_history_active: 47,
            recurrent_history_staged: 53,
            recurrent_state_active: 59,
            recurrent_state_staged: 61,
            numerical_status: 67,
        };
        let demand = Qwen35NativeExecutionDeviceDemand::from_bytes(bytes)
            .map_err(|error| error.to_string())?;

        assert_eq!(demand.resident_bytes(), 11);
        assert_eq!(
            demand.session_bytes(),
            472,
            "per-session demand must include only mutable state and workspaces"
        );
        assert_eq!(demand.step_bytes(), 37);
        assert_eq!(demand.output_bytes(), 31);
        assert_eq!(
            demand.total_bytes(),
            demand.resident_bytes()
                + demand.session_bytes()
                + demand.step_bytes()
                + demand.output_bytes(),
            "all demand categories must derive from the one checked resource plan"
        );
        Ok(())
    }
}
