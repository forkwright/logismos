//! Blocking unsafe qualification session for one native Qwen3.5 attention block.

use hipcore::{Device, DeviceBuffer};
use snafu::ResultExt;

use crate::error::{ArithmeticOverflowSnafu, NativePagedKvSnafu, NativeSessionStateSnafu};
use crate::qwen35_native::plan::{DeviceByteDemand, DeviceFullAttentionPlan};
use crate::qwen35_native::resources::DeviceResources;
use crate::qwen35_native::{
    BeginError, CompletionError, CompletionResource, ResourceOwner, ResourceState,
};
use crate::{Qwen35Weights, Result};

/// Checked device-allocation demand for one full-attention-block qualification session.
///
/// The categories are requested allocation extents, not observed residency,
/// throughput, capacity, or a device qualification result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35NativeLayerDeviceDemand {
    bytes: DeviceByteDemand,
    total: usize,
}

impl Qwen35NativeLayerDeviceDemand {
    fn from_bytes(bytes: DeviceByteDemand) -> Result<Self> {
        Ok(Self {
            bytes,
            total: bytes.total()?,
        })
    }

    /// Requested immutable weight bytes.
    #[must_use]
    pub const fn weight_bytes(self) -> usize {
        self.bytes.weights
    }

    /// Requested reusable scratch bytes.
    #[must_use]
    pub const fn scratch_bytes(self) -> usize {
        self.bytes.scratch
    }

    /// Requested one-token input bytes.
    #[must_use]
    pub const fn input_bytes(self) -> usize {
        self.bytes.input
    }

    /// Requested one-token output bytes.
    #[must_use]
    pub const fn output_bytes(self) -> usize {
        self.bytes.output
    }

    /// Requested mRoPE-control bytes.
    #[must_use]
    pub const fn control_bytes(self) -> usize {
        self.bytes.controls
    }

    /// Requested separate native K/V backing bytes.
    #[must_use]
    pub const fn key_value_bytes(self) -> usize {
        self.bytes.key_values
    }

    /// Requested native page-table bytes.
    #[must_use]
    pub const fn table_bytes(self) -> usize {
        self.bytes.table
    }

    /// Checked total requested device bytes across this one owned session.
    #[must_use]
    pub const fn total_bytes(self) -> usize {
        self.total
    }
}

/// Artifact-bound plan for one native Qwen3.5 full-attention block.
///
/// The plan borrows the exact verified weights used during upload, preventing a
/// same-shape but different artifact from being substituted at session creation.
#[derive(Debug)]
pub struct Qwen35NativeLayerPlan<'weights, 'artifact> {
    weights: &'weights Qwen35Weights<'artifact>,
    plan: DeviceFullAttentionPlan,
    demand: Qwen35NativeLayerDeviceDemand,
}

impl<'weights, 'artifact> Qwen35NativeLayerPlan<'weights, 'artifact> {
    /// Derive a checked native one-block plan from verified Qwen3.5 weights.
    ///
    /// This performs no device initialization, allocation, upload, or kernel
    /// submission.
    ///
    /// # Errors
    ///
    /// Refuses unsupported block roles, metadata, matrix formats or geometry,
    /// out-of-range context, and unrepresentable allocation extents.
    pub fn try_from_weights(
        weights: &'weights Qwen35Weights<'artifact>,
        block: usize,
        max_context: usize,
        page_tokens: kernels::attention::NativePageTokens,
    ) -> Result<Self> {
        let plan = DeviceFullAttentionPlan::from_weights(weights, block, max_context, page_tokens)?;
        let demand = Qwen35NativeLayerDeviceDemand::from_bytes(plan.bytes)?;
        Ok(Self {
            weights,
            plan,
            demand,
        })
    }

    /// Return the exact checked allocation request for this plan.
    #[must_use]
    pub const fn device_demand(&self) -> Qwen35NativeLayerDeviceDemand {
        self.demand
    }

    /// Upload this exact plan's verified weights and create its owned session.
    ///
    /// # Errors
    ///
    /// Returns the typed allocation, upload, stream, or weight-binding failure.
    /// No inference kernel is submitted during construction.
    ///
    /// # Safety
    ///
    /// `device` must be a qualified `gfx1100` device for the selected native
    /// kernels. This constructor establishes ownership, not numerical or
    /// hardware qualification.
    pub unsafe fn into_session(self, device: &Device) -> Result<Qwen35NativeLayerSession> {
        let resources = DeviceResources::new(self.weights, self.plan, device)?;
        Ok(Qwen35NativeLayerSession {
            owner: ResourceOwner::new(resources),
        })
    }
}

/// Observable blocking-session state after completed or failed submissions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Qwen35NativeSessionState {
    /// No work is submitted and the owned bundle may accept its next token.
    Ready,
    /// A submission reached a known-idle state but the session is permanently poisoned.
    PoisonedKnownIdle,
    /// Completion remains uncertain; the complete owned bundle is retained or forgotten.
    PoisonedCompletionUncertain,
}

/// Backwards-compatible state name for the single-layer qualification session.
pub type Qwen35NativeLayerSessionState = Qwen35NativeSessionState;

pub(super) fn session_state<Resource: CompletionResource>(
    owner: &ResourceOwner<Resource>,
) -> Qwen35NativeSessionState {
    match owner.state() {
        Some(ResourceState::Ready(_)) => Qwen35NativeSessionState::Ready,
        Some(ResourceState::PoisonedIdle(_)) => Qwen35NativeSessionState::PoisonedKnownIdle,
        Some(ResourceState::InFlight(_) | ResourceState::PoisonedUncertain(_)) | None => {
            Qwen35NativeSessionState::PoisonedCompletionUncertain
        }
    }
}

/// One owned blocking native full-attention-block qualification session.
///
/// It owns the uploaded verified weights, K/V ledger and device mirrors,
/// stream, scratch, and every per-step input/output buffer until completion is
/// proved. It is not a model runtime, serving session, or hardware result.
pub struct Qwen35NativeLayerSession {
    owner: ResourceOwner<DeviceResources>,
}

impl Qwen35NativeLayerSession {
    /// Return the session's externally observable completion state.
    #[must_use]
    pub fn state(&self) -> Qwen35NativeLayerSessionState {
        session_state(&self.owner)
    }

    /// Execute one token through exactly one admitted native full-attention block.
    ///
    /// # Errors
    ///
    /// Refuses a poisoned session, wrong-device or wrong-sized input, exhausted
    /// context, or failed allocation. Preflight refusal leaves the session
    /// ready. Any error after submission permanently poisons it; use
    /// [`Self::state`] to distinguish known-idle from uncertain completion.
    /// No failed step publishes KV, advances position, or returns partial output.
    ///
    /// # Safety
    ///
    /// `input` must be a live one-row allocation on this session's qualified
    /// `gfx1100` device. The caller guarantees finite normal-or-zero values for
    /// every input, weight, K/V, control, and intermediate value required by
    /// the native numerical domain, and no pending external operation may use
    /// `input`. This blocking boundary does not independently establish those
    /// conditions or hardware parity.
    pub unsafe fn step(&mut self, input: DeviceBuffer<f32>) -> Result<DeviceBuffer<f32>> {
        let mut in_flight = self.owner.begin().map_err(begin_error)?;
        {
            let resources = in_flight.resource().map_err(completion_error)?;
            resources.prepare_step(input)?;
        }
        in_flight.mark_submitted();
        {
            let resources = in_flight.resource().map_err(completion_error)?;
            // SAFETY: the caller supplied the raw numerical-domain and device-lifetime guarantees for this synchronous step.
            unsafe { resources.submit_step()? };
        }
        in_flight
            .complete(publish_completed_step)
            .map_err(completion_error)
    }
}

fn publish_completed_step(resources: &mut DeviceResources) -> Result<DeviceBuffer<f32>> {
    let next_position = resources.position.checked_add(1).ok_or_else(|| {
        ArithmeticOverflowSnafu {
            context: "native committed position",
        }
        .build()
    })?;
    let step = resources.step.take().ok_or_else(|| {
        NativeSessionStateSnafu {
            rule: "native publication requires one prepared step",
        }
        .build()
    })?;
    // SAFETY: ResourceOwner synchronized the ordered stream before invoking this callback; all fallible output and position work above precedes logical publication.
    unsafe {
        resources
            .kv
            .commit_prepared_after_completion()
            .context(NativePagedKvSnafu)?;
    }
    resources.position = next_position;
    Ok(step.output)
}

pub(super) fn begin_error(error: BeginError) -> crate::Error {
    let rule = match error {
        BeginError::MissingResource => "native session lost its owned resource bundle",
        BeginError::NotReady => "native session is permanently poisoned after submission",
    };
    NativeSessionStateSnafu { rule }.build()
}

pub(super) fn completion_error(error: CompletionError<crate::Error>) -> crate::Error {
    match error {
        CompletionError::Commit { source, .. }
        | CompletionError::Synchronization { source, .. } => source,
        CompletionError::MissingResource => NativeSessionStateSnafu {
            rule: "native in-flight guard lost its owned resource bundle",
        }
        .build(),
        CompletionError::NotSubmitted => NativeSessionStateSnafu {
            rule: "native completion requires a marked submission",
        }
        .build(),
    }
}
