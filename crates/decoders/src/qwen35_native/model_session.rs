//! Blocking unsafe qualification session for one native Qwen3.5 main-model token.

use hipcore::{Device, DeviceBuffer};
use std::sync::Arc;

use super::ResourceOwner;
pub use super::custody::NativeBuildSource;
use super::custody::{NativeBufferParts, NativeBuildScope};
use super::model_plan::{DeviceModelPlan, ModelDeviceByteDemand};
use super::model_resources::{
    ModelSessionResources, ModelSessionTeardown, ModelSessionTeardownParts,
    ModelSessionTeardownState, NativeResidentModelResources, NativeResidentTeardown,
    NativeResidentTeardownParts, ResidentRetention, StreamRetention,
};
use super::session::{Qwen35NativeSessionState, begin_error, completion_error, session_state};
use crate::{Qwen35Weights, Result};

enum NativeBuildCustody {
    Resident {
        scope: NativeBuildScope,
        device: Device,
    },
    Session {
        scope: NativeBuildScope,
        stream: Option<StreamRetention>,
        resident: ResidentRetention<NativeResidentModelResources>,
    },
    Standalone {
        scope: NativeBuildScope,
        stream: Option<StreamRetention>,
        device: Device,
    },
}

/// A failed native construction retaining its exact source and partial owners.
///
/// Dropping this value performs no HIP operation. Completed allocations remain
/// inert in one recursive scope; an indeterminate allocation or stream output
/// remains in its typed source quarantine and never enters ordinary teardown.
#[must_use = "failed native construction retains explicit resource custody"]
pub struct NativeBuildFailure {
    source: Box<NativeBuildSource>,
    custody: Box<NativeBuildCustody>,
}

impl NativeBuildFailure {
    pub(super) fn resident(
        source: NativeBuildSource,
        scope: NativeBuildScope,
        device: Device,
    ) -> Self {
        Self {
            source: Box::new(source),
            custody: Box::new(NativeBuildCustody::Resident { scope, device }),
        }
    }

    pub(super) fn session(
        source: NativeBuildSource,
        scope: NativeBuildScope,
        stream: Option<StreamRetention>,
        resident: ResidentRetention<NativeResidentModelResources>,
    ) -> Self {
        Self {
            source: Box::new(source),
            custody: Box::new(NativeBuildCustody::Session {
                scope,
                stream,
                resident,
            }),
        }
    }

    pub(super) fn standalone(
        source: NativeBuildSource,
        scope: NativeBuildScope,
        stream: Option<StreamRetention>,
        device: Device,
    ) -> Self {
        Self {
            source: Box::new(source),
            custody: Box::new(NativeBuildCustody::Standalone {
                scope,
                stream,
                device,
            }),
        }
    }

    /// Borrow the exact failure that stopped native construction.
    pub fn source_error(&self) -> &NativeBuildSource {
        self.source.as_ref()
    }

    /// Whether the source retains a non-null output from a failed HIP creation.
    ///
    /// This is independent of release of known completed owners: terminal
    /// creation custody has no destructor, retry, or ordinary inventory path.
    #[must_use]
    pub fn has_creation_quarantine(&self) -> bool {
        source_has_creation_quarantine(self.source.as_ref())
    }

    /// Begin explicit teardown of every known completed construction owner.
    ///
    /// A live private construction guard prevents detachment and is returned as
    /// [`NativeBuildReleaseState::ConstructionPending`]. Safe public callers
    /// only receive a failure after those guards have unwound.
    #[must_use = "construction release retains the source and all native custody"]
    pub fn begin_release(self) -> NativeBuildRelease {
        NativeBuildRelease::from_failure(self)
    }

    fn custody_kind(&self) -> &'static str {
        match self.custody.as_ref() {
            NativeBuildCustody::Resident { .. } => "resident",
            NativeBuildCustody::Session { .. } => "session",
            NativeBuildCustody::Standalone { .. } => "standalone",
        }
    }
}

impl core::fmt::Debug for NativeBuildFailure {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("NativeBuildFailure")
            .field("source", &self.source)
            .field("custody", &self.custody_kind())
            .finish_non_exhaustive()
    }
}

impl core::fmt::Display for NativeBuildFailure {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.source.fmt(formatter)
    }
}

impl std::error::Error for NativeBuildFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source.as_ref())
    }
}

enum NativeBuildReleaseCustody {
    Construction(NativeBuildCustody),
    NoResources {
        resident: Option<ResidentRetention<NativeResidentModelResources>>,
    },
    Unadmitted {
        _buffers: NativeBufferParts,
        _resident: Option<ResidentRetention<NativeResidentModelResources>>,
    },
    Resident(NativeResidentTeardown),
    Session(ModelSessionTeardown),
}

/// Explicit release custody for a failed native construction.
///
/// Its overall state prioritizes any terminal creation quarantine. The
/// subordinate known-owner state remains available separately, and dropping
/// this value performs no HIP operation.
#[must_use = "construction release outcomes retain source and native custody"]
pub struct NativeBuildRelease {
    source: Box<NativeBuildSource>,
    custody: NativeBuildReleaseCustody,
}

/// Overall release state retained after native construction failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NativeBuildReleaseState {
    /// A failed HIP creation retains an indeterminate non-null output.
    CreationQuarantined,
    /// A private typed guard still prevents lossless custody materialization.
    ConstructionPending,
    /// No successfully created owner requires a HIP destructor.
    NoResources,
    /// Completed buffers exist but no owned stream can admit their teardown.
    Unadmitted,
    /// Checked aggregate accounting retained admitted and unadmitted ownership.
    PartiallyAdmitted,
    /// HIP preflight retained the complete aggregate before a destructor call.
    Pending,
    /// Stream completion remains unproved; only reconciliation is sound.
    SynchronizationUnconfirmed,
    /// A destructor outcome is indeterminate and cannot be retried.
    Quarantined,
    /// HIP acknowledged all destructors for the known completed owners.
    Released,
}

impl NativeBuildRelease {
    fn from_failure(failure: NativeBuildFailure) -> Self {
        let NativeBuildFailure { source, custody } = failure;
        let custody = match *custody {
            NativeBuildCustody::Resident { scope, device } => {
                materialize_resident_build(scope, device)
            }
            NativeBuildCustody::Session {
                scope,
                stream,
                resident,
            } => materialize_session_build(scope, stream, resident),
            NativeBuildCustody::Standalone {
                scope,
                stream,
                device,
            } => materialize_standalone_build(scope, stream, device),
        };
        Self { source, custody }
    }

    /// Borrow the exact source that stopped construction.
    pub fn source_error(&self) -> &NativeBuildSource {
        self.source.as_ref()
    }

    /// Whether a failed creation separately retains an indeterminate output.
    #[must_use]
    pub fn has_creation_quarantine(&self) -> bool {
        source_has_creation_quarantine(self.source.as_ref())
    }

    /// Classify teardown of known successfully created owners.
    #[must_use]
    pub fn state(&self) -> NativeBuildReleaseState {
        if self.has_creation_quarantine() {
            return NativeBuildReleaseState::CreationQuarantined;
        }
        self.known_resource_state()
    }

    /// Classify only the independently retained known completed owners.
    ///
    /// This subordinate state is never a whole-construction release result
    /// while [`Self::state`] reports creation quarantine.
    #[must_use]
    pub fn known_resource_state(&self) -> NativeBuildReleaseState {
        match &self.custody {
            NativeBuildReleaseCustody::Construction(_) => {
                NativeBuildReleaseState::ConstructionPending
            }
            NativeBuildReleaseCustody::NoResources { .. } => NativeBuildReleaseState::NoResources,
            NativeBuildReleaseCustody::Unadmitted { .. } => NativeBuildReleaseState::Unadmitted,
            NativeBuildReleaseCustody::Resident(teardown) => build_release_state(teardown.state()),
            NativeBuildReleaseCustody::Session(teardown) => {
                build_release_state(teardown.known_state())
            }
        }
    }

    /// Retry only construction materialization or HIP preflight-pending release.
    pub fn retry(self) -> Self {
        let Self { source, custody } = self;
        match custody {
            NativeBuildReleaseCustody::Construction(custody) => {
                Self::from_failure(NativeBuildFailure {
                    source,
                    custody: Box::new(custody),
                })
            }
            NativeBuildReleaseCustody::Resident(teardown) => Self {
                source,
                custody: NativeBuildReleaseCustody::Resident(teardown.retry_pending()),
            },
            NativeBuildReleaseCustody::Session(teardown) => Self {
                source,
                custody: NativeBuildReleaseCustody::Session(teardown.retry_pending()),
            },
            custody => Self { source, custody },
        }
    }

    /// Reconcile only a synchronization-unconfirmed known-owner release.
    pub fn reconcile(self) -> Self {
        let Self { source, custody } = self;
        let custody = match custody {
            NativeBuildReleaseCustody::Resident(teardown) => {
                NativeBuildReleaseCustody::Resident(teardown.reconcile_synchronization())
            }
            NativeBuildReleaseCustody::Session(teardown) => {
                NativeBuildReleaseCustody::Session(teardown.reconcile_synchronization())
            }
            other => other,
        };
        Self { source, custody }
    }

    /// Recover the stopped source after all known owners are absent or released.
    ///
    /// # Errors
    ///
    /// Returns unchanged custody if release is incomplete or a session still
    /// retains its resident model anchor.
    pub fn into_released_source(self) -> core::result::Result<NativeBuildSource, Box<Self>> {
        if self.has_creation_quarantine() {
            return Err(Box::new(self));
        }
        let Self { source, custody } = self;
        match custody {
            NativeBuildReleaseCustody::NoResources { resident: None } => Ok(*source),
            NativeBuildReleaseCustody::Resident(teardown) => match teardown.into_released() {
                Ok(()) => Ok(*source),
                Err(teardown) => Err(Box::new(Self {
                    source,
                    custody: NativeBuildReleaseCustody::Resident(*teardown),
                })),
            },
            NativeBuildReleaseCustody::Session(teardown) => {
                release_source_from_session(source, teardown)
            }
            custody => Err(Box::new(Self { source, custody })),
        }
    }

    /// Recover the exact resident model after failed session resources released.
    ///
    /// The returned residual custody retains the exact source independently of
    /// the recovered model. In particular, a terminal creation quarantine can
    /// never disappear merely because all known session owners were released.
    /// Resident construction failures have no model to recover.
    ///
    /// # Errors
    ///
    /// Returns unchanged custody until every known session owner is released.
    pub fn into_released_model(
        self,
    ) -> core::result::Result<(Qwen35NativeExecutionModel, Self), Box<Self>> {
        let Self { source, custody } = self;
        match custody {
            NativeBuildReleaseCustody::NoResources {
                resident: Some(resident),
            } => {
                let model = Qwen35NativeExecutionModel {
                    resources: resident.recover(),
                };
                let residual = Self {
                    source,
                    custody: NativeBuildReleaseCustody::NoResources { resident: None },
                };
                Ok((model, residual))
            }
            NativeBuildReleaseCustody::Session(teardown) => {
                release_model_from_session(source, teardown)
            }
            custody => Err(Box::new(Self { source, custody })),
        }
    }
}

fn materialize_resident_build(
    scope: NativeBuildScope,
    device: Device,
) -> NativeBuildReleaseCustody {
    let parts = match scope.try_into_parts() {
        Ok(parts) => parts,
        Err(scope) => {
            return NativeBuildReleaseCustody::Construction(NativeBuildCustody::Resident {
                scope,
                device,
            });
        }
    };
    if parts.is_empty() {
        return NativeBuildReleaseCustody::NoResources { resident: None };
    }
    NativeBuildReleaseCustody::Resident(
        NativeResidentTeardownParts::new(device, parts.into_buffers()).begin_release(),
    )
}

fn materialize_session_build(
    scope: NativeBuildScope,
    stream: Option<StreamRetention>,
    resident: ResidentRetention<NativeResidentModelResources>,
) -> NativeBuildReleaseCustody {
    let parts = match scope.try_into_parts() {
        Ok(parts) => parts,
        Err(scope) => {
            return NativeBuildReleaseCustody::Construction(NativeBuildCustody::Session {
                scope,
                stream,
                resident,
            });
        }
    };
    let Some(stream) = stream else {
        return if parts.is_empty() {
            NativeBuildReleaseCustody::NoResources {
                resident: Some(resident),
            }
        } else {
            NativeBuildReleaseCustody::Unadmitted {
                _buffers: parts,
                _resident: Some(resident),
            }
        };
    };
    let parts = ModelSessionTeardownParts::new(
        stream.recover(),
        parts.into_buffers(),
        Some(resident),
        None,
    );
    NativeBuildReleaseCustody::Session(parts.begin_release())
}

fn materialize_standalone_build(
    scope: NativeBuildScope,
    stream: Option<StreamRetention>,
    device: Device,
) -> NativeBuildReleaseCustody {
    let parts = match scope.try_into_parts() {
        Ok(parts) => parts,
        Err(scope) => {
            return NativeBuildReleaseCustody::Construction(NativeBuildCustody::Standalone {
                scope,
                stream,
                device,
            });
        }
    };
    let Some(stream) = stream else {
        return if parts.is_empty() {
            NativeBuildReleaseCustody::NoResources { resident: None }
        } else {
            NativeBuildReleaseCustody::Unadmitted {
                _buffers: parts,
                _resident: None,
            }
        };
    };
    drop(device);
    let parts = ModelSessionTeardownParts::new(stream.recover(), parts.into_buffers(), None, None);
    NativeBuildReleaseCustody::Session(parts.begin_release())
}

fn release_source_from_session(
    source: Box<NativeBuildSource>,
    teardown: ModelSessionTeardown,
) -> core::result::Result<NativeBuildSource, Box<NativeBuildRelease>> {
    match teardown.into_released() {
        Ok(None) => Ok(*source),
        Ok(Some(resident)) => Err(Box::new(NativeBuildRelease {
            source,
            custody: NativeBuildReleaseCustody::NoResources {
                resident: Some(ResidentRetention::new(resident)),
            },
        })),
        Err(teardown) => Err(Box::new(NativeBuildRelease {
            source,
            custody: NativeBuildReleaseCustody::Session(*teardown),
        })),
    }
}

fn release_model_from_session(
    source: Box<NativeBuildSource>,
    teardown: ModelSessionTeardown,
) -> core::result::Result<(Qwen35NativeExecutionModel, NativeBuildRelease), Box<NativeBuildRelease>>
{
    match teardown.into_released() {
        Ok(Some(resources)) => {
            let model = Qwen35NativeExecutionModel { resources };
            let residual = NativeBuildRelease {
                source,
                custody: NativeBuildReleaseCustody::NoResources { resident: None },
            };
            Ok((model, residual))
        }
        Ok(None) => Err(Box::new(NativeBuildRelease {
            source,
            custody: NativeBuildReleaseCustody::NoResources { resident: None },
        })),
        Err(teardown) => Err(Box::new(NativeBuildRelease {
            source,
            custody: NativeBuildReleaseCustody::Session(*teardown),
        })),
    }
}

fn build_release_state(state: ModelSessionTeardownState) -> NativeBuildReleaseState {
    match state {
        ModelSessionTeardownState::Released => NativeBuildReleaseState::Released,
        ModelSessionTeardownState::Unadmitted => NativeBuildReleaseState::Unadmitted,
        ModelSessionTeardownState::PartiallyAdmitted => NativeBuildReleaseState::PartiallyAdmitted,
        ModelSessionTeardownState::Pending => NativeBuildReleaseState::Pending,
        ModelSessionTeardownState::SynchronizationUnconfirmed => {
            NativeBuildReleaseState::SynchronizationUnconfirmed
        }
        ModelSessionTeardownState::Quarantined => NativeBuildReleaseState::Quarantined,
    }
}

fn source_has_creation_quarantine(source: &NativeBuildSource) -> bool {
    match source {
        NativeBuildSource::Decoder(_) => false,
        NativeBuildSource::BufferAllocation(error) => {
            !matches!(error.as_ref(), hipcore::BufferAllocationError::NoHandle(_))
        }
        NativeBuildSource::StreamCreation(error) => {
            !matches!(error.as_ref(), hipcore::StreamCreationError::NoHandle(_))
        }
    }
}

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
    pub unsafe fn into_model(
        self,
        device: &Device,
    ) -> core::result::Result<Qwen35NativeExecutionModel, NativeBuildFailure> {
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
    pub unsafe fn into_session(
        self,
        device: &Device,
    ) -> core::result::Result<Qwen35NativeExecutionSession, NativeBuildFailure> {
        // SAFETY: the caller supplies the qualified device required to upload the immutable model.
        let model = unsafe { self.into_model(device) }?;
        model.new_session()
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
    pub fn new_session(
        &self,
    ) -> core::result::Result<Qwen35NativeExecutionSession, NativeBuildFailure> {
        let plan = self
            .plan_session(self.resources.context_ceiling())
            .map_err(|error| {
                NativeBuildFailure::session(
                    NativeBuildSource::decoder(error),
                    NativeBuildScope::new(),
                    None,
                    ResidentRetention::new(Arc::clone(&self.resources)),
                )
            })?;
        plan.into_session()
    }

    /// Consume this model into explicit resident teardown when no use retains it.
    ///
    /// A live session or planned use keeps the immutable resident `Arc` alive,
    /// so this returns that exact model unchanged rather than treating a shared
    /// reference count as an eviction acknowledgement.
    pub fn close(self) -> Qwen35NativeExecutionModelClose {
        match Arc::try_unwrap(self.resources) {
            Ok(resources) => Qwen35NativeExecutionModelClose::Teardown(Box::new(
                Qwen35NativeExecutionModelTeardown {
                    inner: resources.into_teardown_parts().begin_release(),
                },
            )),
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
    Teardown(Box<Qwen35NativeExecutionModelTeardown>),
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
    pub fn retry(self) -> Self {
        Self {
            inner: self.inner.retry_pending(),
        }
    }

    /// Deliberately retry only stream synchronization after completion was unproved.
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
    pub fn into_session(
        self,
    ) -> core::result::Result<Qwen35NativeExecutionSession, NativeBuildFailure> {
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

    /// Return the subordinate release state of the owned stream and buffers.
    ///
    /// This can report release progress while [`Self::state`] remains
    /// quarantined by a separate indeterminate output from failed allocation.
    /// It is never a complete session-release acknowledgement on its own.
    #[must_use]
    pub fn known_resource_state(&self) -> Qwen35NativeExecutionSessionTeardownState {
        teardown_state(self.inner.known_state())
    }

    /// Retry HIP aggregate teardown only after its preflight-pending outcome.
    ///
    /// Other outcomes retain their exact custody unchanged. In particular, a
    /// synchronization-unconfirmed result requires [`Self::reconcile`], never
    /// an ordinary destructor retry.
    pub fn retry(self) -> Self {
        Self {
            inner: self.inner.retry_pending(),
        }
    }

    /// Deliberately retry only stream synchronization after completion was unproved.
    ///
    /// This leaves every other teardown outcome unchanged and never retries a
    /// destructor after a quarantined result.
    pub fn reconcile(self) -> Self {
        Self {
            inner: self.inner.reconcile_synchronization(),
        }
    }

    /// Recover the exact resident model only after HIP acknowledged every
    /// session buffer and its ordered stream. All other custody remains owned
    /// by the returned teardown value without invoking HIP on drop.
    ///
    /// # Errors
    ///
    /// Returns the unchanged boxed teardown custody unless HIP acknowledged the
    /// complete session inventory and ordered stream.
    pub fn into_released_model(
        self,
    ) -> core::result::Result<Qwen35NativeExecutionModel, Box<Self>> {
        match self.inner.into_released_resident() {
            Ok(resources) => Ok(Qwen35NativeExecutionModel { resources }),
            Err(inner) => Err(Box::new(Self { inner: *inner })),
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
