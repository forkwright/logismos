//! Explicit one-way teardown state for HIP-owned resources.

use core::fmt;

use snafu::Snafu;

use crate::device::Device;
use crate::error::Error;
use crate::memory::{
    BufferRelease, BufferTeardownHandle, DeviceBuffer, TeardownBuffer, attempt_buffer_release,
};
use crate::pod::BytePod;
use crate::stream::{
    NonOwnedStream, Stream, StreamTeardownHandle, attempt_stream_destroy, attempt_stream_quiesce,
};

/// Kind of HIP resource represented by a teardown outcome.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum ResourceKind {
    /// A `hipMalloc` allocation.
    Buffer,
    /// An owned, non-default HIP stream.
    Stream,
}

/// Stable local identity of an entry captured for explicit teardown.
///
/// IDs are meaningful only inside the inventory or standalone transition that
/// issued them. They are deliberately not process-global resource identities.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum TeardownEntryId {
    /// A leaf transition outside an aggregate inventory.
    Standalone,
    /// The owned stream of an aggregate inventory.
    Stream,
    /// A buffer's zero-based registration position in an aggregate inventory.
    Buffer(usize),
}

/// Stage at which explicit teardown stopped.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum TeardownPhase {
    /// Selecting the recorded owning device failed before an operation call.
    Preflight,
    /// A stream did not prove that its queued work had completed.
    Synchronization,
    /// The runtime returned a non-success status from a destructor call.
    Destructor,
}

/// Immutable accounting facts for one HIP resource.
#[derive(Debug, Clone)]
pub struct ResourceMetadata {
    kind: ResourceKind,
    requested_bytes: usize,
    device: Device,
    entry: TeardownEntryId,
}

impl ResourceMetadata {
    pub(crate) fn new(
        kind: ResourceKind,
        requested_bytes: usize,
        device: Device,
        entry: TeardownEntryId,
    ) -> Self {
        Self {
            kind,
            requested_bytes,
            device,
            entry,
        }
    }

    /// Resource category requested from HIP.
    #[must_use]
    pub const fn kind(&self) -> ResourceKind {
        self.kind
    }

    /// Exact number of bytes requested for this resource.
    #[must_use]
    pub const fn requested_bytes(&self) -> usize {
        self.requested_bytes
    }

    /// Device recorded when the resource was created.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Local identity assigned when the teardown owner captured this resource.
    #[must_use]
    pub const fn entry(&self) -> TeardownEntryId {
        self.entry
    }
}

/// Typed reason an explicit teardown transition stopped.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum TeardownError {
    /// Device selection failed before an operation was invoked.
    #[snafu(display(
        "could not prepare {:?} teardown on device {}: {source}",
        resource.kind,
        resource.device.ordinal()
    ))]
    Preflight {
        /// Resource facts retained for accounting.
        resource: ResourceMetadata,
        /// The HIP wrapper failure.
        source: Error,
        /// Source code location where the failure was captured.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Synchronization could not establish that queued stream work completed.
    #[snafu(display(
        "could not synchronize {:?} teardown on device {}: {source}",
        resource.kind,
        resource.device.ordinal()
    ))]
    Synchronization {
        /// Resource facts retained for accounting.
        resource: ResourceMetadata,
        /// The HIP wrapper failure.
        source: Error,
        /// Source code location where the failure was captured.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// HIP did not acknowledge the destructor call, so ownership is indeterminate.
    #[snafu(display(
        "HIP did not acknowledge {:?} destruction on device {}: {source}",
        resource.kind,
        resource.device.ordinal()
    ))]
    Destructor {
        /// Resource facts retained for accounting.
        resource: ResourceMetadata,
        /// The HIP wrapper failure.
        source: Error,
        /// Source code location where the failure was captured.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}

impl TeardownError {
    /// Resource facts retained with this failure.
    #[must_use]
    pub fn resource(&self) -> &ResourceMetadata {
        match self {
            Self::Preflight { resource, .. }
            | Self::Synchronization { resource, .. }
            | Self::Destructor { resource, .. } => resource,
        }
    }

    /// Transition phase that stopped.
    #[must_use]
    pub const fn phase(&self) -> TeardownPhase {
        match self {
            Self::Preflight { .. } => TeardownPhase::Preflight,
            Self::Synchronization { .. } => TeardownPhase::Synchronization,
            Self::Destructor { .. } => TeardownPhase::Destructor,
        }
    }
}

/// Logical acknowledgement that HIP accepted one destructor request.
#[derive(Debug)]
pub struct ReleaseReceipt {
    resource: ResourceMetadata,
}

impl ReleaseReceipt {
    pub(crate) fn new(resource: ResourceMetadata) -> Self {
        Self { resource }
    }

    /// Resource facts whose destructor call HIP acknowledged.
    #[must_use]
    pub fn resource(&self) -> &ResourceMetadata {
        &self.resource
    }
}

/// Opaque terminal accounting record for a destructor with indeterminate outcome.
///
/// Dropping this record never performs HIP work. It deliberately exposes no
/// pointer, handle, or retry operation because HIP did not establish whether
/// the original resource is still live.
#[derive(Debug)]
pub struct TeardownTombstone {
    error: TeardownError,
}

impl TeardownTombstone {
    pub(crate) fn new(error: TeardownError) -> Self {
        Self { error }
    }

    /// Indeterminate destructor failure recorded by this tombstone.
    #[must_use]
    pub fn error(&self) -> &TeardownError {
        &self.error
    }
}

/// Private, already-disarmed owner used by the transition coordinator.
///
/// `R` must be an inert handle whose Rust destructor performs no HIP work.
/// `DeviceBuffer` and `Stream` are converted to such handles before entering
/// this type, so every ordinary drop path is non-destructive.
#[derive(Debug)]
pub(crate) struct ReleaseOwner<R> {
    resource: R,
    metadata: ResourceMetadata,
}

impl<R> ReleaseOwner<R> {
    pub(crate) fn new(resource: R, metadata: ResourceMetadata) -> Self {
        Self { resource, metadata }
    }

    pub(crate) fn metadata(&self) -> &ResourceMetadata {
        &self.metadata
    }

    pub(crate) fn with_entry(mut self, entry: TeardownEntryId) -> Self {
        self.metadata.entry = entry;
        self
    }

    pub(crate) fn attempt(
        self,
        phase: TeardownPhase,
        prepare: impl FnOnce(&mut R, &ResourceMetadata) -> Result<(), Error>,
        destroy: impl FnOnce(&mut R, &ResourceMetadata) -> Result<(), Error>,
    ) -> ReleaseAttempt<R> {
        let mut owner = match self.prepare(phase, prepare) {
            Ok(owner) => owner,
            Err(pending) => return ReleaseAttempt::Pending(*pending),
        };
        if let Err(source) = destroy(&mut owner.resource, &owner.metadata) {
            let error = TeardownError::Destructor {
                resource: owner.metadata.clone(),
                source,
                location: core::panic::Location::caller(),
            };
            return ReleaseAttempt::Quarantined {
                owner,
                tombstone: TeardownTombstone::new(error),
            };
        }
        let Self { resource, metadata } = owner;
        drop(resource);
        ReleaseAttempt::Released(ReleaseReceipt::new(metadata))
    }

    pub(crate) fn prepare(
        mut self,
        phase: TeardownPhase,
        operation: impl FnOnce(&mut R, &ResourceMetadata) -> Result<(), Error>,
    ) -> core::result::Result<Self, Box<PendingOwner<R>>> {
        if let Err(source) = operation(&mut self.resource, &self.metadata) {
            return Err(Box::new(self.pending(phase, source)));
        }
        Ok(self)
    }

    pub(crate) fn pending(self, phase: TeardownPhase, source: Error) -> PendingOwner<R> {
        let error = match phase {
            TeardownPhase::Preflight => TeardownError::Preflight {
                resource: self.metadata.clone(),
                source,
                location: core::panic::Location::caller(),
            },
            TeardownPhase::Synchronization => TeardownError::Synchronization {
                resource: self.metadata.clone(),
                source,
                location: core::panic::Location::caller(),
            },
            TeardownPhase::Destructor => TeardownError::Destructor {
                resource: self.metadata.clone(),
                source,
                location: core::panic::Location::caller(),
            },
        };
        PendingOwner { owner: self, error }
    }
}

/// Internal result from one destructor transition.
pub(crate) enum ReleaseAttempt<R> {
    Released(ReleaseReceipt),
    Pending(PendingOwner<R>),
    Quarantined {
        owner: ReleaseOwner<R>,
        tombstone: TeardownTombstone,
    },
}

/// Internal result from the two-stage stream quiescence transition.
pub(crate) enum QuiesceAttempt<R> {
    Quiescent(ReleaseOwner<R>),
    PreflightPending(PendingOwner<R>),
    SynchronizationUnconfirmed(PendingOwner<R>),
}

/// Non-usable retained owner after a call known not to have reached destruction.
#[derive(Debug)]
pub(crate) struct PendingOwner<R> {
    owner: ReleaseOwner<R>,
    error: TeardownError,
}

impl<R> PendingOwner<R> {
    pub(crate) fn error(&self) -> &TeardownError {
        &self.error
    }

    pub(crate) fn into_owner(self) -> ReleaseOwner<R> {
        self.owner
    }
}

/// Immutable expected entries and accumulated destructor acknowledgements.
///
/// The byte total is the sum of requested allocation sizes, not measured
/// physical residency. Every aggregate outcome retains this same full total.
#[derive(Debug)]
pub struct InventoryEvidence {
    requested_bytes: usize,
    entries: Vec<ResourceMetadata>,
    released: Vec<ReleaseReceipt>,
}

impl InventoryEvidence {
    fn new(stream: &ResourceMetadata) -> Self {
        Self {
            requested_bytes: stream.requested_bytes(),
            entries: vec![stream.clone()],
            released: Vec::new(),
        }
    }

    fn register(
        &mut self,
        resource: &ResourceMetadata,
    ) -> core::result::Result<(), InventoryAccountingError> {
        let requested_bytes = self
            .requested_bytes
            .checked_add(resource.requested_bytes())
            .ok_or_else(|| InventoryAccountingError {
                current_requested_bytes: self.requested_bytes,
                additional_requested_bytes: resource.requested_bytes(),
            })?;
        self.requested_bytes = requested_bytes;
        self.entries.push(resource.clone());
        Ok(())
    }

    fn acknowledge(&mut self, receipt: ReleaseReceipt) {
        self.released.push(receipt);
    }

    /// Full conservative requested-byte extent captured before teardown.
    #[must_use]
    pub const fn requested_bytes(&self) -> usize {
        self.requested_bytes
    }

    /// Every entry captured by this inventory, in registration order.
    #[must_use]
    pub fn entries(&self) -> &[ResourceMetadata] {
        &self.entries
    }

    /// Entries whose destructors have been acknowledged, in release order.
    #[must_use]
    pub fn released(&self) -> &[ReleaseReceipt] {
        &self.released
    }
}

/// Checked-accounting failure while adding a buffer to an inventory.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct InventoryAccountingError {
    current_requested_bytes: usize,
    additional_requested_bytes: usize,
}

impl InventoryAccountingError {
    /// Requested bytes already captured by the inventory.
    #[must_use]
    pub const fn current_requested_bytes(&self) -> usize {
        self.current_requested_bytes
    }

    /// Requested bytes of the rejected buffer.
    #[must_use]
    pub const fn additional_requested_bytes(&self) -> usize {
        self.additional_requested_bytes
    }
}

impl fmt::Display for InventoryAccountingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "teardown inventory requested-byte extent overflow: {} + {}",
            self.current_requested_bytes, self.additional_requested_bytes
        )
    }
}

impl std::error::Error for InventoryAccountingError {}

/// Rejected buffer retained after checked inventory accounting overflowed.
///
/// The buffer was already disarmed before this value was returned. Dropping
/// this value performs no HIP work; call [`Self::begin_release`] to attempt an
/// explicit standalone release.
#[must_use = "the rejected allocation remains owned and must stay accounted"]
pub struct InventoryPushError {
    error: InventoryAccountingError,
    buffer: TeardownBuffer,
}

impl InventoryPushError {
    /// Checked accounting failure that rejected the buffer.
    #[must_use]
    pub const fn error(&self) -> &InventoryAccountingError {
        &self.error
    }

    /// Metadata for the rejected, still-owned allocation.
    #[must_use]
    pub fn resource(&self) -> &ResourceMetadata {
        self.buffer.resource()
    }

    /// Recover the rejected inert buffer for retention or another inventory.
    pub fn into_buffer(self) -> TeardownBuffer {
        self.buffer
    }

    /// Explicitly release the rejected allocation as a standalone buffer.
    pub fn begin_release(self) -> BufferRelease {
        self.buffer.begin_release()
    }
}

/// Heterogeneous teardown owner for one explicit stream and its touched buffers.
///
/// Every buffer is type-erased only after ownership is transferred into an
/// inert raw-handle owner. `u8`, `f32`, `u32`, and other [`BytePod`] buffers can
/// therefore share the same inventory without any ordinary resource destructor
/// remaining reachable.
#[must_use = "dropping an inventory abandons native handles without acknowledging release"]
pub struct TeardownInventory {
    stream: ReleaseOwner<StreamTeardownHandle>,
    buffers: Vec<ReleaseOwner<BufferTeardownHandle>>,
    evidence: InventoryEvidence,
}

impl TeardownInventory {
    /// Begin an inventory around an owned, non-default stream.
    ///
    /// # Errors
    ///
    /// Returns the original stream inside [`NonOwnedStream`] when `stream` is
    /// the non-owned NULL stream. This refusal occurs before any buffer can be
    /// registered or disarmed.
    pub fn try_new(stream: Stream) -> core::result::Result<Self, NonOwnedStream> {
        let stream = stream.into_release_owner(TeardownEntryId::Stream)?;
        let evidence = InventoryEvidence::new(stream.metadata());
        Ok(Self {
            stream,
            buffers: Vec::new(),
            evidence,
        })
    }

    /// Retain a typed buffer and assign its stable inventory-local identity.
    ///
    /// # Errors
    ///
    /// Returns an opaque, explicitly releasable owner if adding the requested
    /// byte extent would overflow `usize`. No HIP operation is attempted.
    pub fn push_buffer<T: BytePod>(
        &mut self,
        buffer: DeviceBuffer<T>,
    ) -> core::result::Result<TeardownEntryId, InventoryPushError> {
        self.push_teardown_buffer(buffer.into_teardown())
    }

    /// Retain an already-disarmed buffer and assign its inventory-local identity.
    ///
    /// # Errors
    ///
    /// Returns the same opaque buffer owner if checked byte accounting overflows.
    pub fn push_teardown_buffer(
        &mut self,
        buffer: TeardownBuffer,
    ) -> core::result::Result<TeardownEntryId, InventoryPushError> {
        let entry = TeardownEntryId::Buffer(self.buffers.len());
        let buffer = buffer.with_entry(entry);
        if let Err(error) = self.evidence.register(buffer.metadata()) {
            return Err(InventoryPushError {
                error,
                buffer: TeardownBuffer::from_owner(buffer),
            });
        }
        self.buffers.push(buffer);
        Ok(entry)
    }

    /// Full inventory evidence captured so far.
    #[must_use]
    pub const fn evidence(&self) -> &InventoryEvidence {
        &self.evidence
    }

    /// Consume, quiesce, then explicitly release every retained resource.
    pub fn begin_release(self) -> InventoryRelease {
        let Self {
            stream,
            buffers,
            evidence,
        } = self;
        let mut operations = HipInventoryOperations;
        map_coordinator_outcome(begin_coordinator(
            stream,
            buffers,
            evidence,
            &mut operations,
        ))
    }
}

/// Successful evidence that every captured destructor was acknowledged.
#[derive(Debug)]
pub struct InventoryReceipt {
    evidence: InventoryEvidence,
}

impl InventoryReceipt {
    /// Complete expected-entry and acknowledgement evidence.
    #[must_use]
    pub const fn evidence(&self) -> &InventoryEvidence {
        &self.evidence
    }

    /// Per-resource HIP destructor acknowledgements in release order.
    #[must_use]
    pub fn released(&self) -> &[ReleaseReceipt] {
        self.evidence.released()
    }
}

/// Explicit aggregate teardown outcome.
#[non_exhaustive]
#[must_use = "teardown outcomes carry native ownership and accounting evidence"]
pub enum InventoryRelease {
    /// Every captured buffer and the quiescent stream acknowledged destruction.
    Released(InventoryReceipt),
    /// A pre-call failure retained a retryable owner without invoking a destructor.
    Pending(PendingInventory),
    /// Stream completion remains unproved; only explicit reconciliation is allowed.
    SynchronizationUnconfirmed(InventorySynchronizationUnconfirmed),
    /// A destructor outcome was indeterminate; no retry or native access remains.
    Quarantined(InventoryQuarantine),
}

/// Retryable aggregate retained after a failure before the relevant HIP call.
#[must_use = "the retained inventory must remain accounted or be explicitly retried"]
pub struct PendingInventory {
    state: PendingState<StreamTeardownHandle, BufferTeardownHandle>,
    evidence: InventoryEvidence,
}

impl PendingInventory {
    /// Failure that stopped the known-not-attempted transition.
    #[must_use]
    pub fn error(&self) -> &TeardownError {
        self.state.error()
    }

    /// Full expected inventory and acknowledgements completed before failure.
    #[must_use]
    pub const fn evidence(&self) -> &InventoryEvidence {
        &self.evidence
    }

    /// Retry only the transition known not to have reached its HIP call.
    pub fn retry(self) -> InventoryRelease {
        let mut operations = HipInventoryOperations;
        map_coordinator_outcome(retry_pending(self.state, self.evidence, &mut operations))
    }
}

/// Aggregate retained because `hipStreamSynchronize` did not prove completion.
///
/// This is not an ordinary retryable eviction failure. The stream and all
/// buffers remain non-usable and fully charged. A caller may deliberately run
/// [`Self::reconcile`] to issue another non-destructive synchronization attempt.
#[must_use = "completion is unproved; retain the full charge or reconcile explicitly"]
pub struct InventorySynchronizationUnconfirmed {
    stream: PendingOwner<StreamTeardownHandle>,
    buffers: Vec<ReleaseOwner<BufferTeardownHandle>>,
    evidence: InventoryEvidence,
}

impl InventorySynchronizationUnconfirmed {
    /// Synchronization failure that left completion unproved.
    #[must_use]
    pub fn error(&self) -> &TeardownError {
        self.stream.error()
    }

    /// Full expected inventory; no buffer release follows a failed synchronization.
    #[must_use]
    pub const fn evidence(&self) -> &InventoryEvidence {
        &self.evidence
    }

    /// Deliberately attempt to reconcile completion by synchronizing again.
    pub fn reconcile(self) -> InventoryRelease {
        let mut operations = HipInventoryOperations;
        map_coordinator_outcome(reconcile_synchronization(
            self.stream,
            self.buffers,
            self.evidence,
            &mut operations,
        ))
    }
}

/// Opaque conservative charge after an indeterminate destructor outcome.
///
/// This state exposes evidence but neither raw native handles nor a retry. Its
/// full requested-byte extent remains charged even when some entries have
/// acknowledged release; it does not claim measured retained physical bytes.
#[must_use = "the quarantined inventory must retain its full conservative reservation"]
pub struct InventoryQuarantine {
    evidence: InventoryEvidence,
    tombstone: TeardownTombstone,
    _held: HeldInventory<StreamTeardownHandle, BufferTeardownHandle>,
}

impl InventoryQuarantine {
    /// Full conservative requested-byte extent of the original inventory.
    #[must_use]
    pub const fn requested_bytes(&self) -> usize {
        self.evidence.requested_bytes()
    }

    /// Expected entries and per-entry acknowledgements completed before uncertainty.
    #[must_use]
    pub const fn evidence(&self) -> &InventoryEvidence {
        &self.evidence
    }

    /// Indeterminate destructor failure that stopped teardown.
    #[must_use]
    pub fn error(&self) -> &TeardownError {
        self.tombstone.error()
    }
}

trait InventoryOperations<S, B> {
    fn quiesce(&mut self, stream: ReleaseOwner<S>) -> QuiesceAttempt<S>;
    fn release_buffer(&mut self, buffer: ReleaseOwner<B>) -> ReleaseAttempt<B>;
    fn release_stream(&mut self, stream: ReleaseOwner<S>) -> ReleaseAttempt<S>;
}

struct HipInventoryOperations;

impl InventoryOperations<StreamTeardownHandle, BufferTeardownHandle> for HipInventoryOperations {
    fn quiesce(
        &mut self,
        stream: ReleaseOwner<StreamTeardownHandle>,
    ) -> QuiesceAttempt<StreamTeardownHandle> {
        attempt_stream_quiesce(stream)
    }

    fn release_buffer(
        &mut self,
        buffer: ReleaseOwner<BufferTeardownHandle>,
    ) -> ReleaseAttempt<BufferTeardownHandle> {
        attempt_buffer_release(buffer)
    }

    fn release_stream(
        &mut self,
        stream: ReleaseOwner<StreamTeardownHandle>,
    ) -> ReleaseAttempt<StreamTeardownHandle> {
        attempt_stream_destroy(stream)
    }
}

enum PendingState<S, B> {
    Synchronizing {
        stream: PendingOwner<S>,
        buffers: Vec<ReleaseOwner<B>>,
    },
    Releasing {
        stream: ReleaseOwner<S>,
        buffer: PendingOwner<B>,
        buffers: Vec<ReleaseOwner<B>>,
    },
    Destroying {
        stream: PendingOwner<S>,
    },
}

impl<S, B> PendingState<S, B> {
    fn error(&self) -> &TeardownError {
        match self {
            Self::Synchronizing { stream, .. } | Self::Destroying { stream } => stream.error(),
            Self::Releasing { buffer, .. } => buffer.error(),
        }
    }
}

struct HeldInventory<S, B> {
    _stream: Option<ReleaseOwner<S>>,
    _buffer: Option<ReleaseOwner<B>>,
    _buffers: Vec<ReleaseOwner<B>>,
}

enum CoordinatorOutcome<S, B> {
    Released(InventoryEvidence),
    Pending {
        state: PendingState<S, B>,
        evidence: InventoryEvidence,
    },
    SynchronizationUnconfirmed {
        stream: PendingOwner<S>,
        buffers: Vec<ReleaseOwner<B>>,
        evidence: InventoryEvidence,
    },
    Quarantined {
        evidence: InventoryEvidence,
        tombstone: TeardownTombstone,
        held: HeldInventory<S, B>,
    },
}

fn begin_coordinator<S, B, O>(
    stream: ReleaseOwner<S>,
    buffers: Vec<ReleaseOwner<B>>,
    evidence: InventoryEvidence,
    operations: &mut O,
) -> CoordinatorOutcome<S, B>
where
    O: InventoryOperations<S, B>,
{
    let quiesce = operations.quiesce(stream);
    continue_after_quiesce(quiesce, buffers, evidence, operations)
}

fn continue_after_quiesce<S, B, O>(
    quiesce: QuiesceAttempt<S>,
    buffers: Vec<ReleaseOwner<B>>,
    evidence: InventoryEvidence,
    operations: &mut O,
) -> CoordinatorOutcome<S, B>
where
    O: InventoryOperations<S, B>,
{
    match quiesce {
        QuiesceAttempt::Quiescent(stream) => {
            resume_buffers(stream, None, buffers, evidence, operations)
        }
        QuiesceAttempt::PreflightPending(stream) => CoordinatorOutcome::Pending {
            state: PendingState::Synchronizing { stream, buffers },
            evidence,
        },
        QuiesceAttempt::SynchronizationUnconfirmed(stream) => {
            CoordinatorOutcome::SynchronizationUnconfirmed {
                stream,
                buffers,
                evidence,
            }
        }
    }
}

fn retry_pending<S, B, O>(
    state: PendingState<S, B>,
    evidence: InventoryEvidence,
    operations: &mut O,
) -> CoordinatorOutcome<S, B>
where
    O: InventoryOperations<S, B>,
{
    match state {
        PendingState::Synchronizing { stream, buffers } => {
            let quiesce = operations.quiesce(stream.into_owner());
            continue_after_quiesce(quiesce, buffers, evidence, operations)
        }
        PendingState::Releasing {
            stream,
            buffer,
            buffers,
        } => resume_buffers(
            stream,
            Some(buffer.into_owner()),
            buffers,
            evidence,
            operations,
        ),
        PendingState::Destroying { stream } => {
            finish_inventory(stream.into_owner(), evidence, operations)
        }
    }
}

fn reconcile_synchronization<S, B, O>(
    stream: PendingOwner<S>,
    buffers: Vec<ReleaseOwner<B>>,
    evidence: InventoryEvidence,
    operations: &mut O,
) -> CoordinatorOutcome<S, B>
where
    O: InventoryOperations<S, B>,
{
    let quiesce = operations.quiesce(stream.into_owner());
    match quiesce {
        QuiesceAttempt::Quiescent(stream) => {
            resume_buffers(stream, None, buffers, evidence, operations)
        }
        QuiesceAttempt::PreflightPending(stream)
        | QuiesceAttempt::SynchronizationUnconfirmed(stream) => {
            CoordinatorOutcome::SynchronizationUnconfirmed {
                stream,
                buffers,
                evidence,
            }
        }
    }
}

fn resume_buffers<S, B, O>(
    stream: ReleaseOwner<S>,
    current: Option<ReleaseOwner<B>>,
    mut buffers: Vec<ReleaseOwner<B>>,
    mut evidence: InventoryEvidence,
    operations: &mut O,
) -> CoordinatorOutcome<S, B>
where
    O: InventoryOperations<S, B>,
{
    let mut current = current;
    loop {
        let Some(buffer) = current.take().or_else(|| buffers.pop()) else {
            return finish_inventory(stream, evidence, operations);
        };
        match operations.release_buffer(buffer) {
            ReleaseAttempt::Released(receipt) => evidence.acknowledge(receipt),
            ReleaseAttempt::Pending(buffer) => {
                return CoordinatorOutcome::Pending {
                    state: PendingState::Releasing {
                        stream,
                        buffer,
                        buffers,
                    },
                    evidence,
                };
            }
            ReleaseAttempt::Quarantined {
                owner: buffer,
                tombstone,
            } => {
                return CoordinatorOutcome::Quarantined {
                    evidence,
                    tombstone,
                    held: HeldInventory {
                        _stream: Some(stream),
                        _buffer: Some(buffer),
                        _buffers: buffers,
                    },
                };
            }
        }
    }
}

fn finish_inventory<S, B, O>(
    stream: ReleaseOwner<S>,
    mut evidence: InventoryEvidence,
    operations: &mut O,
) -> CoordinatorOutcome<S, B>
where
    O: InventoryOperations<S, B>,
{
    match operations.release_stream(stream) {
        ReleaseAttempt::Released(receipt) => {
            evidence.acknowledge(receipt);
            CoordinatorOutcome::Released(evidence)
        }
        ReleaseAttempt::Pending(stream) => CoordinatorOutcome::Pending {
            state: PendingState::Destroying { stream },
            evidence,
        },
        ReleaseAttempt::Quarantined {
            owner: stream,
            tombstone,
        } => CoordinatorOutcome::Quarantined {
            evidence,
            tombstone,
            held: HeldInventory {
                _stream: Some(stream),
                _buffer: None,
                _buffers: Vec::new(),
            },
        },
    }
}

fn map_coordinator_outcome(
    outcome: CoordinatorOutcome<StreamTeardownHandle, BufferTeardownHandle>,
) -> InventoryRelease {
    match outcome {
        CoordinatorOutcome::Released(evidence) => {
            InventoryRelease::Released(InventoryReceipt { evidence })
        }
        CoordinatorOutcome::Pending { state, evidence } => {
            InventoryRelease::Pending(PendingInventory { state, evidence })
        }
        CoordinatorOutcome::SynchronizationUnconfirmed {
            stream,
            buffers,
            evidence,
        } => InventoryRelease::SynchronizationUnconfirmed(InventorySynchronizationUnconfirmed {
            stream,
            buffers,
            evidence,
        }),
        CoordinatorOutcome::Quarantined {
            evidence,
            tombstone,
            held,
        } => InventoryRelease::Quarantined(InventoryQuarantine {
            evidence,
            tombstone,
            _held: held,
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::io;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;

    type TestResult = core::result::Result<(), Box<dyn std::error::Error>>;

    #[derive(Debug)]
    struct FakeHandle;

    #[derive(Debug)]
    struct DropTrackedHandle {
        drops: Arc<AtomicUsize>,
    }

    impl Drop for DropTrackedHandle {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[derive(Clone, Copy)]
    enum QuiesceStep {
        Success,
        PreflightFailure,
        SynchronizationFailure,
    }

    #[derive(Clone, Copy)]
    enum ReleaseStep {
        Success,
        PreflightFailure,
        DestructorFailure,
    }

    struct ScriptedOperations {
        quiesce: VecDeque<QuiesceStep>,
        buffers: VecDeque<ReleaseStep>,
        stream: VecDeque<ReleaseStep>,
        calls: Vec<String>,
    }

    impl ScriptedOperations {
        fn new(
            quiesce: impl IntoIterator<Item = QuiesceStep>,
            buffers: impl IntoIterator<Item = ReleaseStep>,
            stream: impl IntoIterator<Item = ReleaseStep>,
        ) -> Self {
            Self {
                quiesce: quiesce.into_iter().collect(),
                buffers: buffers.into_iter().collect(),
                stream: stream.into_iter().collect(),
                calls: Vec::new(),
            }
        }

        fn release(
            calls: &mut Vec<String>,
            step: ReleaseStep,
            owner: ReleaseOwner<FakeHandle>,
        ) -> ReleaseAttempt<FakeHandle> {
            calls.push(format!("preflight:{:?}", owner.metadata().entry()));
            match step {
                ReleaseStep::PreflightFailure => {
                    ReleaseAttempt::Pending(owner.pending(TeardownPhase::Preflight, failure()))
                }
                ReleaseStep::Success => {
                    calls.push(format!("destroy:{:?}", owner.metadata().entry()));
                    owner.attempt(TeardownPhase::Preflight, |_, _| Ok(()), |_, _| Ok(()))
                }
                ReleaseStep::DestructorFailure => {
                    calls.push(format!("destroy:{:?}", owner.metadata().entry()));
                    owner.attempt(
                        TeardownPhase::Preflight,
                        |_, _| Ok(()),
                        |_, _| Err(failure()),
                    )
                }
            }
        }
    }

    impl InventoryOperations<FakeHandle, FakeHandle> for ScriptedOperations {
        fn quiesce(&mut self, stream: ReleaseOwner<FakeHandle>) -> QuiesceAttempt<FakeHandle> {
            self.calls.push("stream-preflight".to_string());
            let step = self
                .quiesce
                .pop_front()
                .unwrap_or(QuiesceStep::SynchronizationFailure);
            match step {
                QuiesceStep::PreflightFailure => QuiesceAttempt::PreflightPending(
                    stream.pending(TeardownPhase::Preflight, failure()),
                ),
                QuiesceStep::SynchronizationFailure => {
                    self.calls.push("stream-synchronize".to_string());
                    QuiesceAttempt::SynchronizationUnconfirmed(
                        stream.pending(TeardownPhase::Synchronization, failure()),
                    )
                }
                QuiesceStep::Success => {
                    self.calls.push("stream-synchronize".to_string());
                    QuiesceAttempt::Quiescent(stream)
                }
            }
        }

        fn release_buffer(
            &mut self,
            buffer: ReleaseOwner<FakeHandle>,
        ) -> ReleaseAttempt<FakeHandle> {
            let step = self
                .buffers
                .pop_front()
                .unwrap_or(ReleaseStep::DestructorFailure);
            Self::release(&mut self.calls, step, buffer)
        }

        fn release_stream(
            &mut self,
            stream: ReleaseOwner<FakeHandle>,
        ) -> ReleaseAttempt<FakeHandle> {
            let step = self
                .stream
                .pop_front()
                .unwrap_or(ReleaseStep::DestructorFailure);
            Self::release(&mut self.calls, step, stream)
        }
    }

    fn failure() -> Error {
        Error::runtime(1, "synthetic teardown")
    }

    fn metadata(entry: TeardownEntryId, bytes: usize) -> ResourceMetadata {
        ResourceMetadata::new(
            if matches!(entry, TeardownEntryId::Stream) {
                ResourceKind::Stream
            } else {
                ResourceKind::Buffer
            },
            bytes,
            Device::for_test(0),
            entry,
        )
    }

    fn owner(entry: TeardownEntryId, bytes: usize) -> ReleaseOwner<FakeHandle> {
        ReleaseOwner::new(FakeHandle, metadata(entry, bytes))
    }

    fn fixture() -> core::result::Result<
        (
            ReleaseOwner<FakeHandle>,
            Vec<ReleaseOwner<FakeHandle>>,
            InventoryEvidence,
        ),
        InventoryAccountingError,
    > {
        let stream = owner(TeardownEntryId::Stream, 0);
        let mut evidence = InventoryEvidence::new(stream.metadata());
        let first = owner(TeardownEntryId::Buffer(0), 16);
        let second = owner(TeardownEntryId::Buffer(1), 16);
        evidence.register(first.metadata())?;
        evidence.register(second.metadata())?;
        Ok((stream, vec![first, second], evidence))
    }

    #[test]
    fn production_coordinator_retries_only_unstarted_entry() -> TestResult {
        let (stream, buffers, evidence) = fixture()?;
        let mut operations = ScriptedOperations::new(
            [QuiesceStep::Success],
            [
                ReleaseStep::Success,
                ReleaseStep::PreflightFailure,
                ReleaseStep::Success,
            ],
            [ReleaseStep::Success],
        );
        let outcome = begin_coordinator(stream, buffers, evidence, &mut operations);
        let CoordinatorOutcome::Pending { state, evidence } = outcome else {
            return Err(io::Error::other("second buffer must remain pending").into());
        };
        assert_eq!(evidence.released().len(), 1);
        assert_eq!(
            evidence.released()[0].resource().entry(),
            TeardownEntryId::Buffer(1)
        );

        let outcome = retry_pending(state, evidence, &mut operations);
        let CoordinatorOutcome::Released(evidence) = outcome else {
            return Err(io::Error::other("retry must finish the retained suffix").into());
        };
        assert_eq!(evidence.released().len(), 3);
        assert_eq!(evidence.requested_bytes(), 32);
        assert_eq!(
            operations.calls.join("|"),
            "stream-preflight|stream-synchronize|preflight:Buffer(1)|\
             destroy:Buffer(1)|preflight:Buffer(0)|preflight:Buffer(0)|\
             destroy:Buffer(0)|preflight:Stream|destroy:Stream"
        );
        Ok(())
    }

    #[test]
    fn synchronization_failure_releases_nothing_until_reconciled() -> TestResult {
        let (stream, buffers, evidence) = fixture()?;
        let mut operations = ScriptedOperations::new(
            [QuiesceStep::SynchronizationFailure, QuiesceStep::Success],
            [ReleaseStep::Success, ReleaseStep::Success],
            [ReleaseStep::Success],
        );
        let outcome = begin_coordinator(stream, buffers, evidence, &mut operations);
        let CoordinatorOutcome::SynchronizationUnconfirmed {
            stream,
            buffers,
            evidence,
        } = outcome
        else {
            return Err(io::Error::other("failed synchronization must be a distinct state").into());
        };
        assert!(evidence.released().is_empty());
        assert_eq!(
            operations.calls.join("|"),
            "stream-preflight|stream-synchronize"
        );

        let outcome = reconcile_synchronization(stream, buffers, evidence, &mut operations);
        assert!(matches!(outcome, CoordinatorOutcome::Released(_)));
        Ok(())
    }

    #[test]
    fn destructor_failure_quarantines_without_touching_suffix_or_stream() -> TestResult {
        let (stream, buffers, evidence) = fixture()?;
        let mut operations = ScriptedOperations::new(
            [QuiesceStep::Success],
            [ReleaseStep::DestructorFailure, ReleaseStep::Success],
            [ReleaseStep::Success],
        );
        let outcome = begin_coordinator(stream, buffers, evidence, &mut operations);
        let CoordinatorOutcome::Quarantined {
            evidence,
            tombstone,
            ..
        } = outcome
        else {
            return Err(
                io::Error::other("destructor failure must quarantine the inventory").into(),
            );
        };
        assert!(evidence.released().is_empty());
        assert_eq!(evidence.requested_bytes(), 32);
        assert_eq!(
            tombstone.error().resource().entry(),
            TeardownEntryId::Buffer(1)
        );
        assert_eq!(
            operations.calls.join("|"),
            "stream-preflight|stream-synchronize|preflight:Buffer(1)|destroy:Buffer(1)"
        );
        Ok(())
    }

    #[test]
    fn preflight_failure_never_invokes_destructor() {
        let resource = owner(TeardownEntryId::Standalone, 16);
        let mut destructor_calls = 0;
        let outcome = resource.attempt(
            TeardownPhase::Preflight,
            |_, _| Err(failure()),
            |_, _| {
                destructor_calls += 1;
                Ok(())
            },
        );
        assert!(matches!(outcome, ReleaseAttempt::Pending(_)));
        assert_eq!(destructor_calls, 0);
    }

    #[test]
    fn acknowledged_release_drops_inert_rust_metadata_once() {
        let drops = Arc::new(AtomicUsize::new(0));
        let owner = ReleaseOwner::new(
            DropTrackedHandle {
                drops: Arc::clone(&drops),
            },
            metadata(TeardownEntryId::Standalone, 16),
        );
        let outcome = owner.attempt(TeardownPhase::Preflight, |_, _| Ok(()), |_, _| Ok(()));
        assert!(matches!(outcome, ReleaseAttempt::Released(_)));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn duplicate_extents_retain_distinct_entry_identity() -> TestResult {
        let (_, _, evidence) = fixture()?;
        assert_eq!(evidence.entries().len(), 3);
        assert_eq!(evidence.entries()[1].requested_bytes(), 16);
        assert_eq!(evidence.entries()[2].requested_bytes(), 16);
        assert_ne!(evidence.entries()[1].entry(), evidence.entries()[2].entry());
        Ok(())
    }

    #[test]
    fn scripted_preflight_can_be_retried_without_synchronizing_early() -> TestResult {
        let (stream, buffers, evidence) = fixture()?;
        let mut operations = ScriptedOperations::new(
            [QuiesceStep::PreflightFailure, QuiesceStep::Success],
            [ReleaseStep::Success, ReleaseStep::Success],
            [ReleaseStep::Success],
        );
        let outcome = begin_coordinator(stream, buffers, evidence, &mut operations);
        let CoordinatorOutcome::Pending { state, evidence } = outcome else {
            return Err(io::Error::other("preflight failure must remain retryable").into());
        };
        assert_eq!(operations.calls.join("|"), "stream-preflight");
        assert!(matches!(
            retry_pending(state, evidence, &mut operations),
            CoordinatorOutcome::Released(_)
        ));
        Ok(())
    }

    #[test]
    fn reconciliation_preflight_failure_remains_completion_unconfirmed() -> TestResult {
        let (stream, buffers, evidence) = fixture()?;
        let mut operations = ScriptedOperations::new(
            [
                QuiesceStep::SynchronizationFailure,
                QuiesceStep::PreflightFailure,
            ],
            [],
            [],
        );
        let outcome = begin_coordinator(stream, buffers, evidence, &mut operations);
        let CoordinatorOutcome::SynchronizationUnconfirmed {
            stream,
            buffers,
            evidence,
        } = outcome
        else {
            return Err(io::Error::other("initial sync failure must be unconfirmed").into());
        };
        assert!(matches!(
            reconcile_synchronization(stream, buffers, evidence, &mut operations),
            CoordinatorOutcome::SynchronizationUnconfirmed { .. }
        ));
        assert_eq!(
            operations.calls.join("|"),
            "stream-preflight|stream-synchronize|stream-preflight"
        );
        Ok(())
    }
}
