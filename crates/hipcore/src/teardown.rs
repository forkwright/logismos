//! Explicit one-way teardown state for HIP-owned resources.

use core::mem::ManuallyDrop;

use snafu::Snafu;

use crate::device::Device;
use crate::error::{Error, InternalSnafu};
use crate::memory::{DeviceBuffer, attempt_buffer_release};
use crate::pod::BytePod;
use crate::stream::{Stream, StreamQuiesce, attempt_stream_release, quiesce_stream_owner};

/// Kind of HIP resource represented by a teardown outcome.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum ResourceKind {
    /// A `hipMalloc` allocation.
    Buffer,
    /// A non-default HIP stream.
    Stream,
}
/// Stage at which explicit teardown stopped.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum TeardownPhase {
    /// Selecting the recorded owning device failed before a destructor call.
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

    /// Stable entry identity within the teardown owner that captured it.
    #[must_use]
    pub const fn entry(&self) -> TeardownEntryId {
        self.entry
    }
}

/// Stable local identity of an entry captured for explicit teardown.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct TeardownEntryId(usize);

impl TeardownEntryId {
    pub(crate) const fn standalone() -> Self {
        Self(0)
    }

    pub(crate) const fn new(value: usize) -> Self {
        Self(value)
    }

    /// Zero-based position assigned when the teardown owner captured this entry.
    #[must_use]
    pub const fn position(self) -> usize {
        self.0
    }
}

/// Typed reason an explicit teardown transition stopped.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum TeardownError {
    /// Device selection failed before a destructor was invoked.
    #[snafu(display("could not prepare {:?} teardown on device {}: {source}", resource.kind, resource.device.ordinal()))]
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
    #[snafu(display("could not synchronize {:?} teardown on device {}: {source}", resource.kind, resource.device.ordinal()))]
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
    #[snafu(display("HIP did not acknowledge {:?} destruction on device {}: {source}", resource.kind, resource.device.ordinal()))]
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

/// Private, disarmed owner shared by real HIP resources and pure transition tests.
///
/// `ManuallyDrop` is installed before any transition callback runs. Therefore a
/// failure cannot fall back to an ordinary resource `Drop` implementation.
pub(crate) struct ReleaseOwner<R> {
    resource: ManuallyDrop<R>,
    metadata: ResourceMetadata,
    dispose_after_ack: fn(R),
}

impl<R> ReleaseOwner<R> {
    pub(crate) fn new(resource: R, metadata: ResourceMetadata, dispose_after_ack: fn(R)) -> Self {
        Self {
            resource: ManuallyDrop::new(resource),
            metadata,
            dispose_after_ack,
        }
    }

    pub(crate) fn requested_bytes(&self) -> usize {
        self.metadata.requested_bytes
    }

    pub(crate) fn attempt(
        self,
        phase: TeardownPhase,
        prepare: impl FnOnce(&mut R) -> Result<(), Error>,
        destroy: impl FnOnce(&mut R) -> Result<(), Error>,
    ) -> ReleaseAttempt<R> {
        let mut owner = match self.prepare(phase, prepare) {
            Ok(owner) => owner,
            Err(pending) => return ReleaseAttempt::Pending(pending),
        };
        if let Err(source) = destroy(&mut owner.resource) {
            let error = TeardownError::Destructor {
                resource: owner.metadata.clone(),
                source,
                location: snafu::Location::caller(),
            };
            return ReleaseAttempt::Quarantined {
                owner,
                tombstone: TeardownTombstone::new(error),
            };
        }
        let resource = ManuallyDrop::into_inner(owner.resource);
        (owner.dispose_after_ack)(resource);
        ReleaseAttempt::Released(ReleaseReceipt::new(owner.metadata))
    }

    pub(crate) fn prepare(
        mut self,
        phase: TeardownPhase,
        operation: impl FnOnce(&mut R) -> Result<(), Error>,
    ) -> core::result::Result<Self, PendingOwner<R>> {
        if let Err(source) = operation(&mut self.resource) {
            return Err(self.pending(phase, source));
        }
        Ok(self)
    }
}

/// Internal result from the shared transition machine.
pub(crate) enum ReleaseAttempt<R> {
    Released(ReleaseReceipt),
    Pending(PendingOwner<R>),
    Quarantined {
        owner: ReleaseOwner<R>,
        tombstone: TeardownTombstone,
    },
}

/// Non-usable retained owner after a pre-destructor failure.
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

/// Explicit heterogeneous teardown owner for one stream and its touched buffers.
///
/// Buffers remain ordinary owners until [`Self::begin_release`] consumes this
/// inventory. That transition disarms every leaf before synchronizing the
/// stream, so neither a failed synchronization nor a later partial release can
/// fall back to ordinary HIP `Drop` behavior.
pub struct TeardownInventory<T: BytePod> {
    stream: Stream,
    buffers: Vec<DeviceBuffer<T>>,
}

impl<T: BytePod> TeardownInventory<T> {
    /// Start an inventory with the stream that touched every registered buffer.
    #[must_use]
    pub fn new(stream: Stream) -> Self {
        Self {
            stream,
            buffers: Vec::new(),
        }
    }

    /// Retain a buffer whose work was submitted to this inventory's stream.
    pub fn push_buffer(&mut self, buffer: DeviceBuffer<T>) {
        self.buffers.push(buffer);
    }

    /// Consume, quiesce, then explicitly release every retained resource.
    #[must_use]
    pub fn begin_release(self) -> InventoryRelease<T> {
        let TeardownInventory { stream, buffers } = self;
        let buffers = buffers
            .into_iter()
            .enumerate()
            .map(|(entry, buffer)| buffer.into_release_owner(TeardownEntryId::new(entry)))
            .collect();
        if !stream.owns_explicit_handle() {
            return InventoryRelease::NotOwned(NonOwnedInventory { _buffers: buffers });
        }
        let stream = stream.into_release_owner(TeardownEntryId::new(usize::MAX));
        if let Err(error) = inventory_extent(&stream, &buffers) {
            return InventoryRelease::Pending(PendingInventory {
                state: PendingInventoryState::Admission {
                    stream: stream.pending(TeardownPhase::Preflight, error),
                    buffers,
                },
            });
        }
        match quiesce_stream_owner(stream) {
            StreamQuiesce::NotOwned => {
                InventoryRelease::NotOwned(NonOwnedInventory { _buffers: buffers })
            }
            StreamQuiesce::Quiescent(stream) => {
                resume_buffers(stream.owner, None, buffers, Vec::new())
            }
            StreamQuiesce::Pending(stream) => InventoryRelease::Pending(PendingInventory {
                state: PendingInventoryState::Synchronizing {
                    stream: stream.pending,
                    buffers,
                },
            }),
        }
    }
}

/// Successful logical acknowledgements from an inventory release.
#[derive(Debug)]
pub struct InventoryReceipt {
    released: Vec<ReleaseReceipt>,
}

impl InventoryReceipt {
    /// Per-resource HIP destructor acknowledgements in release order.
    #[must_use]
    pub fn released(&self) -> &[ReleaseReceipt] {
        &self.released
    }
}

/// Explicit inventory teardown outcome.
#[non_exhaustive]
pub enum InventoryRelease<T: BytePod> {
    /// The inventory was given the non-owned NULL stream; no destroy acknowledgement exists.
    NotOwned(NonOwnedInventory<T>),
    /// Every buffer and the quiescent stream acknowledged destruction.
    Released(InventoryReceipt),
    /// No uncertain resource is released automatically; retry is explicit.
    Pending(PendingInventory<T>),
    /// A destructor outcome was indeterminate; all unreleased leaves remain quarantined.
    Quarantined(InventoryQuarantine<T>),
}

/// Non-usable inventory retained after a known pre-destructor failure.
pub struct PendingInventory<T: BytePod> {
    state: PendingInventoryState<T>,
}

/// Opaque retained buffers from an inventory rejected for a non-owned stream.
pub struct NonOwnedInventory<T: BytePod> {
    _buffers: Vec<ReleaseOwner<DeviceBuffer<T>>>,
}

enum PendingInventoryState<T: BytePod> {
    Admission {
        stream: PendingOwner<Stream>,
        buffers: Vec<ReleaseOwner<DeviceBuffer<T>>>,
    },
    Synchronizing {
        stream: PendingOwner<Stream>,
        buffers: Vec<ReleaseOwner<DeviceBuffer<T>>>,
    },
    Releasing {
        stream: ReleaseOwner<Stream>,
        buffer: PendingOwner<DeviceBuffer<T>>,
        buffers: Vec<ReleaseOwner<DeviceBuffer<T>>>,
        receipts: Vec<ReleaseReceipt>,
    },
    Destroying {
        stream: PendingOwner<Stream>,
        receipts: Vec<ReleaseReceipt>,
    },
}

impl<T: BytePod> PendingInventory<T> {
    /// Retry the exact transition that last stopped.
    #[must_use]
    pub fn retry(self) -> InventoryRelease<T> {
        match self.state {
            PendingInventoryState::Admission { stream, buffers } => {
                let stream = stream.into_owner();
                if let Err(error) = inventory_extent(&stream, &buffers) {
                    return InventoryRelease::Pending(Self {
                        state: PendingInventoryState::Admission {
                            stream: stream.pending(TeardownPhase::Preflight, error),
                            buffers,
                        },
                    });
                }
                match quiesce_stream_owner(stream) {
                    StreamQuiesce::NotOwned => {
                        InventoryRelease::NotOwned(NonOwnedInventory { _buffers: buffers })
                    }
                    StreamQuiesce::Quiescent(stream) => {
                        resume_buffers(stream.owner, None, buffers, Vec::new())
                    }
                    StreamQuiesce::Pending(stream) => InventoryRelease::Pending(Self {
                        state: PendingInventoryState::Synchronizing {
                            stream: stream.pending,
                            buffers,
                        },
                    }),
                }
            }
            PendingInventoryState::Synchronizing { stream, buffers } => {
                match quiesce_stream_owner(stream.into_owner()) {
                    StreamQuiesce::NotOwned => {
                        InventoryRelease::NotOwned(NonOwnedInventory { _buffers: buffers })
                    }
                    StreamQuiesce::Quiescent(stream) => {
                        resume_buffers(stream.owner, None, buffers, Vec::new())
                    }
                    StreamQuiesce::Pending(stream) => InventoryRelease::Pending(Self {
                        state: PendingInventoryState::Synchronizing {
                            stream: stream.pending,
                            buffers,
                        },
                    }),
                }
            }
            PendingInventoryState::Releasing {
                stream,
                buffer,
                buffers,
                receipts,
            } => resume_buffers(stream, Some(buffer.into_owner()), buffers, receipts),
            PendingInventoryState::Destroying { stream, receipts } => {
                finish_inventory(stream.into_owner(), receipts)
            }
        }
    }
}

fn inventory_extent<T: BytePod>(
    stream: &ReleaseOwner<Stream>,
    buffers: &[ReleaseOwner<DeviceBuffer<T>>],
) -> core::result::Result<usize, Error> {
    buffers
        .iter()
        .try_fold(stream.requested_bytes(), |extent, buffer| {
            extent.checked_add(buffer.requested_bytes()).ok_or_else(|| {
                InternalSnafu {
                    message: "teardown inventory requested-byte extent overflow".to_string(),
                }
                .build()
            })
        })
}

/// Opaque conservative charge after a partial inventory destructor failure.
///
/// This tombstone intentionally has neither per-buffer access nor retry. It
/// retains all not-yet-acknowledged owners without dropping them, so an upper
/// executor must keep the reported requested-byte charge reserved.
pub struct InventoryQuarantine<T: BytePod> {
    requested_bytes: usize,
    tombstone: TeardownTombstone,
    _retained: InventoryHeld<T>,
}

impl<T: BytePod> InventoryQuarantine<T> {
    /// Conservative byte charge for the entire partially released inventory.
    #[must_use]
    pub const fn requested_bytes(&self) -> usize {
        self.requested_bytes
    }

    /// Indeterminate destructor failure that prevented further teardown.
    #[must_use]
    pub fn error(&self) -> &TeardownError {
        self.tombstone.error()
    }
}

struct InventoryHeld<T: BytePod> {
    _stream: Option<ReleaseOwner<Stream>>,
    _buffer: Option<ReleaseOwner<DeviceBuffer<T>>>,
    _buffers: Vec<ReleaseOwner<DeviceBuffer<T>>>,
}

fn resume_buffers<T: BytePod>(
    stream: ReleaseOwner<Stream>,
    current: Option<ReleaseOwner<DeviceBuffer<T>>>,
    mut buffers: Vec<ReleaseOwner<DeviceBuffer<T>>>,
    mut receipts: Vec<ReleaseReceipt>,
) -> InventoryRelease<T> {
    let mut current = current;
    loop {
        let Some(buffer) = current.take().or_else(|| buffers.pop()) else {
            return finish_inventory(stream, receipts);
        };
        match attempt_buffer_release(buffer) {
            ReleaseAttempt::Released(receipt) => receipts.push(receipt),
            ReleaseAttempt::Pending(buffer) => {
                return InventoryRelease::Pending(PendingInventory {
                    state: PendingInventoryState::Releasing {
                        stream,
                        buffer,
                        buffers,
                        receipts,
                    },
                });
            }
            ReleaseAttempt::Quarantined {
                owner: buffer,
                tombstone,
            } => {
                return InventoryRelease::Quarantined(quarantine_inventory(
                    stream,
                    Some(buffer),
                    buffers,
                    receipts,
                    tombstone,
                ));
            }
        }
    }
}

fn finish_inventory<T: BytePod>(
    stream: ReleaseOwner<Stream>,
    mut receipts: Vec<ReleaseReceipt>,
) -> InventoryRelease<T> {
    match attempt_stream_release(stream) {
        ReleaseAttempt::Released(receipt) => {
            receipts.push(receipt);
            InventoryRelease::Released(InventoryReceipt { released: receipts })
        }
        ReleaseAttempt::Pending(stream) => InventoryRelease::Pending(PendingInventory {
            state: PendingInventoryState::Destroying { stream, receipts },
        }),
        ReleaseAttempt::Quarantined {
            owner: stream,
            tombstone,
        } => InventoryRelease::Quarantined(quarantine_inventory(
            stream,
            None,
            Vec::new(),
            receipts,
            tombstone,
        )),
    }
}

fn quarantine_inventory<T: BytePod>(
    stream: ReleaseOwner<Stream>,
    buffer: Option<ReleaseOwner<DeviceBuffer<T>>>,
    buffers: Vec<ReleaseOwner<DeviceBuffer<T>>>,
    receipts: Vec<ReleaseReceipt>,
    tombstone: TeardownTombstone,
) -> InventoryQuarantine<T> {
    let requested_bytes = receipts
        .iter()
        .map(|receipt| receipt.resource().requested_bytes())
        .sum::<usize>()
        + stream.requested_bytes()
        + buffer.as_ref().map_or(0, ReleaseOwner::requested_bytes)
        + buffers
            .iter()
            .map(ReleaseOwner::requested_bytes)
            .sum::<usize>();
    InventoryQuarantine {
        requested_bytes,
        tombstone,
        _retained: InventoryHeld {
            _stream: Some(stream),
            _buffer: buffer,
            _buffers: buffers,
        },
    }
}

impl<R> ReleaseOwner<R> {
    pub(crate) fn pending(self, phase: TeardownPhase, source: Error) -> PendingOwner<R> {
        let error = match phase {
            TeardownPhase::Preflight => TeardownError::Preflight {
                resource: self.metadata.clone(),
                source,
                location: snafu::Location::caller(),
            },
            TeardownPhase::Synchronization => TeardownError::Synchronization {
                resource: self.metadata.clone(),
                source,
                location: snafu::Location::caller(),
            },
            TeardownPhase::Destructor => TeardownError::Destructor {
                resource: self.metadata.clone(),
                source,
                location: snafu::Location::caller(),
            },
        };
        PendingOwner { owner: self, error }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;
    use crate::device::Device;
    use crate::error::Error;

    fn owner(value: u8) -> ReleaseOwner<u8> {
        ReleaseOwner::new(
            value,
            ResourceMetadata::new(
                ResourceKind::Buffer,
                16,
                Device::for_test(0),
                TeardownEntryId::standalone(),
            ),
            |_| {},
        )
    }

    fn failure() -> Error {
        Error::runtime(1, "synthetic teardown")
    }

    struct RustMetadata {
        drops: Arc<AtomicUsize>,
    }

    impl Drop for RustMetadata {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn successful_transition_invokes_each_callback_once() {
        let mut prepares = 0;
        let mut destructors = 0;
        let outcome = owner(1).attempt(
            TeardownPhase::Preflight,
            |_| {
                prepares += 1;
                Ok(())
            },
            |_| {
                destructors += 1;
                Ok(())
            },
        );
        assert!(matches!(outcome, ReleaseAttempt::Released(_)));
        assert_eq!(prepares, 1);
        assert_eq!(destructors, 1);
    }

    #[test]
    fn acknowledged_release_disposes_rust_metadata_without_repeating_destructor() {
        let drops = Arc::new(AtomicUsize::new(0));
        let owner = ReleaseOwner::new(
            RustMetadata {
                drops: Arc::clone(&drops),
            },
            ResourceMetadata::new(
                ResourceKind::Buffer,
                16,
                Device::for_test(0),
                TeardownEntryId::standalone(),
            ),
            |_| {},
        );
        let outcome = owner.attempt(TeardownPhase::Preflight, |_| Ok(()), |_| Ok(()));
        assert!(matches!(outcome, ReleaseAttempt::Released(_)));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn preflight_failure_never_invokes_destructor_and_requires_explicit_retry() {
        let mut destructors = 0;
        let outcome = owner(1).attempt(
            TeardownPhase::Preflight,
            |_| Err(failure()),
            |_| {
                destructors += 1;
                Ok(())
            },
        );
        let ReleaseAttempt::Pending(pending) = outcome else {
            panic!("preflight failure must retain a pending owner");
        };
        assert_eq!(destructors, 0);
        drop(pending);
        assert_eq!(destructors, 0);
    }

    #[test]
    fn synchronization_failure_never_invokes_destructor() {
        let mut destructors = 0;
        let outcome = owner(1).attempt(
            TeardownPhase::Synchronization,
            |_| Err(failure()),
            |_| {
                destructors += 1;
                Ok(())
            },
        );
        let ReleaseAttempt::Pending(pending) = outcome else {
            panic!("synchronization failure must retain every resource");
        };
        drop(pending);
        assert_eq!(destructors, 0);
    }

    #[test]
    fn pending_retry_runs_the_destructor_exactly_once() {
        let pending =
            match owner(1).attempt(TeardownPhase::Preflight, |_| Err(failure()), |_| Ok(())) {
                ReleaseAttempt::Pending(pending) => pending,
                _ => panic!("preflight failure must create a pending transition"),
            };
        let mut destructors = 0;
        let outcome = pending.into_owner().attempt(
            TeardownPhase::Preflight,
            |_| Ok(()),
            |_| {
                destructors += 1;
                Ok(())
            },
        );
        assert!(matches!(outcome, ReleaseAttempt::Released(_)));
        assert_eq!(destructors, 1);
    }

    #[test]
    fn partial_batch_retains_the_unstarted_remainder() {
        let first = owner(1).attempt(TeardownPhase::Preflight, |_| Ok(()), |_| Ok(()));
        assert!(matches!(first, ReleaseAttempt::Released(_)));

        let mut destructors = 0;
        let remainder = owner(2).attempt(
            TeardownPhase::Preflight,
            |_| Err(failure()),
            |_| {
                destructors += 1;
                Ok(())
            },
        );
        let ReleaseAttempt::Pending(remainder) = remainder else {
            panic!("second resource must remain pending after its preflight failure");
        };
        drop(remainder);
        assert_eq!(destructors, 0);
    }

    #[test]
    fn destructor_failure_is_terminal_and_drop_does_not_retry() {
        let mut destructors = 0;
        let outcome = owner(1).attempt(
            TeardownPhase::Preflight,
            |_| Ok(()),
            |_| {
                destructors += 1;
                Err(failure())
            },
        );
        let ReleaseAttempt::Quarantined { tombstone, .. } = outcome else {
            panic!("destructor failure must quarantine the resource");
        };
        drop(tombstone);
        assert_eq!(destructors, 1);
    }
}
