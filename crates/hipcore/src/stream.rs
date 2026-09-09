//! HIP streams and events.

use core::mem::ManuallyDrop;
use std::fmt;
use std::io::{self, Write};
use std::ptr;

use crate::creation::{
    CreationAttempt, CreationPort, CreationResolution, attempt_creation, resolve_creation,
};
use crate::device::Device;
use crate::error::{Error, InternalSnafu, Result, check, hipError_t_code};
use crate::ffi;
use crate::teardown::{
    PendingOwner, QuiesceAttempt, ReleaseAttempt, ReleaseOwner, ReleaseReceipt, ResourceKind,
    ResourceMetadata, TeardownEntryId, TeardownError, TeardownPhase, TeardownTombstone,
};

/// Handle to a HIP stream.
///
/// Streams are created non-blocking with `hipStreamNonBlocking`; the
/// default (NULL) stream serialises against every other stream and is
/// exposed through [`Stream::null`] only for ergonomic fallback code.
pub struct Stream {
    handle: ffi::hipStream_t,
    device: Device,
    owns_handle: bool,
}

// SAFETY: `hipStream_t` is an opaque pointer; the HIP runtime
// supports submitting work from any thread as long as the device is
// current. `Send` is sufficient; we deliberately do not implement
// `Sync` (concurrent use from two threads is undefined per HIP docs).
unsafe impl Send for Stream {}

impl Stream {
    /// Create a new non-blocking stream on `device`.
    ///
    /// This compatibility entry point returns the historical [`crate::Result`]
    /// shape. It cannot return custody when a failed stream creation call
    /// anomalously writes a non-null output handle. Construction transactions
    /// that require truthful native ownership must use [`Self::new_tracked`].
    ///
    /// # Errors
    ///
    /// [`crate::Error::Runtime`] on HIP failure.
    pub fn new(device: &Device) -> Result<Self> {
        match resolve_creation(
            stream_creation_attempt(device),
            |handle| Self::from_created_handle(device, handle),
            |_, error| error,
        ) {
            CreationResolution::Created(stream) => Ok(stream),
            CreationResolution::NoHandle(error) | CreationResolution::Quarantined(error) => {
                Err(error)
            }
        }
    }

    /// Create a non-blocking stream with explicit output-handle custody.
    ///
    /// A preflight failure or HIP error whose output slot remains null returns
    /// [`StreamCreationError::NoHandle`]. A HIP error accompanied by a non-null
    /// output returns an opaque terminal [`StreamCreationQuarantine`]; the
    /// handle is never exposed, destroyed, or retried because HIP did not
    /// establish that it denotes an owned stream.
    ///
    /// # Errors
    ///
    /// Returns [`StreamCreationError`] for wrapper rejection or HIP failure.
    pub fn new_tracked(device: &Device) -> core::result::Result<Self, StreamCreationError> {
        match resolve_creation(
            stream_creation_attempt(device),
            |handle| Self::from_created_handle(device, handle),
            |handle, error| StreamCreationQuarantine {
                error,
                resource: ResourceMetadata::new(
                    ResourceKind::Stream,
                    0,
                    device.clone(),
                    TeardownEntryId::Standalone,
                ),
                _handle: StreamTeardownHandle { handle },
            },
        ) {
            CreationResolution::Created(stream) => Ok(stream),
            CreationResolution::NoHandle(error) => Err(StreamCreationError::NoHandle(error)),
            CreationResolution::Quarantined(quarantine) => {
                Err(StreamCreationError::Quarantined(quarantine))
            }
        }
    }

    fn from_created_handle(device: &Device, handle: ffi::hipStream_t) -> Result<Self> {
        if handle.is_null() {
            return InternalSnafu {
                message: "hipStreamCreateWithFlags returned success with null handle",
            }
            .fail();
        }
        Ok(Self {
            handle,
            device: device.clone(),
            owns_handle: true,
        })
    }

    /// The NULL (default) stream on `device`. Inexpensive to create;
    /// does not own a handle (no destroy on drop).
    #[must_use]
    pub fn null(device: &Device) -> Self {
        Self {
            handle: ptr::null_mut(),
            device: device.clone(),
            owns_handle: false,
        }
    }

    /// Raw stream handle for passing to FFI.
    #[must_use]
    pub fn raw(&self) -> ffi::hipStream_t {
        self.handle
    }

    /// Device this stream belongs to.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Make this stream's owning device current on the calling thread.
    ///
    /// Kernel launch adapters must call this immediately before passing
    /// [`Self::raw`] to HIP. This is required even for an explicit stream and
    /// is essential for a NULL stream, whose device is otherwise only the
    /// calling thread's ambient current device.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Runtime`] on HIP failure.
    pub fn make_current(&self) -> Result<()> {
        self.device.make_current()
    }

    /// Block the calling thread until all work on this stream completes.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Runtime`] on HIP failure.
    pub fn synchronize(&self) -> Result<()> {
        self.make_current()?;
        // SAFETY: FFI call; handle is valid (null is accepted by HIP
        // to mean the default stream).
        check(
            unsafe { ffi::hipStreamSynchronize(self.handle) },
            "hipStreamSynchronize",
        )
    }

    /// Consume this stream and prove that its queued work has completed.
    ///
    /// HIP documents that `hipStreamDestroy` may destroy a queue while work is
    /// still inflight. Explicit teardown therefore has no direct stream
    /// destructor from this state: callers must first receive a
    /// [`QuiescentStream`], then destroy it after every buffer touched by the
    /// stream has been released.
    pub fn begin_quiesce(self) -> StreamQuiesce {
        match self.into_release_owner(TeardownEntryId::Standalone) {
            Ok(owner) => map_stream_quiesce(attempt_stream_quiesce(owner)),
            Err(stream) => StreamQuiesce::NotOwned(stream),
        }
    }

    /// Queue `event` on this stream.
    ///
    /// # Errors
    ///
    /// [`crate::Error::DeviceMismatch`] if `event` belongs to another
    /// device, or [`crate::Error::Runtime`] on HIP failure.
    pub fn record(&self, event: &Event) -> Result<()> {
        self.device
            .ensure_same_device(&event.device, "hipEventRecord")?;
        self.make_current()?;
        // SAFETY: FFI call; handles validated at construction.
        check(
            unsafe { ffi::hipEventRecord(event.handle, self.handle) },
            "hipEventRecord",
        )
    }

    pub(crate) fn into_release_owner(
        self,
        entry: TeardownEntryId,
    ) -> core::result::Result<ReleaseOwner<StreamTeardownHandle>, NonOwnedStream> {
        if !self.owns_handle || self.handle.is_null() {
            return Err(NonOwnedStream { stream: self });
        }
        let stream = ManuallyDrop::new(self);
        let handle = stream.handle;
        // SAFETY: `stream` is never ordinarily dropped. Moving its sole
        // Rust-owned field into metadata retires it exactly once while the
        // copied opaque handle becomes inert teardown state.
        let device = unsafe { core::ptr::read(&stream.device) };
        let metadata = ResourceMetadata::new(ResourceKind::Stream, 0, device, entry);
        Ok(ReleaseOwner::new(StreamTeardownHandle { handle }, metadata))
    }
}

struct StreamCreationPort<'a> {
    device: &'a Device,
}

impl CreationPort for StreamCreationPort<'_> {
    type Handle = ffi::hipStream_t;

    fn null_handle(&self) -> Self::Handle {
        ptr::null_mut()
    }

    fn is_null(&self, handle: Self::Handle) -> bool {
        handle.is_null()
    }

    fn preflight(&mut self) -> Result<()> {
        self.device.make_current()
    }

    fn create(&mut self, output: &mut Self::Handle) -> Result<()> {
        // SAFETY: `output` is a valid writable stream-handle slot.
        check(
            unsafe { ffi::hipStreamCreateWithFlags(output, ffi::hipStreamNonBlocking) },
            "hipStreamCreateWithFlags",
        )
    }

    fn success_with_null_error(&self) -> Error {
        InternalSnafu {
            message: "hipStreamCreateWithFlags returned success with null handle",
        }
        .build()
    }
}

fn stream_creation_attempt(device: &Device) -> CreationAttempt<ffi::hipStream_t> {
    let mut port = StreamCreationPort { device };
    attempt_creation(&mut port)
}

/// Failure from an explicitly tracked stream creation attempt.
#[derive(Debug)]
#[non_exhaustive]
pub enum StreamCreationError {
    /// The call was rejected before creation or its output slot remained null.
    ///
    /// This state owns no native handle and makes no claim about runtime-side
    /// cleanup or resource reclamation.
    NoHandle(Error),
    /// HIP returned an error after writing a non-null, indeterminate output.
    Quarantined(StreamCreationQuarantine),
}

impl StreamCreationError {
    /// Underlying wrapper or HIP failure.
    #[must_use]
    pub fn error(&self) -> &Error {
        match self {
            Self::NoHandle(error) => error,
            Self::Quarantined(quarantine) => quarantine.error(),
        }
    }
}

impl fmt::Display for StreamCreationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoHandle(error) => fmt::Display::fmt(error, formatter),
            Self::Quarantined(quarantine) => write!(
                formatter,
                "HIP stream creation returned an indeterminate non-null output: {}",
                quarantine.error()
            ),
        }
    }
}

impl std::error::Error for StreamCreationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.error())
    }
}

/// Terminal custody for a non-null output from failed HIP stream creation.
///
/// The output is neither assumed valid nor passed to `hipStreamDestroy`. This
/// value exposes accounting facts and the creation error, but no handle, retry,
/// or release transition. Dropping it performs no HIP work.
#[must_use = "the indeterminate stream output must remain conservatively accounted"]
pub struct StreamCreationQuarantine {
    error: Error,
    resource: ResourceMetadata,
    _handle: StreamTeardownHandle,
}

impl fmt::Debug for StreamCreationQuarantine {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StreamCreationQuarantine")
            .field("error", &self.error)
            .field("resource", &self.resource)
            .finish_non_exhaustive()
    }
}

impl StreamCreationQuarantine {
    /// HIP failure returned with the indeterminate output.
    #[must_use]
    pub const fn error(&self) -> &Error {
        &self.error
    }

    /// Stream creation facts retained for conservative accounting.
    #[must_use]
    pub const fn resource(&self) -> &ResourceMetadata {
        &self.resource
    }
}

/// Original NULL stream rejected by owned-stream teardown admission.
///
/// No HIP call or ownership conversion occurs before this value is returned.
#[must_use = "the non-owned stream is returned intact"]
pub struct NonOwnedStream {
    stream: Stream,
}

impl NonOwnedStream {
    /// Recover the original non-owned stream.
    #[must_use]
    pub fn into_stream(self) -> Stream {
        self.stream
    }
}

/// Inert opaque stream handle used only by explicit teardown internals.
#[derive(Debug)]
pub(crate) struct StreamTeardownHandle {
    handle: ffi::hipStream_t,
}

// SAFETY: this is an inert, inaccessible copy of an owned HIP stream handle.
// Moving it between threads cannot submit work; controlled transitions select
// the recorded device before passing it back to HIP.
unsafe impl Send for StreamTeardownHandle {}

/// Outcome of consuming a stream for explicit synchronization.
#[non_exhaustive]
#[must_use = "quiescence outcomes retain the consumed stream"]
pub enum StreamQuiesce {
    /// This is the non-owned NULL stream, so no destroy acknowledgement exists.
    NotOwned(NonOwnedStream),
    /// Synchronization proved that submitted stream work completed.
    Quiescent(QuiescentStream),
    /// Device selection failed before synchronization was invoked.
    Pending(PendingStreamQuiesce),
    /// Synchronization returned non-success, so completion remains unproved.
    SynchronizationUnconfirmed(StreamSynchronizationUnconfirmed),
}

/// A stream proven quiescent but not yet destroyed.
#[must_use = "the quiescent stream still requires explicit destruction"]
pub struct QuiescentStream {
    pub(crate) owner: ReleaseOwner<StreamTeardownHandle>,
}

impl QuiescentStream {
    /// Explicitly destroy this already-quiescent stream.
    pub fn destroy(self) -> StreamRelease {
        release_stream_owner(self.owner)
    }
}

/// Non-usable stream retained after a failed quiescence transition.
#[must_use = "the retained stream must remain accounted or be explicitly retried"]
pub struct PendingStreamQuiesce {
    pub(crate) pending: PendingOwner<StreamTeardownHandle>,
}

impl PendingStreamQuiesce {
    /// Recorded pre-synchronization failure and resource facts.
    #[must_use]
    pub fn error(&self) -> &TeardownError {
        self.pending.error()
    }

    /// Retry the explicit quiescence transition.
    pub fn retry(self) -> StreamQuiesce {
        map_stream_quiesce(attempt_stream_quiesce(self.pending.into_owner()))
    }
}

/// Non-usable stream retained after synchronization failed to prove completion.
#[must_use = "completion remains unproved until explicitly reconciled"]
pub struct StreamSynchronizationUnconfirmed {
    pending: PendingOwner<StreamTeardownHandle>,
}

impl StreamSynchronizationUnconfirmed {
    /// Synchronization failure and resource facts.
    #[must_use]
    pub fn error(&self) -> &TeardownError {
        self.pending.error()
    }

    /// Deliberately issue another non-destructive synchronization attempt.
    pub fn reconcile(self) -> StreamQuiesce {
        map_stream_reconciliation(attempt_stream_quiesce(self.pending.into_owner()))
    }
}

/// Outcome of destroying a quiescent stream.
#[non_exhaustive]
#[must_use = "release outcomes carry native ownership or acknowledgement evidence"]
pub enum StreamRelease {
    /// HIP acknowledged `hipStreamDestroy` for the owned stream.
    Released(ReleaseReceipt),
    /// Device selection failed before `hipStreamDestroy` was invoked.
    Pending(PendingStreamDestroy),
    /// `hipStreamDestroy` returned non-success, leaving ownership indeterminate.
    Quarantined(StreamTeardownQuarantine),
}

/// Non-usable quiescent stream retained after a pre-destruction failure.
#[must_use = "the retained stream must remain accounted or be explicitly retried"]
pub struct PendingStreamDestroy {
    pending: PendingOwner<StreamTeardownHandle>,
}

impl PendingStreamDestroy {
    /// Recorded preflight failure and resource facts.
    #[must_use]
    pub fn error(&self) -> &TeardownError {
        self.pending.error()
    }

    /// Retry the explicit stream destruction.
    pub fn retry(self) -> StreamRelease {
        release_stream_owner(self.pending.into_owner())
    }
}

/// Opaque terminal stream record after an indeterminate destroy outcome.
pub struct StreamTeardownQuarantine {
    tombstone: TeardownTombstone,
    _owner: ReleaseOwner<StreamTeardownHandle>,
}

impl fmt::Debug for StreamTeardownQuarantine {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StreamTeardownQuarantine")
            .field("tombstone", &self.tombstone)
            .finish_non_exhaustive()
    }
}

impl StreamTeardownQuarantine {
    /// Indeterminate destructor failure retained for accounting.
    #[must_use]
    pub fn error(&self) -> &TeardownError {
        self.tombstone.error()
    }
}

pub(crate) fn attempt_stream_quiesce(
    owner: ReleaseOwner<StreamTeardownHandle>,
) -> QuiesceAttempt<StreamTeardownHandle> {
    let owner = match owner.prepare(TeardownPhase::Preflight, |_, metadata| {
        metadata.device().make_current()
    }) {
        Ok(owner) => owner,
        Err(pending) => return QuiesceAttempt::PreflightPending(*pending),
    };
    match owner.prepare(TeardownPhase::Synchronization, |stream, _| {
        // SAFETY: preflight selected the owner device and `handle` is retained
        // by the disarmed stream owner for the full synchronization call.
        check(
            unsafe { ffi::hipStreamSynchronize(stream.handle) },
            "hipStreamSynchronize",
        )
    }) {
        Ok(owner) => QuiesceAttempt::Quiescent(owner),
        Err(pending) => QuiesceAttempt::SynchronizationUnconfirmed(*pending),
    }
}

pub(crate) fn attempt_stream_destroy(
    owner: ReleaseOwner<StreamTeardownHandle>,
) -> ReleaseAttempt<StreamTeardownHandle> {
    owner.attempt(
        TeardownPhase::Preflight,
        |_, metadata| metadata.device().make_current(),
        |stream, _| {
            // SAFETY: admission accepts only owned, non-null handles; the
            // original Stream destructor was disarmed before this transition.
            check(
                unsafe { ffi::hipStreamDestroy(stream.handle) },
                "hipStreamDestroy",
            )
        },
    )
}

fn map_stream_quiesce(attempt: QuiesceAttempt<StreamTeardownHandle>) -> StreamQuiesce {
    match attempt {
        QuiesceAttempt::Quiescent(owner) => StreamQuiesce::Quiescent(QuiescentStream { owner }),
        QuiesceAttempt::PreflightPending(pending) => {
            StreamQuiesce::Pending(PendingStreamQuiesce { pending })
        }
        QuiesceAttempt::SynchronizationUnconfirmed(pending) => {
            StreamQuiesce::SynchronizationUnconfirmed(StreamSynchronizationUnconfirmed { pending })
        }
    }
}

fn map_stream_reconciliation(attempt: QuiesceAttempt<StreamTeardownHandle>) -> StreamQuiesce {
    match attempt {
        QuiesceAttempt::Quiescent(owner) => StreamQuiesce::Quiescent(QuiescentStream { owner }),
        QuiesceAttempt::PreflightPending(pending)
        | QuiesceAttempt::SynchronizationUnconfirmed(pending) => {
            StreamQuiesce::SynchronizationUnconfirmed(StreamSynchronizationUnconfirmed { pending })
        }
    }
}

fn release_stream_owner(owner: ReleaseOwner<StreamTeardownHandle>) -> StreamRelease {
    match attempt_stream_destroy(owner) {
        ReleaseAttempt::Released(receipt) => StreamRelease::Released(receipt),
        ReleaseAttempt::Pending(pending) => {
            StreamRelease::Pending(PendingStreamDestroy { pending })
        }
        ReleaseAttempt::Quarantined { owner, tombstone } => {
            StreamRelease::Quarantined(StreamTeardownQuarantine {
                tombstone,
                _owner: owner,
            })
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if self.owns_handle && !self.handle.is_null() {
            // WARNING: do not call `hipStreamDestroy` when `make_current`
            // fails — same reasoning as `DeviceBuffer::drop`: the destroy
            // targets whichever device is current on this thread, not
            // the device the stream was created on. Skip and leak
            // instead of acting against an unknown device context.
            if let Err(error) = self.device.make_current() {
                if writeln!(
                    io::stderr().lock(),
                    "hipcore: make_current before hipStreamDestroy failed: {error} — leaking stream handle"
                )
                .is_err()
                {
                    // Drop cannot surface secondary stderr failures.
                }
                return;
            }
            // SAFETY: handle owned by this wrapper and not yet freed;
            // the owning device was just made current above. Errors
            // during teardown are logged but cannot be returned from Drop.
            let status = unsafe { ffi::hipStreamDestroy(self.handle) };
            if status != ffi::hipError_t::hipSuccess
                && writeln!(
                    io::stderr().lock(),
                    "hipcore: hipStreamDestroy failed (code {}) — leaking stream handle",
                    hipError_t_code(status)
                )
                .is_err()
            {
                // Drop cannot surface secondary stderr failures.
            }
        }
    }
}

/// Handle to a HIP event, used for intra-stream timing and cross-stream
/// ordering.
pub struct Event {
    handle: ffi::hipEvent_t,
    device: Device,
    owns_handle: bool,
}

// SAFETY: `hipEvent_t` is an opaque pointer; HIP permits cross-thread
// recording / synchronisation once the device is current. `Event`
// retains the creating `Device` so `Drop` can restore its context
// before destroying the handle, however this value migrates threads.
unsafe impl Send for Event {}

impl Event {
    /// Create a default event (timing enabled).
    ///
    /// # Errors
    ///
    /// [`crate::Error::Runtime`] on HIP failure.
    pub fn new(device: &Device) -> Result<Self> {
        device.make_current()?;
        let mut handle: ffi::hipEvent_t = ptr::null_mut();
        // SAFETY: FFI call; `&mut handle` valid.
        check(
            unsafe { ffi::hipEventCreate(&mut handle) },
            "hipEventCreate",
        )?;
        Ok(Self {
            handle,
            device: device.clone(),
            owns_handle: true,
        })
    }

    /// Block until this event has fired.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Runtime`] on HIP failure.
    pub fn synchronize(&self) -> Result<()> {
        self.device.make_current()?;
        // SAFETY: FFI call; handle owned.
        check(
            unsafe { ffi::hipEventSynchronize(self.handle) },
            "hipEventSynchronize",
        )
    }

    /// Elapsed time in milliseconds between two recorded events.
    ///
    /// # Errors
    ///
    /// [`crate::Error::DeviceMismatch`] if the events belong to different
    /// devices, or [`crate::Error::Runtime`] on HIP failure.
    pub fn elapsed_ms(start: &Event, end: &Event) -> Result<f32> {
        start
            .device
            .ensure_same_device(&end.device, "hipEventElapsedTime")?;
        start.device.make_current()?;
        let mut ms: f32 = 0.0;
        // SAFETY: FFI call; handles owned, output pointer valid.
        check(
            unsafe { ffi::hipEventElapsedTime(&mut ms, start.handle, end.handle) },
            "hipEventElapsedTime",
        )?;
        Ok(ms)
    }

    /// Raw handle for FFI passthrough.
    #[must_use]
    pub fn raw(&self) -> ffi::hipEvent_t {
        self.handle
    }

    /// Device this event was created on.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if self.owns_handle && !self.handle.is_null() {
            // WARNING: do not call `hipEventDestroy` when `make_current`
            // fails — same reasoning as `DeviceBuffer::drop` and
            // `Stream::drop`: destroy targets whichever device is
            // current on this thread, not the device the event was
            // created on. `Event` is `Send`, so a handle created on one
            // device can be dropped on a thread where a different
            // device is current; skip and leak rather than destroying
            // against an unknown (possibly wrong) device context.
            if let Err(error) = self.device.make_current() {
                if writeln!(
                    io::stderr().lock(),
                    "hipcore: make_current before hipEventDestroy failed: {error} — leaking event handle"
                )
                .is_err()
                {
                    // Drop cannot surface secondary stderr failures.
                }
                return;
            }
            // SAFETY: handle owned and not yet destroyed; the owning
            // device was just made current above.
            let status = unsafe { ffi::hipEventDestroy(self.handle) };
            if status != ffi::hipError_t::hipSuccess
                && writeln!(
                    io::stderr().lock(),
                    "hipcore: hipEventDestroy failed (code {}) — leaking event handle",
                    hipError_t_code(status)
                )
                .is_err()
            {
                // Drop cannot surface secondary stderr failures.
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "test assertions use expect_err() directly"
    )]

    use super::*;

    fn borrowed_event(device: &Device) -> Event {
        Event {
            handle: ptr::null_mut(),
            device: device.clone(),
            owns_handle: false,
        }
    }

    #[test]
    fn gpu_boundary_pure_record_rejects_cross_device_event_before_ffi() {
        let stream = Stream::null(&Device::for_test(2));
        let event = borrowed_event(&Device::for_test(7));

        let error = stream
            .record(&event)
            .expect_err("cross-device event must be rejected");
        assert!(matches!(
            error,
            crate::Error::DeviceMismatch {
                op: "hipEventRecord",
                expected: 2,
                actual: 7,
                ..
            }
        ));
    }

    #[test]
    fn gpu_boundary_pure_elapsed_time_rejects_cross_device_events_before_ffi() {
        let start = borrowed_event(&Device::for_test(3));
        let end = borrowed_event(&Device::for_test(8));

        let error =
            Event::elapsed_ms(&start, &end).expect_err("cross-device events must be rejected");
        assert!(matches!(
            error,
            crate::Error::DeviceMismatch {
                op: "hipEventElapsedTime",
                expected: 3,
                actual: 8,
                ..
            }
        ));
    }
}
