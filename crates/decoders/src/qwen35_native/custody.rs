//! Inert device-buffer custody used between native owners and HIP teardown.

use core::mem::ManuallyDrop;
use std::sync::{Arc, Mutex, MutexGuard};

use hipcore::{BufferAllocationError, Device, DeviceBuffer, StreamCreationError, TeardownBuffer};

/// Source failure from an ownership-preserving native construction transaction.
#[derive(Debug)]
#[non_exhaustive]
#[must_use = "a creation quarantine in this source must remain accounted"]
pub enum NativeBuildSource {
    /// A decoder validation, upload, or initialization operation failed.
    Decoder(Box<crate::Error>),
    /// A tracked device allocation did not produce an ordinary buffer owner.
    BufferAllocation(Box<BufferAllocationError>),
    /// A tracked stream creation did not produce an ordinary stream owner.
    StreamCreation(Box<StreamCreationError>),
}

impl NativeBuildSource {
    pub(super) fn decoder(error: crate::Error) -> Self {
        Self::Decoder(Box::new(error))
    }

    pub(super) fn buffer(error: BufferAllocationError) -> Self {
        Self::BufferAllocation(Box::new(error))
    }

    pub(super) fn stream(error: StreamCreationError) -> Self {
        Self::StreamCreation(Box::new(error))
    }
}

impl core::fmt::Display for NativeBuildSource {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Decoder(error) => error.fmt(formatter),
            Self::BufferAllocation(error) => error.fmt(formatter),
            Self::StreamCreation(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for NativeBuildSource {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Decoder(error) => Some(error.as_ref()),
            Self::BufferAllocation(error) => Some(error.as_ref()),
            Self::StreamCreation(error) => Some(error.as_ref()),
        }
    }
}

pub(super) type NativeBuildResult<T> = core::result::Result<T, NativeBuildSource>;

/// Consumes an original typed buffer only by disarming ordinary destruction.
///
/// Implementors retain the returned opaque owner until they can pass it to the
/// one real HIP teardown inventory. Dropping an accepted buffer owner performs
/// no HIP work, so a later admission failure cannot accidentally free memory
/// that an ordered native stream may still reference.
pub(super) trait NativeBufferSink {
    /// Disarm and retain one serialized-weight byte buffer.
    fn push_u8(&mut self, buffer: DeviceBuffer<u8>) {
        self.push_teardown_buffer(buffer.into_teardown());
    }

    /// Disarm and retain one f32 buffer.
    fn push_f32(&mut self, buffer: DeviceBuffer<f32>) {
        self.push_teardown_buffer(buffer.into_teardown());
    }

    /// Disarm and retain one u32 buffer.
    fn push_u32(&mut self, buffer: DeviceBuffer<u32>) {
        self.push_teardown_buffer(buffer.into_teardown());
    }

    /// Retain an already-disarmed heterogeneous buffer owner.
    fn push_teardown_buffer(&mut self, buffer: TeardownBuffer);
}

/// Inert heterogeneous native buffers before their stream is admitted to HIP
/// aggregate teardown.
///
/// This type deliberately owns no stream and performs no release. It is only
/// the lossless bridge from typed resource structs to `TeardownInventory`.
#[derive(Default)]
pub(super) struct NativeBufferParts {
    buffers: Vec<TeardownBuffer>,
}

impl NativeBufferParts {
    /// Start empty typed-buffer custody.
    pub(super) const fn new() -> Self {
        Self {
            buffers: Vec::new(),
        }
    }

    /// Consume the inert owners after all typed fields have been disarmed.
    pub(super) fn into_buffers(self) -> Vec<TeardownBuffer> {
        self.buffers
    }

    /// Whether this custody contains no disarmed native allocation.
    pub(super) fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }

    /// Merge already-disarmed ownership without invoking HIP.
    pub(super) fn append_to(self, destination: &mut Self) {
        destination.buffers.extend(self.buffers);
    }
}

impl NativeBufferSink for NativeBufferParts {
    fn push_teardown_buffer(&mut self, buffer: TeardownBuffer) {
        self.buffers.push(buffer);
    }
}

trait NativeBuildAccumulator: Default {
    fn append(&mut self, source: Self);
}

impl NativeBuildAccumulator for NativeBufferParts {
    fn append(&mut self, source: Self) {
        source.append_to(self);
    }
}

/// One recursive native construction scope with inert failure custody.
///
/// Every completed typed owner is armed in a [`NativeBuildGuard`] before the
/// next fallible operation. A failed constructor unwinds those guards into the
/// shared inert accumulator; the public failure is the sole remaining scope
/// owner before it can materialize real teardown parts.
pub(super) struct NativeBuildScope {
    buffers: Arc<Mutex<NativeBufferParts>>,
}

impl NativeBuildScope {
    /// Start one empty recursive construction scope.
    pub(super) fn new() -> Self {
        Self {
            buffers: Arc::new(Mutex::new(NativeBufferParts::new())),
        }
    }

    /// Arm one completed typed owner for failure-only disarming.
    pub(super) fn guard<T>(
        &self,
        value: T,
        retain: fn(T, &mut NativeBufferParts),
    ) -> NativeBuildGuard<T> {
        NativeBuildGuard {
            value: ManuallyDrop::new(value),
            armed: true,
            retain,
            buffers: Arc::clone(&self.buffers),
        }
    }

    /// Allocate and immediately arm one f32 buffer.
    pub(super) fn allocate_f32(
        &self,
        device: &Device,
        elements: usize,
    ) -> NativeBuildResult<NativeBuildGuard<DeviceBuffer<f32>>> {
        let buffer =
            DeviceBuffer::alloc_tracked(device, elements).map_err(NativeBuildSource::buffer)?;
        Ok(self.guard(buffer, |buffer, sink| sink.push_f32(buffer)))
    }

    /// Allocate and immediately arm one u32 buffer.
    pub(super) fn allocate_u32(
        &self,
        device: &Device,
        elements: usize,
    ) -> NativeBuildResult<NativeBuildGuard<DeviceBuffer<u32>>> {
        let buffer =
            DeviceBuffer::alloc_tracked(device, elements).map_err(NativeBuildSource::buffer)?;
        Ok(self.guard(buffer, |buffer, sink| sink.push_u32(buffer)))
    }

    /// Allocate and immediately arm one byte buffer.
    pub(super) fn allocate_u8(
        &self,
        device: &Device,
        elements: usize,
    ) -> NativeBuildResult<NativeBuildGuard<DeviceBuffer<u8>>> {
        let buffer =
            DeviceBuffer::alloc_tracked(device, elements).map_err(NativeBuildSource::buffer)?;
        Ok(self.guard(buffer, |buffer, sink| sink.push_u8(buffer)))
    }

    /// Disarm one rejected ownership carrier directly into this scope.
    pub(super) fn retain<T>(&self, value: T, retain: fn(T, &mut NativeBufferParts)) {
        retain_outside_lock(&self.buffers, value, retain);
    }

    /// Materialize custody only when no typed construction guard remains.
    pub(super) fn try_into_parts(self) -> core::result::Result<NativeBufferParts, Self> {
        match Arc::try_unwrap(self.buffers) {
            Ok(buffers) => Ok(match buffers.into_inner() {
                Ok(parts) => parts,
                Err(poisoned) => poisoned.into_inner(),
            }),
            Err(buffers) => Err(Self { buffers }),
        }
    }
}

/// A completed typed owner that can only commit or become inert custody.
pub(super) struct NativeBuildGuard<T> {
    value: ManuallyDrop<T>,
    armed: bool,
    retain: fn(T, &mut NativeBufferParts),
    buffers: Arc<Mutex<NativeBufferParts>>,
}

impl<T> NativeBuildGuard<T> {
    /// Commit the exact typed owner by one infallible move.
    pub(super) fn commit(mut self) -> T {
        self.armed = false;
        // SAFETY: `armed` was true on construction and only this consuming
        // method or `Drop` can take the value. Clearing it prevents a second take.
        unsafe { ManuallyDrop::take(&mut self.value) }
    }
}

impl<T> core::ops::Deref for NativeBuildGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> core::ops::DerefMut for NativeBuildGuard<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<T> Drop for NativeBuildGuard<T> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        self.armed = false;
        // SAFETY: this armed guard still owns the only live `T`; clearing
        // `armed` before the take prevents any second extraction.
        let value = unsafe { ManuallyDrop::take(&mut self.value) };
        retain_outside_lock(&self.buffers, value, self.retain);
    }
}

fn retain_outside_lock<T>(
    shared: &Arc<Mutex<NativeBufferParts>>,
    value: T,
    retain: fn(T, &mut NativeBufferParts),
) {
    let mut captured = NativeBufferParts::new();
    retain(value, &mut captured);
    lock_parts(shared).append(captured);
}

fn lock_parts(shared: &Arc<Mutex<NativeBufferParts>>) -> MutexGuard<'_, NativeBufferParts> {
    match shared.lock() {
        Ok(parts) => parts,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use super::{NativeBufferParts, NativeBuildScope};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FakeOwner {
        ordinary_drops: Arc<AtomicUsize>,
        retained: Arc<AtomicUsize>,
    }

    impl Drop for FakeOwner {
        fn drop(&mut self) {
            self.ordinary_drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn retain_fake(owner: FakeOwner, _: &mut NativeBufferParts) {
        owner.retained.fetch_add(1, Ordering::SeqCst);
        core::mem::forget(owner);
    }

    fn fake_owner(ordinary_drops: &Arc<AtomicUsize>, retained: &Arc<AtomicUsize>) -> FakeOwner {
        FakeOwner {
            ordinary_drops: Arc::clone(ordinary_drops),
            retained: Arc::clone(retained),
        }
    }

    #[test]
    fn failed_scope_disarms_every_completed_owner_once() -> Result<(), String> {
        let ordinary_drops = Arc::new(AtomicUsize::new(0));
        let retained = Arc::new(AtomicUsize::new(0));
        let scope = NativeBuildScope::new();
        let first = scope.guard(fake_owner(&ordinary_drops, &retained), retain_fake);
        let second = scope.guard(fake_owner(&ordinary_drops, &retained), retain_fake);
        drop(first);
        drop(second);
        let _parts = scope
            .try_into_parts()
            .map_err(|_| "guards remained after explicit drops".to_string())?;
        assert_eq!(ordinary_drops.load(Ordering::SeqCst), 0);
        assert_eq!(retained.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[test]
    fn live_guard_prevents_custody_detachment() -> Result<(), String> {
        let ordinary_drops = Arc::new(AtomicUsize::new(0));
        let retained = Arc::new(AtomicUsize::new(0));
        let scope = NativeBuildScope::new();
        let owner = scope.guard(fake_owner(&ordinary_drops, &retained), retain_fake);
        let scope = scope
            .try_into_parts()
            .err()
            .ok_or_else(|| "live guard allowed custody detachment".to_string())?;
        drop(owner);
        let _parts = scope
            .try_into_parts()
            .map_err(|_| "dropped guard still retained a scope clone".to_string())?;
        assert_eq!(ordinary_drops.load(Ordering::SeqCst), 0);
        assert_eq!(retained.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn committed_owner_never_enters_failure_custody() -> Result<(), String> {
        let ordinary_drops = Arc::new(AtomicUsize::new(0));
        let retained = Arc::new(AtomicUsize::new(0));
        let scope = NativeBuildScope::new();
        let owner = scope.guard(fake_owner(&ordinary_drops, &retained), retain_fake);
        let committed = owner.commit();
        let _parts = scope
            .try_into_parts()
            .map_err(|_| "committed owner retained a scope clone".to_string())?;
        assert_eq!(retained.load(Ordering::SeqCst), 0);
        drop(committed);
        assert_eq!(ordinary_drops.load(Ordering::SeqCst), 1);
        Ok(())
    }
}
