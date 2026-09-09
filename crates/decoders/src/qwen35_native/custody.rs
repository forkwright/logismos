//! Inert device-buffer custody used between native owners and HIP teardown.

use hipcore::{DeviceBuffer, TeardownBuffer};

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
