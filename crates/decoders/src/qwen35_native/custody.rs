//! Typed device-buffer transfer seam for native construction and teardown.

use hipcore::DeviceBuffer;

/// Consumes original typed native buffer owners without allocating or releasing them.
///
/// The eventual teardown inventory implements this only after it has accepted
/// its quiescing stream. Until then, a construction transaction retains typed
/// resources and transfers them through this seam only on a known cleanup path.
pub(super) trait NativeBufferSink {
    /// Consume one serialized-weight byte buffer.
    fn push_u8(&mut self, buffer: DeviceBuffer<u8>);

    /// Consume one f32 device buffer.
    fn push_f32(&mut self, buffer: DeviceBuffer<f32>);

    /// Consume one u32 device buffer.
    fn push_u32(&mut self, buffer: DeviceBuffer<u32>);
}
