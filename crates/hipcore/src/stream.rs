//! HIP streams and events.

use std::io::{self, Write};
use std::ptr;

use crate::device::Device;
use crate::error::{Result, check, hipError_t_code};
use crate::ffi;

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
    /// # Errors
    ///
    /// [`crate::Error::Runtime`] on HIP failure.
    pub fn new(device: &Device) -> Result<Self> {
        device.make_current()?;
        let mut handle: ffi::hipStream_t = ptr::null_mut();
        // SAFETY: FFI call; `&mut handle` valid.
        check(
            unsafe { ffi::hipStreamCreateWithFlags(&mut handle, ffi::hipStreamNonBlocking) },
            "hipStreamCreateWithFlags",
        )?;
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
