//! Device memory allocation and host↔device copy.

use core::marker::PhantomData;
use core::ptr::NonNull;
use std::ffi::c_void;
use std::io::{self, Write};

use crate::device::Device;
use crate::error::{Error, Result, check, hipError_t_code};
use crate::ffi;
use crate::pod::BytePod;
use crate::stream::Stream;

/// Owned allocation in device memory.
///
/// `T: BytePod` enforces that every bit pattern of `T` is a valid
/// inhabitant, so reads back from device memory cannot produce an
/// invalid `T`. `Drop` makes the owning device current, then calls
/// `hipFree`; a failure at either step is logged (via stderr) and
/// leaked rather than panicking, per library policy — `hipFree`
/// targets whichever device is current on the calling thread, not the
/// device the pointer was allocated on, so freeing after a failed
/// context switch would act against an unknown device.
pub struct DeviceBuffer<T: BytePod> {
    ptr: NonNull<T>,
    len: usize,
    device: Device,
    _marker: PhantomData<T>,
}

// SAFETY: a `DeviceBuffer` is a handle to device memory; moving it
// between threads is safe as long as the device is made current before
// use.
unsafe impl<T: BytePod + Send> Send for DeviceBuffer<T> {}

// SAFETY: `&DeviceBuffer` exposes only read-only access to the handle
// itself; mutation of device memory always goes through a `&mut` on
// this wrapper. Concurrent reads from multiple threads are safe; any
// cross-stream ordering is the caller's responsibility via events.
unsafe impl<T: BytePod + Sync> Sync for DeviceBuffer<T> {}

impl<T: BytePod> DeviceBuffer<T> {
    /// Allocate `len` elements of `T` on `device`.
    ///
    /// # Errors
    ///
    /// - [`Error::OutOfMemory`] when `hipMalloc` returns an
    ///   out-of-memory status.
    /// - [`Error::Runtime`] for any other HIP failure.
    pub fn alloc(device: &Device, len: usize) -> Result<Self> {
        device.make_current()?;
        let bytes = len
            .checked_mul(core::mem::size_of::<T>())
            .ok_or_else(|| Error::Internal("allocation size overflow".into()))?;
        let mut ptr: *mut c_void = core::ptr::null_mut();
        // SAFETY: FFI call; `&mut ptr` valid. HIP returns either a
        // non-null pointer with status success, or null with a failure
        // code; we validate both.
        let status = unsafe { ffi::hipMalloc(&mut ptr, bytes) };
        if status != ffi::hipError_t::hipSuccess {
            let free = device.memory_budget().map(|b| b.free).unwrap_or(0);
            return Err(if status == ffi::hipError_t::hipErrorOutOfMemory {
                Error::OutOfMemory {
                    requested: bytes,
                    free,
                }
            } else {
                Error::runtime(hipError_t_code(status), "hipMalloc")
            });
        }
        let nn = NonNull::new(ptr.cast::<T>())
            .ok_or_else(|| Error::Internal("hipMalloc returned success with null ptr".into()))?;
        Ok(Self {
            ptr: nn,
            len,
            device: device.clone(),
            _marker: PhantomData,
        })
    }

    /// Allocate and copy `data` to device memory.
    ///
    /// # Errors
    ///
    /// As [`Self::alloc`] plus [`Error::Runtime`] from the memcpy.
    pub fn from_host(device: &Device, data: &[T]) -> Result<Self> {
        let mut buf = Self::alloc(device, data.len())?;
        buf.copy_from_host(data)?;
        Ok(buf)
    }

    /// Number of `T` elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if the buffer holds zero elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Size in bytes.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.len * core::mem::size_of::<T>()
    }

    /// Raw device pointer (for passing to kernels).
    #[must_use]
    pub fn as_device_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Owning [`Device`].
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Host → device memcpy (synchronous).
    ///
    /// # Errors
    ///
    /// - [`Error::Internal`] if `data.len() != self.len()`.
    /// - [`Error::Runtime`] on HIP failure.
    pub(crate) fn copy_from_host(&mut self, data: &[T]) -> Result<()> {
        if data.len() != self.len {
            return Err(Error::Internal(format!(
                "copy_from_host length mismatch: buffer {} vs slice {}",
                self.len,
                data.len()
            )));
        }
        self.device.make_current()?;
        // SAFETY: destination pointer is owned and sized; source slice is valid.
        check(
            unsafe {
                ffi::hipMemcpy(
                    self.ptr.as_ptr().cast::<c_void>(),
                    data.as_ptr().cast::<c_void>(),
                    self.byte_len(),
                    ffi::hipMemcpyKind::hipMemcpyHostToDevice,
                )
            },
            "hipMemcpy(HtoD)",
        )
    }

    /// Device → host memcpy (synchronous).
    ///
    /// # Errors
    ///
    /// - [`Error::Internal`] if `dst.len() != self.len()`.
    /// - [`Error::Runtime`] on HIP failure.
    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<()> {
        if dst.len() != self.len {
            return Err(Error::Internal(format!(
                "copy_to_host length mismatch: buffer {} vs slice {}",
                self.len,
                dst.len()
            )));
        }
        self.device.make_current()?;
        // SAFETY: destination slice is valid; source pointer owned.
        check(
            unsafe {
                ffi::hipMemcpy(
                    dst.as_mut_ptr().cast::<c_void>(),
                    self.ptr.as_ptr().cast::<c_void>(),
                    self.byte_len(),
                    ffi::hipMemcpyKind::hipMemcpyDeviceToHost,
                )
            },
            "hipMemcpy(DtoH)",
        )
    }

    /// Async host → device memcpy on `stream`.
    ///
    /// Takes ownership of `data`, returning a [`PendingCopy`] that holds
    /// it until the copy completes. A safe caller cannot free or mutate
    /// the source buffer while the GPU DMA may still be reading it:
    /// `data` is moved into this call, so any attempt to reuse the
    /// caller's original binding afterward is a compile-time "use of
    /// moved value" error. Even `std::mem::forget`-ing the returned
    /// [`PendingCopy`] only leaks the owned buffer (safe) — it can
    /// never free memory the DMA is still reading (unsound), which is
    /// the failure mode a borrow-plus-`Drop`-guard design would not
    /// have closed.
    ///
    /// Call [`PendingCopy::wait`] to block until the copy finishes and
    /// reclaim `data`, or simply drop the value — its `Drop` blocks on
    /// the stream for you.
    ///
    /// # Errors
    ///
    /// As [`Self::copy_from_host`].
    pub fn copy_from_host_async<'s>(
        &mut self,
        data: Vec<T>,
        stream: &'s Stream,
    ) -> Result<PendingCopy<'s, T>> {
        if data.len() != self.len {
            return Err(Error::Internal(format!(
                "copy_from_host_async length mismatch: buffer {} vs slice {}",
                self.len,
                data.len()
            )));
        }
        self.device.make_current()?;
        // SAFETY: destination pointer is owned and sized. `data` is
        // moved into the `PendingCopy` this returns on success, which
        // keeps the allocation (and therefore the source pointer)
        // alive until the caller synchronizes with `stream` via
        // `PendingCopy::wait` or `Drop` — so the pointer stays valid
        // for the full duration of the async copy regardless of what
        // the caller does with the return value.
        check(
            unsafe {
                ffi::hipMemcpyAsync(
                    self.ptr.as_ptr().cast::<c_void>(),
                    data.as_ptr().cast::<c_void>(),
                    self.byte_len(),
                    ffi::hipMemcpyKind::hipMemcpyHostToDevice,
                    stream.raw(),
                )
            },
            "hipMemcpyAsync(HtoD)",
        )?;
        Ok(PendingCopy {
            data,
            stream,
            synced: false,
        })
    }
}

/// A host → device copy still in flight on a [`Stream`].
///
/// Returned by [`DeviceBuffer::copy_from_host_async`]. Owns the source
/// host buffer until the copy completes, so a safe caller cannot free
/// or mutate it while the GPU DMA may still be reading it — see that
/// method's docs for why ownership (rather than a borrow) is what
/// makes this sound even against `std::mem::forget`.
#[must_use = "dropping this immediately still blocks in `Drop` to keep the copy \
              sound, but discarding it early gives up the chance to overlap host \
              work with the copy — call `wait()` to reclaim `data` explicitly"]
pub struct PendingCopy<'s, T: BytePod> {
    data: Vec<T>,
    stream: &'s Stream,
    synced: bool,
}

impl<T: BytePod> PendingCopy<'_, T> {
    /// Block until the copy completes, returning the host buffer.
    ///
    /// # Errors
    ///
    /// [`Error::Runtime`] on HIP failure.
    pub fn wait(mut self) -> Result<Vec<T>> {
        self.synced = true;
        self.stream.synchronize()?;
        Ok(core::mem::take(&mut self.data))
    }
}

impl<T: BytePod> Drop for PendingCopy<'_, T> {
    fn drop(&mut self) {
        // WARNING: this is the safety net for a caller that never calls
        // `wait()`. It must block on the stream, not skip the sync — if
        // it returned without synchronizing, `self.data` would drop
        // (freeing the host buffer) immediately afterward as part of
        // this same `Drop`, while the DMA reading it could still be in
        // flight. Best-effort like the sibling `Drop` impls below: a
        // synchronize failure is logged, not retried or panicked on,
        // since `Drop` cannot propagate an error.
        if self.synced {
            return;
        }
        if let Err(error) = self.stream.synchronize() {
            if writeln!(
                io::stderr().lock(),
                "hipcore: stream synchronize before PendingCopy drop failed: {error} — \
                 the host buffer may still be read by an in-flight DMA"
            )
            .is_err()
            {
                // Drop cannot surface secondary stderr failures.
            }
        }
    }
}

impl<T: BytePod> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        // WARNING: do not call `hipFree` when `make_current` fails. HIP
        // frees against whichever device is current on this thread, not
        // the device `ptr` was allocated on; proceeding here would free
        // against an unknown (possibly wrong) device context. Skip and
        // leak instead — consistent with the log-and-leak policy below.
        if let Err(error) = self.device.make_current() {
            if writeln!(
                io::stderr().lock(),
                "hipcore: make_current before hipFree failed: {error} — leaking buffer"
            )
            .is_err()
            {
                // Drop cannot surface secondary stderr failures.
            }
            return;
        }
        // SAFETY: `ptr` was returned by `hipMalloc` and is owned here;
        // the owning device was just made current above.
        let status = unsafe { ffi::hipFree(self.ptr.as_ptr().cast::<c_void>()) };
        if status != ffi::hipError_t::hipSuccess
            && writeln!(
                io::stderr().lock(),
                "hipcore: hipFree failed (code {}) — leaking buffer",
                hipError_t_code(status)
            )
            .is_err()
        {
            // Drop cannot surface secondary stderr failures.
        }
    }
}
