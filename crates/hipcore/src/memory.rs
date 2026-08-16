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

    /// Zero-fill the entire allocation via `hipMemset`.
    ///
    /// `hipMalloc` does not zero device memory — callers that need a
    /// genuinely zeroed buffer (as opposed to one about to be fully
    /// overwritten, e.g. by [`Self::copy_from_host`]) must call this
    /// explicitly.
    ///
    /// # Errors
    ///
    /// [`Error::Runtime`] on HIP failure.
    pub fn zero_fill(&mut self) -> Result<()> {
        self.device.make_current()?;
        // SAFETY: `ptr` is owned, sized `byte_len()` bytes, and the
        // device was just made current. `hipMemset` writes exactly
        // `byte_len()` bytes starting at `ptr`, matching the allocation.
        check(
            unsafe { ffi::hipMemset(self.ptr.as_ptr().cast::<c_void>(), 0, self.byte_len()) },
            "hipMemset",
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
    /// The caller must ensure `data` outlives the copy (normally via
    /// `stream.synchronize()` before `data` is dropped).
    ///
    /// # Errors
    ///
    /// As [`Self::copy_from_host`].
    pub fn copy_from_host_async(&mut self, data: &[T], stream: &Stream) -> Result<()> {
        if data.len() != self.len {
            return Err(Error::Internal(format!(
                "copy_from_host_async length mismatch: buffer {} vs slice {}",
                self.len,
                data.len()
            )));
        }
        self.device.make_current()?;
        // SAFETY: pointers valid for at least the duration of the
        // submission; caller is responsible for outliving the async op.
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
        )
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
