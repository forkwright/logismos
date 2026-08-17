//! Device memory allocation and host↔device copy.

use core::marker::PhantomData;
use core::mem::ManuallyDrop;
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
    /// Takes ownership of both `self` (the destination) and `data` (the
    /// source), returning a [`PendingCopy`] that holds them both until
    /// the copy completes. A safe caller cannot free or mutate EITHER
    /// buffer while the GPU DMA may still be touching it: `data` is
    /// moved into this call exactly as it always was, and now `self` is
    /// too, so any attempt to reuse the caller's original `DeviceBuffer`
    /// binding afterward is also a compile-time "use of moved value"
    /// error — the same guarantee this already gave the source side,
    /// extended to the destination (forkwright/logismos#104: the
    /// original signature borrowed `self` only for the duration of this
    /// call, so a caller could `drop` the destination the instant this
    /// returned, while the queued DMA was still writing into it). Even
    /// `std::mem::forget`-ing the returned [`PendingCopy`] only leaks
    /// the owned destination (safe) — it can never free memory the DMA
    /// is still writing (unsound). A borrow-plus-`Drop` guard would not
    /// have closed that: `mem::forget`-ing a guard that only *borrows*
    /// its protected value lets NLL end the borrow at the `forget` call
    /// itself, so the caller could still free the borrowed buffer right
    /// after — owning it is what makes `forget` merely leak rather than
    /// un-protect, on both sides of this copy.
    ///
    /// If `self`'s length does not match `data`'s, or the underlying HIP
    /// call fails before anything is queued, `self` and `data` are
    /// simply dropped as part of returning `Err` — no DMA has been
    /// queued onto `stream` in either failure case, so there is nothing
    /// in flight to protect against.
    ///
    /// Call [`PendingCopy::wait`] to block until the copy finishes and
    /// reclaim both the destination buffer and `data`, or simply drop
    /// the value — its `Drop` blocks on the stream for you and then
    /// releases both.
    ///
    /// # Errors
    ///
    /// As [`Self::copy_from_host`].
    pub fn copy_from_host_async(self, data: Vec<T>, stream: &Stream) -> Result<PendingCopy<'_, T>> {
        if data.len() != self.len {
            return Err(Error::Internal(format!(
                "copy_from_host_async length mismatch: buffer {} vs slice {}",
                self.len,
                data.len()
            )));
        }
        self.device.make_current()?;
        // SAFETY: destination pointer is owned and sized. `self` and
        // `data` are both moved into the `PendingCopy` this returns on
        // success, which keeps the destination allocation and the
        // source allocation alive until the caller synchronizes with
        // `stream` via `PendingCopy::wait` or `Drop` — so both pointers
        // stay valid for the full duration of the async copy regardless
        // of what the caller does with the return value.
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
            buf: ManuallyDrop::new(self),
            stream,
            synced: false,
        })
    }
}

/// A host → device copy still in flight on a [`Stream`].
///
/// Returned by [`DeviceBuffer::copy_from_host_async`]. Owns BOTH the
/// destination [`DeviceBuffer`] and the source host buffer until the
/// copy completes, so a safe caller cannot free or mutate either one
/// while the GPU DMA may still be touching it — see that method's docs
/// for why ownership (rather than a borrow) is what makes this sound
/// even against `std::mem::forget`, on both sides
/// (forkwright/logismos#25 source, forkwright/logismos#104 destination).
#[must_use = "dropping this immediately still blocks in `Drop` to keep the copy \
              sound, but discarding it early gives up the chance to overlap host \
              work with the copy — call `wait()` to reclaim the destination buffer \
              and `data` explicitly"]
pub struct PendingCopy<'s, T: BytePod> {
    data: Vec<T>,
    // INVARIANT: holds its `DeviceBuffer` for the entire life of a
    // `PendingCopy`, except for the single instant inside `wait` between
    // `ManuallyDrop::take` and that value's return. `Drop` below decides
    // explicitly — based on `synced` and the outcome of a fresh
    // `stream.synchronize()` — whether to actually run the wrapped
    // buffer's destructor (safe once sync confirms the DMA is done) or
    // leave it untouched (leak, not free, when sync could not confirm
    // that). A plain `DeviceBuffer<T>` field here would free it
    // unconditionally via ordinary field-drop-glue regardless of that
    // outcome, reopening forkwright/logismos#104.
    buf: ManuallyDrop<DeviceBuffer<T>>,
    stream: &'s Stream,
    synced: bool,
}

impl<T: BytePod> PendingCopy<'_, T> {
    /// Block until the copy completes, returning the destination buffer
    /// and the host buffer.
    ///
    /// # Errors
    ///
    /// [`Error::Runtime`] on HIP failure. On error, neither buffer is
    /// released here — `self` drops immediately afterward and its
    /// `Drop` impl leaks both rather than freeing either, for the same
    /// reason a bare `drop(pending)` does: a failed synchronize does not
    /// prove the queued copy has stopped touching them.
    pub fn wait(mut self) -> Result<(DeviceBuffer<T>, Vec<T>)> {
        self.stream.synchronize()?;
        // WARNING: this must be set only AFTER `synchronize` succeeds,
        // not before it. Setting it unconditionally up front (as this
        // used to, prior to forkwright/logismos#104) would make `Drop`
        // return early via the `if self.synced` check below without
        // ever attempting its own synchronize-or-leak fallback — so on
        // the `?` above returning `Err`, ordinary field-drop-glue would
        // free `self.data` (and, now, `self.buf`) unconditionally right
        // after this function returns, on `wait`'s error path, despite
        // synchronize having just failed. That reopens exactly the
        // use-after-free this type exists to prevent, one call frame up
        // from the `Drop` path #25/#104's own reproductions target.
        self.synced = true;
        // SAFETY: the synchronize above just succeeded, so the DMA this
        // copy queued is done touching `buf` (write) and `data` (read)
        // — safe to hand real ownership of `buf` back to the caller.
        // `self` drops when this function returns (it was taken by
        // value); `Drop` checks `self.synced` (just set `true`) first
        // and returns without touching `buf` again, so this is the only
        // place `buf` is ever taken out of its `ManuallyDrop`.
        let buf = unsafe { ManuallyDrop::take(&mut self.buf) };
        Ok((buf, core::mem::take(&mut self.data)))
    }
}

impl<T: BytePod> Drop for PendingCopy<'_, T> {
    fn drop(&mut self) {
        // WARNING: this is the safety net for a caller that never calls
        // `wait()` (and the fallback for `wait()`'s own error path — see
        // the WARNING there). It must block on the stream, not skip the
        // sync — if it returned without synchronizing, `self.data` and
        // the wrapped destination buffer would drop (freeing both)
        // immediately afterward as part of this same `Drop`, while the
        // DMA touching them could still be in flight.
        if self.synced {
            return;
        }
        match self.stream.synchronize() {
            Ok(()) => {
                // SAFETY: the synchronize immediately above succeeded,
                // so the DMA is done writing into `buf` — safe to
                // actually run its destructor (`hipFree`) now.
                // `self.data` is left alone here; falling through lets
                // ordinary field-drop-glue free it right after this
                // function returns, which is sound for the same reason.
                unsafe { ManuallyDrop::drop(&mut self.buf) };
            }
            Err(error) => {
                if writeln!(
                    io::stderr().lock(),
                    "hipcore: stream synchronize before PendingCopy drop failed: {error} — \
                     leaking the destination buffer and the host buffer, not freeing \
                     either, because an in-flight DMA may still be touching them"
                )
                .is_err()
                {
                    // Drop cannot surface secondary stderr failures.
                }
                // WARNING: a failed synchronize does not prove the queued
                // hipMemcpyAsync has stopped touching `buf` or `data` — it
                // only proves we could not confirm either way. Freeing
                // either here would be the exact use-after-free #25
                // (source) / #104 (destination) exist to prevent, now on
                // the error path instead of the caller's. Skip-and-leak
                // both instead, matching the policy `DeviceBuffer::drop` /
                // `Stream::drop` / `Event::drop` use for the same "acting
                // under uncertainty is unsound" case:
                // - `data`: `mem::take` swaps in an empty, non-allocating
                //   `Vec` so the drop glue that still runs after this
                //   function returns is a no-op, and `mem::forget`
                //   discards the real one without running its destructor.
                //   `T: BytePod: Copy`, so `Vec<T>` has no destructor
                //   beyond freeing its buffer — nothing else is leaked.
                // - `buf`: simply never calling `ManuallyDrop::drop` on it
                //   means its destructor (`hipFree`) never runs at all —
                //   `ManuallyDrop`'s own `Drop` impl is a no-op, so
                //   leaving it untouched leaks the device allocation
                //   (safe) instead of racing the in-flight DMA by freeing
                //   it (unsound).
                core::mem::forget(core::mem::take(&mut self.data));
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

#[cfg(test)]
mod tests {
    //! Negative-case fixtures for PR #101 review finding 1 and
    //! forkwright/logismos#104 (`PendingCopy::drop` / `PendingCopy::wait`
    //! freeing `data` and/or the destination `DeviceBuffer` on a failed
    //! `stream.synchronize()`, reopening the exact use-after-free
    //! forkwright/logismos#25 and #104 close on the success path).
    //!
    //! `Device::invalid_for_test` drives a real, deterministic
    //! `hipSetDevice` failure through the shipped `Stream::synchronize` ->
    //! `Device::make_current` path with no physical GPU required, so these
    //! tests exercise the actual `Drop for PendingCopy` / `PendingCopy::wait`
    //! impls above, not a re-implementation of them. Two independent
    //! observers tell "leaked" from "freed" apart without reading memory
    //! that may have been freed, which would itself be the unsound thing
    //! these tests exist to rule out:
    //!
    //! - `WatchingAlloc` (source side, #25/#101): intercepts the system
    //!   allocator's `dealloc` and records whether it is ever called
    //!   against the exact address `data` was allocated at.
    //! - `Device::strong_count` (destination side, #104): a fabricated
    //!   `DeviceBuffer`'s `device: Device` field always gets dropped by
    //!   ordinary field-drop-glue once `DeviceBuffer::drop`'s custom body
    //!   returns, success or logged-skip alike — `hipFree` itself is never
    //!   reachable here (the device is always-invalid, so
    //!   `DeviceBuffer::drop`'s own `make_current` guard always skips it),
    //!   which is exactly why the strong count, not the FFI call, is the
    //!   signal: it is what actually distinguishes "the buffer's
    //!   destructor ran" (bug) from "the buffer was correctly left
    //!   untouched" (fix).

    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// Builds a `DeviceBuffer` standing in for a real destination
    /// allocation, without ever making a real FFI call: `ptr` is never
    /// dereferenced (nothing here ever reads or writes through it), and
    /// `device` is always `Device::invalid_for_test`'s always-fails
    /// handle, so `DeviceBuffer::drop`'s `make_current` guard skips
    /// `hipFree` even if this value's `Drop` does run. Bypasses
    /// `DeviceBuffer::alloc` deliberately: `alloc` itself calls
    /// `device.make_current()` first and would fail against an invalid
    /// device before ever returning something to fabricate a test with.
    fn fake_device_buffer(device: &Device, len: usize) -> DeviceBuffer<u8> {
        DeviceBuffer {
            ptr: NonNull::dangling(),
            len,
            device: device.clone(),
            _marker: PhantomData,
        }
    }

    /// Address the running test is watching, and whether `dealloc` has
    /// been called against it. `nextest` runs each `#[test]` in its own
    /// process (the fleet's standard runner here — see
    /// `.github/workflows/gate-attestation.yml`'s `nextest_cmd`), so this
    /// process-wide allocator swap cannot observe or be polluted by any
    /// other test's allocations.
    static WATCH_PTR: AtomicUsize = AtomicUsize::new(0);
    static WATCH_FREED: AtomicUsize = AtomicUsize::new(0);

    struct WatchingAlloc;

    // SAFETY: every method delegates directly to `System`, passing
    // through the same pointer/layout `System` already validates;
    // `dealloc` additionally reads (never mutates) two `AtomicUsize`s.
    unsafe impl GlobalAlloc for WatchingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // SAFETY: `layout` is the caller's, unmodified; forwarded
            // verbatim to `System`.
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            if WATCH_PTR.load(Ordering::SeqCst) == ptr as usize {
                WATCH_FREED.store(1, Ordering::SeqCst);
            }
            // SAFETY: `ptr`/`layout` are the caller's, unmodified;
            // forwarded verbatim to `System`.
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    #[global_allocator]
    static ALLOC: WatchingAlloc = WatchingAlloc;

    #[test]
    fn drop_leaks_data_instead_of_freeing_it_when_synchronize_fails() {
        // WHY: `Stream::synchronize` calls `self.device.make_current()`
        // (`hipSetDevice`) before touching the stream handle at all, so
        // an invalid device fails it deterministically without ever
        // needing a real stream or a physical GPU.
        let device = Device::invalid_for_test();
        let stream = Stream::null(&device);

        let data: Vec<u8> = vec![1, 2, 3, 4];
        let watched_ptr = data.as_ptr() as usize;
        WATCH_PTR.store(watched_ptr, Ordering::SeqCst);
        WATCH_FREED.store(0, Ordering::SeqCst);
        let data_len = data.len();

        // Bypasses `DeviceBuffer::copy_from_host_async` deliberately: that
        // constructor would itself fail against `device` before ever
        // returning a `PendingCopy`, since it also calls `make_current`.
        // This is the same `PendingCopy` the real constructor returns —
        // same fields, same `Drop` impl — just assembled directly, which
        // this module (a descendant of `memory`) can do because the
        // fields are private to the crate, not faked at another layer.
        // WHY `data_len` is captured before the literal below rather than
        // called as `data.len()` inline: struct-literal fields evaluate
        // left-to-right in the order written, not declaration order, and
        // the `data` field's shorthand moves `data` before a later `buf`
        // field could still borrow it.
        let pending = PendingCopy {
            data,
            buf: ManuallyDrop::new(fake_device_buffer(&device, data_len)),
            stream: &stream,
            synced: false,
        };

        drop(pending);

        assert_eq!(
            WATCH_FREED.load(Ordering::SeqCst),
            0,
            "PendingCopy::drop freed `data` after stream.synchronize() failed \
             — the DMA this copy started may still have been reading it. \
             This reopens the exact use-after-free forkwright/logismos#25 \
             exists to prevent, now on the error path (PR #101 review \
             finding 1) instead of the caller's."
        );
    }

    /// Negative-case fixture for forkwright/logismos#104: the destination
    /// side of the same `Drop for PendingCopy` this module's #101 fixture
    /// (above) already covers for the source side. `copy_from_host_async`
    /// used to borrow the destination `DeviceBuffer` only for the
    /// duration of the call, so a safe caller could `drop` it the instant
    /// the call returned, while the queued DMA was still writing into it.
    /// The fix wraps the destination in `PendingCopy` too, so it can only
    /// be released once a synchronize against `stream` has actually
    /// succeeded.
    #[test]
    fn drop_leaks_destination_buffer_instead_of_freeing_it_when_synchronize_fails() {
        let device = Device::invalid_for_test();
        let stream = Stream::null(&device);

        let data: Vec<u8> = vec![1, 2, 3, 4];
        let data_len = data.len();
        let pending = PendingCopy {
            data,
            buf: ManuallyDrop::new(fake_device_buffer(&device, data_len)),
            stream: &stream,
            synced: false,
        };

        let before_drop = device.strong_count();
        drop(pending);

        assert_eq!(
            device.strong_count(),
            before_drop,
            "PendingCopy::drop released the destination DeviceBuffer's Device handle \
             after stream.synchronize() failed — the DMA this copy started may still \
             have been writing into it. This reopens forkwright/logismos#104 on the \
             error path."
        );
    }

    /// Negative-case fixture for the `PendingCopy::wait` ordering bug
    /// found while implementing forkwright/logismos#104: `wait` used to
    /// set `self.synced = true` BEFORE calling `stream.synchronize()`,
    /// not after. On a failed synchronize this made `Drop::drop`'s `if
    /// self.synced { return; }` guard fire immediately and skip its own
    /// synchronize-or-leak fallback entirely, letting ordinary
    /// field-drop-glue free `data` (and, with #104's destination guard
    /// added naively on top of the same ordering, the destination buffer
    /// too) unconditionally right after `wait` returned `Err` —
    /// reopening the exact use-after-free on `wait`'s error path instead
    /// of a bare `drop(pending)`'s. Fixed by moving `self.synced = true`
    /// to after a successful `synchronize`, so a failed one falls through
    /// to the same `Drop` fallback the bare-drop path above already
    /// relies on.
    #[test]
    fn wait_leaks_both_buffers_instead_of_freeing_them_when_synchronize_fails() {
        let device = Device::invalid_for_test();
        let stream = Stream::null(&device);

        let data: Vec<u8> = vec![1, 2, 3, 4];
        let watched_ptr = data.as_ptr() as usize;
        WATCH_PTR.store(watched_ptr, Ordering::SeqCst);
        WATCH_FREED.store(0, Ordering::SeqCst);
        let data_len = data.len();

        let pending = PendingCopy {
            data,
            buf: ManuallyDrop::new(fake_device_buffer(&device, data_len)),
            stream: &stream,
            synced: false,
        };

        let before_drop = device.strong_count();
        let result = pending.wait();

        assert!(
            result.is_err(),
            "wait() must return Err when stream.synchronize() fails against an \
             invalid device"
        );
        assert_eq!(
            WATCH_FREED.load(Ordering::SeqCst),
            0,
            "wait() freed `data` on its own failed synchronize() before falling \
             through to Drop's fallback — reopens forkwright/logismos#25 on \
             wait()'s error path."
        );
        assert_eq!(
            device.strong_count(),
            before_drop,
            "wait() released the destination DeviceBuffer's Device handle on its \
             own failed synchronize() before falling through to Drop's fallback \
             — reopens forkwright/logismos#104 on wait()'s error path."
        );
    }
}
