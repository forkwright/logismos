//! Tensor storage variants.

use std::sync::Arc;

use hipcore::{BytePod, Device, DeviceBuffer};

use crate::dtype::DType;
use crate::error::{Error, Result};

/// Type-erased CPU storage. One variant per supported dtype that has
/// a native Rust type; other dtypes live in raw-byte storage later.
#[derive(Debug)]
#[non_exhaustive]
pub enum CpuStorage {
    /// `f32` buffer.
    F32(Vec<f32>),
    /// `half::f16` buffer.
    F16(Vec<half::f16>),
    /// `half::bf16` buffer.
    BF16(Vec<half::bf16>),
    /// `i32` buffer.
    I32(Vec<i32>),
    /// `i8` buffer.
    I8(Vec<i8>),
    /// `u8` buffer.
    U8(Vec<u8>),
}

impl CpuStorage {
    /// Dtype of this storage.
    #[must_use]
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F16(_) => DType::F16,
            Self::BF16(_) => DType::BF16,
            Self::I32(_) => DType::I32,
            Self::I8(_) => DType::I8,
            Self::U8(_) => DType::U8,
        }
    }

    /// Element count.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F16(v) => v.len(),
            Self::BF16(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::U8(v) => v.len(),
        }
    }

    /// Is the storage empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reinterpret as `&[f32]`, checking dtype.
    ///
    /// # Errors
    ///
    /// [`Error::DTypeMismatch`] when dtype is not `F32`.
    pub fn as_slice_f32(&self) -> Result<&[f32]> {
        match self {
            Self::F32(v) => Ok(v),
            _ => Err(Error::DTypeMismatch {
                op: "as_slice_f32",
                expected: DType::F32,
                got: self.dtype(),
            }),
        }
    }

    /// Reinterpret as `&[f16]`, checking dtype.
    ///
    /// # Errors
    ///
    /// [`Error::DTypeMismatch`] when dtype is not `F16`.
    pub fn as_slice_f16(&self) -> Result<&[half::f16]> {
        match self {
            Self::F16(v) => Ok(v),
            _ => Err(Error::DTypeMismatch {
                op: "as_slice_f16",
                expected: DType::F16,
                got: self.dtype(),
            }),
        }
    }

    /// Reinterpret as `&[bf16]`, checking dtype.
    ///
    /// # Errors
    ///
    /// [`Error::DTypeMismatch`] when dtype is not `BF16`.
    pub fn as_slice_bf16(&self) -> Result<&[half::bf16]> {
        match self {
            Self::BF16(v) => Ok(v),
            _ => Err(Error::DTypeMismatch {
                op: "as_slice_bf16",
                expected: DType::BF16,
                got: self.dtype(),
            }),
        }
    }
}

/// Device-side storage: raw byte buffer + dtype + element count.
///
/// The buffer is `Arc` so tensor clones are cheap (metadata-only) and
/// cross-stream aliasing is possible. Storage is immutable from the
/// Rust side once allocated; in-place kernels operate on raw pointers
/// through a unique-access API not required in Phase 1.
pub struct HipStorage {
    buffer: Arc<DeviceBuffer<u8>>,
    dtype: DType,
    elem_count: usize,
    device: Device,
}

// SAFETY: `HipStorage` is a device-memory handle. HIP supports
// cross-thread access once the device is current. `DeviceBuffer` is
// `Send` (enforced via its own `unsafe impl Send`). We additionally
// assert `Sync` because `Tensor` holds `Arc<Storage>` for cheap
// clones; Phase-1 kernels never mutate device memory through a shared
// reference (matmul allocates its own output buffer), so the `Sync`
// invariant (shared immutable access is safe) holds.
unsafe impl Send for HipStorage {}
// SAFETY: see above.
unsafe impl Sync for HipStorage {}

impl HipStorage {
    /// Allocate a zeroed device buffer sized for `elem_count` of
    /// `dtype` on `device`.
    ///
    /// # Errors
    ///
    /// [`Error::Hip`] on allocation or zero-fill failure.
    pub fn alloc(device: &Device, dtype: DType, elem_count: usize) -> Result<Self> {
        let bytes = dtype.byte_count(elem_count);
        let mut buffer = DeviceBuffer::<u8>::alloc(device, bytes)?;
        // WARNING: `hipMalloc` does not zero device memory. This
        // constructor's name and contract promise zeroed output (see
        // `Tensor::zeros_hip`), so callers must never observe residual
        // memory from a prior allocation — closes forkwright/logismos#26.
        buffer.zero_fill()?;
        Ok(Self {
            buffer: Arc::new(buffer),
            dtype,
            elem_count,
            device: device.clone(),
        })
    }

    /// Copy a typed host slice to a freshly allocated HIP storage.
    ///
    /// `T`'s size must exactly match `dtype`'s declared element size —
    /// a caller-supplied `dtype` that disagrees with the actual host
    /// buffer would otherwise construct a storage whose `dtype`,
    /// `elem_count`, and `byte_len` are mutually inconsistent, letting
    /// dtype-dispatched kernels reinterpret the bytes at a distance
    /// from this call site (forkwright/logismos#40).
    ///
    /// # Errors
    ///
    /// [`Error::Msg`] when `dtype`'s element size disagrees with
    /// `size_of::<T>()`, or when `dtype` is sub-byte-packed (no
    /// `T: BytePod` slice can represent it).
    /// [`Error::Hip`] on allocation or memcpy failure.
    pub fn from_host<T: BytePod>(device: &Device, dtype: DType, data: &[T]) -> Result<Self> {
        Self::validate_dtype_matches::<T>(dtype)?;
        // SAFETY: `T: BytePod` guarantees every bit pattern is valid
        // and the type is `Copy`. Transmuting the slice to a byte
        // view is defined.
        let bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(data.as_ptr().cast::<u8>(), core::mem::size_of_val(data))
        };
        let buffer = DeviceBuffer::<u8>::from_host(device, bytes)?;
        Ok(Self {
            buffer: Arc::new(buffer),
            dtype,
            elem_count: data.len(),
            device: device.clone(),
        })
    }

    /// Reject a `T` / `dtype` pair whose element sizes disagree, or a
    /// `dtype` no `T: BytePod` slice can represent (sub-byte-packed).
    ///
    /// Pure host-side arithmetic — no device required — so it is
    /// unit-testable without HIP/ROCm and is the sole gate `from_host`
    /// runs through before touching device memory.
    ///
    /// # Errors
    ///
    /// [`Error::Msg`] on disagreement, per [`Self::from_host`].
    fn validate_dtype_matches<T: BytePod>(dtype: DType) -> Result<()> {
        let elem_size = core::mem::size_of::<T>();
        match dtype.size_in_bytes_exact() {
            Some(expected) if expected == elem_size => Ok(()),
            Some(expected) => Err(Error::Msg(format!(
                "from_host dtype/element-size mismatch: dtype {dtype:?} is {expected}B/elem, \
                 but T is {elem_size}B (elem_count and byte_len would disagree with dtype)"
            ))),
            None => Err(Error::Msg(format!(
                "from_host: dtype {dtype:?} is sub-byte-packed and cannot be represented by \
                 a T: BytePod slice"
            ))),
        }
    }

    /// Element count.
    #[must_use]
    pub fn elem_count(&self) -> usize {
        self.elem_count
    }

    /// Element type of the stored buffer.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Underlying device handle.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Raw device pointer (bytes).
    #[must_use]
    pub fn as_device_ptr(&self) -> *const u8 {
        self.buffer.as_device_ptr()
    }

    /// Mutable device pointer (bytes).
    ///
    /// # Safety
    ///
    /// `HipStorage` does not enforce unique access — it is held behind
    /// `Arc` so `Tensor` clones stay cheap, and kernel launches are
    /// asynchronous, so no Rust borrow scope can bound the true
    /// in-flight write window on the device. The caller must ensure no
    /// other live pointer (from this handle, a cloned `Arc`, or another
    /// in-flight kernel) reads or writes this allocation for as long as
    /// the returned pointer is used for a write.
    #[must_use]
    pub unsafe fn as_mut_device_ptr(&self) -> *mut u8 {
        self.buffer.as_device_ptr()
    }

    /// Reference to the underlying allocation.
    #[must_use]
    pub fn buffer(&self) -> &DeviceBuffer<u8> {
        &self.buffer
    }

    /// Byte length of the underlying allocation.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.buffer.byte_len()
    }

    /// Copy the entire device allocation back to host as bytes.
    ///
    /// # Errors
    ///
    /// [`Error::Hip`] on memcpy failure.
    pub fn to_host_bytes(&self) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.byte_len()];
        self.buffer.copy_to_host(&mut out)?;
        Ok(out)
    }

    /// Copy device storage into a typed host buffer.
    ///
    /// # Errors
    ///
    /// [`Error::DTypeMismatch`] when `T` disagrees with `self.dtype`
    /// (size check only — callers typically dispatch on `dtype` first).
    /// [`Error::Hip`] on memcpy failure.
    pub(crate) fn to_host<T: BytePod>(&self, dtype_check: DType) -> Result<Vec<T>> {
        if self.dtype != dtype_check {
            return Err(Error::DTypeMismatch {
                op: "HipStorage::to_host",
                expected: dtype_check,
                got: self.dtype,
            });
        }
        if core::mem::size_of::<T>() * self.elem_count != self.byte_len() {
            return Err(Error::Msg(format!(
                "to_host size mismatch: T={}B × {}elems != {}B",
                core::mem::size_of::<T>(),
                self.elem_count,
                self.byte_len()
            )));
        }
        let mut out: Vec<T> = Vec::with_capacity(self.elem_count);
        // SAFETY: `T: BytePod` means a byte-write followed by `set_len`
        // yields well-formed `T` values. Capacity was just allocated.
        unsafe {
            let dst_bytes: &mut [u8] =
                core::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), self.byte_len());
            self.buffer.copy_to_host(dst_bytes)?;
            out.set_len(self.elem_count);
        }
        Ok(out)
    }
}

impl std::fmt::Debug for HipStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipStorage")
            .field("dtype", &self.dtype)
            .field("elem_count", &self.elem_count)
            .field("byte_len", &self.byte_len())
            .finish_non_exhaustive()
    }
}

/// Tensor storage: either CPU (native Rust `Vec`) or HIP (device
/// allocation).
#[derive(Debug)]
#[non_exhaustive]
pub enum Storage {
    /// Host (CPU) storage.
    Cpu(CpuStorage),
    /// Device (HIP) storage.
    Hip(HipStorage),
}

impl Storage {
    /// Dtype of the underlying data.
    #[must_use]
    pub fn dtype(&self) -> DType {
        match self {
            Self::Cpu(c) => c.dtype(),
            Self::Hip(h) => h.dtype(),
        }
    }

    /// Element count.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Cpu(c) => c.len(),
            Self::Hip(h) => h.elem_count(),
        }
    }

    /// Is the storage empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // WHY: pure host-side arithmetic (dtype byte-size vs `size_of::<T>()`) —
    // no `Device`/HIP runtime involved, so these run without ROCm/GPU
    // hardware, unlike every other `HipStorage` constructor.

    #[test]
    fn validate_dtype_matches_accepts_agreeing_pair() {
        assert!(HipStorage::validate_dtype_matches::<f32>(DType::F32).is_ok());
        assert!(HipStorage::validate_dtype_matches::<half::f16>(DType::F16).is_ok());
        assert!(HipStorage::validate_dtype_matches::<u8>(DType::U8).is_ok());
    }

    #[test]
    fn validate_dtype_matches_rejects_size_mismatch() {
        // WHY: the exact confusion #40 describes — T=f32 (4B) paired with a
        // caller-supplied dtype declaring 2B/elem. Pre-fix, `from_host`
        // accepted this silently and stored `dtype=F16` with
        // `elem_count` counted in f32 units — internally inconsistent
        // metadata a dtype-dispatched kernel would misread.
        assert!(matches!(
            HipStorage::validate_dtype_matches::<f32>(DType::F16),
            Err(Error::Msg(_))
        ));
    }

    #[test]
    fn validate_dtype_matches_rejects_sub_byte_dtype() {
        // WHY: no `T: BytePod` slice can represent a packed sub-byte dtype
        // regardless of `T`; `size_in_bytes_exact()` is `None` for I4.
        assert!(matches!(
            HipStorage::validate_dtype_matches::<u8>(DType::I4),
            Err(Error::Msg(_))
        ));
    }
}
