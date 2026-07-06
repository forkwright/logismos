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
    /// [`Error::Hip`] on allocation failure.
    pub fn alloc(device: &Device, dtype: DType, elem_count: usize) -> Result<Self> {
        let bytes = dtype.byte_count(elem_count);
        let buffer = DeviceBuffer::<u8>::alloc(device, bytes)?;
        Ok(Self {
            buffer: Arc::new(buffer),
            dtype,
            elem_count,
            device: device.clone(),
        })
    }

    /// Copy a typed host slice to a freshly allocated HIP storage.
    ///
    /// # Errors
    ///
    /// [`Error::Hip`] on allocation or memcpy failure.
    pub fn from_host<T: BytePod>(device: &Device, dtype: DType, data: &[T]) -> Result<Self> {
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

    /// Mutable device pointer (bytes). Callers must uphold aliasing
    /// themselves; `HipStorage` does not enforce unique access.
    #[must_use]
    pub fn as_mut_device_ptr(&self) -> *mut u8 {
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
