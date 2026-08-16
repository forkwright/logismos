//! `Tensor` — the public tier-1 data vessel.

use std::sync::Arc;

use hipcore::{BytePod, Device};

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::shape::Shape;
use crate::storage::{CpuStorage, HipStorage, Storage};

/// Cheap-clone tensor handle.
#[derive(Clone)]
pub struct Tensor {
    inner: Arc<TensorInner>,
}

struct TensorInner {
    storage: Arc<Storage>,
    layout: Layout,
    dtype: DType,
}

impl Tensor {
    /// Construct a CPU tensor from a typed `Vec`.
    #[must_use]
    pub fn from_cpu(storage: CpuStorage, shape: Shape) -> Self {
        let dtype = storage.dtype();
        let layout = Layout::contiguous(shape);
        Self {
            inner: Arc::new(TensorInner {
                dtype,
                storage: Arc::new(Storage::Cpu(storage)),
                layout,
            }),
        }
    }

    /// Construct a HIP tensor from a host slice of `f32`.
    ///
    /// # Errors
    ///
    /// [`Error::ShapeMismatch`] when `data.len() != shape.elem_count()`.
    /// [`Error::Hip`] on device allocation or copy failure.
    pub fn from_host_f32(device: &Device, data: &[f32], shape: Shape) -> Result<Self> {
        Self::from_host_typed(device, data, shape, DType::F32)
    }

    /// Construct a HIP tensor from a host slice of `half::f16`.
    ///
    /// # Errors
    ///
    /// See [`Self::from_host_f32`].
    pub fn from_host_f16(device: &Device, data: &[half::f16], shape: Shape) -> Result<Self> {
        Self::from_host_typed(device, data, shape, DType::F16)
    }

    /// Construct a HIP tensor from a host slice of `half::bf16`.
    ///
    /// # Errors
    ///
    /// See [`Self::from_host_f32`].
    pub fn from_host_bf16(device: &Device, data: &[half::bf16], shape: Shape) -> Result<Self> {
        Self::from_host_typed(device, data, shape, DType::BF16)
    }

    fn from_host_typed<T: BytePod>(
        device: &Device,
        data: &[T],
        shape: Shape,
        dtype: DType,
    ) -> Result<Self> {
        if data.len() != shape.elem_count() {
            return Err(Error::ShapeMismatch {
                op: "from_host_typed",
                msg: format!(
                    "data.len()={} != shape.elem_count()={}",
                    data.len(),
                    shape.elem_count()
                ),
            });
        }
        let storage = HipStorage::from_host(device, dtype, data)?;
        let layout = Layout::contiguous(shape);
        Ok(Self {
            inner: Arc::new(TensorInner {
                dtype,
                storage: Arc::new(Storage::Hip(storage)),
                layout,
            }),
        })
    }

    /// Allocate a HIP tensor of the given shape + dtype, zero-filled.
    ///
    /// # Errors
    ///
    /// [`Error::Hip`] on allocation or zero-fill failure.
    pub fn zeros_hip(device: &Device, dtype: DType, shape: Shape) -> Result<Self> {
        let elem = shape.elem_count();
        let storage = HipStorage::alloc(device, dtype, elem)?;
        let layout = Layout::contiguous(shape);
        Ok(Self {
            inner: Arc::new(TensorInner {
                dtype,
                storage: Arc::new(Storage::Hip(storage)),
                layout,
            }),
        })
    }

    /// Element type of this tensor.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.inner.dtype
    }

    /// Strided memory layout (shape + offsets + contiguity) of this tensor.
    #[must_use]
    pub fn layout(&self) -> &Layout {
        &self.inner.layout
    }

    /// Multi-axis extent of this tensor (shortcut for `layout().shape()`).
    #[must_use]
    pub fn shape(&self) -> &Shape {
        self.inner.layout.shape()
    }

    /// Dimensions.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        self.inner.layout.dims()
    }

    /// Element count.
    #[must_use]
    pub fn elem_count(&self) -> usize {
        self.inner.layout.elem_count()
    }

    /// Storage reference.
    #[must_use]
    pub fn storage(&self) -> &Storage {
        &self.inner.storage
    }

    /// True when the layout is row-major contiguous starting at offset 0.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        self.inner.layout.is_contiguous()
    }

    /// True when the tensor lives on a HIP device.
    #[must_use]
    pub fn is_on_device(&self) -> bool {
        matches!(self.inner.storage.as_ref(), Storage::Hip(_))
    }

    /// Access the HIP storage, if any.
    #[must_use]
    pub fn hip_storage(&self) -> Option<&HipStorage> {
        match self.inner.storage.as_ref() {
            Storage::Hip(h) => Some(h),
            Storage::Cpu(_) => None,
        }
    }

    /// Access the CPU storage, if any.
    #[must_use]
    pub fn cpu_storage(&self) -> Option<&CpuStorage> {
        match self.inner.storage.as_ref() {
            Storage::Cpu(c) => Some(c),
            Storage::Hip(_) => None,
        }
    }

    /// Device, if the tensor is on HIP.
    #[must_use]
    pub fn device(&self) -> Option<&Device> {
        self.hip_storage().map(HipStorage::device)
    }

    /// Copy device tensor back to a typed host `Vec<f32>`.
    ///
    /// # Errors
    ///
    /// [`Error::WrongStorage`] if the tensor is CPU-backed.
    /// [`Error::DTypeMismatch`] if the dtype is not `F32`.
    /// [`Error::Hip`] on copy failure.
    pub fn to_host_f32(&self) -> Result<Vec<f32>> {
        match self.inner.storage.as_ref() {
            Storage::Hip(h) => h.to_host::<f32>(DType::F32),
            Storage::Cpu(CpuStorage::F32(v)) => Ok(v.clone()),
            Storage::Cpu(_) => Err(Error::DTypeMismatch {
                op: "to_host_f32",
                expected: DType::F32,
                got: self.inner.dtype,
            }),
        }
    }

    /// Copy device tensor back to a typed host `Vec<f16>`.
    ///
    /// # Errors
    ///
    /// See [`Self::to_host_f32`].
    pub fn to_host_f16(&self) -> Result<Vec<half::f16>> {
        match self.inner.storage.as_ref() {
            Storage::Hip(h) => h.to_host::<half::f16>(DType::F16),
            Storage::Cpu(CpuStorage::F16(v)) => Ok(v.clone()),
            Storage::Cpu(_) => Err(Error::DTypeMismatch {
                op: "to_host_f16",
                expected: DType::F16,
                got: self.inner.dtype,
            }),
        }
    }

    /// Copy device tensor back to a typed host `Vec<bf16>`.
    ///
    /// # Errors
    ///
    /// See [`Self::to_host_f32`].
    pub fn to_host_bf16(&self) -> Result<Vec<half::bf16>> {
        match self.inner.storage.as_ref() {
            Storage::Hip(h) => h.to_host::<half::bf16>(DType::BF16),
            Storage::Cpu(CpuStorage::BF16(v)) => Ok(v.clone()),
            Storage::Cpu(_) => Err(Error::DTypeMismatch {
                op: "to_host_bf16",
                expected: DType::BF16,
                got: self.inner.dtype,
            }),
        }
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("dtype", &self.inner.dtype)
            .field("dims", &self.inner.layout.dims())
            .field("storage", self.inner.storage.as_ref())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_tensor_constructs() {
        let t = Tensor::from_cpu(
            CpuStorage::F32(vec![1.0, 2.0, 3.0, 4.0]),
            Shape::new(&[2, 2]),
        );
        assert_eq!(t.dims(), &[2, 2]);
        assert_eq!(t.dtype(), DType::F32);
        assert!(t.is_contiguous());
        assert!(!t.is_on_device());
    }

    // INVARIANT: `zeros_hip` must produce an all-zero buffer (forkwright/logismos#26).
    // Real shipped-code exercise, not a copy — but the assertion can only run
    // against actual device memory, and no box in this fleet nor any GH-hosted
    // CI runner exposes a physical ROCm GPU (CI installs `libamdhip64-dev`
    // headers/link only; see .github/workflows/gate-attestation.yml). The
    // ignore annotation below documents and preserves the check for the day
    // this runs on ROCm hardware; until then the invariant rests on code
    // review of the `hipMemset` call at `HipStorage::alloc` (storage.rs)
    // covering exactly `byte_len()` bytes.
    #[test]
    #[ignore = "requires a physical ROCm GPU device — unavailable on any fleet \
                box or GH-hosted CI runner; run manually on ROCm hardware — see #26"]
    fn zeros_hip_produces_zero_filled_buffer() -> Result<()> {
        let device = Device::new(0)?;
        let t = Tensor::zeros_hip(&device, DType::F32, Shape::new(&[4, 4]))?;
        let host = t.to_host_f32()?;
        assert_eq!(host, vec![0.0_f32; 16]);
        Ok(())
    }
}
