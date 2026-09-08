//! Q8_0 matrix-vector product: HIP launcher plus checked CPU reference.
//!
//! The matrix is raw GGML `Q8_0` storage in row-major `[rows, width]`
//! order. It multiplies one finite f32 vector of `width` values and produces
//! one f32 value per row. This is a correctness baseline, not a WMMA or
//! whole-model execution path.

pub mod cpu;

#[cfg(feature = "gpu")]
use std::ffi::c_void;

#[cfg(feature = "gpu")]
use hipcore::Stream;
use snafu::ResultExt;

#[cfg(all(feature = "gpu", not(logismos_no_gpu_kernels)))]
use crate::error::LaunchSnafu;
#[cfg(all(feature = "gpu", logismos_no_gpu_kernels))]
use crate::error::NoGpuBuildSnafu;
use crate::error::{QuantSnafu, Result, UnsupportedShapeSnafu};

const KERNEL: &str = "q8_0_gemv_f32";

#[cfg(all(feature = "gpu", not(logismos_no_gpu_kernels)))]
unsafe extern "C" {
    fn logismos_launch_q8_0_gemv_f32(
        matrix_q8_0: *const c_void,
        activations_f32: *const c_void,
        output_f32: *mut c_void,
        rows: i32,
        width: i32,
        stream: *mut c_void,
    ) -> u32;
}

/// Validated Q8_0 matrix/vector extents shared by CPU execution and HIP launch.
#[derive(Clone, Copy, Debug)]
pub struct Q8GemvShape {
    rows: usize,
    width: usize,
    row_bytes: usize,
    matrix_bytes: usize,
}

/// Check one raw Q8_0 GEMV shape before allocation or launch.
///
/// The row byte count is delegated to the Q8_0 format owner. This keeps the
/// GGML block geometry and the CPU/GPU contracts on one authority.
impl Q8GemvShape {
    /// Validate the exact raw matrix, activation, and output extents.
    ///
    /// The shape retains only validated geometry, so CPU and HIP callers do
    /// not separately reinterpret dimensions or buffer lengths.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::UnsupportedShape`] when an extent is zero,
    /// inconsistent, too large for a Rust allocation layout, or too large for
    /// the HIP ABI; returns [`crate::Error::Quant`] for invalid Q8_0 width.
    pub fn new(
        rows: usize,
        width: usize,
        matrix_bytes: usize,
        activation_len: usize,
        output_len: usize,
    ) -> Result<Self> {
        if rows == 0 {
            return UnsupportedShapeSnafu {
                kernel: KERNEL,
                msg: "rows must be positive".to_string(),
            }
            .fail();
        }
        if activation_len != width {
            return UnsupportedShapeSnafu {
                kernel: KERNEL,
                msg: format!("activation length {activation_len} must equal width {width}"),
            }
            .fail();
        }
        if output_len != rows {
            return UnsupportedShapeSnafu {
                kernel: KERNEL,
                msg: format!("output length {output_len} must equal rows {rows}"),
            }
            .fail();
        }
        let row_bytes = quant::q8_0::row_byte_len(width).context(QuantSnafu)?;
        let matrix_bytes_expected = rows.checked_mul(row_bytes).ok_or_else(|| {
            UnsupportedShapeSnafu {
                kernel: KERNEL,
                msg: format!("rows * Q8_0 row bytes overflows usize ({rows} * {row_bytes})"),
            }
            .build()
        })?;
        checked_layout::<u8>(matrix_bytes_expected, "matrix bytes")?;
        checked_layout::<f32>(activation_len, "activation length")?;
        checked_layout::<f32>(output_len, "output length")?;
        if matrix_bytes != matrix_bytes_expected {
            return UnsupportedShapeSnafu {
                kernel: KERNEL,
                msg: format!(
                    "matrix bytes {matrix_bytes} must equal {matrix_bytes_expected} for {rows} Q8_0 rows"
                ),
            }
            .fail();
        }
        i32::try_from(rows).map_err(|_| {
            UnsupportedShapeSnafu {
                kernel: KERNEL,
                msg: format!("rows {rows} exceeds the HIP ABI i32 domain"),
            }
            .build()
        })?;
        i32::try_from(width).map_err(|_| {
            UnsupportedShapeSnafu {
                kernel: KERNEL,
                msg: format!("width {width} exceeds the HIP ABI i32 domain"),
            }
            .build()
        })?;
        Ok(Self {
            rows,
            width,
            row_bytes,
            matrix_bytes: matrix_bytes_expected,
        })
    }

    /// Number of row outputs.
    #[must_use]
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Logical F32 activation width.
    #[must_use]
    pub const fn width(self) -> usize {
        self.width
    }

    pub(crate) const fn row_bytes(self) -> usize {
        self.row_bytes
    }

    pub(crate) const fn matrix_bytes(self) -> usize {
        self.matrix_bytes
    }
}

fn checked_layout<T>(elements: usize, label: &str) -> Result<()> {
    std::alloc::Layout::array::<T>(elements).map_err(|_| {
        UnsupportedShapeSnafu {
            kernel: KERNEL,
            msg: format!("{label} {elements} exceeds the Rust allocation layout domain"),
        }
        .build()
    })?;
    Ok(())
}

#[cfg(feature = "gpu")]
/// Launch one raw Q8_0 row-major matrix-vector product on `stream`.
///
/// The GPU kernel is intentionally a sequential per-row correctness baseline:
/// each lane computes `scale * signed_i8`, then one f32 product and one f32
/// accumulation in serialized block/lane order. It does not validate finite
/// device inputs or device results; callers must complete that admission
/// before crossing this unsafe boundary.
///
/// # Errors
///
/// Returns [`crate::Error::UnsupportedShape`] for invalid geometry, exact
/// buffer lengths, allocation layouts, arithmetic products, or HIP ABI widths;
/// [`crate::Error::NoGpuBuild`] before device initialization for a CPU-only
/// build; [`crate::Error::Hip`] when the stream device cannot be restored; or
/// [`crate::Error::Launch`] when HIP rejects the submitted kernel.
///
/// # Safety
///
/// `matrix_q8_0`, `activations_f32`, and `output_f32` must be non-null device
/// buffers on `stream`'s device for the complete launch. Their exact byte or
/// element lengths must match `shape`, and f32 buffers must be aligned for f32.
/// Inputs and every sequential product/accumulator must remain finite within
/// the admitted numerical domain. Inputs must not be concurrently modified;
/// `output_f32` must not alias either input or another concurrent access, and
/// all buffers must outlive the stream's completion.
/// This ABI has no per-row status channel, so it cannot reproduce
/// the CPU reference's non-finite product or accumulator refusals.
pub unsafe fn launch_q8_0_gemv_f32(
    shape: Q8GemvShape,
    matrix_q8_0: *const c_void,
    activations_f32: *const c_void,
    output_f32: *mut c_void,
    stream: &Stream,
) -> Result<()> {
    #[cfg(logismos_no_gpu_kernels)]
    {
        let _ = (
            matrix_q8_0,
            activations_f32,
            output_f32,
            shape.rows,
            shape.width,
            shape.row_bytes,
            shape.matrix_bytes,
            stream,
        );
        no_gpu_build_refusal()
    }

    #[cfg(not(logismos_no_gpu_kernels))]
    {
        let rows = i32::try_from(shape.rows).map_err(|_| {
            UnsupportedShapeSnafu {
                kernel: KERNEL,
                msg: format!("rows {} exceeds the HIP ABI i32 domain", shape.rows),
            }
            .build()
        })?;
        let width = i32::try_from(shape.width).map_err(|_| {
            UnsupportedShapeSnafu {
                kernel: KERNEL,
                msg: format!("width {} exceeds the HIP ABI i32 domain", shape.width),
            }
            .build()
        })?;
        stream.make_current()?;
        // SAFETY: caller upholds device buffer extent, lifetime, non-aliasing,
        // and finite-input obligations documented above; dimensions came from
        // the checked Q8_0 format owner and fit the ABI.
        let code = unsafe {
            logismos_launch_q8_0_gemv_f32(
                matrix_q8_0,
                activations_f32,
                output_f32,
                rows,
                width,
                stream.raw().cast::<c_void>(),
            )
        };
        if code == 0 {
            Ok(())
        } else {
            LaunchSnafu {
                kernel: KERNEL,
                kind: hipcore::ErrorKind::from_raw(code),
                code,
            }
            .fail()
        }
    }
}

#[cfg(all(feature = "gpu", logismos_no_gpu_kernels))]
fn no_gpu_build_refusal() -> Result<()> {
    NoGpuBuildSnafu { kernel: KERNEL }.fail()
}

#[cfg(test)]
mod tests {
    use super::{KERNEL, Q8GemvShape};
    use crate::Error;

    #[test]
    fn checked_shape_refuses_zero_rows_and_incomplete_q8_widths() {
        assert!(matches!(
            Q8GemvShape::new(0, 32, 0, 32, 0),
            Err(Error::UnsupportedShape { kernel: KERNEL, .. })
        ));
        assert!(matches!(
            Q8GemvShape::new(1, 31, 0, 31, 1),
            Err(Error::Quant { .. })
        ));
    }

    #[test]
    fn checked_shape_refuses_overflow_before_any_output_allocation() {
        assert!(matches!(
            Q8GemvShape::new(usize::MAX, 32, 0, 32, usize::MAX),
            Err(Error::UnsupportedShape { kernel: KERNEL, .. })
        ));
    }

    #[test]
    fn checked_shape_refuses_extents_outside_the_rust_allocation_domain() {
        let too_large_for_f32 = usize::try_from(isize::MAX)
            .expect("isize::MAX is non-negative")
            .checked_div(core::mem::size_of::<f32>())
            .and_then(|elements| elements.checked_add(1))
            .expect("test platform has a representable layout boundary");
        assert!(matches!(
            super::checked_layout::<f32>(too_large_for_f32, "test f32 extent"),
            Err(Error::UnsupportedShape { kernel: KERNEL, .. })
        ));
    }

    #[cfg(all(feature = "gpu", logismos_no_gpu_kernels))]
    #[test]
    fn cpu_only_build_refuses_gpu_launch_without_a_device() {
        assert!(matches!(
            super::no_gpu_build_refusal(),
            Err(Error::NoGpuBuild { .. })
        ));
    }
}
