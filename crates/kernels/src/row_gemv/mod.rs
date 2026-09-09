//! Checked serialized-row matrix-vector products for executable `quant` formats.

pub mod cpu;

#[cfg(feature = "gpu")]
use std::ffi::c_void;

#[cfg(feature = "gpu")]
use hipcore::Stream;
use snafu::ResultExt;

#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
use crate::device_span::{
    checked_f32_device_span, checked_u8_device_span, reject_overlapping_device_spans,
};
#[cfg(all(feature = "gpu", not(logismos_no_gpu_kernels)))]
use crate::error::LaunchSnafu;
#[cfg(all(feature = "gpu", logismos_no_gpu_kernels))]
use crate::error::NoGpuBuildSnafu;
use crate::error::{QuantSnafu, Result, RowGemvAllocationSnafu, UnsupportedShapeSnafu};

const KERNEL: &str = "row_gemv_f32";

#[cfg(all(feature = "gpu", not(logismos_no_gpu_kernels)))]
unsafe extern "C" {
    fn logismos_launch_f32_row_gemv_f32(
        matrix: *const c_void,
        activations: *const c_void,
        output: *mut c_void,
        rows: i32,
        width: i32,
        stream: *mut c_void,
    ) -> u32;
    fn logismos_launch_q8_0_row_gemv_f32(
        matrix: *const c_void,
        activations: *const c_void,
        output: *mut c_void,
        rows: i32,
        width: i32,
        stream: *mut c_void,
    ) -> u32;
    fn logismos_launch_q4_k_row_gemv_f32(
        matrix: *const c_void,
        activations: *const c_void,
        output: *mut c_void,
        rows: i32,
        width: i32,
        stream: *mut c_void,
    ) -> u32;
    fn logismos_launch_q5_k_row_gemv_f32(
        matrix: *const c_void,
        activations: *const c_void,
        output: *mut c_void,
        rows: i32,
        width: i32,
        stream: *mut c_void,
    ) -> u32;
    fn logismos_launch_q6_k_row_gemv_f32(
        matrix: *const c_void,
        activations: *const c_void,
        output: *mut c_void,
        rows: i32,
        width: i32,
        stream: *mut c_void,
    ) -> u32;
    fn logismos_launch_iq4_nl_row_gemv_f32(
        matrix: *const c_void,
        activations: *const c_void,
        output: *mut c_void,
        rows: i32,
        width: i32,
        stream: *mut c_void,
    ) -> u32;
    fn logismos_launch_iq4_xs_row_gemv_f32(
        matrix: *const c_void,
        activations: *const c_void,
        output: *mut c_void,
        rows: i32,
        width: i32,
        stream: *mut c_void,
    ) -> u32;
}

/// Validated serialized matrix/vector extents shared by CPU execution and HIP launch.
#[derive(Clone, Copy, Debug)]
pub struct RowGemvShape {
    format: quant::RowFormat,
    rows: usize,
    width: usize,
    row_bytes: usize,
    matrix_bytes: usize,
}

impl RowGemvShape {
    /// Validate one raw row-major serialized matrix and f32 vector shape.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::UnsupportedShape`] for zero, inconsistent, or
    /// unrepresentable extents, and [`crate::Error::Quant`] when `width` is
    /// not one complete row for `format`.
    pub fn new(
        format: quant::RowFormat,
        rows: usize,
        width: usize,
        matrix_bytes: usize,
        activation_len: usize,
        output_len: usize,
    ) -> Result<Self> {
        if rows == 0 {
            return unsupported_shape("rows must be positive");
        }
        if activation_len != width {
            return unsupported_shape(format!(
                "activation length {activation_len} must equal width {width}"
            ));
        }
        if output_len != rows {
            return unsupported_shape(format!("output length {output_len} must equal rows {rows}"));
        }
        let row_bytes = quant::row_byte_len(format, width).context(QuantSnafu)?;
        let matrix_bytes_expected = rows.checked_mul(row_bytes).ok_or_else(|| {
            UnsupportedShapeSnafu {
                kernel: KERNEL,
                msg: format!("rows * row bytes overflows usize ({rows} * {row_bytes})"),
            }
            .build()
        })?;
        checked_layout::<u8>(matrix_bytes_expected, "matrix bytes")?;
        checked_layout::<f32>(activation_len, "activation length")?;
        checked_layout::<f32>(output_len, "output length")?;
        if matrix_bytes != matrix_bytes_expected {
            return unsupported_shape(format!(
                "matrix bytes {matrix_bytes} must equal {matrix_bytes_expected} for {rows} {format} rows"
            ));
        }
        i32::try_from(rows).map_err(|_| abi_error("rows", rows))?;
        i32::try_from(width).map_err(|_| abi_error("width", width))?;
        Ok(Self {
            format,
            rows,
            width,
            row_bytes,
            matrix_bytes: matrix_bytes_expected,
        })
    }

    /// Serialized row format.
    #[must_use]
    pub const fn format(self) -> quant::RowFormat {
        self.format
    }

    /// Number of matrix rows and output values.
    #[must_use]
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Logical f32 activation width.
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

fn checked_layout<T>(elements: usize, label: &'static str) -> Result<()> {
    std::alloc::Layout::array::<T>(elements).map_err(|_| {
        UnsupportedShapeSnafu {
            kernel: KERNEL,
            msg: format!("{label} {elements} exceeds the Rust allocation layout domain"),
        }
        .build()
    })?;
    Ok(())
}

fn unsupported_shape<T>(msg: impl Into<String>) -> Result<T> {
    UnsupportedShapeSnafu {
        kernel: KERNEL,
        msg: msg.into(),
    }
    .fail()
}

fn abi_error(label: &'static str, value: usize) -> crate::Error {
    UnsupportedShapeSnafu {
        kernel: KERNEL,
        msg: format!("{label} {value} exceeds the HIP ABI i32 domain"),
    }
    .build()
}

#[cfg(feature = "gpu")]
/// Launch one raw serialized row-major matrix-vector product on `stream`.
///
/// The format-specific HIP kernels preserve serial row/block/lane f32 decode,
/// product, and accumulation order. They are correctness baselines, not
/// packed-matrix or whole-model execution paths.
///
/// # Errors
///
/// Returns [`crate::Error::UnsupportedShape`] for invalid geometry, device
/// spans, overlap, layouts, or ABI widths; [`crate::Error::NoGpuBuild`] for a
/// CPU-only build; and HIP launch errors after submission.
///
/// # Safety
///
/// `matrix`, `activations`, and `output` must name exact device allocations on
/// `stream`'s device; each supplied length must describe that full allocation
/// and equal the corresponding checked `shape` extent. Their contents must remain immutable through stream
/// completion, while `output` requires exclusive access through completion.
/// Every stored fp16 scale, f32 operand, decoded value, product, and accumulator
/// must be finite and either zero or normal. This ABI has no device numerical
/// status channel and cannot reproduce CPU arithmetic refusals.
pub unsafe fn launch_row_gemv_f32(
    shape: RowGemvShape,
    matrix: *const u8,
    matrix_bytes: usize,
    activations: *const f32,
    activation_len: usize,
    output: *mut f32,
    output_len: usize,
    stream: &Stream,
) -> Result<()> {
    #[cfg(logismos_no_gpu_kernels)]
    {
        let _ = (
            shape,
            matrix,
            matrix_bytes,
            activations,
            activation_len,
            output,
            output_len,
            stream,
        );
        no_gpu_build_refusal()
    }

    #[cfg(not(logismos_no_gpu_kernels))]
    {
        validate_device_buffers(
            shape,
            matrix,
            matrix_bytes,
            activations,
            activation_len,
            output,
            output_len,
        )?;
        let rows = i32::try_from(shape.rows).map_err(|_| abi_error("rows", shape.rows))?;
        let width = i32::try_from(shape.width).map_err(|_| abi_error("width", shape.width))?;
        stream.make_current()?;
        // SAFETY: caller and validator establish the documented device spans,
        // ownership, alignment, and numerical domain before private ABI entry.
        let code = unsafe {
            launch_format(
                shape.format,
                matrix.cast::<c_void>(),
                activations.cast::<c_void>(),
                output.cast::<c_void>(),
                rows,
                width,
                stream.raw().cast::<c_void>(),
            )?
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

#[cfg(all(feature = "gpu", not(logismos_no_gpu_kernels)))]
unsafe fn launch_format(
    format: quant::RowFormat,
    matrix: *const c_void,
    activations: *const c_void,
    output: *mut c_void,
    rows: i32,
    width: i32,
    stream: *mut c_void,
) -> Result<u32> {
    // SAFETY: caller chooses the format-specific private entry point after all
    // Rust shape and device-span checks; each C ABI has the same pointer ABI.
    unsafe {
        match format {
            quant::RowFormat::F32 => Ok(logismos_launch_f32_row_gemv_f32(
                matrix,
                activations,
                output,
                rows,
                width,
                stream,
            )),
            quant::RowFormat::Q8_0 => Ok(logismos_launch_q8_0_row_gemv_f32(
                matrix,
                activations,
                output,
                rows,
                width,
                stream,
            )),
            quant::RowFormat::Q4K => Ok(logismos_launch_q4_k_row_gemv_f32(
                matrix,
                activations,
                output,
                rows,
                width,
                stream,
            )),
            quant::RowFormat::Q5K => Ok(logismos_launch_q5_k_row_gemv_f32(
                matrix,
                activations,
                output,
                rows,
                width,
                stream,
            )),
            quant::RowFormat::Q6K => Ok(logismos_launch_q6_k_row_gemv_f32(
                matrix,
                activations,
                output,
                rows,
                width,
                stream,
            )),
            quant::RowFormat::IQ4NL => Ok(logismos_launch_iq4_nl_row_gemv_f32(
                matrix,
                activations,
                output,
                rows,
                width,
                stream,
            )),
            quant::RowFormat::IQ4XS => Ok(logismos_launch_iq4_xs_row_gemv_f32(
                matrix,
                activations,
                output,
                rows,
                width,
                stream,
            )),
            _ => unsupported_shape(format!("native HIP row GEMV does not support {format}")),
        }
    }
}

#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
fn validate_device_buffers(
    shape: RowGemvShape,
    matrix: *const u8,
    matrix_bytes: usize,
    activations: *const f32,
    activation_len: usize,
    output: *mut f32,
    output_len: usize,
) -> Result<()> {
    if matrix_bytes != shape.matrix_bytes {
        return unsupported_shape(format!(
            "matrix byte length {matrix_bytes} must equal checked {}",
            shape.matrix_bytes
        ));
    }
    if activation_len != shape.width {
        return unsupported_shape(format!(
            "activation length {activation_len} must equal checked {}",
            shape.width
        ));
    }
    if output_len != shape.rows {
        return unsupported_shape(format!(
            "output length {output_len} must equal checked {}",
            shape.rows
        ));
    }
    let matrix = checked_u8_device_span(KERNEL, matrix, shape.matrix_bytes, "matrix")?;
    let activations = checked_f32_device_span(KERNEL, activations, shape.width, "activations")?;
    let output = checked_f32_device_span(KERNEL, output.cast_const(), shape.rows, "output")?;
    reject_overlapping_device_spans(KERNEL, output, matrix)?;
    reject_overlapping_device_spans(KERNEL, output, activations)
}

#[cfg(all(feature = "gpu", logismos_no_gpu_kernels))]
fn no_gpu_build_refusal() -> Result<()> {
    NoGpuBuildSnafu { kernel: KERNEL }.fail()
}

pub(crate) fn reserve_output(rows: usize) -> Result<Vec<f32>> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(rows)
        .context(RowGemvAllocationSnafu {
            requested_len: rows,
        })?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{KERNEL, RowGemvShape};
    use crate::Error;

    #[test]
    fn shape_uses_the_format_owner_for_every_supported_row_layout()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        for format in [
            quant::RowFormat::F32,
            quant::RowFormat::Q8_0,
            quant::RowFormat::Q4K,
            quant::RowFormat::Q5K,
            quant::RowFormat::Q6K,
            quant::RowFormat::IQ4NL,
            quant::RowFormat::IQ4XS,
        ] {
            let width = match format {
                quant::RowFormat::F32 => 3,
                quant::RowFormat::Q8_0 | quant::RowFormat::IQ4NL => 32,
                quant::RowFormat::Q4K
                | quant::RowFormat::Q5K
                | quant::RowFormat::Q6K
                | quant::RowFormat::IQ4XS => 256,
                _ => return Err("unknown RowFormat in test".into()),
            };
            let row_bytes = quant::row_byte_len(format, width)?;
            let shape = RowGemvShape::new(format, 2, width, row_bytes * 2, width, 2)?;
            assert_eq!(shape.format(), format);
            assert_eq!(shape.row_bytes(), row_bytes);
            assert_eq!(shape.matrix_bytes(), row_bytes * 2);
        }
        Ok(())
    }

    #[test]
    fn shape_refuses_zero_partial_and_overflowing_geometry() {
        assert!(matches!(
            RowGemvShape::new(quant::RowFormat::F32, 0, 1, 4, 1, 0),
            Err(Error::UnsupportedShape { kernel: KERNEL, .. })
        ));
        assert!(matches!(
            RowGemvShape::new(quant::RowFormat::Q4K, 1, 255, 0, 255, 1),
            Err(Error::Quant { .. })
        ));
        assert!(matches!(
            RowGemvShape::new(quant::RowFormat::F32, usize::MAX, 1, 0, 1, usize::MAX),
            Err(Error::UnsupportedShape { kernel: KERNEL, .. })
        ));
        let too_large_for_f32 = isize::MAX.unsigned_abs() / core::mem::size_of::<f32>() + 1;
        assert!(matches!(
            super::checked_layout::<f32>(too_large_for_f32, "test f32 extent"),
            Err(Error::UnsupportedShape { kernel: KERNEL, .. })
        ));
    }

    #[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
    #[test]
    fn device_validator_isolates_length_alignment_and_alias_refusals()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let matrix = [0_u8; 16];
        let activations = [0.0_f32; 2];
        let mut output = [0.0_f32; 2];
        let shape = RowGemvShape::new(quant::RowFormat::F32, 2, 2, 16, 2, 2)?;
        super::validate_device_buffers(
            shape,
            matrix.as_ptr(),
            matrix.len(),
            activations.as_ptr(),
            activations.len(),
            output.as_mut_ptr(),
            output.len(),
        )?;
        assert!(
            super::validate_device_buffers(
                shape,
                matrix.as_ptr(),
                matrix.len() - 1,
                activations.as_ptr(),
                activations.len(),
                output.as_mut_ptr(),
                output.len(),
            )
            .is_err()
        );
        assert!(
            super::validate_device_buffers(
                shape,
                matrix.as_ptr(),
                matrix.len(),
                activations.as_ptr(),
                activations.len() - 1,
                output.as_mut_ptr(),
                output.len(),
            )
            .is_err()
        );
        assert!(
            super::validate_device_buffers(
                shape,
                matrix.as_ptr(),
                matrix.len(),
                activations.as_ptr(),
                activations.len(),
                output.as_mut_ptr(),
                output.len() - 1,
            )
            .is_err()
        );
        assert!(
            super::validate_device_buffers(
                shape,
                core::ptr::null(),
                matrix.len(),
                activations.as_ptr(),
                activations.len(),
                output.as_mut_ptr(),
                output.len(),
            )
            .is_err()
        );
        assert!(
            super::validate_device_buffers(
                shape,
                matrix.as_ptr(),
                matrix.len(),
                activations.as_ptr(),
                activations.len(),
                output.as_mut_ptr().wrapping_byte_add(1),
                output.len(),
            )
            .is_err()
        );
        assert!(
            super::validate_device_buffers(
                shape,
                matrix.as_ptr(),
                matrix.len(),
                activations.as_ptr(),
                activations.len(),
                activations.as_ptr().cast_mut(),
                output.len(),
            )
            .is_err()
        );
        let mut aligned_matrix = [0.0_f32; 4];
        assert!(
            super::validate_device_buffers(
                shape,
                aligned_matrix.as_ptr().cast::<u8>(),
                matrix.len(),
                activations.as_ptr(),
                activations.len(),
                aligned_matrix.as_mut_ptr(),
                output.len(),
            )
            .is_err()
        );
        assert!(matches!(
            crate::device_span::checked_u8_device_span(
                KERNEL,
                (usize::MAX - 1) as *const u8,
                2,
                "overflowing matrix",
            ),
            Err(Error::UnsupportedShape { kernel: KERNEL, .. })
        ));
        Ok(())
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
