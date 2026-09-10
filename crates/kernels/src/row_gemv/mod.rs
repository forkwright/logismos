//! Checked serialized-row matrix-vector products for executable `quant` formats.

pub mod cpu;

#[cfg(all(feature = "gpu", not(logismos_no_gpu_kernels)))]
use std::ffi::c_void;

#[cfg(feature = "gpu")]
use hipcore::Stream;
use snafu::ResultExt;

#[cfg(feature = "gpu")]
use crate::numerical_status::NativeNumericalStatus;

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
const ROW_GEMV_THREADS_PER_BLOCK: usize = 256;

#[cfg(all(feature = "gpu", not(logismos_no_gpu_kernels)))]
unsafe extern "C" {
    fn logismos_launch_f32_row_gemv_f32_checked(
        matrix: *const c_void,
        activations: *const c_void,
        output: *mut c_void,
        rows: i32,
        width: i32,
        tokens: i32,
        numerical_status: *mut c_void,
        stream: *mut c_void,
    ) -> u32;
    fn logismos_launch_q8_0_row_gemv_f32_checked(
        matrix: *const c_void,
        activations: *const c_void,
        output: *mut c_void,
        rows: i32,
        width: i32,
        tokens: i32,
        numerical_status: *mut c_void,
        stream: *mut c_void,
    ) -> u32;
    fn logismos_launch_q4_k_row_gemv_f32_checked(
        matrix: *const c_void,
        activations: *const c_void,
        output: *mut c_void,
        rows: i32,
        width: i32,
        tokens: i32,
        numerical_status: *mut c_void,
        stream: *mut c_void,
    ) -> u32;
    fn logismos_launch_q5_k_row_gemv_f32_checked(
        matrix: *const c_void,
        activations: *const c_void,
        output: *mut c_void,
        rows: i32,
        width: i32,
        tokens: i32,
        numerical_status: *mut c_void,
        stream: *mut c_void,
    ) -> u32;
    fn logismos_launch_q6_k_row_gemv_f32_checked(
        matrix: *const c_void,
        activations: *const c_void,
        output: *mut c_void,
        rows: i32,
        width: i32,
        tokens: i32,
        numerical_status: *mut c_void,
        stream: *mut c_void,
    ) -> u32;
    fn logismos_launch_iq4_nl_row_gemv_f32_checked(
        matrix: *const c_void,
        activations: *const c_void,
        output: *mut c_void,
        rows: i32,
        width: i32,
        tokens: i32,
        numerical_status: *mut c_void,
        stream: *mut c_void,
    ) -> u32;
    fn logismos_launch_iq4_xs_row_gemv_f32_checked(
        matrix: *const c_void,
        activations: *const c_void,
        output: *mut c_void,
        rows: i32,
        width: i32,
        tokens: i32,
        numerical_status: *mut c_void,
        stream: *mut c_void,
    ) -> u32;
    fn logismos_launch_f32_row_decode_f32_checked(
        matrix: *const c_void,
        output: *mut c_void,
        row: i32,
        width: i32,
        numerical_status: *mut c_void,
        stream: *mut c_void,
    ) -> u32;
    fn logismos_launch_q8_0_row_decode_f32_checked(
        matrix: *const c_void,
        output: *mut c_void,
        row: i32,
        width: i32,
        numerical_status: *mut c_void,
        stream: *mut c_void,
    ) -> u32;
    fn logismos_launch_q4_k_row_decode_f32_checked(
        matrix: *const c_void,
        output: *mut c_void,
        row: i32,
        width: i32,
        numerical_status: *mut c_void,
        stream: *mut c_void,
    ) -> u32;
    fn logismos_launch_q5_k_row_decode_f32_checked(
        matrix: *const c_void,
        output: *mut c_void,
        row: i32,
        width: i32,
        numerical_status: *mut c_void,
        stream: *mut c_void,
    ) -> u32;
    fn logismos_launch_q6_k_row_decode_f32_checked(
        matrix: *const c_void,
        output: *mut c_void,
        row: i32,
        width: i32,
        numerical_status: *mut c_void,
        stream: *mut c_void,
    ) -> u32;
    fn logismos_launch_iq4_nl_row_decode_f32_checked(
        matrix: *const c_void,
        output: *mut c_void,
        row: i32,
        width: i32,
        numerical_status: *mut c_void,
        stream: *mut c_void,
    ) -> u32;
    fn logismos_launch_iq4_xs_row_decode_f32_checked(
        matrix: *const c_void,
        output: *mut c_void,
        row: i32,
        width: i32,
        numerical_status: *mut c_void,
        stream: *mut c_void,
    ) -> u32;
}

/// Validated serialized matrix/vector extents shared by CPU execution and HIP launch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

/// Checked `[tokens, width]` to `[tokens, rows]` geometry for one matrix shape.
///
/// The source [`RowGemvShape`] remains the only owner of serialized matrix
/// layout and one-token projection dimensions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowGemvBatchPlan {
    shape: RowGemvShape,
    tokens: usize,
    input_elements: usize,
    output_elements: usize,
    tokens_i32: i32,
}

impl RowGemvBatchPlan {
    /// Derive and validate exact dense input/output extents for `token_count`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::UnsupportedShape`] when token count is zero,
    /// extents overflow, supplied spans differ from the derived extents, or a
    /// launch dimension cannot cross the HIP ABI.
    pub fn try_from_shape(
        shape: RowGemvShape,
        token_count: usize,
        input_elements: usize,
        output_elements: usize,
    ) -> Result<Self> {
        if token_count == 0 {
            return unsupported_shape("token count must be positive");
        }
        let derived_input = token_count.checked_mul(shape.width).ok_or_else(|| {
            UnsupportedShapeSnafu {
                kernel: KERNEL,
                msg: format!(
                    "token count * activation width overflows usize ({token_count} * {})",
                    shape.width
                ),
            }
            .build()
        })?;
        let derived_output = token_count.checked_mul(shape.rows).ok_or_else(|| {
            UnsupportedShapeSnafu {
                kernel: KERNEL,
                msg: format!(
                    "token count * output rows overflows usize ({token_count} * {})",
                    shape.rows
                ),
            }
            .build()
        })?;
        checked_layout::<f32>(derived_input, "batched activation length")?;
        checked_layout::<f32>(derived_output, "batched output length")?;
        let blocks = derived_output
            .checked_add(ROW_GEMV_THREADS_PER_BLOCK - 1)
            .ok_or_else(|| {
                UnsupportedShapeSnafu {
                    kernel: KERNEL,
                    msg: "batched projection launch blocks overflow usize".to_string(),
                }
                .build()
            })?
            / ROW_GEMV_THREADS_PER_BLOCK;
        u32::try_from(blocks).map_err(|_| abi_error("batched projection grid blocks", blocks))?;
        if input_elements != derived_input {
            return unsupported_shape(format!(
                "batched activation length {input_elements} must equal {derived_input}"
            ));
        }
        if output_elements != derived_output {
            return unsupported_shape(format!(
                "batched output length {output_elements} must equal {derived_output}"
            ));
        }
        let tokens_i32 =
            i32::try_from(token_count).map_err(|_| abi_error("token count", token_count))?;
        Ok(Self {
            shape,
            tokens: token_count,
            input_elements: derived_input,
            output_elements: derived_output,
            tokens_i32,
        })
    }

    /// Return the sole serialized matrix-shape authority.
    #[must_use]
    pub const fn shape(self) -> RowGemvShape {
        self.shape
    }

    /// Return the number of independent input rows.
    #[must_use]
    pub const fn tokens(self) -> usize {
        self.tokens
    }

    /// Return the exact dense input span in f32 elements.
    #[must_use]
    pub const fn input_elements(self) -> usize {
        self.input_elements
    }

    /// Return the exact dense output span in f32 elements.
    #[must_use]
    pub const fn output_elements(self) -> usize {
        self.output_elements
    }
}

/// Checked selected-row lookup geometry derived from one serialized matrix shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowDecodePlan {
    shape: RowGemvShape,
    row: usize,
    row_offset: usize,
    row_i32: i32,
    width_i32: i32,
}

impl RowDecodePlan {
    /// Select one stored row from an already-admitted serialized matrix shape.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::UnsupportedShape`] when `row` lies outside the
    /// checked matrix or cannot cross the HIP ABI. The source shape remains
    /// the sole owner of format, matrix extent, row bytes, and output width.
    pub fn try_from_shape(shape: RowGemvShape, row: usize) -> Result<Self> {
        if row >= shape.rows {
            return unsupported_shape(format!(
                "selected row {row} must be smaller than checked row count {}",
                shape.rows
            ));
        }
        let row_offset = row.checked_mul(shape.row_bytes).ok_or_else(|| {
            UnsupportedShapeSnafu {
                kernel: KERNEL,
                msg: format!(
                    "selected row offset overflows usize ({row} * {})",
                    shape.row_bytes
                ),
            }
            .build()
        })?;
        let row_end = row_offset.checked_add(shape.row_bytes).ok_or_else(|| {
            UnsupportedShapeSnafu {
                kernel: KERNEL,
                msg: "selected row end overflows usize".to_owned(),
            }
            .build()
        })?;
        if row_end > shape.matrix_bytes {
            return unsupported_shape("selected row exceeds the checked matrix extent");
        }
        Ok(Self {
            shape,
            row,
            row_offset,
            row_i32: i32::try_from(row).map_err(|_| abi_error("selected row", row))?,
            width_i32: i32::try_from(shape.width).map_err(|_| abi_error("width", shape.width))?,
        })
    }

    /// Return the sole source matrix-shape authority.
    #[must_use]
    pub const fn shape(self) -> RowGemvShape {
        self.shape
    }

    /// Return the selected stored row index.
    #[must_use]
    pub const fn row(self) -> usize {
        self.row
    }

    /// Return the checked serialized-byte offset of the selected row.
    #[must_use]
    pub const fn row_offset(self) -> usize {
        self.row_offset
    }

    /// Return the exact f32 output extent.
    #[must_use]
    pub const fn output_elements(self) -> usize {
        self.shape.width
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
/// batched GEMM or whole-model execution paths.
///
/// # Errors
///
/// Returns [`crate::Error::UnsupportedShape`] for invalid geometry, device
/// spans, overlap, layouts, or ABI widths; [`crate::Error::NoGpuBuild`] for a
/// CPU-only build; and HIP launch errors after submission.
///
/// # Safety
///
/// `matrix`, `activations`, and `output` must identify valid device spans on
/// `stream`'s device whose supplied lengths exactly equal the checked `shape`
/// extents. Inputs must remain immutable through stream completion; `output`
/// requires exclusive access through completion. Every span must remain live,
/// correctly aligned, and non-overlapping with the writable output.
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
    let batch = RowGemvBatchPlan::try_from_shape(shape, 1, activation_len, output_len)?;
    // SAFETY: the raw caller retains the documented allocation and numerical obligations.
    unsafe {
        launch_row_gemv_f32_rows_impl(
            batch,
            matrix,
            matrix_bytes,
            activations,
            activation_len,
            output,
            output_len,
            stream,
            None,
        )
    }
}

/// Launch one serialized row-major matrix-vector product while recording
/// explicit native numerical-domain failures in `status`.
///
/// The checked path preserves the raw launcher's format-specific decode and
/// serial f32 operation order. It records original f32/fp16 inputs and every
/// explicit reconstruction, product, and accumulation result; it does not
/// qualify hidden math-library internals or physical-device denormal behavior.
///
/// # Errors
///
/// Returns the same pre-submission and launch failures as
/// [`launch_row_gemv_f32`]. Numerical failures are sticky and are reported
/// when the synchronized status owner is read.
///
/// # Safety
///
/// The matrix, activation, output, and stream obligations match
/// [`launch_row_gemv_f32`], except that numerical classification is performed
/// by the checked kernel. `status` must be a distinct allocation on the same
/// stream device and remain live through completion. The selected device,
/// emitted f32 mode, and hidden device-math behavior remain caller-qualified.
#[cfg(feature = "gpu")]
pub unsafe fn launch_row_gemv_f32_checked(
    shape: RowGemvShape,
    matrix: *const u8,
    matrix_bytes: usize,
    activations: *const f32,
    activation_len: usize,
    output: *mut f32,
    output_len: usize,
    stream: &Stream,
    status: &NativeNumericalStatus,
) -> Result<()> {
    let batch = RowGemvBatchPlan::try_from_shape(shape, 1, activation_len, output_len)?;
    // SAFETY: the checked caller retains all device and status allocation obligations.
    unsafe {
        launch_row_gemv_f32_rows_impl(
            batch,
            matrix,
            matrix_bytes,
            activations,
            activation_len,
            output,
            output_len,
            stream,
            Some(status),
        )
    }
}

/// Launch a checked batched serialized row-major projection on `stream`.
///
/// Each `(token, output row)` preserves the format owner's original serial
/// decode, product, and accumulation order. This is a correctness primitive,
/// not a batched GEMM path.
///
/// # Safety
///
/// `matrix`, `activations`, and `output` must identify the exact distinct
/// device spans admitted by `batch` on `stream`'s device and remain live
/// through completion. `status` is a distinct live allocation on that device.
#[cfg(feature = "gpu")]
pub unsafe fn launch_row_gemv_f32_rows_checked(
    batch: RowGemvBatchPlan,
    matrix: *const u8,
    matrix_bytes: usize,
    activations: *const f32,
    activation_len: usize,
    output: *mut f32,
    output_len: usize,
    stream: &Stream,
    status: &NativeNumericalStatus,
) -> Result<()> {
    // SAFETY: the caller retains the documented checked device and status obligations.
    unsafe {
        launch_row_gemv_f32_rows_impl(
            batch,
            matrix,
            matrix_bytes,
            activations,
            activation_len,
            output,
            output_len,
            stream,
            Some(status),
        )
    }
}

#[cfg(feature = "gpu")]
#[expect(
    clippy::too_many_arguments,
    reason = "one launch owner validates both raw and checked serialized-row GEMV calls"
)]
unsafe fn launch_row_gemv_f32_rows_impl(
    batch: RowGemvBatchPlan,
    matrix: *const u8,
    matrix_bytes: usize,
    activations: *const f32,
    activation_len: usize,
    output: *mut f32,
    output_len: usize,
    stream: &Stream,
    status: Option<&NativeNumericalStatus>,
) -> Result<()> {
    #[cfg(logismos_no_gpu_kernels)]
    {
        let _ = (
            batch,
            matrix,
            matrix_bytes,
            activations,
            activation_len,
            output,
            output_len,
            stream,
            status,
        );
        no_gpu_build_refusal()
    }

    #[cfg(not(logismos_no_gpu_kernels))]
    {
        validate_device_buffers(
            batch,
            matrix,
            matrix_bytes,
            activations,
            activation_len,
            output,
            output_len,
        )?;
        let rows =
            i32::try_from(batch.shape.rows).map_err(|_| abi_error("rows", batch.shape.rows))?;
        let width =
            i32::try_from(batch.shape.width).map_err(|_| abi_error("width", batch.shape.width))?;
        let numerical_status = match status {
            Some(status) => {
                // SAFETY: the caller retains this distinct status on the stream device.
                unsafe { status.as_device_ptr().cast::<c_void>() }
            }
            None => core::ptr::null_mut(),
        };
        stream.make_current()?;
        // SAFETY: caller and validator establish the documented device spans,
        // ownership, alignment, and raw-or-checked numerical contract before
        // the sole private format dispatch.
        let code = unsafe {
            launch_format(
                batch.shape.format,
                matrix.cast::<c_void>(),
                activations.cast::<c_void>(),
                output.cast::<c_void>(),
                rows,
                width,
                batch.tokens_i32,
                numerical_status,
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

#[cfg(feature = "gpu")]
/// Decode one selected serialized row into a distinct dense f32 device buffer.
///
/// This is a direct token-to-logits lookup primitive, not a basis-vector GEMV:
/// the checked plan's stored row selects the sole decoded output.
///
/// # Errors
///
/// Returns typed exact-span, layout, selected-row, CPU-only, stream, or HIP
/// launch failures.
///
/// # Safety
///
/// `matrix` and `output` must identify live exact device spans on `stream`'s
/// device through completion. The serialized matrix remains immutable and the
/// output remains exclusively writable and disjoint from it. Every decoded
/// scale, reconstructed value, and output must be finite and normal-or-zero;
/// the device ABI has no numerical status channel.
pub unsafe fn launch_row_decode_f32(
    plan: RowDecodePlan,
    matrix: *const u8,
    matrix_bytes: usize,
    output: *mut f32,
    output_len: usize,
    stream: &Stream,
) -> Result<()> {
    // SAFETY: the raw caller retains the documented allocation and numerical obligations.
    unsafe {
        launch_row_decode_f32_impl(plan, matrix, matrix_bytes, output, output_len, stream, None)
    }
}

/// Decode one selected serialized row while recording every explicit native
/// input and reconstruction failure in `status`.
///
/// # Errors
///
/// Returns the same pre-submission and launch failures as
/// [`launch_row_decode_f32`]. Numerical failures are reported only after the
/// synchronized status allocation is read.
///
/// # Safety
///
/// The matrix, output, and stream obligations match
/// [`launch_row_decode_f32`], except that the checked kernel owns explicit
/// numerical classification. `status` must be a distinct allocation on the
/// same stream device and remain live through completion. Physical-device
/// f32-mode and hidden device conversion behavior remain caller-qualified.
#[cfg(feature = "gpu")]
pub unsafe fn launch_row_decode_f32_checked(
    plan: RowDecodePlan,
    matrix: *const u8,
    matrix_bytes: usize,
    output: *mut f32,
    output_len: usize,
    stream: &Stream,
    status: &NativeNumericalStatus,
) -> Result<()> {
    // SAFETY: the checked caller retains all device and status allocation obligations.
    unsafe {
        launch_row_decode_f32_impl(
            plan,
            matrix,
            matrix_bytes,
            output,
            output_len,
            stream,
            Some(status),
        )
    }
}

#[cfg(feature = "gpu")]
unsafe fn launch_row_decode_f32_impl(
    plan: RowDecodePlan,
    matrix: *const u8,
    matrix_bytes: usize,
    output: *mut f32,
    output_len: usize,
    stream: &Stream,
    status: Option<&NativeNumericalStatus>,
) -> Result<()> {
    #[cfg(logismos_no_gpu_kernels)]
    {
        let _ = (
            plan,
            matrix,
            matrix_bytes,
            output,
            output_len,
            stream,
            status,
        );
        no_gpu_build_refusal()
    }
    #[cfg(not(logismos_no_gpu_kernels))]
    {
        validate_decode_buffers(plan, matrix, matrix_bytes, output, output_len)?;
        let numerical_status = match status {
            Some(status) => {
                // SAFETY: the caller retains this distinct status on the stream device.
                unsafe { status.as_device_ptr().cast::<c_void>() }
            }
            None => core::ptr::null_mut(),
        };
        stream.make_current()?;
        // SAFETY: the plan, exact spans, and raw-or-checked numerical contract
        // establish the sole private format dispatch.
        let code = unsafe {
            launch_decode_format(
                plan.shape.format,
                matrix.cast::<c_void>(),
                output.cast::<c_void>(),
                plan.row_i32,
                plan.width_i32,
                numerical_status,
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
    tokens: i32,
    numerical_status: *mut c_void,
    stream: *mut c_void,
) -> Result<u32> {
    // SAFETY: caller chooses the format-specific checked private entry after
    // exact span validation and retains any status allocation through completion.
    unsafe {
        match format {
            quant::RowFormat::F32 => Ok(logismos_launch_f32_row_gemv_f32_checked(
                matrix,
                activations,
                output,
                rows,
                width,
                tokens,
                numerical_status,
                stream,
            )),
            quant::RowFormat::Q8_0 => Ok(logismos_launch_q8_0_row_gemv_f32_checked(
                matrix,
                activations,
                output,
                rows,
                width,
                tokens,
                numerical_status,
                stream,
            )),
            quant::RowFormat::Q4K => Ok(logismos_launch_q4_k_row_gemv_f32_checked(
                matrix,
                activations,
                output,
                rows,
                width,
                tokens,
                numerical_status,
                stream,
            )),
            quant::RowFormat::Q5K => Ok(logismos_launch_q5_k_row_gemv_f32_checked(
                matrix,
                activations,
                output,
                rows,
                width,
                tokens,
                numerical_status,
                stream,
            )),
            quant::RowFormat::Q6K => Ok(logismos_launch_q6_k_row_gemv_f32_checked(
                matrix,
                activations,
                output,
                rows,
                width,
                tokens,
                numerical_status,
                stream,
            )),
            quant::RowFormat::IQ4NL => Ok(logismos_launch_iq4_nl_row_gemv_f32_checked(
                matrix,
                activations,
                output,
                rows,
                width,
                tokens,
                numerical_status,
                stream,
            )),
            quant::RowFormat::IQ4XS => Ok(logismos_launch_iq4_xs_row_gemv_f32_checked(
                matrix,
                activations,
                output,
                rows,
                width,
                tokens,
                numerical_status,
                stream,
            )),
            _ => unsupported_shape(format!("native HIP row GEMV does not support {format}")),
        }
    }
}

#[cfg(all(feature = "gpu", not(logismos_no_gpu_kernels)))]
unsafe fn launch_decode_format(
    format: quant::RowFormat,
    matrix: *const c_void,
    output: *mut c_void,
    row: i32,
    width: i32,
    numerical_status: *mut c_void,
    stream: *mut c_void,
) -> Result<u32> {
    // SAFETY: caller selected the checked private format owner and retains any
    // status allocation with the exact matrix/output spans through completion.
    unsafe {
        match format {
            quant::RowFormat::F32 => Ok(logismos_launch_f32_row_decode_f32_checked(
                matrix,
                output,
                row,
                width,
                numerical_status,
                stream,
            )),
            quant::RowFormat::Q8_0 => Ok(logismos_launch_q8_0_row_decode_f32_checked(
                matrix,
                output,
                row,
                width,
                numerical_status,
                stream,
            )),
            quant::RowFormat::Q4K => Ok(logismos_launch_q4_k_row_decode_f32_checked(
                matrix,
                output,
                row,
                width,
                numerical_status,
                stream,
            )),
            quant::RowFormat::Q5K => Ok(logismos_launch_q5_k_row_decode_f32_checked(
                matrix,
                output,
                row,
                width,
                numerical_status,
                stream,
            )),
            quant::RowFormat::Q6K => Ok(logismos_launch_q6_k_row_decode_f32_checked(
                matrix,
                output,
                row,
                width,
                numerical_status,
                stream,
            )),
            quant::RowFormat::IQ4NL => Ok(logismos_launch_iq4_nl_row_decode_f32_checked(
                matrix,
                output,
                row,
                width,
                numerical_status,
                stream,
            )),
            quant::RowFormat::IQ4XS => Ok(logismos_launch_iq4_xs_row_decode_f32_checked(
                matrix,
                output,
                row,
                width,
                numerical_status,
                stream,
            )),
            _ => unsupported_shape(format!("native HIP row decode does not support {format}")),
        }
    }
}

#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
fn validate_device_buffers(
    batch: RowGemvBatchPlan,
    matrix: *const u8,
    matrix_bytes: usize,
    activations: *const f32,
    activation_len: usize,
    output: *mut f32,
    output_len: usize,
) -> Result<()> {
    if matrix_bytes != batch.shape.matrix_bytes {
        return unsupported_shape(format!(
            "matrix byte length {matrix_bytes} must equal checked {}",
            batch.shape.matrix_bytes
        ));
    }
    if activation_len != batch.input_elements {
        return unsupported_shape(format!(
            "activation length {activation_len} must equal checked {}",
            batch.input_elements
        ));
    }
    if output_len != batch.output_elements {
        return unsupported_shape(format!(
            "output length {output_len} must equal checked {}",
            batch.output_elements
        ));
    }
    let matrix = checked_u8_device_span(KERNEL, matrix, batch.shape.matrix_bytes, "matrix")?;
    let activations =
        checked_f32_device_span(KERNEL, activations, batch.input_elements, "activations")?;
    let output =
        checked_f32_device_span(KERNEL, output.cast_const(), batch.output_elements, "output")?;
    reject_overlapping_device_spans(KERNEL, output, matrix)?;
    reject_overlapping_device_spans(KERNEL, output, activations)
}

#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
fn validate_decode_buffers(
    plan: RowDecodePlan,
    matrix: *const u8,
    matrix_bytes: usize,
    output: *mut f32,
    output_len: usize,
) -> Result<()> {
    if matrix_bytes != plan.shape.matrix_bytes {
        return unsupported_shape(format!(
            "matrix byte length {matrix_bytes} must equal checked {}",
            plan.shape.matrix_bytes
        ));
    }
    if output_len != plan.shape.width {
        return unsupported_shape(format!(
            "output length {output_len} must equal checked {}",
            plan.shape.width
        ));
    }
    let matrix = checked_u8_device_span(KERNEL, matrix, plan.shape.matrix_bytes, "matrix")?;
    let output = checked_f32_device_span(
        KERNEL,
        output.cast_const(),
        plan.shape.width,
        "decoded row output",
    )?;
    reject_overlapping_device_spans(KERNEL, output, matrix)
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
    use super::{KERNEL, RowDecodePlan, RowGemvBatchPlan, RowGemvShape};
    use crate::Error;
    use crate::numerical_status::NativeNumericalStatusCategory;

    #[test]
    fn batch_plan_derives_t1_t2_t3_extents_from_one_matrix_shape()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let shape = RowGemvShape::new(quant::RowFormat::F32, 2, 3, 24, 3, 2)?;
        for tokens in 1..=3 {
            let batch = RowGemvBatchPlan::try_from_shape(shape, tokens, tokens * 3, tokens * 2)?;
            assert_eq!(batch.shape(), shape);
            assert_eq!(batch.tokens(), tokens);
            assert_eq!(batch.input_elements(), tokens * 3);
            assert_eq!(batch.output_elements(), tokens * 2);
        }
        assert!(RowGemvBatchPlan::try_from_shape(shape, 2, 5, 4).is_err());
        assert!(RowGemvBatchPlan::try_from_shape(shape, 2, 6, 3).is_err());
        assert!(RowGemvBatchPlan::try_from_shape(shape, 0, 0, 0).is_err());
        Ok(())
    }

    #[test]
    fn batch_plan_widens_representable_token_row_products_without_allocating()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let rows = usize::try_from(i32::MAX)?;
        let matrix_bytes = rows
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| std::io::Error::other("matrix bytes"))?;
        let shape = RowGemvShape::new(quant::RowFormat::F32, rows, 1, matrix_bytes, 1, rows)?;
        let outputs = rows
            .checked_mul(2)
            .ok_or_else(|| std::io::Error::other("output elements"))?;
        let batch = RowGemvBatchPlan::try_from_shape(shape, 2, 2, outputs)?;
        assert_eq!(batch.output_elements(), outputs);
        Ok(())
    }

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
    fn shape_equality_retains_format_width_and_row_identity()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let baseline = RowGemvShape::new(quant::RowFormat::F32, 1, 2, 8, 2, 1)?;
        let other_format = RowGemvShape::new(quant::RowFormat::Q8_0, 1, 32, 34, 32, 1)?;
        let other_width = RowGemvShape::new(quant::RowFormat::F32, 1, 3, 12, 3, 1)?;
        let other_rows = RowGemvShape::new(quant::RowFormat::F32, 2, 2, 16, 2, 2)?;
        assert_eq!(baseline, baseline);
        assert_ne!(baseline, other_format);
        assert_ne!(baseline, other_width);
        assert_ne!(baseline, other_rows);
        Ok(())
    }

    #[test]
    fn row_decode_plan_derives_selected_row_extent_from_one_shape_owner()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let shape = RowGemvShape::new(quant::RowFormat::F32, 3, 3, 36, 3, 3)?;
        let plan = RowDecodePlan::try_from_shape(shape, 2)?;
        assert_eq!(plan.shape(), shape);
        assert_eq!(plan.row(), 2);
        assert_eq!(plan.row_offset(), 24);
        assert_eq!(plan.output_elements(), 3);
        assert!(
            RowDecodePlan::try_from_shape(shape, 3).is_err(),
            "a selected row beyond the checked matrix must be refused"
        );
        Ok(())
    }

    #[test]
    fn selected_row_fixtures_match_independent_f64_packed_oracles()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        for format in formats() {
            let fixture = decode_fixture(format)?;
            let shape = RowGemvShape::new(
                format,
                fixture.rows,
                fixture.width,
                fixture.matrix.len(),
                fixture.width,
                fixture.rows,
            )?;
            let plan = RowDecodePlan::try_from_shape(shape, fixture.selected_row)?;
            let row =
                &fixture.matrix[plan.row_offset()..plan.row_offset() + plan.shape().row_bytes()];
            let expected = independent_decode(format, row, fixture.width)?;
            let decoded = quant::row_decode_f32(format, row, fixture.width)?;

            assert_decode_matches_f64(&decoded, &expected, &format.to_string());
            assert_eq!(
                expected.first().map(|value| value.to_bits()),
                Some(fixture.known_first.to_bits()),
                "{format} known first value"
            );
            assert_eq!(
                expected.last().map(|value| value.to_bits()),
                Some(fixture.known_last.to_bits()),
                "{format} known tail value"
            );
            assert!(
                decoded.iter().all(
                    |value| value.is_normal() || value.classify() == std::num::FpCategory::Zero
                ),
                "{format} selected row must remain in the native numerical domain"
            );
            let preceding = quant::row_decode_f32(
                format,
                &fixture.matrix[..plan.shape().row_bytes()],
                fixture.width,
            )?;
            assert!(
                preceding
                    .iter()
                    .all(|value| value.classify() == std::num::FpCategory::Zero),
                "{format} preceding row must discriminate the selected row"
            );
        }
        Ok(())
    }

    #[test]
    fn independent_status_oracle_catches_bad_inputs_and_hidden_underflow() {
        let (_, nonfinite_input) = independent_checked_f32_dot(&[f32::NAN], &[1.0]);
        assert_status_bit(
            nonfinite_input,
            NativeNumericalStatusCategory::InputNonFinite,
            "non-finite serialized f32 input",
        );

        let (_, subnormal_input) = independent_checked_f32_dot(&[1.0], &[f32::from_bits(1)]);
        assert_status_bit(
            subnormal_input,
            NativeNumericalStatusCategory::InputSubnormal,
            "subnormal activation input",
        );

        let (_, overflow) = independent_checked_f32_dot(&[f32::MAX], &[2.0]);
        assert_status_bit(
            overflow,
            NativeNumericalStatusCategory::ArithmeticNonFinite,
            "overflowing product",
        );

        let (output, hidden_underflow) =
            independent_checked_f32_dot(&[f32::MIN_POSITIVE, 1.0], &[0.5, 1.0]);
        assert_eq!(
            output.to_bits(),
            1.0_f32.to_bits(),
            "the later normal term deliberately hides the earlier subnormal product"
        );
        assert_status_bit(
            hidden_underflow,
            NativeNumericalStatusCategory::ArithmeticSubnormal,
            "hidden subnormal product",
        );
        assert_eq!(
            hidden_underflow
                & (NativeNumericalStatusCategory::InputSubnormal.bit()
                    | NativeNumericalStatusCategory::InputNonFinite.bit()),
            0,
            "the hidden-result witness must keep every original input in-domain"
        );
    }

    #[test]
    fn shape_refuses_zero_partial_and_overflowing_geometry() {
        assert!(matches!(
            RowGemvShape::new(quant::RowFormat::F32, 0, 1, 0, 1, 0),
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

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn shape_refuses_values_beyond_the_signed_hip_abi() {
        let beyond_i32 = i32::MAX as usize + 1;
        let rows_result = RowGemvShape::new(
            quant::RowFormat::F32,
            beyond_i32,
            1,
            beyond_i32 * core::mem::size_of::<f32>(),
            1,
            beyond_i32,
        );
        assert!(matches!(
            rows_result,
            Err(Error::UnsupportedShape { kernel: KERNEL, .. })
        ));
        let width_result = RowGemvShape::new(
            quant::RowFormat::F32,
            1,
            beyond_i32,
            beyond_i32 * core::mem::size_of::<f32>(),
            beyond_i32,
            1,
        );
        assert!(matches!(
            width_result,
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
        let batch = RowGemvBatchPlan::try_from_shape(shape, 1, activations.len(), output.len())?;
        super::validate_device_buffers(
            batch,
            matrix.as_ptr(),
            matrix.len(),
            activations.as_ptr(),
            activations.len(),
            output.as_mut_ptr(),
            output.len(),
        )?;
        let misaligned_matrix = [0.0_f32; 5];
        super::validate_device_buffers(
            batch,
            misaligned_matrix.as_ptr().cast::<u8>().wrapping_byte_add(1),
            matrix.len(),
            activations.as_ptr(),
            activations.len(),
            output.as_mut_ptr(),
            output.len(),
        )?;
        let batched_activations = [0.0_f32; 4];
        let mut batched_output = [0.0_f32; 4];
        let batch_two = RowGemvBatchPlan::try_from_shape(
            shape,
            2,
            batched_activations.len(),
            batched_output.len(),
        )?;
        super::validate_device_buffers(
            batch_two,
            matrix.as_ptr(),
            matrix.len(),
            batched_activations.as_ptr(),
            batched_activations.len(),
            batched_output.as_mut_ptr(),
            batched_output.len(),
        )?;
        assert!(
            super::validate_device_buffers(
                batch_two,
                matrix.as_ptr(),
                matrix.len(),
                batched_activations.as_ptr(),
                batched_activations.len() - 1,
                batched_output.as_mut_ptr(),
                batched_output.len(),
            )
            .is_err(),
            "T=2 activation span must equal the batch-derived extent"
        );
        assert!(
            super::validate_device_buffers(
                batch_two,
                matrix.as_ptr(),
                matrix.len(),
                batched_activations.as_ptr(),
                batched_activations.len(),
                batched_output.as_mut_ptr(),
                batched_output.len() - 1,
            )
            .is_err(),
            "T=2 output span must equal the batch-derived extent"
        );
        assert!(
            super::validate_device_buffers(
                batch,
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
                batch,
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
                batch,
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
                batch,
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
                batch,
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
                batch,
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
                batch,
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

    #[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
    #[test]
    fn row_decode_validator_binds_exact_matrix_and_distinct_output()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let matrix = [0_u8; 24];
        let mut output = [0.0_f32; 3];
        let shape = RowGemvShape::new(quant::RowFormat::F32, 2, 3, matrix.len(), 3, 2)?;
        let plan = RowDecodePlan::try_from_shape(shape, 1)?;
        super::validate_decode_buffers(
            plan,
            matrix.as_ptr(),
            matrix.len(),
            output.as_mut_ptr(),
            output.len(),
        )?;
        assert!(
            super::validate_decode_buffers(
                plan,
                matrix.as_ptr(),
                matrix.len() - 1,
                output.as_mut_ptr(),
                output.len(),
            )
            .is_err(),
            "short serialized matrix must be refused"
        );
        assert!(
            super::validate_decode_buffers(
                plan,
                matrix.as_ptr(),
                matrix.len(),
                output.as_mut_ptr(),
                output.len() - 1,
            )
            .is_err(),
            "short decoded output must be refused"
        );
        assert!(
            super::validate_decode_buffers(
                plan,
                core::ptr::null(),
                matrix.len(),
                output.as_mut_ptr(),
                output.len(),
            )
            .is_err(),
            "null serialized matrix must be refused"
        );
        assert!(
            super::validate_decode_buffers(
                plan,
                matrix.as_ptr(),
                matrix.len(),
                output.as_mut_ptr().wrapping_byte_add(1),
                output.len(),
            )
            .is_err(),
            "misaligned decoded output must be refused"
        );
        let mut aligned_matrix = [0.0_f32; 6];
        assert!(
            super::validate_decode_buffers(
                plan,
                aligned_matrix.as_ptr().cast::<u8>(),
                matrix.len(),
                aligned_matrix.as_mut_ptr(),
                output.len(),
            )
            .is_err(),
            "decoded output must not alias serialized matrix storage"
        );
        Ok(())
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "requires an explicitly reserved HIP device; absent devices are a failure"]
    fn reserved_device_decodes_selected_row_for_all_formats_and_preserves_tail()
    -> core::result::Result<(), String> {
        use hipcore::{Device, DeviceBuffer, Stream};

        const SENTINEL: f32 = -1234.5;
        const CANARY_VALUES: usize = 17;

        let device = Device::new(0).map_err(|error| format!("open reserved device 0: {error}"))?;
        let stream = Stream::new(&device).map_err(|error| format!("create stream: {error}"))?;
        for format in formats() {
            let fixture = decode_fixture(format).map_err(|error| error.to_string())?;
            let shape = RowGemvShape::new(
                format,
                fixture.rows,
                fixture.width,
                fixture.matrix.len(),
                fixture.width,
                fixture.rows,
            )
            .map_err(|error| format!("validate {format} shape: {error}"))?;
            let plan = RowDecodePlan::try_from_shape(shape, fixture.selected_row)
                .map_err(|error| format!("select {format} row: {error}"))?;
            let row =
                &fixture.matrix[plan.row_offset()..plan.row_offset() + plan.shape().row_bytes()];
            let expected = independent_decode(format, row, fixture.width)
                .map_err(|error| format!("derive independent {format} result: {error}"))?;
            let owner_decoded = quant::row_decode_f32(format, row, fixture.width)
                .map_err(|error| format!("decode {format} with quant owner: {error}"))?;
            assert_decode_matches_f64(&owner_decoded, &expected, &format.to_string());

            let mut padded_matrix = Vec::with_capacity(fixture.matrix.len() + 1);
            padded_matrix.push(0xa5);
            padded_matrix.extend_from_slice(&fixture.matrix);
            let matrix = DeviceBuffer::<u8>::from_host(&device, &padded_matrix)
                .map_err(|error| format!("upload unaligned {format} matrix: {error}"))?;
            let initialized_output = vec![SENTINEL; fixture.width + CANARY_VALUES];
            let output = DeviceBuffer::<f32>::from_host(&device, &initialized_output)
                .map_err(|error| format!("allocate {format} output and tail: {error}"))?;
            let status = crate::numerical_status::NativeNumericalStatus::new(&device)
                .map_err(|error| format!("allocate {format} numerical status: {error}"))?;
            // SAFETY: the one-byte offset deliberately exercises byte-unaligned
            // little-endian weights. All three distinct allocations remain live;
            // matrix stays immutable and output/status stay exclusive through
            // synchronization.
            unsafe {
                super::launch_row_decode_f32_checked(
                    plan,
                    matrix.as_device_ptr().wrapping_add(1),
                    fixture.matrix.len(),
                    output.as_device_ptr(),
                    fixture.width,
                    &stream,
                    &status,
                )
            }
            .map_err(|error| format!("launch {format} selected-row decode: {error}"))?;
            stream
                .synchronize()
                .map_err(|error| format!("synchronize {format} selected-row decode: {error}"))?;
            status
                .read_after_synchronization()
                .map_err(|error| format!("read clean {format} numerical status: {error}"))?;

            let mut actual = vec![f32::NAN; output.len()];
            output
                .copy_to_host(&mut actual)
                .map_err(|error| format!("read {format} decoded row: {error}"))?;
            assert_decode_matches_f64(&actual[..fixture.width], &expected, &format.to_string());
            if actual[fixture.width..]
                .iter()
                .any(|value| value.to_bits() != SENTINEL.to_bits())
            {
                return Err(format!("{format} launch modified the output canary tail"));
            }
            let mut matrix_after = vec![0_u8; matrix.len()];
            matrix
                .copy_to_host(&mut matrix_after)
                .map_err(|error| format!("read {format} matrix: {error}"))?;
            if matrix_after != padded_matrix {
                return Err(format!("{format} launch modified immutable matrix bytes"));
            }
        }
        Ok(())
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "requires an explicitly reserved HIP device; absent devices are a failure"]
    fn reserved_device_checked_gemv_refuses_hidden_and_serialized_input_failures()
    -> core::result::Result<(), String> {
        use hipcore::{Device, DeviceBuffer, Stream};

        let device = Device::new(0).map_err(|error| format!("open reserved device 0: {error}"))?;
        let stream = Stream::new(&device).map_err(|error| format!("create stream: {error}"))?;

        let hidden_matrix = [f32::MIN_POSITIVE, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let hidden_activations = [0.5_f32, 1.0];
        let hidden_shape = RowGemvShape::new(
            quant::RowFormat::F32,
            1,
            hidden_activations.len(),
            hidden_matrix.len(),
            hidden_activations.len(),
            1,
        )
        .map_err(|error| format!("validate hidden-result shape: {error}"))?;
        let hidden_matrix = DeviceBuffer::<u8>::from_host(&device, &hidden_matrix)
            .map_err(|error| format!("upload hidden-result matrix: {error}"))?;
        let hidden_activations = DeviceBuffer::<f32>::from_host(&device, &hidden_activations)
            .map_err(|error| format!("upload hidden-result activations: {error}"))?;
        let hidden_output = DeviceBuffer::<f32>::from_host(&device, &[f32::NAN])
            .map_err(|error| format!("allocate hidden-result output: {error}"))?;
        let hidden_status = crate::numerical_status::NativeNumericalStatus::new(&device)
            .map_err(|error| format!("allocate hidden-result status: {error}"))?;
        // SAFETY: exact distinct allocations remain immutable/exclusive and live
        // with the status owner through synchronization.
        unsafe {
            super::launch_row_gemv_f32_checked(
                hidden_shape,
                hidden_matrix.as_device_ptr(),
                hidden_matrix.len(),
                hidden_activations.as_device_ptr(),
                hidden_activations.len(),
                hidden_output.as_device_ptr(),
                hidden_output.len(),
                &stream,
                &hidden_status,
            )
        }
        .map_err(|error| format!("launch hidden-result GEMV: {error}"))?;
        stream
            .synchronize()
            .map_err(|error| format!("synchronize hidden-result GEMV: {error}"))?;
        let mut hidden_actual = [f32::NAN];
        hidden_output
            .copy_to_host(&mut hidden_actual)
            .map_err(|error| format!("read hidden-result output: {error}"))?;
        if hidden_actual[0].to_bits() != 1.0_f32.to_bits() {
            return Err(format!(
                "hidden-result output must finish normal: got {}",
                hidden_actual[0]
            ));
        }
        require_observed_status(
            &hidden_status,
            NativeNumericalStatusCategory::ArithmeticSubnormal,
            "hidden subnormal product",
        )?;

        let mut packed_matrix = known_q8_block();
        packed_matrix[..2].copy_from_slice(&1_u16.to_le_bytes());
        let packed_activations = [1.0_f32; 32];
        let packed_shape = RowGemvShape::new(
            quant::RowFormat::Q8_0,
            1,
            packed_activations.len(),
            packed_matrix.len(),
            packed_activations.len(),
            1,
        )
        .map_err(|error| format!("validate subnormal-f16 shape: {error}"))?;
        let packed_matrix = DeviceBuffer::<u8>::from_host(&device, &packed_matrix)
            .map_err(|error| format!("upload subnormal-f16 matrix: {error}"))?;
        let packed_activations = DeviceBuffer::<f32>::from_host(&device, &packed_activations)
            .map_err(|error| format!("upload subnormal-f16 activations: {error}"))?;
        let packed_output = DeviceBuffer::<f32>::from_host(&device, &[f32::NAN])
            .map_err(|error| format!("allocate subnormal-f16 output: {error}"))?;
        let packed_status = crate::numerical_status::NativeNumericalStatus::new(&device)
            .map_err(|error| format!("allocate subnormal-f16 status: {error}"))?;
        // SAFETY: exact distinct allocations remain immutable/exclusive and live
        // with the status owner through synchronization.
        unsafe {
            super::launch_row_gemv_f32_checked(
                packed_shape,
                packed_matrix.as_device_ptr(),
                packed_matrix.len(),
                packed_activations.as_device_ptr(),
                packed_activations.len(),
                packed_output.as_device_ptr(),
                packed_output.len(),
                &stream,
                &packed_status,
            )
        }
        .map_err(|error| format!("launch subnormal-f16 GEMV: {error}"))?;
        stream
            .synchronize()
            .map_err(|error| format!("synchronize subnormal-f16 GEMV: {error}"))?;
        require_observed_status(
            &packed_status,
            NativeNumericalStatusCategory::InputSubnormal,
            "serialized fp16 subnormal",
        )
    }

    #[cfg(all(feature = "gpu", logismos_no_gpu_kernels))]
    #[test]
    fn cpu_only_build_refuses_gpu_launch_without_a_device() {
        assert!(matches!(
            super::no_gpu_build_refusal(),
            Err(Error::NoGpuBuild { .. })
        ));
    }

    fn formats() -> [quant::RowFormat; 7] {
        [
            quant::RowFormat::F32,
            quant::RowFormat::Q8_0,
            quant::RowFormat::Q4K,
            quant::RowFormat::Q5K,
            quant::RowFormat::Q6K,
            quant::RowFormat::IQ4NL,
            quant::RowFormat::IQ4XS,
        ]
    }

    struct DecodeFixture {
        rows: usize,
        width: usize,
        selected_row: usize,
        matrix: Vec<u8>,
        known_first: f64,
        known_last: f64,
    }

    fn decode_fixture(
        format: quant::RowFormat,
    ) -> core::result::Result<DecodeFixture, Box<dyn std::error::Error>> {
        let (width, selected, known_first, known_last) = match format {
            quant::RowFormat::F32 => (263, known_f32_row(263), -3.5, 0.5),
            quant::RowFormat::Q8_0 => (288, repeat_block(known_q8_block(), 9), -8.0, 7.5),
            quant::RowFormat::Q4K => (512, repeat_block(known_q4_block(), 2), 0.5, 662.5),
            quant::RowFormat::Q5K => (512, repeat_block(known_q5_block(), 2), 16.5, 1670.5),
            quant::RowFormat::Q6K => (512, repeat_block(known_q6_block(), 2), -31.0, -27.0),
            quant::RowFormat::IQ4NL => (288, repeat_block(known_iq4_nl_block(), 9), -63.5, 56.5),
            quant::RowFormat::IQ4XS => (512, repeat_block(known_iq4_xs_block(), 2), -63.5, -1751.5),
            _ => return Err("unknown executable row format".into()),
        };
        let row_bytes = quant::row_byte_len(format, width)?;
        if selected.len() != row_bytes {
            return Err(format!(
                "independent {format} fixture has {} bytes, owner requires {row_bytes}",
                selected.len()
            )
            .into());
        }
        let rows = 3;
        let selected_row = 1;
        let mut matrix = vec![0_u8; row_bytes];
        matrix.extend(selected);
        matrix.extend(vec![0_u8; row_bytes]);
        Ok(DecodeFixture {
            rows,
            width,
            selected_row,
            matrix,
            known_first,
            known_last,
        })
    }

    fn repeat_block(block: Vec<u8>, count: usize) -> Vec<u8> {
        block.repeat(count)
    }

    fn known_f32_row(width: usize) -> Vec<u8> {
        const VALUES: [f32; 5] = [-3.5, -1.25, 0.5, 2.0, 4.0];
        (0..width)
            .flat_map(|index| VALUES[index % VALUES.len()].to_le_bytes())
            .collect()
    }

    fn known_q8_block() -> Vec<u8> {
        const VALUES: [i8; 32] = [
            -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4,
            5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        ];
        let mut block = 0x3800_u16.to_le_bytes().to_vec();
        block.extend(VALUES.into_iter().flat_map(i8::to_le_bytes));
        block
    }

    fn known_k_scales() -> [u8; 12] {
        let scales = [1_u8, 2, 3, 4, 17, 34, 51, 63];
        let minima = [5_u8, 6, 7, 8, 18, 35, 52, 61];
        let mut packed = [0_u8; 12];
        for group in 0..4 {
            packed[group] = scales[group] | ((scales[group + 4] >> 4) << 6);
            packed[group + 4] = minima[group] | ((minima[group + 4] >> 4) << 6);
            packed[group + 8] = (scales[group + 4] & 0x0f) | ((minima[group + 4] & 0x0f) << 4);
        }
        packed
    }

    fn known_q4_block() -> Vec<u8> {
        let mut block = vec![0_u8; 144];
        block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        block[2..4].copy_from_slice(&0x3800_u16.to_le_bytes());
        block[4..16].copy_from_slice(&known_k_scales());
        block[16..].fill(0xb3);
        block
    }

    fn known_q5_block() -> Vec<u8> {
        let mut block = vec![0_u8; 176];
        block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        block[2..4].copy_from_slice(&0x3800_u16.to_le_bytes());
        block[4..16].copy_from_slice(&known_k_scales());
        block[16..48].fill(0xa5);
        block[48..].fill(0xb3);
        block
    }

    fn known_q6_block() -> Vec<u8> {
        let scales = [2_i8, -1, 3, -2, 4, -3, 5, -4, 6, -5, 7, -6, 8, -7, 9, -3];
        let mut block = vec![0x21_u8; 128];
        block.extend([0xe4_u8; 64]);
        block.extend(scales.into_iter().flat_map(i8::to_le_bytes));
        block.extend(0x3800_u16.to_le_bytes());
        block
    }

    fn known_iq4_nl_block() -> Vec<u8> {
        let mut block = 0x3800_u16.to_le_bytes().to_vec();
        block.extend([0xf0_u8; 16]);
        block
    }

    fn known_iq4_xs_block() -> Vec<u8> {
        let raw_scales = [33_u8, 34, 35, 36, 49, 27, 63, 1];
        let mut scale_high = 0_u16;
        let mut scale_low = [0_u8; 4];
        for (group, raw) in raw_scales.into_iter().enumerate() {
            scale_low[group / 2] |= (raw & 0x0f) << ((group % 2) * 4);
            scale_high |= u16::from(raw >> 4) << (group * 2);
        }
        let mut block = 0x3800_u16.to_le_bytes().to_vec();
        block.extend(scale_high.to_le_bytes());
        block.extend(scale_low);
        block.extend([0xf0_u8; 128]);
        block
    }

    fn independent_decode(
        format: quant::RowFormat,
        row: &[u8],
        width: usize,
    ) -> core::result::Result<Vec<f64>, Box<dyn std::error::Error>> {
        (0..width)
            .map(|index| independent_weight(format, row, index))
            .collect()
    }

    fn independent_weight(
        format: quant::RowFormat,
        row: &[u8],
        index: usize,
    ) -> core::result::Result<f64, Box<dyn std::error::Error>> {
        match format {
            quant::RowFormat::F32 => {
                let offset = index * 4;
                Ok(f64::from(f32::from_le_bytes(
                    row[offset..offset + 4].try_into()?,
                )))
            }
            quant::RowFormat::Q8_0 => {
                let block = &row[index / 32 * 34..];
                Ok(independent_f16(block)? * f64::from(i8::from_le_bytes([block[2 + index % 32]])))
            }
            quant::RowFormat::Q4K => independent_q4(row, index),
            quant::RowFormat::Q5K => independent_q5(row, index),
            quant::RowFormat::Q6K => independent_q6(row, index),
            quant::RowFormat::IQ4NL => independent_iq4_nl(row, index),
            quant::RowFormat::IQ4XS => independent_iq4_xs(row, index),
            _ => Err("unknown executable row format".into()),
        }
    }

    fn independent_k_scale_min(scales: &[u8], group: usize) -> (u8, u8) {
        if group < 4 {
            (scales[group] & 0x3f, scales[group + 4] & 0x3f)
        } else {
            (
                (scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4),
                (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
            )
        }
    }

    fn independent_q4(
        row: &[u8],
        index: usize,
    ) -> core::result::Result<f64, Box<dyn std::error::Error>> {
        let block = &row[index / 256 * 144..];
        let group = index % 256 / 32;
        let lane = index % 32;
        let (scale, minimum) = independent_k_scale_min(&block[4..16], group);
        let packed = block[16 + group / 2 * 32 + lane];
        let quantized = if group.is_multiple_of(2) {
            packed & 0x0f
        } else {
            packed >> 4
        };
        Ok(
            independent_f16(block)? * f64::from(scale) * f64::from(quantized)
                - independent_f16(&block[2..])? * f64::from(minimum),
        )
    }

    fn independent_q5(
        row: &[u8],
        index: usize,
    ) -> core::result::Result<f64, Box<dyn std::error::Error>> {
        let block = &row[index / 256 * 176..];
        let group = index % 256 / 32;
        let lane = index % 32;
        let (scale, minimum) = independent_k_scale_min(&block[4..16], group);
        let packed = block[48 + group / 2 * 32 + lane];
        let fifth = if block[16 + lane] & (1 << group) == 0 {
            0
        } else {
            16
        };
        let quantized = (if group.is_multiple_of(2) {
            packed & 0x0f
        } else {
            packed >> 4
        }) + fifth;
        Ok(
            independent_f16(block)? * f64::from(scale) * f64::from(quantized)
                - independent_f16(&block[2..])? * f64::from(minimum),
        )
    }

    fn independent_q6(
        row: &[u8],
        index: usize,
    ) -> core::result::Result<f64, Box<dyn std::error::Error>> {
        let block = &row[index / 256 * 210..];
        let local = index % 256;
        let half = local / 128;
        let quarter = local % 128 / 32;
        let lane = local % 32;
        let low = block[half * 64 + (quarter % 2) * 32 + lane];
        let lower = if quarter < 2 { low & 0x0f } else { low >> 4 };
        let upper = (block[128 + half * 32 + lane] >> (quarter * 2)) & 0x03;
        let quantized = i16::from((upper << 4) | lower) - 32;
        let scale = i8::from_le_bytes([block[192 + half * 8 + quarter * 2 + lane / 16]]);
        Ok(independent_f16(&block[208..])? * f64::from(scale) * f64::from(quantized))
    }

    fn independent_iq4_nl(
        row: &[u8],
        index: usize,
    ) -> core::result::Result<f64, Box<dyn std::error::Error>> {
        let block = &row[index / 32 * 18..];
        let lane = index % 32;
        let packed = block[2 + lane % 16];
        let code = if lane < 16 {
            packed & 0x0f
        } else {
            packed >> 4
        };
        Ok(independent_f16(block)? * f64::from(independent_iq4_value(code)))
    }

    fn independent_iq4_xs(
        row: &[u8],
        index: usize,
    ) -> core::result::Result<f64, Box<dyn std::error::Error>> {
        let block = &row[index / 256 * 136..];
        let local = index % 256;
        let group = local / 32;
        let lane = local % 32;
        let low = if group.is_multiple_of(2) {
            block[4 + group / 2] & 0x0f
        } else {
            block[4 + group / 2] >> 4
        };
        let high = u8::try_from((u16::from_le_bytes([block[2], block[3]]) >> (group * 2)) & 0x03)?;
        let group_scale = i16::from(low | (high << 4)) - 32;
        let packed = block[8 + group * 16 + lane % 16];
        let code = if lane < 16 {
            packed & 0x0f
        } else {
            packed >> 4
        };
        Ok(independent_f16(block)?
            * f64::from(group_scale)
            * f64::from(independent_iq4_value(code)))
    }

    fn independent_iq4_value(index: u8) -> i8 {
        [
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        ][usize::from(index)]
    }

    fn independent_f16(bytes: &[u8]) -> core::result::Result<f64, Box<dyn std::error::Error>> {
        let bits = u16::from_le_bytes(bytes[..2].try_into()?);
        let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
        let exponent = i32::from((bits >> 10) & 0x1f);
        let fraction = u32::from(bits & 0x03ff);
        match exponent {
            0 => Ok(sign * f64::from(fraction) * 2_f64.powi(-24)),
            31 => Err("synthetic fixture contains non-finite fp16".into()),
            _ => Ok(sign * (1.0 + f64::from(fraction) / 1024.0) * 2_f64.powi(exponent - 15)),
        }
    }

    fn independent_checked_f32_dot(weights: &[f32], activations: &[f32]) -> (f32, u32) {
        assert_eq!(weights.len(), activations.len());
        let mut status = 0_u32;
        let mut accumulator = 0.0_f32;
        for (&weight, &activation) in weights.iter().zip(activations) {
            independent_record_input(weight, &mut status);
            independent_record_input(activation, &mut status);
            independent_record_result(weight, &mut status);
            independent_record_result(activation, &mut status);
            let product = weight * activation;
            independent_record_result(product, &mut status);
            independent_record_result(accumulator, &mut status);
            independent_record_result(product, &mut status);
            accumulator += product;
            independent_record_result(accumulator, &mut status);
        }
        (accumulator, status)
    }

    fn independent_record_input(value: f32, status: &mut u32) {
        match value.classify() {
            std::num::FpCategory::Subnormal => {
                *status |= NativeNumericalStatusCategory::InputSubnormal.bit();
            }
            std::num::FpCategory::Infinite | std::num::FpCategory::Nan => {
                *status |= NativeNumericalStatusCategory::InputNonFinite.bit();
            }
            std::num::FpCategory::Normal | std::num::FpCategory::Zero => {}
        }
    }

    fn independent_record_result(value: f32, status: &mut u32) {
        match value.classify() {
            std::num::FpCategory::Subnormal => {
                *status |= NativeNumericalStatusCategory::ArithmeticSubnormal.bit();
            }
            std::num::FpCategory::Infinite | std::num::FpCategory::Nan => {
                *status |= NativeNumericalStatusCategory::ArithmeticNonFinite.bit();
            }
            std::num::FpCategory::Normal | std::num::FpCategory::Zero => {}
        }
    }

    fn assert_status_bit(bits: u32, category: NativeNumericalStatusCategory, label: &str) {
        assert_ne!(bits & category.bit(), 0, "{label} must set {category:?}");
    }

    #[cfg(feature = "gpu")]
    fn require_observed_status(
        status: &crate::numerical_status::NativeNumericalStatus,
        category: NativeNumericalStatusCategory,
        label: &str,
    ) -> core::result::Result<(), String> {
        match status.read_after_synchronization() {
            Err(Error::NumericalStatus {
                source: crate::numerical_status::NativeNumericalStatusError::Observed { mask, .. },
                ..
            }) if mask.contains(category) => Ok(()),
            Err(error) => Err(format!(
                "{label} returned unexpected status failure: {error}"
            )),
            Ok(()) => Err(format!("{label} must set {category:?}")),
        }
    }

    fn assert_decode_matches_f64(actual: &[f32], expected: &[f64], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} decoded length");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                actual.is_finite() && expected.is_finite(),
                "{label} value {index} must be finite: got {actual}, expected {expected}"
            );
            // WHY: every synthetic operand is dyadic and every expected result
            // is exactly representable; this is not a global tolerance policy.
            assert_eq!(
                f64::from(*actual).to_bits(),
                expected.to_bits(),
                "{label} value {index}: got {actual}, expected {expected}"
            );
        }
    }
}
