//! Checked native f32 primitives used by the Qwen full-attention block.

#[cfg(not(logismos_no_gpu_kernels))]
use std::ffi::c_void;

use hipcore::Stream;
use num_traits::ToPrimitive;
use snafu::ResultExt;

#[cfg(any(test, not(logismos_no_gpu_kernels)))]
use crate::device_span::{checked_f32_device_span, reject_overlapping_f32_spans};
#[cfg(not(logismos_no_gpu_kernels))]
use crate::error::LaunchSnafu;
#[cfg(logismos_no_gpu_kernels)]
use crate::error::NoGpuBuildSnafu;
use crate::error::{Result, UnsupportedShapeSnafu};

const RMS_NORM_KERNEL: &str = "decoder_rms_norm_f32";
const ROTARY_KERNEL: &str = "decoder_rotary_half_split_f32";
const SPLIT_Q_GATE_KERNEL: &str = "decoder_split_q_gate_f32";
const SIGMOID_MUL_KERNEL: &str = "decoder_sigmoid_mul_f32";
const SILU_MUL_KERNEL: &str = "decoder_silu_mul_f32";
const RESIDUAL_ADD_KERNEL: &str = "decoder_residual_add_f32";
const WAVE_SIZE: usize = 32;
const ELEMENTWISE_THREADS: usize = 256;

#[cfg(not(logismos_no_gpu_kernels))]
unsafe extern "C" {
    fn logismos_launch_decoder_rms_norm_f32(
        input_f32: *const c_void,
        weight_f32: *const c_void,
        output_f32: *mut c_void,
        rows: u32,
        width: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> u32;

    fn logismos_launch_decoder_rotary_half_split_f32(
        values_f32: *mut c_void,
        cos_f32: *const c_void,
        sin_f32: *const c_void,
        heads: u32,
        width: u32,
        rotary_width: u32,
        stream: *mut c_void,
    ) -> u32;

    fn logismos_launch_decoder_split_q_gate_f32(
        q_gate_f32: *const c_void,
        query_f32: *mut c_void,
        gate_f32: *mut c_void,
        heads: u32,
        key_width: u32,
        stream: *mut c_void,
    ) -> u32;

    fn logismos_launch_decoder_sigmoid_mul_f32(
        value_f32: *const c_void,
        gate_f32: *const c_void,
        output_f32: *mut c_void,
        elements: u32,
        stream: *mut c_void,
    ) -> u32;

    fn logismos_launch_decoder_silu_mul_f32(
        gate_f32: *const c_void,
        up_f32: *const c_void,
        output_f32: *mut c_void,
        elements: u32,
        stream: *mut c_void,
    ) -> u32;

    fn logismos_launch_decoder_residual_add_f32(
        left_f32: *const c_void,
        right_f32: *const c_void,
        output_f32: *mut c_void,
        elements: u32,
        stream: *mut c_void,
    ) -> u32;
}

/// Checked native geometry for a dense f32 RMSNorm call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RmsNormF32Plan {
    rows: usize,
    width: usize,
    elements: usize,
    epsilon: f32,
    rows_u32: u32,
    width_u32: u32,
}

impl RmsNormF32Plan {
    /// Admit one positive dense `[rows, width]` f32 RMSNorm geometry.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::UnsupportedShape`] when a dimension is zero,
    /// a product or f32 layout overflows, a HIP ABI dimension cannot be
    /// represented, or `epsilon` is not positive normal finite f32.
    pub fn try_from_dimensions(rows: usize, width: usize, epsilon: f32) -> Result<Self> {
        validate_nonzero(RMS_NORM_KERNEL, "rows", rows)?;
        validate_nonzero(RMS_NORM_KERNEL, "width", width)?;
        validate_positive_normal(RMS_NORM_KERNEL, "epsilon", epsilon)?;
        let elements = checked_product(RMS_NORM_KERNEL, rows, width, "rows * width")?;
        validate_f32_layout(RMS_NORM_KERNEL, "input/output", elements)?;
        validate_f32_layout(RMS_NORM_KERNEL, "weight", width)?;
        Ok(Self {
            rows,
            width,
            elements,
            epsilon,
            rows_u32: abi_u32(RMS_NORM_KERNEL, "rows", rows)?,
            width_u32: abi_u32(RMS_NORM_KERNEL, "width", width)?,
        })
    }

    /// Return the admitted row count.
    #[must_use]
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Return the admitted row width.
    #[must_use]
    pub const fn width(self) -> usize {
        self.width
    }

    /// Return the exact input and output extent.
    #[must_use]
    pub const fn elements(self) -> usize {
        self.elements
    }

    /// Return the positive numerical epsilon used by the kernel.
    #[must_use]
    pub const fn epsilon(self) -> f32 {
        self.epsilon
    }
}

/// Checked native geometry for an in-place half-split rotary operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RotaryHalfSplitF32Plan {
    heads: usize,
    width: usize,
    rotary_width: usize,
    pairs: usize,
    elements: usize,
    heads_u32: u32,
    width_u32: u32,
    rotary_width_u32: u32,
}

impl RotaryHalfSplitF32Plan {
    /// Admit dense `[heads, width]` half-split rotation geometry.
    ///
    /// `rotary_width` must be positive, even, and no larger than `width`.
    /// Coordinates in the remaining per-head tail are intentionally preserved.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::UnsupportedShape`] for invalid dimensions,
    /// overflowed spans/layouts, an unrepresentable HIP ABI value, or a grid
    /// whose block count exceeds the HIP grid domain.
    pub fn try_from_dimensions(heads: usize, width: usize, rotary_width: usize) -> Result<Self> {
        validate_nonzero(ROTARY_KERNEL, "heads", heads)?;
        validate_nonzero(ROTARY_KERNEL, "width", width)?;
        validate_nonzero(ROTARY_KERNEL, "rotary_width", rotary_width)?;
        if !rotary_width.is_multiple_of(2) || rotary_width > width {
            return unsupported_shape(
                ROTARY_KERNEL,
                format!(
                    "rotary_width {rotary_width} must be even and no greater than width {width}"
                ),
            );
        }
        let pairs = rotary_width / 2;
        let elements = checked_product(ROTARY_KERNEL, heads, width, "heads * width")?;
        let work_items = checked_product(ROTARY_KERNEL, heads, pairs, "heads * rotary pairs")?;
        validate_f32_layout(ROTARY_KERNEL, "values", elements)?;
        validate_f32_layout(ROTARY_KERNEL, "cosine coefficients", pairs)?;
        validate_f32_layout(ROTARY_KERNEL, "sine coefficients", pairs)?;
        validate_grid(ROTARY_KERNEL, work_items, ELEMENTWISE_THREADS)?;
        Ok(Self {
            heads,
            width,
            rotary_width,
            pairs,
            elements,
            heads_u32: abi_u32(ROTARY_KERNEL, "heads", heads)?,
            width_u32: abi_u32(ROTARY_KERNEL, "width", width)?,
            rotary_width_u32: abi_u32(ROTARY_KERNEL, "rotary_width", rotary_width)?,
        })
    }

    /// Return the admitted head count.
    #[must_use]
    pub const fn heads(self) -> usize {
        self.heads
    }

    /// Return the complete per-head width.
    #[must_use]
    pub const fn width(self) -> usize {
        self.width
    }

    /// Return the rotated leading width.
    #[must_use]
    pub const fn rotary_width(self) -> usize {
        self.rotary_width
    }

    /// Return the exact cosine/sine coefficient extent.
    #[must_use]
    pub const fn coefficient_elements(self) -> usize {
        self.pairs
    }

    /// Return the exact in-place value extent.
    #[must_use]
    pub const fn elements(self) -> usize {
        self.elements
    }
}

/// Checked native geometry for Qwen's per-head interleaved Q/gate projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitQGateF32Plan {
    heads: usize,
    key_width: usize,
    input_elements: usize,
    output_elements: usize,
    heads_u32: u32,
    key_width_u32: u32,
}

impl SplitQGateF32Plan {
    /// Admit a Q/gate source `[heads, 2 * key_width]` and two outputs.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::UnsupportedShape`] for a zero dimension,
    /// overflowed geometry/layout, an unrepresentable HIP ABI dimension, or
    /// an unrepresentable launch grid.
    pub fn try_from_dimensions(heads: usize, key_width: usize) -> Result<Self> {
        validate_nonzero(SPLIT_Q_GATE_KERNEL, "heads", heads)?;
        validate_nonzero(SPLIT_Q_GATE_KERNEL, "key_width", key_width)?;
        let output_elements =
            checked_product(SPLIT_Q_GATE_KERNEL, heads, key_width, "heads * key_width")?;
        let input_elements = checked_product(
            SPLIT_Q_GATE_KERNEL,
            output_elements,
            2,
            "heads * 2 * key_width",
        )?;
        validate_f32_layout(SPLIT_Q_GATE_KERNEL, "Q/gate source", input_elements)?;
        validate_f32_layout(SPLIT_Q_GATE_KERNEL, "query output", output_elements)?;
        validate_f32_layout(SPLIT_Q_GATE_KERNEL, "gate output", output_elements)?;
        validate_grid(SPLIT_Q_GATE_KERNEL, output_elements, ELEMENTWISE_THREADS)?;
        Ok(Self {
            heads,
            key_width,
            input_elements,
            output_elements,
            heads_u32: abi_u32(SPLIT_Q_GATE_KERNEL, "heads", heads)?,
            key_width_u32: abi_u32(SPLIT_Q_GATE_KERNEL, "key_width", key_width)?,
        })
    }

    /// Return the admitted head count.
    #[must_use]
    pub const fn heads(self) -> usize {
        self.heads
    }

    /// Return the admitted key/query width per head.
    #[must_use]
    pub const fn key_width(self) -> usize {
        self.key_width
    }

    /// Return the exact Q/gate source extent.
    #[must_use]
    pub const fn input_elements(self) -> usize {
        self.input_elements
    }

    /// Return the exact extent of each split output.
    #[must_use]
    pub const fn output_elements(self) -> usize {
        self.output_elements
    }
}

/// Checked native geometry shared by the three elementwise decoder operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElementwiseF32Plan {
    elements: usize,
    elements_u32: u32,
}

impl ElementwiseF32Plan {
    /// Admit one nonempty dense f32 elementwise extent.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::UnsupportedShape`] when the extent cannot form
    /// a Rust f32 allocation layout, fit the u32 HIP ABI, or fit its grid.
    pub fn try_from_elements(elements: usize) -> Result<Self> {
        validate_nonzero(SIGMOID_MUL_KERNEL, "elements", elements)?;
        validate_f32_layout(SIGMOID_MUL_KERNEL, "elementwise values", elements)?;
        validate_grid(SIGMOID_MUL_KERNEL, elements, ELEMENTWISE_THREADS)?;
        Ok(Self {
            elements,
            elements_u32: abi_u32(SIGMOID_MUL_KERNEL, "elements", elements)?,
        })
    }

    /// Return the exact admitted dense extent.
    #[must_use]
    pub const fn elements(self) -> usize {
        self.elements
    }
}

/// Launch f32 RMSNorm with one wave32 per row.
///
/// # Errors
///
/// Returns [`crate::Error::UnsupportedShape`] when any declared extent or
/// device span violates `plan`; returns [`crate::Error::NoGpuBuild`] for a
/// CPU-only build without initializing HIP; propagates stream-current and HIP
/// submission failures.
///
/// # Safety
///
/// Every nonempty pointer must identify a live, correctly aligned allocation
/// on `stream`'s device through completion. `output_f32` must not alias either
/// input and is exclusively writable through completion. Inputs and every
/// reduction/output intermediate must be finite and normal-or-zero; this ABI
/// has no device status channel for checked numerical refusals.
pub unsafe fn launch_rms_norm_f32(
    plan: RmsNormF32Plan,
    input_f32: *const f32,
    input_elements: usize,
    weight_f32: *const f32,
    weight_elements: usize,
    output_f32: *mut f32,
    output_elements: usize,
    stream: &Stream,
) -> Result<()> {
    #[cfg(logismos_no_gpu_kernels)]
    {
        let _ = (
            plan,
            input_f32,
            input_elements,
            weight_f32,
            weight_elements,
            output_f32,
            output_elements,
            stream,
        );
        no_gpu_refusal(RMS_NORM_KERNEL)
    }
    #[cfg(not(logismos_no_gpu_kernels))]
    {
        validate_rms_launch(
            plan,
            input_f32,
            input_elements,
            weight_f32,
            weight_elements,
            output_f32,
            output_elements,
        )?;
        stream.make_current()?;
        // SAFETY: the caller's ownership and finite-domain contract plus the
        // checked plan/spans establish the private ABI's preconditions.
        let code = unsafe {
            logismos_launch_decoder_rms_norm_f32(
                input_f32.cast::<c_void>(),
                weight_f32.cast::<c_void>(),
                output_f32.cast::<c_void>(),
                plan.rows_u32,
                plan.width_u32,
                plan.epsilon,
                stream.raw().cast::<c_void>(),
            )
        };
        launch_result(RMS_NORM_KERNEL, code)
    }
}

/// Launch in-place f32 half-split rotary embedding.
///
/// # Errors
///
/// Returns typed shape/span failures, a CPU-only [`crate::Error::NoGpuBuild`],
/// or propagated stream and launch failures.
///
/// # Safety
///
/// `values_f32` must be exclusively writable through completion and must not
/// alias either coefficient span. All values, coefficients, products, and
/// rotated results must be finite and normal-or-zero on `stream`'s device.
pub unsafe fn launch_rotary_half_split_f32_in_place(
    plan: RotaryHalfSplitF32Plan,
    values_f32: *mut f32,
    value_elements: usize,
    cos_f32: *const f32,
    cos_elements: usize,
    sin_f32: *const f32,
    sin_elements: usize,
    stream: &Stream,
) -> Result<()> {
    #[cfg(logismos_no_gpu_kernels)]
    {
        let _ = (
            plan,
            values_f32,
            value_elements,
            cos_f32,
            cos_elements,
            sin_f32,
            sin_elements,
            stream,
        );
        no_gpu_refusal(ROTARY_KERNEL)
    }
    #[cfg(not(logismos_no_gpu_kernels))]
    {
        validate_rotary_launch(
            plan,
            values_f32,
            value_elements,
            cos_f32,
            cos_elements,
            sin_f32,
            sin_elements,
        )?;
        stream.make_current()?;
        // SAFETY: validated spans and the caller's device/numerical contract
        // establish the private ABI's preconditions.
        let code = unsafe {
            logismos_launch_decoder_rotary_half_split_f32(
                values_f32.cast::<c_void>(),
                cos_f32.cast::<c_void>(),
                sin_f32.cast::<c_void>(),
                plan.heads_u32,
                plan.width_u32,
                plan.rotary_width_u32,
                stream.raw().cast::<c_void>(),
            )
        };
        launch_result(ROTARY_KERNEL, code)
    }
}

/// Launch per-head Q/gate split from `[heads, 2 * key_width]`.
///
/// # Errors
///
/// Returns typed exact-span, CPU-only, stream, or HIP-launch failures.
///
/// # Safety
///
/// The source is immutable through completion. Both outputs are exclusive and
/// may not alias each other or the source. Every device operand is finite and
/// normal-or-zero through this copy-only operation.
pub unsafe fn launch_split_q_gate_f32(
    plan: SplitQGateF32Plan,
    q_gate_f32: *const f32,
    q_gate_elements: usize,
    query_f32: *mut f32,
    query_elements: usize,
    gate_f32: *mut f32,
    gate_elements: usize,
    stream: &Stream,
) -> Result<()> {
    #[cfg(logismos_no_gpu_kernels)]
    {
        let _ = (
            plan,
            q_gate_f32,
            q_gate_elements,
            query_f32,
            query_elements,
            gate_f32,
            gate_elements,
            stream,
        );
        no_gpu_refusal(SPLIT_Q_GATE_KERNEL)
    }
    #[cfg(not(logismos_no_gpu_kernels))]
    {
        validate_split_launch(
            plan,
            q_gate_f32,
            q_gate_elements,
            query_f32,
            query_elements,
            gate_f32,
            gate_elements,
        )?;
        stream.make_current()?;
        // SAFETY: validated spans and the caller's ownership contract establish the ABI.
        let code = unsafe {
            logismos_launch_decoder_split_q_gate_f32(
                q_gate_f32.cast::<c_void>(),
                query_f32.cast::<c_void>(),
                gate_f32.cast::<c_void>(),
                plan.heads_u32,
                plan.key_width_u32,
                stream.raw().cast::<c_void>(),
            )
        };
        launch_result(SPLIT_Q_GATE_KERNEL, code)
    }
}

/// Launch `value * sigmoid(gate)` over one checked exact extent.
///
/// # Errors
///
/// Returns typed exact-span, CPU-only, stream, or HIP-launch failures.
///
/// # Safety
///
/// Inputs remain immutable and output remains exclusively writable through
/// completion. Output must not alias either input. Inputs and all sigmoid and
/// product intermediates must be finite and normal-or-zero.
pub unsafe fn sigmoid_mul(
    plan: ElementwiseF32Plan,
    value_f32: *const f32,
    value_elements: usize,
    gate_f32: *const f32,
    gate_elements: usize,
    output_f32: *mut f32,
    output_elements: usize,
    stream: &Stream,
) -> Result<()> {
    // SAFETY: this public function preserves the caller's documented device
    // ownership and numerical-domain obligations for the shared launch path.
    unsafe {
        launch_elementwise(
            plan,
            value_f32,
            value_elements,
            gate_f32,
            gate_elements,
            output_f32,
            output_elements,
            stream,
            SIGMOID_MUL_KERNEL,
        )
    }
}

/// Launch `SiLU(gate) * up` over one checked exact extent.
///
/// # Errors
///
/// Returns typed exact-span, CPU-only, stream, or HIP-launch failures.
///
/// # Safety
///
/// Inputs remain immutable and output remains exclusively writable through
/// completion. Output must not alias either input. Inputs and all exponential,
/// activation, and product intermediates must be finite and normal-or-zero.
pub unsafe fn silu_mul(
    plan: ElementwiseF32Plan,
    gate_f32: *const f32,
    gate_elements: usize,
    up_f32: *const f32,
    up_elements: usize,
    output_f32: *mut f32,
    output_elements: usize,
    stream: &Stream,
) -> Result<()> {
    // SAFETY: this public function preserves the caller's documented device
    // ownership and numerical-domain obligations for the shared launch path.
    unsafe {
        launch_elementwise(
            plan,
            gate_f32,
            gate_elements,
            up_f32,
            up_elements,
            output_f32,
            output_elements,
            stream,
            SILU_MUL_KERNEL,
        )
    }
}

/// Launch `left + right` over one checked exact extent.
///
/// # Errors
///
/// Returns typed exact-span, CPU-only, stream, or HIP-launch failures.
///
/// # Safety
///
/// Inputs remain immutable and output remains exclusively writable through
/// completion. Output must not alias either input. Inputs and every sum must
/// be finite and normal-or-zero.
pub unsafe fn residual_add(
    plan: ElementwiseF32Plan,
    left_f32: *const f32,
    left_elements: usize,
    right_f32: *const f32,
    right_elements: usize,
    output_f32: *mut f32,
    output_elements: usize,
    stream: &Stream,
) -> Result<()> {
    // SAFETY: this public function preserves the caller's documented device
    // ownership and numerical-domain obligations for the shared launch path.
    unsafe {
        launch_elementwise(
            plan,
            left_f32,
            left_elements,
            right_f32,
            right_elements,
            output_f32,
            output_elements,
            stream,
            RESIDUAL_ADD_KERNEL,
        )
    }
}

unsafe fn launch_elementwise(
    plan: ElementwiseF32Plan,
    left_f32: *const f32,
    left_elements: usize,
    right_f32: *const f32,
    right_elements: usize,
    output_f32: *mut f32,
    output_elements: usize,
    stream: &Stream,
    kernel: &'static str,
) -> Result<()> {
    #[cfg(logismos_no_gpu_kernels)]
    {
        let _ = (
            plan,
            left_f32,
            left_elements,
            right_f32,
            right_elements,
            output_f32,
            output_elements,
            stream,
        );
        no_gpu_refusal(kernel)
    }
    #[cfg(not(logismos_no_gpu_kernels))]
    {
        validate_elementwise_launch(
            plan,
            left_f32,
            left_elements,
            right_f32,
            right_elements,
            output_f32,
            output_elements,
            kernel,
        )?;
        stream.make_current()?;
        // SAFETY: this function preserves the public caller's safety contract
        // after exact-span validation and selects the matching private ABI.
        let code = unsafe {
            match kernel {
                SIGMOID_MUL_KERNEL => logismos_launch_decoder_sigmoid_mul_f32(
                    left_f32.cast::<c_void>(),
                    right_f32.cast::<c_void>(),
                    output_f32.cast::<c_void>(),
                    plan.elements_u32,
                    stream.raw().cast::<c_void>(),
                ),
                SILU_MUL_KERNEL => logismos_launch_decoder_silu_mul_f32(
                    left_f32.cast::<c_void>(),
                    right_f32.cast::<c_void>(),
                    output_f32.cast::<c_void>(),
                    plan.elements_u32,
                    stream.raw().cast::<c_void>(),
                ),
                RESIDUAL_ADD_KERNEL => logismos_launch_decoder_residual_add_f32(
                    left_f32.cast::<c_void>(),
                    right_f32.cast::<c_void>(),
                    output_f32.cast::<c_void>(),
                    plan.elements_u32,
                    stream.raw().cast::<c_void>(),
                ),
                _ => {
                    return unsupported_shape(
                        kernel,
                        "unknown elementwise native operation".to_owned(),
                    );
                }
            }
        };
        launch_result(kernel, code)
    }
}

#[cfg(logismos_no_gpu_kernels)]
fn no_gpu_refusal(kernel: &'static str) -> Result<()> {
    NoGpuBuildSnafu { kernel }.fail()
}

#[cfg(not(logismos_no_gpu_kernels))]
fn launch_result(kernel: &'static str, code: u32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        LaunchSnafu {
            kernel,
            kind: hipcore::ErrorKind::from_raw(code),
            code,
        }
        .fail()
    }
}

#[cfg(any(test, not(logismos_no_gpu_kernels)))]
fn validate_rms_launch(
    plan: RmsNormF32Plan,
    input_f32: *const f32,
    input_elements: usize,
    weight_f32: *const f32,
    weight_elements: usize,
    output_f32: *mut f32,
    output_elements: usize,
) -> Result<()> {
    validate_length(RMS_NORM_KERNEL, "input", input_elements, plan.elements)?;
    validate_length(RMS_NORM_KERNEL, "weight", weight_elements, plan.width)?;
    validate_length(RMS_NORM_KERNEL, "output", output_elements, plan.elements)?;
    let input = checked_f32_device_span(RMS_NORM_KERNEL, input_f32, input_elements, "input")?;
    let weight = checked_f32_device_span(RMS_NORM_KERNEL, weight_f32, weight_elements, "weight")?;
    let output = checked_f32_device_span(
        RMS_NORM_KERNEL,
        output_f32.cast_const(),
        output_elements,
        "output",
    )?;
    reject_overlapping_f32_spans(RMS_NORM_KERNEL, output, input)?;
    reject_overlapping_f32_spans(RMS_NORM_KERNEL, output, weight)
}

#[cfg(any(test, not(logismos_no_gpu_kernels)))]
fn validate_rotary_launch(
    plan: RotaryHalfSplitF32Plan,
    values_f32: *mut f32,
    value_elements: usize,
    cos_f32: *const f32,
    cos_elements: usize,
    sin_f32: *const f32,
    sin_elements: usize,
) -> Result<()> {
    validate_length(ROTARY_KERNEL, "values", value_elements, plan.elements)?;
    validate_length(
        ROTARY_KERNEL,
        "cosine coefficients",
        cos_elements,
        plan.pairs,
    )?;
    validate_length(ROTARY_KERNEL, "sine coefficients", sin_elements, plan.pairs)?;
    let values = checked_f32_device_span(
        ROTARY_KERNEL,
        values_f32.cast_const(),
        value_elements,
        "values",
    )?;
    let cosine =
        checked_f32_device_span(ROTARY_KERNEL, cos_f32, cos_elements, "cosine coefficients")?;
    let sine = checked_f32_device_span(ROTARY_KERNEL, sin_f32, sin_elements, "sine coefficients")?;
    reject_overlapping_f32_spans(ROTARY_KERNEL, values, cosine)?;
    reject_overlapping_f32_spans(ROTARY_KERNEL, values, sine)
}

#[cfg(any(test, not(logismos_no_gpu_kernels)))]
fn validate_split_launch(
    plan: SplitQGateF32Plan,
    q_gate_f32: *const f32,
    q_gate_elements: usize,
    query_f32: *mut f32,
    query_elements: usize,
    gate_f32: *mut f32,
    gate_elements: usize,
) -> Result<()> {
    validate_length(
        SPLIT_Q_GATE_KERNEL,
        "Q/gate source",
        q_gate_elements,
        plan.input_elements,
    )?;
    validate_length(
        SPLIT_Q_GATE_KERNEL,
        "query output",
        query_elements,
        plan.output_elements,
    )?;
    validate_length(
        SPLIT_Q_GATE_KERNEL,
        "gate output",
        gate_elements,
        plan.output_elements,
    )?;
    let source = checked_f32_device_span(
        SPLIT_Q_GATE_KERNEL,
        q_gate_f32,
        q_gate_elements,
        "Q/gate source",
    )?;
    let query = checked_f32_device_span(
        SPLIT_Q_GATE_KERNEL,
        query_f32.cast_const(),
        query_elements,
        "query output",
    )?;
    let gate = checked_f32_device_span(
        SPLIT_Q_GATE_KERNEL,
        gate_f32.cast_const(),
        gate_elements,
        "gate output",
    )?;
    reject_overlapping_f32_spans(SPLIT_Q_GATE_KERNEL, query, source)?;
    reject_overlapping_f32_spans(SPLIT_Q_GATE_KERNEL, gate, source)?;
    reject_overlapping_f32_spans(SPLIT_Q_GATE_KERNEL, query, gate)
}

#[cfg(any(test, not(logismos_no_gpu_kernels)))]
fn validate_elementwise_launch(
    plan: ElementwiseF32Plan,
    left_f32: *const f32,
    left_elements: usize,
    right_f32: *const f32,
    right_elements: usize,
    output_f32: *mut f32,
    output_elements: usize,
    kernel: &'static str,
) -> Result<()> {
    validate_length(kernel, "left operand", left_elements, plan.elements)?;
    validate_length(kernel, "right operand", right_elements, plan.elements)?;
    validate_length(kernel, "output", output_elements, plan.elements)?;
    let left = checked_f32_device_span(kernel, left_f32, left_elements, "left operand")?;
    let right = checked_f32_device_span(kernel, right_f32, right_elements, "right operand")?;
    let output =
        checked_f32_device_span(kernel, output_f32.cast_const(), output_elements, "output")?;
    reject_overlapping_f32_spans(kernel, output, left)?;
    reject_overlapping_f32_spans(kernel, output, right)
}

fn validate_nonzero(kernel: &'static str, name: &'static str, value: usize) -> Result<()> {
    if value == 0 {
        unsupported_shape(kernel, format!("{name} must be greater than zero"))
    } else {
        Ok(())
    }
}

fn validate_positive_normal(kernel: &'static str, name: &'static str, value: f32) -> Result<()> {
    if value.is_normal() && value.is_sign_positive() {
        Ok(())
    } else {
        unsupported_shape(kernel, format!("{name} must be positive normal finite f32"))
    }
}

fn checked_product(
    kernel: &'static str,
    left: usize,
    right: usize,
    name: &'static str,
) -> Result<usize> {
    left.checked_mul(right).ok_or_else(|| {
        UnsupportedShapeSnafu {
            kernel,
            msg: format!("{name} overflows usize"),
        }
        .build()
    })
}

fn validate_f32_layout(kernel: &'static str, name: &'static str, elements: usize) -> Result<()> {
    std::alloc::Layout::array::<f32>(elements).map_err(|_| {
        UnsupportedShapeSnafu {
            kernel,
            msg: format!("{name} length {elements} exceeds the Rust allocation layout domain"),
        }
        .build()
    })?;
    Ok(())
}

fn abi_u32(kernel: &'static str, name: &'static str, value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        UnsupportedShapeSnafu {
            kernel,
            msg: format!("{name} {value} exceeds the u32 HIP ABI"),
        }
        .build()
    })
}

fn validate_grid(kernel: &'static str, work_items: usize, threads: usize) -> Result<()> {
    let rounded = work_items.checked_add(threads - 1).ok_or_else(|| {
        UnsupportedShapeSnafu {
            kernel,
            msg: "native launch-grid rounding overflows usize".to_owned(),
        }
        .build()
    })?;
    let blocks = rounded / threads;
    abi_u32(kernel, "native launch-grid blocks", blocks).map(|_| ())
}

fn validate_length(
    kernel: &'static str,
    name: &'static str,
    actual: usize,
    expected: usize,
) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        unsupported_shape(
            kernel,
            format!("{name} length {actual} must equal checked {expected}"),
        )
    }
}

fn unsupported_shape<T>(kernel: &'static str, msg: String) -> Result<T> {
    UnsupportedShapeSnafu { kernel, msg }.fail()
}

fn reserve_native_reference(operation: &'static str, elements: usize) -> Result<Vec<f32>> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(elements)
        .context(crate::error::CpuF32AllocationSnafu {
            operation,
            requested_len: elements,
        })?;
    Ok(output)
}

fn validate_reference_length(
    operation: &'static str,
    name: &'static str,
    actual: usize,
    expected: usize,
) -> Result<()> {
    validate_length(operation, name, actual, expected)
}

fn rms_norm_native_order_reference(
    plan: RmsNormF32Plan,
    input: &[f32],
    weight: &[f32],
) -> Result<Vec<f32>> {
    validate_reference_length(RMS_NORM_KERNEL, "input", input.len(), plan.elements)?;
    validate_reference_length(RMS_NORM_KERNEL, "weight", weight.len(), plan.width)?;
    let mut output = reserve_native_reference("decoder RMSNorm reference", plan.elements)?;
    for input_row in input.chunks_exact(plan.width) {
        let mut lanes = [0.0_f32; WAVE_SIZE];
        for lane in 0..WAVE_SIZE {
            let mut column = lane;
            while column < plan.width {
                let value = input_row.get(column).copied().ok_or_else(|| {
                    UnsupportedShapeSnafu {
                        kernel: RMS_NORM_KERNEL,
                        msg: "checked row access failed".to_owned(),
                    }
                    .build()
                })?;
                let partial = lanes.get(lane).copied().ok_or_else(|| {
                    UnsupportedShapeSnafu {
                        kernel: RMS_NORM_KERNEL,
                        msg: "fixed wave lane access failed".to_owned(),
                    }
                    .build()
                })?;
                let slot = lanes.get_mut(lane).ok_or_else(|| {
                    UnsupportedShapeSnafu {
                        kernel: RMS_NORM_KERNEL,
                        msg: "fixed wave lane mutation failed".to_owned(),
                    }
                    .build()
                })?;
                *slot = partial + value * value;
                column = column.checked_add(WAVE_SIZE).ok_or_else(|| {
                    UnsupportedShapeSnafu {
                        kernel: RMS_NORM_KERNEL,
                        msg: "lane-strided column overflow".to_owned(),
                    }
                    .build()
                })?;
            }
        }
        for offset in [16_usize, 8, 4, 2, 1] {
            for lane in 0..offset {
                let partial = lanes.get(lane).copied().ok_or_else(|| {
                    UnsupportedShapeSnafu {
                        kernel: RMS_NORM_KERNEL,
                        msg: "fixed wave lane access failed".to_owned(),
                    }
                    .build()
                })?;
                let peer = lanes.get(lane + offset).copied().ok_or_else(|| {
                    UnsupportedShapeSnafu {
                        kernel: RMS_NORM_KERNEL,
                        msg: "fixed wave peer access failed".to_owned(),
                    }
                    .build()
                })?;
                let slot = lanes.get_mut(lane).ok_or_else(|| {
                    UnsupportedShapeSnafu {
                        kernel: RMS_NORM_KERNEL,
                        msg: "fixed wave lane mutation failed".to_owned(),
                    }
                    .build()
                })?;
                *slot = partial + peer;
            }
        }
        let sum = lanes.first().copied().ok_or_else(|| {
            UnsupportedShapeSnafu {
                kernel: RMS_NORM_KERNEL,
                msg: "fixed wave has no lane zero".to_owned(),
            }
            .build()
        })?;
        let width = plan.width.to_f32().ok_or_else(|| {
            UnsupportedShapeSnafu {
                kernel: RMS_NORM_KERNEL,
                msg: "checked RMSNorm width cannot convert to f32".to_owned(),
            }
            .build()
        })?;
        let inverse = (sum / width + plan.epsilon).sqrt().recip();
        for (value, scale) in input_row.iter().zip(weight.iter()) {
            output.push(*value * inverse * *scale);
        }
    }
    Ok(output)
}

fn rotary_half_split_native_order_reference(
    plan: RotaryHalfSplitF32Plan,
    values: &[f32],
    cos: &[f32],
    sin: &[f32],
) -> Result<Vec<f32>> {
    validate_reference_length(ROTARY_KERNEL, "values", values.len(), plan.elements)?;
    validate_reference_length(ROTARY_KERNEL, "cosine coefficients", cos.len(), plan.pairs)?;
    validate_reference_length(ROTARY_KERNEL, "sine coefficients", sin.len(), plan.pairs)?;
    let mut output = reserve_native_reference("decoder rotary reference", plan.elements)?;
    output.extend_from_slice(values);
    for row in output.chunks_exact_mut(plan.width) {
        for pair in 0..plan.pairs {
            let partner = pair.checked_add(plan.pairs).ok_or_else(|| {
                UnsupportedShapeSnafu {
                    kernel: ROTARY_KERNEL,
                    msg: "half-split pair index overflow".to_owned(),
                }
                .build()
            })?;
            let left = row.get(pair).copied().ok_or_else(|| {
                UnsupportedShapeSnafu {
                    kernel: ROTARY_KERNEL,
                    msg: "checked left pair access failed".to_owned(),
                }
                .build()
            })?;
            let right = row.get(partner).copied().ok_or_else(|| {
                UnsupportedShapeSnafu {
                    kernel: ROTARY_KERNEL,
                    msg: "checked right pair access failed".to_owned(),
                }
                .build()
            })?;
            let cosine = cos.get(pair).copied().ok_or_else(|| {
                UnsupportedShapeSnafu {
                    kernel: ROTARY_KERNEL,
                    msg: "checked cosine access failed".to_owned(),
                }
                .build()
            })?;
            let sine = sin.get(pair).copied().ok_or_else(|| {
                UnsupportedShapeSnafu {
                    kernel: ROTARY_KERNEL,
                    msg: "checked sine access failed".to_owned(),
                }
                .build()
            })?;
            let left_slot = row.get_mut(pair).ok_or_else(|| {
                UnsupportedShapeSnafu {
                    kernel: ROTARY_KERNEL,
                    msg: "checked left pair mutation failed".to_owned(),
                }
                .build()
            })?;
            *left_slot = left * cosine - right * sine;
            let right_slot = row.get_mut(partner).ok_or_else(|| {
                UnsupportedShapeSnafu {
                    kernel: ROTARY_KERNEL,
                    msg: "checked right pair mutation failed".to_owned(),
                }
                .build()
            })?;
            *right_slot = left * sine + right * cosine;
        }
    }
    Ok(output)
}

fn split_q_gate_native_order_reference(
    plan: SplitQGateF32Plan,
    source: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    validate_reference_length(
        SPLIT_Q_GATE_KERNEL,
        "Q/gate source",
        source.len(),
        plan.input_elements,
    )?;
    let mut query = reserve_native_reference("decoder Q split reference", plan.output_elements)?;
    let mut gate = reserve_native_reference("decoder gate split reference", plan.output_elements)?;
    for head in source.chunks_exact(plan.key_width * 2) {
        let query_part = head.get(..plan.key_width).ok_or_else(|| {
            UnsupportedShapeSnafu {
                kernel: SPLIT_Q_GATE_KERNEL,
                msg: "checked query half access failed".to_owned(),
            }
            .build()
        })?;
        let gate_part = head.get(plan.key_width..).ok_or_else(|| {
            UnsupportedShapeSnafu {
                kernel: SPLIT_Q_GATE_KERNEL,
                msg: "checked gate half access failed".to_owned(),
            }
            .build()
        })?;
        query.extend_from_slice(query_part);
        gate.extend_from_slice(gate_part);
    }
    Ok((query, gate))
}

fn sigmoid_mul_native_order_reference(
    plan: ElementwiseF32Plan,
    value: &[f32],
    gate: &[f32],
) -> Result<Vec<f32>> {
    elementwise_native_order_reference(plan, value, gate, SIGMOID_MUL_KERNEL, |left, right| {
        left * (1.0_f32 / (1.0_f32 + (-right).exp()))
    })
}

fn silu_mul_native_order_reference(
    plan: ElementwiseF32Plan,
    gate: &[f32],
    up: &[f32],
) -> Result<Vec<f32>> {
    elementwise_native_order_reference(plan, gate, up, SILU_MUL_KERNEL, |left, right| {
        (left / (1.0_f32 + (-left).exp())) * right
    })
}

fn residual_add_native_order_reference(
    plan: ElementwiseF32Plan,
    left: &[f32],
    right: &[f32],
) -> Result<Vec<f32>> {
    elementwise_native_order_reference(plan, left, right, RESIDUAL_ADD_KERNEL, |left, right| {
        left + right
    })
}

fn elementwise_native_order_reference(
    plan: ElementwiseF32Plan,
    left: &[f32],
    right: &[f32],
    operation: &'static str,
    transform: impl Fn(f32, f32) -> f32,
) -> Result<Vec<f32>> {
    validate_reference_length(operation, "left operand", left.len(), plan.elements)?;
    validate_reference_length(operation, "right operand", right.len(), plan.elements)?;
    let mut output = reserve_native_reference(operation, plan.elements)?;
    output.extend(
        left.iter()
            .zip(right.iter())
            .map(|(left, right)| transform(*left, *right)),
    );
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f32 = 1e-3;

    #[test]
    fn rms_wave_order_tracks_f64_oracle_on_asymmetric_tail()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let plan = RmsNormF32Plan::try_from_dimensions(2, 37, 1e-5)?;
        let mut input = Vec::with_capacity(plan.elements());
        for index in 0..plan.elements() {
            input.push((f32::from(u16::try_from(index)?) * 0.173 - 2.1) / 3.7);
        }
        let mut weight = Vec::with_capacity(plan.width());
        for index in 0..plan.width() {
            weight.push(0.2 + f32::from(u16::try_from(index)?) * 0.031);
        }
        let actual = rms_norm_native_order_reference(plan, &input, &weight)?;
        let expected = rms_f64_oracle(&input, &weight, plan.width(), f64::from(plan.epsilon()))?;
        assert_close_f64(&actual, &expected, "wave32 RMSNorm");
        Ok(())
    }

    #[test]
    fn half_split_rotation_preserves_tail_and_refuses_adjacent_pairing()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let plan = RotaryHalfSplitF32Plan::try_from_dimensions(2, 7, 4)?;
        let values = [
            1.0_f32, 10.0, 2.0, 20.0, 101.0, 102.0, 103.0, -3.0, 30.0, 4.0, 40.0, 201.0, 202.0,
            203.0,
        ];
        let cos = [0.0_f32, 1.0];
        let sin = [1.0_f32, 0.0];
        let actual = rotary_half_split_native_order_reference(plan, &values, &cos, &sin)?;
        assert_eq!(
            actual.get(..4),
            Some(&[-2.0, 10.0, 1.0, 20.0][..]),
            "first head must rotate half-split pairs"
        );
        assert_eq!(
            actual.get(4..7),
            Some(&values[4..7]),
            "leading partial rotary span must preserve its tail"
        );
        assert_eq!(
            actual.get(7..11),
            Some(&[-4.0, 30.0, -3.0, 40.0][..]),
            "second head keeps its own half-split pairing"
        );
        Ok(())
    }

    #[test]
    fn split_and_elementwise_operations_preserve_qwen_order()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let split = SplitQGateF32Plan::try_from_dimensions(2, 3)?;
        let source = [
            1.0_f32, 2.0, 3.0, -2.0, 0.0, 2.0, 10.0, 20.0, 30.0, 3.0, -1.0, 0.5,
        ];
        let (query, gate) = split_q_gate_native_order_reference(split, &source)?;
        assert_eq!(
            query,
            vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "query must select the first per-head half"
        );
        assert_eq!(
            gate,
            vec![-2.0, 0.0, 2.0, 3.0, -1.0, 0.5],
            "gate must select the second per-head half"
        );
        let plan = ElementwiseF32Plan::try_from_elements(query.len())?;
        let sigmoid = sigmoid_mul_native_order_reference(plan, &query, &gate)?;
        let silu = silu_mul_native_order_reference(plan, &gate, &query)?;
        let residual = residual_add_native_order_reference(plan, &sigmoid, &silu)?;
        for (index, got) in residual.iter().enumerate() {
            let q = f64::from(*query.get(index).ok_or("query index")?);
            let g = f64::from(*gate.get(index).ok_or("gate index")?);
            let expected = q * (1.0 / (1.0 + (-g).exp())) + (g / (1.0 + (-g).exp())) * q;
            assert!(
                (f64::from(*got) - expected).abs() <= f64::from(TOLERANCE),
                "index {index}: got {got}, expected {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn plans_and_device_validators_refuse_shape_alias_alignment_and_abi_errors()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        assert!(
            RmsNormF32Plan::try_from_dimensions(0, 1, 1e-5).is_err(),
            "zero row count is invalid"
        );
        assert!(
            RmsNormF32Plan::try_from_dimensions(1, 1, f32::MIN_POSITIVE / 2.0).is_err(),
            "subnormal epsilon violates native-domain qualification"
        );
        assert!(
            RotaryHalfSplitF32Plan::try_from_dimensions(1, 3, 3).is_err(),
            "odd rotary width is invalid"
        );
        assert!(
            SplitQGateF32Plan::try_from_dimensions(usize::MAX, 2).is_err(),
            "overflowing split geometry is invalid"
        );
        let plan = ElementwiseF32Plan::try_from_elements(2)?;
        let values = [1.0_f32, 2.0];
        let mut output = [0.0_f32; 2];
        validate_elementwise_launch(
            plan,
            values.as_ptr(),
            values.len(),
            values.as_ptr(),
            values.len(),
            output.as_mut_ptr(),
            output.len(),
            RESIDUAL_ADD_KERNEL,
        )?;
        assert!(
            validate_elementwise_launch(
                plan,
                values.as_ptr(),
                1,
                values.as_ptr(),
                values.len(),
                output.as_mut_ptr(),
                output.len(),
                RESIDUAL_ADD_KERNEL
            )
            .is_err(),
            "short declared span must be refused"
        );
        assert!(
            validate_elementwise_launch(
                plan,
                values.as_ptr(),
                values.len(),
                values.as_ptr(),
                values.len(),
                values.as_ptr().cast_mut(),
                values.len(),
                RESIDUAL_ADD_KERNEL
            )
            .is_err(),
            "output alias must be refused"
        );
        assert!(
            validate_elementwise_launch(
                plan,
                values.as_ptr(),
                values.len(),
                values.as_ptr(),
                values.len(),
                output.as_mut_ptr().wrapping_byte_add(1),
                output.len(),
                RESIDUAL_ADD_KERNEL
            )
            .is_err(),
            "misaligned output must be refused"
        );
        assert!(
            matches!(
                checked_f32_device_span(
                    RESIDUAL_ADD_KERNEL,
                    core::ptr::NonNull::<f32>::dangling().as_ptr(),
                    usize::MAX,
                    "overflow"
                ),
                Err(crate::Error::UnsupportedShape { .. })
            ),
            "unrepresentable device span must be refused"
        );
        if let Ok(too_wide) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert!(
                ElementwiseF32Plan::try_from_elements(too_wide).is_err(),
                "u32 ABI overflow must be refused"
            );
        }
        Ok(())
    }

    #[cfg(logismos_no_gpu_kernels)]
    #[test]
    fn cpu_only_build_refuses_each_native_entry_without_hip()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        assert!(
            matches!(
                no_gpu_refusal(RMS_NORM_KERNEL),
                Err(crate::Error::NoGpuBuild { .. })
            ),
            "RMSNorm must report typed no-GPU refusal"
        );
        assert!(
            matches!(
                no_gpu_refusal(ROTARY_KERNEL),
                Err(crate::Error::NoGpuBuild { .. })
            ),
            "rotary must report typed no-GPU refusal"
        );
        assert!(
            matches!(
                no_gpu_refusal(SPLIT_Q_GATE_KERNEL),
                Err(crate::Error::NoGpuBuild { .. })
            ),
            "split must report typed no-GPU refusal"
        );
        assert!(
            matches!(
                no_gpu_refusal(SIGMOID_MUL_KERNEL),
                Err(crate::Error::NoGpuBuild { .. })
            ),
            "sigmoid multiplication must report typed no-GPU refusal"
        );
        Ok(())
    }

    #[cfg(not(logismos_no_gpu_kernels))]
    #[test]
    #[ignore = "requires an operator-reserved HIP device; source tests do not qualify hardware"]
    fn reserved_device_decoder_ops_match_native_references() -> core::result::Result<(), String> {
        use hipcore::{Device, DeviceBuffer};

        let device = Device::new(0).map_err(|error| format!("open reserved device: {error}"))?;
        let stream = Stream::new(&device).map_err(|error| format!("create stream: {error}"))?;

        let rms_plan = RmsNormF32Plan::try_from_dimensions(1, 3, 1e-5)
            .map_err(|error| format!("plan RMSNorm: {error}"))?;
        let rms_input_host = [1.0_f32, -2.0, 3.0];
        let rms_weight_host = [0.5_f32, 1.5, -1.0];
        let rms_input = DeviceBuffer::from_host(&device, &rms_input_host)
            .map_err(|error| format!("upload RMSNorm input: {error}"))?;
        let rms_weight = DeviceBuffer::from_host(&device, &rms_weight_host)
            .map_err(|error| format!("upload RMSNorm weight: {error}"))?;
        let rms_output = DeviceBuffer::alloc(&device, rms_plan.elements())
            .map_err(|error| format!("allocate RMSNorm output: {error}"))?;
        // SAFETY: distinct buffers satisfy the exact plan spans and remain live through synchronization.
        unsafe {
            launch_rms_norm_f32(
                rms_plan,
                rms_input.as_device_ptr(),
                rms_input.len(),
                rms_weight.as_device_ptr(),
                rms_weight.len(),
                rms_output.as_device_ptr(),
                rms_output.len(),
                &stream,
            )
        }
        .map_err(|error| format!("launch RMSNorm: {error}"))?;
        stream
            .synchronize()
            .map_err(|error| format!("synchronize RMSNorm: {error}"))?;
        let rms_actual = read_device(&rms_output)?;
        let rms_expected =
            rms_norm_native_order_reference(rms_plan, &rms_input_host, &rms_weight_host)
                .map_err(|error| format!("RMSNorm reference: {error}"))?;
        assert_close_f32(&rms_actual, &rms_expected, "device RMSNorm");

        let rotary_plan = RotaryHalfSplitF32Plan::try_from_dimensions(1, 5, 4)
            .map_err(|error| format!("plan rotary: {error}"))?;
        let rotary_values_host = [1.0_f32, 2.0, 3.0, 4.0, 99.0];
        let cosine_host = [0.0_f32, 1.0];
        let sine_host = [1.0_f32, 0.0];
        let rotary_values = DeviceBuffer::from_host(&device, &rotary_values_host)
            .map_err(|error| format!("upload rotary values: {error}"))?;
        let cosine = DeviceBuffer::from_host(&device, &cosine_host)
            .map_err(|error| format!("upload cosine: {error}"))?;
        let sine = DeviceBuffer::from_host(&device, &sine_host)
            .map_err(|error| format!("upload sine: {error}"))?;
        // SAFETY: values is exclusive and does not alias either coefficient buffer.
        unsafe {
            launch_rotary_half_split_f32_in_place(
                rotary_plan,
                rotary_values.as_device_ptr(),
                rotary_values.len(),
                cosine.as_device_ptr(),
                cosine.len(),
                sine.as_device_ptr(),
                sine.len(),
                &stream,
            )
        }
        .map_err(|error| format!("launch rotary: {error}"))?;
        stream
            .synchronize()
            .map_err(|error| format!("synchronize rotary: {error}"))?;
        let rotary_actual = read_device(&rotary_values)?;
        let rotary_expected = rotary_half_split_native_order_reference(
            rotary_plan,
            &rotary_values_host,
            &cosine_host,
            &sine_host,
        )
        .map_err(|error| format!("rotary reference: {error}"))?;
        assert_close_f32(&rotary_actual, &rotary_expected, "device rotary");

        let split_plan = SplitQGateF32Plan::try_from_dimensions(1, 3)
            .map_err(|error| format!("plan split: {error}"))?;
        let split_source_host = [1.0_f32, 2.0, 3.0, -2.0, 0.0, 2.0];
        let split_source = DeviceBuffer::from_host(&device, &split_source_host)
            .map_err(|error| format!("upload split source: {error}"))?;
        let split_query = DeviceBuffer::alloc(&device, split_plan.output_elements())
            .map_err(|error| format!("allocate split query: {error}"))?;
        let split_gate = DeviceBuffer::alloc(&device, split_plan.output_elements())
            .map_err(|error| format!("allocate split gate: {error}"))?;
        // SAFETY: source is immutable and the two outputs are distinct exact plan spans.
        unsafe {
            launch_split_q_gate_f32(
                split_plan,
                split_source.as_device_ptr(),
                split_source.len(),
                split_query.as_device_ptr(),
                split_query.len(),
                split_gate.as_device_ptr(),
                split_gate.len(),
                &stream,
            )
        }
        .map_err(|error| format!("launch split: {error}"))?;
        stream
            .synchronize()
            .map_err(|error| format!("synchronize split: {error}"))?;
        let (query_expected, gate_expected) =
            split_q_gate_native_order_reference(split_plan, &split_source_host)
                .map_err(|error| format!("split reference: {error}"))?;
        assert_close_f32(
            &read_device(&split_query)?,
            &query_expected,
            "device Q split",
        );
        assert_close_f32(
            &read_device(&split_gate)?,
            &gate_expected,
            "device gate split",
        );

        let elementwise = ElementwiseF32Plan::try_from_elements(query_expected.len())
            .map_err(|error| format!("plan elementwise: {error}"))?;
        let elementwise_output = DeviceBuffer::alloc(&device, elementwise.elements())
            .map_err(|error| format!("allocate elementwise output: {error}"))?;
        // SAFETY: all source/output buffers are distinct exact plan spans.
        unsafe {
            sigmoid_mul(
                elementwise,
                split_query.as_device_ptr(),
                split_query.len(),
                split_gate.as_device_ptr(),
                split_gate.len(),
                elementwise_output.as_device_ptr(),
                elementwise_output.len(),
                &stream,
            )
        }
        .map_err(|error| format!("launch sigmoid multiplication: {error}"))?;
        stream
            .synchronize()
            .map_err(|error| format!("synchronize sigmoid multiplication: {error}"))?;
        let sigmoid_actual = read_device(&elementwise_output)?;
        let sigmoid_expected =
            sigmoid_mul_native_order_reference(elementwise, &query_expected, &gate_expected)
                .map_err(|error| format!("sigmoid reference: {error}"))?;
        assert_close_f32(
            &sigmoid_actual,
            &sigmoid_expected,
            "device sigmoid multiplication",
        );
        // SAFETY: inputs remain immutable and output remains an exclusive exact plan span.
        unsafe {
            silu_mul(
                elementwise,
                split_gate.as_device_ptr(),
                split_gate.len(),
                split_query.as_device_ptr(),
                split_query.len(),
                elementwise_output.as_device_ptr(),
                elementwise_output.len(),
                &stream,
            )
        }
        .map_err(|error| format!("launch SiLU multiplication: {error}"))?;
        stream
            .synchronize()
            .map_err(|error| format!("synchronize SiLU multiplication: {error}"))?;
        let silu_actual = read_device(&elementwise_output)?;
        let silu_expected =
            silu_mul_native_order_reference(elementwise, &gate_expected, &query_expected)
                .map_err(|error| format!("SiLU reference: {error}"))?;
        assert_close_f32(&silu_actual, &silu_expected, "device SiLU multiplication");
        // SAFETY: both input buffers and the output buffer remain distinct through completion.
        unsafe {
            residual_add(
                elementwise,
                split_query.as_device_ptr(),
                split_query.len(),
                split_gate.as_device_ptr(),
                split_gate.len(),
                elementwise_output.as_device_ptr(),
                elementwise_output.len(),
                &stream,
            )
        }
        .map_err(|error| format!("launch residual addition: {error}"))?;
        stream
            .synchronize()
            .map_err(|error| format!("synchronize residual addition: {error}"))?;
        let residual_actual = read_device(&elementwise_output)?;
        let residual_expected =
            residual_add_native_order_reference(elementwise, &query_expected, &gate_expected)
                .map_err(|error| format!("residual reference: {error}"))?;
        assert_close_f32(
            &residual_actual,
            &residual_expected,
            "device residual addition",
        );
        Ok(())
    }

    fn rms_f64_oracle(
        input: &[f32],
        weight: &[f32],
        width: usize,
        epsilon: f64,
    ) -> Result<Vec<f64>> {
        let width_u32 = u32::try_from(width).map_err(|_| {
            UnsupportedShapeSnafu {
                kernel: RMS_NORM_KERNEL,
                msg: "f64 oracle width exceeds the checked u32 plan domain".to_owned(),
            }
            .build()
        })?;
        let width_f64 = f64::from(width_u32);
        let mut output = Vec::with_capacity(input.len());
        for row in input.chunks_exact(width) {
            let sum = row
                .iter()
                .map(|value| f64::from(*value) * f64::from(*value))
                .sum::<f64>();
            let inverse = (sum / width_f64 + epsilon).sqrt().recip();
            output.extend(
                row.iter()
                    .zip(weight.iter())
                    .map(|(value, scale)| f64::from(*value) * inverse * f64::from(*scale)),
            );
        }
        Ok(output)
    }

    fn assert_close_f64(actual: &[f32], expected: &[f64], operation: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{operation} output lengths must match"
        );
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (f64::from(*actual) - expected).abs() <= f64::from(TOLERANCE),
                "{operation} index {index}: got {actual}, expected {expected}"
            );
        }
    }

    fn assert_close_f32(actual: &[f32], expected: &[f32], operation: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{operation} output lengths must match"
        );
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= TOLERANCE,
                "{operation} index {index}: got {actual}, expected {expected}"
            );
        }
    }

    fn read_device(buffer: &hipcore::DeviceBuffer<f32>) -> core::result::Result<Vec<f32>, String> {
        let mut host = vec![0.0_f32; buffer.len()];
        buffer
            .copy_to_host(&mut host)
            .map_err(|error| format!("copy device buffer: {error}"))?;
        Ok(host)
    }
}
