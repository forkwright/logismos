//! Bounded CPU reference for dense causal convolution.
//!
//! This module is a model-agnostic primitive: it has no activation, bias,
//! normalization, packing, cache, device, or model-family semantics. Its
//! explicit raw-input history permits callers to compose chunked evaluation
//! without hidden mutable state.

#[cfg(all(feature = "gpu", not(logismos_no_gpu_kernels)))]
use std::ffi::c_void;

#[cfg(feature = "gpu")]
use hipcore::Stream;
use snafu::{ResultExt, Snafu};

#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
use crate::device_span::{checked_f32_device_span, reject_overlapping_f32_spans};
#[cfg(all(feature = "gpu", not(logismos_no_gpu_kernels)))]
use crate::error::LaunchSnafu;
#[cfg(all(feature = "gpu", logismos_no_gpu_kernels))]
use crate::error::NoGpuBuildSnafu;
#[cfg(feature = "gpu")]
use crate::error::Result;
#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
use crate::error::UnsupportedShapeSnafu;

const CAUSAL_CONVOLUTION: &str = "causal_conv_fwd";
#[cfg(feature = "gpu")]
const CAUSAL_CONV_STEP_KERNEL: &str = "causal_conv_step_f32";

#[cfg(all(feature = "gpu", not(logismos_no_gpu_kernels)))]
unsafe extern "C" {
    fn logismos_launch_causal_conv_step_f32_checked(
        input_f32: *const c_void,
        weights_f32: *const c_void,
        history_in_f32: *const c_void,
        history_out_f32: *mut c_void,
        output_f32: *mut c_void,
        channel_count: u32,
        width: u32,
        numerical_status: *mut c_void,
        stream: *mut c_void,
    ) -> u32;
}

/// Result alias for the bounded causal-convolution reference.
pub type CausalConvResult<T> = core::result::Result<T, CausalConvError>;

/// Exact logical `f32` capacities owned by one causal-convolution evaluation.
///
/// WHY: model executors can compose the kernel's checked allocation requests
/// without duplicating its shape arithmetic or mistaking requested capacity for
/// allocator capacity or resident memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CausalConvAllocationPlan {
    token_count: usize,
    channel_count: usize,
    width: usize,
    output: usize,
    weights: usize,
    history: usize,
}

impl CausalConvAllocationPlan {
    /// Derive the allocation requests for one admitted convolution shape.
    ///
    /// # Errors
    ///
    /// Returns [`CausalConvError`] when a required dimension is zero or a
    /// declared-shape calculation overflows.
    pub fn try_from_dimensions(
        token_count: usize,
        channel_count: usize,
        width: usize,
    ) -> CausalConvResult<Self> {
        validate_nonzero_dimension("channel_count", channel_count)?;
        validate_nonzero_dimension("width", width)?;
        let history_width = checked_subtract(width, 1, "width - 1")?;
        Ok(Self {
            token_count,
            channel_count,
            width,
            output: checked_product(token_count, channel_count, "token_count * channel_count")?,
            weights: checked_product(channel_count, width, "channel_count * width")?,
            history: checked_product(channel_count, history_width, "channel_count * (width - 1)")?,
        })
    }

    /// Return the exact requested output capacity.
    #[must_use]
    pub const fn output_elements(self) -> usize {
        self.output
    }

    /// Return the exact requested final-history capacity.
    #[must_use]
    pub const fn history_elements(self) -> usize {
        self.history
    }

    /// Return the admitted token count.
    #[must_use]
    pub const fn token_count(self) -> usize {
        self.token_count
    }

    /// Return the admitted channel count.
    #[must_use]
    pub const fn channel_count(self) -> usize {
        self.channel_count
    }

    /// Return the admitted causal-filter width.
    #[must_use]
    pub const fn width(self) -> usize {
        self.width
    }

    /// Return the exact requested channel-major weight capacity.
    #[must_use]
    pub const fn weight_elements(self) -> usize {
        self.weights
    }
}

/// Failures while admitting or evaluating the bounded causal-convolution reference.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum CausalConvError {
    /// A required channel or kernel-width dimension was zero.
    #[snafu(display("{CAUSAL_CONVOLUTION}: {dimension} must be greater than zero"))]
    ZeroDimension {
        /// The rejected dimension.
        dimension: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Multiplying dimensions or positions could not be represented by `usize`.
    #[snafu(display("{CAUSAL_CONVOLUTION}: {dimensions} overflows usize"))]
    DimensionOverflow {
        /// The operation whose result overflowed.
        dimensions: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A dense input did not match its declared shape.
    #[snafu(display(
        "{CAUSAL_CONVOLUTION}: {input} length {actual} does not match expected {expected}"
    ))]
    LengthMismatch {
        /// Input name.
        input: &'static str,
        /// Required element count.
        expected: usize,
        /// Supplied element count.
        actual: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// An admitted scalar was not finite.
    #[snafu(display("{CAUSAL_CONVOLUTION}: {input}[{index}] is not finite"))]
    NonFiniteInput {
        /// Input name.
        input: &'static str,
        /// Flat element index.
        index: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// An intermediate product, accumulation, or output was not finite.
    #[snafu(display("{CAUSAL_CONVOLUTION}: non-finite value during {stage} at index {index}"))]
    NonFiniteArithmetic {
        /// Named computation stage.
        stage: &'static str,
        /// Flat element index within that stage.
        index: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A bounded-by-shape output allocation could not be reserved.
    #[snafu(display(
        "{CAUSAL_CONVOLUTION}: could not reserve {elements} elements for {allocation}"
    ))]
    Allocation {
        /// Allocation role.
        allocation: &'static str,
        /// Requested element count.
        elements: usize,
        /// Allocation failure reported by the standard library.
        source: std::collections::TryReserveError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}

/// Validated dense input for a causal convolution.
///
/// The supported layout is input `[T, C]`, weights `[C, W]`, and initial
/// raw-input history `[C, W - 1]`; all are row-major `f32`. Taps and history
/// are ordered oldest-to-newest. `T` may be zero, preserving the history.
#[derive(Debug, Clone, Copy)]
pub struct CausalConvInput<'a> {
    input: &'a [f32],
    weights: &'a [f32],
    history: &'a [f32],
    allocations: CausalConvAllocationPlan,
}

impl<'a> CausalConvInput<'a> {
    /// Admit an exact dense causal-convolution input.
    ///
    /// # Errors
    ///
    /// Returns [`CausalConvError`] when a required dimension is zero, a
    /// declared-shape calculation overflows, a buffer length differs from its
    /// declared shape, or a supplied scalar is non-finite.
    pub fn new(
        input: &'a [f32],
        weights: &'a [f32],
        history: &'a [f32],
        token_count: usize,
        channel_count: usize,
        width: usize,
    ) -> CausalConvResult<Self> {
        let allocations =
            CausalConvAllocationPlan::try_from_dimensions(token_count, channel_count, width)?;

        validate_length("input", input.len(), allocations.output_elements())?;
        validate_length("weights", weights.len(), allocations.weight_elements())?;
        validate_length("history", history.len(), allocations.history_elements())?;
        validate_scalars("input", input)?;
        validate_scalars("weights", weights)?;
        validate_scalars("history", history)?;

        Ok(Self {
            input,
            weights,
            history,
            allocations,
        })
    }

    fn history_width(&self) -> CausalConvResult<usize> {
        checked_subtract(self.allocations.width(), 1, "width - 1")
    }

    fn window_sample(&self, channel_index: usize, window_position: usize) -> CausalConvResult<f32> {
        let history_width = self.history_width()?;
        if window_position < history_width {
            let history_start = checked_product(
                channel_index,
                history_width,
                "channel index * history width",
            )?;
            let history_index = checked_add(history_start, window_position, "history index")?;
            return read_scalar(self.history, history_index, "history", self.history.len());
        }

        let token_index = checked_subtract(
            window_position,
            history_width,
            "window position - history width",
        )?;
        let input_start = checked_product(
            token_index,
            self.allocations.channel_count(),
            "token index * channel count",
        )?;
        let input_index = checked_add(input_start, channel_index, "input index")?;
        read_scalar(self.input, input_index, "input", self.input.len())
    }

    fn weight(&self, channel_index: usize, tap_index: usize) -> CausalConvResult<f32> {
        let weight_start = checked_product(
            channel_index,
            self.allocations.width(),
            "channel index * width",
        )?;
        let weight_index = checked_add(weight_start, tap_index, "weight index")?;
        read_scalar(self.weights, weight_index, "weights", self.weights.len())
    }
}

/// Output and final raw-input history from [`causal_conv_fwd`].
#[derive(Debug, Clone, PartialEq)]
pub struct CausalConvOutput {
    output: Vec<f32>,
    history: Vec<f32>,
}

impl CausalConvOutput {
    /// Return the dense `[T, C]` output in row-major order.
    #[must_use]
    pub fn output(&self) -> &[f32] {
        &self.output
    }

    /// Return the final raw-input `[C, W - 1]` history in row-major order.
    #[must_use]
    pub fn history(&self) -> &[f32] {
        &self.history
    }
}

/// Evaluate the bounded causal-convolution reference.
///
/// For every output `[t, c]`, weights `[c, :]` multiply the oldest-to-newest
/// window ending at input `[t, c]`. The first `W - 1` window positions come
/// from the immutable input history. It performs no activation, bias,
/// normalization, packing, or model-specific interpretation.
///
/// # Errors
///
/// Returns [`CausalConvError::NonFiniteArithmetic`] when an otherwise finite
/// product or accumulation exceeds the `f32` domain, or
/// [`CausalConvError::Allocation`] if an output reservation fails.
pub fn causal_conv_fwd(input: &CausalConvInput<'_>) -> CausalConvResult<CausalConvOutput> {
    let history_width = input.history_width()?;
    let mut output = reserve_f32("output", input.allocations.output_elements())?;

    for token_index in 0..input.allocations.token_count() {
        for channel_index in 0..input.allocations.channel_count() {
            let output_index = checked_add(
                checked_product(
                    token_index,
                    input.allocations.channel_count(),
                    "token index * channel count",
                )?,
                channel_index,
                "output index",
            )?;
            let mut accumulator = 0.0_f32;
            for tap_index in 0..input.allocations.width() {
                let window_position =
                    checked_add(token_index, tap_index, "token index + tap index")?;
                let sample = input.window_sample(channel_index, window_position)?;
                let product = sample * input.weight(channel_index, tap_index)?;
                ensure_finite(product, "tap product", output_index)?;
                accumulator += product;
                ensure_finite(accumulator, "output accumulation", output_index)?;
            }
            output.push(accumulator);
        }
    }

    let mut final_history = reserve_f32("final history", input.allocations.history_elements())?;
    for channel_index in 0..input.allocations.channel_count() {
        for history_index in 0..history_width {
            let window_position = checked_add(
                input.allocations.token_count(),
                history_index,
                "token count + history index",
            )?;
            final_history.push(input.window_sample(channel_index, window_position)?);
        }
    }

    Ok(CausalConvOutput {
        output,
        history: final_history,
    })
}

#[cfg(feature = "gpu")]
/// Launch one staged dense-f32 causal-convolution decode step on `stream`.
///
/// `plan` is the sole owner of this operation's shape: the admitted native
/// step has `T = 1`, input/output `[1, C]`, channel-major weights `[C, W]`,
/// and immutable plus separately staged raw histories `[C, W - 1]`. Taps and
/// history are oldest-to-newest. This operation applies no activation, bias,
/// packing, model state, or decoder policy.
///
/// One GPU thread evaluates one channel, serializing tap products and
/// additions in the same order as [`causal_conv_fwd`]. The source-scoped HIP
/// flags disable fast math and contraction. `W = 1` has no history footprint:
/// neither history pointer is accessed by the kernel.
///
/// # Errors
///
/// Returns [`crate::Error::UnsupportedShape`] when `plan` is not a one-token
/// dense-f32 step, a declared length differs from `plan`, a nonempty span is
/// null, unaligned, unrepresentable, overlaps a writable result, or a launch
/// dimension cannot cross the `u32` HIP ABI. A CPU-only build returns
/// [`crate::Error::NoGpuBuild`] without initializing HIP. It propagates
/// stream-current failures and reports a HIP submission failure as
/// [`crate::Error::Launch`].
///
/// # Safety
///
/// Each nonempty pointer must designate a live allocation on `stream`'s
/// device for the exact declared `f32` count through stream completion.
/// `history_in_f32` remains immutable. `history_out_f32` and `output_f32`
/// must not alias one another or any input, and each requires exclusive access
/// through stream completion: no other GPU command or host alias may read or
/// write either span. No producer may modify any input through stream
/// completion.
///
/// Device contents are not inspectable at this boundary. Callers must ensure
/// every input and intermediate is finite and either zero or normal `f32`; in
/// particular every product, accumulation, output, and staged history value
/// must remain finite. Subnormal-dependent behavior is not qualified. The
/// kernel has no status channel and therefore cannot reproduce the CPU
/// reference's non-finite-input or arithmetic refusals.
#[expect(
    clippy::too_many_arguments,
    reason = "the five staged causal-convolution buffers and stream are the fixed native step ABI"
)]
pub unsafe fn launch_causal_conv_step_f32(
    plan: CausalConvAllocationPlan,
    input_f32: *const f32,
    input_elements: usize,
    weights_f32: *const f32,
    weight_elements: usize,
    history_in_f32: *const f32,
    history_in_elements: usize,
    history_out_f32: *mut f32,
    history_out_elements: usize,
    output_f32: *mut f32,
    output_elements: usize,
    stream: &Stream,
) -> Result<()> {
    // SAFETY: the raw caller retains the documented allocation and numerical obligations.
    unsafe {
        launch_causal_conv_step_f32_impl(
            plan,
            input_f32,
            input_elements,
            weights_f32,
            weight_elements,
            history_in_f32,
            history_in_elements,
            history_out_f32,
            history_out_elements,
            output_f32,
            output_elements,
            stream,
            None,
        )
    }
}

#[cfg(feature = "gpu")]
/// Submit the staged operation with sticky input and intermediate numerical checks.
///
/// Submission success does not validate arithmetic. Read `status` after proven
/// stream completion before publishing any staged state or output.
///
/// # Safety
///
/// The allocation, lifetime, aliasing and stream requirements of [`launch_causal_conv_step_f32`]
/// still apply. `status` must be on the same device, nonaliasing and retained
/// through completion. The execution environment must qualify the checked
/// kernel's denorm-preserving compiler and math-library behavior; caller
/// prequalification of each explicit arithmetic intermediate is not required.
#[expect(
    clippy::too_many_arguments,
    reason = "the checked operation retains the exact staged-buffer ABI"
)]
pub unsafe fn launch_causal_conv_step_f32_checked(
    plan: CausalConvAllocationPlan,
    input_f32: *const f32,
    input_elements: usize,
    weights_f32: *const f32,
    weight_elements: usize,
    history_in_f32: *const f32,
    history_in_elements: usize,
    history_out_f32: *mut f32,
    history_out_elements: usize,
    output_f32: *mut f32,
    output_elements: usize,
    stream: &Stream,
    status: &crate::numerical_status::NativeNumericalStatus,
) -> Result<()> {
    // SAFETY: the checked caller retains all device and status allocation obligations.
    unsafe {
        launch_causal_conv_step_f32_impl(
            plan,
            input_f32,
            input_elements,
            weights_f32,
            weight_elements,
            history_in_f32,
            history_in_elements,
            history_out_f32,
            history_out_elements,
            output_f32,
            output_elements,
            stream,
            Some(status),
        )
    }
}

#[cfg(feature = "gpu")]
#[expect(
    clippy::too_many_arguments,
    reason = "one launch owner validates both raw and checked staged-buffer calls"
)]
unsafe fn launch_causal_conv_step_f32_impl(
    plan: CausalConvAllocationPlan,
    input_f32: *const f32,
    input_elements: usize,
    weights_f32: *const f32,
    weight_elements: usize,
    history_in_f32: *const f32,
    history_in_elements: usize,
    history_out_f32: *mut f32,
    history_out_elements: usize,
    output_f32: *mut f32,
    output_elements: usize,
    stream: &Stream,
    status: Option<&crate::numerical_status::NativeNumericalStatus>,
) -> Result<()> {
    #[cfg(logismos_no_gpu_kernels)]
    {
        let _ = (
            status,
            plan,
            input_f32,
            input_elements,
            weights_f32,
            weight_elements,
            history_in_f32,
            history_in_elements,
            history_out_f32,
            history_out_elements,
            output_f32,
            output_elements,
            stream,
        );
        no_gpu_causal_conv_step_refusal()
    }

    #[cfg(not(logismos_no_gpu_kernels))]
    {
        let abi = validate_causal_conv_step_launch(
            plan,
            input_f32,
            input_elements,
            weights_f32,
            weight_elements,
            history_in_f32,
            history_in_elements,
            history_out_f32,
            history_out_elements,
            output_f32,
            output_elements,
        )?;
        let numerical_status = match status {
            Some(status) => {
                // SAFETY: the caller retains this nonaliasing status on the stream device.
                unsafe { status.as_device_ptr().cast::<c_void>() }
            }
            None => core::ptr::null_mut(),
        };
        stream.make_current()?;
        // SAFETY: the caller upholds device ownership, lifetime, concurrent
        // access, and numerical-domain obligations documented above; checked
        // spans and the allocation-plan owner established exact extents,
        // alignment, non-aliasing results, and ABI dimensions.
        let code = unsafe {
            logismos_launch_causal_conv_step_f32_checked(
                input_f32.cast::<c_void>(),
                weights_f32.cast::<c_void>(),
                history_in_f32.cast::<c_void>(),
                history_out_f32.cast::<c_void>(),
                output_f32.cast::<c_void>(),
                abi.channel_count,
                abi.width,
                numerical_status,
                stream.raw().cast::<c_void>(),
            )
        };
        if code == 0 {
            Ok(())
        } else {
            LaunchSnafu {
                kernel: CAUSAL_CONV_STEP_KERNEL,
                kind: hipcore::ErrorKind::from_raw(code),
                code,
            }
            .fail()
        }
    }
}

#[cfg(all(feature = "gpu", logismos_no_gpu_kernels))]
fn no_gpu_causal_conv_step_refusal() -> Result<()> {
    NoGpuBuildSnafu {
        kernel: CAUSAL_CONV_STEP_KERNEL,
    }
    .fail()
}

#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
#[derive(Clone, Copy)]
struct CausalConvStepAbi {
    channel_count: u32,
    width: u32,
}

#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
#[expect(
    clippy::too_many_arguments,
    reason = "validation receives the fixed raw staged causal-convolution ABI without constructing a second shape owner"
)]
fn validate_causal_conv_step_launch(
    plan: CausalConvAllocationPlan,
    input_f32: *const f32,
    input_elements: usize,
    weights_f32: *const f32,
    weight_elements: usize,
    history_in_f32: *const f32,
    history_in_elements: usize,
    history_out_f32: *mut f32,
    history_out_elements: usize,
    output_f32: *mut f32,
    output_elements: usize,
) -> Result<CausalConvStepAbi> {
    if plan.token_count() != 1 {
        return unsupported_causal_conv_step_shape(format!(
            "only dense-f32 T=1 decode steps are supported, got T={}",
            plan.token_count()
        ));
    }
    validate_causal_conv_step_length("input", input_elements, plan.output_elements())?;
    validate_causal_conv_step_length("weights", weight_elements, plan.weight_elements())?;
    validate_causal_conv_step_length("history_in", history_in_elements, plan.history_elements())?;
    validate_causal_conv_step_length("history_out", history_out_elements, plan.history_elements())?;
    validate_causal_conv_step_length("output", output_elements, plan.output_elements())?;

    let inputs = [
        checked_f32_device_span(CAUSAL_CONV_STEP_KERNEL, input_f32, input_elements, "input")?,
        checked_f32_device_span(
            CAUSAL_CONV_STEP_KERNEL,
            weights_f32,
            weight_elements,
            "weights",
        )?,
        checked_f32_device_span(
            CAUSAL_CONV_STEP_KERNEL,
            history_in_f32,
            history_in_elements,
            "history_in",
        )?,
    ];
    let history_out = checked_f32_device_span(
        CAUSAL_CONV_STEP_KERNEL,
        history_out_f32.cast_const(),
        history_out_elements,
        "history_out",
    )?;
    let output = checked_f32_device_span(
        CAUSAL_CONV_STEP_KERNEL,
        output_f32.cast_const(),
        output_elements,
        "output",
    )?;
    for input in inputs {
        reject_overlapping_f32_spans(CAUSAL_CONV_STEP_KERNEL, history_out, input)?;
        reject_overlapping_f32_spans(CAUSAL_CONV_STEP_KERNEL, output, input)?;
    }
    reject_overlapping_f32_spans(CAUSAL_CONV_STEP_KERNEL, history_out, output)?;

    Ok(CausalConvStepAbi {
        channel_count: causal_conv_step_u32("channel_count", plan.channel_count())?,
        width: causal_conv_step_u32("width", plan.width())?,
    })
}

#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
fn validate_causal_conv_step_length(
    name: &'static str,
    actual: usize,
    expected: usize,
) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        unsupported_causal_conv_step_shape(format!(
            "{name} length {actual} does not match allocation-plan extent {expected}"
        ))
    }
}

#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
fn causal_conv_step_u32(name: &'static str, value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        UnsupportedShapeSnafu {
            kernel: CAUSAL_CONV_STEP_KERNEL,
            msg: format!("{name} {value} exceeds the HIP ABI u32 domain"),
        }
        .build()
    })
}

#[cfg(all(feature = "gpu", any(test, not(logismos_no_gpu_kernels))))]
fn unsupported_causal_conv_step_shape<T>(msg: String) -> Result<T> {
    UnsupportedShapeSnafu {
        kernel: CAUSAL_CONV_STEP_KERNEL,
        msg,
    }
    .fail()
}

fn checked_product(left: usize, right: usize, dimensions: &'static str) -> CausalConvResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| DimensionOverflowSnafu { dimensions }.build())
}

fn checked_add(left: usize, right: usize, dimensions: &'static str) -> CausalConvResult<usize> {
    left.checked_add(right)
        .ok_or_else(|| DimensionOverflowSnafu { dimensions }.build())
}

fn checked_subtract(
    left: usize,
    right: usize,
    dimensions: &'static str,
) -> CausalConvResult<usize> {
    left.checked_sub(right)
        .ok_or_else(|| DimensionOverflowSnafu { dimensions }.build())
}

fn validate_nonzero_dimension(dimension: &'static str, value: usize) -> CausalConvResult<()> {
    if value == 0 {
        return ZeroDimensionSnafu { dimension }.fail();
    }
    Ok(())
}

fn validate_length(input: &'static str, actual: usize, expected: usize) -> CausalConvResult<()> {
    if actual != expected {
        return LengthMismatchSnafu {
            input,
            expected,
            actual,
        }
        .fail();
    }
    Ok(())
}

fn validate_scalars(input: &'static str, values: &[f32]) -> CausalConvResult<()> {
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return NonFiniteInputSnafu { input, index }.fail();
        }
    }
    Ok(())
}

fn read_scalar(
    values: &[f32],
    index: usize,
    input: &'static str,
    expected: usize,
) -> CausalConvResult<f32> {
    values.get(index).copied().ok_or_else(|| {
        LengthMismatchSnafu {
            input,
            expected,
            actual: values.len(),
        }
        .build()
    })
}

fn ensure_finite(value: f32, stage: &'static str, index: usize) -> CausalConvResult<()> {
    if !value.is_finite() {
        return NonFiniteArithmeticSnafu { stage, index }.fail();
    }
    Ok(())
}

fn reserve_f32(allocation: &'static str, elements: usize) -> CausalConvResult<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .context(AllocationSnafu {
            allocation,
            elements,
        })?;
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHANNEL_COUNT: usize = 2;
    const TOKEN_COUNT: usize = 4;
    const WIDTH: usize = 3;
    const ORACLE_TOLERANCE: f64 = 1e-6;

    #[test]
    fn asymmetric_taps_pin_oldest_to_newest_order() -> CausalConvResult<()> {
        let input = [7.0, 11.0];
        let weights = [2.0, 3.0, 5.0];
        let history = [10.0, 20.0];
        let admitted = CausalConvInput::new(&input, &weights, &history, 2, 1, 3)?;
        let actual = causal_conv_fwd(&admitted)?;

        assert_eq!(
            actual.output(),
            &[115.0, 116.0],
            "tap ordering must be oldest-to-newest"
        );
        assert_eq!(
            actual.history(),
            &[7.0, 11.0],
            "final history must retain newest raw inputs"
        );
        Ok(())
    }

    #[test]
    fn channels_are_independent() -> CausalConvResult<()> {
        let input = [4.0, 8.0, 6.0, 9.0];
        let weights = [2.0, 3.0, 5.0, 7.0];
        let history = [1.0, 10.0];
        let admitted = CausalConvInput::new(&input, &weights, &history, 2, 2, 2)?;
        let actual = causal_conv_fwd(&admitted)?;

        assert_eq!(
            actual.output(),
            &[14.0, 106.0, 26.0, 103.0],
            "channels must use their own taps"
        );
        assert_eq!(
            actual.history(),
            &[6.0, 9.0],
            "each channel must retain its own history"
        );
        Ok(())
    }

    #[test]
    fn width_one_has_no_history() -> CausalConvResult<()> {
        let input = [2.0, 3.0];
        let weights = [4.0];
        let admitted = CausalConvInput::new(&input, &weights, &[], 2, 1, 1)?;
        let actual = causal_conv_fwd(&admitted)?;

        assert_eq!(
            actual.output(),
            &[8.0, 12.0],
            "width-one output must use the current sample"
        );
        assert!(
            actual.history().is_empty(),
            "width one must return no history"
        );
        Ok(())
    }

    #[test]
    fn hidden_subnormal_tap_remains_a_counterexample_after_normal_output() -> CausalConvResult<()> {
        let history = [f32::MIN_POSITIVE];
        let weights = [0.5_f32, 1.0];
        let input = [1.0_f32];
        let subnormal_tap = history[0] * weights[0];
        assert!(history[0].is_normal(), "history operand must begin normal");
        assert!(weights[0].is_normal(), "weight operand must begin normal");
        assert!(
            subnormal_tap.is_subnormal(),
            "first tap product must be subnormal"
        );

        let admitted = CausalConvInput::new(&input, &weights, &history, 1, 1, 2)?;
        let actual = causal_conv_fwd(&admitted)?;
        assert_eq!(
            actual.output(),
            &[1.0],
            "the final CPU output remains normal"
        );
        assert!(
            actual.output()[0].is_normal(),
            "a normal final output must not hide the earlier subnormal tap"
        );
        Ok(())
    }

    #[test]
    fn empty_sequence_preserves_history() -> CausalConvResult<()> {
        let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let history = [10.0, 20.0, 30.0, 40.0];
        let admitted = CausalConvInput::new(&[], &weights, &history, 0, 2, 3)?;
        let actual = causal_conv_fwd(&admitted)?;

        assert!(
            actual.output().is_empty(),
            "an empty sequence must produce no output"
        );
        assert_eq!(
            actual.history(),
            history,
            "an empty sequence must preserve history"
        );
        Ok(())
    }

    #[test]
    fn causal_convolution_matches_independent_f64_oracle() -> CausalConvResult<()> {
        let input = [0.25, -1.0, 1.5, 0.75, -0.5, 2.0, 1.25, -0.25];
        let weights = [0.5, -1.0, 0.25, 1.25, 0.75, -0.5];
        let history = [-2.0, 0.5, 1.0, -1.5];
        let admitted = CausalConvInput::new(
            &input,
            &weights,
            &history,
            TOKEN_COUNT,
            CHANNEL_COUNT,
            WIDTH,
        )?;
        let actual = causal_conv_fwd(&admitted)?;
        let (expected_output, expected_history) = oracle_causal_conv(
            &input,
            &weights,
            &history,
            TOKEN_COUNT,
            CHANNEL_COUNT,
            WIDTH,
        )?;

        assert_close_f64(actual.output(), &expected_output, "oracle output");
        assert_close_f64(actual.history(), &expected_history, "oracle history");
        Ok(())
    }

    #[test]
    fn every_chunk_partition_matches_full_evaluation() -> CausalConvResult<()> {
        let input = [0.25, -1.0, 1.5, 0.75, -0.5, 2.0, 1.25, -0.25];
        let weights = [0.5, -1.0, 0.25, 1.25, 0.75, -0.5];
        let history = [-2.0, 0.5, 1.0, -1.5];
        let full_input = CausalConvInput::new(
            &input,
            &weights,
            &history,
            TOKEN_COUNT,
            CHANNEL_COUNT,
            WIDTH,
        )?;
        let full = causal_conv_fwd(&full_input)?;

        for split_token in 0..=TOKEN_COUNT {
            let first_token_count = split_token;
            let first_input_len = checked_product(
                first_token_count,
                CHANNEL_COUNT,
                "test first token count * channel count",
            )?;
            let first_admitted = CausalConvInput::new(
                input.get(..first_input_len).ok_or_else(|| {
                    LengthMismatchSnafu {
                        input: "test input prefix",
                        expected: first_input_len,
                        actual: input.len(),
                    }
                    .build()
                })?,
                &weights,
                &history,
                first_token_count,
                CHANNEL_COUNT,
                WIDTH,
            )?;
            let first = causal_conv_fwd(&first_admitted)?;
            let remaining_token_count =
                checked_subtract(TOKEN_COUNT, first_token_count, "test remaining token count")?;
            let second_admitted = CausalConvInput::new(
                input.get(first_input_len..).ok_or_else(|| {
                    LengthMismatchSnafu {
                        input: "test input suffix",
                        expected: input.len(),
                        actual: first_input_len,
                    }
                    .build()
                })?,
                &weights,
                first.history(),
                remaining_token_count,
                CHANNEL_COUNT,
                WIDTH,
            )?;
            let second = causal_conv_fwd(&second_admitted)?;
            let mut joined_output = first.output().to_vec();
            joined_output.extend_from_slice(second.output());

            assert_eq!(
                full.output(),
                joined_output,
                "split at token {split_token} must preserve every output"
            );
            assert_eq!(
                full.history(),
                second.history(),
                "split at token {split_token} must preserve final history"
            );
        }
        Ok(())
    }

    #[test]
    fn output_does_not_depend_on_future_input() -> CausalConvResult<()> {
        let prefix = [0.25, -1.0, 1.5, 0.75];
        let first_input = [0.25, -1.0, 1.5, 0.75, -0.5, 2.0, 1.25, -0.25];
        let second_input = [0.25, -1.0, 1.5, 0.75, 99.0, -200.0, 300.0, -400.0];
        let weights = [0.5, -1.0, 0.25, 1.25, 0.75, -0.5];
        let history = [-2.0, 0.5, 1.0, -1.5];
        let first_admitted = CausalConvInput::new(
            &first_input,
            &weights,
            &history,
            TOKEN_COUNT,
            CHANNEL_COUNT,
            WIDTH,
        )?;
        let second_admitted = CausalConvInput::new(
            &second_input,
            &weights,
            &history,
            TOKEN_COUNT,
            CHANNEL_COUNT,
            WIDTH,
        )?;
        let first = causal_conv_fwd(&first_admitted)?;
        let second = causal_conv_fwd(&second_admitted)?;
        let prefix_len = prefix.len();

        assert_eq!(
            first.output().get(..prefix_len),
            second.output().get(..prefix_len),
            "future samples must not affect the shared output prefix"
        );
        Ok(())
    }

    #[cfg(feature = "gpu")]
    struct ValidCausalConvStepBuffers {
        input: Vec<f32>,
        weights: Vec<f32>,
        history_in: Vec<f32>,
        history_out: Vec<f32>,
        output: Vec<f32>,
    }

    #[cfg(feature = "gpu")]
    impl ValidCausalConvStepBuffers {
        fn from_plan(plan: CausalConvAllocationPlan) -> Self {
            Self {
                input: vec![1.0_f32; plan.output_elements()],
                weights: vec![1.0_f32; plan.weight_elements()],
                history_in: vec![0.0_f32; plan.history_elements()],
                history_out: vec![0.0_f32; plan.history_elements()],
                output: vec![0.0_f32; plan.output_elements()],
            }
        }

        fn validate(&mut self, plan: CausalConvAllocationPlan) -> Result<CausalConvStepAbi> {
            validate_causal_conv_step_launch(
                plan,
                self.input.as_ptr(),
                self.input.len(),
                self.weights.as_ptr(),
                self.weights.len(),
                self.history_in.as_ptr(),
                self.history_in.len(),
                self.history_out.as_mut_ptr(),
                self.history_out.len(),
                self.output.as_mut_ptr(),
                self.output.len(),
            )
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn staged_gpu_step_validates_exact_spans_and_refuses_one_invalidity()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let plan = CausalConvAllocationPlan::try_from_dimensions(1, 2, 3)?;
        let mut buffers = ValidCausalConvStepBuffers::from_plan(plan);
        let abi = buffers.validate(plan)?;
        assert_eq!(
            (abi.channel_count, abi.width),
            (2, 3),
            "the ABI must preserve allocation-owner dimensions"
        );

        let zero_token_plan = CausalConvAllocationPlan::try_from_dimensions(0, 2, 3)?;
        let mut zero_token_buffers = ValidCausalConvStepBuffers::from_plan(zero_token_plan);
        assert!(matches!(
            zero_token_buffers.validate(zero_token_plan),
            Err(crate::Error::UnsupportedShape { .. })
        ));
        let two_token_plan = CausalConvAllocationPlan::try_from_dimensions(2, 2, 3)?;
        let mut two_token_buffers = ValidCausalConvStepBuffers::from_plan(two_token_plan);
        assert!(matches!(
            two_token_buffers.validate(two_token_plan),
            Err(crate::Error::UnsupportedShape { .. })
        ));

        buffers.input.clear();
        assert!(matches!(
            buffers.validate(plan),
            Err(crate::Error::UnsupportedShape { .. })
        ));
        buffers.input.resize(plan.output_elements(), 1.0);
        buffers.weights.clear();
        assert!(matches!(
            buffers.validate(plan),
            Err(crate::Error::UnsupportedShape { .. })
        ));
        buffers.weights.resize(plan.weight_elements(), 1.0);
        buffers.history_in.clear();
        assert!(matches!(
            buffers.validate(plan),
            Err(crate::Error::UnsupportedShape { .. })
        ));
        buffers.history_in.resize(plan.history_elements(), 0.0);
        buffers.history_out.clear();
        assert!(matches!(
            buffers.validate(plan),
            Err(crate::Error::UnsupportedShape { .. })
        ));
        buffers.history_out.resize(plan.history_elements(), 0.0);
        buffers.output.clear();
        assert!(matches!(
            buffers.validate(plan),
            Err(crate::Error::UnsupportedShape { .. })
        ));
        buffers.output.resize(plan.output_elements(), 0.0);

        let plan = CausalConvAllocationPlan::try_from_dimensions(1, 1, 2)?;
        let mut aligned = ValidCausalConvStepBuffers::from_plan(plan);
        let aligned_f32 = [0.0_f32; 2];
        assert!(matches!(
            validate_causal_conv_step_launch(
                plan,
                aligned_f32.as_ptr().wrapping_byte_add(1),
                aligned.input.len(),
                aligned.weights.as_ptr(),
                aligned.weights.len(),
                aligned.history_in.as_ptr(),
                aligned.history_in.len(),
                aligned.history_out.as_mut_ptr(),
                aligned.history_out.len(),
                aligned.output.as_mut_ptr(),
                aligned.output.len(),
            ),
            Err(crate::Error::UnsupportedShape { .. })
        ));
        assert!(matches!(
            validate_causal_conv_step_launch(
                plan,
                aligned.input.as_ptr(),
                aligned.input.len(),
                core::ptr::null(),
                aligned.weights.len(),
                aligned.history_in.as_ptr(),
                aligned.history_in.len(),
                aligned.history_out.as_mut_ptr(),
                aligned.history_out.len(),
                aligned.output.as_mut_ptr(),
                aligned.output.len(),
            ),
            Err(crate::Error::UnsupportedShape { .. })
        ));
        assert!(matches!(
            validate_causal_conv_step_launch(
                plan,
                aligned.input.as_ptr(),
                aligned.input.len(),
                aligned.weights.as_ptr(),
                aligned.weights.len(),
                aligned.history_in.as_ptr(),
                aligned.history_in.len(),
                aligned.output.as_mut_ptr(),
                aligned.output.len(),
                aligned.output.as_mut_ptr(),
                aligned.output.len(),
            ),
            Err(crate::Error::UnsupportedShape { .. })
        ));
        assert!(matches!(
            validate_causal_conv_step_launch(
                plan,
                aligned.input.as_ptr(),
                aligned.input.len(),
                aligned.weights.as_ptr(),
                aligned.weights.len(),
                aligned.history_in.as_ptr(),
                aligned.history_in.len(),
                aligned.history_out.as_mut_ptr(),
                aligned.history_out.len(),
                aligned.input.as_mut_ptr(),
                aligned.input.len(),
            ),
            Err(crate::Error::UnsupportedShape { .. })
        ));
        assert!(matches!(
            checked_f32_device_span(
                CAUSAL_CONV_STEP_KERNEL,
                core::ptr::NonNull::<f32>::dangling().as_ptr(),
                usize::MAX,
                "layout overflow",
            ),
            Err(crate::Error::UnsupportedShape { .. })
        ));
        let address_overflow = (usize::MAX - 3) as *const f32;
        assert!(matches!(
            checked_f32_device_span(
                CAUSAL_CONV_STEP_KERNEL,
                address_overflow,
                1,
                "address overflow"
            ),
            Err(crate::Error::UnsupportedShape { .. })
        ));
        if let Ok(abi_overflow) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert!(matches!(
                causal_conv_step_u32("abi overflow", abi_overflow),
                Err(crate::Error::UnsupportedShape { .. })
            ));
        }
        Ok(())
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn width_one_step_has_absent_null_history_spans()
    -> core::result::Result<(), Box<dyn std::error::Error>> {
        let plan = CausalConvAllocationPlan::try_from_dimensions(1, 2, 1)?;
        let mut buffers = ValidCausalConvStepBuffers::from_plan(plan);
        validate_causal_conv_step_launch(
            plan,
            buffers.input.as_ptr(),
            buffers.input.len(),
            buffers.weights.as_ptr(),
            buffers.weights.len(),
            core::ptr::null(),
            0,
            core::ptr::null_mut(),
            0,
            buffers.output.as_mut_ptr(),
            buffers.output.len(),
        )?;
        Ok(())
    }

    #[cfg(all(feature = "gpu", logismos_no_gpu_kernels))]
    #[test]
    fn staged_gpu_step_cpu_only_witness_never_initializes_hip() {
        assert!(matches!(
            no_gpu_causal_conv_step_refusal(),
            Err(crate::Error::NoGpuBuild { .. })
        ));
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "requires an explicitly reserved HIP device; absent devices are a failure"]
    fn initialized_status_classifies_causal_step_inputs_and_hidden_intermediates()
    -> core::result::Result<(), String> {
        use hipcore::{Device, DeviceBuffer, Stream};

        let device = Device::new(0).map_err(|error| format!("open reserved device 0: {error}"))?;
        let stream = Stream::new(&device).map_err(|error| format!("create stream: {error}"))?;
        let plan = CausalConvAllocationPlan::try_from_dimensions(1, 1, 2)
            .map_err(|error| format!("build causal status plan: {error}"))?;
        let cases = [
            ("clean", [1.0_f32], [0.5_f32, 1.0], [1.0_f32], 0_u32),
            (
                "input subnormal",
                [f32::from_bits(1)],
                [1.0_f32, 1.0],
                [1.0_f32],
                crate::numerical_status::NativeNumericalStatusCategory::InputSubnormal.bit()
                    | crate::numerical_status::NativeNumericalStatusCategory::ArithmeticSubnormal
                        .bit(),
            ),
            (
                "hidden arithmetic underflow",
                [1.0_f32],
                [0.5_f32, 1.0],
                [f32::MIN_POSITIVE],
                crate::numerical_status::NativeNumericalStatusCategory::ArithmeticSubnormal.bit(),
            ),
            (
                "input nonfinite",
                [f32::INFINITY],
                [1.0_f32, 1.0],
                [1.0_f32],
                crate::numerical_status::NativeNumericalStatusCategory::InputNonFinite.bit()
                    | crate::numerical_status::NativeNumericalStatusCategory::ArithmeticNonFinite
                        .bit(),
            ),
            (
                "arithmetic overflow",
                [f32::MAX],
                [0.5_f32, 2.0],
                [1.0_f32],
                crate::numerical_status::NativeNumericalStatusCategory::ArithmeticNonFinite.bit(),
            ),
        ];

        for (name, input_host, weights_host, history_host, expected_bits) in cases {
            let input = DeviceBuffer::from_host(&device, &input_host)
                .map_err(|error| format!("{name}: upload input: {error}"))?;
            let weights = DeviceBuffer::from_host(&device, &weights_host)
                .map_err(|error| format!("{name}: upload weights: {error}"))?;
            let history_in = DeviceBuffer::from_host(&device, &history_host)
                .map_err(|error| format!("{name}: upload history: {error}"))?;
            let history_out = DeviceBuffer::from_host(&device, &[-1234.5_f32])
                .map_err(|error| format!("{name}: initialize staged history: {error}"))?;
            let output = DeviceBuffer::from_host(&device, &[-1234.5_f32])
                .map_err(|error| format!("{name}: initialize staged output: {error}"))?;
            let status = crate::numerical_status::NativeNumericalStatus::new(&device)
                .map_err(|error| format!("{name}: initialize status: {error}"))?;
            // SAFETY: these distinct owned buffers match the checked plan and remain live through synchronization.
            unsafe {
                launch_causal_conv_step_f32_checked(
                    plan,
                    input.as_device_ptr(),
                    input.len(),
                    weights.as_device_ptr(),
                    weights.len(),
                    history_in.as_device_ptr(),
                    history_in.len(),
                    history_out.as_device_ptr(),
                    history_out.len(),
                    output.as_device_ptr(),
                    output.len(),
                    &stream,
                    &status,
                )
            }
            .map_err(|error| format!("{name}: launch checked causal step: {error}"))?;
            stream
                .synchronize()
                .map_err(|error| format!("{name}: synchronize: {error}"))?;
            assert_eq!(
                native_status_bits(&status)?,
                expected_bits,
                "{name}: initialized status must contain the exact typed mask"
            );
            assert_device_values(&input, &input_host, name, "input")?;
            assert_device_values(&weights, &weights_host, name, "weights")?;
            assert_device_values(&history_in, &history_host, name, "history")?;
            if expected_bits == 0 {
                let admitted =
                    CausalConvInput::new(&input_host, &weights_host, &history_host, 1, 1, 2)
                        .map_err(|error| format!("{name}: admit CPU oracle: {error}"))?;
                let expected = causal_conv_fwd(&admitted)
                    .map_err(|error| format!("{name}: evaluate CPU oracle: {error}"))?;
                assert_device_values(&output, expected.output(), name, "clean output")?;
                assert_device_values(&history_out, expected.history(), name, "clean history")?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "requires an explicitly reserved HIP device; absent devices are a failure"]
    fn reserved_device_step_matches_oracle_continuation_and_width_one()
    -> core::result::Result<(), String> {
        use hipcore::{Device, DeviceBuffer, Stream};

        let device = Device::new(0).map_err(|error| format!("open reserved device 0: {error}"))?;
        let stream = Stream::new(&device).map_err(|error| format!("create stream: {error}"))?;
        let weights_host = [0.5, -1.0, 0.25, 1.25, 0.75, -0.5, -0.25, 0.5, 1.0];
        let initial_history = [-2.0, 0.5, 1.0, -1.5, 0.25, 2.0];
        let first_input = [0.25, -1.0, 1.5];
        let second_input = [0.75, -0.5, 2.0];
        let plan = CausalConvAllocationPlan::try_from_dimensions(1, 3, 3)
            .map_err(|error| format!("build one-token plan: {error}"))?;
        let (first_output_oracle, first_history_oracle) =
            oracle_causal_conv(&first_input, &weights_host, &initial_history, 1, 3, 3)
                .map_err(|error| format!("first oracle: {error}"))?;
        let first_history_expected: Vec<f32> = first_history_oracle
            .iter()
            .map(|value| *value as f32)
            .collect();
        let (second_output_oracle, second_history_oracle) = oracle_causal_conv(
            &second_input,
            &weights_host,
            &first_history_expected,
            1,
            3,
            3,
        )
        .map_err(|error| format!("continuation oracle: {error}"))?;

        let input = DeviceBuffer::<f32>::from_host(&device, &first_input)
            .map_err(|error| format!("upload first input: {error}"))?;
        let weights = DeviceBuffer::<f32>::from_host(&device, &weights_host)
            .map_err(|error| format!("upload weights: {error}"))?;
        let history_in = DeviceBuffer::<f32>::from_host(&device, &initial_history)
            .map_err(|error| format!("upload initial history: {error}"))?;
        let history_out = DeviceBuffer::<f32>::alloc(&device, plan.history_elements())
            .map_err(|error| format!("allocate staged history: {error}"))?;
        let output = DeviceBuffer::<f32>::alloc(&device, plan.output_elements())
            .map_err(|error| format!("allocate staged output: {error}"))?;
        // SAFETY: these distinct device buffers have the exact plan extents;
        // the test synchronizes before any host read or reuse.
        unsafe {
            launch_causal_conv_step_f32(
                plan,
                input.as_device_ptr(),
                input.len(),
                weights.as_device_ptr(),
                weights.len(),
                history_in.as_device_ptr(),
                history_in.len(),
                history_out.as_device_ptr(),
                history_out.len(),
                output.as_device_ptr(),
                output.len(),
                &stream,
            )
        }
        .map_err(|error| format!("launch first causal-convolution step: {error}"))?;
        stream
            .synchronize()
            .map_err(|error| format!("synchronize first step: {error}"))?;
        let mut first_output = vec![0.0_f32; output.len()];
        output
            .copy_to_host(&mut first_output)
            .map_err(|error| format!("read first output: {error}"))?;
        let mut first_history = vec![0.0_f32; history_out.len()];
        history_out
            .copy_to_host(&mut first_history)
            .map_err(|error| format!("read first staged history: {error}"))?;
        let mut preserved_first_input = vec![0.0_f32; input.len()];
        input
            .copy_to_host(&mut preserved_first_input)
            .map_err(|error| format!("read immutable first input: {error}"))?;
        let mut preserved_initial_history = vec![0.0_f32; history_in.len()];
        history_in
            .copy_to_host(&mut preserved_initial_history)
            .map_err(|error| format!("read immutable initial history: {error}"))?;
        let mut preserved_weights = vec![0.0_f32; weights.len()];
        weights
            .copy_to_host(&mut preserved_weights)
            .map_err(|error| format!("read immutable weights: {error}"))?;
        assert_close_f64(&first_output, &first_output_oracle, "device first output");
        assert_close_f64(
            &first_history,
            &first_history_oracle,
            "device first history",
        );
        assert_eq!(
            preserved_first_input, first_input,
            "device input must remain immutable"
        );
        assert_eq!(
            preserved_initial_history, initial_history,
            "device initial history must remain immutable"
        );
        assert_eq!(
            preserved_weights, weights_host,
            "device weights must remain immutable"
        );

        let second_input = DeviceBuffer::<f32>::from_host(&device, &second_input)
            .map_err(|error| format!("upload continuation input: {error}"))?;
        let second_history_out = DeviceBuffer::<f32>::alloc(&device, plan.history_elements())
            .map_err(|error| format!("allocate continuation history: {error}"))?;
        let second_output = DeviceBuffer::<f32>::alloc(&device, plan.output_elements())
            .map_err(|error| format!("allocate continuation output: {error}"))?;
        // SAFETY: the first staged history is immutable input to this distinct
        // continuation result pair, with exact plan extents through completion.
        unsafe {
            launch_causal_conv_step_f32(
                plan,
                second_input.as_device_ptr(),
                second_input.len(),
                weights.as_device_ptr(),
                weights.len(),
                history_out.as_device_ptr(),
                history_out.len(),
                second_history_out.as_device_ptr(),
                second_history_out.len(),
                second_output.as_device_ptr(),
                second_output.len(),
                &stream,
            )
        }
        .map_err(|error| format!("launch continuation causal-convolution step: {error}"))?;
        stream
            .synchronize()
            .map_err(|error| format!("synchronize continuation step: {error}"))?;
        let mut actual_second_output = vec![0.0_f32; second_output.len()];
        second_output
            .copy_to_host(&mut actual_second_output)
            .map_err(|error| format!("read continuation output: {error}"))?;
        let mut actual_second_history = vec![0.0_f32; second_history_out.len()];
        second_history_out
            .copy_to_host(&mut actual_second_history)
            .map_err(|error| format!("read continuation history: {error}"))?;
        let mut preserved_first_history = vec![0.0_f32; history_out.len()];
        history_out
            .copy_to_host(&mut preserved_first_history)
            .map_err(|error| format!("read immutable first staged history: {error}"))?;
        assert_close_f64(
            &actual_second_output,
            &second_output_oracle,
            "device continuation output",
        );
        assert_close_f64(
            &actual_second_history,
            &second_history_oracle,
            "device continuation history",
        );
        assert_close_f64(
            &preserved_first_history,
            &first_history_oracle,
            "device immutable first staged history",
        );

        let width_one_input = [2.0_f32, -3.0];
        let width_one_weights = [4.0_f32, -0.5];
        let width_one_plan = CausalConvAllocationPlan::try_from_dimensions(1, 2, 1)
            .map_err(|error| format!("build width-one plan: {error}"))?;
        let (width_one_output_oracle, width_one_history_oracle) =
            oracle_causal_conv(&width_one_input, &width_one_weights, &[], 1, 2, 1)
                .map_err(|error| format!("width-one oracle: {error}"))?;
        let width_one_input = DeviceBuffer::<f32>::from_host(&device, &width_one_input)
            .map_err(|error| format!("upload width-one input: {error}"))?;
        let width_one_weights = DeviceBuffer::<f32>::from_host(&device, &width_one_weights)
            .map_err(|error| format!("upload width-one weights: {error}"))?;
        let width_one_output =
            DeviceBuffer::<f32>::alloc(&device, width_one_plan.output_elements())
                .map_err(|error| format!("allocate width-one output: {error}"))?;
        // SAFETY: width one has no history footprint, so null history pointers
        // carry zero lengths while the remaining distinct buffers match `plan`.
        unsafe {
            launch_causal_conv_step_f32(
                width_one_plan,
                width_one_input.as_device_ptr(),
                width_one_input.len(),
                width_one_weights.as_device_ptr(),
                width_one_weights.len(),
                core::ptr::null(),
                0,
                core::ptr::null_mut(),
                0,
                width_one_output.as_device_ptr(),
                width_one_output.len(),
                &stream,
            )
        }
        .map_err(|error| format!("launch width-one causal-convolution step: {error}"))?;
        stream
            .synchronize()
            .map_err(|error| format!("synchronize width-one step: {error}"))?;
        let mut actual_width_one_output = vec![0.0_f32; width_one_output.len()];
        width_one_output
            .copy_to_host(&mut actual_width_one_output)
            .map_err(|error| format!("read width-one output: {error}"))?;
        assert_close_f64(
            &actual_width_one_output,
            &width_one_output_oracle,
            "device width-one output",
        );
        assert!(
            width_one_history_oracle.is_empty(),
            "width one must retain no history"
        );
        Ok(())
    }

    #[test]
    fn malformed_nonfinite_overflow_and_arithmetic_inputs_are_rejected() -> CausalConvResult<()> {
        let finite = [1.0_f32; 4];
        let malformed = CausalConvInput::new(&finite[..3], &finite, &finite[..2], 2, 2, 2);
        assert!(
            matches!(
                malformed,
                Err(CausalConvError::LengthMismatch { input: "input", .. })
            ),
            "mismatched input length must fail admission"
        );

        let zero_channel = CausalConvInput::new(&[], &[], &[], 0, 0, 1);
        assert!(
            matches!(
                zero_channel,
                Err(CausalConvError::ZeroDimension {
                    dimension: "channel_count",
                    ..
                })
            ),
            "zero channel count must fail admission"
        );
        let zero_width = CausalConvInput::new(&[], &[], &[], 0, 1, 0);
        assert!(
            matches!(
                zero_width,
                Err(CausalConvError::ZeroDimension {
                    dimension: "width",
                    ..
                })
            ),
            "zero width must fail admission"
        );

        let overflowing_shape = CausalConvInput::new(&[], &[], &[], 0, 2, usize::MAX);
        assert!(
            matches!(
                overflowing_shape,
                Err(CausalConvError::DimensionOverflow {
                    dimensions: "channel_count * width",
                    ..
                })
            ),
            "overflowing shape product must fail admission"
        );

        let nonfinite = CausalConvInput::new(&[1.0], &[f32::NAN], &[], 1, 1, 1);
        assert!(
            matches!(
                nonfinite,
                Err(CausalConvError::NonFiniteInput {
                    input: "weights",
                    ..
                })
            ),
            "non-finite weights must fail admission"
        );
        let nonfinite_input = CausalConvInput::new(&[f32::INFINITY], &[1.0], &[], 1, 1, 1);
        assert!(
            matches!(
                nonfinite_input,
                Err(CausalConvError::NonFiniteInput { input: "input", .. })
            ),
            "non-finite input must fail admission"
        );
        let nonfinite_history =
            CausalConvInput::new(&[1.0], &[1.0, 1.0], &[f32::NEG_INFINITY], 1, 1, 2);
        assert!(
            matches!(
                nonfinite_history,
                Err(CausalConvError::NonFiniteInput {
                    input: "history",
                    ..
                })
            ),
            "non-finite history must fail admission"
        );

        let initial_history = [f32::MAX];
        let arithmetic = CausalConvInput::new(&[1.0], &[f32::MAX, 1.0], &initial_history, 1, 1, 2)?;
        let arithmetic_result = causal_conv_fwd(&arithmetic);
        assert!(
            matches!(
                arithmetic_result,
                Err(CausalConvError::NonFiniteArithmetic {
                    stage: "tap product",
                    ..
                })
            ),
            "non-finite arithmetic must return no partial result"
        );
        assert_eq!(
            initial_history
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![f32::MAX.to_bits()],
            "the immutable caller history must remain bitwise unchanged after rejection"
        );
        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn native_status_bits(
        status: &crate::numerical_status::NativeNumericalStatus,
    ) -> core::result::Result<u32, String> {
        match status.read_after_synchronization() {
            Ok(()) => Ok(0),
            Err(crate::Error::NumericalStatus {
                source: crate::numerical_status::NativeNumericalStatusError::Observed { mask },
                ..
            }) => Ok(mask.bits()),
            Err(error) => Err(format!("read typed native status: {error}")),
        }
    }

    #[cfg(feature = "gpu")]
    fn assert_device_values(
        buffer: &hipcore::DeviceBuffer<f32>,
        expected: &[f32],
        case: &str,
        name: &str,
    ) -> core::result::Result<(), String> {
        let mut actual = vec![0.0_f32; buffer.len()];
        buffer
            .copy_to_host(&mut actual)
            .map_err(|error| format!("{case}: read {name}: {error}"))?;
        assert_eq!(actual, expected, "{case}: {name} must match exactly");
        Ok(())
    }

    fn oracle_causal_conv(
        input: &[f32],
        weights: &[f32],
        history: &[f32],
        token_count: usize,
        channel_count: usize,
        width: usize,
    ) -> CausalConvResult<(Vec<f64>, Vec<f64>)> {
        let history_width = checked_subtract(width, 1, "oracle width - 1")?;
        let output_len = checked_product(
            token_count,
            channel_count,
            "oracle token count * channel count",
        )?;
        let history_len = checked_product(
            channel_count,
            history_width,
            "oracle channel count * history width",
        )?;
        let weight_len = checked_product(channel_count, width, "oracle channel count * width")?;
        let mut output = Vec::with_capacity(output_len);
        for token_index in 0..token_count {
            for channel_index in 0..channel_count {
                let mut accumulator = 0.0_f64;
                for tap_index in 0..width {
                    let position =
                        checked_add(token_index, tap_index, "oracle token index + tap index")?;
                    let sample = if position < history_width {
                        let index = checked_add(
                            checked_product(
                                channel_index,
                                history_width,
                                "oracle channel * history width",
                            )?,
                            position,
                            "oracle history index",
                        )?;
                        history.get(index).copied().ok_or_else(|| {
                            LengthMismatchSnafu {
                                input: "oracle history",
                                expected: history_len,
                                actual: history.len(),
                            }
                            .build()
                        })?
                    } else {
                        let token = checked_subtract(
                            position,
                            history_width,
                            "oracle position - history width",
                        )?;
                        let index = checked_add(
                            checked_product(token, channel_count, "oracle token * channel count")?,
                            channel_index,
                            "oracle input index",
                        )?;
                        input.get(index).copied().ok_or_else(|| {
                            LengthMismatchSnafu {
                                input: "oracle input",
                                expected: output_len,
                                actual: input.len(),
                            }
                            .build()
                        })?
                    };
                    let weight_index = checked_add(
                        checked_product(channel_index, width, "oracle channel * width")?,
                        tap_index,
                        "oracle weight index",
                    )?;
                    let weight = weights.get(weight_index).copied().ok_or_else(|| {
                        LengthMismatchSnafu {
                            input: "oracle weights",
                            expected: weight_len,
                            actual: weights.len(),
                        }
                        .build()
                    })?;
                    accumulator += f64::from(sample) * f64::from(weight);
                }
                output.push(accumulator);
            }
        }

        let mut final_history = Vec::with_capacity(history_len);
        for channel_index in 0..channel_count {
            for history_index in 0..history_width {
                let position = checked_add(
                    token_count,
                    history_index,
                    "oracle token count + history index",
                )?;
                let sample = if position < history_width {
                    let index = checked_add(
                        checked_product(
                            channel_index,
                            history_width,
                            "oracle channel * history width",
                        )?,
                        position,
                        "oracle history index",
                    )?;
                    history.get(index).copied().ok_or_else(|| {
                        LengthMismatchSnafu {
                            input: "oracle history",
                            expected: history_len,
                            actual: history.len(),
                        }
                        .build()
                    })?
                } else {
                    let token = checked_subtract(
                        position,
                        history_width,
                        "oracle position - history width",
                    )?;
                    let index = checked_add(
                        checked_product(token, channel_count, "oracle token * channel count")?,
                        channel_index,
                        "oracle input index",
                    )?;
                    input.get(index).copied().ok_or_else(|| {
                        LengthMismatchSnafu {
                            input: "oracle input",
                            expected: output_len,
                            actual: input.len(),
                        }
                        .build()
                    })?
                };
                final_history.push(f64::from(sample));
            }
        }
        Ok((output, final_history))
    }

    fn assert_close_f64(actual: &[f32], expected: &[f64], name: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{name} length must match oracle"
        );
        for (index, (actual_value, expected_value)) in actual.iter().zip(expected).enumerate() {
            let difference = (f64::from(*actual_value) - *expected_value).abs();
            assert!(
                difference <= ORACLE_TOLERANCE,
                "{name}[{index}] differs from independent oracle by {difference}"
            );
        }
    }
}
