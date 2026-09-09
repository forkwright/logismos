//! `kernels` error surface.

use snafu::Snafu;

#[cfg(feature = "gpu")]
use crate::numerical_status::NativeNumericalStatusError;

/// Result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// Validate a softmax input extent before either CPU implementation indexes it.
///
/// Both CPU softmax implementations keep their numerical loops independent, but
/// share this boundary contract so a malformed shape receives the same typed
/// error in every build profile.
pub(crate) fn checked_softmax_input_elements(
    kernel: &'static str,
    rows: usize,
    width: usize,
    actual_len: usize,
) -> Result<usize> {
    if width == 0 {
        return SoftmaxInvalidDimensionSnafu {
            kernel,
            rows,
            width,
        }
        .fail();
    }
    let expected_len = rows.checked_mul(width).ok_or_else(|| {
        SoftmaxSizeOverflowSnafu {
            kernel,
            rows,
            width,
        }
        .build()
    })?;
    if actual_len != expected_len {
        return SoftmaxShapeSnafu {
            kernel,
            rows,
            width,
            expected_len,
            actual_len,
        }
        .fail();
    }
    Ok(expected_len)
}

/// The checked step of the CPU RMSNorm reference that rejected an input or result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RmsNormStage {
    /// Input activation values or their declared extent.
    Input,
    /// Learned scale values or their declared extent.
    Weight,
    /// Squaring an input activation.
    Square,
    /// Accumulating squared activations.
    Sum,
    /// Dividing the squared sum by the row width.
    Mean,
    /// Adding the numerical epsilon.
    Epsilon,
    /// Taking the reciprocal root mean square.
    Inverse,
    /// Scaling an activation by the reciprocal root mean square.
    Scale,
    /// Scaling the normalized activation by its learned weight.
    Output,
}

/// The checked unit-normalization step that observed a non-finite value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnitNormalizationStage {
    /// A value supplied by the caller.
    Input,
    /// The accumulated squared L2 norm.
    Accumulation,
    /// A value rounded back to the output `f32` representation.
    Scale,
}

/// Errors surfaced by the kernel launchers and CPU references.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum Error {
    #[cfg(feature = "gpu")]
    /// Propagated HIP failure.
    #[snafu(transparent)]
    Hip {
        /// Source HIP error.
        source: hipcore::Error,
    },

    #[cfg(feature = "gpu")]
    /// Propagated tensor failure.
    #[snafu(transparent)]
    Taxis {
        /// Source tensor error.
        source: taxis::Error,
    },

    #[cfg(feature = "gpu")]
    /// Checked native arithmetic reported an invalid numerical-domain value.
    #[snafu(display("native numerical status failure: {source}"))]
    NumericalStatus {
        /// Typed native numerical-status failure.
        source: NativeNumericalStatusError,
        /// Source code location where the status was read.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Propagated checked serialized-row format failure.
    #[snafu(display("serialized-row projection format failure: {source}"))]
    Quant {
        /// Source quantization error.
        source: quant::Error,
        /// Source code location where the error was propagated.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    #[cfg(feature = "gpu")]
    /// Kernel launch failed — HIP reported a non-success status after
    /// kernel submission.
    #[snafu(display("kernel {kernel}: launch failed: {kind:?} (code {code})"))]
    Launch {
        /// Symbolic kernel name.
        kernel: &'static str,
        /// Classified HIP error kind for `code`, so a launch failure
        /// is diagnosable without cross-referencing the HIP headers.
        kind: hipcore::ErrorKind,
        /// Raw HIP error code.
        code: u32,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Tensor shape is not supported by this kernel.
    #[snafu(display("kernel {kernel}: unsupported shape: {msg}"))]
    UnsupportedShape {
        /// Symbolic kernel name.
        kernel: &'static str,
        /// Description.
        msg: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// CPU softmax received a zero-width last axis.
    #[snafu(display("softmax {kernel} rejects rows={rows}, width={width}"))]
    SoftmaxInvalidDimension {
        /// Symbolic implementation name.
        kernel: &'static str,
        /// Declared row count.
        rows: usize,
        /// Declared last-axis width.
        width: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// CPU softmax's declared element count overflowed `usize`.
    #[snafu(display("softmax {kernel} element count overflows for rows={rows}, width={width}"))]
    SoftmaxSizeOverflow {
        /// Symbolic implementation name.
        kernel: &'static str,
        /// Declared row count.
        rows: usize,
        /// Declared last-axis width.
        width: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// CPU softmax's supplied input extent did not match its declared shape.
    #[snafu(display(
        "softmax {kernel} input length {actual_len} does not equal expected {expected_len} for rows={rows}, width={width}"
    ))]
    SoftmaxShape {
        /// Symbolic implementation name.
        kernel: &'static str,
        /// Declared row count.
        rows: usize,
        /// Declared last-axis width.
        width: usize,
        /// Computed required length.
        expected_len: usize,
        /// Supplied input length.
        actual_len: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// CPU softmax received a non-finite value other than an intentional mask.
    #[snafu(display(
        "softmax {kernel} rejects non-finite value {value} at row {row}, column {column}"
    ))]
    SoftmaxNonFinite {
        /// Symbolic implementation name.
        kernel: &'static str,
        /// Row containing the rejected value.
        row: usize,
        /// Column containing the rejected value.
        column: usize,
        /// Rejected value (`NaN` or positive infinity).
        value: f32,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// CPU softmax could not reserve its output backing.
    #[snafu(display("softmax {kernel} output allocation for {requested_len} elements failed"))]
    SoftmaxAllocation {
        /// Symbolic implementation name.
        kernel: &'static str,
        /// Requested output element count.
        requested_len: usize,
        /// Allocation failure.
        source: std::collections::TryReserveError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// CPU RMSNorm received a zero-width tensor dimension.
    #[snafu(display("RMSNorm rejects rows={rows}, width={width}"))]
    RmsNormInvalidDimension {
        /// Declared row count.
        rows: usize,
        /// Declared row width.
        width: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// CPU RMSNorm's declared element count overflowed `usize`.
    #[snafu(display("RMSNorm element count overflows for rows={rows}, width={width}"))]
    RmsNormSizeOverflow {
        /// Declared row count.
        rows: usize,
        /// Declared row width.
        width: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// CPU RMSNorm's supplied slice length did not match its declared shape.
    #[snafu(display(
        "RMSNorm {stage:?} length {actual_len} does not equal expected {expected_len} for rows={rows}, width={width}"
    ))]
    RmsNormShape {
        /// Input or weight extent that failed validation.
        stage: RmsNormStage,
        /// Declared row count.
        rows: usize,
        /// Declared row width.
        width: usize,
        /// Computed required length.
        expected_len: usize,
        /// Supplied slice length.
        actual_len: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// CPU RMSNorm received an invalid scalar parameter.
    #[snafu(display("RMSNorm rejects epsilon {epsilon}"))]
    RmsNormInvalidParameter {
        /// Rejected parameter value.
        epsilon: f32,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// CPU RMSNorm observed a non-finite input or intermediate value.
    #[snafu(display(
        "RMSNorm {stage:?} produced or received non-finite value {value} at row {row}, column {column}"
    ))]
    RmsNormNonFinite {
        /// Checked RMSNorm stage.
        stage: RmsNormStage,
        /// Row containing the rejected value.
        row: usize,
        /// Column containing the rejected value.
        column: usize,
        /// Rejected non-finite value.
        value: f32,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// CPU RMSNorm could not reserve its output backing.
    #[snafu(display("RMSNorm output allocation for {requested_len} elements failed"))]
    RmsNormAllocation {
        /// Requested output element count.
        requested_len: usize,
        /// Allocation failure.
        source: std::collections::TryReserveError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// CPU unit normalization observed a non-finite input or intermediate.
    #[snafu(display(
        "unit normalization {stage:?} rejected non-finite value {value} at index {index}"
    ))]
    UnitNormalizationNonFinite {
        /// Checked unit-normalization step.
        stage: UnitNormalizationStage,
        /// Input position associated with the rejected value.
        index: usize,
        /// Rejected non-finite value represented at accumulator precision.
        value: f64,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// CPU unit normalization received an empty or all-zero vector.
    #[snafu(display("unit normalization rejects zero L2 norm for {elements} elements"))]
    UnitNormalizationZeroNorm {
        /// Number of input elements whose squared norm was zero.
        elements: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Rounded CPU unit-normalization output did not meet its unit-norm contract.
    #[snafu(display(
        "unit normalization output norm {norm} exceeds tolerance {tolerance} from one"
    ))]
    UnitNormalizationNonUnit {
        /// L2 norm recomputed from the rounded `f32` output values.
        norm: f64,
        /// Maximum accepted absolute distance from one.
        tolerance: f64,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A checked CPU elementwise reference could not reserve its result.
    #[snafu(display("CPU {operation} allocation for {requested_len} elements failed"))]
    CpuF32Allocation {
        /// Elementwise operation requesting output storage.
        operation: &'static str,
        /// Exact output element count.
        requested_len: usize,
        /// Allocation failure.
        source: std::collections::TryReserveError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// CPU elementwise operands did not share one exact extent.
    #[snafu(display("CPU {operation} operand lengths differ: {left} versus {right}"))]
    CpuF32Shape {
        /// Elementwise operation rejecting the input lengths.
        operation: &'static str,
        /// Left operand length.
        left: usize,
        /// Right operand length.
        right: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// CPU serialized-row GEMV could not reserve its checked output backing.
    #[snafu(display("serialized-row GEMV output allocation for {requested_len} rows failed"))]
    RowGemvAllocation {
        /// Exact requested output row count.
        requested_len: usize,
        /// Allocation failure.
        source: std::collections::TryReserveError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    #[cfg(feature = "gpu")]
    /// Build was produced without the HIP kernel archive (e.g. `hipcc`
    /// was absent). CPU references still work; GPU paths return this.
    #[snafu(display("kernel {kernel}: no-GPU build (set HIPCC or install ROCm to enable)"))]
    NoGpuBuild {
        /// Symbolic kernel name.
        kernel: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    #[test]
    fn launch_error_carries_symbolic_kind() {
        // WHY(forkwright/logismos#59): `Error::Launch` used to carry
        // only the raw `u32` HIP code, so diagnosing a launch failure
        // meant manually cross-referencing the HIP headers. `kind` is
        // derived from the same code via `hipcore::ErrorKind::from_raw`
        // (the mapping this finding says already existed) and its
        // `Display` now names the failure instead of just the number.
        let err = LaunchSnafu {
            kernel: "matmul_naive_fp16",
            kind: hipcore::ErrorKind::from_raw(2), // hipErrorOutOfMemory
            code: 2u32,
        }
        .build();
        assert_eq!(
            hipcore::ErrorKind::from_raw(2),
            hipcore::ErrorKind::OutOfMemory
        );
        let rendered = err.to_string();
        assert!(
            rendered.contains("OutOfMemory"),
            "Display must name the classified kind, got: {rendered}"
        );
        assert!(
            rendered.contains('2'),
            "Display must still carry the raw code, got: {rendered}"
        );
    }
}
