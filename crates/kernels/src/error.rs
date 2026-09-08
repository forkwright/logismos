//! `kernels` error surface.

use snafu::Snafu;

/// Result alias.
pub type Result<T> = core::result::Result<T, Error>;

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

/// The checked step of the CPU softmax reference that rejected an input or result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SoftmaxStage {
    /// An input logit before masking policy is applied.
    Input,
    /// An exponential intermediate.
    Exponent,
    /// The row denominator.
    Denominator,
    /// A normalized output probability.
    Output,
}

/// The checked step of in-place CPU L2 normalization that rejected a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum L2NormalizeStage {
    /// An input vector component.
    Input,
    /// A scaled squared component used for robust accumulation.
    Square,
    /// The scaled sum of squares.
    Sum,
    /// The derived vector norm.
    Norm,
    /// A normalized output component.
    Output,
    /// The final unit-norm postcondition.
    Verification,
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

    /// Propagated checked Q8_0 row-format failure.
    #[snafu(display("Q8_0 projection format failure: {source}"))]
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

    /// CPU softmax rejects an empty last axis.
    #[snafu(display("softmax rejects rows={rows}, width={width}"))]
    SoftmaxInvalidDimension {
        /// Declared row count.
        rows: usize,
        /// Declared row width.
        width: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// CPU softmax's declared element count overflowed `usize`.
    #[snafu(display("softmax element count overflows for rows={rows}, width={width}"))]
    SoftmaxSizeOverflow {
        /// Declared row count.
        rows: usize,
        /// Declared row width.
        width: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// CPU softmax's supplied slice length did not match its declared shape.
    #[snafu(display(
        "softmax input length {actual_len} does not equal expected {expected_len} for rows={rows}, width={width}"
    ))]
    SoftmaxShape {
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

    /// CPU softmax observed a non-finite value outside the all-negative-infinity mask policy.
    #[snafu(display(
        "softmax {stage:?} produced or received non-finite value {value} at row {row}, column {column}"
    ))]
    SoftmaxNonFinite {
        /// Checked softmax stage.
        stage: SoftmaxStage,
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

    /// CPU softmax could not reserve checked output backing.
    #[snafu(display("softmax {kernel} output allocation for {requested_len} elements failed"))]
    SoftmaxAllocation {
        /// Symbolic CPU reference name.
        kernel: &'static str,
        /// Exact requested output element count.
        requested_len: usize,
        /// Allocation failure.
        source: std::collections::TryReserveError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// CPU L2 normalization rejected a vector with no nonzero component.
    #[snafu(display("L2 normalization rejects a zero vector"))]
    L2NormalizeZero {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// CPU L2 normalization observed a non-finite input or intermediate value.
    #[snafu(display(
        "L2 normalization {stage:?} produced or received non-finite value {value} at index {index}"
    ))]
    L2NormalizeNonFinite {
        /// Checked normalization stage.
        stage: L2NormalizeStage,
        /// Component associated with the rejected value.
        index: usize,
        /// Rejected non-finite value.
        value: f64,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// CPU L2 normalization produced a finite result outside its unit-norm tolerance.
    #[snafu(display("L2 normalization output norm {norm} is outside unit tolerance"))]
    L2NormalizeNotUnit {
        /// Observed output L2 norm.
        norm: f64,
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

    /// CPU Q8_0 GEMV could not reserve its checked output backing.
    #[snafu(display("Q8_0 GEMV output allocation for {requested_len} rows failed"))]
    Q8GemvAllocation {
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
