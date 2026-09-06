//! Typed failures at decoder structural-admission boundaries.

use snafu::Snafu;

/// Result alias used throughout `decoders`.
pub type Result<T> = std::result::Result<T, Error>;

/// Qwen3.5 observation failures.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum Error {
    /// A required architecture metadata key was absent.
    #[snafu(display("qwen35 structural profile is missing metadata key `{key}`"))]
    MissingMetadata {
        /// Exact GGUF key required by the structural contract.
        key: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A metadata key had an incompatible GGUF value type.
    #[snafu(display("qwen35 metadata key `{key}` must be {expected}, got {actual:?}"))]
    MetadataType {
        /// Exact GGUF key required by the structural contract.
        key: &'static str,
        /// Required GGUF value type.
        expected: &'static str,
        /// Parsed value type supplied by the observed artifact.
        actual: loader::gguf::MetaValueType,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A typed metadata value did not meet a source-derived relation.
    #[snafu(display("qwen35 metadata key `{key}` violates {rule}"))]
    MetadataRelation {
        /// Exact GGUF key responsible for the failure.
        key: &'static str,
        /// Source-derived relation that the value must satisfy.
        rule: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A tensor required by the structural contract was absent.
    #[snafu(display("qwen35 structural profile is missing tensor `{name}`"))]
    MissingTensor {
        /// Tensor role and block index expected by the profile.
        name: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A tensor name appeared more than once.
    #[snafu(display("qwen35 structural profile has duplicate tensor `{name}`"))]
    DuplicateTensor {
        /// Duplicate tensor name.
        name: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A tensor is not part of the recognized Qwen3.5 structural role set.
    #[snafu(display("qwen35 structural profile has unclassified tensor `{name}`"))]
    UnclassifiedTensor {
        /// Unexpected tensor name.
        name: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A recognized tensor had a shape inconsistent with typed metadata.
    #[snafu(display("qwen35 tensor `{name}` shape must be {expected:?}, got {actual:?}"))]
    TensorShape {
        /// Tensor name.
        name: String,
        /// Shape derived from typed architecture metadata.
        expected: Vec<u64>,
        /// Shape recorded in the observation.
        actual: Vec<u64>,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A checked arithmetic operation required to derive a tensor shape overflowed.
    #[snafu(display("qwen35 structural profile arithmetic overflow while deriving {context}"))]
    ArithmeticOverflow {
        /// Shape relation that overflowed.
        context: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The verified payload could not supply the named tensor.
    #[snafu(display("qwen35 verified payload cannot provide tensor `{name}`: {source}"))]
    PayloadTensor {
        /// Tensor requested by the bounded projection operation.
        name: String,
        /// Verified-payload lookup failure.
        source: loader::Error,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A structurally recognized tensor does not use an executable row storage type.
    #[snafu(display(
        "qwen35 tensor `{name}` has no executable row format for this projection, got {actual:?}"
    ))]
    ProjectionDtype {
        /// Tensor requested by the bounded projection operation.
        name: String,
        /// Observed GGML storage type.
        actual: loader::gguf::GgmlType,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A structurally recognized tensor is not a matrix.
    #[snafu(display(
        "qwen35 tensor `{name}` must be rank 2 for this projection, got rank {actual}"
    ))]
    ProjectionRank {
        /// Tensor requested by the bounded projection operation.
        name: String,
        /// Observed tensor rank.
        actual: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The supplied activation width does not match the matrix input dimension.
    #[snafu(display(
        "qwen35 tensor `{name}` projection input must have width {expected}, got {actual}"
    ))]
    ProjectionInputWidth {
        /// Tensor requested by the bounded projection operation.
        name: String,
        /// Matrix input dimension.
        expected: usize,
        /// Supplied activation length.
        actual: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A matrix input dimension cannot form an integral executable serialized row.
    #[snafu(display("qwen35 tensor `{name}` has invalid executable row layout: {source}"))]
    ProjectionLayout {
        /// Tensor requested by the bounded projection operation.
        name: String,
        /// Row-geometry failure from the canonical quantization utility.
        source: quant::Error,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Tensor bytes do not form the contiguous rows implied by its validated dimensions.
    #[snafu(display("qwen35 tensor `{name}` projection bytes must be {expected}, got {actual}"))]
    ProjectionBytes {
        /// Tensor requested by the bounded projection operation.
        name: String,
        /// Required serialized row-major byte length.
        expected: usize,
        /// Observed borrowed payload length.
        actual: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Output allocation failed before any projection result could escape.
    #[snafu(display(
        "qwen35 tensor `{name}` could not reserve {output_width} projection outputs: {source}"
    ))]
    ProjectionAllocation {
        /// Tensor requested by the bounded projection operation.
        name: String,
        /// Matrix output dimension.
        output_width: usize,
        /// Allocation failure.
        source: std::collections::TryReserveError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// One serialized row was not executable as finite CPU arithmetic.
    #[snafu(display("qwen35 tensor `{name}` projection row {row} failed: {source}"))]
    ProjectionRow {
        /// Tensor requested by the bounded projection operation.
        name: String,
        /// Zero-based output-row index.
        row: usize,
        /// Executable row-decoding or arithmetic failure.
        source: quant::Error,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A requested block is not a recurrent main decoder block.
    #[snafu(display("qwen35 block {block_index} cannot run recurrent attention: {rule}"))]
    RecurrentLayer {
        /// Main-block index requested by the caller.
        block_index: u64,
        /// Source-derived reason this block cannot use the recurrent path.
        rule: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The recurrent step input is not a complete hidden-width token sequence.
    #[snafu(display(
        "qwen35 recurrent input must contain complete hidden rows of width {hidden}, got {actual} values"
    ))]
    RecurrentInput {
        /// Typed hidden width expected by the selected layer.
        hidden: usize,
        /// Supplied scalar count.
        actual: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Recurrent execution could not allocate an all-or-nothing local result.
    #[snafu(display("qwen35 recurrent {target} could not reserve {length} f32 values: {source}"))]
    RecurrentAllocation {
        /// Named local buffer that could not be reserved.
        target: &'static str,
        /// Exact scalar capacity requested.
        length: usize,
        /// Allocation failure.
        source: std::collections::TryReserveError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A recurrent allocation owner disagreed with its precomputed request.
    #[snafu(display(
        "qwen35 recurrent {target} derived {derived} f32 values after planning {planned}"
    ))]
    RecurrentAllocationPlan {
        /// Named allocation whose owner/request relation drifted.
        target: &'static str,
        /// Capacity precomputed by the recurrent owner plan.
        planned: usize,
        /// Capacity independently implied at the reservation site.
        derived: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Native execution could not allocate an all-or-nothing local result.
    #[snafu(display("qwen35 execution {target} could not reserve {length} elements: {source}"))]
    ExecutionAllocation {
        /// Named local buffer that could not be reserved.
        target: &'static str,
        /// Exact element capacity requested.
        length: usize,
        /// Allocation failure retained as the error source.
        source: std::collections::TryReserveError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// An executor allocation owner disagreed with its precomputed request.
    #[snafu(display(
        "qwen35 execution {target} derived {derived} f32 values after planning {planned}"
    ))]
    ExecutionAllocationPlan {
        /// Named allocation whose owner/request relation drifted.
        target: &'static str,
        /// Capacity precomputed by the executor owner plan.
        planned: usize,
        /// Capacity independently implied at the reservation site.
        derived: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Finite recurrent execution produced a non-finite intermediate.
    #[snafu(display(
        "qwen35 recurrent arithmetic became non-finite during {stage} at index {index}"
    ))]
    RecurrentArithmetic {
        /// Named mathematical stage.
        stage: &'static str,
        /// Flat scalar index within that stage.
        index: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The shared finite CPU `RMSNorm` rejected recurrent execution.
    #[snafu(display("qwen35 recurrent RMSNorm failed: {source}"))]
    RecurrentRmsNorm {
        /// Checked shared CPU `RMSNorm` failure.
        source: kernels::Error,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A fallible CPU elementwise reference rejected recurrent execution.
    #[snafu(display("qwen35 recurrent CPU elementwise operation failed: {source}"))]
    RecurrentCpu {
        /// Checked shared CPU reference failure.
        source: kernels::Error,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The depthwise causal-convolution reference rejected a recurrent step.
    #[snafu(display("qwen35 recurrent causal convolution failed: {source}"))]
    RecurrentConvolution {
        /// Checked causal-convolution failure.
        source: kernels::CausalConvError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The grouped GDN reference rejected a recurrent step.
    #[snafu(display("qwen35 recurrent GDN failed: {source}"))]
    RecurrentGdn {
        /// Checked grouped-GDN failure.
        source: kernels::GdnError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A bounded text-session context request is invalid for this artifact.
    #[snafu(display("qwen35 execution context {requested} violates {rule}"))]
    ExecutionContext {
        /// Requested or reached token count.
        requested: usize,
        /// Execution invariant that refused the request.
        rule: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A token identifier does not name one row of the artifact vocabulary.
    #[snafu(display("qwen35 token id {token_id} is outside vocabulary {vocabulary}"))]
    ExecutionToken {
        /// Supplied token id.
        token_id: u32,
        /// Artifact-derived vocabulary count.
        vocabulary: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// CPU execution produced a non-finite scalar.
    #[snafu(display(
        "qwen35 execution arithmetic became non-finite during {stage} at index {index}"
    ))]
    ExecutionArithmetic {
        /// Mathematical stage.
        stage: &'static str,
        /// Flat scalar index within that stage.
        index: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A fallible CPU elementwise reference rejected hybrid execution.
    #[snafu(display("qwen35 execution CPU elementwise operation failed: {source}"))]
    ExecutionCpu {
        /// Checked shared CPU reference failure.
        source: kernels::Error,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}
