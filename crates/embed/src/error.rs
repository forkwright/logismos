//! Error type for `embed`.

use snafu::Snafu;

/// Embed crate errors.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum Error {
    /// Required Qwen3 GGUF metadata was absent or had the wrong type.
    #[snafu(display("invalid Qwen3 metadata `{key}`"))]
    Metadata {
        /// GGUF metadata key.
        key: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Native Qwen3 decoder failure.
    #[snafu(display("native Qwen3 decoder: {source}"))]
    Decoders {
        /// Source decoder failure.
        source: decoders::Error,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Native Qwen3 tokenizer or artifact-token compatibility failure.
    #[snafu(display("native Qwen3 tokenize: {source}"))]
    Qwen3Tokenizer {
        /// Source tokenizer error.
        source: tokenize::Error,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// A semantic query role lacked trusted setup instructions.
    #[snafu(display("no trusted instruction is configured for {role}"))]
    UnresolvedPromptRole {
        /// Requested semantic role.
        role: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// An embedding vector could not satisfy the shared unit-norm contract.
    #[snafu(display("embedding unit normalization: {source}"))]
    UnitNormalization {
        /// Source checked CPU normalization failure.
        source: kernels::Error,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Native Qwen3 embedding limits are internally inconsistent.
    #[snafu(display("invalid Qwen3 embedding limits: {rule}"))]
    InvalidLimits {
        /// Violated setup or request-limit invariant.
        rule: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Qwen3 embedding requirement composition overflowed its logical `f32` payload bounds.
    #[snafu(display("Qwen3 embedding CPU requirements overflowed while composing {target}"))]
    RequirementsOverflow {
        /// Named requirement component whose checked byte arithmetic overflowed.
        target: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// A bounded request-local allocation could not be reserved.
    #[snafu(display("Qwen3 embedding could not reserve {target}"))]
    Allocation {
        /// Allocation purpose.
        target: &'static str,
        /// Allocator failure.
        source: std::collections::TryReserveError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// The prefixed request text exceeds its configured byte bound.
    #[snafu(display("embedding request has {actual} UTF-8 bytes, limit {limit}"))]
    InputBytesTooLong {
        /// Actual combined prefix-and-text byte count.
        actual: usize,
        /// Configured maximum byte count.
        limit: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// The prefix and input lengths could not be represented together.
    #[snafu(display("embedding prefix-and-input byte length overflowed usize"))]
    InputByteLengthOverflow {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// An embedding request had no input text before trusted prefix policy.
    #[snafu(display("embedding request text must not be empty"))]
    EmptyInput {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// A batch exceeded the adapter's trusted bounded-work policy.
    #[snafu(display("embedding batch has {actual} inputs, limit {limit}"))]
    BatchTooLarge {
        /// Number of requested inputs.
        actual: usize,
        /// Maximum number accepted by the derived batch policy.
        limit: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Model directory lookup failed.
    #[snafu(display("io: {message}"))]
    Io {
        /// Free-form description.
        message: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Encoder crate bubbled an error.
    #[snafu(transparent)]
    #[cfg(feature = "stella")]
    Encoders {
        /// Source encoder error.
        source: encoders::Error,
    },
    /// Loader crate bubbled an error.
    #[snafu(transparent)]
    Loader {
        /// Source loader error.
        source: loader::Error,
    },
    /// Tokenizer crate bubbled an error.
    #[snafu(transparent)]
    Tokenize {
        /// Source tokenizer error.
        source: tokenize::Error,
    },
    /// Caller asked for a dim the model does not support.
    #[snafu(display("unsupported dim {dim}"))]
    UnsupportedDim {
        /// The unsupported dimension requested.
        dim: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Input token length exceeds configured max.
    #[snafu(display("input too long: got {got}, limit {limit}"))]
    InputTooLong {
        /// Actual token count.
        got: usize,
        /// Configured maximum.
        limit: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        IoSnafu {
            message: e.to_string(),
        }
        .build()
    }
}

/// Crate-local result alias.
pub type Result<T> = std::result::Result<T, Error>;
