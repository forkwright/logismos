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
}
