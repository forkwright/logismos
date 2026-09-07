//! Typed failures for the artifact-bound native text pipeline.

use snafu::Snafu;

/// Result alias used throughout `text`.
pub type Result<T> = std::result::Result<T, Error>;

/// Failures that leave no usable generated response.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum Error {
    /// A bounded request-local allocation could not be reserved.
    #[snafu(display("text could not reserve bounded {target} storage"))]
    Allocation {
        /// Allocation purpose.
        target: &'static str,
        /// Allocation failure returned by the standard library.
        source: std::collections::TryReserveError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Required artifact metadata was absent or had the wrong GGUF type.
    #[snafu(display("text artifact metadata `{key}` is missing or has the wrong type"))]
    Metadata {
        /// GGUF metadata key.
        key: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A tokenizer does not exactly match the artifact token table.
    #[snafu(display("text tokenizer/artifact vocabulary mismatch at token id {id}"))]
    VocabularyMismatch {
        /// Mismatched serialized token ID.
        id: u32,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The model's special-token policy was inconsistent with its token table.
    #[snafu(display("text artifact special-token policy is invalid: {rule}"))]
    SpecialTokenPolicy {
        /// Violated policy rule.
        rule: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Template compilation or rendering failed.
    #[snafu(display("text chat template failed: {source}"))]
    Template {
        /// Template engine failure retaining its error chain.
        source: minijinja::Error,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A newer shared renderer failure has no text-specific legacy equivalent.
    #[snafu(display("text template renderer failed: {source}"))]
    TemplateRenderer {
        /// Shared renderer failure retaining its error chain.
        source: templates::Error,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Tokenizer parsing, encoding, or decoding failed.
    #[snafu(display("text tokenizer failed: {source}"))]
    Tokenizer {
        /// Tokenizer failure retaining its error chain.
        source: tokenize::Error,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The artifact-bound decoder session failed.
    #[snafu(display("text decoder failed: {source}"))]
    Decoder {
        /// Decoder failure retaining its error chain.
        source: decoders::Error,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The checked greedy sampler rejected decoder logits.
    #[snafu(display("text greedy selection failed: {source}"))]
    Decode {
        /// Checked sampler failure retaining its error chain.
        source: decode::Error,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Decoder logits did not contain the single bounded vocabulary row.
    #[snafu(display("text decoder logits have length {actual}, expected {expected}"))]
    LogitShape {
        /// Actual logit count.
        actual: usize,
        /// Required exact vocabulary width.
        expected: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Rendered bytes violated the UTF-8 invariant required by a text template.
    #[snafu(display("text template emitted invalid UTF-8: {source}"))]
    RenderedUtf8 {
        /// UTF-8 conversion failure retaining its error chain.
        source: std::string::FromUtf8Error,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A rendered prompt contains no model token IDs.
    #[snafu(display("text chat template rendered an empty prompt"))]
    EmptyPrompt {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A request exceeded one declared pipeline bound.
    #[snafu(display("text request {field} {actual} exceeds limit {limit}"))]
    LimitExceeded {
        /// Bounded request dimension.
        field: &'static str,
        /// Caller-supplied or derived value.
        actual: usize,
        /// Maximum accepted value.
        limit: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The caller cancelled at a defined pipeline boundary.
    #[snafu(display("text generation was cancelled before {boundary}"))]
    Cancelled {
        /// Last unstarted pipeline boundary.
        boundary: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A pipeline configuration was internally inconsistent.
    #[snafu(display("text pipeline configuration violates {rule}"))]
    InvalidConfiguration {
        /// Exact invariant that failed.
        rule: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}
