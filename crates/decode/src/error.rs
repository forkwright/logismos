//! Typed failures at the decode-policy boundary.

use snafu::Snafu;

/// Result alias used throughout `decode`.
pub type Result<T> = core::result::Result<T, Error>;

/// Decode-surface errors.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum Error {
    /// No vocabulary logits were supplied.
    #[snafu(display("decode logits must not be empty"))]
    EmptyLogits {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// One logit is not a supported finite score or negative-infinity mask.
    #[snafu(display("decode logit at index {index} must not be {kind}"))]
    NonFiniteLogit {
        /// Position in the vocabulary-logit row.
        index: usize,
        /// Unsupported non-finite representation.
        kind: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Every vocabulary entry is masked.
    #[snafu(display("decode logits must retain at least one finite candidate"))]
    AllLogitsMasked {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A decode-policy configuration value violates its contract.
    #[snafu(display("decode parameter `{name}` violates {rule}"))]
    InvalidParameter {
        /// Parameter name.
        name: &'static str,
        /// Contract violated by the supplied value.
        rule: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A configured history token does not identify this vocabulary row.
    #[snafu(display("decode token id {token_id} is outside vocabulary {vocabulary}"))]
    TokenOutOfRange {
        /// Configured token identifier.
        token_id: u32,
        /// Number of logits in the current vocabulary row.
        vocabulary: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A selected vocabulary position cannot be represented as a token id.
    #[snafu(display("decode vocabulary index {index} does not fit in u32"))]
    TokenIndexOutOfRange {
        /// Selected vocabulary position.
        index: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A configured processor is intentionally not implemented.
    #[snafu(display("decode processor `{processor}` is not implemented"))]
    UnsupportedProcessor {
        /// Processor name.
        processor: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A decode-local buffer could not reserve its required capacity.
    #[snafu(display("decode could not reserve {length} entries for {target}: {source}"))]
    Allocation {
        /// Buffer whose allocation was refused.
        target: &'static str,
        /// Exact requested entry count.
        length: usize,
        /// Allocation failure retained for callers.
        source: std::collections::TryReserveError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}
