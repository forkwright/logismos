//! Error types for the `tokenize` crate.

use snafu::Snafu;

/// Result alias used throughout `tokenize`.
pub type Result<T> = core::result::Result<T, Error>;

/// Tokenizer-surface errors.
///
/// The upstream crate uses `Box<dyn std::error::Error + Send + Sync>`
/// which we decline to re-export: it would leak the upstream-specific
/// error hierarchy across the facade. Instead, the upstream's
/// `Display` is captured as a string.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum Error {
    /// A byte-limit configuration was zero and therefore cannot admit a tokenizer.
    #[snafu(display("tokenizer byte limit must be non-zero, got {maximum}"))]
    InvalidByteLimit {
        /// Invalid requested maximum.
        maximum: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Upstream `tokenizers` failure (loading, encode, decode).
    #[snafu(display("tokenizers upstream: {message}"))]
    Upstream {
        /// Upstream error rendered without exposing its type hierarchy.
        message: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Tokenizer bytes exceeded the caller-supplied parsing limit.
    #[snafu(display("tokenizer bytes have length {actual}, exceeding limit {limit}"))]
    ByteLimitExceeded {
        /// Maximum permitted byte length.
        limit: usize,
        /// Actual supplied byte length.
        actual: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Tokenizer bytes did not have the expected exact length.
    #[snafu(display("tokenizer bytes have length {actual}, expected {expected}"))]
    ByteLengthMismatch {
        /// Required byte length from the explicit tokenizer identity.
        expected: usize,
        /// Actual supplied byte length.
        actual: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Tokenizer bytes did not have the expected SHA-256 digest.
    #[snafu(display("tokenizer bytes do not match the required SHA-256 digest"))]
    DigestMismatch {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A verified tokenizer retains upstream padding configuration.
    #[snafu(display("tokenizer retains configured upstream padding"))]
    ConfiguredPadding {
        /// Source code location where the refusal was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A verified tokenizer retains upstream truncation configuration.
    #[snafu(display("tokenizer retains configured upstream truncation"))]
    ConfiguredTruncation {
        /// Source code location where the refusal was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The expected vocabulary count differs from the tokenizer vocabulary count.
    #[snafu(display("tokenizer vocabulary has {actual} entries, expected {expected} entries"))]
    VocabularyLengthMismatch {
        /// Required number of token IDs.
        expected: usize,
        /// Number of tokenizer IDs, including added tokens.
        actual: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The supplied expected-vocabulary iterator did not match its declared count.
    #[snafu(display(
        "expected vocabulary iterator produced {actual} entries, declared count was {expected}"
    ))]
    ExpectedVocabularyLengthMismatch {
        /// Declared expected vocabulary size.
        expected: usize,
        /// Minimum observed iterator entry count.
        actual: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// An expected vocabulary position cannot be represented by a tokenizer ID.
    #[snafu(display("expected vocabulary position {index} exceeds the u32 tokenizer-ID domain"))]
    VocabularyIdOutOfRange {
        /// Expected vocabulary position that could not become a tokenizer ID.
        index: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A tokenizer ID does not match the expected spelling at that ID.
    #[snafu(display("tokenizer vocabulary does not match the expected spelling at token ID {id}"))]
    VocabularyMismatch {
        /// Mismatched token ID.
        id: u32,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A declared special-token ID lies outside the expected vocabulary.
    #[snafu(display(
        "declared special token ID {id} lies outside the expected vocabulary of {vocabulary_size} entries"
    ))]
    SpecialTokenIdOutOfRange {
        /// Declared special token ID.
        id: u32,
        /// Expected vocabulary size.
        vocabulary_size: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A declared special-token ID has no tokenizer spelling.
    #[snafu(display("declared special token ID {id} has no tokenizer spelling"))]
    SpecialTokenMissing {
        /// Declared special token ID.
        id: u32,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A declared special-token ID is not marked special by the tokenizer.
    #[snafu(display("declared special token ID {id} is not marked special by tokenizer.json"))]
    SpecialTokenNotMarked {
        /// Declared special token ID.
        id: u32,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A declared special-token spelling does not encode to its declared ID alone.
    #[snafu(display(
        "declared special token ID {id} does not encode to exactly that one token ID"
    ))]
    SpecialTokenEncodingMismatch {
        /// Declared special token ID.
        id: u32,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// The immutable collective decoder program could not be derived.
    #[snafu(display("tokenizer collective decoder program is invalid: {message}"))]
    DecoderProgram {
        /// Exact derivation failure.
        message: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A checked collective decoder storage dimension overflowed `usize`.
    #[snafu(display("tokenizer collective decoder storage overflowed while deriving {target}"))]
    DecodePlanOverflow {
        /// Storage dimension whose checked arithmetic overflowed.
        target: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A bounded collective decoder allocation could not be reserved.
    #[snafu(display("tokenizer could not reserve bounded {target} storage"))]
    DecodeStorageAllocation {
        /// Allocation purpose.
        target: &'static str,
        /// Allocation failure returned by the standard library.
        source: std::collections::TryReserveError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// More generated token IDs were supplied than the acquired storage admits.
    #[snafu(display(
        "tokenizer collective decode received {actual} token IDs, exceeding capacity {capacity}"
    ))]
    DecodeTokenCapacityExceeded {
        /// Number of IDs after the refused append.
        actual: usize,
        /// Acquired generated-ID capacity.
        capacity: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// A decoder write exceeded its pre-acquired arena.
    #[snafu(display(
        "tokenizer collective decoder needed {needed} {target} units, exceeding acquired capacity {capacity}"
    ))]
    DecodeStorageExhausted {
        /// Arena or index being written.
        target: &'static str,
        /// Exact capacity needed for the refused write.
        needed: usize,
        /// Acquired logical capacity.
        capacity: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// An internal collective decoder byte span violated its UTF-8 invariant.
    #[snafu(display("tokenizer collective decoder produced an invalid UTF-8 span: {source}"))]
    DecodeUtf8Invariant {
        /// UTF-8 validation failure for the internal span.
        source: std::str::Utf8Error,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// Collective decoded output exceeded the caller's retained byte limit.
    #[snafu(display(
        "tokenizer collective decoded output has {actual} bytes, exceeding limit {limit}"
    ))]
    DecodedByteLimitExceeded {
        /// Maximum retained decoded bytes.
        limit: usize,
        /// Exact decoded byte length before any output write.
        actual: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}
