//! Error type for `rerank`.

use snafu::Snafu;

/// Reranker errors.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum Error {
    /// Native GTE ModernBERT backend is not loaded.
    #[snafu(display("not loaded: {message}"))]
    NotLoaded {
        /// Human-readable reason why the backend is not loaded.
        message: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Native GTE ModernBERT backend is unavailable.
    #[snafu(display("backend unavailable: {message}"))]
    BackendUnavailable {
        /// Human-readable reason why the backend is unavailable.
        message: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Operation is not yet implemented (preflight surface).
    #[snafu(display("not implemented: {message}"))]
    NotImplemented {
        /// Human-readable description of the missing operation.
        message: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Configuration deserialization or validation failure.
    #[snafu(display("config: {message}"))]
    Config {
        /// Human-readable description of the configuration problem.
        message: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Config `model_type` is not ModernBERT.
    #[snafu(display("unsupported model type `{model_type}`"))]
    UnsupportedModelType {
        /// The model type that was rejected.
        model_type: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Config does not declare a sequence-classification head.
    #[snafu(display("missing sequence-classification head"))]
    MissingClassifierHead {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Shape or structural violation.
    #[snafu(display("shape: {message}"))]
    Shape {
        /// Human-readable description of the shape problem.
        message: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Prediction map is missing a row for a batch item.
    #[snafu(display("missing prediction row {index}"))]
    MissingPrediction {
        /// Batch index that is missing from predictions.
        index: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Prediction map contains a row outside the batch.
    #[snafu(display("unknown prediction row {index}"))]
    UnknownPrediction {
        /// Batch index that is outside the valid range.
        index: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Prediction row contains no scores.
    #[snafu(display("empty prediction row {index}"))]
    EmptyPrediction {
        /// Batch index that has an empty prediction vector.
        index: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Input validation failure.
    #[snafu(display("input: {message}"))]
    Input {
        /// Human-readable description of the input problem.
        message: String,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Batch has no items.
    #[snafu(display("empty rerank batch"))]
    EmptyBatch {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Batch item has an empty query.
    #[snafu(display("empty query at batch item {index}"))]
    EmptyQuery {
        /// Index of the offending batch item.
        index: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Batch item has an empty document.
    #[snafu(display("empty document at batch item {index}"))]
    EmptyDocument {
        /// Index of the offending batch item.
        index: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Native Qwen3 rank decoder failure.
    #[snafu(display("native Qwen3 rank decoder: {source}"))]
    Qwen3Decoder {
        /// Source decoder failure.
        source: decoders::Error,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Native Qwen3 tokenizer or artifact-token compatibility failure.
    #[snafu(display("native Qwen3 rerank tokenize: {source}"))]
    Qwen3Tokenizer {
        /// Source tokenizer failure.
        source: tokenize::Error,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Artifact-owned Qwen3 rerank template failure.
    #[snafu(display("native Qwen3 rerank template: {source}"))]
    Qwen3Template {
        /// Source bounded-template failure.
        source: templates::Error,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Required Qwen3 rerank metadata was absent or had an incompatible type.
    #[snafu(display("invalid Qwen3 rerank metadata `{key}`"))]
    Qwen3Metadata {
        /// Exact GGUF metadata key.
        key: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Qwen3 rerank setup limits were internally inconsistent.
    #[snafu(display("invalid Qwen3 rerank limits: {rule}"))]
    Qwen3Limits {
        /// Violated setup-limit invariant.
        rule: &'static str,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// One framed Qwen3 rerank item exceeds its explicit byte bound.
    #[snafu(display("Qwen3 rerank item {index} has {actual} input bytes, limit {limit}"))]
    Qwen3InputBytesTooLong {
        /// Batch item index.
        index: usize,
        /// Checked instruction, query, and document byte total.
        actual: usize,
        /// Trusted setup limit.
        limit: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// One framed Qwen3 rerank item exceeds its explicit token bound.
    #[snafu(display("Qwen3 rerank item {index} has {actual} tokens, limit {limit}"))]
    Qwen3InputTokensTooLong {
        /// Batch item index.
        index: usize,
        /// Encoded token count.
        actual: usize,
        /// Trusted setup limit.
        limit: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Qwen3 rerank input lengths could not be represented together.
    #[snafu(display("Qwen3 rerank input byte length overflowed usize"))]
    Qwen3InputByteLengthOverflow {
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// A Qwen3 rerank batch exceeds its explicit item bound.
    #[snafu(display("Qwen3 rerank batch has {actual} items, limit {limit}"))]
    Qwen3BatchTooLarge {
        /// Number of submitted pairs.
        actual: usize,
        /// Trusted setup limit.
        limit: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// A bounded Qwen3 rerank allocation could not be reserved.
    #[snafu(display("Qwen3 rerank could not reserve bounded {target} storage"))]
    Qwen3Allocation {
        /// Allocation purpose.
        target: &'static str,
        /// Allocation failure returned by the standard library.
        source: std::collections::TryReserveError,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
    /// Qwen3 rerank raw logits could not produce one finite signed score.
    #[snafu(display("Qwen3 rerank item {index} produced a non-finite relevance score"))]
    Qwen3NonFiniteScore {
        /// Batch item index.
        index: usize,
        /// Source code location where the error was reported.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}

/// Crate-local result alias.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests intentionally unwrap successful fixture calls to keep failure locations precise"
)]
mod tests {
    use super::*;

    #[test]
    fn not_loaded_display() {
        let err = NotLoadedSnafu {
            message: "weights missing".to_string(),
        }
        .fail::<()>()
        .unwrap_err();
        assert_eq!(err.to_string(), "not loaded: weights missing");
    }

    #[test]
    fn backend_unavailable_display() {
        let err = BackendUnavailableSnafu {
            message: "native path not built".to_string(),
        }
        .fail::<()>()
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            "backend unavailable: native path not built"
        );
    }

    #[test]
    fn not_implemented_display() {
        let err = NotImplementedSnafu {
            message: "foo".to_string(),
        }
        .fail::<()>()
        .unwrap_err();
        assert_eq!(err.to_string(), "not implemented: foo");
    }

    #[test]
    fn config_display() {
        let err = ConfigSnafu {
            message: "bad json".to_string(),
        }
        .fail::<()>()
        .unwrap_err();
        assert_eq!(err.to_string(), "config: bad json");
    }

    #[test]
    fn shape_display() {
        let err = ShapeSnafu {
            message: "mismatched dims".to_string(),
        }
        .fail::<()>()
        .unwrap_err();
        assert_eq!(err.to_string(), "shape: mismatched dims");
    }

    #[test]
    fn input_display() {
        let err = InputSnafu {
            message: "empty batch".to_string(),
        }
        .fail::<()>()
        .unwrap_err();
        assert_eq!(err.to_string(), "input: empty batch");
    }
}
