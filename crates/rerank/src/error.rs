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
