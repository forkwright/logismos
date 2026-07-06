//! Batch and I/O types for the reranker contract.
//!
//! Conceptually matches TEI's `Batch` -> `Predictions` contract:
//! a batch of query-document pairs goes in; a score per pair comes out.

use std::collections::BTreeMap;

use crate::error::{
    EmptyBatchSnafu, EmptyDocumentSnafu, EmptyPredictionSnafu, EmptyQuerySnafu,
    MissingPredictionSnafu, Result, UnknownPredictionSnafu,
};

/// A single query-document pair to be scored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RerankItem {
    /// Query text.
    pub query: String,
    /// Document text.
    pub document: String,
}

/// A batch of items submitted to a reranker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RerankBatch {
    /// Ordered list of query-document pairs.
    pub items: Vec<RerankItem>,
}

impl RerankBatch {
    /// Build a validated rerank batch.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmptyBatch`] when `items` is empty,
    /// [`Error::EmptyQuery`] when a query is blank, or
    /// [`Error::EmptyDocument`] when a document is blank.
    pub fn new(items: Vec<RerankItem>) -> Result<Self> {
        if items.is_empty() {
            return EmptyBatchSnafu.fail();
        }

        for (index, item) in items.iter().enumerate() {
            if item.query.trim().is_empty() {
                return EmptyQuerySnafu { index }.fail();
            }
            if item.document.trim().is_empty() {
                return EmptyDocumentSnafu { index }.fail();
            }
        }

        Ok(Self { items })
    }

    /// Number of query-document pairs in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the batch has no query-document pairs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// Prediction map keyed by batch item index.
///
/// This mirrors TEI's `Predictions = IntMap<usize, Vec<f32>>` boundary
/// while using a deterministic standard-library map until a faster integer
/// map earns a dependency.
pub type Predictions = BTreeMap<usize, Vec<f32>>;

/// Output of a reranker prediction pass.
///
/// For a binary cross-encoder reranker each value vector typically has length
/// 1 and contains the relevance logit for the corresponding batch item.
#[derive(Debug, Clone, PartialEq)]
pub struct RerankScores {
    /// Score vectors keyed by input batch position.
    pub predictions: Predictions,
}

impl RerankScores {
    /// Build validated scores for a batch.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnknownPrediction`] for keys outside the batch,
    /// [`Error::MissingPrediction`] for absent batch rows, and
    /// [`Error::EmptyPrediction`] when a row has no logits.
    pub fn new(batch: &RerankBatch, predictions: Predictions) -> Result<Self> {
        for index in predictions.keys() {
            if *index >= batch.len() {
                return UnknownPredictionSnafu { index: *index }.fail();
            }
        }

        for index in 0..batch.len() {
            let Some(scores) = predictions.get(&index) else {
                return MissingPredictionSnafu { index }.fail();
            };
            if scores.is_empty() {
                return EmptyPredictionSnafu { index }.fail();
            }
        }

        Ok(Self { predictions })
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests intentionally unwrap successful fixture calls to keep failure locations precise"
)]
mod tests {
    use super::*;

    #[test]
    fn batch_construction() {
        let batch = RerankBatch::new(vec![
            RerankItem {
                query: "what is rust".into(),
                document: "Rust is a systems language".into(),
            },
            RerankItem {
                query: "what is rust".into(),
                document: "Python is a scripting language".into(),
            },
        ])
        .unwrap();
        assert_eq!(batch.items.len(), 2);
    }

    #[test]
    fn scores_match_batch_indices() {
        let mut predictions = Predictions::new();
        predictions.insert(0, vec![2.5]);
        predictions.insert(1, vec![-1.0]);
        let batch = RerankBatch::new(vec![
            RerankItem {
                query: "what is rust".into(),
                document: "Rust is a systems language".into(),
            },
            RerankItem {
                query: "what is rust".into(),
                document: "Python is a scripting language".into(),
            },
        ])
        .unwrap();
        let scores = RerankScores::new(&batch, predictions).unwrap();
        assert_eq!(scores.predictions.len(), 2);
        let first = scores.predictions.get(&0).and_then(|v| v.first()).copied();
        assert_eq!(first, Some(2.5));
    }

    #[test]
    fn batch_rejects_blank_query() {
        let result = RerankBatch::new(vec![RerankItem {
            query: " ".into(),
            document: "Rust is a systems language".into(),
        }]);

        assert!(matches!(
            result,
            Err(crate::error::Error::EmptyQuery { index: 0, .. })
        ));
    }

    #[test]
    fn scores_reject_missing_row() {
        let batch = RerankBatch::new(vec![RerankItem {
            query: "what is rust".into(),
            document: "Rust is a systems language".into(),
        }])
        .unwrap();
        let predictions = Predictions::new();

        let result = RerankScores::new(&batch, predictions);

        assert!(matches!(
            result,
            Err(crate::error::Error::MissingPrediction { index: 0, .. })
        ));
    }
}
