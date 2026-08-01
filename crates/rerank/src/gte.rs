//! Preflight surface for the `gte-reranker-modernbert-base` model.
//!
//! Phase 5 Option A: named surface only. Every inference method fails
//! loudly with a precise [`Error::NotLoaded`].

use crate::batch::{Predictions, RerankBatch};
use crate::config::ModernBertConfig;
use crate::error::{EmptyBatchSnafu, NotLoadedSnafu, Result};
use crate::reranker::Reranker;

/// Preflight surface for the `gte-reranker-modernbert-base` model.
///
/// Holds the deserialized configuration but has no weights and no
/// forward implementation. Calling [`Reranker::predict`] fails loudly with
/// a precise error rather than silently falling back or returning
/// dummy values.
#[derive(Debug, Clone)]
pub struct GteReranker {
    cfg: ModernBertConfig,
}

impl GteReranker {
    /// Create a preflight reranker from configuration.
    ///
    /// No weights are loaded; the returned instance exists only to
    /// validate the config shape and serve as a type-system anchor
    /// for downstream consumers.
    ///
    /// # Errors
    ///
    /// Propagates [`ModernBertConfig::preflight_gte_reranker`] failures.
    pub fn from_config(cfg: ModernBertConfig) -> Result<Self> {
        cfg.preflight_gte_reranker()?;
        Ok(Self { cfg })
    }

    /// Load a model checkpoint from disk.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotLoaded`] because the HIP weight-loading path
    /// is not yet implemented.
    pub fn load(_path: &std::path::Path) -> Result<Self> {
        NotLoadedSnafu {
            message:
                "GteReranker::load() is not yet implemented; HIP weight loading path is not wired"
                    .to_string(),
        }
        .fail()
    }

    /// Borrows the configuration.
    #[must_use]
    pub fn config(&self) -> &ModernBertConfig {
        &self.cfg
    }
}

impl Reranker for GteReranker {
    fn predict(&self, batch: RerankBatch) -> Result<Predictions> {
        if batch.is_empty() {
            return EmptyBatchSnafu.fail();
        }
        NotLoadedSnafu {
            message: "GteReranker: HIP inference path not yet wired; load a model checkpoint via GteReranker::load() first".to_string(),
        }.fail()
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests intentionally unwrap successful fixture calls to keep failure locations precise"
)]
mod tests {
    use super::*;
    use crate::batch::{RerankBatch, RerankItem};
    use crate::error::Error;

    #[test]
    fn gte_reranker_from_config_holds_config() {
        let cfg = ModernBertConfig::gte_reranker_modernbert_base();
        let reranker = GteReranker::from_config(cfg.clone()).unwrap();
        assert_eq!(reranker.config().vocab_size, cfg.vocab_size);
    }

    #[test]
    fn gte_reranker_predict_fails_loudly() {
        let cfg = ModernBertConfig::gte_reranker_modernbert_base();
        let reranker = GteReranker::from_config(cfg).unwrap();
        let batch = RerankBatch {
            items: vec![RerankItem {
                query: "test".into(),
                document: "doc".into(),
            }],
        };
        let result = reranker.predict(batch);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not loaded"),
            "error should declare not loaded, got: {msg}"
        );
        assert!(
            msg.contains("GteReranker: HIP inference path not yet wired"),
            "error should name the precise surface, got: {msg}"
        );
        assert!(
            msg.contains("load()"),
            "error should suggest load(), got: {msg}"
        );
    }

    #[test]
    fn gte_reranker_predict_does_not_return_dummy_scores() {
        let cfg = ModernBertConfig::gte_reranker_modernbert_base();
        let reranker = GteReranker::from_config(cfg).unwrap();
        let batch = RerankBatch {
            items: vec![
                RerankItem {
                    query: "q".into(),
                    document: "d1".into(),
                },
                RerankItem {
                    query: "q".into(),
                    document: "d2".into(),
                },
            ],
        };
        let result = reranker.predict(batch);
        // Must be Err, never Ok with placeholder scores.
        assert!(
            matches!(result, Err(Error::NotLoaded { .. })),
            "preflight must never return Ok(...)"
        );
    }

    #[test]
    fn gte_reranker_rejects_invalid_config() {
        let mut cfg = ModernBertConfig::gte_reranker_modernbert_base();
        cfg.num_labels = 0;
        cfg.id2label.clear();

        let result = GteReranker::from_config(cfg);

        assert!(matches!(result, Err(Error::MissingClassifierHead { .. })));
    }

    #[test]
    fn reranker_rejects_empty_batch() {
        let cfg = ModernBertConfig::gte_reranker_modernbert_base();
        let reranker = GteReranker::from_config(cfg).unwrap();
        let batch = RerankBatch { items: vec![] };

        let result = reranker.predict(batch);

        assert!(matches!(result, Err(Error::EmptyBatch { .. })));
    }

    #[test]
    fn reranker_not_loaded_returns_precise_error() {
        let cfg = ModernBertConfig::gte_reranker_modernbert_base();
        let reranker = GteReranker::from_config(cfg).unwrap();
        let batch = RerankBatch {
            items: vec![RerankItem {
                query: "query".into(),
                document: "document".into(),
            }],
        };

        let result = reranker.predict(batch);

        assert!(matches!(result, Err(Error::NotLoaded { .. })));
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("HIP inference path not yet wired"),
            "expected precise HIP not-wired message, got: {msg}"
        );
        assert!(
            msg.contains("GteReranker::load()"),
            "expected actionable load() suggestion, got: {msg}"
        );
    }
}
