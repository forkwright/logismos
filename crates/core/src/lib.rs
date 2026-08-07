//! # core
//!
//! Public trait surface for the logismos platform. Every consumer
//! that does not need crate-specific types depends on `core` only,
//! plus whichever concrete implementation it chooses at the trait
//! boundary.
//!
//! ## Phase 3 status
//!
//! [`EmbeddingModel`] is finalised in Phase 3 and used by
//! `embed::StellaModel`. Other traits (Reranker, Classifier,
//! DecoderModel, SpeechRecognizer, etc.) land with their respective
//! phases per the roadmap.
//!
//! ## Traits (forward-looking)
//!
//! - `EmbeddingModel` — text → vector (**finalised, Phase 3**)
//! - `Reranker` — (query, document) → score (Phase 5)
//! - `Classifier` — input → labels (+ scores) (Phase 5)
//! - `Extractor` — input → structured value (Phase 5)
//! - `DecoderModel` — prompt → token stream (Phase 6)
//! - `SpeechRecognizer` — audio → text (Phase 8)
//! - `SpeechSynthesizer` — text → audio (Phase 9)
//! - `DiffusionModel` — latent → denoised latent (Phase 11)

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::doc_markdown
)]

use std::fmt;

/// Prompt templates that sentence-transformer checkpoints ship with.
///
/// Different retrieval setups benefit from different prompt prefixes; the
/// checkpoint itself owns the exact string. Consumers pick a role, the
/// model resolves the string. `Custom` escape-hatches arbitrary prefixes.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Prompt {
    /// Symmetric "find semantically similar text" prefix.
    S2sQuery,
    /// Asymmetric "web-search query → passage" prefix.
    S2pQuery,
    /// Arbitrary caller-supplied prefix.
    Custom(String),
}

/// Encoding options for [`EmbeddingModel::encode`].
///
/// All fields are optional; `None` means "use the model default".
#[derive(Debug, Clone, Default)]
pub struct EncodeOpts {
    /// Target output dimensionality. Must be in
    /// [`EmbeddingModel::supported_dims`].
    pub dim: Option<usize>,
    /// Override the model's default max input length (tokens). Input
    /// that tokenises longer than this returns
    /// [`EmbeddingError::InputTooLong`] rather than being truncated.
    pub max_tokens: Option<usize>,
    /// Prompt prefix to prepend to the input text.
    pub prompt: Option<Prompt>,
}

/// Error type for embedding models.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EmbeddingError {
    /// Input produced more tokens than the model's configured limit.
    #[error("input exceeds max_tokens: got {got}, limit {limit}")]
    InputTooLong {
        /// Token count produced by tokenisation.
        got: usize,
        /// Configured maximum.
        limit: usize,
    },
    /// Caller asked for a dim the model does not support.
    #[error("unsupported dim {0}")]
    UnsupportedDim(usize),
    /// Tokenisation failed.
    #[error("tokenize: {0}")]
    Tokenize(String),
    /// Compute / kernel failure.
    #[error("compute: {0}")]
    Compute(String),
    /// IO / weight-loading failure.
    #[error("io: {0}")]
    Io(String),
}

/// Contract every embedding model implements.
///
/// Implementations must be `Send + Sync` so they can live behind an
/// `Arc` inside a server.
pub trait EmbeddingModel: Send + Sync + fmt::Debug {
    /// Dimension returned when [`EncodeOpts::dim`] is `None`.
    fn default_dim(&self) -> usize;

    /// Every dim this model can produce. Single-element for fixed-dim
    /// models; multi-element for Matryoshka heads.
    fn supported_dims(&self) -> &[usize];

    /// Maximum input token length this model accepts.
    fn max_tokens(&self) -> usize;

    /// Encode a single text. Returns a unit-norm fp32 vector of length
    /// `opts.dim` (or `default_dim()` when `None`).
    fn encode(&self, text: &str, opts: &EncodeOpts) -> Result<Vec<f32>, EmbeddingError>;

    /// Encode a batch. The default calls [`Self::encode`] in a loop; a
    /// model is free to override with a padded-batch fast path.
    fn encode_batch(
        &self,
        texts: &[&str],
        opts: &EncodeOpts,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        texts.iter().map(|t| self.encode(t, opts)).collect()
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests intentionally unwrap successful fixture calls to keep failure locations precise"
)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct StubModel {
        dim: usize,
        dims: Vec<usize>,
    }

    impl EmbeddingModel for StubModel {
        fn default_dim(&self) -> usize {
            self.dim
        }

        fn supported_dims(&self) -> &[usize] {
            &self.dims
        }

        fn max_tokens(&self) -> usize {
            512
        }

        fn encode(&self, text: &str, opts: &EncodeOpts) -> Result<Vec<f32>, EmbeddingError> {
            let dim = opts.dim.unwrap_or(self.dim);
            if !self.supported_dims().contains(&dim) {
                return Err(EmbeddingError::UnsupportedDim(dim));
            }
            if text == "fail" {
                return Err(EmbeddingError::Compute("boom".into()));
            }
            Ok(vec![1.0; dim])
        }
    }

    #[test]
    fn encode_opts_default_all_none() {
        let opts = EncodeOpts::default();
        assert!(opts.dim.is_none());
        assert!(opts.max_tokens.is_none());
        assert!(opts.prompt.is_none());
    }

    #[test]
    fn prompt_cross_variant_inequality() {
        // WHY: all three variants must compare unequal to each other; S2pQuery != S2sQuery
        // is the cross-variant gate; Custom inequality ensures field comparison is active.
        assert_ne!(Prompt::S2sQuery, Prompt::S2pQuery);
        assert_ne!(Prompt::S2sQuery, Prompt::Custom("s2s_query".into()));
        assert_ne!(Prompt::S2pQuery, Prompt::Custom("s2p_query".into()));
        assert_ne!(Prompt::Custom("foo".into()), Prompt::Custom("bar".into()));
    }

    #[test]
    fn embedding_error_input_too_long_display() {
        let err = EmbeddingError::InputTooLong {
            got: 100,
            limit: 50,
        };
        let msg = err.to_string();
        assert!(msg.contains("input exceeds max_tokens"));
        assert!(msg.contains("got 100"));
        assert!(msg.contains("limit 50"));
    }

    #[test]
    fn embedding_error_unsupported_dim_display() {
        let err = EmbeddingError::UnsupportedDim(7);
        assert_eq!(err.to_string(), "unsupported dim 7");
    }

    #[test]
    fn stub_model_encode_happy_path() {
        let model = StubModel {
            dim: 4,
            dims: vec![4],
        };
        let opts = EncodeOpts::default();
        let vec = model.encode("hello", &opts).unwrap();
        assert_eq!(vec.len(), 4);
        assert!(vec.iter().all(|&v| (v - 1.0).abs() < f32::EPSILON));
    }

    #[test]
    fn encode_batch_default_calls_encode_per_input() {
        let model = StubModel {
            dim: 3,
            dims: vec![3],
        };
        let opts = EncodeOpts::default();
        let batch = model.encode_batch(&["a", "b", "c"], &opts).unwrap();
        assert_eq!(batch.len(), 3);
        for vec in &batch {
            assert_eq!(vec.len(), 3);
            assert!(vec.iter().all(|&v| (v - 1.0).abs() < f32::EPSILON));
        }
    }

    #[test]
    fn encode_batch_default_propagates_errors() {
        let model = StubModel {
            dim: 2,
            dims: vec![2],
        };
        let opts = EncodeOpts::default();
        let result = model.encode_batch(&["ok", "fail", "also ok"], &opts);
        assert!(result.is_err());
    }
}
