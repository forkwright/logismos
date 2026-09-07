//! # rerank
//!
//! Cross-encoder rerank wrappers. Score (query, document) pairs
//! directly for hybrid-retrieval post-ranking.
//!
//! Native CPU Qwen3 and CPU ModernBERT reranking surfaces.
//! - [`ModernBertConfig`] - serde-deserializable config shape.
//! - [`Reranker`] - trait contract matching TEI `Backend::predict`.
//! - [`GteReranker`] - named preflight surface; fails loudly.
//! - [`Qwen3Reranker`] - artifact-bound CPU causal cross-encoder adapter.
//!
//! ## Responsibility
//!
//! - `Reranker` impls backed by cross-encoder transformers
//! - Existing ModernBERT CPU execution behind the default `modernbert` feature
//! - Qwen3 rank GGUF payloads with artifact-owned chat framing
//!
//! Native Qwen3 consumers may disable default features for a GPU-free graph.
//! CPU execution does not establish model-quality or hardware qualification.
#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![expect(
    clippy::doc_markdown,
    reason = "crate docs intentionally reference TEI and GTE model names"
)]

pub mod batch;
pub mod config;
#[cfg(feature = "modernbert")]
pub mod cpu_reranker;
pub mod error;
pub mod gte;
pub mod qwen3;
pub mod reranker;

pub use crate::batch::{Predictions, RerankBatch, RerankItem, RerankScores};
pub use crate::config::{ModernBertConfig, ModernBertPreflight};
#[cfg(feature = "modernbert")]
pub use crate::cpu_reranker::{ClassifierHead, ModernBertCpuReranker};
pub use crate::error::{Error, Result};
pub use crate::gte::GteReranker;
pub use crate::qwen3::{Qwen3Reranker, Qwen3RerankerCpuRequirements, Qwen3RerankerLimits};
pub use crate::reranker::Reranker;

#[cfg(test)]
const CRATE_NAME: &str = "rerank";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_identity_matches_role() {
        assert_eq!(env!("CARGO_PKG_NAME"), CRATE_NAME);
    }
}
