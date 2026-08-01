//! # rerank
//!
//! Cross-encoder rerank wrappers. Score (query, document) pairs
//! directly for hybrid-retrieval post-ranking.
//!
//! Phase 5 Option A: contract and preflight surface only.
//! - [`ModernBertConfig`] - serde-deserializable config shape.
//! - [`Reranker`] - trait contract matching TEI `Backend::predict`.
//! - [`GteReranker`] - named preflight surface; fails loudly.
//!
//! ## Responsibility
//!
//! - `Reranker` impls backed by cross-encoder transformers
//! - GTE-reranker-modernbert-base (aletheia Phase 06 target, 149 M)
//! - bge-reranker family
//!
//! Lands in Phase 5. Consumers: kanon/mnemosyne Phase 04f hybrid
//! rerank, aletheia's memory recall.
#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![expect(
    clippy::doc_markdown,
    reason = "crate docs intentionally reference TEI and GTE model names"
)]

pub mod batch;
pub mod config;
pub mod cpu_reranker;
pub mod error;
pub mod gte;
pub mod reranker;

pub use crate::batch::{Predictions, RerankBatch, RerankItem, RerankScores};
pub use crate::config::{ModernBertConfig, ModernBertPreflight};
pub use crate::cpu_reranker::{ClassifierHead, ModernBertCpuReranker};
pub use crate::error::{Error, Result};
pub use crate::gte::GteReranker;
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
