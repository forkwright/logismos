//! The reranker trait contract.
//!
//! Implementations live beside their backend: [`crate::qwen3`] for native
//! Qwen3, `cpu_reranker` for feature-gated ModernBERT, and [`crate::gte`]
//! for the preflight surface.

use crate::batch::{Predictions, RerankBatch};
use crate::error::Result;

/// Contract for cross-encoder rerankers.
///
/// Matches the conceptual shape of TEI's `Backend::predict`.
pub trait Reranker: Send + Sync {
    /// Score every item in `batch`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotLoaded`] for preflight stubs, or
    /// [`Error::EmptyBatch`] for malformed batches.
    fn predict(&self, batch: RerankBatch) -> Result<Predictions>;
}
