//! The reranker trait contract.
//!
//! Implementations live beside their backend: [`crate::cpu_reranker`] for
//! the CPU cross-encoder, [`crate::gte`] for the preflight surface.

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
