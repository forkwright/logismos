//! # hermeneus — ἑρμηνεύς
//!
//! Protocol interpreter. Translates external request shapes (`OpenAI`
//! HTTP, MCP stdio/SSE, eventually others) into calls against the
//! logismos inference stack. Role inherited from aletheia's Claude
//! API-client crate and generalised; MCP folded in here rather than
//! living in a separate crate because both protocols are the same
//! role at different transports.
//!
//! Phase 0 scaffold. No functional code yet.
//!
//! ## Responsibility
//!
//! HTTP surface (phase 7):
//! - `/v1/chat/completions` (+ SSE streaming)
//! - `/v1/completions`, `/v1/embeddings`
//! - `/v1/rerank` (non-standard but matches BGE/Cohere convention)
//! - `/v1/models`
//! - `/v1/audio/transcriptions` (phase 8), `/v1/audio/speech` (phase 9)
//! - `/v1/images/generations` (phase 11)
//!
//! MCP surface (phase 7):
//! - `inference.complete`, `inference.embed`, `inference.rerank`,
//!   `inference.models` over stdio + SSE
//!
//! Shared machinery (one implementation, two transports):
//! - Request shaping + response normalisation across model families
//! - Admission control delegation to `sched`
//! - Grammar delegation to `decode`
//! - SLO target: P50 first-token ≤300 ms, P95 ≤1 s, ≥8 concurrent
//!
//! Lands in Phase 7 (fleet cutover milestone). Replaces llama-server
//! as the dispatch fleet's `local` provider backend.
#![deny(missing_docs)]

#[cfg(test)]
const CRATE_NAME: &str = "hermeneus";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_identity_matches_role() {
        assert_eq!(env!("CARGO_PKG_NAME"), CRATE_NAME);
    }
}
