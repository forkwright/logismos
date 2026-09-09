//! # hermeneus — ἑρμηνεύς
//!
//! Protocol interpreter. Translates external request shapes (`OpenAI`
//! HTTP, MCP stdio/SSE, eventually others) into calls against the
//! logismos inference stack. Role inherited from aletheia's Claude
//! API-client crate and generalised; MCP folded in here rather than
//! living in a separate crate because both protocols are the same
//! role at different transports.
//!
//! Protocol scaffold. No functional service or published wire contract yet.
//! The private service adapter belongs here: it will bind the existing text,
//! decoder, placement and scheduler owners without moving HIP into the pure
//! request/planning crates or creating a second resource ledger.
//!
//! ## Responsibility
//!
//! Planned HTTP surface (not an advertised capability):
//! - `/v1/chat/completions` (+ SSE streaming)
//! - `/v1/completions`, `/v1/embeddings`
//! - `/v1/rerank` (non-standard but matches BGE/Cohere convention)
//! - `/v1/models`
//! - `/v1/audio/transcriptions` (phase 8), `/v1/audio/speech` (phase 9)
//! - `/v1/images/generations` (phase 11)
//!
//! Planned MCP surface (not an advertised capability):
//! - `inference.complete`, `inference.embed`, `inference.rerank`,
//!   `inference.models` over stdio + SSE
//!
//! Shared machinery (one implementation, two transports):
//! - Request shaping + response normalisation across model families
//! - Admission control delegation to `sched`
//! - Grammar delegation to `decode`
//! - Dispatch-time validation of the exact request, resident and current grant
//! - Shared residency, independent per-use state and retained-result ownership
//! - Explicit release acknowledgement; cancellation alone does not free a lease
//!
//! Aletheia retains workload/session intent and privacy policy. Host grants,
//! modes and external service lifecycle remain host-owned. Serving, measured
//! capacity/quality/performance, consumer conformance and any intentional
//! provider transition have separate gates; implementing this crate does not
//! automatically replace llama-server or retarget the fleet's `local` provider.
#![deny(missing_docs)]

#[cfg(feature = "gpu")]
mod native_text;

#[cfg(feature = "gpu")]
pub use native_text::{
    NativeTextDriverError, NativeTextGenerationFailure, NativeTextResident,
    NativeTextResidentBuildFailure, NativeTextResidentClose, NativeTextResidentTeardown,
    NativeTextUseClose, NativeTextUseConstructionCustody, NativeTextUsePlan,
    NativeTextUsePlanFailure,
};

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
