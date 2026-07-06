//! # sched
//!
//! Inference scheduler: batching, admission control, preemption,
//! priority queueing. Multi-tenant safety for shared inference
//! endpoints.
//!
//! Phase 0 scaffold. No functional code yet.
//!
//! ## Responsibility
//!
//! - Continuous-batching state machine (vLLM-style)
//! - Admission control with memory + KV budget
//! - Priority preemption
//! - Cooperates with `cache` paged allocator
//!
//! Lands in Phase 7 (serving + fleet cutover).
#![deny(missing_docs)]

#[cfg(test)]
const CRATE_NAME: &str = "sched";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_identity_matches_role() {
        assert_eq!(env!("CARGO_PKG_NAME"), CRATE_NAME);
    }
}
