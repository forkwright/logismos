//! # optim
//!
//! Optimisers and learning-rate schedulers.
//!
//! Phase 0 scaffold. No functional code yet.
//!
//! ## Responsibility
//!
//! - `AdamW`, Adafactor, Lion, Sophia
//! - Warmup + cosine + linear + polynomial schedulers
//! - Gradient accumulation hooks
//!
//! Lands in Phase 10. Consumer: `melete`.
#![deny(missing_docs)]

#[cfg(test)]
const CRATE_NAME: &str = "optim";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_identity_matches_role() {
        assert_eq!(env!("CARGO_PKG_NAME"), CRATE_NAME);
    }
}
