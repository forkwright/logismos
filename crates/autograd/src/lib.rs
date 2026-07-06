//! # autograd
//!
//! Backward pass, gradient checkpointing, activation recomputation.
//!
//! Phase 0 scaffold. No functional code yet.
//!
//! ## Responsibility
//!
//! - Tape-based autograd over `taxis` tensors
//! - Checkpoint-recompute framework for long backprop chains
//! - Mixed-precision (fp16/bf16 forward, fp32 master gradients)
//!
//! Lands in Phase 10. Consumer: `melete`.
#![deny(missing_docs)]

#[cfg(test)]
const CRATE_NAME: &str = "autograd";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_identity_matches_role() {
        assert_eq!(env!("CARGO_PKG_NAME"), CRATE_NAME);
    }
}
