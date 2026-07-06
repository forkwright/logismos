//! # melete — μελέτη
//!
//! Training orchestration: `LoRA` / `QLoRA` adapter surgery, distillation,
//! training loops. Role inherited from aletheia's distillation crate;
//! generalised here to any training practice (careful, attentive
//! rehearsal toward a learned weight).
//!
//! Phase 0 scaffold. No functional code yet.
//!
//! ## Responsibility
//!
//! - `LoRA` / `QLoRA` adapter injection and merge/unmerge
//! - Distillation loops (teacher → student)
//! - Direct preference optimisation variants (DPO, ORPO, KTO)
//! - Training-loop orchestration over `autograd` + `optim` + `data`
//!
//! Lands in Phase 10. Consumers: aletheia Phase 06b memory policy,
//! gnomon routing-overlay training, domain-specific Stella fine-tunes.
#![deny(missing_docs)]

#[cfg(test)]
const CRATE_NAME: &str = "melete";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_identity_matches_role() {
        assert_eq!(env!("CARGO_PKG_NAME"), CRATE_NAME);
    }
}
