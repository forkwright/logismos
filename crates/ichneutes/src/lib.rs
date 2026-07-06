//! # ichneutes — ἰχνευτής
//!
//! Classification, named-entity recognition, structured extraction.
//! Role inherited from akroasis Phase 4 (RF signal classification);
//! generalized here to text + audio + any domain-specific classifier
//! tracked via the same trait surface.
//!
//! Phase 0 scaffold. No functional code yet.
//!
//! ## Responsibility
//!
//! - Multi-label classification (`ModernBERT` intent — aletheia Phase 06,
//!   139 M)
//! - NER (GLiNER-Large v2.5 — aletheia Phase 06, 300 M)
//! - Structured extraction (NuExtract-tiny — aletheia Phase 06, 500 M)
//! - Extensible classifier registry so akroasis's RF-signal variant
//!   plugs in cleanly
//!
//! Lands in Phase 5.
#![deny(missing_docs)]

#[cfg(test)]
const CRATE_NAME: &str = "ichneutes";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_identity_matches_role() {
        assert_eq!(env!("CARGO_PKG_NAME"), CRATE_NAME);
    }
}
