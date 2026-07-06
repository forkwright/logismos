//! # ekphrasis — ἔκφρασις
//!
//! Whisper-family speech recognition. Role inherited from thumos's
//! on-device voice-to-text crate; logismos carries the desktop /
//! GPU-backed path, thumos retains its constrained on-device variant.
//!
//! Phase 0 scaffold. No functional code yet.
//!
//! ## Responsibility
//!
//! - Whisper encoder-decoder forward pass on `hipcore` + `praxis`
//! - Streaming chunking + token-level timestamp recovery
//! - Concrete models: Whisper large-v3, large-v3-turbo
//!
//! Lands in Phase 8 (downstream companion-consumer unblock).
#![deny(missing_docs)]

#[cfg(test)]
const CRATE_NAME: &str = "ekphrasis";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_identity_matches_role() {
        assert_eq!(env!("CARGO_PKG_NAME"), CRATE_NAME);
    }
}
