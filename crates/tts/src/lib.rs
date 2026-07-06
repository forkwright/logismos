//! # tts
//!
//! Text-to-speech: Kokoro (ONNX-weights-equivalent), F5-TTS (flow
//! matching), XTTS (speaker-conditioned).
//!
//! Phase 0 scaffold. No functional code yet.
//!
//! ## Responsibility
//!
//! - Kokoro decoder (54 voices, sub-0.3 s first sample)
//! - F5-TTS flow-matching with 5–15 s voice-clone reference
//! - XTTS multilingual
//!
//! Lands in Phase 9 (downstream companion-consumer lipsync unblock).
#![deny(missing_docs)]

#[cfg(test)]
const CRATE_NAME: &str = "tts";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_identity_matches_role() {
        assert_eq!(env!("CARGO_PKG_NAME"), CRATE_NAME);
    }
}
