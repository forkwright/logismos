//! # decoders
//!
//! Decoder-only LLM family: Qwen2 / 3 (including GDN hybrid), Llama.
//!
//! Phase 0 scaffold. No functional code yet.
//!
//! ## Responsibility
//!
//! - Autoregressive forward pass with KV cache
//! - Qwen2/3 architecture (GQA + `RoPE` + `SwiGLU` + `RMSNorm`)
//! - Qwen3 GDN hybrid (48 GDN + 16 full attention) — gnomon target
//! - Llama family
//!
//! Lands in Phase 6 alongside paged cache + speculative decoding.
//! Consumers: `hermeneus` for serving, `bin` for CLI, downstream
//! repos via `core::DecoderModel`.
#![deny(missing_docs)]

#[cfg(test)]
const CRATE_NAME: &str = "decoders";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_identity_matches_role() {
        assert_eq!(env!("CARGO_PKG_NAME"), CRATE_NAME);
    }
}
