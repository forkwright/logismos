//! # decoders
//!
//! Decoder-only LLM family: Qwen2 / 3 (including GDN hybrid), Llama.
//!
//! The current implementation is limited to Qwen3.5 GGUF structural preflight
//! over an opaque loader observation. It neither decodes weights nor provides a
//! forward model, tokenizer execution, cache, or runtime admission.
//!
//! ## Responsibility
//!
//! - Autoregressive forward pass with KV cache
//! - Qwen2/3 architecture (GQA + `RoPE` + `SwiGLU` + `RMSNorm`)
//! - Qwen3 GDN hybrid (48 GDN + 16 full attention) — gnomon target
//! - Llama family
//!
//! Forward execution lands in Phase 6 alongside paged cache + speculative decoding.
//! Consumers: `hermeneus` for serving, `bin` for CLI, downstream
//! repos via `core::DecoderModel`.
#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod error;
pub mod qwen35;

pub use crate::error::{Error, Result};
pub use crate::qwen35::Qwen35StructuralProfile;

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
