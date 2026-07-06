//! # transformers
//!
//! Shared transformer building blocks: attention modules (MHA / GQA /
//! MQA), MLP variants (SwiGLU / GeGLU), rotary-position-embedding
//! utilities, and optional KV-cache wiring for decoders.
//!
//! Phase 3 ships the **Qwen2-flavour** stack required by Stella 1.5B v5:
//!
//! - [`QwenAttention`] — Grouped-Query Attention with Q/K/V biases, RoPE
//!   (halves rotation), padding-only additive mask, scaled dot-product
//!   attention, and no KV cache (encoder usage).
//! - [`SwiGluMlp`] — SiLU * up-gate followed by a down projection.
//! - Thin wrappers around `kernels::cpu_f32::rms_norm` so callers
//!   compose with the same surface they'll compose the HIP path with
//!   later.
//!
//! All blocks are **fp32 on CPU** in Phase 3. The Phase-6 decoder port
//! will extend the surface to fp16 on HIP with a KV-cache wiring; the
//! block constructors already reserve the space for that.
//!
//! ## Why `Vec<f32>` and not `taxis::Tensor`?
//!
//! Phase 3's exit gate is a correctness gate — the 1e-3 bit-exact parity
//! is easier to chase with flat f32 buffers and explicit shapes than
//! with a tensor abstraction. The blocks expose flat buffers at the
//! edges, which the calling encoder crate wraps in tensors for its
//! users. When Phase 6 lands the GPU path, the block bodies get a
//! matching `forward_hip` implementation that takes `taxis::Tensor`;
//! the CPU reference path stays as the parity anchor.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::doc_markdown,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::many_single_char_names
)]

pub mod attention;
pub mod error;
pub mod mlp;
pub mod norm;
pub mod rope;

pub use crate::attention::{QwenAttention, QwenAttentionConfig, QwenAttentionWeights};
pub use crate::error::{Error, Result};
pub use crate::mlp::{SwiGluMlp, SwiGluMlpWeights};
pub use crate::norm::rms_norm_f32;
pub use crate::rope::RopeTable;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_config_kv_width_multiplies_heads_by_dim() {
        let cfg = QwenAttentionConfig {
            hidden: 16,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
        };
        assert_eq!(cfg.kv_width(), 8);
    }
}

pub mod modernbert;

pub use crate::modernbert::{
    GeGluMlp, GeGluMlpWeights, ModernBertAttention, ModernBertAttentionConfig,
    ModernBertAttentionWeights, gelu, layer_norm_f32,
};
