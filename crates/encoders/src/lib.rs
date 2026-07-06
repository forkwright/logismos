//! # encoders
//!
//! Encoder-only model family. Phase 3 ships `StellaEncoder` — the
//! Qwen2-architecture decoder run in encoder mode (no causal mask,
//! no KV cache), which is exactly how the Stella 1.5B v5
//! sentence-transformers checkpoint produces its token-level
//! representations.
//!
//! The encoder emits `[seq, hidden]` last-hidden-states; the `embed`
//! crate owns pooling, the matryoshka dense head, and the final L2
//! normalisation.
//!
//! Phase 3 runs **fp32 on CPU**. Phase 6 will port the same forward
//! shape to fp16 on HIP. The forward signature is stable across both
//! paths; implementations will dispatch by tensor placement.

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
    clippy::too_many_lines,
    clippy::similar_names
)]

pub mod error;
pub mod stella;

pub use crate::error::{Error, Result};
pub use crate::stella::{
    StellaConfig, StellaEncoder, StellaLayer, StellaLayerWeights, StellaWeights,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stella_config_matches_qwen2_base_shape() {
        let cfg = StellaConfig::stella_1_5b();
        assert_eq!(cfg.hidden, 1536);
        assert_eq!(cfg.n_layers, 28);
    }
}

pub mod modernbert;

pub use crate::modernbert::{
    ModernBertEncoder, ModernBertEncoderConfig, ModernBertLayerWeights, ModernBertWeights,
};
