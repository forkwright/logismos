//! # embed
//!
//! Sentence-transformer heads, Matryoshka projection, and the concrete
//! `StellaModel` that implements [`core::EmbeddingModel`] — the first
//! production model landing in logismos.
//!
//! The Stella pipeline matches the sentence-transformers reference at
//! `/models/stella-1.5b-v5/`:
//!
//! 1. Tokenise via `tokenize::Tokenizer::from_file` (adds EOS by
//!    default, matching the HF `post_processor`).
//! 2. Optionally prepend a role prompt (`s2s_query`, `s2p_query`, or a
//!    caller-supplied string).
//! 3. Forward through `encoders::StellaEncoder` (fp32 CPU in Phase 3).
//! 4. Mean-pool the last-hidden-states using the attention mask.
//! 5. L2-normalise the pooled vector.
//! 6. Project through the selected Matryoshka dense head (fp32
//!    `linear.weight` + `linear.bias`).
//! 7. L2-normalise the projected vector.
//!
//! ## Prompt prefixes
//!
//! The checkpoint bundles its own prompt strings in
//! `config_sentence_transformers.json`. The model parses the file at
//! load time and keeps a map `Prompt -> String`. Consumers pick a
//! prompt role; the model resolves the string.

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
    clippy::cast_sign_loss
)]

pub mod error;
pub mod stella;

pub use crate::error::{Error, Result};
pub use crate::stella::{StellaDim, StellaModel};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stella_dims_include_default_width() {
        assert!(StellaDim::all().contains(&StellaDim::Dim1024));
    }
}
