//! # tokenize
//!
//! Tokenizer facade over the upstream `huggingface/tokenizers` crate.
//! Per Phase-0 dossier, tokenizers is a Cody-approved sovereignty cut:
//! the format (`tokenizer.json`) is the real API boundary, and
//! re-implementing HF's BPE / SentencePiece / WordPiece / Unigram stack
//! is zero marginal value.
//!
//! This crate wraps `tokenizers::Tokenizer` in a thin [`Tokenizer`]
//! struct that exposes only the surface logismos needs:
//!
//! - `encode(text, add_special_tokens) -> Vec<u32>`
//! - `decode(ids, skip_special_tokens) -> String`
//! - `vocab_size() -> usize`
//!
//! Chat-template rendering is **not** done here. `tokenizer.json`
//! doesn't carry the chat template — that lives in
//! `tokenizer_config.json`. Higher layers feed the raw template into
//! minijinja; this crate stays pure.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::doc_markdown
)]

pub mod error;

use std::path::Path;

pub use crate::error::{Error, Result};

/// Thin facade over `tokenizers::Tokenizer`.
///
/// Consumers never see the upstream type: changes to the upstream API
/// live behind this boundary.
pub struct Tokenizer {
    inner: ::tokenizers::Tokenizer,
}

impl Tokenizer {
    /// Load a HuggingFace `tokenizer.json` from disk.
    ///
    /// # Errors
    ///
    /// [`Error::Upstream`] when the file is absent / malformed.
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner =
            ::tokenizers::Tokenizer::from_file(path).map_err(|e| Error::Upstream(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Encode a single string to a token-id vector.
    ///
    /// `add_special_tokens` controls whether the tokenizer's post-processor
    /// prepends / appends its special tokens (BOS, EOS, etc.). For
    /// most HF-compatible models the answer is `true`.
    ///
    /// # Errors
    ///
    /// [`Error::Upstream`] if the underlying encoder fails.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| Error::Upstream(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode an id slice back to a string.
    ///
    /// `skip_special_tokens` mirrors HF semantics: when true, tokens
    /// flagged as "special" in the vocabulary are elided from the
    /// output string.
    ///
    /// # Errors
    ///
    /// [`Error::Upstream`] on decoder failure.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| Error::Upstream(e.to_string()))
    }

    /// Vocabulary size, including added tokens.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Vocabulary size, base model only (no added tokens).
    pub fn vocab_size_base(&self) -> usize {
        self.inner.get_vocab_size(false)
    }
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.vocab_size())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    /// Build a tiny WordLevel `tokenizer.json` on disk so the
    /// round-trip test runs without pulling any real model file.
    fn write_trivial_tokenizer(path: &Path) -> std::io::Result<()> {
        let json = r#"{
          "version": "1.0",
          "truncation": null,
          "padding": null,
          "added_tokens": [],
          "normalizer": null,
          "pre_tokenizer": { "type": "Whitespace" },
          "post_processor": null,
          "decoder": null,
          "model": {
            "type": "WordLevel",
            "vocab": {
              "[UNK]": 0,
              "hello": 1,
              "world": 2,
              "the": 3,
              "quick": 4,
              "fox": 5
            },
            "unk_token": "[UNK]"
          }
        }"#;
        let mut f = std::fs::File::create(path)?;
        f.write_all(json.as_bytes())
    }

    #[test]
    fn round_trip_fixture() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "logismos-tokenize-test-{}.json",
            std::process::id()
        ));
        write_trivial_tokenizer(&tmp).map_err(|e| Error::Upstream(e.to_string()))?;

        let tok = Tokenizer::from_file(&tmp)?;
        assert_eq!(tok.vocab_size(), 6);

        let ids = tok.encode("hello world", false)?;
        assert_eq!(ids, vec![1, 2]);

        let text = tok.decode(&ids, false)?;
        assert!(text.contains("hello"));
        assert!(text.contains("world"));

        let _ = std::fs::remove_file(&tmp);
        Ok(())
    }

    #[test]
    fn unknown_token_maps_to_unk() -> Result<()> {
        let tmp =
            std::env::temp_dir().join(format!("logismos-tokenize-unk-{}.json", std::process::id()));
        write_trivial_tokenizer(&tmp).map_err(|e| Error::Upstream(e.to_string()))?;
        let tok = Tokenizer::from_file(&tmp)?;
        let ids = tok.encode("zephyr", false)?;
        assert_eq!(ids, vec![0]);
        let _ = std::fs::remove_file(&tmp);
        Ok(())
    }

    #[test]
    fn from_file_returns_upstream_error_for_missing_path() {
        let result = Tokenizer::from_file(Path::new("/tmp/nonexistent-tokenizer-12345.json"));
        assert!(matches!(result, Err(Error::Upstream(_))));
    }

    #[test]
    fn vocab_size_base_matches_vocab_size_for_trivial_fixture() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "logismos-tokenize-vocab-{}.json",
            std::process::id()
        ));
        write_trivial_tokenizer(&tmp).map_err(|e| Error::Upstream(e.to_string()))?;
        let tok = Tokenizer::from_file(&tmp)?;
        assert_eq!(tok.vocab_size(), 6);
        assert_eq!(tok.vocab_size_base(), 6);
        assert_eq!(tok.vocab_size(), tok.vocab_size_base());
        let _ = std::fs::remove_file(&tmp);
        Ok(())
    }

    #[test]
    fn encode_add_special_tokens_true_vs_false() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "logismos-tokenize-special-{}.json",
            std::process::id()
        ));
        write_trivial_tokenizer(&tmp).map_err(|e| Error::Upstream(e.to_string()))?;
        let tok = Tokenizer::from_file(&tmp)?;
        let ids_no_special = tok.encode("hello world", false)?;
        let ids_with_special = tok.encode("hello world", true)?;
        assert_eq!(ids_no_special, vec![1, 2]);
        assert_eq!(ids_with_special, ids_no_special);
        let _ = std::fs::remove_file(&tmp);
        Ok(())
    }

    #[test]
    fn decode_round_trips_through_encode() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "logismos-tokenize-roundtrip-{}.json",
            std::process::id()
        ));
        write_trivial_tokenizer(&tmp).map_err(|e| Error::Upstream(e.to_string()))?;
        let tok = Tokenizer::from_file(&tmp)?;
        let original = "hello world";
        let ids = tok.encode(original, false)?;
        let decoded = tok.decode(&ids, false)?;
        assert_eq!(decoded, original);
        let _ = std::fs::remove_file(&tmp);
        Ok(())
    }

    #[test]
    fn encode_empty_string() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "logismos-tokenize-empty-{}.json",
            std::process::id()
        ));
        write_trivial_tokenizer(&tmp).map_err(|e| Error::Upstream(e.to_string()))?;
        let tok = Tokenizer::from_file(&tmp)?;
        let ids = tok.encode("", false)?;
        assert!(ids.is_empty());
        let _ = std::fs::remove_file(&tmp);
        Ok(())
    }

    #[test]
    fn debug_impl_includes_vocab_size() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "logismos-tokenize-debug-{}.json",
            std::process::id()
        ));
        write_trivial_tokenizer(&tmp).map_err(|e| Error::Upstream(e.to_string()))?;
        let tok = Tokenizer::from_file(&tmp)?;
        let dbg = format!("{tok:?}");
        assert!(dbg.contains("Tokenizer"));
        assert!(dbg.contains("vocab_size"));
        assert!(dbg.contains('6'));
        let _ = std::fs::remove_file(&tmp);
        Ok(())
    }
}
