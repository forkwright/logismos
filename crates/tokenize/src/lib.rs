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
//! - exact byte identity verification before a higher layer admits a tokenizer
//!
//! Chat-template rendering is **not** done here. A higher layer must bind any
//! template to the same verified model artifact; this crate stays pure.

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

use std::fmt;
use std::num::NonZeroUsize;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::{
    ByteLengthMismatchSnafu, ByteLimitExceededSnafu, DigestMismatchSnafu, InvalidByteLimitSnafu,
    UpstreamSnafu,
};

pub use crate::error::{Error, Result};

const SHA256_BYTES: usize = 32;

/// SHA-256 identity of one exact tokenizer byte sequence.
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct TokenizerDigest([u8; SHA256_BYTES]);

impl TokenizerDigest {
    /// Construct a digest from canonical SHA-256 bytes.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; SHA256_BYTES]) -> Self {
        Self(bytes)
    }

    /// Borrow canonical digest bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; SHA256_BYTES] {
        &self.0
    }
}

impl fmt::Debug for TokenizerDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("TokenizerDigest(sha256)")
    }
}

/// Maximum tokenizer bytes accepted for verification and parser input.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct TokenizerByteLimit(NonZeroUsize);

impl TokenizerByteLimit {
    /// Construct a byte limit from an ordinary maximum.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidByteLimit`] when `maximum` is zero.
    pub fn try_new(maximum: usize) -> Result<Self> {
        let maximum =
            NonZeroUsize::new(maximum).ok_or_else(|| InvalidByteLimitSnafu { maximum }.build())?;
        Ok(Self(maximum))
    }

    /// Construct a non-zero maximum byte length.
    #[must_use]
    pub const fn new(maximum: NonZeroUsize) -> Self {
        Self(maximum)
    }

    /// Return the maximum accepted byte length.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0.get()
    }
}

/// Required identity for a separately supplied tokenizer byte sequence.
///
/// A tokenizer companion has no implicit path relationship to a GGUF file.
/// Callers must carry both its exact byte length and digest from an explicit
/// receipt before constructing [`VerifiedTokenizer`]. A digest establishes
/// content identity only; it does not establish publisher authenticity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TokenizerIdentity {
    byte_length: usize,
    digest: TokenizerDigest,
}

impl TokenizerIdentity {
    /// Construct an expected identity from independently recorded facts.
    #[must_use]
    pub const fn new(byte_length: usize, digest: TokenizerDigest) -> Self {
        Self {
            byte_length,
            digest,
        }
    }

    /// Return the required serialized byte length.
    #[must_use]
    pub const fn byte_length(self) -> usize {
        self.byte_length
    }

    /// Return the required SHA-256 digest.
    #[must_use]
    pub const fn digest(self) -> TokenizerDigest {
        self.digest
    }
}

/// A tokenizer parsed only after exact companion-byte verification.
///
/// This type does not establish compatibility with a model artifact. The text
/// pipeline performs that separate metadata-bound check before use.
pub struct VerifiedTokenizer {
    tokenizer: Tokenizer,
    identity: TokenizerIdentity,
}

impl VerifiedTokenizer {
    /// Verify and parse tokenizer bytes against an explicit identity.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ByteLengthMismatch`] or [`Error::DigestMismatch`] before
    /// parsing when the supplied bytes do not meet `limit` or `identity`, or
    /// [`Error::Upstream`] when matched bytes are not a supported tokenizer.
    pub fn from_bytes(
        bytes: &[u8],
        identity: TokenizerIdentity,
        limit: TokenizerByteLimit,
    ) -> Result<Self> {
        if bytes.len() > limit.get() {
            return ByteLimitExceededSnafu {
                limit: limit.get(),
                actual: bytes.len(),
            }
            .fail();
        }
        if bytes.len() != identity.byte_length() {
            return ByteLengthMismatchSnafu {
                expected: identity.byte_length(),
                actual: bytes.len(),
            }
            .fail();
        }
        let actual = TokenizerDigest::from_bytes(Sha256::digest(bytes).into());
        if actual != identity.digest() {
            return DigestMismatchSnafu.fail();
        }
        Ok(Self {
            tokenizer: Tokenizer::from_bytes(bytes)?,
            identity,
        })
    }

    /// Return the identity verified before parsing this tokenizer.
    #[must_use]
    pub const fn identity(&self) -> TokenizerIdentity {
        self.identity
    }

    /// Borrow the parsed tokenizer facade.
    #[must_use]
    pub const fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

impl fmt::Debug for VerifiedTokenizer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerifiedTokenizer")
            .field("identity", &self.identity)
            .field("vocab_size", &self.tokenizer.vocab_size())
            .finish()
    }
}

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
        let inner = ::tokenizers::Tokenizer::from_file(path)
            .map_err(|error| upstream_error(error.to_string()))?;
        Ok(Self { inner })
    }

    /// Parse a HuggingFace `tokenizer.json` byte sequence.
    ///
    /// Callers that receive a companion tokenizer should prefer
    /// [`VerifiedTokenizer::from_bytes`] so the byte identity is checked before
    /// parsing.
    ///
    /// # Errors
    ///
    /// [`Error::Upstream`] when the bytes are malformed.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let inner = ::tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|error| upstream_error(error.to_string()))?;
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
            .map_err(|error| upstream_error(error.to_string()))?;
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
            .map_err(|error| upstream_error(error.to_string()))
    }

    /// Vocabulary size, including added tokens.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Vocabulary size, base model only (no added tokens).
    pub fn vocab_size_base(&self) -> usize {
        self.inner.get_vocab_size(false)
    }

    /// Return the tokenizer string recorded for one token ID.
    #[must_use]
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Return the token ID recorded for one exact tokenizer string.
    #[must_use]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Return whether an added token ID is marked special by `tokenizer.json`.
    ///
    /// Base-vocabulary entries have no such marker and return `false`.
    #[must_use]
    pub fn is_special_token(&self, id: u32) -> bool {
        self.inner
            .get_added_tokens_decoder()
            .get(&id)
            .is_some_and(|token| token.special)
    }
}

#[track_caller]
fn upstream_error(message: String) -> Error {
    UpstreamSnafu { message }.build()
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

    use sha2::{Digest, Sha256};

    use super::*;

    const TRIVIAL_TOKENIZER: &str = r#"{
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

    /// Build a tiny WordLevel `tokenizer.json` on disk so the
    /// round-trip test runs without pulling any real model file.
    fn write_trivial_tokenizer(path: &Path) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        f.write_all(TRIVIAL_TOKENIZER.as_bytes())
    }

    #[test]
    fn round_trip_fixture() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "logismos-tokenize-test-{}.json",
            std::process::id()
        ));
        write_trivial_tokenizer(&tmp).map_err(|error| upstream_error(error.to_string()))?;

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
        write_trivial_tokenizer(&tmp).map_err(|error| upstream_error(error.to_string()))?;
        let tok = Tokenizer::from_file(&tmp)?;
        let ids = tok.encode("zephyr", false)?;
        assert_eq!(ids, vec![0]);
        let _ = std::fs::remove_file(&tmp);
        Ok(())
    }

    #[test]
    fn from_file_returns_upstream_error_for_missing_path() {
        let result = Tokenizer::from_file(Path::new("/tmp/nonexistent-tokenizer-12345.json"));
        assert!(matches!(result, Err(Error::Upstream { .. })));
    }

    #[test]
    fn vocab_size_base_matches_vocab_size_for_trivial_fixture() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "logismos-tokenize-vocab-{}.json",
            std::process::id()
        ));
        write_trivial_tokenizer(&tmp).map_err(|error| upstream_error(error.to_string()))?;
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
        write_trivial_tokenizer(&tmp).map_err(|error| upstream_error(error.to_string()))?;
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
        write_trivial_tokenizer(&tmp).map_err(|error| upstream_error(error.to_string()))?;
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
        write_trivial_tokenizer(&tmp).map_err(|error| upstream_error(error.to_string()))?;
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
        write_trivial_tokenizer(&tmp).map_err(|error| upstream_error(error.to_string()))?;
        let tok = Tokenizer::from_file(&tmp)?;
        let dbg = format!("{tok:?}");
        assert!(dbg.contains("Tokenizer"));
        assert!(dbg.contains("vocab_size"));
        assert!(dbg.contains('6'));
        let _ = std::fs::remove_file(&tmp);
        Ok(())
    }

    #[test]
    fn verified_tokenizer_rejects_any_identity_mismatch() -> Result<()> {
        let bytes = TRIVIAL_TOKENIZER.as_bytes();
        let digest = TokenizerDigest::from_bytes(Sha256::digest(bytes).into());
        let identity = TokenizerIdentity::new(bytes.len(), digest);
        let limit = TokenizerByteLimit::try_new(bytes.len())?;
        let verified = VerifiedTokenizer::from_bytes(bytes, identity, limit)?;
        assert_eq!(verified.identity(), identity);
        assert_eq!(
            verified.tokenizer().id_to_token(1).as_deref(),
            Some("hello")
        );
        assert_eq!(verified.tokenizer().token_to_id("world"), Some(2));

        let wrong_length = TokenizerIdentity::new(bytes.len() + 1, digest);
        assert!(matches!(
            VerifiedTokenizer::from_bytes(bytes, wrong_length, limit),
            Err(Error::ByteLengthMismatch { .. })
        ));

        let wrong_digest =
            TokenizerIdentity::new(bytes.len(), TokenizerDigest::from_bytes([0; 32]));
        assert!(matches!(
            VerifiedTokenizer::from_bytes(bytes, wrong_digest, limit),
            Err(Error::DigestMismatch { .. })
        ));
        Ok(())
    }
}
