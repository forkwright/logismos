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

mod collective;

use std::fmt;
use std::num::NonZeroUsize;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::{
    ByteLengthMismatchSnafu, ByteLimitExceededSnafu, ConfiguredPaddingSnafu,
    ConfiguredTruncationSnafu, DigestMismatchSnafu, ExpectedVocabularyLengthMismatchSnafu,
    InvalidByteLimitSnafu, SpecialTokenEncodingMismatchSnafu, SpecialTokenIdOutOfRangeSnafu,
    SpecialTokenMissingSnafu, SpecialTokenNotMarkedSnafu, UpstreamSnafu,
    VocabularyIdOutOfRangeSnafu, VocabularyLengthMismatchSnafu, VocabularyMismatchSnafu,
};

pub use crate::collective::{DecodeStorage, DecodeStoragePlan};
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

    /// Refuse a tokenizer that can silently pad or truncate native input.
    ///
    /// Native text, embedding, and reranking boundaries own their explicit
    /// request limits and special-token policies. This check leaves upstream
    /// settings intact so ordinary tokenizer consumers retain their configured
    /// behavior.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ConfiguredPadding`] or [`Error::ConfiguredTruncation`]
    /// when the verified upstream tokenizer retains either setting.
    pub fn verify_unpadded_untruncated(&self) -> Result<()> {
        if self.tokenizer.inner.get_padding().is_some() {
            return ConfiguredPaddingSnafu.fail();
        }
        if self.tokenizer.inner.get_truncation().is_some() {
            return ConfiguredTruncationSnafu.fail();
        }
        Ok(())
    }

    /// Verify an expected ordered vocabulary against this exact tokenizer.
    ///
    /// `expected_count` is supplied separately so callers can stream borrowed
    /// spellings from an artifact without allocating a second vocabulary. The
    /// iterator must produce exactly that count.
    /// Each expected spelling must appear at precisely its sequential token ID,
    /// and resolve back to that same ID through the tokenizer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::VocabularyLengthMismatch`] when the expected and
    /// tokenizer vocabulary counts differ,
    /// [`Error::ExpectedVocabularyLengthMismatch`] when the iterator disagrees
    /// with `expected_count`, or [`Error::VocabularyMismatch`] when either
    /// direction of an expected ID/spelling association differs, or
    /// [`Error::VocabularyIdOutOfRange`] when a sequential position cannot fit
    /// the tokenizer's `u32` ID domain.
    pub fn verify_exact_vocabulary<'expected>(
        &self,
        expected_count: usize,
        expected: impl Iterator<Item = &'expected str>,
    ) -> Result<()> {
        let actual_count = self.tokenizer.vocab_size();
        if expected_count != actual_count {
            return VocabularyLengthMismatchSnafu {
                expected: expected_count,
                actual: actual_count,
            }
            .fail();
        }
        let mut expected = expected;
        for index in 0..expected_count {
            let spelling = expected.next().ok_or_else(|| {
                ExpectedVocabularyLengthMismatchSnafu {
                    expected: expected_count,
                    actual: index,
                }
                .build()
            })?;
            let id = vocabulary_id(index)?;
            if self.tokenizer.id_to_token(id).as_deref() != Some(spelling)
                || self.tokenizer.token_to_id(spelling) != Some(id)
            {
                return VocabularyMismatchSnafu { id }.fail();
            }
        }
        if expected.next().is_some() {
            let actual = expected_count.checked_add(1).ok_or_else(|| {
                ExpectedVocabularyLengthMismatchSnafu {
                    expected: expected_count,
                    actual: expected_count,
                }
                .build()
            })?;
            return ExpectedVocabularyLengthMismatchSnafu {
                expected: expected_count,
                actual,
            }
            .fail();
        }
        Ok(())
    }

    /// Verify one declared special-token ID against an already selected vocabulary.
    ///
    /// The caller supplies the selected vocabulary size because this tokenizer
    /// does not own an artifact vocabulary. Callers should first use
    /// [`Self::verify_exact_vocabulary`] for that vocabulary, then use this
    /// method for each policy-selected special ID.
    ///
    /// # Errors
    ///
    /// Returns a typed special-token error when `id` is outside the selected
    /// vocabulary, lacks a tokenizer spelling, is not marked special, or does
    /// not encode by itself back to `id`; propagates [`Error::Upstream`] when
    /// the tokenizer cannot encode the declared spelling.
    pub fn verify_declared_special_id(&self, vocabulary_size: usize, id: u32) -> Result<()> {
        let index = usize::try_from(id).map_err(|_| {
            SpecialTokenIdOutOfRangeSnafu {
                id,
                vocabulary_size,
            }
            .build()
        })?;
        if index >= vocabulary_size {
            return SpecialTokenIdOutOfRangeSnafu {
                id,
                vocabulary_size,
            }
            .fail();
        }
        let spelling = self
            .tokenizer
            .id_to_token(id)
            .ok_or_else(|| SpecialTokenMissingSnafu { id }.build())?;
        if !self.tokenizer.is_special_token(id) {
            return SpecialTokenNotMarkedSnafu { id }.fail();
        }
        let encoded = self.tokenizer.encode(&spelling, false)?;
        if encoded.as_slice() != [id] {
            return SpecialTokenEncodingMismatchSnafu { id }.fail();
        }
        Ok(())
    }
}

fn vocabulary_id(index: usize) -> Result<u32> {
    u32::try_from(index).map_err(|_| VocabularyIdOutOfRangeSnafu { index }.build())
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
    collective_decoder: collective::CollectiveDecoder,
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
        Self::from_inner(inner)
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
        Self::from_inner(inner)
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

    /// Derive the complete owned storage needed by one bounded collective decode.
    ///
    /// The returned plan covers generated IDs, retained output, both transform
    /// arenas, their sparse span indexes, and byte-oriented transform scratch.
    /// It does not claim to bound allocator metadata or the native Onig regex
    /// engine's internal region and match-stack allocations.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DecodePlanOverflow`] when any derived capacity or
    /// requested byte-accounting sum exceeds `usize`.
    pub fn decode_storage_plan(
        &self,
        token_capacity: usize,
        output_byte_limit: usize,
    ) -> Result<DecodeStoragePlan> {
        self.collective_decoder
            .storage_plan(token_capacity, output_byte_limit)
    }

    /// Return owned persistent bytes attributable to the compiled decoder containers.
    ///
    /// This includes actual capacities for the packed spelling slab, sparse
    /// index, flattened stage vector, and its owned configuration strings. It
    /// deliberately excludes the upstream tokenizer, allocator metadata, and
    /// native Onig regions and match stacks; those require separate residency
    /// and runtime qualification.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DecodePlanOverflow`] if summing actual container
    /// capacities exceeds `usize`.
    pub fn collective_decoder_resident_bytes(&self) -> Result<usize> {
        self.collective_decoder.owned_resident_bytes()
    }

    /// Decode acquired generated-ID storage without any container growth.
    ///
    /// The returned string and ID vector are moved directly out of `storage`;
    /// all transform scratch is dropped before this method returns.
    ///
    /// # Errors
    ///
    /// Returns a typed capacity, UTF-8-invariant, or decoded-byte-limit error.
    pub fn decode_with_storage(
        &self,
        storage: DecodeStorage,
        skip_special_tokens: bool,
    ) -> Result<(String, Vec<u32>)> {
        self.collective_decoder.decode(storage, skip_special_tokens)
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

    fn from_inner(inner: ::tokenizers::Tokenizer) -> Result<Self> {
        let collective_decoder = collective::CollectiveDecoder::compile(&inner)?;
        Ok(Self {
            inner,
            collective_decoder,
        })
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

    const SPECIAL_TOKENIZER: &str = r#"{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [
        {"id": 1, "content": "<special>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
      ],
      "normalizer": null,
      "pre_tokenizer": { "type": "Whitespace" },
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "WordLevel",
        "vocab": {
          "[UNK]": 0,
          "<special>": 1,
          "hello": 2
        },
        "unk_token": "[UNK]"
      }
    }"#;

    const SPARSE_TOKENIZER: &str = r#"{
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
          "hello": 2
        },
        "unk_token": "[UNK]"
      }
    }"#;

    const CONFIGURED_TRUNCATION: &str = r#"{
      "version": "1.0",
      "truncation": {"direction":"Right","max_length":1,"strategy":"LongestFirst","stride":0},
      "padding": null,
      "added_tokens": [],
      "normalizer": null,
      "pre_tokenizer": { "type": "Whitespace" },
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "WordLevel",
        "vocab": {"[UNK]":0,"hello":1,"world":2},
        "unk_token": "[UNK]"
      }
    }"#;

    const CONFIGURED_PADDING: &str = r#"{
      "version": "1.0",
      "truncation": null,
      "padding": {"strategy":{"Fixed":4},"direction":"Right","pad_to_multiple_of":null,"pad_id":0,"pad_type_id":0,"pad_token":"[UNK]"},
      "added_tokens": [],
      "normalizer": null,
      "pre_tokenizer": { "type": "Whitespace" },
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "WordLevel",
        "vocab": {"[UNK]":0,"hello":1,"world":2},
        "unk_token": "[UNK]"
      }
    }"#;

    fn verified_tokenizer(bytes: &[u8]) -> Result<VerifiedTokenizer> {
        let digest = TokenizerDigest::from_bytes(Sha256::digest(bytes).into());
        let identity = TokenizerIdentity::new(bytes.len(), digest);
        let limit = TokenizerByteLimit::try_new(bytes.len())?;
        VerifiedTokenizer::from_bytes(bytes, identity, limit)
    }

    #[test]
    fn verified_tokenizer_accepts_null_padding_and_truncation() -> Result<()> {
        verified_tokenizer(TRIVIAL_TOKENIZER.as_bytes())?.verify_unpadded_untruncated()
    }

    #[test]
    fn verified_tokenizer_refuses_configured_truncation_without_mutating_it() -> Result<()> {
        let ordinary = Tokenizer::from_bytes(CONFIGURED_TRUNCATION.as_bytes())?;
        let ordinary_ids = ordinary.encode("hello world", false)?;
        assert_eq!(
            ordinary_ids,
            vec![1],
            "configured truncation must affect ordinary encode"
        );
        let verified = verified_tokenizer(CONFIGURED_TRUNCATION.as_bytes())?;
        assert!(
            matches!(
                verified.verify_unpadded_untruncated(),
                Err(Error::ConfiguredTruncation { .. })
            ),
            "verified tokenizer must refuse configured truncation"
        );
        assert_eq!(
            verified.tokenizer().encode("hello world", false)?,
            ordinary_ids,
            "verification must not mutate ordinary tokenizer truncation"
        );
        Ok(())
    }

    #[test]
    fn verified_tokenizer_refuses_configured_padding_without_mutating_it() -> Result<()> {
        let ordinary = Tokenizer::from_bytes(CONFIGURED_PADDING.as_bytes())?;
        let ordinary_ids = ordinary.encode("hello", false)?;
        assert_eq!(
            ordinary_ids.len(),
            4,
            "configured padding must affect ordinary encode"
        );
        let verified = verified_tokenizer(CONFIGURED_PADDING.as_bytes())?;
        assert!(
            matches!(
                verified.verify_unpadded_untruncated(),
                Err(Error::ConfiguredPadding { .. })
            ),
            "verified tokenizer must refuse configured padding"
        );
        assert_eq!(
            verified.tokenizer().encode("hello", false)?,
            ordinary_ids,
            "verification must not mutate ordinary tokenizer padding"
        );
        Ok(())
    }

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

    #[test]
    fn verified_tokenizer_requires_an_exact_ordered_vocabulary() -> Result<()> {
        let verified = verified_tokenizer(TRIVIAL_TOKENIZER.as_bytes())?;
        let expected = ["[UNK]", "hello", "world", "the", "quick", "fox"];
        verified.verify_exact_vocabulary(expected.len(), expected.into_iter())?;

        assert!(matches!(
            verified.verify_exact_vocabulary(expected.len() - 1, expected.into_iter()),
            Err(Error::VocabularyLengthMismatch { .. })
        ));
        assert!(matches!(
            verified.verify_exact_vocabulary(expected.len() + 1, expected.into_iter()),
            Err(Error::VocabularyLengthMismatch { .. })
        ));
        assert!(matches!(
            verified.verify_exact_vocabulary(
                expected.len(),
                ["[UNK]", "hello", "hello", "the", "quick", "fox"].into_iter(),
            ),
            Err(Error::VocabularyMismatch { id: 2, .. })
        ));
        assert!(matches!(
            verified.verify_exact_vocabulary(expected.len(), expected[..5].iter().copied()),
            Err(Error::ExpectedVocabularyLengthMismatch { actual: 5, .. })
        ));
        assert!(matches!(
            verified.verify_exact_vocabulary(
                expected.len(),
                ["[UNK]", "hello", "world", "the", "quick", "fox", "extra"].into_iter(),
            ),
            Err(Error::ExpectedVocabularyLengthMismatch { actual: 7, .. })
        ));
        Ok(())
    }

    #[test]
    fn vocabulary_position_refuses_u32_domain_overflow() {
        let Some(index) = usize::try_from(u32::MAX)
            .ok()
            .and_then(|maximum| maximum.checked_add(1))
        else {
            return;
        };
        assert!(matches!(
            vocabulary_id(index),
            Err(Error::VocabularyIdOutOfRange { index: actual, .. }) if actual == index
        ));
    }

    #[test]
    fn verified_tokenizer_checks_declared_special_ids() -> Result<()> {
        let special = verified_tokenizer(SPECIAL_TOKENIZER.as_bytes())?;
        let expected = ["[UNK]", "<special>", "hello"];
        special.verify_exact_vocabulary(expected.len(), expected.into_iter())?;
        special.verify_declared_special_id(expected.len(), 1)?;
        assert!(matches!(
            special.verify_declared_special_id(expected.len(), 3),
            Err(Error::SpecialTokenIdOutOfRange { .. })
        ));

        let ordinary = verified_tokenizer(TRIVIAL_TOKENIZER.as_bytes())?;
        assert!(matches!(
            ordinary.verify_declared_special_id(ordinary.tokenizer().vocab_size(), 1),
            Err(Error::SpecialTokenNotMarked { id: 1, .. })
        ));

        let sparse = verified_tokenizer(SPARSE_TOKENIZER.as_bytes())?;
        assert!(matches!(
            sparse.verify_declared_special_id(sparse.tokenizer().vocab_size(), 1),
            Err(Error::SpecialTokenMissing { id: 1, .. })
        ));
        Ok(())
    }
}

#[cfg(test)]
mod collective_tests;
