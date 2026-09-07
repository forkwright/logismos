//! Bounded CPU Qwen3 retrieval embeddings.

use decoders::Qwen3Weights;
use loader::gguf::{MetaValue, VerifiedArtifact};
use logismos_core::{
    ComputeSnafu as CoreComputeSnafu, EmbeddingError, EmbeddingModel, EncodeOpts,
    InputTooLongSnafu as CoreInputTooLongSnafu, Prompt, TokenizeSnafu as CoreTokenizeSnafu,
    UnsupportedDimSnafu as CoreUnsupportedDimSnafu,
    UnsupportedPromptSnafu as CoreUnsupportedPromptSnafu,
};
use snafu::ResultExt;
use tokenize::VerifiedTokenizer;

use crate::error::{
    AllocationSnafu, BatchTooLargeSnafu, DecodersSnafu, EmptyInputSnafu,
    InputByteLengthOverflowSnafu, InputBytesTooLongSnafu, InputTooLongSnafu, InvalidLimitsSnafu,
    MetadataSnafu, NonNormalizableSnafu, Qwen3TokenizerSnafu, Result, UnresolvedPromptRoleSnafu,
    UnsupportedDimSnafu,
};

const TOKENS: &str = "tokenizer.ggml.tokens";
const BOS: &str = "tokenizer.ggml.bos_token_id";
const EOS: &str = "tokenizer.ggml.eos_token_id";
const ADD_BOS: &str = "tokenizer.ggml.add_bos_token";
const ADD_EOS: &str = "tokenizer.ggml.add_eos_token";
const UNIT_NORM_TOLERANCE: f64 = 1e-6;

/// Trusted instructions for semantic retrieval roles.
#[derive(Clone, Debug, Default)]
pub struct Qwen3RolePrefixes {
    /// Exact trusted source-pinned instruction for sentence-to-sentence queries.
    pub s2s_query: Option<String>,
    /// Exact trusted source-pinned instruction for sentence-to-passage queries.
    pub s2p_query: Option<String>,
}

/// Explicit CPU request limits for Qwen3 embeddings.
#[derive(Clone, Copy, Debug)]
pub struct Qwen3EmbeddingLimits {
    /// Maximum UTF-8 bytes across the trusted prefix and request text.
    pub max_text_bytes: usize,
    /// Maximum token IDs after prefix and GGUF-declared special-token policy.
    pub max_tokens: usize,
    /// Maximum input objects accepted by one bounded batch request.
    pub max_batch_items: usize,
}

/// Artifact-bound CPU Qwen3 embedding model with last-token pooling.
#[derive(Debug)]
pub struct Qwen3EmbeddingModel<'artifact> {
    weights: Qwen3Weights<'artifact>,
    tokenizer: VerifiedTokenizer,
    max_text_bytes: usize,
    max_tokens: usize,
    max_batch_items: usize,
    prefixes: Qwen3RolePrefixes,
    bos: Option<u32>,
    eos: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    supported: [usize; 1],
}

impl<'artifact> Qwen3EmbeddingModel<'artifact> {
    /// Construct a bounded CPU-only Qwen3 embedding adapter from trusted setup.
    pub fn from_verified_cpu(
        artifact: &'artifact VerifiedArtifact,
        tokenizer: VerifiedTokenizer,
        limits: Qwen3EmbeddingLimits,
        prefixes: Qwen3RolePrefixes,
    ) -> Result<Self> {
        if limits.max_text_bytes == 0 || limits.max_tokens == 0 || limits.max_batch_items == 0 {
            return InvalidLimitsSnafu {
                rule: "setup byte, token, and batch-item limits must be nonzero",
            }
            .fail();
        }
        let metadata = artifact.observation().metadata();
        let values = match metadata.get(TOKENS) {
            Some(MetaValue::Array(values)) => values.values(),
            _ => return MetadataSnafu { key: TOKENS }.fail(),
        };
        if values
            .iter()
            .any(|value| !matches!(value, MetaValue::String(_)))
        {
            return MetadataSnafu { key: TOKENS }.fail();
        }
        tokenizer
            .verify_exact_vocabulary(
                values.len(),
                values.iter().filter_map(|value| match value {
                    MetaValue::String(value) => Some(value.as_str()),
                    _ => None,
                }),
            )
            .context(Qwen3TokenizerSnafu)?;
        tokenizer
            .verify_unpadded_untruncated()
            .context(Qwen3TokenizerSnafu)?;
        let insert_bos = flag(metadata, ADD_BOS)?;
        let append_eos = flag(metadata, ADD_EOS)?;
        let bos = id(metadata, BOS)?;
        let eos = id(metadata, EOS)?;
        if insert_bos {
            tokenizer
                .verify_declared_special_id(
                    values.len(),
                    bos.ok_or_else(|| MetadataSnafu { key: BOS }.build())?,
                )
                .context(Qwen3TokenizerSnafu)?;
        }
        if append_eos {
            tokenizer
                .verify_declared_special_id(
                    values.len(),
                    eos.ok_or_else(|| MetadataSnafu { key: EOS }.build())?,
                )
                .context(Qwen3TokenizerSnafu)?;
        }
        let weights = Qwen3Weights::try_from_verified(artifact).context(DecodersSnafu)?;
        let hidden = weights.hidden_width();
        let max_context = weights.max_context();
        if limits.max_tokens > max_context {
            return InvalidLimitsSnafu {
                rule: "setup token limit exceeds artifact context",
            }
            .fail();
        }
        Ok(Self {
            weights,
            tokenizer,
            max_text_bytes: limits.max_text_bytes,
            max_tokens: limits.max_tokens,
            max_batch_items: limits.max_batch_items,
            prefixes,
            bos,
            eos,
            add_bos: insert_bos,
            add_eos: append_eos,
            supported: [hidden],
        })
    }
    /// Encode with detailed native errors.
    pub fn encode_cpu(&self, text: &str, opts: &EncodeOpts) -> Result<Vec<f32>> {
        self.encode_with_policy(text, opts, self.request_policy(opts)?)
    }

    fn encode_with_policy(
        &self,
        text: &str,
        opts: &EncodeOpts,
        policy: RequestPolicy,
    ) -> Result<Vec<f32>> {
        if text.is_empty() {
            return EmptyInputSnafu.fail();
        }
        let text = prefix(
            &self.prefixes,
            opts.prompt.as_ref(),
            text,
            self.max_text_bytes,
        )?;
        let mut ids = self
            .tokenizer
            .tokenizer()
            .encode(&text, false)
            .context(Qwen3TokenizerSnafu)?;
        ids.try_reserve(2).context(AllocationSnafu {
            target: "embedding special-token policy",
        })?;
        if self.add_bos {
            ids.insert(
                0,
                self.bos
                    .ok_or_else(|| UnresolvedPromptRoleSnafu { role: "BOS" }.build())?,
            );
        }
        if self.add_eos {
            ids.push(
                self.eos
                    .ok_or_else(|| UnresolvedPromptRoleSnafu { role: "EOS" }.build())?,
            );
        }
        if ids.is_empty() || ids.len() > policy.max_tokens {
            return InputTooLongSnafu {
                got: ids.len(),
                limit: policy.max_tokens,
            }
            .fail();
        }
        let hidden = self
            .weights
            .execution(policy.max_tokens)
            .context(DecodersSnafu)?
            .last_hidden(&ids)
            .context(DecodersSnafu)?;
        normalize(hidden)
    }

    fn request_policy(&self, opts: &EncodeOpts) -> Result<RequestPolicy> {
        let hidden = self.supported[0];
        if let Some(dim) = opts.dim
            && dim != hidden
        {
            return UnsupportedDimSnafu { dim }.fail();
        }
        let limit = opts.max_tokens.unwrap_or(self.max_tokens);
        if limit == 0 || limit > self.max_tokens {
            return InvalidLimitsSnafu {
                rule: "request token limit must be nonzero and no greater than setup limit",
            }
            .fail();
        }
        Ok(RequestPolicy { max_tokens: limit })
    }

    fn encode_batch_cpu(&self, texts: &[&str], opts: &EncodeOpts) -> Result<Vec<Vec<f32>>> {
        let policy = self.request_policy(opts)?;
        if texts.len() > self.max_batch_items {
            return BatchTooLargeSnafu {
                actual: texts.len(),
                limit: self.max_batch_items,
            }
            .fail();
        }
        let mut vectors = Vec::new();
        vectors
            .try_reserve_exact(texts.len())
            .context(AllocationSnafu {
                target: "embedding batch result vectors",
            })?;
        for text in texts {
            vectors.push(self.encode_with_policy(text, opts, policy)?);
        }
        Ok(vectors)
    }
}

#[derive(Clone, Copy)]
struct RequestPolicy {
    max_tokens: usize,
}

impl EmbeddingModel for Qwen3EmbeddingModel<'_> {
    fn default_dim(&self) -> usize {
        self.supported[0]
    }
    fn supported_dims(&self) -> &[usize] {
        &self.supported
    }
    fn max_tokens(&self) -> usize {
        self.max_tokens
    }
    fn encode(
        &self,
        text: &str,
        opts: &EncodeOpts,
    ) -> std::result::Result<Vec<f32>, EmbeddingError> {
        self.encode_cpu(text, opts).map_err(map_stable_error)
    }

    fn encode_batch(
        &self,
        texts: &[&str],
        opts: &EncodeOpts,
    ) -> std::result::Result<Vec<Vec<f32>>, EmbeddingError> {
        self.encode_batch_cpu(texts, opts).map_err(map_stable_error)
    }
}

fn map_stable_error(error: crate::error::Error) -> EmbeddingError {
    match error {
        crate::error::Error::UnsupportedDim { dim, .. } => CoreUnsupportedDimSnafu { dim }.build(),
        crate::error::Error::InputTooLong { got, limit, .. } => {
            CoreInputTooLongSnafu { got, limit }.build()
        }
        crate::error::Error::Qwen3Tokenizer { source, .. } => CoreTokenizeSnafu {
            message: source.to_string(),
        }
        .build(),
        crate::error::Error::UnresolvedPromptRole { .. } => CoreUnsupportedPromptSnafu.build(),
        error => CoreComputeSnafu {
            message: error.to_string(),
        }
        .build(),
    }
}
fn flag(
    metadata: &std::collections::HashMap<String, MetaValue>,
    key: &'static str,
) -> Result<bool> {
    match metadata.get(key) {
        None => Ok(false),
        Some(MetaValue::Bool(value)) => Ok(*value),
        _ => MetadataSnafu { key }.fail(),
    }
}
fn id(
    metadata: &std::collections::HashMap<String, MetaValue>,
    key: &'static str,
) -> Result<Option<u32>> {
    match metadata.get(key) {
        None => Ok(None),
        Some(MetaValue::U32(value)) => Ok(Some(*value)),
        _ => MetadataSnafu { key }.fail(),
    }
}
fn prefix(
    prefixes: &Qwen3RolePrefixes,
    prompt: Option<&Prompt>,
    text: &str,
    limit: usize,
) -> Result<String> {
    let prefix = match prompt {
        None => "",
        Some(Prompt::Custom(prefix)) => prefix,
        Some(Prompt::S2sQuery) => prefixes
            .s2s_query
            .as_deref()
            .ok_or_else(|| UnresolvedPromptRoleSnafu { role: "S2sQuery" }.build())?,
        Some(Prompt::S2pQuery) => prefixes
            .s2p_query
            .as_deref()
            .ok_or_else(|| UnresolvedPromptRoleSnafu { role: "S2pQuery" }.build())?,
        Some(_) => return UnresolvedPromptRoleSnafu { role: "unknown" }.fail(),
    };
    let bytes = prefix
        .len()
        .checked_add(text.len())
        .ok_or_else(|| InputByteLengthOverflowSnafu.build())?;
    if bytes > limit {
        return InputBytesTooLongSnafu {
            actual: bytes,
            limit,
        }
        .fail();
    }
    let mut combined = String::new();
    combined.try_reserve_exact(bytes).context(AllocationSnafu {
        target: "embedding prefixed text",
    })?;
    combined.push_str(prefix);
    combined.push_str(text);
    Ok(combined)
}
fn normalize(mut values: Vec<f32>) -> Result<Vec<f32>> {
    let sum = values.iter().try_fold(0_f64, |sum, value| {
        if value.is_finite() {
            Ok(sum + f64::from(*value) * f64::from(*value))
        } else {
            Err(())
        }
    });
    let Ok(sum) = sum else {
        return NonNormalizableSnafu.fail();
    };
    let norm = sum.sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return NonNormalizableSnafu.fail();
    }
    for value in &mut values {
        *value = (f64::from(*value) / norm) as f32;
    }
    if values.iter().any(|value| !value.is_finite()) {
        return NonNormalizableSnafu.fail();
    }
    let output_norm = values
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    if !output_norm.is_finite() || (output_norm - 1.0).abs() > UNIT_NORM_TOLERANCE {
        return NonNormalizableSnafu.fail();
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;
    use std::num::{NonZeroU64, NonZeroUsize};

    use loader::gguf::{ArtifactByteLimit, Sha256Digest};
    use sha2::{Digest, Sha256};
    use test_fixtures::{RawGguf, RawMetadata, RawMetadataValue, RawTensor, serialize_raw_gguf};
    use tokenize::{TokenizerByteLimit, TokenizerDigest, TokenizerIdentity};

    use super::*;

    const HIDDEN: u64 = 3;
    const HEADS: u64 = 2;
    const KV_HEADS: u64 = 1;
    const HEAD_DIM: u64 = 2;
    const FEED_FORWARD: u64 = 4;
    const VOCABULARY: u64 = 4;
    const CONTEXT: u32 = 4;

    #[test]
    fn prefix_obeys_explicit_roles_and_combined_byte_cap() -> Result<()> {
        let prefixes = Qwen3RolePrefixes {
            s2s_query: Some("query: ".to_string()),
            s2p_query: None,
        };
        assert_eq!(prefix(&prefixes, None, "doc", 3)?, "doc");
        assert_eq!(
            prefix(&prefixes, Some(&Prompt::Custom("x:".to_string())), "doc", 5)?,
            "x:doc"
        );
        assert_eq!(
            prefix(&prefixes, Some(&Prompt::S2sQuery), "doc", 10)?,
            "query: doc"
        );
        assert!(matches!(
            prefix(&prefixes, Some(&Prompt::S2pQuery), "doc", 10),
            Err(crate::error::Error::UnresolvedPromptRole { .. })
        ));
        assert!(matches!(
            prefix(&prefixes, Some(&Prompt::S2sQuery), "doc", 9),
            Err(crate::error::Error::InputBytesTooLong {
                actual: 10,
                limit: 9,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn normalization_returns_finite_unit_vectors_and_refuses_invalid_input() -> Result<()> {
        let values = normalize(vec![3.0, 4.0])?;
        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "normalization must produce a unit vector"
        );
        assert!(matches!(
            normalize(vec![0.0, 0.0]),
            Err(crate::error::Error::NonNormalizable { .. })
        ));
        assert!(matches!(
            normalize(vec![f32::NAN]),
            Err(crate::error::Error::NonNormalizable { .. })
        ));
        Ok(())
    }

    #[test]
    fn verified_qwen3_uses_only_explicit_prefix_and_special_token_policy()
    -> std::result::Result<(), Box<dyn StdError>> {
        let artifact = artifact(&fixture()?)?;
        let word_level = verified_tokenizer(TokenizerModel::WordLevel)?;
        let word_piece = verified_tokenizer(TokenizerModel::WordPiece)?;
        let limits = Qwen3EmbeddingLimits {
            max_text_bytes: 32,
            max_tokens: 4,
            max_batch_items: 4,
        };
        let prefixes = Qwen3RolePrefixes {
            s2s_query: Some("alice ".to_owned()),
            s2p_query: None,
        };
        let level = Qwen3EmbeddingModel::from_verified_cpu(
            &artifact,
            word_level,
            limits,
            prefixes.clone(),
        )?;
        let piece =
            Qwen3EmbeddingModel::from_verified_cpu(&artifact, word_piece, limits, prefixes)?;
        let default = EncodeOpts::default();
        let expected = normalize(
            Qwen3Weights::try_from_verified(&artifact)?
                .execution(4)?
                .last_hidden(&[0, 2, 1])?,
        )?;
        assert_eq!(level.encode_cpu("alice", &default)?, expected);
        assert_eq!(piece.encode_cpu("alice", &default)?, expected);
        let query = EncodeOpts {
            prompt: Some(Prompt::S2sQuery),
            ..EncodeOpts::default()
        };
        let expected_query = normalize(
            Qwen3Weights::try_from_verified(&artifact)?
                .execution(4)?
                .last_hidden(&[0, 2, 2, 1])?,
        )?;
        assert_eq!(level.encode_cpu("alice", &query)?, expected_query);
        assert_eq!(
            level.encode_cpu(
                "alice",
                &EncodeOpts {
                    prompt: Some(Prompt::Custom("alice ".to_owned())),
                    ..EncodeOpts::default()
                },
            )?,
            expected_query
        );
        Ok(())
    }

    #[test]
    fn qwen3_refuses_invalid_setup_and_request_limits() -> std::result::Result<(), Box<dyn StdError>>
    {
        let artifact = artifact(&fixture()?)?;
        assert!(matches!(
            Qwen3EmbeddingModel::from_verified_cpu(
                &artifact,
                verified_tokenizer(TokenizerModel::WordLevel)?,
                Qwen3EmbeddingLimits {
                    max_text_bytes: 20,
                    max_tokens: 3,
                    max_batch_items: 0,
                },
                Qwen3RolePrefixes::default(),
            ),
            Err(crate::error::Error::InvalidLimits { .. })
        ));
        let model = Qwen3EmbeddingModel::from_verified_cpu(
            &artifact,
            verified_tokenizer(TokenizerModel::WordLevel)?,
            Qwen3EmbeddingLimits {
                max_text_bytes: 20,
                max_tokens: 3,
                max_batch_items: 3,
            },
            Qwen3RolePrefixes::default(),
        )?;
        let opts = EncodeOpts::default();
        assert!(matches!(
            model.encode_cpu("", &opts),
            Err(crate::error::Error::EmptyInput { .. })
        ));
        assert!(matches!(
            model.encode_cpu("alice alice", &opts),
            Err(crate::error::Error::InputTooLong {
                got: 4,
                limit: 3,
                ..
            })
        ));
        assert!(matches!(
            model.encode_cpu(
                "alice alice",
                &EncodeOpts {
                    max_tokens: Some(4),
                    ..EncodeOpts::default()
                }
            ),
            Err(crate::error::Error::InvalidLimits { .. })
        ));
        assert!(matches!(
            model.encode_cpu("012345678901234567890", &opts),
            Err(crate::error::Error::InputBytesTooLong {
                actual: 21,
                limit: 20,
                ..
            })
        ));
        assert!(matches!(
            model.encode_cpu(
                "alice",
                &EncodeOpts {
                    dim: Some(2),
                    ..EncodeOpts::default()
                }
            ),
            Err(crate::error::Error::UnsupportedDim { dim: 2, .. })
        ));
        assert!(matches!(
            EmbeddingModel::encode(&model, "alice alice", &opts),
            Err(EmbeddingError::InputTooLong {
                got: 4,
                limit: 3,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn qwen3_batch_limits_preserve_pristine_retry() -> std::result::Result<(), Box<dyn StdError>> {
        let artifact = artifact(&fixture()?)?;
        let model = Qwen3EmbeddingModel::from_verified_cpu(
            &artifact,
            verified_tokenizer(TokenizerModel::WordLevel)?,
            Qwen3EmbeddingLimits {
                max_text_bytes: 20,
                max_tokens: 3,
                max_batch_items: 3,
            },
            Qwen3RolePrefixes::default(),
        )?;
        let opts = EncodeOpts::default();
        let first = model.encode_cpu("alice", &opts)?;
        let over_batch = ["alice", "alice", "alice", "alice"];
        assert!(matches!(
            model.encode_batch_cpu(&over_batch, &opts),
            Err(crate::error::Error::BatchTooLarge {
                actual: 4,
                limit: 3,
                ..
            })
        ));
        assert_eq!(model.encode_cpu("alice", &opts)?, first);
        let batch = ["alice", "bob", "alice"];
        let vectors = EmbeddingModel::encode_batch(&model, &batch, &opts)?;
        assert_eq!(vectors.len(), 3);
        assert!(vectors.iter().flatten().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn compatibility_failure_retains_the_tokenizer_source_chain()
    -> std::result::Result<(), Box<dyn StdError>> {
        let mut raw = fixture()?;
        raw.metadata.retain(|entry| entry.key != TOKENS);
        raw.metadata.push(RawMetadata {
            key: TOKENS.to_owned(),
            value: RawMetadataValue::StringArray(vec![
                "[BOS]".to_owned(),
                "[EOS]".to_owned(),
                "mallory".to_owned(),
                "bob".to_owned(),
            ]),
        });
        let Err(error) = Qwen3EmbeddingModel::from_verified_cpu(
            &artifact(&raw)?,
            verified_tokenizer(TokenizerModel::WordLevel)?,
            Qwen3EmbeddingLimits {
                max_text_bytes: 32,
                max_tokens: 4,
                max_batch_items: 4,
            },
            Qwen3RolePrefixes::default(),
        ) else {
            return Err("mismatched GGUF spelling must be rejected".into());
        };
        assert!(matches!(error, crate::error::Error::Qwen3Tokenizer { .. }));
        assert!(StdError::source(&error).is_some());
        Ok(())
    }

    #[test]
    fn qwen3_refuses_configured_tokenizer_padding_and_truncation()
    -> std::result::Result<(), Box<dyn StdError>> {
        let artifact = artifact(&fixture()?)?;
        for (settings, expected) in [
            (TokenizerSettings::ConfiguredPadding, "padding"),
            (TokenizerSettings::ConfiguredTruncation, "truncation"),
        ] {
            let Err(error) = Qwen3EmbeddingModel::from_verified_cpu(
                &artifact,
                verified_tokenizer_with_settings(TokenizerModel::WordLevel, settings)?,
                Qwen3EmbeddingLimits {
                    max_text_bytes: 32,
                    max_tokens: 4,
                    max_batch_items: 4,
                },
                Qwen3RolePrefixes::default(),
            ) else {
                return Err(format!("configured tokenizer {expected} must be rejected").into());
            };
            let crate::error::Error::Qwen3Tokenizer { source, .. } = error else {
                return Err(
                    format!("configured tokenizer {expected} must retain its source").into(),
                );
            };
            assert!(
                matches!(
                    (settings, source),
                    (
                        TokenizerSettings::ConfiguredPadding,
                        tokenize::Error::ConfiguredPadding { .. }
                    ) | (
                        TokenizerSettings::ConfiguredTruncation,
                        tokenize::Error::ConfiguredTruncation { .. }
                    )
                ),
                "native Qwen3 setup must retain the typed {expected} refusal"
            );
        }
        Ok(())
    }

    #[derive(Clone, Copy)]
    enum TokenizerModel {
        WordLevel,
        WordPiece,
    }

    #[derive(Clone, Copy)]
    enum TokenizerSettings {
        Null,
        ConfiguredPadding,
        ConfiguredTruncation,
    }

    fn verified_tokenizer(
        kind: TokenizerModel,
    ) -> std::result::Result<VerifiedTokenizer, Box<dyn StdError>> {
        verified_tokenizer_with_settings(kind, TokenizerSettings::Null)
    }

    fn verified_tokenizer_with_settings(
        kind: TokenizerModel,
        settings: TokenizerSettings,
    ) -> std::result::Result<VerifiedTokenizer, Box<dyn StdError>> {
        let model = match kind {
            TokenizerModel::WordLevel => {
                r#"{"type":"WordLevel","vocab":{"[BOS]":0,"[EOS]":1,"alice":2,"bob":3},"unk_token":"[UNK]"}"#
            }
            TokenizerModel::WordPiece => {
                r###"{"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100,"vocab":{"[BOS]":0,"[EOS]":1,"alice":2,"bob":3}}"###
            }
        };
        let (truncation, padding) = match settings {
            TokenizerSettings::Null => ("null", "null"),
            TokenizerSettings::ConfiguredPadding => (
                "null",
                r#"{"strategy":{"Fixed":4},"direction":"Right","pad_to_multiple_of":null,"pad_id":0,"pad_type_id":0,"pad_token":"[BOS]"}"#,
            ),
            TokenizerSettings::ConfiguredTruncation => (
                r#"{"direction":"Right","max_length":1,"strategy":"LongestFirst","stride":0}"#,
                "null",
            ),
        };
        let json = format!(
            r#"{{"version":"1.0","truncation":{truncation},"padding":{padding},"added_tokens":[{{"id":0,"content":"[BOS]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}},{{"id":1,"content":"[EOS]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}}],"normalizer":null,"pre_tokenizer":{{"type":"Whitespace"}},"post_processor":{{"type":"TemplateProcessing","single":[{{"Sequence":{{"id":"A","type_id":0}}}},{{"SpecialToken":{{"id":"[EOS]","type_id":0}}}}],"pair":[{{"Sequence":{{"id":"A","type_id":0}}}},{{"Sequence":{{"id":"B","type_id":1}}}},{{"SpecialToken":{{"id":"[EOS]","type_id":0}}}}],"special_tokens":{{"[EOS]":{{"id":"[EOS]","ids":[1],"tokens":["[EOS]"]}}}}}},"decoder":null,"model":{model}}}"#
        );
        let bytes = json.as_bytes();
        let length = NonZeroUsize::new(bytes.len()).ok_or("empty synthetic tokenizer")?;
        let identity = TokenizerIdentity::new(
            bytes.len(),
            TokenizerDigest::from_bytes(Sha256::digest(bytes).into()),
        );
        Ok(VerifiedTokenizer::from_bytes(
            bytes,
            identity,
            TokenizerByteLimit::new(length),
        )?)
    }

    fn artifact(raw: &RawGguf) -> std::result::Result<VerifiedArtifact, Box<dyn StdError>> {
        let serialized = serialize_raw_gguf(raw)?;
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("qwen3-embed.gguf");
        std::fs::write(&path, &serialized.bytes)?;
        let limit = NonZeroU64::new(serialized.byte_len + 1).ok_or("invalid fixture limit")?;
        Ok(VerifiedArtifact::load(
            &path,
            Sha256Digest::from_bytes(serialized.sha256),
            ArtifactByteLimit::new(limit),
        )?)
    }

    fn fixture() -> std::result::Result<RawGguf, Box<dyn StdError>> {
        let mut tensors = vec![
            tensor("token_embd.weight", vec![HIDDEN, VOCABULARY], 0.125)?,
            tensor("output_norm.weight", vec![HIDDEN], 1.0)?,
        ];
        for (role, dimensions, seed) in [
            ("attn_norm.weight", vec![HIDDEN], 1.0),
            ("attn_q_norm.weight", vec![HEAD_DIM], 1.0),
            ("attn_k_norm.weight", vec![HEAD_DIM], 1.0),
            ("ffn_norm.weight", vec![HIDDEN], 1.0),
            ("attn_q.weight", vec![HIDDEN, HEADS * HEAD_DIM], 0.0625),
            ("attn_k.weight", vec![HIDDEN, KV_HEADS * HEAD_DIM], 0.09375),
            ("attn_v.weight", vec![HIDDEN, KV_HEADS * HEAD_DIM], 0.125),
            (
                "attn_output.weight",
                vec![HEADS * HEAD_DIM, HIDDEN],
                0.15625,
            ),
            ("ffn_gate.weight", vec![HIDDEN, FEED_FORWARD], 0.1875),
            ("ffn_up.weight", vec![HIDDEN, FEED_FORWARD], 0.21875),
            ("ffn_down.weight", vec![FEED_FORWARD, HIDDEN], 0.25),
        ] {
            tensors.push(tensor(&format!("blk.0.{role}"), dimensions, seed)?);
        }
        Ok(RawGguf {
            metadata: vec![
                metadata_string("general.architecture", "qwen3"),
                metadata_u32("qwen3.block_count", 1),
                metadata_u32("qwen3.context_length", CONTEXT),
                metadata_u32("qwen3.embedding_length", u32::try_from(HIDDEN)?),
                metadata_u32("qwen3.feed_forward_length", u32::try_from(FEED_FORWARD)?),
                metadata_u32("qwen3.attention.head_count", u32::try_from(HEADS)?),
                metadata_u32("qwen3.attention.head_count_kv", u32::try_from(KV_HEADS)?),
                metadata_u32("qwen3.attention.key_length", u32::try_from(HEAD_DIM)?),
                metadata_u32("qwen3.attention.value_length", u32::try_from(HEAD_DIM)?),
                metadata_f32("qwen3.attention.layer_norm_rms_epsilon", 0.001),
                metadata_u32("qwen3.rope.dimension_count", u32::try_from(HEAD_DIM)?),
                metadata_f32("qwen3.rope.freq_base", 10_000.0),
                metadata_u32("qwen3.pooling_type", 3),
                RawMetadata {
                    key: TOKENS.to_owned(),
                    value: RawMetadataValue::StringArray(vec![
                        "[BOS]".to_owned(),
                        "[EOS]".to_owned(),
                        "alice".to_owned(),
                        "bob".to_owned(),
                    ]),
                },
                metadata_u32(BOS, 0),
                metadata_u32(EOS, 1),
                metadata_bool(ADD_BOS, true),
                metadata_bool(ADD_EOS, true),
            ],
            tensors,
        })
    }

    fn metadata_u32(key: &str, value: u32) -> RawMetadata {
        RawMetadata {
            key: key.to_owned(),
            value: RawMetadataValue::U32(value),
        }
    }

    fn metadata_f32(key: &str, value: f32) -> RawMetadata {
        RawMetadata {
            key: key.to_owned(),
            value: RawMetadataValue::F32(value),
        }
    }

    fn metadata_bool(key: &str, value: bool) -> RawMetadata {
        RawMetadata {
            key: key.to_owned(),
            value: RawMetadataValue::Bool(value),
        }
    }

    fn metadata_string(key: &str, value: &str) -> RawMetadata {
        RawMetadata {
            key: key.to_owned(),
            value: RawMetadataValue::String(value.to_owned()),
        }
    }

    fn tensor(
        name: &str,
        dims: Vec<u64>,
        seed: f32,
    ) -> std::result::Result<RawTensor, Box<dyn StdError>> {
        let count = dims.iter().try_fold(
            1_usize,
            |count, dimension| -> std::result::Result<usize, Box<dyn StdError>> {
                let dimension = usize::try_from(*dimension)?;
                count
                    .checked_mul(dimension)
                    .ok_or_else(|| std::io::Error::other("fixture element count overflow").into())
            },
        )?;
        let payload_len = count
            .checked_mul(4)
            .ok_or_else(|| std::io::Error::other("fixture payload overflow"))?;
        let mut payload = Vec::with_capacity(payload_len);
        for index in 0..count {
            payload.extend_from_slice(&(seed + index as f32 * 0.007_812_5).to_le_bytes());
        }
        Ok(RawTensor {
            name: name.to_owned(),
            dims,
            format: 0,
            payload,
        })
    }
}
