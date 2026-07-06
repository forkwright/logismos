//! ModernBERT configuration shape.
//!
//! Deserializes from `config.json` shipped with ModernBERT checkpoints.
//! Field names match the Hugging Face / TEI convention.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{ConfigSnafu, MissingClassifierHeadSnafu, Result, UnsupportedModelTypeSnafu};

/// ModernBERT configuration.
///
/// Sourced from the published `Alibaba-NLP/gte-reranker-modernbert-base`
/// `config.json` and TEI's ModernBERT implementation. Not every field is
/// used in Phase 5 preflight, but the shape accepts published checkpoint
/// metadata without losing the inference-critical values logismos needs.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "checkpoint config mirrors upstream JSON booleans one-to-one"
)]
#[non_exhaustive]
pub struct ModernBertConfig {
    /// Original model path recorded by the checkpoint.
    #[serde(default, rename = "_name_or_path")]
    pub name_or_path: String,
    /// Model architecture names recorded by Transformers.
    #[serde(default)]
    pub architectures: Vec<String>,
    /// Whether attention projections include bias.
    #[serde(default)]
    pub attention_bias: bool,
    /// `vocab_size`.
    pub vocab_size: usize,
    /// `hidden_size` (d_model).
    pub hidden_size: usize,
    /// `num_hidden_layers`.
    pub num_hidden_layers: usize,
    /// `num_attention_heads`.
    pub num_attention_heads: usize,
    /// `intermediate_size`.
    pub intermediate_size: usize,
    /// `max_position_embeddings`.
    pub max_position_embeddings: usize,
    /// `num_labels` for sequence classification (reranker head).
    #[serde(default)]
    pub num_labels: usize,
    /// Local attention window size.
    #[serde(default)]
    pub local_attention: usize,
    /// Apply global attention every N layers.
    #[serde(default = "default_global_every")]
    pub global_attn_every_n_layers: usize,
    /// RoPE theta for local attention.
    #[serde(default = "default_local_rope_theta")]
    pub local_rope_theta: f64,
    /// RoPE theta for global attention.
    #[serde(default = "default_global_rope_theta")]
    pub global_rope_theta: f64,
    /// LayerNorm epsilon.
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    /// Alias epsilon used by recent ModernBERT configs.
    #[serde(default = "default_layer_norm_eps")]
    pub norm_eps: f64,
    /// Hidden activation function name (e.g. `"gelu"`).
    #[serde(default = "default_hidden_activation", alias = "hidden_act")]
    pub hidden_activation: String,
    /// Dropout probability (unused at inference, but present in config).
    #[serde(default, alias = "hidden_dropout_prob")]
    pub embedding_dropout: f64,
    /// Attention dropout probability.
    #[serde(default, alias = "attention_probs_dropout_prob")]
    pub attention_dropout: f64,
    /// BOS token id.
    #[serde(default)]
    pub bos_token_id: usize,
    /// EOS token id.
    #[serde(default)]
    pub eos_token_id: usize,
    /// Token type vocabulary size.
    #[serde(default = "default_type_vocab_size")]
    pub type_vocab_size: usize,
    /// Initializer range.
    #[serde(default = "default_initializer_range")]
    pub initializer_range: f64,
    /// Pad token id.
    #[serde(default)]
    pub pad_token_id: usize,
    /// CLS token id.
    #[serde(default)]
    pub cls_token_id: usize,
    /// SEP token id.
    #[serde(default)]
    pub sep_token_id: usize,
    /// Whether the decoder (classifier) bias is enabled.
    #[serde(default)]
    pub decoder_bias: bool,
    /// Classifier activation.
    #[serde(default = "default_hidden_activation")]
    pub classifier_activation: String,
    /// Whether the classifier head includes bias.
    #[serde(default)]
    pub classifier_bias: bool,
    /// Classifier dropout probability.
    #[serde(default)]
    pub classifier_dropout: f64,
    /// Classifier pooling mode.
    #[serde(default = "default_classifier_pooling")]
    pub classifier_pooling: String,
    /// Whether deterministic flash attention was requested at training time.
    #[serde(default)]
    pub deterministic_flash_attn: bool,
    /// Whether gradient checkpointing is enabled in the saved config.
    #[serde(default)]
    pub gradient_checkpointing: bool,
    /// Label id-to-name map.
    #[serde(default)]
    pub id2label: BTreeMap<String, String>,
    /// Label name-to-id map.
    #[serde(default)]
    pub label2id: BTreeMap<String, usize>,
    /// Initializer cutoff factor.
    #[serde(default = "default_initializer_cutoff_factor")]
    pub initializer_cutoff_factor: f64,
    /// Whether MLP layers include bias.
    #[serde(default)]
    pub mlp_bias: bool,
    /// MLP dropout probability.
    #[serde(default)]
    pub mlp_dropout: f64,
    /// Transformers model type.
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// Whether normalization layers include bias.
    #[serde(default)]
    pub norm_bias: bool,
    /// Normalization type (e.g. `"layernorm"`).
    #[serde(default = "default_normalization_type")]
    pub normalization_type: String,
    /// Number of global tokens.
    #[serde(default)]
    pub num_global_tokens: usize,
    /// Position embedding type.
    #[serde(default = "default_position_embedding_type")]
    pub position_embedding_type: String,
    /// Sparse prediction ignore index.
    #[serde(default = "default_sparse_pred_ignore_index")]
    pub sparse_pred_ignore_index: isize,
    /// Whether sparse prediction is enabled.
    #[serde(default)]
    pub sparse_prediction: bool,
    /// Original checkpoint dtype.
    #[serde(default = "default_torch_dtype")]
    pub torch_dtype: String,
    /// Transformers version that wrote the config.
    #[serde(default)]
    pub transformers_version: String,
}

fn default_global_every() -> usize {
    3
}

fn default_local_rope_theta() -> f64 {
    10_000.0
}

fn default_global_rope_theta() -> f64 {
    160_000.0
}

fn default_layer_norm_eps() -> f64 {
    1e-5
}

fn default_hidden_activation() -> String {
    "gelu".into()
}

fn default_type_vocab_size() -> usize {
    2
}

fn default_initializer_range() -> f64 {
    0.02
}

fn default_initializer_cutoff_factor() -> f64 {
    2.0
}

fn default_classifier_pooling() -> String {
    "mean".into()
}

fn default_model_type() -> String {
    "modernbert".into()
}

fn default_normalization_type() -> String {
    "layernorm".into()
}

fn default_position_embedding_type() -> String {
    "absolute".into()
}

fn default_sparse_pred_ignore_index() -> isize {
    -100
}

fn default_torch_dtype() -> String {
    "float32".into()
}

impl ModernBertConfig {
    /// Canonical configuration for `Alibaba-NLP/gte-reranker-modernbert-base`.
    ///
    /// These values match the published checkpoint's `config.json`
    /// (149 M params, max input length 8192).
    #[must_use]
    pub fn gte_reranker_modernbert_base() -> Self {
        Self {
            name_or_path: "gte-reranker-modernbert-base".into(),
            architectures: vec!["ModernBertForSequenceClassification".into()],
            attention_bias: false,
            vocab_size: 50_368,
            hidden_size: 768,
            num_hidden_layers: 22,
            num_attention_heads: 12,
            intermediate_size: 1_152,
            max_position_embeddings: 8_192,
            num_labels: 1,
            local_attention: 128,
            global_attn_every_n_layers: 3,
            local_rope_theta: 10_000.0,
            global_rope_theta: 160_000.0,
            layer_norm_eps: 1e-5,
            norm_eps: 1e-5,
            hidden_activation: "gelu".into(),
            embedding_dropout: 0.0,
            attention_dropout: 0.0,
            bos_token_id: 50_281,
            eos_token_id: 50_282,
            type_vocab_size: 2,
            initializer_range: 0.02,
            pad_token_id: 50_283,
            cls_token_id: 50_281,
            sep_token_id: 50_282,
            decoder_bias: true,
            classifier_activation: "gelu".into(),
            classifier_bias: false,
            classifier_dropout: 0.0,
            classifier_pooling: "mean".into(),
            deterministic_flash_attn: false,
            gradient_checkpointing: false,
            id2label: BTreeMap::from([("0".into(), "LABEL_0".into())]),
            label2id: BTreeMap::from([("LABEL_0".into(), 0)]),
            initializer_cutoff_factor: 2.0,
            mlp_bias: false,
            mlp_dropout: 0.0,
            model_type: "modernbert".into(),
            norm_bias: false,
            normalization_type: "layernorm".into(),
            num_global_tokens: 0,
            position_embedding_type: "absolute".into(),
            sparse_pred_ignore_index: -100,
            sparse_prediction: false,
            torch_dtype: "float32".into(),
            transformers_version: "4.48.0.dev0".into(),
        }
    }

    /// Validate the config fields required before loading a GTE reranker.
    ///
    /// # Errors
    ///
    /// Returns a typed config error when the checkpoint is not ModernBERT,
    /// lacks a classifier head, or has incompatible attention dimensions.
    pub fn preflight_gte_reranker(&self) -> Result<ModernBertPreflight> {
        if self.model_type != "modernbert" {
            return UnsupportedModelTypeSnafu {
                model_type: self.model_type.clone(),
            }
            .fail();
        }

        if self.num_labels == 0 && self.id2label.is_empty() {
            return MissingClassifierHeadSnafu.fail();
        }

        if self.num_attention_heads == 0 {
            return ConfigSnafu {
                message: "num_attention_heads must be nonzero".to_string(),
            }
            .fail();
        }

        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return ConfigSnafu {
                message: format!(
                    "hidden_size {} must be divisible by num_attention_heads {}",
                    self.hidden_size, self.num_attention_heads
                ),
            }
            .fail();
        }

        if self.local_attention == 0 {
            return ConfigSnafu {
                message: "local_attention must be nonzero".to_string(),
            }
            .fail();
        }

        if self.global_attn_every_n_layers == 0 {
            return ConfigSnafu {
                message: "global_attn_every_n_layers must be nonzero".to_string(),
            }
            .fail();
        }

        Ok(ModernBertPreflight {
            hidden_size: self.hidden_size,
            head_dim: self.hidden_size / self.num_attention_heads,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            max_position_embeddings: self.max_position_embeddings,
            local_attention: self.local_attention,
            global_attn_every_n_layers: self.global_attn_every_n_layers,
            classifier_labels: self.num_labels.max(self.id2label.len()),
        })
    }
}

/// Validated GTE ModernBERT shape needed for backend planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModernBertPreflight {
    /// Hidden width.
    pub hidden_size: usize,
    /// Attention head width.
    pub head_dim: usize,
    /// Transformer layer count.
    pub num_hidden_layers: usize,
    /// Attention head count.
    pub num_attention_heads: usize,
    /// Maximum configured sequence length.
    pub max_position_embeddings: usize,
    /// Sliding local attention window.
    pub local_attention: usize,
    /// Global attention layer frequency.
    pub global_attn_every_n_layers: usize,
    /// Classifier label count.
    pub classifier_labels: usize,
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests intentionally unwrap successful fixture calls to keep failure locations precise"
)]
mod tests {
    use super::*;

    #[test]
    fn canonical_config_matches_expected_shape() {
        let cfg = ModernBertConfig::gte_reranker_modernbert_base();
        assert_eq!(cfg.vocab_size, 50_368);
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_hidden_layers, 22);
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.intermediate_size, 1_152);
        assert_eq!(cfg.max_position_embeddings, 8_192);
        assert_eq!(cfg.num_labels, 1);
        assert_eq!(cfg.local_attention, 128);
        assert_eq!(cfg.global_attn_every_n_layers, 3);
        assert_eq!(cfg.pad_token_id, 50_283);
        assert_eq!(cfg.cls_token_id, 50_281);
        assert_eq!(cfg.sep_token_id, 50_282);
        assert!(cfg.decoder_bias);
        assert!((cfg.local_rope_theta - 10_000.0).abs() < f64::EPSILON);
        assert!((cfg.global_rope_theta - 160_000.0).abs() < f64::EPSILON);
        assert!((cfg.layer_norm_eps - 1e-5).abs() < f64::EPSILON);
    }

    #[test]
    fn deserialize_minimal_json() {
        let json = r#"{
            "vocab_size": 100,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "max_position_embeddings": 512
        }"#;
        let cfg: ModernBertConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.vocab_size, 100);
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.num_hidden_layers, 2);
        assert_eq!(cfg.num_attention_heads, 4);
        assert_eq!(cfg.intermediate_size, 128);
        assert_eq!(cfg.max_position_embeddings, 512);
        // Defaults applied for omitted fields.
        assert_eq!(cfg.num_labels, 0);
        assert_eq!(cfg.local_attention, 0);
        assert_eq!(cfg.global_attn_every_n_layers, 3);
        assert!((cfg.local_rope_theta - 10_000.0).abs() < f64::EPSILON);
        assert!((cfg.global_rope_theta - 160_000.0).abs() < f64::EPSILON);
        assert!((cfg.layer_norm_eps - 1e-5).abs() < f64::EPSILON);
        assert_eq!(cfg.hidden_activation, "gelu");
        assert_eq!(cfg.type_vocab_size, 2);
        assert!((cfg.initializer_range - 0.02).abs() < f64::EPSILON);
        assert_eq!(cfg.normalization_type, "layernorm");
    }

    #[test]
    fn modernbert_config_deserializes_from_json() {
        let json = r#"{
            "vocab_size": 100,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "max_position_embeddings": 512
        }"#;
        let cfg: ModernBertConfig = serde_json::from_str(json).unwrap();
        let serialized = serde_json::to_string(&cfg).unwrap();
        let round_tripped: ModernBertConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(round_tripped.vocab_size, 100);
        assert_eq!(round_tripped.hidden_size, 64);
        assert_eq!(round_tripped.num_hidden_layers, 2);
        assert_eq!(round_tripped.num_attention_heads, 4);
        assert_eq!(round_tripped.intermediate_size, 128);
        assert_eq!(round_tripped.max_position_embeddings, 512);
    }

    #[test]
    fn deserialize_published_gte_reranker_config() {
        let json = r#"{
            "_name_or_path": "gte-reranker-modernbert-base",
            "architectures": ["ModernBertForSequenceClassification"],
            "attention_bias": false,
            "attention_dropout": 0.0,
            "bos_token_id": 50281,
            "classifier_activation": "gelu",
            "classifier_bias": false,
            "classifier_dropout": 0.0,
            "classifier_pooling": "mean",
            "cls_token_id": 50281,
            "decoder_bias": true,
            "deterministic_flash_attn": false,
            "embedding_dropout": 0.0,
            "eos_token_id": 50282,
            "vocab_size": 50368,
            "hidden_size": 768,
            "num_hidden_layers": 22,
            "num_attention_heads": 12,
            "intermediate_size": 1152,
            "max_position_embeddings": 8192,
            "local_attention": 128,
            "global_attn_every_n_layers": 3,
            "local_rope_theta": 10000.0,
            "global_rope_theta": 160000.0,
            "gradient_checkpointing": false,
            "hidden_activation": "gelu",
            "id2label": {"0": "LABEL_0"},
            "initializer_cutoff_factor": 2.0,
            "initializer_range": 0.02,
            "label2id": {"LABEL_0": 0},
            "layer_norm_eps": 1e-05,
            "mlp_bias": false,
            "mlp_dropout": 0.0,
            "model_type": "modernbert",
            "norm_bias": false,
            "norm_eps": 1e-05,
            "pad_token_id": 50283,
            "position_embedding_type": "absolute",
            "sep_token_id": 50282,
            "sparse_pred_ignore_index": -100,
            "sparse_prediction": false,
            "torch_dtype": "float32",
            "transformers_version": "4.48.0.dev0"
        }"#;
        let cfg: ModernBertConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.vocab_size, 50_368);
        assert_eq!(cfg.intermediate_size, 1_152);
        assert_eq!(cfg.cls_token_id, 50_281);
        assert_eq!(cfg.pad_token_id, 50_283);
        assert_eq!(cfg.sep_token_id, 50_282);
        assert!(cfg.decoder_bias);
        assert_eq!(cfg.hidden_activation, "gelu");
        assert!((cfg.embedding_dropout - 0.0).abs() < f64::EPSILON);
        assert!((cfg.attention_dropout - 0.0).abs() < f64::EPSILON);
        assert!((cfg.layer_norm_eps - 1e-5).abs() < f64::EPSILON);
    }

    #[test]
    fn deserialize_invalid_json_fails() {
        let json = r#"{"vocab_size": "not_a_number"}"#;
        let result = serde_json::from_str::<ModernBertConfig>(json);
        assert!(result.is_err());
    }

    #[test]
    fn preflight_keeps_attention_and_classifier_shape() {
        let cfg = ModernBertConfig::gte_reranker_modernbert_base();

        let preflight = cfg.preflight_gte_reranker().unwrap();

        assert_eq!(
            preflight,
            ModernBertPreflight {
                hidden_size: 768,
                head_dim: 64,
                num_hidden_layers: 22,
                num_attention_heads: 12,
                max_position_embeddings: 8_192,
                local_attention: 128,
                global_attn_every_n_layers: 3,
                classifier_labels: 1,
            }
        );
    }

    #[test]
    fn preflight_rejects_wrong_model_type() {
        let mut cfg = ModernBertConfig::gte_reranker_modernbert_base();
        cfg.model_type = "bert".into();

        let result = cfg.preflight_gte_reranker();

        assert!(
            matches!(result, Err(crate::error::Error::UnsupportedModelType { model_type, .. }) if model_type == "bert")
        );
    }
}
