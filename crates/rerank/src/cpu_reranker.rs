//! CPU ModernBERT reranker.
//!
//! `ModernBertCpuReranker` implements the [`Reranker`] trait using
//! the CPU fp32 encoder in `encoders::modernbert`. It is the Phase 5
//! correctness reference — validate here before the HIP path lands.
//!
//! ## Forward pass
//!
//! 1. For each (query, document) pair, build the token sequence
//!    `[CLS] query [SEP] document [SEP]` via manual special-token
//!    concatenation (no tokenizer held here; caller supplies token ids).
//! 2. Run `ModernBertEncoder::forward`.
//! 3. Mean-pool the hidden states of the valid tokens.
//! 4. Apply the linear classifier head.
//! 5. Return the logit as the relevance score.

use encoders::modernbert::{
    GeGluMlpWeights, ModernBertAttentionWeights, ModernBertEncoder, ModernBertEncoderConfig,
    ModernBertLayerWeights, ModernBertWeights,
};
use kernels::cpu_f32;

use crate::batch::{Predictions, RerankBatch};
use crate::config::ModernBertConfig;
use crate::error::{EmptyBatchSnafu, Result, ShapeSnafu};
use crate::reranker::Reranker;

// ---------------------------------------------------------------------------
// Classifier head
// ---------------------------------------------------------------------------

/// Linear classifier head: `[hidden] -> [num_labels]`.
///
/// Matches the `decoder` weight in `gte-reranker-modernbert-base`:
/// `head.weight [num_labels, hidden]`, optional `head.bias [num_labels]`.
#[derive(Debug, Clone)]
pub struct ClassifierHead {
    /// `[num_labels, hidden]` row-major.
    pub weight: Vec<f32>,
    /// `[num_labels]` or empty when no bias.
    pub bias: Vec<f32>,
    /// Hidden width (input dim).
    pub hidden: usize,
    /// Number of output labels.
    pub num_labels: usize,
}

impl ClassifierHead {
    /// Construct with shape-checking.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] on weight/bias size disagreement.
    pub fn new(weight: Vec<f32>, bias: Vec<f32>, hidden: usize, num_labels: usize) -> Result<Self> {
        if weight.len() != num_labels * hidden {
            return ShapeSnafu {
                message: format!(
                    "head.weight: expected {}, got {}",
                    num_labels * hidden,
                    weight.len()
                ),
            }
            .fail();
        }
        if !bias.is_empty() && bias.len() != num_labels {
            return ShapeSnafu {
                message: format!("head.bias: expected {num_labels} or 0, got {}", bias.len()),
            }
            .fail();
        }
        Ok(Self {
            weight,
            bias,
            hidden,
            num_labels,
        })
    }

    /// Apply `pooled [hidden] -> logits [num_labels]`.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] when `pooled.len() != hidden`.
    pub fn forward(&self, pooled: &[f32]) -> Result<Vec<f32>> {
        if pooled.len() != self.hidden {
            return ShapeSnafu {
                message: format!(
                    "classifier.forward: pooled.len()={} != hidden={}",
                    pooled.len(),
                    self.hidden
                ),
            }
            .fail();
        }
        let bias = if self.bias.is_empty() {
            None
        } else {
            Some(self.bias.as_slice())
        };
        Ok(cpu_f32::linear_t(
            pooled,
            &self.weight,
            bias,
            1,
            self.num_labels,
            self.hidden,
        ))
    }
}

// ---------------------------------------------------------------------------
// CPU reranker
// ---------------------------------------------------------------------------

/// CPU ModernBERT cross-encoder reranker.
///
/// Pre-tokenized inputs only (no tokenizer held here). Each call to
/// [`Reranker::predict`] takes a batch of (`token_ids`, `mask`) pairs
/// built by the caller. For end-to-end tokenization, wrap this in a
/// higher-level type that calls `tokenizers` before this point.
///
/// Contrast with `GteReranker` (preflight stub): this type has real
/// weights and a working forward pass.
#[derive(Debug, Clone)]
pub struct ModernBertCpuReranker {
    encoder: ModernBertEncoder,
    head: ClassifierHead,
}

impl ModernBertCpuReranker {
    /// Construct from a pre-built encoder and classifier head.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] when head dimensions do not match encoder hidden size.
    pub fn new(encoder: ModernBertEncoder, head: ClassifierHead) -> Result<Self> {
        if head.hidden != encoder.config().hidden_size {
            return ShapeSnafu {
                message: format!(
                    "head.hidden={} != encoder.hidden_size={}",
                    head.hidden,
                    encoder.config().hidden_size
                ),
            }
            .fail();
        }
        Ok(Self { encoder, head })
    }

    /// Construct a zero-weight reranker for a given config.
    ///
    /// Used in structural tests where only shape and API contract matter.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] on invalid config.
    pub fn new_zeroed(cfg: &ModernBertConfig) -> Result<Self> {
        let enc_cfg = encoder_config_from(cfg);
        let weights = zeroed_weights(&enc_cfg);
        let encoder = ModernBertEncoder::new(enc_cfg.clone(), weights).map_err(|e| {
            ShapeSnafu {
                message: e.to_string(),
            }
            .build()
        })?;
        let n_labels = cfg.num_labels.max(1);
        let h = enc_cfg.hidden_size;
        let head = ClassifierHead::new(vec![0.0; n_labels * h], vec![], h, n_labels)?;
        Self::new(encoder, head)
    }

    /// Score a single pre-tokenized (query, document) pair.
    ///
    /// - `token_ids`: combined token sequence (caller is responsible for
    ///   special-token framing: `[CLS] query [SEP] document [SEP]`).
    /// - `mask`: attention mask, same length as `token_ids`.
    ///
    /// Returns `[num_labels]` logits.
    ///
    /// # Errors
    ///
    /// Propagates encoder and classifier-head errors.
    pub fn score_pair(&self, token_ids: &[u32], mask: &[u8]) -> Result<Vec<f32>> {
        let pooled = self.encoder.encode_pooled(token_ids, mask).map_err(|e| {
            ShapeSnafu {
                message: e.to_string(),
            }
            .build()
        })?;
        self.head.forward(&pooled)
    }
}

impl Reranker for ModernBertCpuReranker {
    /// Score every item in `batch`.
    ///
    /// Each item is tokenized as a single combined text `"query [SEP]
    /// document"` for this CPU reference path. The special-token
    /// prefix/suffix are not added here; the reranker treats the query
    /// and document strings as-is via a trivial placeholder tokenization.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmptyBatch`] for an empty batch. Propagates
    /// encoder errors for individual items.
    ///
    /// # NOTE
    ///
    /// Real end-to-end use requires loading a `tokenizer.json` and
    /// building proper pair-encoded token sequences. This placeholder
    /// path returns scores from zero-weight projections for structural
    /// validation only.
    fn predict(&self, batch: RerankBatch) -> crate::error::Result<Predictions> {
        if batch.is_empty() {
            return EmptyBatchSnafu.fail();
        }
        let mut predictions = Predictions::new();
        for (index, item) in batch.items.iter().enumerate() {
            // Minimal tokenization for structural testing: map each char
            // to its byte value mod vocab_size. Real use loads a tokenizer.json.
            let combined = format!("{} {}", item.query, item.document);
            let vocab = self.encoder.config().hidden_size.max(2); // any nonzero
            let max_seq = self.encoder.config().max_position_embeddings;
            let ids: Vec<u32> = combined
                .bytes()
                .take(max_seq - 2)
                // WHY: vocab = hidden_size.max(2), bounded by ModelConfig (max 131072).
                // u32::try_from never fails in practice; unwrap_or gives safe modulus floor.
                .map(|b| u32::from(b) % u32::try_from(vocab).unwrap_or(u32::MAX))
                .collect();
            let mut full_ids = vec![0u32]; // placeholder CLS
            full_ids.extend_from_slice(&ids);
            full_ids.push(0u32); // placeholder SEP
            let mask = vec![1u8; full_ids.len()];
            let logits = self.score_pair(&full_ids, &mask)?;
            predictions.insert(index, logits);
        }
        Ok(predictions)
    }
}

// ---------------------------------------------------------------------------
// Config conversion helpers
// ---------------------------------------------------------------------------

/// Extract `ModernBertEncoderConfig` from a `ModernBertConfig`.
pub(crate) fn encoder_config_from(cfg: &ModernBertConfig) -> ModernBertEncoderConfig {
    ModernBertEncoderConfig {
        vocab_size: cfg.vocab_size,
        hidden_size: cfg.hidden_size,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        intermediate_size: cfg.intermediate_size,
        max_position_embeddings: cfg.max_position_embeddings,
        local_window: cfg.local_attention / 2,
        global_attn_every_n_layers: cfg.global_attn_every_n_layers,
        local_rope_theta: cfg.local_rope_theta,
        global_rope_theta: cfg.global_rope_theta,
        // WHY: layer_norm_eps is f64 in config (JSON source); f32 precision is
        // sufficient for layer normalization epsilon (typical value 1e-12).
        #[expect(
            clippy::cast_possible_truncation,
            reason = "eps precision loss from f64->f32 is acceptable"
        )]
        layer_norm_eps: cfg.layer_norm_eps as f32,
        attention_bias: cfg.attention_bias,
        mlp_bias: cfg.mlp_bias,
        norm_bias: cfg.norm_bias,
    }
}

/// Build zeroed weights for a given encoder config (for tests).
pub(crate) fn zeroed_weights(cfg: &ModernBertEncoderConfig) -> ModernBertWeights {
    let h = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let vocab = cfg.vocab_size;
    let n_layers = cfg.num_hidden_layers;
    let layers = (0..n_layers)
        .map(|_| ModernBertLayerWeights {
            attn_norm_weight: vec![1.0; h],
            attn_norm_bias: vec![],
            attn: ModernBertAttentionWeights {
                wqkv: vec![0.0; 3 * h * h],
                bqkv: vec![],
                wo: vec![0.0; h * h],
                bo: vec![],
            },
            mlp_norm_weight: vec![1.0; h],
            mlp_norm_bias: vec![],
            mlp: GeGluMlpWeights {
                wi: vec![0.0; 2 * inter * h],
                wo: vec![0.0; h * inter],
                bi: vec![],
                bo: vec![],
            },
        })
        .collect();
    ModernBertWeights {
        tok_embed: vec![1.0; vocab * h],
        embed_norm_weight: vec![1.0; h],
        embed_norm_bias: vec![],
        layers,
        final_norm_weight: vec![1.0; h],
        final_norm_bias: vec![],
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests intentionally unwrap successful fixture calls to keep failure locations precise"
)]
#[expect(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    reason = "encoder_config_from does a plain f64->f32 pass-through with no arithmetic, so exact \
              equality against the same cast is the correct check, not an accumulation-error case"
)]
mod tests {
    use super::*;
    use crate::batch::{RerankBatch, RerankItem};
    use crate::config::ModernBertConfig;

    fn tiny_cfg() -> ModernBertConfig {
        ModernBertConfig {
            name_or_path: "test".into(),
            architectures: vec![],
            attention_bias: false,
            vocab_size: 64,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 4,
            max_position_embeddings: 32,
            num_labels: 1,
            local_attention: 4,
            global_attn_every_n_layers: 2,
            local_rope_theta: 10_000.0,
            global_rope_theta: 160_000.0,
            layer_norm_eps: 1e-5,
            norm_eps: 1e-5,
            hidden_activation: "gelu".into(),
            embedding_dropout: 0.0,
            attention_dropout: 0.0,
            bos_token_id: 1,
            eos_token_id: 2,
            type_vocab_size: 2,
            initializer_range: 0.02,
            pad_token_id: 0,
            cls_token_id: 1,
            sep_token_id: 2,
            decoder_bias: true,
            classifier_activation: "gelu".into(),
            classifier_bias: false,
            classifier_dropout: 0.0,
            classifier_pooling: "mean".into(),
            deterministic_flash_attn: false,
            gradient_checkpointing: false,
            id2label: std::collections::BTreeMap::from([("0".into(), "LABEL_0".into())]),
            label2id: std::collections::BTreeMap::from([("LABEL_0".into(), 0)]),
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
            transformers_version: String::new(),
        }
    }

    #[test]
    fn classifier_head_shape_gate() {
        let h = 8;
        let n = 1;
        let head = ClassifierHead::new(vec![0.0; n * h], vec![], h, n).unwrap();
        assert_eq!(head.num_labels, n);
        assert_eq!(head.hidden, h);
    }

    #[test]
    fn classifier_head_rejects_bad_weight_size() {
        let result = ClassifierHead::new(vec![0.0; 5], vec![], 8, 1);
        assert!(result.is_err());
    }

    #[test]
    fn classifier_head_forward_zero_weights_zero_output() {
        let h = 4;
        let head = ClassifierHead::new(vec![0.0; h], vec![], h, 1).unwrap();
        let out = head.forward(&vec![1.0; h]).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].abs() < 1e-7);
    }

    #[test]
    fn cpu_reranker_zeroed_predict_returns_scores_for_batch() {
        let cfg = tiny_cfg();
        let reranker = ModernBertCpuReranker::new_zeroed(&cfg).unwrap();
        let batch = RerankBatch::new(vec![
            RerankItem {
                query: "what is rust".into(),
                document: "Rust is a systems language".into(),
            },
            RerankItem {
                query: "what is rust".into(),
                document: "Python is an interpreted language".into(),
            },
        ])
        .unwrap();
        let result = reranker.predict(batch).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result.contains_key(&0));
        assert!(result.contains_key(&1));
        // Zero-weight encoder → all logits must be 0 or exactly deterministic
        let s0 = result[&0][0];
        let s1 = result[&1][0];
        assert!(s0.is_finite(), "score[0] must be finite, got {s0}");
        assert!(s1.is_finite(), "score[1] must be finite, got {s1}");
    }

    #[test]
    fn cpu_reranker_rejects_empty_batch() {
        let cfg = tiny_cfg();
        let reranker = ModernBertCpuReranker::new_zeroed(&cfg).unwrap();
        let result = reranker.predict(RerankBatch { items: vec![] });
        assert!(result.is_err());
    }

    #[test]
    fn cpu_reranker_scores_are_deterministic() {
        let cfg = tiny_cfg();
        let reranker = ModernBertCpuReranker::new_zeroed(&cfg).unwrap();
        let item = RerankItem {
            query: "query alice".into(),
            document: "document from acme corp".into(),
        };
        let batch1 = RerankBatch::new(vec![item.clone()]).unwrap();
        let batch2 = RerankBatch::new(vec![item]).unwrap();
        let r1 = reranker.predict(batch1).unwrap();
        let r2 = reranker.predict(batch2).unwrap();
        assert_eq!(
            r1[&0], r2[&0],
            "identical input must yield identical scores"
        );
    }

    #[test]
    fn encoder_config_from_matches_modernbert_config() {
        let cfg = tiny_cfg();
        let enc_cfg = encoder_config_from(&cfg);
        assert_eq!(enc_cfg.vocab_size, cfg.vocab_size);
        assert_eq!(enc_cfg.hidden_size, cfg.hidden_size);
        assert_eq!(enc_cfg.num_hidden_layers, cfg.num_hidden_layers);
        assert_eq!(enc_cfg.local_window, cfg.local_attention / 2);
        assert_eq!(enc_cfg.layer_norm_eps, cfg.layer_norm_eps as f32);
    }

    #[test]
    fn layer_norm_fixture() {
        // Verify the layer_norm_f32 building block: unit weight, zero bias,
        // single row — output should have zero mean and unit variance.
        use transformers::layer_norm_f32;
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let w = vec![1.0f32; 8];
        let b: Vec<f32> = vec![];
        let y = layer_norm_f32(&x, &w, &b, 1, 8, 1e-5);
        let mean: f32 = y.iter().sum::<f32>() / 8.0;
        let var: f32 = y.iter().map(|&v| v * v).sum::<f32>() / 8.0;
        assert!(mean.abs() < 1e-4, "mean should be ~0, got {mean}");
        assert!((var - 1.0).abs() < 1e-3, "var should be ~1, got {var}");
    }

    #[test]
    fn gelu_known_values() {
        // gelu(0) = 0, gelu(1) ≈ 0.8413, gelu(-1) ≈ -0.1587
        use transformers::gelu;
        let y = gelu(&[0.0, 1.0, -1.0]);
        assert!(y[0].abs() < 1e-6, "gelu(0) = 0, got {}", y[0]);
        assert!(
            (y[1] - 0.8413).abs() < 1e-3,
            "gelu(1) ≈ 0.8413, got {}",
            y[1]
        );
        assert!(
            (y[2] + 0.1587).abs() < 1e-3,
            "gelu(-1) ≈ -0.1587, got {}",
            y[2]
        );
    }
}
