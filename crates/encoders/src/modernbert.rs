//! ModernBERT encoder (CPU fp32 reference).
//!
//! Bidirectional encoder: token embeddings → N transformer layers
//! (alternating local/global attention) → last-hidden-states `[seq, hidden]`.
//!
//! Phase 5 scope: fp32 CPU only. No batch dimension (B=1). Correctness
//! target is 1e-3 parity against HF Transformers `AutoModel`.

use std::path::Path;

use kernels::cpu_f32;
use loader::WeightProvider;
use loader::safetensors::Reader;
use taxis;
use transformers::{
    GeGluMlp, ModernBertAttention, ModernBertAttentionConfig, RopeTable, layer_norm_f32,
};
pub use transformers::{GeGluMlpWeights, ModernBertAttentionWeights};

use crate::error::{Error, Result};

/// Encoder-side configuration extracted from a ModernBERT checkpoint config.
///
/// This is the subset of `rerank::config::ModernBertConfig` that the encoder
/// needs, expressed as plain values so there is no crate-level circular
/// dependency between `encoders` and `rerank`.
#[derive(Debug, Clone)]
pub struct ModernBertEncoderConfig {
    /// `vocab_size`.
    pub vocab_size: usize,
    /// `hidden_size`.
    pub hidden_size: usize,
    /// `num_hidden_layers`.
    pub num_hidden_layers: usize,
    /// `num_attention_heads`.
    pub num_attention_heads: usize,
    /// `intermediate_size`.
    pub intermediate_size: usize,
    /// `max_position_embeddings`.
    pub max_position_embeddings: usize,
    /// Local attention half-window (from `local_attention / 2`).
    pub local_window: usize,
    /// Global attention layer period (`global_attn_every_n_layers`).
    pub global_attn_every_n_layers: usize,
    /// RoPE theta for local attention layers.
    pub local_rope_theta: f64,
    /// RoPE theta for global attention layers.
    pub global_rope_theta: f64,
    /// LayerNorm epsilon.
    pub layer_norm_eps: f32,
    /// Whether attention projections have bias.
    pub attention_bias: bool,
    /// Whether MLP projections have bias.
    pub mlp_bias: bool,
    /// Whether LayerNorm layers have bias.
    pub norm_bias: bool,
}

/// Per-layer weights for a ModernBERT encoder layer.
#[derive(Debug, Clone)]
pub struct ModernBertLayerWeights {
    /// Pre-attention LayerNorm weight — `[hidden]`.
    pub attn_norm_weight: Vec<f32>,
    /// Pre-attention LayerNorm bias — `[hidden]` or empty.
    pub attn_norm_bias: Vec<f32>,
    /// Attention weights.
    pub attn: ModernBertAttentionWeights,
    /// Pre-MLP LayerNorm weight — `[hidden]`.
    pub mlp_norm_weight: Vec<f32>,
    /// Pre-MLP LayerNorm bias — `[hidden]` or empty.
    pub mlp_norm_bias: Vec<f32>,
    /// MLP weights (GeGLU).
    pub mlp: GeGluMlpWeights,
}

/// Full ModernBERT encoder weight bundle.
#[derive(Debug, Clone)]
pub struct ModernBertWeights {
    /// `embeddings.tok_embeddings.weight` — `[vocab_size, hidden]`.
    pub tok_embed: Vec<f32>,
    /// `embeddings.norm.weight` — `[hidden]`.
    pub embed_norm_weight: Vec<f32>,
    /// `embeddings.norm.bias` — `[hidden]` or empty.
    pub embed_norm_bias: Vec<f32>,
    /// Per-layer weights, length `num_hidden_layers`.
    pub layers: Vec<ModernBertLayerWeights>,
    /// `final_norm.weight` — `[hidden]`.
    pub final_norm_weight: Vec<f32>,
    /// `final_norm.bias` — `[hidden]` or empty.
    pub final_norm_bias: Vec<f32>,
}

impl ModernBertWeights {
    /// Load encoder weights from a safetensors archive.
    ///
    /// # Errors
    ///
    /// [`Error::Loader`] for I/O failures; [`Error::Layout`] for name or shape
    /// disagreement.
    pub fn load(path: &Path, cfg: &ModernBertEncoderConfig) -> Result<Self> {
        let reader = Reader::open(path)?;
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let n_layers = cfg.num_hidden_layers;
        let vocab = cfg.vocab_size;

        let tok_embed = read_f32(
            &reader,
            "model.embeddings.tok_embeddings.weight",
            &[vocab, h],
        )?;
        let embed_norm_weight = read_f32(&reader, "model.embeddings.norm.weight", &[h])?;
        let embed_norm_bias = if cfg.norm_bias {
            read_f32(&reader, "model.embeddings.norm.bias", &[h])?
        } else {
            Vec::new()
        };

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let lw = load_layer(
                &reader,
                i,
                h,
                inter,
                cfg.attention_bias,
                cfg.mlp_bias,
                cfg.norm_bias,
            )?;
            layers.push(lw);
        }

        let final_norm_weight = read_f32(&reader, "model.final_norm.weight", &[h])?;
        let final_norm_bias = if cfg.norm_bias {
            read_f32(&reader, "model.final_norm.bias", &[h])?
        } else {
            Vec::new()
        };

        Ok(Self {
            tok_embed,
            embed_norm_weight,
            embed_norm_bias,
            layers,
            final_norm_weight,
            final_norm_bias,
        })
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "per-layer safetensors shape/bias-presence parameters are clearest as explicit arguments"
)]
fn load_layer(
    reader: &Reader,
    i: usize,
    h: usize,
    inter: usize,
    attn_bias: bool,
    mlp_bias: bool,
    norm_bias: bool,
) -> Result<ModernBertLayerWeights> {
    let p = format!("model.layers.{i}");

    let attn_norm_weight = read_f32(reader, &format!("{p}.attn_norm.weight"), &[h])?;
    let attn_norm_bias = if norm_bias {
        read_f32(reader, &format!("{p}.attn_norm.bias"), &[h])?
    } else {
        Vec::new()
    };

    let wqkv = read_f32(reader, &format!("{p}.attn.Wqkv.weight"), &[3 * h, h])?;
    let bqkv = if attn_bias {
        read_f32(reader, &format!("{p}.attn.Wqkv.bias"), &[3 * h])?
    } else {
        Vec::new()
    };
    let wo = read_f32(reader, &format!("{p}.attn.Wo.weight"), &[h, h])?;
    let bo = if attn_bias {
        read_f32(reader, &format!("{p}.attn.Wo.bias"), &[h])?
    } else {
        Vec::new()
    };

    let mlp_norm_weight = read_f32(reader, &format!("{p}.mlp_norm.weight"), &[h])?;
    let mlp_norm_bias = if norm_bias {
        read_f32(reader, &format!("{p}.mlp_norm.bias"), &[h])?
    } else {
        Vec::new()
    };

    let wi = read_f32(reader, &format!("{p}.mlp.Wi.weight"), &[2 * inter, h])?;
    let bi = if mlp_bias {
        read_f32(reader, &format!("{p}.mlp.Wi.bias"), &[2 * inter])?
    } else {
        Vec::new()
    };
    let wo_mlp = read_f32(reader, &format!("{p}.mlp.Wo.weight"), &[h, inter])?;
    let bo_mlp = if mlp_bias {
        read_f32(reader, &format!("{p}.mlp.Wo.bias"), &[h])?
    } else {
        Vec::new()
    };

    Ok(ModernBertLayerWeights {
        attn_norm_weight,
        attn_norm_bias,
        attn: ModernBertAttentionWeights { wqkv, bqkv, wo, bo },
        mlp_norm_weight,
        mlp_norm_bias,
        mlp: GeGluMlpWeights {
            wi,
            wo: wo_mlp,
            bi,
            bo: bo_mlp,
        },
    })
}

fn read_f32(reader: &Reader, name: &str, expected_shape: &[usize]) -> Result<Vec<f32>> {
    let view = reader.get(name)?;
    if view.dtype != taxis::DType::F32 {
        return Err(Error::Layout(format!(
            "{name}: expected F32, got {:?}",
            view.dtype
        )));
    }
    if view.shape != expected_shape {
        return Err(Error::Layout(format!(
            "{name}: expected shape {:?}, got {:?}",
            expected_shape, view.shape
        )));
    }
    let mut out = Vec::with_capacity(view.bytes.len() / 4);
    for chunk in view.bytes.chunks_exact(4) {
        let mut b = [0u8; 4];
        b.copy_from_slice(chunk);
        out.push(f32::from_le_bytes(b));
    }
    Ok(out)
}

/// CPU ModernBERT encoder.
///
/// Owns the weights and configuration. `forward()` runs the full encoder
/// forward pass over a single sequence.
#[derive(Debug, Clone)]
pub struct ModernBertEncoder {
    cfg: ModernBertEncoderConfig,
    weights: ModernBertWeights,
    /// Two RoPE tables: index 0 = local theta, index 1 = global theta.
    rope: [RopeTable; 2],
    /// Attention blocks, one per layer.
    attn_blocks: Vec<ModernBertAttention>,
    /// MLP blocks, one per layer.
    mlp_blocks: Vec<GeGluMlp>,
}

impl ModernBertEncoder {
    /// Construct from pre-loaded weights and config.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] when any weight shape contradicts the config.
    pub fn new(cfg: ModernBertEncoderConfig, weights: ModernBertWeights) -> Result<Self> {
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        if n_heads == 0 || !h.is_multiple_of(n_heads) {
            return Err(Error::Shape(format!(
                "hidden_size {h} not divisible by num_attention_heads {n_heads}"
            )));
        }
        let head_dim = h / n_heads;
        let global_every = cfg.global_attn_every_n_layers;

        let rope_local =
            RopeTable::new(cfg.max_position_embeddings, head_dim, cfg.local_rope_theta);
        let rope_global =
            RopeTable::new(cfg.max_position_embeddings, head_dim, cfg.global_rope_theta);

        let mut attn_blocks = Vec::with_capacity(cfg.num_hidden_layers);
        let mut mlp_blocks = Vec::with_capacity(cfg.num_hidden_layers);

        for (i, lw) in weights.layers.iter().enumerate() {
            let is_global = (i + 1) % global_every == 0;
            let acfg = ModernBertAttentionConfig {
                hidden: h,
                n_heads,
                head_dim,
                local_window: cfg.local_window,
                is_global,
                attention_bias: cfg.attention_bias,
            };
            let attn = ModernBertAttention::new(acfg, lw.attn.clone())
                .map_err(|e| Error::Shape(e.to_string()))?;
            attn_blocks.push(attn);

            let mlp = GeGluMlp::new(h, cfg.intermediate_size, lw.mlp.clone())
                .map_err(|e| Error::Shape(e.to_string()))?;
            mlp_blocks.push(mlp);
        }

        Ok(Self {
            cfg,
            weights,
            rope: [rope_local, rope_global],
            attn_blocks,
            mlp_blocks,
        })
    }

    /// Load from a safetensors file on disk.
    ///
    /// # Errors
    ///
    /// Propagates loader or shape errors.
    pub fn load(path: &Path, cfg: ModernBertEncoderConfig) -> Result<Self> {
        let weights = ModernBertWeights::load(path, &cfg)?;
        Self::new(cfg, weights)
    }

    /// Run the encoder forward pass.
    ///
    /// - `token_ids`: `[seq]` u32 token identifiers.
    /// - `mask`: `[seq]` attention mask (1=valid, 0=padding).
    ///
    /// Returns `[seq, hidden]` last-hidden-states (pre-pooling).
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] when `token_ids` is empty or `mask.len() != seq`.
    pub fn forward(&self, token_ids: &[u32], mask: &[u8]) -> Result<Vec<f32>> {
        let seq = token_ids.len();
        if seq == 0 {
            return Err(Error::Shape("encoder.forward: empty input".into()));
        }
        if mask.len() != seq {
            return Err(Error::Shape(format!(
                "encoder.forward: mask.len()={} != seq={}",
                mask.len(),
                seq
            )));
        }
        let h = self.cfg.hidden_size;
        let n_layers = self.cfg.num_hidden_layers;
        let global_every = self.cfg.global_attn_every_n_layers;
        let norm_eps = self.cfg.layer_norm_eps;
        let positions: Vec<usize> = (0..seq).collect();

        // Token embeddings lookup: [seq, h]
        let mut hidden =
            cpu_f32::embed_lookup(&self.weights.tok_embed, h, self.cfg.vocab_size, token_ids);

        // Embedding LayerNorm
        hidden = layer_norm_f32(
            &hidden,
            &self.weights.embed_norm_weight,
            &self.weights.embed_norm_bias,
            seq,
            h,
            norm_eps,
        );

        // Transformer layers
        for i in 0..n_layers {
            // Pre-attention norm + attention + residual
            let normed_attn = layer_norm_f32(
                &hidden,
                &self.weights.layers[i].attn_norm_weight,
                &self.weights.layers[i].attn_norm_bias,
                seq,
                h,
                norm_eps,
            );
            let is_global = (i + 1) % global_every == 0;
            let rope = if is_global {
                &self.rope[1]
            } else {
                &self.rope[0]
            };
            let attn_out = self.attn_blocks[i]
                .forward(&normed_attn, rope, &positions, mask)
                .map_err(|e| Error::Shape(e.to_string()))?;
            for (xv, av) in hidden.iter_mut().zip(attn_out.iter()) {
                *xv += av;
            }

            // Pre-MLP norm + MLP + residual
            let normed_mlp = layer_norm_f32(
                &hidden,
                &self.weights.layers[i].mlp_norm_weight,
                &self.weights.layers[i].mlp_norm_bias,
                seq,
                h,
                norm_eps,
            );
            let mlp_out = self.mlp_blocks[i]
                .forward(&normed_mlp)
                .map_err(|e| Error::Shape(e.to_string()))?;
            for (xv, mv) in hidden.iter_mut().zip(mlp_out.iter()) {
                *xv += mv;
            }
        }

        // Final LayerNorm
        hidden = layer_norm_f32(
            &hidden,
            &self.weights.final_norm_weight,
            &self.weights.final_norm_bias,
            seq,
            h,
            norm_eps,
        );

        Ok(hidden)
    }

    /// Run the encoder and mean-pool over valid (unmasked) tokens.
    ///
    /// Returns `[hidden]` pooled representation.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::forward`] errors.
    pub fn encode_pooled(&self, token_ids: &[u32], mask: &[u8]) -> Result<Vec<f32>> {
        let hidden = self.forward(token_ids, mask)?;
        let seq = token_ids.len();
        Ok(cpu_f32::mean_pool_masked(
            &hidden,
            mask,
            seq,
            self.cfg.hidden_size,
        ))
    }

    /// Returns a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &ModernBertEncoderConfig {
        &self.cfg
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests intentionally unwrap successful fixture calls to keep failure locations precise"
)]
mod tests {
    use super::*;

    fn tiny_cfg() -> ModernBertEncoderConfig {
        ModernBertEncoderConfig {
            vocab_size: 16,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 4,
            max_position_embeddings: 32,
            local_window: 2,
            global_attn_every_n_layers: 2,
            local_rope_theta: 10_000.0,
            global_rope_theta: 160_000.0,
            layer_norm_eps: 1e-5,
            attention_bias: false,
            mlp_bias: false,
            norm_bias: false,
        }
    }

    fn zero_weights(cfg: &ModernBertEncoderConfig) -> ModernBertWeights {
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

    #[test]
    fn encoder_forward_output_shape() {
        let cfg = tiny_cfg();
        let weights = zero_weights(&cfg);
        let encoder = ModernBertEncoder::new(cfg.clone(), weights).unwrap();
        let ids = vec![1u32, 5, 2];
        let mask = vec![1u8; 3];
        let out = encoder.forward(&ids, &mask).unwrap();
        assert_eq!(out.len(), 3 * cfg.hidden_size);
    }

    #[test]
    fn encoder_pooled_output_shape() {
        let cfg = tiny_cfg();
        let weights = zero_weights(&cfg);
        let encoder = ModernBertEncoder::new(cfg.clone(), weights).unwrap();
        let ids = vec![1u32, 5, 2];
        let mask = vec![1u8; 3];
        let out = encoder.encode_pooled(&ids, &mask).unwrap();
        assert_eq!(out.len(), cfg.hidden_size);
    }

    #[test]
    fn encoder_rejects_empty_input() {
        let cfg = tiny_cfg();
        let weights = zero_weights(&cfg);
        let encoder = ModernBertEncoder::new(cfg, weights).unwrap();
        assert!(encoder.forward(&[], &[]).is_err());
    }

    #[test]
    fn encoder_rejects_mask_length_mismatch() {
        let cfg = tiny_cfg();
        let weights = zero_weights(&cfg);
        let encoder = ModernBertEncoder::new(cfg, weights).unwrap();
        assert!(encoder.forward(&[1u32, 2], &[1u8]).is_err());
    }

    #[test]
    fn encoder_output_is_finite() {
        let cfg = tiny_cfg();
        let weights = zero_weights(&cfg);
        let encoder = ModernBertEncoder::new(cfg, weights).unwrap();
        let ids = vec![1u32, 5, 2];
        let mask = vec![1u8; 3];
        let out = encoder.forward(&ids, &mask).unwrap();
        assert!(
            out.iter().all(|v| v.is_finite()),
            "all output values must be finite"
        );
    }
}
