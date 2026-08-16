//! Stella 1.5B v5 encoder.
//!
//! Qwen2 decoder architecture run in encoder mode: bidirectional
//! attention (no causal mask), no KV cache, mean-pooled over
//! attention-masked tokens in the downstream `embed` crate.
//!
//! Phase 3 scope: fp32 CPU reference implementation. The 1e-3 parity
//! gate is against a PyTorch reference.

use std::path::Path;

use loader::WeightProvider;
use loader::safetensors::Reader;
use transformers::{
    QwenAttention, QwenAttentionConfig, QwenAttentionWeights, RopeTable, SwiGluMlp,
    SwiGluMlpWeights, rms_norm_f32,
};

use crate::error::{Error, Result};

/// Tensors [`load_layer`] reads per transformer layer, feeding the
/// checkpoint-completeness guard in [`StellaWeights::load`]:
/// `input_layernorm.weight`, `post_attention_layernorm.weight`,
/// `self_attn.{q,k,v}_proj.{weight,bias}` (6), `self_attn.o_proj.weight`
/// (no bias), `mlp.{gate,up,down}_proj.weight` (3). A Stella variant with
/// a different per-layer tensor count must update this alongside
/// `load_layer` — the checkpoint-completeness tests below exercise both
/// together against a real fixture, so a drift between this count and
/// what `load_layer` actually reads fails there.
const TENSORS_PER_LAYER: usize = 12;

/// Stella configuration, sourced from `config.json`.
#[derive(Debug, Clone, Copy)]
pub struct StellaConfig {
    /// `vocab_size`.
    pub vocab_size: usize,
    /// `hidden_size` (d_model).
    pub hidden: usize,
    /// `intermediate_size`.
    pub intermediate: usize,
    /// `num_hidden_layers`.
    pub n_layers: usize,
    /// `num_attention_heads`.
    pub n_heads: usize,
    /// `num_key_value_heads`.
    pub n_kv_heads: usize,
    /// `rope_theta`.
    pub rope_theta: f64,
    /// `rms_norm_eps`.
    pub rms_eps: f32,
    /// `max_position_embeddings` (we pre-build RoPE up to this or a cap).
    pub max_pos: usize,
}

impl StellaConfig {
    /// Canonical Stella 1.5B v5 config. Matches `/models/stella-1.5b-v5/config.json`.
    #[must_use]
    pub fn stella_1_5b() -> Self {
        Self {
            vocab_size: 151_646,
            hidden: 1536,
            intermediate: 8960,
            n_layers: 28,
            n_heads: 12,
            n_kv_heads: 2,
            rope_theta: 1_000_000.0,
            rms_eps: 1e-6,
            // cap at 32768 for memory; Phase-3 usage tops out at 512.
            max_pos: 32_768,
        }
    }

    /// Per-head dimension.
    #[must_use]
    pub(crate) fn head_dim(&self) -> usize {
        self.hidden / self.n_heads
    }
}

/// Per-layer weights (unnamed, ordered).
#[derive(Debug, Clone)]
pub struct StellaLayerWeights {
    /// Pre-attention RMSNorm gain.
    pub norm1: Vec<f32>,
    /// Post-attention, pre-MLP RMSNorm gain.
    pub norm2: Vec<f32>,
    /// Attention weights.
    pub attn: QwenAttentionWeights,
    /// MLP weights (Qwen2 = SwiGLU).
    pub mlp: SwiGluMlpWeights,
}

/// Full Stella weight bundle.
#[derive(Debug, Clone)]
pub struct StellaWeights {
    /// `embed_tokens.weight` — `[vocab_size, hidden]`.
    pub tok_embed: Vec<f32>,
    /// Final `model.norm.weight` — `[hidden]`.
    pub final_norm: Vec<f32>,
    /// Per-layer weights, length `n_layers`.
    pub layers: Vec<StellaLayerWeights>,
}

impl StellaWeights {
    /// Load Stella weights from a safetensors archive on disk.
    ///
    /// The loader accounts every tensor it reads: unknown, missing, or
    /// shape-mismatched tensors return [`Error::Layout`].
    ///
    /// # Errors
    ///
    /// [`Error::Loader`] for I/O or parsing failure; [`Error::Layout`]
    /// for name / shape disagreement.
    pub fn load(path: &Path, cfg: &StellaConfig) -> Result<Self> {
        let reader = Reader::open(path)?;

        let mut consumed = std::collections::HashSet::<String>::new();
        let expected_global = ["model.embed_tokens.weight", "model.norm.weight"];

        let tok_embed = read_f32(
            &reader,
            "model.embed_tokens.weight",
            &[cfg.vocab_size, cfg.hidden],
        )?;
        consumed.insert("model.embed_tokens.weight".into());

        let final_norm = read_f32(&reader, "model.norm.weight", &[cfg.hidden])?;
        consumed.insert("model.norm.weight".into());

        let kv_width = cfg.n_kv_heads * cfg.head_dim();
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let lw = load_layer(&reader, cfg, i, kv_width, &mut consumed)?;
            layers.push(lw);
        }

        // PROOF-BRANCH: this is the pre-fix guard ordering (checked
        // AFTER every per-tensor read), deliberately restored here to
        // demonstrate the new tests fail against it. Not for merge.
        let have: Vec<String> = reader.names();
        let total_expected = expected_global.len() + cfg.n_layers * TENSORS_PER_LAYER;
        if have.len() != total_expected {
            return Err(Error::Layout(format!(
                "expected {} tensors in stella archive, got {}",
                total_expected,
                have.len()
            )));
        }
        for name in &have {
            if !consumed.contains(name) {
                return Err(Error::Layout(format!(
                    "unconsumed tensor in stella archive: {name}"
                )));
            }
        }

        Ok(Self {
            tok_embed,
            final_norm,
            layers,
        })
    }
}

/// One Stella encoder layer: owns its attention and MLP.
#[derive(Debug, Clone)]
pub struct StellaLayer {
    /// Pre-attention norm gain.
    pub norm1: Vec<f32>,
    /// Pre-MLP norm gain.
    pub norm2: Vec<f32>,
    /// Attention block.
    pub attn: QwenAttention,
    /// MLP block.
    pub mlp: SwiGluMlp,
}

impl StellaLayer {
    /// Run this layer over `x` (`[seq, hidden]`): pre-attention RMSNorm →
    /// attention → residual, then pre-MLP RMSNorm → MLP → residual.
    /// Updates `x` in place.
    ///
    /// # Errors
    ///
    /// Propagates attention / MLP shape errors.
    fn forward(&self, x: &mut [f32], mask: &[u8], rope: &RopeTable, rms_eps: f32) -> Result<()> {
        let hidden = self.norm1.len();
        let seq = mask.len();
        let norm = rms_norm_f32(x, &self.norm1, seq, hidden, rms_eps)?;
        let attn_out = self.attn.forward(&norm, mask, rope)?;
        for (xi, ai) in x.iter_mut().zip(attn_out.iter()) {
            *xi += ai;
        }

        let norm = rms_norm_f32(x, &self.norm2, seq, hidden, rms_eps)?;
        let mlp_out = self.mlp.forward(&norm)?;
        for (xi, mi) in x.iter_mut().zip(mlp_out.iter()) {
            *xi += mi;
        }

        Ok(())
    }
}

/// Stella encoder — embedding lookup → 28 layers → final norm.
#[derive(Debug, Clone)]
pub struct StellaEncoder {
    /// Config.
    pub cfg: StellaConfig,
    /// Token embedding matrix, `[vocab_size, hidden]` fp32 flat.
    pub tok_embed: Vec<f32>,
    /// Final RMSNorm gain.
    pub final_norm: Vec<f32>,
    /// Layer stack.
    pub layers: Vec<StellaLayer>,
    /// Shared RoPE table.
    pub rope: RopeTable,
}

impl StellaEncoder {
    /// Assemble an encoder from loaded weights.
    ///
    /// # Errors
    ///
    /// Propagates [`transformers`]-level shape checks.
    pub(crate) fn from_weights(cfg: StellaConfig, weights: StellaWeights) -> Result<Self> {
        let attn_cfg = QwenAttentionConfig {
            hidden: cfg.hidden,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim(),
        };
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for lw in weights.layers {
            let attn = QwenAttention::new(attn_cfg, lw.attn)?;
            let mlp = SwiGluMlp::new(cfg.hidden, cfg.intermediate, lw.mlp)?;
            layers.push(StellaLayer {
                norm1: lw.norm1,
                norm2: lw.norm2,
                attn,
                mlp,
            });
        }
        let rope = RopeTable::new(cfg.max_pos, cfg.head_dim(), cfg.rope_theta);
        Ok(Self {
            cfg,
            tok_embed: weights.tok_embed,
            final_norm: weights.final_norm,
            layers,
            rope,
        })
    }

    /// Load + assemble in one call.
    ///
    /// # Errors
    ///
    /// Propagates load / shape failures.
    pub fn load(path: &Path, cfg: StellaConfig) -> Result<Self> {
        let w = StellaWeights::load(path, &cfg)?;
        Self::from_weights(cfg, w)
    }

    /// Forward pass over a single sentence.
    ///
    /// - `ids`: `[seq]` token ids.
    /// - `mask`: `[seq]` attention mask (1 = real, 0 = pad).
    ///
    /// Returns `[seq, hidden]` fp32 last-hidden-states.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] on mismatched input lengths.
    pub fn forward(&self, ids: &[u32], mask: &[u8]) -> Result<Vec<f32>> {
        if ids.len() != mask.len() {
            return Err(Error::Shape(format!(
                "forward: ids.len()={} != mask.len()={}",
                ids.len(),
                mask.len()
            )));
        }
        let seq = ids.len();
        if seq > self.cfg.max_pos {
            return Err(Error::Shape(format!(
                "forward: seq {} > max_pos {}",
                seq, self.cfg.max_pos
            )));
        }

        // 1. Embedding lookup.
        let mut x = kernels::cpu_f32::embed_lookup(
            &self.tok_embed,
            self.cfg.hidden,
            self.cfg.vocab_size,
            ids,
        );

        // 2. 28 transformer layers.
        for layer in &self.layers {
            layer.forward(&mut x, mask, &self.rope, self.cfg.rms_eps)?;
        }

        // 3. Final RMSNorm.
        let y = rms_norm_f32(&x, &self.final_norm, seq, self.cfg.hidden, self.cfg.rms_eps)?;
        Ok(y)
    }
}

// -----------------------------------------------------------------------------
// weight-name mapping + reader helpers
// -----------------------------------------------------------------------------

fn load_layer(
    r: &Reader,
    cfg: &StellaConfig,
    i: usize,
    kv_width: usize,
    consumed: &mut std::collections::HashSet<String>,
) -> Result<StellaLayerWeights> {
    let norm1 = layer_weight(r, i, "input_layernorm.weight", &[cfg.hidden], consumed)?;
    let norm2 = layer_weight(
        r,
        i,
        "post_attention_layernorm.weight",
        &[cfg.hidden],
        consumed,
    )?;

    let wq = layer_weight(
        r,
        i,
        "self_attn.q_proj.weight",
        &[cfg.hidden, cfg.hidden],
        consumed,
    )?;
    let bq = layer_weight(r, i, "self_attn.q_proj.bias", &[cfg.hidden], consumed)?;
    let wk = layer_weight(
        r,
        i,
        "self_attn.k_proj.weight",
        &[kv_width, cfg.hidden],
        consumed,
    )?;
    let bk = layer_weight(r, i, "self_attn.k_proj.bias", &[kv_width], consumed)?;
    let wv = layer_weight(
        r,
        i,
        "self_attn.v_proj.weight",
        &[kv_width, cfg.hidden],
        consumed,
    )?;
    let bv = layer_weight(r, i, "self_attn.v_proj.bias", &[kv_width], consumed)?;
    let wo = layer_weight(
        r,
        i,
        "self_attn.o_proj.weight",
        &[cfg.hidden, cfg.hidden],
        consumed,
    )?;

    let w_gate = layer_weight(
        r,
        i,
        "mlp.gate_proj.weight",
        &[cfg.intermediate, cfg.hidden],
        consumed,
    )?;
    let w_up = layer_weight(
        r,
        i,
        "mlp.up_proj.weight",
        &[cfg.intermediate, cfg.hidden],
        consumed,
    )?;
    let w_down = layer_weight(
        r,
        i,
        "mlp.down_proj.weight",
        &[cfg.hidden, cfg.intermediate],
        consumed,
    )?;

    Ok(StellaLayerWeights {
        norm1,
        norm2,
        attn: QwenAttentionWeights {
            wq,
            bq,
            wk,
            bk,
            wv,
            bv,
            wo,
        },
        mlp: SwiGluMlpWeights {
            w_gate,
            w_up,
            w_down,
        },
    })
}

fn layer_weight(
    r: &Reader,
    i: usize,
    suffix: &str,
    expected_shape: &[usize],
    consumed: &mut std::collections::HashSet<String>,
) -> Result<Vec<f32>> {
    let name = format!("model.layers.{i}.{suffix}");
    let v = read_f32(r, &name, expected_shape)?;
    consumed.insert(name);
    Ok(v)
}

fn read_f32(r: &Reader, name: &str, expected_shape: &[usize]) -> Result<Vec<f32>> {
    let view = r.get(name)?;
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap as StdMap;
    use std::path::PathBuf;

    use safetensors::serialize_to_file;
    use safetensors::tensor::{Dtype as UpstreamDtype, TensorView as UpstreamView};

    use super::*;

    /// Tiny Stella-shaped config: same field structure as
    /// [`StellaConfig::stella_1_5b`], scaled down so the fixture below
    /// stays trivial.
    fn tiny_cfg() -> StellaConfig {
        StellaConfig {
            vocab_size: 4,
            hidden: 2,
            intermediate: 2,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            rope_theta: 10_000.0,
            rms_eps: 1e-6,
            max_pos: 8,
        }
    }

    /// Every tensor `StellaWeights::load` expects for `cfg`, as
    /// `(name, shape)` pairs in `load_layer`'s read order. A
    /// hand-written second copy of that read order, deliberately: the
    /// tests below run `StellaWeights::load` against a real fixture
    /// built from this list, so a drift between `load_layer`'s actual
    /// reads and `TENSORS_PER_LAYER` surfaces as a real test failure
    /// rather than staying invisible.
    fn expected_tensor_shapes(cfg: &StellaConfig) -> Vec<(String, Vec<usize>)> {
        let h = cfg.hidden;
        let kv = cfg.n_kv_heads * cfg.head_dim();
        let inter = cfg.intermediate;
        let mut shapes = vec![
            (
                "model.embed_tokens.weight".to_string(),
                vec![cfg.vocab_size, h],
            ),
            ("model.norm.weight".to_string(), vec![h]),
        ];
        for i in 0..cfg.n_layers {
            let p = format!("model.layers.{i}");
            shapes.extend([
                (format!("{p}.input_layernorm.weight"), vec![h]),
                (format!("{p}.post_attention_layernorm.weight"), vec![h]),
                (format!("{p}.self_attn.q_proj.weight"), vec![h, h]),
                (format!("{p}.self_attn.q_proj.bias"), vec![h]),
                (format!("{p}.self_attn.k_proj.weight"), vec![kv, h]),
                (format!("{p}.self_attn.k_proj.bias"), vec![kv]),
                (format!("{p}.self_attn.v_proj.weight"), vec![kv, h]),
                (format!("{p}.self_attn.v_proj.bias"), vec![kv]),
                (format!("{p}.self_attn.o_proj.weight"), vec![h, h]),
                (format!("{p}.mlp.gate_proj.weight"), vec![inter, h]),
                (format!("{p}.mlp.up_proj.weight"), vec![inter, h]),
                (format!("{p}.mlp.down_proj.weight"), vec![h, inter]),
            ]);
        }
        assert_eq!(
            shapes.len(),
            2 + cfg.n_layers * TENSORS_PER_LAYER,
            "fixture tensor list drifted from TENSORS_PER_LAYER"
        );
        shapes
    }

    /// Writes a safetensors fixture at `path` with one all-`1.0` tensor
    /// per `(name, shape)` entry.
    fn write_fixture(
        path: &Path,
        shapes: &[(String, Vec<usize>)],
    ) -> std::result::Result<(), String> {
        let bufs: Vec<Vec<u8>> = shapes
            .iter()
            .map(|(_, shape)| {
                let n: usize = shape.iter().product();
                (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect()
            })
            .collect();
        let mut tensors: StdMap<String, UpstreamView<'_>> = StdMap::new();
        for ((name, shape), buf) in shapes.iter().zip(bufs.iter()) {
            let tv = UpstreamView::new(UpstreamDtype::F32, shape.clone(), buf)
                .map_err(|e| format!("{name}: {e}"))?;
            tensors.insert(name.clone(), tv);
        }
        serialize_to_file(&tensors, None, path).map_err(|e| format!("serialize_to_file: {e}"))
    }

    /// A per-test-unique path on the shared process temp dir (`tag` +
    /// pid disambiguate concurrent test processes).
    fn fixture_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "logismos-stella-load-test-{tag}-{}.safetensors",
            std::process::id()
        ))
    }

    #[test]
    fn load_accepts_a_complete_checkpoint() -> std::result::Result<(), String> {
        let cfg = tiny_cfg();
        let shapes = expected_tensor_shapes(&cfg);
        let path = fixture_path("complete");
        write_fixture(&path, &shapes)?;

        let result = StellaWeights::load(&path, &cfg);
        let _ = std::fs::remove_file(&path);
        let Ok(weights) = result else {
            return Err(format!(
                "a checkpoint with exactly the expected tensors must load, got {result:?}"
            ));
        };
        assert_eq!(weights.layers.len(), cfg.n_layers);
        assert_eq!(weights.tok_embed.len(), cfg.vocab_size * cfg.hidden);
        Ok(())
    }

    #[test]
    fn load_rejects_checkpoint_with_too_few_tensors() -> std::result::Result<(), String> {
        let cfg = tiny_cfg();
        let mut shapes = expected_tensor_shapes(&cfg);
        // WHY: drop one required tensor so the archive is short by
        // exactly one entry. `load` checks the archive's raw tensor
        // count against `total_expected` before reading any tensor by
        // name, so this exercises the count guard's too-few branch
        // directly rather than a per-name lookup miss.
        shapes.pop();
        let path = fixture_path("too-few");
        write_fixture(&path, &shapes)?;

        let result = StellaWeights::load(&path, &cfg);
        let _ = std::fs::remove_file(&path);
        assert!(
            matches!(&result, Err(Error::Layout(_))),
            "a checkpoint short of the expected tensor count must be rejected by the count guard as a layout error, got {result:?}"
        );
        Ok(())
    }

    #[test]
    fn load_rejects_checkpoint_with_unconsumed_extra_tensor() -> std::result::Result<(), String> {
        let cfg = tiny_cfg();
        let mut shapes = expected_tensor_shapes(&cfg);
        shapes.push((
            "model.layers.0.extra_unexpected_tensor".to_string(),
            vec![cfg.hidden],
        ));
        let path = fixture_path("extra");
        write_fixture(&path, &shapes)?;

        let result = StellaWeights::load(&path, &cfg);
        let _ = std::fs::remove_file(&path);
        assert!(
            matches!(&result, Err(Error::Layout(_))),
            "a checkpoint with an unconsumed extra tensor must be rejected as a layout error, got {result:?}"
        );
        Ok(())
    }
}
