//! `StellaModel` — the first production embedding model in logismos.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use encoders::{StellaConfig, StellaEncoder};
use kernels::cpu_f32;
use loader::WeightProvider;
use loader::safetensors::Reader;
use logismos_core::{EmbeddingError, EmbeddingModel, EncodeOpts, Prompt};
use tokenize::Tokenizer;

use crate::error::{Error, Result};

/// Matryoshka output dimensionality for Stella.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StellaDim {
    /// 256-dim head.
    Dim256,
    /// 768-dim head.
    Dim768,
    /// 1024-dim head — logismos + mnemosyne default.
    Dim1024,
    /// 2048-dim head.
    Dim2048,
    /// 4096-dim head.
    Dim4096,
    /// 6144-dim head.
    Dim6144,
    /// 8192-dim head.
    Dim8192,
}

impl StellaDim {
    /// Integer width of the head.
    #[must_use]
    pub(crate) fn width(self) -> usize {
        match self {
            Self::Dim256 => 256,
            Self::Dim768 => 768,
            Self::Dim1024 => 1024,
            Self::Dim2048 => 2048,
            Self::Dim4096 => 4096,
            Self::Dim6144 => 6144,
            Self::Dim8192 => 8192,
        }
    }

    /// All dims exposed on disk. Matches the `2_Dense_{N}` directories.
    #[must_use]
    pub fn all() -> &'static [StellaDim] {
        &[
            Self::Dim256,
            Self::Dim768,
            Self::Dim1024,
            Self::Dim2048,
            Self::Dim4096,
            Self::Dim6144,
            Self::Dim8192,
        ]
    }

    /// Resolve by integer width.
    #[must_use]
    pub fn from_width(width: usize) -> Option<Self> {
        Self::all().iter().copied().find(|d| d.width() == width)
    }

    /// Folder name under the checkpoint root.
    #[must_use]
    pub(crate) fn folder(self) -> &'static str {
        match self {
            Self::Dim256 => "2_Dense_256",
            Self::Dim768 => "2_Dense_768",
            Self::Dim1024 => "2_Dense_1024",
            Self::Dim2048 => "2_Dense_2048",
            Self::Dim4096 => "2_Dense_4096",
            Self::Dim6144 => "2_Dense_6144",
            Self::Dim8192 => "2_Dense_8192",
        }
    }
}

#[derive(Debug, Clone)]
struct DenseHead {
    /// `[dim, hidden]` flat, fp32.
    weight: Vec<f32>,
    /// `[dim]`, fp32.
    bias: Vec<f32>,
    /// Output dim (width).
    dim: usize,
}

/// Fully-assembled Stella embedding model.
pub struct StellaModel {
    cfg: StellaConfig,
    encoder: StellaEncoder,
    tokenizer: Tokenizer,
    heads: HashMap<usize, DenseHead>,
    default_dim: usize,
    supported: Vec<usize>,
    max_tokens: usize,
    prompts: HashMap<String, String>,
    pool: rayon::ThreadPool,
}

impl std::fmt::Debug for StellaModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StellaModel")
            .field("default_dim", &self.default_dim)
            .field("supported", &self.supported)
            .field("max_tokens", &self.max_tokens)
            .field("n_layers", &self.encoder.layers.len())
            .field("heads", &self.heads.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl StellaModel {
    /// Default max-input-token length (sentence_bert_config.json).
    pub(crate) const DEFAULT_MAX_TOKENS: usize = 512;

    /// Load the full model (encoder + tokenizer + dense heads) from a
    /// checkpoint root directory such as `/models/stella-1.5b-v5/`.
    ///
    /// # Errors
    ///
    /// [`Error::Loader`], [`Error::Encoders`], [`Error::Tokenize`] on
    /// any subsystem failure.
    pub fn load(root: &Path, dims: &[StellaDim]) -> Result<Self> {
        let cfg = StellaConfig::stella_1_5b();
        let encoder = StellaEncoder::load(&root.join("model.safetensors"), cfg)?;
        let tokenizer = Tokenizer::from_file(&root.join("tokenizer.json"))?;

        let mut heads: HashMap<usize, DenseHead> = HashMap::new();
        let mut supported: Vec<usize> = Vec::new();
        for &d in dims {
            let head = load_dense_head(root, d)?;
            let w = d.width();
            heads.insert(w, head);
            supported.push(w);
        }
        supported.sort_unstable();

        let default_dim = if heads.contains_key(&1024) {
            1024
        } else {
            // Fall back to the first loaded head.
            *supported
                .first()
                .ok_or_else(|| Error::Io("no heads loaded".into()))?
        };

        // Prompt metadata is optional for the checkpoint; keep model loading
        // usable when that sidecar is absent or unreadable.
        let prompts = match load_prompts(root) {
            Ok(prompts) => prompts,
            Err(Error::Io(_)) => HashMap::new(),
            Err(err) => return Err(err),
        };

        // Pin to physical cores. On most x86_64 hosts this is half of
        // `num_cpus::get()`. We peek at `/proc/cpuinfo` for the most
        // accurate number and fall back to a generous default.
        let n_physical = detect_physical_cores();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_physical)
            .build()
            .map_err(|e| Error::Io(format!("rayon pool: {e}")))?;

        Ok(Self {
            cfg,
            encoder,
            tokenizer,
            heads,
            default_dim,
            supported,
            max_tokens: Self::DEFAULT_MAX_TOKENS,
            prompts,
            pool,
        })
    }

    /// Model config.
    #[must_use]
    pub fn config(&self) -> &StellaConfig {
        &self.cfg
    }

    /// Prompt strings keyed by role name (`"s2s_query"`, `"s2p_query"`).
    #[must_use]
    pub fn prompts(&self) -> &HashMap<String, String> {
        &self.prompts
    }

    /// Internal forward — no prompt, no dim lookup, pure compute.
    /// Returns the L2-normalised vector at the requested dim.
    ///
    /// # Errors
    ///
    /// Propagates encoder / shape failures.
    pub(crate) fn encode_raw(&self, ids: &[u32], mask: &[u8], dim: usize) -> Result<Vec<f32>> {
        let head = self.heads.get(&dim).ok_or(Error::UnsupportedDim(dim))?;

        // Forward through the encoder: [seq, hidden]
        let hidden_states = self.encoder.forward(ids, mask)?;

        // Mean pool
        let pooled = cpu_f32::mean_pool_masked(&hidden_states, mask, ids.len(), self.cfg.hidden);
        // L2-normalise pooled (matches sentence-transformers pipeline).
        let mut pooled_n = pooled;
        cpu_f32::l2_normalize_in_place(&mut pooled_n);

        // Dense head projection (fp32).
        let mut y = cpu_f32::linear_t(
            &pooled_n,
            &head.weight,
            Some(&head.bias),
            1,
            head.dim,
            self.cfg.hidden,
        );
        // Final L2-normalise.
        cpu_f32::l2_normalize_in_place(&mut y);
        Ok(y)
    }
}

impl EmbeddingModel for StellaModel {
    fn default_dim(&self) -> usize {
        self.default_dim
    }

    fn supported_dims(&self) -> &[usize] {
        &self.supported
    }

    fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    fn encode_batch(
        &self,
        texts: &[&str],
        opts: &EncodeOpts,
    ) -> std::result::Result<Vec<Vec<f32>>, EmbeddingError> {
        // Phase-3 batch: process sentences in parallel via rayon. Each
        // worker runs the full forward pass for its sentence. matmul
        // remains single-thread internally (matrixmultiply without
        // threading) so the available cores fan out across sentences,
        // which is the shape that scales best at batch >= cores.
        //
        // We pin to the physical-core count (not logical thread count)
        // the first time `encode_batch` runs — SMT hyperthreads hurt
        // matmul perf by ~2× on this workload. Users that want a
        // different pool size can set `RAYON_NUM_THREADS` in env before
        // process start.
        self.pool.install(|| {
            use rayon::prelude::*;
            texts.par_iter().map(|t| self.encode(t, opts)).collect()
        })
    }

    fn encode(
        &self,
        text: &str,
        opts: &EncodeOpts,
    ) -> std::result::Result<Vec<f32>, EmbeddingError> {
        let dim = opts.dim.unwrap_or(self.default_dim);
        if !self.supported.contains(&dim) {
            return Err(EmbeddingError::UnsupportedDim(dim));
        }
        let max_tokens = opts.max_tokens.unwrap_or(self.max_tokens);

        let text_with_prompt = match opts.prompt.as_ref() {
            Some(Prompt::S2sQuery) => {
                if let Some(p) = self.prompts.get("s2s_query") {
                    format!("{p}{text}")
                } else {
                    text.to_string()
                }
            }
            Some(Prompt::S2pQuery) => {
                if let Some(p) = self.prompts.get("s2p_query") {
                    format!("{p}{text}")
                } else {
                    text.to_string()
                }
            }
            Some(Prompt::Custom(s)) => format!("{s}{text}"),
            None | Some(_) => text.to_string(),
        };

        let ids: Vec<u32> = self
            .tokenizer
            .encode(&text_with_prompt, true)
            .map_err(|e| EmbeddingError::Tokenize(e.to_string()))?;
        if ids.len() > max_tokens {
            return Err(EmbeddingError::InputTooLong {
                got: ids.len(),
                limit: max_tokens,
            });
        }
        let mask = vec![1u8; ids.len()];
        self.encode_raw(&ids, &mask, dim)
            .map_err(|e| EmbeddingError::Compute(e.to_string()))
    }
}

fn load_dense_head(root: &Path, dim: StellaDim) -> Result<DenseHead> {
    let path: PathBuf = root.join(dim.folder()).join("model.safetensors");
    let reader = Reader::open(&path)?;
    let w_view = reader.get("linear.weight")?;
    let b_view = reader.get("linear.bias")?;
    if w_view.dtype != taxis::DType::F32 || b_view.dtype != taxis::DType::F32 {
        return Err(Error::Io(format!(
            "dense head {}: expected F32, got weight={:?}, bias={:?}",
            dim.folder(),
            w_view.dtype,
            b_view.dtype
        )));
    }
    let expected_w = [dim.width(), 1536];
    if w_view.shape != expected_w {
        return Err(Error::Io(format!(
            "dense head {}: expected weight shape {:?}, got {:?}",
            dim.folder(),
            expected_w,
            w_view.shape
        )));
    }
    if b_view.shape != [dim.width()] {
        return Err(Error::Io(format!(
            "dense head {}: expected bias shape {:?}, got {:?}",
            dim.folder(),
            [dim.width()],
            b_view.shape
        )));
    }
    let weight = bytes_to_f32(w_view.bytes);
    let bias = bytes_to_f32(b_view.bytes);
    Ok(DenseHead {
        weight,
        bias,
        dim: dim.width(),
    })
}

/// Detect physical-core count on Linux via `/proc/cpuinfo`. Falls back
/// to `rayon::current_num_threads()` on non-Linux.
fn detect_physical_cores() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/cpuinfo") {
            let mut cores = std::collections::HashSet::new();
            for line in s.lines() {
                if let Some(v) = line.strip_prefix("core id\t: ")
                    && let Ok(id) = v.trim().parse::<u32>()
                {
                    cores.insert(id);
                }
            }
            if !cores.is_empty() {
                return cores.len();
            }
        }
    }
    std::thread::available_parallelism().map_or(4, std::num::NonZero::get)
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let mut b = [0u8; 4];
        b.copy_from_slice(chunk);
        out.push(f32::from_le_bytes(b));
    }
    out
}

/// Minimal ad-hoc JSON parser for the two string values we need
/// (`prompts.s2s_query` and `prompts.s2p_query`). Avoids adding a
/// `serde_json` dependency for a two-field lookup.
///
/// Not robust for general JSON; fine for this checkpoint file whose
/// layout is fixed by sentence-transformers 3.x.
fn load_prompts(root: &Path) -> Result<HashMap<String, String>> {
    let path = root.join("config_sentence_transformers.json");
    let text = std::fs::read_to_string(&path)?;
    let mut out = HashMap::new();
    for key in ["s2s_query", "s2p_query"] {
        if let Some(v) = extract_json_string_field(&text, key) {
            out.insert(key.to_string(), v);
        }
    }
    Ok(out)
}

/// Find `"<key>": "<value>"` in JSON text and return `<value>` with
/// escape decoding for `\n`, `\t`, `\"`, `\\`. Returns None on absence.
fn extract_json_string_field(text: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let kstart = text.find(&needle)?;
    let after_key = text.get(kstart + needle.len()..)?;
    let colon = after_key.find(':')?;
    let rest = after_key.get(colon + 1..)?;
    let qs = rest.find('"')?;
    // Walk until the closing (unescaped) quote.
    let mut out = String::new();
    let mut chars = rest.get(qs + 1..)?.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => match chars.next()? {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                'r' => out.push('\r'),
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                other => out.push(other),
            },
            other => out.push(other),
        }
    }
    None
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "test assertions use expect() directly")]

    use super::*;

    #[test]
    fn extract_json_string_field_happy_path_escapes() {
        let text = r#"{"name": "line1\nline2\ttab\rcr\"quote\\backslash"}"#;
        let got = extract_json_string_field(text, "name").expect("field present");
        assert_eq!(got, "line1\nline2\ttab\rcr\"quote\\backslash");
    }

    #[test]
    fn extract_json_string_field_unicode_escape_is_not_decoded() {
        // WHY(forkwright/logismos#43): locks the current (non-decoding)
        // behaviour as a stated contract rather than an unverified
        // assumption. `\uXXXX` is walked as five literal characters
        // ('u','0','0','e','9'), not decoded to the code point U+00E9.
        let text = "{\"name\": \"caf\\u00e9\"}";
        let got = extract_json_string_field(text, "name").expect("field present");
        assert_eq!(got, "cafu00e9");
    }

    #[test]
    fn extract_json_string_field_missing_key_returns_none() {
        let text = r#"{"other": "value"}"#;
        assert_eq!(extract_json_string_field(text, "name"), None);
    }

    #[test]
    fn extract_json_string_field_unterminated_string_returns_none() {
        let text = "{\"name\": \"no closing quote";
        assert_eq!(extract_json_string_field(text, "name"), None);
    }

    #[test]
    fn extract_json_string_field_duplicate_key_resolves_to_first_match() {
        // Locks the documented "first match wins" behaviour: a later
        // repeated key does not override an earlier one.
        let text = r#"{"name": "first", "other": 1, "name": "second"}"#;
        let got = extract_json_string_field(text, "name").expect("field present");
        assert_eq!(got, "first");
    }

    #[test]
    fn extract_json_string_field_key_text_in_a_value_matches_wrong_field() {
        // Locks the documented "bare substring search" limitation: the
        // needle is a plain `text.find`, with no check that the match is
        // actually in key position. If the target key's quoted text
        // happens to appear as *another* field's value earlier in the
        // document, the parser walks forward from there — past that
        // unrelated field's own colon and quoted value — and returns a
        // different field's value instead of `None`.
        let text = r#"{"note": "name", "real": "wrong_pick"}"#;
        let got = extract_json_string_field(text, "name").expect("false match");
        assert_eq!(got, "wrong_pick");
    }
}
