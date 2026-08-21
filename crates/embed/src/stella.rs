//! `StellaModel` — the first production embedding model in logismos.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use encoders::{StellaConfig, StellaEncoder};
use kernels::cpu_f32;
use loader::WeightProvider;
use loader::safetensors::Reader;
use logismos_core::{
    ComputeSnafu as CoreComputeSnafu, EmbeddingError, EmbeddingModel, EncodeOpts,
    InputTooLongSnafu as CoreInputTooLongSnafu, Prompt, TokenizeSnafu as CoreTokenizeSnafu,
    UnsupportedDimSnafu as CoreUnsupportedDimSnafu,
    UnsupportedPromptSnafu as CoreUnsupportedPromptSnafu,
};
use tokenize::Tokenizer;

use crate::error::{IoSnafu, Result, UnsupportedDimSnafu};

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
    /// any subsystem failure; [`Error::Io`] on a dense-head shape
    /// mismatch, a non-absent prompt-sidecar IO fault, or a rayon
    /// pool-build failure.
    pub fn load(root: &Path, dims: &[StellaDim]) -> Result<Self> {
        let cfg = StellaConfig::stella_1_5b();
        let encoder = StellaEncoder::load(&root.join("model.safetensors"), cfg)?;
        let tokenizer = Tokenizer::from_file(&root.join("tokenizer.json"))?;

        let mut heads: HashMap<usize, DenseHead> = HashMap::new();
        let mut supported: Vec<usize> = Vec::new();
        for &d in dims {
            let head = load_dense_head(root, d, cfg.hidden)?;
            let w = d.width();
            heads.insert(w, head);
            supported.push(w);
        }
        supported.sort_unstable();

        let default_dim = if heads.contains_key(&1024) {
            1024
        } else {
            // Fall back to the first loaded head.
            *supported.first().ok_or_else(|| {
                IoSnafu {
                    message: "no heads loaded",
                }
                .build()
            })?
        };

        // Prompt metadata is optional for the checkpoint: an absent
        // sidecar falls back to no prompts. `load_prompts` itself draws
        // that line at `io::ErrorKind::NotFound` only — a permissions
        // fault or a truncated read is not "absent" and propagates.
        //
        // WARNING: this bare `?` is the forkwright/logismos#52 fix (a
        // prior match here swallowed every IO fault). No unit test
        // guards this line — see the doc comment on
        // `load_prompts_propagates_non_notfound_io_fault`.
        let prompts = load_prompts(root)?;

        // Pin to physical cores. On most x86_64 hosts this is half of
        // `num_cpus::get()`. We peek at `/proc/cpuinfo` for the most
        // accurate number and fall back to a generous default.
        let n_physical = detect_physical_cores();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_physical)
            .build()
            .map_err(|e| {
                IoSnafu {
                    message: format!("rayon pool: {e}"),
                }
                .build()
            })?;

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
        let head = self
            .heads
            .get(&dim)
            .ok_or_else(|| UnsupportedDimSnafu { dim }.build())?;

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
        //
        // WHY: this DOES fail fast despite being a plain `.collect()`.
        // `Result<C, E>: FromParallelIterator<Result<T, E>>` (rayon's
        // `result.rs`) is built on `while_some()`, which sets a shared
        // `full` flag on the first `Err` and checks it at every
        // work-stealing split boundary before recursing further — splits
        // not yet claimed are never dispatched once the flag is set.
        // Items already mid-flight on other threads at the moment of the
        // error still complete (inherent to any parallel short-circuit),
        // but the batch does not run to completion after an error.
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
            return CoreUnsupportedDimSnafu { dim }.fail();
        }
        let max_tokens = opts.max_tokens.unwrap_or(self.max_tokens);
        let text_with_prompt = apply_prompt(&self.prompts, opts.prompt.as_ref(), text)?;

        let ids: Vec<u32> = self
            .tokenizer
            .encode(&text_with_prompt, true)
            .map_err(|e| CoreTokenizeSnafu { message: e.to_string() }.build())?;
        check_token_limit(ids.len(), max_tokens)?;
        let mask = vec![1u8; ids.len()];
        self.encode_raw(&ids, &mask, dim)
            .map_err(|e| CoreComputeSnafu { message: e.to_string() }.build())
    }
}

/// Resolve `opts.prompt` against the checkpoint's loaded prompt map.
///
/// `None` passes `text` through unprefixed. `S2sQuery`/`S2pQuery`
/// prepend the checkpoint's role string when the sidecar supplied one,
/// and otherwise pass `text` through unprefixed. `Custom` always
/// prepends the caller's literal string. Any other `Prompt` variant —
/// reachable because `Prompt` is `#[non_exhaustive]`, so a caller built
/// against a newer `core` than this model may pass one — is rejected
/// rather than silently falling back to unprefixed text.
///
/// # Errors
///
/// [`EmbeddingError::UnsupportedPrompt`] on an unrecognised variant.
fn apply_prompt(
    prompts: &HashMap<String, String>,
    prompt: Option<&Prompt>,
    text: &str,
) -> std::result::Result<String, EmbeddingError> {
    Ok(match prompt {
        None => text.to_string(),
        Some(Prompt::S2sQuery) => prompts
            .get("s2s_query")
            .map_or_else(|| text.to_string(), |p| format!("{p}{text}")),
        Some(Prompt::S2pQuery) => prompts
            .get("s2p_query")
            .map_or_else(|| text.to_string(), |p| format!("{p}{text}")),
        Some(Prompt::Custom(s)) => format!("{s}{text}"),
        Some(_) => return Err(CoreUnsupportedPromptSnafu.build()),
    })
}

/// Reject a token count over `max_tokens` (already resolved from
/// `opts.max_tokens` or the model default).
///
/// # Errors
///
/// [`EmbeddingError::InputTooLong`] when `token_count > max_tokens`.
fn check_token_limit(
    token_count: usize,
    max_tokens: usize,
) -> std::result::Result<(), EmbeddingError> {
    if token_count > max_tokens {
        CoreInputTooLongSnafu {
            got: token_count,
            limit: max_tokens,
        }
        .fail()
    } else {
        Ok(())
    }
}

fn load_dense_head(root: &Path, dim: StellaDim, hidden: usize) -> Result<DenseHead> {
    let path: PathBuf = root.join(dim.folder()).join("model.safetensors");
    let reader = Reader::open(&path)?;
    let w_view = reader.get("linear.weight")?;
    let b_view = reader.get("linear.bias")?;
    if w_view.dtype != taxis::DType::F32 || b_view.dtype != taxis::DType::F32 {
        return IoSnafu {
            message: format!(
                "dense head {}: expected F32, got weight={:?}, bias={:?}",
                dim.folder(),
                w_view.dtype,
                b_view.dtype
            ),
        }
        .fail();
    }
    validate_head_shapes(dim.folder(), &w_view.shape, &b_view.shape, dim, hidden)?;
    let weight = bytes_to_f32(w_view.bytes);
    let bias = bytes_to_f32(b_view.bytes);
    Ok(DenseHead {
        weight,
        bias,
        dim: dim.width(),
    })
}

/// Validate a dense-head's `linear.weight` / `linear.bias` tensor
/// shapes against the expected `[dim, hidden]` / `[dim]` linear-layer
/// geometry. `hidden` comes from the loaded [`StellaConfig`] rather
/// than a fixed constant, so a checkpoint whose encoder hidden size
/// differs from Stella-1.5B's 1536 is validated correctly instead of
/// being rejected (or wrongly accepted) against the wrong width.
///
/// # Errors
///
/// [`Error::Io`] with a description of the mismatch on any shape
/// disagreement.
fn validate_head_shapes(
    label: &str,
    w_shape: &[usize],
    b_shape: &[usize],
    dim: StellaDim,
    hidden: usize,
) -> Result<()> {
    let expected_w = [dim.width(), hidden];
    if w_shape != expected_w.as_slice() {
        return IoSnafu {
            message: format!(
                "dense head {label}: expected weight shape {expected_w:?}, got {w_shape:?}"
            ),
        }
        .fail();
    }
    let expected_b = [dim.width()];
    if b_shape != expected_b.as_slice() {
        return IoSnafu {
            message: format!(
                "dense head {label}: expected bias shape {expected_b:?}, got {b_shape:?}"
            ),
        }
        .fail();
    }
    Ok(())
}

/// Detect physical-core count on Linux via `/proc/cpuinfo`. Falls back
/// to `rayon::current_num_threads()` on non-Linux.
fn detect_physical_cores() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/cpuinfo") {
            let n = count_physical_cores(&s);
            if n > 0 {
                return n;
            }
        }
    }
    std::thread::available_parallelism().map_or(4, std::num::NonZero::get)
}

/// Count distinct physical cores from `/proc/cpuinfo` text, keyed by
/// `(physical id, core id)`.
///
/// `core id` alone is not a stable key: every CPU socket restarts its
/// `core id` numbering from 0, so a multi-socket host's second socket
/// collides with the first under a `core id`-only `HashSet` and the
/// true physical-core count is undercounted.
///
/// WARNING: `physical id` is not universal — some virtualized/container
/// guests report `core id` with no `physical id` line, and requiring
/// both would zero out the count there, tripping the `n > 0` fallback
/// to the *logical* (SMT-inclusive) thread count. `physical_id`
/// defaults to `0` when absent, so a no-socket-field host still dedupes
/// on `core id` alone.
fn count_physical_cores(cpuinfo: &str) -> usize {
    let mut physical_id: Option<u32> = None;
    let mut core_id: Option<u32> = None;
    let mut cores = std::collections::HashSet::new();
    for line in cpuinfo.lines() {
        if let Some(v) = line.strip_prefix("physical id\t: ")
            && let Ok(id) = v.trim().parse::<u32>()
        {
            physical_id = Some(id);
        } else if let Some(v) = line.strip_prefix("core id\t: ")
            && let Ok(id) = v.trim().parse::<u32>()
        {
            core_id = Some(id);
        } else if line.trim().is_empty() {
            if let Some(c) = core_id {
                cores.insert((physical_id.unwrap_or(0), c));
            }
            physical_id = None;
            core_id = None;
        }
    }
    // The final processor block has no trailing blank line to flush it.
    if let Some(c) = core_id {
        cores.insert((physical_id.unwrap_or(0), c));
    }
    cores.len()
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
///
/// # Errors
///
/// An absent sidecar file (`io::ErrorKind::NotFound`) is not an error:
/// it resolves to an empty map. Every other IO fault — permissions,
/// the path resolving to a directory, a truncated read, ... —
/// propagates as [`Error::Io`] rather than being folded into "absent".
fn load_prompts(root: &Path) -> Result<HashMap<String, String>> {
    let path = root.join("config_sentence_transformers.json");
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(HashMap::new()),
        Err(err) => return Err(err.into()),
    };
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

    #[test]
    fn load_prompts_returns_empty_map_when_sidecar_absent() {
        let dir = tempfile::TempDir::new().expect("create temp dir");
        let got = load_prompts(dir.path()).expect("absent sidecar resolves, does not error");
        assert!(got.is_empty());
    }

    #[test]
    fn load_prompts_propagates_non_notfound_io_fault() {
        // WARNING(scope): guards `load_prompts`'s OWN `NotFound`-only
        // narrowing (stella.rs:490-504), not the originally-reported
        // bug — that lived at `load()`'s call site (removed match, see
        // stella.rs:157-166), and pre-fix `load_prompts` was already a
        // bare `read_to_string(&path)?`, so this exact assertion held
        // pre-fix too. The call site needs a full checkpoint to test
        // (see `phase_3_stella_parity.rs`, ignored for that); it's
        // verified by code review, not by this test.
        let dir = tempfile::TempDir::new().expect("create temp dir");
        std::fs::create_dir(dir.path().join("config_sentence_transformers.json"))
            .expect("create a directory standing in for the sidecar file");
        let result = load_prompts(dir.path());
        assert!(
            result.is_err(),
            "a directory in place of the sidecar file must not be folded into 'absent'"
        );
    }

    #[test]
    fn apply_prompt_none_passes_text_through_unprefixed() {
        let prompts = HashMap::new();
        let got = apply_prompt(&prompts, None, "hello").expect("None never errors");
        assert_eq!(got, "hello");
    }

    #[test]
    fn apply_prompt_s2s_query_prefixes_when_role_present() {
        let mut prompts = HashMap::new();
        prompts.insert("s2s_query".to_string(), "QUERY: ".to_string());
        let got = apply_prompt(&prompts, Some(&Prompt::S2sQuery), "hello")
            .expect("known role never errors");
        assert_eq!(got, "QUERY: hello");
    }

    #[test]
    fn apply_prompt_s2s_query_passes_through_when_role_absent_from_checkpoint() {
        let prompts = HashMap::new();
        let got = apply_prompt(&prompts, Some(&Prompt::S2sQuery), "hello")
            .expect("missing role falls back, does not error");
        assert_eq!(got, "hello");
    }

    #[test]
    fn apply_prompt_s2p_query_prefixes_when_role_present() {
        let mut prompts = HashMap::new();
        prompts.insert("s2p_query".to_string(), "PASSAGE: ".to_string());
        let got = apply_prompt(&prompts, Some(&Prompt::S2pQuery), "hello")
            .expect("known role never errors");
        assert_eq!(got, "PASSAGE: hello");
    }

    #[test]
    fn apply_prompt_custom_always_prefixes_the_caller_literal() {
        let prompts = HashMap::new();
        let got = apply_prompt(&prompts, Some(&Prompt::Custom("X: ".to_string())), "hello")
            .expect("Custom never errors");
        assert_eq!(got, "X: hello");
    }

    #[test]
    fn check_token_limit_ok_at_and_under_the_boundary() {
        assert!(check_token_limit(4, 4).is_ok());
        assert!(check_token_limit(3, 4).is_ok());
    }

    #[test]
    fn check_token_limit_rejects_over_an_overridden_threshold() {
        // Negative fixture for the "max_tokens override has no test"
        // gap: a token count that fits the model default (512) but not
        // a caller-supplied override must be rejected AT the override,
        // proving the override actually takes effect.
        let err = check_token_limit(5, 4).expect_err("over-limit must error");
        assert!(matches!(
            err,
            EmbeddingError::InputTooLong {
                got: 5,
                limit: 4,
                ..
            }
        ));
    }

    #[test]
    fn validate_head_shapes_accepts_hidden_threaded_from_config() {
        // Negative fixture for the hardcoded-1536 finding: `hidden`
        // deliberately differs from the Stella-1.5B literal. A head
        // shaped against THAT hidden must still validate -- proving the
        // width is threaded from config, not fixed.
        let hidden = 2048;
        let result = validate_head_shapes(
            "test",
            &[StellaDim::Dim1024.width(), hidden],
            &[StellaDim::Dim1024.width()],
            StellaDim::Dim1024,
            hidden,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn validate_head_shapes_rejects_weight_hidden_mismatch() {
        let result = validate_head_shapes(
            "test",
            &[StellaDim::Dim1024.width(), 1536],
            &[StellaDim::Dim1024.width()],
            StellaDim::Dim1024,
            2048, // disagrees with the supplied weight shape's second axis
        );
        assert!(result.is_err());
    }

    #[test]
    fn validate_head_shapes_rejects_bias_length_mismatch() {
        let result = validate_head_shapes(
            "test",
            &[StellaDim::Dim1024.width(), 1536],
            &[StellaDim::Dim2048.width()],
            StellaDim::Dim1024,
            1536,
        );
        assert!(result.is_err());
    }

    #[test]
    fn count_physical_cores_single_socket_dedupes_hyperthreads() {
        let cpuinfo = "\
processor\t: 0
core id\t: 0
physical id\t: 0

processor\t: 1
core id\t: 0
physical id\t: 0

processor\t: 2
core id\t: 1
physical id\t: 0

processor\t: 3
core id\t: 1
physical id\t: 0
";
        assert_eq!(count_physical_cores(cpuinfo), 2);
    }

    #[test]
    fn count_physical_cores_multi_socket_does_not_undercount() {
        // Negative fixture for the core-id-only-dedup finding: `core
        // id` restarts from 0 on the second socket. Keying on `core id`
        // alone (the pre-fix behaviour) collapses socket 1's core 0/1
        // onto socket 0's and undercounts 4 physical cores as 2.
        let cpuinfo = "\
processor\t: 0
core id\t: 0
physical id\t: 0

processor\t: 1
core id\t: 1
physical id\t: 0

processor\t: 2
core id\t: 0
physical id\t: 1

processor\t: 3
core id\t: 1
physical id\t: 1
";
        assert_eq!(count_physical_cores(cpuinfo), 4);
    }

    #[test]
    fn count_physical_cores_without_physical_id_field_still_counts_via_core_id() {
        // Negative fixture for requiring BOTH fields: a strict-pairing
        // impl never completes a pair when `physical id` is absent, so
        // this returns 0 -- tripping `detect_physical_cores`'s fallback
        // to the SMT-inclusive logical count. Two `core id` values,
        // zero `physical id` lines: must resolve to 2, not 0.
        let cpuinfo = "\
processor\t: 0
core id\t: 0

processor\t: 1
core id\t: 0

processor\t: 2
core id\t: 1

processor\t: 3
core id\t: 1
";
        assert_eq!(count_physical_cores(cpuinfo), 2);
    }

    #[test]
    fn count_physical_cores_final_block_without_trailing_blank_line_counts() {
        // WHY: /proc/cpuinfo's last processor block has no trailing
        // blank line to flush it through the per-block reset logic.
        let cpuinfo = "processor\t: 0\ncore id\t: 0\nphysical id\t: 0";
        assert_eq!(count_physical_cores(cpuinfo), 1);
    }
}
