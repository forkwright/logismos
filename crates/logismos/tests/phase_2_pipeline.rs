//! Phase 2 exit-gate integration test.
//!
//! Exercises the full tier-2 pipeline end-to-end on real on-disk
//! fixtures wherever possible, falling back to inline fixtures when
//! the disk-resident models aren't present (so the test suite stays
//! green on a fresh checkout).
//!
//! Scenario:
//!
//! 1. Open a safetensors file. Stella's `2_Dense_1024` dense head
//!    (~6 MB) is the default.
//! 2. Enumerate every tensor found.
//! 3. Load a known tensor into a CPU `taxis::Tensor`.
//! 4. Tokenize a fixture string through a tokenizer.json; round-trip
//!    ids → text.
//! 5. Exercise `FlatKvCache::put / get / reset`.
//! 6. Greedy-decode a pre-set logits vector.
//!
//! Per the Phase-2 PLAN exit gate: no forward pass runs here (Phase 3
//! ships that). This test proves weight bytes reach a taxis tensor,
//! tokens round-trip, the cache works, and the sampler works.

use std::io::Write;
use std::path::PathBuf;

use cache::{CacheLayout, FlatKvCache, KvCache};
use decode::{DecodeChain, GreedySampler, TemperatureScale, TokenContext, TopK, TopP};
use loader::WeightProvider;
use taxis::{CpuStorage, Shape, Tensor};

const STELLA_SAFETENSORS: &str = "/models/stella-1.5b-v5/2_Dense_1024/model.safetensors";
const STELLA_TOKENIZER: &str = "/models/stella-1.5b-v5/tokenizer.json";

/// Aggregated error for the Phase-2 integration test. Each variant wraps a
/// concrete upstream error type; `Msg` covers synthetic-fixture write failures
/// where the upstream surfaces a non-`Error` struct.
#[derive(Debug, thiserror::Error)]
enum TestError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("loader: {0}")]
    Loader(#[from] loader::Error),
    #[error("tokenize: {0}")]
    Tokenize(#[from] tokenize::Error),
    #[error("taxis: {0}")]
    Taxis(#[from] taxis::Error),
    #[error("cache: {0}")]
    Cache(#[from] cache::Error),
    #[error("{0}")]
    Msg(String),
}

#[test]
fn phase_2_full_pipeline() -> Result<(), TestError> {
    // -- 1 & 2: open a safetensors file and enumerate tensors. -------
    let (safetensors_path, is_real) = resolve_safetensors()?;
    let archive = loader::Archive::open(&safetensors_path)?;
    let names = archive.names();
    assert!(
        !names.is_empty(),
        "archive must declare at least one tensor"
    );
    eprintln!(
        "[phase-2] safetensors `{}` → {} tensors",
        safetensors_path.display(),
        names.len()
    );
    for n in &names {
        eprintln!("  tensor: {n}");
    }

    // -- 3: load one known tensor into a CPU taxis::Tensor. ----------
    let target_name = names[0].clone();
    let view = archive.get(&target_name)?;
    let tensor = view.to_tensor_cpu()?;
    assert_eq!(tensor.dims(), view.shape.as_slice());
    eprintln!(
        "[phase-2] loaded tensor `{target_name}` into taxis::Tensor dims={:?} dtype={:?}",
        tensor.dims(),
        tensor.dtype()
    );
    if is_real {
        // For Stella's dense head, the tensor is named "linear.weight"
        // or similar and is a rank-2 [1536, 1024] matrix. We don't
        // hard-code the name — just validate the rank.
        assert!(!tensor.dims().is_empty());
    }

    // -- 4: tokenizer round-trip. ------------------------------------
    let (tokenizer_path, tokenizer_is_real) = resolve_tokenizer()?;
    let tok = tokenize::Tokenizer::from_file(&tokenizer_path)?;
    let text = if tokenizer_is_real {
        "hello world"
    } else {
        // The trivial fallback vocabulary knows hello + world + ...
        "hello world the quick fox"
    };
    let ids = tok.encode(text, false)?;
    assert!(!ids.is_empty(), "tokenizer must emit at least one id");
    let round_trip = tok.decode(&ids, false)?;
    eprintln!(
        "[phase-2] tokenize vocab={} ids={ids:?} round_trip={round_trip:?}",
        tok.vocab_size()
    );
    // Bytes need not be bit-equal (whitespace handling differs by
    // model); the core tokens must survive.
    for word in text.split_whitespace() {
        assert!(
            round_trip.to_lowercase().contains(&word.to_lowercase()),
            "round-trip dropped `{word}`: got `{round_trip}`"
        );
    }

    // -- 5: cache put / get / reset. ---------------------------------
    exercise_kv_cache()?;

    // -- 6: greedy decode over a pre-set logits vector. --------------
    exercise_decode();

    eprintln!("[phase-2] ALL STEPS GREEN — exit gate satisfied.");
    Ok(())
}

/// Step 5 of the pipeline: write 3 rows × 2 layers into `FlatKvCache`, read
/// them back, reset, confirm the cache zeroes its per-layer length.
fn exercise_kv_cache() -> Result<(), TestError> {
    let layout = CacheLayout {
        num_layers: 2,
        num_kv_heads: 2,
        head_dim: 4,
        max_seq_len: 16,
        dtype: taxis::DType::F32,
    };
    let mut kv = FlatKvCache::new(layout);
    let row_elems = layout.row_elems();
    let mk_row = |val: f32| -> Tensor {
        Tensor::from_cpu(
            CpuStorage::F32(vec![val; row_elems]),
            Shape::new(&[1, row_elems]),
        )
    };
    for step in 0..3u8 {
        for layer in 0..layout.num_layers {
            let base = f32::from(step);
            let k = mk_row(base + 0.1);
            let v = mk_row(base + 0.9);
            kv.put(layer, &k, &v)?;
        }
    }
    assert_eq!(kv.len_of(0), 3);
    assert_eq!(kv.len_of(1), 3);
    let (k_read, v_read) = kv.get(0, 3)?;
    assert_eq!(k_read.dims(), &[3, row_elems]);
    assert_eq!(v_read.dims(), &[3, row_elems]);
    kv.reset();
    assert_eq!(kv.len_of(0), 0);
    eprintln!("[phase-2] cache: put 3 rows × 2 layers → read back → reset OK");
    Ok(())
}

/// Step 6 of the pipeline: greedy argmax + a four-stage decode chain over
/// a fixed-logits vector; both must pick index 3.
fn exercise_decode() {
    let logits = vec![0.1f32, 0.4, 0.2, 0.8, 0.3];
    assert_eq!(decode::greedy(&logits), 3);

    // And through a chain for the fuller shape.
    let mut chain = DecodeChain::new(GreedySampler)
        .push(TemperatureScale(0.7))
        .push(TopK(3))
        .push(TopP(0.95));
    let mut log_buf = logits.clone();
    let ctx = TokenContext {
        prev_tokens: &[],
        step: 0,
    };
    let id = chain.step(&mut log_buf, &ctx);
    assert_eq!(
        id, 3,
        "chain's greedy pick on [0.1,0.4,0.2,0.8,0.3] is index 3"
    );
    eprintln!("[phase-2] decode: greedy + chain both picked token id 3");
}

// ---------------------------------------------------------------------
// Fixture resolution — prefer on-disk Stella, fall back to synthetic.
// ---------------------------------------------------------------------

fn resolve_safetensors() -> Result<(PathBuf, bool), TestError> {
    let real = PathBuf::from(STELLA_SAFETENSORS);
    if real.exists() {
        return Ok((real, true));
    }
    eprintln!("[phase-2] Stella 2_Dense_1024 not on disk; writing synthetic safetensors fixture.");
    let path = std::env::temp_dir().join(format!(
        "logismos-phase2-{}.safetensors",
        std::process::id()
    ));
    write_synthetic_safetensors(&path)?;
    Ok((path, false))
}

fn resolve_tokenizer() -> Result<(PathBuf, bool), TestError> {
    let real = PathBuf::from(STELLA_TOKENIZER);
    if real.exists() {
        return Ok((real, true));
    }
    eprintln!("[phase-2] Stella tokenizer.json not on disk; writing synthetic fixture.");
    let path = std::env::temp_dir().join(format!(
        "logismos-phase2-{}-tokenizer.json",
        std::process::id()
    ));
    write_synthetic_tokenizer(&path)?;
    Ok((path, false))
}

fn write_synthetic_safetensors(path: &std::path::Path) -> Result<(), TestError> {
    use std::collections::HashMap as M;

    use safetensors::serialize_to_file;
    use safetensors::tensor::{Dtype as UD, TensorView as UV};
    let a_bytes: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let b_bytes: Vec<u8> = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let a =
        UV::new(UD::F32, vec![2, 2], &a_bytes).map_err(|e| TestError::Msg(format!("tv a: {e}")))?;
    let b =
        UV::new(UD::F32, vec![2, 3], &b_bytes).map_err(|e| TestError::Msg(format!("tv b: {e}")))?;
    let mut m: M<String, UV<'_>> = M::new();
    m.insert("dense.weight".into(), a);
    m.insert("dense.bias".into(), b);
    serialize_to_file(&m, None, path).map_err(|e| TestError::Msg(format!("write: {e}")))?;
    Ok(())
}

fn write_synthetic_tokenizer(path: &std::path::Path) -> Result<(), TestError> {
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
    f.write_all(json.as_bytes())?;
    Ok(())
}
