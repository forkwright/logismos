//! Deterministic Phase 3 fixture validation.
//!
//! Verifies that the committed golden artefacts in `phases/03-stella/golden/`
//! are present, well-formed, and internally consistent. This test is part of
//! the normal PR gate and must **never** silently skip - missing or malformed
//! fixtures are a hard failure.

use std::path::PathBuf;

use loader::WeightProvider;
use serde_json::Value;

const GOLDEN_DIR: &str = "phases/03-stella/golden";
const INPUTS: &str = "inputs.txt";
const TOKENS: &str = "tokens.jsonl";
const EMBEDDINGS: &str = "embeddings_dim1024.safetensors";
const BASELINE: &str = "cpu_baseline.json";
const PROVENANCE: &str = "PROVENANCE.json";

/// Aggregated error for the fixture check.
#[derive(Debug, thiserror::Error)]
enum TestError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("loader: {0}")]
    Loader(#[from] loader::Error),
    #[error("fixture: {0}")]
    Msg(String),
}

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one linear top-to-bottom fixture validation; splitting it obscures the \
              existence -> inputs -> tokens -> embeddings -> provenance -> baseline sequence"
)]
fn phase_3_fixture_check() -> Result<(), TestError> {
    let ws = workspace_root();
    let golden = ws.join(GOLDEN_DIR);

    // --- existence -------------------------------------------------------
    for name in [INPUTS, TOKENS, EMBEDDINGS, BASELINE, PROVENANCE] {
        let path = golden.join(name);
        if !path.exists() {
            return Err(TestError::Msg(format!(
                "missing fixture: {}",
                path.display()
            )));
        }
    }

    // --- inputs.txt ------------------------------------------------------
    let inputs_path = golden.join(INPUTS);
    let inputs_text = std::fs::read_to_string(&inputs_path)?;
    let sentences: Vec<&str> = inputs_text.lines().filter(|l| !l.is_empty()).collect();
    if sentences.is_empty() {
        return Err(TestError::Msg("inputs.txt has no sentences".into()));
    }

    let tokens_path = golden.join(TOKENS);
    let tokens_text = std::fs::read_to_string(&tokens_path)?;
    let token_lines: Vec<&str> = tokens_text.lines().collect();
    if token_lines.len() != sentences.len() {
        return Err(TestError::Msg(format!(
            "tokens.jsonl has {} lines but inputs.txt has {} sentences",
            token_lines.len(),
            sentences.len()
        )));
    }

    for (i, (sent, line)) in sentences.iter().zip(token_lines.iter()).enumerate() {
        let record: Value = serde_json::from_str(line)
            .map_err(|e| TestError::Msg(format!("tokens.jsonl line {i}: invalid JSON: {e}")))?;
        let text = json_str(&record, "text", &format!("tokens.jsonl line {i}"))?;
        if text != *sent {
            return Err(TestError::Msg(format!(
                "tokens.jsonl line {i}: text mismatch\n  expected: {sent}\n  got: {text}"
            )));
        }
        let ids = json_u64_array(&record, "ids", &format!("tokens.jsonl line {i}"))?;
        let mask = json_u64_array(&record, "attention_mask", &format!("tokens.jsonl line {i}"))?;
        if ids.len() != mask.len() {
            return Err(TestError::Msg(format!(
                "tokens.jsonl line {i}: ids.len={} != mask.len={}",
                ids.len(),
                mask.len()
            )));
        }
    }

    // --- embeddings ------------------------------------------------------
    let emb_path = golden.join(EMBEDDINGS);
    let reader = loader::safetensors::Reader::open(&emb_path)?;
    let view = reader
        .get("embeddings")
        .map_err(|e| TestError::Msg(format!("embeddings tensor 'embeddings' not found: {e}")))?;
    if view.dtype != taxis::DType::F32 {
        return Err(TestError::Msg(format!(
            "embeddings expected F32, got {:?}",
            view.dtype
        )));
    }
    if view.shape.len() != 2 || view.shape[0] != sentences.len() || view.shape[1] != 1024 {
        return Err(TestError::Msg(format!(
            "embeddings expected shape [{}, 1024], got {:?}",
            sentences.len(),
            view.shape
        )));
    }

    let prov_path = golden.join(PROVENANCE);
    let prov_text = std::fs::read_to_string(&prov_path)?;
    let provenance: Value = serde_json::from_str(&prov_text)
        .map_err(|e| TestError::Msg(format!("PROVENANCE.json: invalid JSON: {e}")))?;
    let prov_sentences = json_u64(&provenance, "num_sentences", "PROVENANCE.json")?;
    if prov_sentences != sentences.len() as u64 {
        return Err(TestError::Msg(format!(
            "PROVENANCE.json num_sentences={} but inputs.txt has {}",
            prov_sentences,
            sentences.len()
        )));
    }
    let prov_dim = json_u64(&provenance, "dim", "PROVENANCE.json")?;
    if prov_dim != 1024 {
        return Err(TestError::Msg(format!(
            "PROVENANCE.json dim={prov_dim} expected 1024"
        )));
    }

    let base_path = golden.join(BASELINE);
    let base_text = std::fs::read_to_string(&base_path)?;
    let baseline: Value = serde_json::from_str(&base_text)
        .map_err(|e| TestError::Msg(format!("cpu_baseline.json: invalid JSON: {e}")))?;
    let _throughput = json_f64(&baseline, "throughput_sent_per_sec", "cpu_baseline.json")?;

    eprintln!(
        "[phase-3-fixture] {} sentences, {} token records, embeddings={:?}, baseline OK",
        sentences.len(),
        token_lines.len(),
        view.shape
    );
    Ok(())
}

fn json_str<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str, TestError> {
    value.get(key).and_then(Value::as_str).ok_or_else(|| {
        TestError::Msg(format!(
            "{context}: missing or malformed string field `{key}`"
        ))
    })
}

fn json_u64(value: &Value, key: &str, context: &str) -> Result<u64, TestError> {
    value.get(key).and_then(Value::as_u64).ok_or_else(|| {
        TestError::Msg(format!(
            "{context}: missing or malformed integer field `{key}`"
        ))
    })
}

fn json_f64(value: &Value, key: &str, context: &str) -> Result<f64, TestError> {
    value.get(key).and_then(Value::as_f64).ok_or_else(|| {
        TestError::Msg(format!(
            "{context}: missing or malformed float field `{key}`"
        ))
    })
}

fn json_u64_array(value: &Value, key: &str, context: &str) -> Result<Vec<u64>, TestError> {
    let values = value.get(key).and_then(Value::as_array).ok_or_else(|| {
        TestError::Msg(format!(
            "{context}: missing or malformed array field `{key}`"
        ))
    })?;
    values
        .iter()
        .enumerate()
        .map(|(i, item)| {
            item.as_u64().ok_or_else(|| {
                TestError::Msg(format!(
                    "{context}: `{key}` item {i} is not an unsigned integer"
                ))
            })
        })
        .collect()
}
