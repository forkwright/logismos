#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::map_unwrap_or,
    clippy::float_cmp,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc
)]
//! Phase 3 exit-gate integration test.
//!
//! Loads the on-disk Stella 1.5B v5 checkpoint, tokenises the fixture
//! sentences in `phases/03-stella/golden/inputs.txt`, runs the forward
//! pass through `embed::StellaModel`, and compares the resulting 1024-d
//! vectors against the Python reference at
//! `phases/03-stella/golden/embeddings_dim1024.safetensors`.
//!
//! **Gate:** every sentence must satisfy
//! `max |a[i] - b[i]| <= 1e-3` AND `cosine(a, b) > 0.999`.
//!
//! This test is ignored by default because it requires the full
//! `/models/stella-1.5b-v5` checkpoint. Run it explicitly with
//! `cargo test -p logismos --test phase_3_stella_parity -- --ignored`.
//! When run, it fails hard if the checkpoint or golden fixtures are
//! missing or if parity diverges.

use std::path::{Path, PathBuf};

use loader::WeightProvider;
use logismos_core::{EmbeddingModel, EncodeOpts};

const STELLA_ROOT: &str = "/models/stella-1.5b-v5";
const INPUTS_REL: &str = "phases/03-stella/golden/inputs.txt";
const GOLDEN_REL: &str = "phases/03-stella/golden/embeddings_dim1024.safetensors";

/// Aggregated error for the Phase-3 parity test.
#[derive(Debug, thiserror::Error)]
enum TestError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("loader: {0}")]
    Loader(#[from] loader::Error),
    #[error("embed: {0}")]
    Embed(#[from] logismos_core::EmbeddingError),
    #[error("stella: {0}")]
    Stella(#[from] embed::Error),
    #[error("{0}")]
    Msg(String),
}

fn workspace_root() -> PathBuf {
    // This file sits at CRATE_ROOT/tests/phase_3_stella_parity.rs; the
    // workspace root is three parents up (crates/logismos/tests → workspace).
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // -> crates
    p.pop(); // -> workspace root
    p
}

#[test]
#[ignore = "requires /models/stella-1.5b-v5 checkpoint - see #26"]
fn phase_3_stella_parity() -> Result<(), TestError> {
    let root = Path::new(STELLA_ROOT);
    let ws = workspace_root();
    let inputs_path = ws.join(INPUTS_REL);
    let golden_path = ws.join(GOLDEN_REL);

    if !root.exists() {
        return Err(TestError::Msg(format!(
            "Stella checkpoint not present at {}",
            root.display()
        )));
    }
    if !inputs_path.exists() {
        return Err(TestError::Msg(format!(
            "golden inputs missing at {}",
            inputs_path.display()
        )));
    }
    if !golden_path.exists() {
        return Err(TestError::Msg(format!(
            "golden embeddings missing at {}",
            golden_path.display()
        )));
    }

    let sentences: Vec<String> = std::fs::read_to_string(&inputs_path)?
        .lines()
        .filter(|l| !l.is_empty())
        .map(ToString::to_string)
        .collect();
    eprintln!("[phase-3] {} sentences", sentences.len());

    let golden = read_golden(&golden_path)?;
    assert_eq!(
        golden.len(),
        sentences.len(),
        "golden rows != sentence count"
    );
    let golden_dim = golden[0].len();
    assert_eq!(golden_dim, 1024, "golden must be dim=1024");

    // Load the model — single forward pass per sentence through the
    // Phase-3 fp32 CPU path.
    let model = embed::StellaModel::load(root, &[embed::StellaDim::Dim1024])?;
    eprintln!("[phase-3] model loaded: {model:?}");

    let opts = EncodeOpts {
        dim: Some(1024),
        max_tokens: None,
        prompt: None,
    };

    let mut max_abs_all: f32 = 0.0;
    let mut min_cos_all: f64 = 1.0;
    let mut failures: Vec<String> = Vec::new();

    for (i, sent) in sentences.iter().enumerate() {
        let got = model.encode(sent, &opts)?;
        assert_eq!(got.len(), golden_dim);
        let (max_abs, cos, worst) = compare_embeddings(&got, &golden[i]);
        if max_abs > max_abs_all {
            max_abs_all = max_abs;
        }
        if cos < min_cos_all {
            min_cos_all = cos;
        }
        eprintln!(
            "[phase-3] [{i:02}] max_abs={max_abs:.5}  cos={cos:.6}  len={}",
            sent.len()
        );
        if max_abs > 1e-3 || cos <= 0.999 {
            failures.push(format!(
                "sentence {i}: max_abs={max_abs:.6}, cos={cos:.6}\n  worst: {worst:?}"
            ));
        }
    }

    eprintln!("[phase-3] SUMMARY max_abs={max_abs_all:.5} min_cos={min_cos_all:.6}");
    if !failures.is_empty() {
        return Err(TestError::Msg(format!(
            "phase-3 parity gate failed:\n  {}",
            failures.join("\n  ")
        )));
    }
    Ok(())
}

/// Compare one (got, want) embedding pair. Returns `(max_abs_diff, cos_sim,
/// worst)` where `worst` is the top-5 worst-diff entries as `(index, got,
/// want, |diff|)`. `cos_sim` is held in f64 because cosine of two normalised
/// vectors lives in `[-1, 1]` and the extra precision matters for the 0.999
/// pass threshold.
fn compare_embeddings(got: &[f32], want: &[f32]) -> (f32, f64, Vec<(usize, f32, f32, f32)>) {
    let mut max_abs: f32 = 0.0;
    let mut dot: f64 = 0.0;
    let mut na: f64 = 0.0;
    let mut nb: f64 = 0.0;
    let mut worst: Vec<(usize, f32, f32, f32)> = Vec::new();
    for (j, (&a, &b)) in got.iter().zip(want.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
        dot += f64::from(a) * f64::from(b);
        na += f64::from(a) * f64::from(a);
        nb += f64::from(b) * f64::from(b);
        if worst.len() < 5 || d > worst.last().map(|w| w.3).unwrap_or(0.0) {
            worst.push((j, a, b, d));
            worst.sort_by(|x, y| y.3.partial_cmp(&x.3).unwrap_or(std::cmp::Ordering::Equal));
            worst.truncate(5);
        }
    }
    let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
    (max_abs, cos, worst)
}

fn read_golden(path: &Path) -> Result<Vec<Vec<f32>>, TestError> {
    let reader = loader::safetensors::Reader::open(path)?;
    let view = reader.get("embeddings")?;
    assert_eq!(view.dtype, taxis::DType::F32);
    assert_eq!(view.shape.len(), 2);
    let rows = view.shape[0];
    let cols = view.shape[1];
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let row_start = r * cols * 4;
        let row_end = row_start + cols * 4;
        let mut row = Vec::with_capacity(cols);
        for c in view.bytes[row_start..row_end].chunks_exact(4) {
            let mut b = [0u8; 4];
            b.copy_from_slice(c);
            row.push(f32::from_le_bytes(b));
        }
        out.push(row);
    }
    Ok(out)
}
