#![allow(
    clippy::cast_precision_loss,
    clippy::needless_continue,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc
)]
//! Throughput benchmark for Stella 1.5B v5 on CPU.
//!
//! Uses `std::time` + a fixed 15-sentence fixture. Prints
//! `sentences/sec`, `mean_latency_ms`, and the ratio over the committed
//! CPU-candle baseline.
//!
//! This benchmark is gated by the `STELLA_BENCH` environment variable.
//! Set `STELLA_BENCH=1` to run; otherwise it exits with an explicit skip
//! message. This prevents silent success in CI when the checkpoint is
//! absent.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::{io, writeln};

use embed::{StellaDim, StellaModel};
use logismos_core::{EmbeddingModel, EncodeOpts};

const STELLA_ROOT: &str = "/models/stella-1.5b-v5";
const INPUTS_REL: &str = "phases/03-stella/golden/inputs.txt";
const BASELINE_REL: &str = "phases/03-stella/golden/cpu_baseline.json";
const BENCH_ENV: &str = "STELLA_BENCH";

/// Aggregated error for the Stella throughput benchmark.
#[derive(Debug, thiserror::Error)]
enum BenchError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("embed: {0}")]
    Embed(#[from] logismos_core::EmbeddingError),
    #[error("stella: {0}")]
    Stella(#[from] embed::Error),
    #[error("{0}")]
    Msg(String),
}

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p
}

fn main() -> Result<(), BenchError> {
    let mut stderr = io::stderr().lock();
    let root = Path::new(STELLA_ROOT);
    let ws = workspace_root();
    let inputs_path = ws.join(INPUTS_REL);
    let baseline_path = ws.join(BASELINE_REL);

    if std::env::var(BENCH_ENV).is_err() {
        writeln!(
            stderr,
            "[bench] SKIP - set {BENCH_ENV}=1 to run Stella throughput benchmark"
        )?;
        return Ok(());
    }
    if !root.exists() {
        return Err(BenchError::Msg(format!(
            "Stella checkpoint not present at {}",
            root.display()
        )));
    }
    if !inputs_path.exists() {
        return Err(BenchError::Msg(format!(
            "Stella inputs not present at {}",
            inputs_path.display()
        )));
    }

    let sentences: Vec<String> = std::fs::read_to_string(&inputs_path)?
        .lines()
        .filter(|l| !l.is_empty())
        .map(ToString::to_string)
        .collect();
    writeln!(stderr, "[bench] loading model...")?;
    let t_load = Instant::now();
    let model = StellaModel::load(root, &[StellaDim::Dim1024])?;
    writeln!(
        stderr,
        "[bench] model loaded in {:.2} s",
        t_load.elapsed().as_secs_f64()
    )?;

    let opts = EncodeOpts {
        dim: Some(1024),
        max_tokens: None,
        prompt: None,
    };

    // Steady-state batch: 32 concurrent sentences (matches the host's
    // physical-core count for maximum per-sentence parallelism without
    // SMT contention). The benchmark runs three passes, skipping the
    // first as warm-up.
    let n_cores = 32;
    let batch: Vec<String> = sentences.iter().cycle().take(n_cores).cloned().collect();
    let refs: Vec<&str> = batch.iter().map(String::as_str).collect();

    // Warm-up pass.
    let _ = model.encode_batch(&refs, &opts)?;

    // Measured: 5 batches, report aggregate throughput.
    let n_iters = 5;
    let total_sentences = refs.len() * n_iters;
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = model.encode_batch(&refs, &opts)?;
    }
    let total = t0.elapsed();
    let total_sentences_f64 = f64::from(u32::try_from(total_sentences).unwrap_or(u32::MAX));
    let throughput = total_sentences_f64 / total.as_secs_f64();
    let mean_ms = total.as_secs_f64() * 1000.0 / total_sentences_f64;
    writeln!(
        stderr,
        "[bench] sentences={total_sentences}  total={:.2}s  throughput={throughput:.3} sent/s  mean_latency={mean_ms:.1} ms",
        total.as_secs_f64()
    )?;

    if baseline_path.exists() {
        let s = std::fs::read_to_string(&baseline_path)?;
        if let Some(bl_throughput) = parse_baseline_throughput(&s) {
            let ratio = throughput / bl_throughput;
            writeln!(
                stderr,
                "[bench] cpu-candle baseline = {bl_throughput:.3} sent/s; speed-up = {ratio:.2}x"
            )?;
            if ratio < 10.0 {
                writeln!(stderr, "[bench] WARN: below 10x gate")?;
            }
        }
    }
    Ok(())
}

fn parse_baseline_throughput(json: &str) -> Option<f64> {
    // Ultra-minimal: find "throughput_sent_per_sec": <number>
    let key = "\"throughput_sent_per_sec\"";
    let idx = json.find(key)?;
    let after = json.get(idx + key.len()..)?;
    let colon = after.find(':')?;
    let rest = after.get(colon + 1..)?;
    let mut end = 0;
    for (i, c) in rest.char_indices() {
        if c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E' || c == '+' {
            end = i + c.len_utf8();
        } else if end > 0 {
            break;
        } else {
            continue;
        }
    }
    rest.get(..end).and_then(|s| s.trim().parse::<f64>().ok())
}
