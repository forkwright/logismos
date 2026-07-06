//! Intermediate-layer diagnostic for parity debugging. Dumps the first
//! few elements of the embedding table lookup, after layer 0, and
//! after the final norm for a fixed three-token input `"hello world"`.
//! Compared by eye against the Python reference at
//! `/tmp/debug_stella.py`.
//!
//! Compiled only under `--features debug-diagnostics`. Run manually
//! with `cargo test -p logismos --features debug-diagnostics --test
//! phase_3_debug`.

#![cfg(feature = "debug-diagnostics")]
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::unreadable_literal
)]

use std::path::Path;

use encoders::{StellaConfig, StellaEncoder};
use kernels::cpu_f32;
use transformers::rms_norm_f32;

const STELLA_ROOT: &str = "/models/stella-1.5b-v5";

/// Aggregated error for the Phase-3 debug diagnostic.
#[derive(Debug, thiserror::Error)]
enum TestError {
    #[error("encoders: {0}")]
    Encoders(#[from] encoders::Error),
    #[error("transformers: {0}")]
    Transformers(#[from] transformers::Error),
}

// Feature-gated to `debug-diagnostics`; the whole test binary does
// not compile in normal builds. No skip attribute needed on top of
// the feature gate -- the attribute would only fire if someone opted
// into the diagnostic, at which point they want the test to run.
#[test]
fn stella_debug_layer0() -> Result<(), TestError> {
    let root = Path::new(STELLA_ROOT);
    if !root.exists() {
        eprintln!("[skip] no stella");
        return Ok(());
    }

    let cfg = StellaConfig::stella_1_5b();
    let encoder = StellaEncoder::load(&root.join("model.safetensors"), cfg)?;

    // "hello world" → ids [14990, 1879, 151643] via tokenizers.
    let ids = [14990_u32, 1879, 151643];
    let mask = [1_u8; 3];

    // Embedding lookup.
    let embed = cpu_f32::embed_lookup(&encoder.tok_embed, cfg.hidden, cfg.vocab_size, &ids);
    eprintln!("embed [row0, :5] = {:?}", &embed[..5]);

    // Run full forward.
    let out = encoder.forward(&ids, &mask)?;
    eprintln!("final [row0, :5] = {:?}", &out[..5]);

    // Also run a single layer manually.
    let x = cpu_f32::embed_lookup(&encoder.tok_embed, cfg.hidden, cfg.vocab_size, &ids);
    let layer = &encoder.layers[0];
    let norm = rms_norm_f32(&x, &layer.norm1, 3, cfg.hidden, cfg.rms_eps)?;
    eprintln!("norm1(embed) [row0, :5] = {:?}", &norm[..5]);
    let attn = layer.attn.forward(&norm, &mask, &encoder.rope)?;
    eprintln!("attn_out [row0, :5] = {:?}", &attn[..5]);
    let mut x_after_attn = x.clone();
    for (xi, ai) in x_after_attn.iter_mut().zip(attn.iter()) {
        *xi += ai;
    }
    let norm2 = rms_norm_f32(&x_after_attn, &layer.norm2, 3, cfg.hidden, cfg.rms_eps)?;
    let mlp_out = layer.mlp.forward(&norm2)?;
    let mut x_after_layer = x_after_attn.clone();
    for (xi, mi) in x_after_layer.iter_mut().zip(mlp_out.iter()) {
        *xi += mi;
    }
    eprintln!("post_layer_0 [row0, :5] = {:?}", &x_after_layer[..5]);

    Ok(())
}
