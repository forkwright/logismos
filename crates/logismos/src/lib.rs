//! # logismos
//!
//! Top-level facade crate. Re-exports ergonomic surfaces from the
//! workspace so a consumer that wants "the whole thing" depends on
//! this one crate instead of enumerating the sub-crates they would
//! otherwise need.
//!
//! The facade carries **no logic** — any helper that grows here should
//! move out into the appropriate sub-crate.
//!
//! Phase 2 surfaces: loader, tokenize, cache, decode, taxis.
//! Phase 3 adds: core (trait surface), encoders, embed (StellaModel).
//!
//! ## Quick start (Phase 3 — Stella)
//!
//! ```ignore
//! use logismos::embed::{StellaDim, StellaModel};
//! use logismos::core::{EmbeddingModel, EncodeOpts};
//! use std::path::Path;
//!
//! let model = StellaModel::load(
//!     Path::new("/models/stella-1.5b-v5"),
//!     &[StellaDim::Dim1024],
//! )?;
//! let vec: Vec<f32> = model.encode("hello world", &EncodeOpts::default())?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![deny(missing_docs)]
#![allow(clippy::doc_markdown)]

pub use cache;
pub use decode;
pub use embed;
pub use encoders;
pub use loader;
pub use logismos_core as core;
pub use taxis;
pub use tokenize;
pub use transformers;

/// Convenience constructor: build a `Box<dyn EmbeddingModel>` pointing
/// at the Stella 1.5B v5 checkpoint, 1024-dim default. This is the
/// signature `mnemosyne` consumes in Phase 4.
///
/// # Errors
///
/// Propagates [`embed::Error`] from the underlying loader.
pub fn stella(
    path: &std::path::Path,
) -> Result<Box<dyn logismos_core::EmbeddingModel>, embed::Error> {
    let m = embed::StellaModel::load(path, &[embed::StellaDim::Dim1024])?;
    Ok(Box::new(m))
}
