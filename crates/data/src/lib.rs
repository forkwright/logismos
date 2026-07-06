//! # data
//!
//! Dataloaders, tokenisation pipelines, streaming dataset iteration
//! for training and evaluation.
//!
//! Phase 0 scaffold. No functional code yet.
//!
//! ## Responsibility
//!
//! - Streaming datasets (JSONL, Parquet, HF Datasets format)
//! - Tokenize-and-pack pipelines
//! - Bucketed batching
//! - Evaluation harness for benchmark sets (MTEB, BEIR, `LongMemEval`)
//!
//! Lands in Phase 10. Consumer: `melete`.
#![deny(missing_docs)]

#[cfg(test)]
const CRATE_NAME: &str = "data";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_identity_matches_role() {
        assert_eq!(env!("CARGO_PKG_NAME"), CRATE_NAME);
    }
}
