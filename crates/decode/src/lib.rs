//! # decode
//!
//! Logit processors + samplers.
//!
//! Phase 2 ships the pure-sampling path: `TemperatureScale`,
//! `TopK`, `TopP`, `MinP`, `RepetitionPenalty`, plus `GreedySampler`
//! and `MultinomialSampler`. Grammar-constrained decoding is deferred
//! to Phase 12 where the PDA engine lands behind the same
//! [`LogitProcessor`] trait.
//!
//! ## Shape
//!
//! ```text
//! logits: &mut [f32]  ──[LogitProcessor chain]──▶  ─▶ Sampler ─▶ u32
//! ```
//!
//! A chain is a `Vec<Box<dyn LogitProcessor>>`; processors mutate the
//! logits vector in place, then a [`Sampler`] emits a token id.
//! Ordering matters (temperature first is the common idiom;
//! top-p second; sampler last). The [`DecodeChain`] type composes
//! processors + sampler and owns the RNG.
//!
//! ## Phase-2 limits
//!
//! - CPU only (the logits vector is a plain `&mut [f32]`). A HIP-side
//!   sampler is Phase 7 work when speculative decoding lands — at that
//!   point the logits will live on device and `LogitProcessor` will
//!   need a device flavour.
//! - `TypicalSampling` is stubbed (no-op) per the PLAN: the full
//!   impl requires Phase 7 context.
//! - Grammar: trait-only hook; no impl.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::needless_pass_by_value,
    clippy::float_cmp
)]

pub mod chain;
pub mod error;
pub mod processor_trait;
pub mod processors;
pub mod sampler;
pub mod sampler_trait;

pub use crate::chain::{DecodeChain, TokenContext};
pub use crate::error::{Error, Result};
pub use crate::processor_trait::LogitProcessor;
pub use crate::processors::{
    MinP, RepetitionPenalty, TemperatureScale, TopK, TopP, TypicalSampling,
};
pub use crate::sampler::{GreedySampler, MultinomialSampler};
pub use crate::sampler_trait::Sampler;

/// Convenience — greedy argmax over a logits slice.
///
/// Equivalent to `GreedySampler.sample(logits)` but usable without
/// instantiating the chain machinery. Saturates to `u32::MAX` if the
/// argmax index does not fit in `u32` (vocabularies are designed to
/// stay well under `u32::MAX`).
#[must_use]
pub fn greedy(logits: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best {
            best = v;
            best_i = i;
        }
    }
    u32::try_from(best_i).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_returns_first_max_index() {
        assert_eq!(greedy(&[0.0, 2.0, 2.0]), 1);
    }
}
