//! # embed
//!
//! Sentence-transformer heads, Matryoshka projection, and the concrete
//! `StellaModel` that implements [`core::EmbeddingModel`] — the first
//! production model landing in logismos.
//!
//! The Stella pipeline matches the sentence-transformers reference at
//! `/models/stella-1.5b-v5/`:
//!
//! 1. Tokenise via `tokenize::Tokenizer::from_file` (adds EOS by
//!    default, matching the HF `post_processor`).
//! 2. Optionally prepend a role prompt (`s2s_query`, `s2p_query`, or a
//!    caller-supplied string).
//! 3. Forward through `encoders::StellaEncoder` (fp32 CPU in Phase 3).
//! 4. Mean-pool the last-hidden-states using the attention mask.
//! 5. L2-normalise the pooled vector.
//! 6. Project through the selected Matryoshka dense head (fp32
//!    `linear.weight` + `linear.bias`).
//! 7. L2-normalise the projected vector.
//!
//! ## Prompt prefixes
//!
//! The checkpoint bundles its own prompt strings in
//! `config_sentence_transformers.json`. The model parses the file at
//! load time and keeps a map `Prompt -> String`. Consumers pick a
//! prompt role; the model resolves the string.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::doc_markdown,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

pub mod error;
pub mod qwen3;
#[cfg(feature = "stella")]
pub mod stella;

pub use crate::error::{Error, Result};
pub use crate::qwen3::{
    Qwen3EmbeddingCpuRequirements, Qwen3EmbeddingLimits, Qwen3EmbeddingModel, Qwen3RolePrefixes,
};
#[cfg(feature = "stella")]
pub use crate::stella::{StellaDim, StellaModel};

#[cfg(any(feature = "stella", test))]
fn project_stella_head(
    mut pooled: Vec<f32>,
    weight: &[f32],
    bias: &[f32],
    output_dim: usize,
    hidden: usize,
) -> Result<Vec<f32>> {
    use snafu::ResultExt;

    kernels::cpu_f32::l2_normalize_in_place(&mut pooled)
        .context(crate::error::UnitNormalizationSnafu)?;
    let mut projected =
        kernels::cpu_f32::linear_t(&pooled, weight, Some(bias), 1, output_dim, hidden);
    kernels::cpu_f32::l2_normalize_in_place(&mut projected)
        .context(crate::error::UnitNormalizationSnafu)?;
    Ok(projected)
}

#[cfg(any(feature = "stella", test))]
fn map_compute_error(error: &Error) -> logismos_core::EmbeddingError {
    logismos_core::ComputeSnafu {
        message: error.to_string(),
    }
    .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use logismos_core::EmbeddingError;

    #[cfg(feature = "stella")]
    #[test]
    fn stella_dims_include_default_width() {
        assert!(StellaDim::all().contains(&StellaDim::Dim1024));
    }

    #[test]
    fn tiny_stella_head_produces_independently_expected_unit_output() -> Result<()> {
        let output = project_stella_head(vec![3.0, 4.0], &[1.0, 0.0, 0.0, 1.0], &[0.0, 0.0], 2, 2)?;
        assert!((output[0] - 0.6).abs() <= 1e-6);
        assert!((output[1] - 0.8).abs() <= 1e-6);
        let norm = output
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() <= kernels::cpu_f32::UNIT_NORM_TOLERANCE);
        Ok(())
    }

    #[test]
    fn tiny_stella_zero_head_maps_refusal_to_public_compute_error() {
        let result = project_stella_head(vec![3.0, 4.0], &[0.0, 0.0, 0.0, 0.0], &[0.0, 0.0], 2, 2)
            .map_err(|error| map_compute_error(&error));
        assert!(matches!(
            result,
            Err(EmbeddingError::Compute { ref message, .. })
                if message.contains("zero L2 norm")
        ));
    }
}
