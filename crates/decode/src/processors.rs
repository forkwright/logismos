//! Checked logit processors.

use std::num::NonZeroUsize;

use crate::Result;
use crate::chain::TokenContext;
use crate::error::{AllLogitsMaskedSnafu, NonFiniteLogitSnafu, UnsupportedProcessorSnafu};
use crate::processor_trait::LogitProcessor;
use crate::validation::{
    reserve, validate_logits, validate_positive, validate_probability, validate_token,
};

#[derive(Debug, Clone, Copy)]
struct Probability(f32);

impl Probability {
    fn new(name: &'static str, value: f32) -> Result<Self> {
        validate_probability(name, value)?;
        Ok(Self(value))
    }

    const fn get(self) -> f32 {
        self.0
    }
}

/// Divide every finite logit by a positive temperature.
#[derive(Debug, Clone, Copy)]
pub struct TemperatureScale(f32);

impl TemperatureScale {
    /// Construct a finite, positive temperature scale.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] when `temperature` is non-finite or not
    /// positive.
    pub fn new(temperature: f32) -> Result<Self> {
        validate_positive("temperature", temperature)?;
        Ok(Self(temperature))
    }
}

impl LogitProcessor for TemperatureScale {
    fn process(&mut self, logits: &mut [f32], _context: &TokenContext<'_>) -> Result<()> {
        validate_logits(logits)?;
        for (index, logit) in logits.iter().copied().enumerate() {
            if logit.is_finite() && !(logit / self.0).is_finite() {
                return NonFiniteLogitSnafu {
                    index,
                    kind: "a non-finite temperature-scaled value",
                }
                .fail();
            }
        }
        for logit in logits {
            *logit /= self.0;
        }
        Ok(())
    }
}

/// Keep the `k` highest-scoring tokens; mask the rest to negative infinity.
///
/// Owns a reusable scratch buffer so the per-step partial sort does not
/// allocate a vocabulary-sized vector on every decode step.
#[derive(Debug, Clone)]
pub struct TopK {
    k: NonZeroUsize,
    scratch: Vec<f32>,
}

impl TopK {
    /// Construct a top-k processor.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] when `k` is zero.
    pub fn new(k: usize) -> Result<Self> {
        let k = NonZeroUsize::new(k).ok_or_else(|| {
            crate::error::InvalidParameterSnafu {
                name: "top_k",
                rule: "must be greater than zero",
            }
            .build()
        })?;
        Ok(Self {
            k,
            scratch: Vec::new(),
        })
    }
}

impl LogitProcessor for TopK {
    fn process(&mut self, logits: &mut [f32], _context: &TokenContext<'_>) -> Result<()> {
        validate_logits(logits)?;
        let k = self.k.get();
        if k >= logits.len() {
            return Ok(());
        }

        self.scratch.clear();
        reserve(&mut self.scratch, "top-k scratch", logits.len())?;
        self.scratch.extend_from_slice(logits);
        self.scratch.sort_by(|left, right| right.total_cmp(left));
        let threshold = self.scratch[k - 1];
        for logit in logits {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }
}

/// Nucleus sampling — keep the smallest prefix whose cumulative probability
/// meets the configured threshold, then mask the rest to negative infinity.
#[derive(Debug, Clone, Copy)]
pub struct TopP(Probability);

impl TopP {
    /// Construct a nucleus-sampling processor with a bounded probability.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] when `probability` is non-finite or outside
    /// `0.0..=1.0`.
    pub fn new(probability: f32) -> Result<Self> {
        Ok(Self(Probability::new("top_p", probability)?))
    }
}

impl LogitProcessor for TopP {
    fn process(&mut self, logits: &mut [f32], _context: &TokenContext<'_>) -> Result<()> {
        let probabilities = probabilities(logits)?;
        if self.0.get() == 1.0 {
            return Ok(());
        }

        let mut indices = Vec::new();
        reserve(&mut indices, "top-p indices", probabilities.len())?;
        indices.extend(0..probabilities.len());
        indices.sort_by(|left, right| probabilities[*right].total_cmp(&probabilities[*left]));

        let mut keep = Vec::new();
        reserve(&mut keep, "top-p keep mask", logits.len())?;
        keep.resize(logits.len(), false);
        let mut cumulative = 0.0f64;
        for index in indices {
            keep[index] = true;
            cumulative += probabilities[index];
            if cumulative >= f64::from(self.0.get()) {
                break;
            }
        }
        for (index, logit) in logits.iter_mut().enumerate() {
            if !keep[index] {
                *logit = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }
}

/// Min-p filter — mask tokens below `min_p × max_probability`.
#[derive(Debug, Clone, Copy)]
pub struct MinP(Probability);

impl MinP {
    /// Construct a min-p processor with a bounded probability.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] when `probability` is non-finite or outside
    /// `0.0..=1.0`.
    pub fn new(probability: f32) -> Result<Self> {
        Ok(Self(Probability::new("min_p", probability)?))
    }
}

impl LogitProcessor for MinP {
    fn process(&mut self, logits: &mut [f32], _context: &TokenContext<'_>) -> Result<()> {
        let probabilities = probabilities(logits)?;
        if self.0.get() == 0.0 {
            return Ok(());
        }
        let max_probability = probabilities.iter().copied().fold(0.0f64, f64::max);
        let threshold = f64::from(self.0.get()) * max_probability;
        for (index, logit) in logits.iter_mut().enumerate() {
            if probabilities[index] < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }
}

/// Downweight recently seen tokens.
#[derive(Debug, Clone)]
pub struct RepetitionPenalty {
    tokens: Vec<u32>,
    penalty: f32,
}

impl RepetitionPenalty {
    /// Construct a finite, positive repetition penalty over explicit history.
    ///
    /// Values greater than one suppress positive logits and values below one
    /// boost them, following the documented repetition-penalty policy.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] when `penalty` is non-finite or not positive.
    pub fn new(tokens: Vec<u32>, penalty: f32) -> Result<Self> {
        validate_positive("repetition_penalty", penalty)?;
        Ok(Self { tokens, penalty })
    }
}

impl LogitProcessor for RepetitionPenalty {
    fn process(&mut self, logits: &mut [f32], _context: &TokenContext<'_>) -> Result<()> {
        validate_logits(logits)?;
        let mut indices = Vec::new();
        reserve(
            &mut indices,
            "repetition-penalty indices",
            self.tokens.len(),
        )?;
        for token_id in &self.tokens {
            let index = validate_token(*token_id, logits.len())?;
            indices.push(index);
        }
        indices.sort_unstable();
        indices.dedup();
        for index in &indices {
            let logit = logits[*index];
            let adjusted = if logit > 0.0 {
                logit / self.penalty
            } else {
                logit * self.penalty
            };
            if logit.is_finite() && !adjusted.is_finite() {
                return NonFiniteLogitSnafu {
                    index: *index,
                    kind: "a non-finite repetition-penalty value",
                }
                .fail();
            }
        }
        for index in indices {
            let logit = &mut logits[index];
            if *logit > 0.0 {
                *logit /= self.penalty;
            } else {
                *logit *= self.penalty;
            }
        }
        Ok(())
    }
}

/// Typical sampling configuration.
///
/// The type remains available for planned policy wiring, but execution refuses
/// it until a real implementation is supplied.
#[derive(Debug, Clone, Copy)]
pub struct TypicalSampling(Probability);

impl TypicalSampling {
    /// Construct a typical-sampling configuration with a bounded probability.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] when `probability` is non-finite or outside
    /// `0.0..=1.0`.
    pub fn new(probability: f32) -> Result<Self> {
        Ok(Self(Probability::new("typical_p", probability)?))
    }

    /// Return the validated typical-sampling probability.
    #[must_use]
    pub const fn probability(&self) -> f32 {
        self.0.get()
    }
}

impl LogitProcessor for TypicalSampling {
    fn process(&mut self, logits: &mut [f32], _context: &TokenContext<'_>) -> Result<()> {
        validate_logits(logits)?;
        UnsupportedProcessorSnafu {
            processor: "typical sampling",
        }
        .fail()
    }
}

fn probabilities(logits: &[f32]) -> Result<Vec<f64>> {
    validate_logits(logits)?;
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = Vec::new();
    reserve(&mut probabilities, "softmax probabilities", logits.len())?;
    let mut sum = 0.0f64;
    for logit in logits {
        let probability = if logit.is_finite() {
            f64::from((*logit - maximum).exp())
        } else {
            0.0
        };
        sum += probability;
        probabilities.push(probability);
    }
    if sum <= 0.0 || !sum.is_finite() {
        return AllLogitsMaskedSnafu.fail();
    }
    for probability in &mut probabilities {
        *probability /= sum;
    }
    Ok(probabilities)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> TokenContext<'static> {
        TokenContext {
            prev_tokens: &[],
            step: 0,
        }
    }

    #[test]
    fn temperature_divides() -> Result<()> {
        let mut logits = vec![1.0, 2.0, 3.0];
        let mut processor = TemperatureScale::new(2.0)?;
        processor.process(&mut logits, &context())?;
        assert_eq!(logits, vec![0.5, 1.0, 1.5]);
        Ok(())
    }

    #[test]
    fn top_k_masks_below_threshold() -> Result<()> {
        let mut logits = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let mut processor = TopK::new(2)?;
        processor.process(&mut logits, &context())?;
        assert_eq!(logits[3], 5.0);
        assert_eq!(logits[4], 4.0);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        Ok(())
    }

    #[test]
    fn top_k_reuses_existing_scratch_capacity() -> Result<()> {
        let mut processor = TopK::new(2)?;
        processor.scratch.reserve(64);
        let pointer = processor.scratch.as_ptr();
        let capacity = processor.scratch.capacity();
        let mut first = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        processor.process(&mut first, &context())?;
        assert_eq!(processor.scratch.as_ptr(), pointer);
        assert!(processor.scratch.capacity() >= capacity);

        let mut second = vec![2.0, 1.0, 4.0, 3.0, 0.0];
        processor.process(&mut second, &context())?;
        assert_eq!(processor.scratch.as_ptr(), pointer);
        Ok(())
    }

    #[test]
    fn top_p_and_min_p_mask_tails() -> Result<()> {
        let mut top_p_logits = vec![10.0, 0.0, 0.0, 0.0];
        TopP::new(0.5)?.process(&mut top_p_logits, &context())?;
        assert_eq!(top_p_logits[0], 10.0);
        assert!(
            top_p_logits[1..]
                .iter()
                .all(|value| *value == f32::NEG_INFINITY)
        );

        let mut min_p_logits = vec![5.0, 0.0, 0.0, 0.0];
        MinP::new(0.5)?.process(&mut min_p_logits, &context())?;
        assert_eq!(min_p_logits[0], 5.0);
        assert!(
            min_p_logits[1..]
                .iter()
                .all(|value| *value == f32::NEG_INFINITY)
        );
        Ok(())
    }

    #[test]
    fn repetition_penalty_rejects_unknown_token_without_mutation() -> Result<()> {
        let mut logits = vec![2.0, 4.0, 1.0];
        let before = logits.clone();
        let mut processor = RepetitionPenalty::new(vec![1, 9], 2.0)?;
        assert!(processor.process(&mut logits, &context()).is_err());
        assert_eq!(logits, before);
        Ok(())
    }

    #[test]
    fn repetition_penalty_applies_each_token_once() -> Result<()> {
        let mut logits = vec![2.0, 4.0, 1.0];
        let mut processor = RepetitionPenalty::new(vec![1, 1], 2.0)?;
        processor.process(&mut logits, &context())?;
        assert_eq!(logits[1], 2.0);
        Ok(())
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        assert!(TemperatureScale::new(0.0).is_err());
        assert!(TemperatureScale::new(f32::NAN).is_err());
        assert!(TopK::new(0).is_err());
        assert!(TopP::new(1.1).is_err());
        assert!(MinP::new(f32::INFINITY).is_err());
        assert!(RepetitionPenalty::new(vec![], -1.0).is_err());
    }

    #[test]
    fn typical_sampling_refuses_without_mutation() -> Result<()> {
        let mut logits = vec![1.0, 2.0];
        let before = logits.clone();
        let mut processor = TypicalSampling::new(0.95)?;
        assert!(processor.process(&mut logits, &context()).is_err());
        assert_eq!(logits, before);
        Ok(())
    }

    #[test]
    fn processors_reject_invalid_logits() -> Result<()> {
        let mut processor = TemperatureScale::new(1.0)?;
        for logits in [
            Vec::new(),
            vec![f32::NAN],
            vec![f32::INFINITY],
            vec![f32::NEG_INFINITY, f32::NEG_INFINITY],
        ] {
            let mut logits = logits;
            assert!(processor.process(&mut logits, &context()).is_err());
        }
        Ok(())
    }
}
