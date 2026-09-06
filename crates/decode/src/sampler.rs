//! Samplers. Given a (possibly processed) logits slice, emit a token id.

use crate::Result;
use crate::error::AllLogitsMaskedSnafu;
use crate::sampler_trait::Sampler;
use crate::validation::{reserve, token_index, validate_logits};
use rand::{Rng, RngExt};

/// Argmax sampler. No RNG.
#[derive(Debug, Clone, Copy, Default)]
pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        crate::greedy(logits)
    }
}

/// Multinomial sampler. Draws from softmax(logits).
///
/// The caller owns and supplies the RNG, so paired tests can reproduce a
/// stream with the same seed. `SmallRng` is non-portable across rand releases;
/// golden fixtures must use fixed input draws instead of its sampled output.
pub struct MultinomialSampler<R: Rng + Send> {
    rng: R,
}

impl<R: Rng + Send> MultinomialSampler<R> {
    /// Wrap an RNG.
    pub fn new(rng: R) -> Self {
        Self { rng }
    }
}

impl<R: Rng + Send> Sampler for MultinomialSampler<R> {
    fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        validate_logits(logits)?;

        // Convert to probabilities, retaining negative-infinity masks as zero.
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probabilities = Vec::new();
        reserve(
            &mut probabilities,
            "multinomial probabilities",
            logits.len(),
        )?;
        let mut sum = 0.0f64;
        for logit in logits {
            let probability = if logit.is_finite() {
                (f64::from(*logit) - f64::from(max)).exp()
            } else {
                0.0
            };
            sum += probability;
            probabilities.push(probability);
        }
        // Input validation guarantees at least one finite logit; subtracting
        // the finite maximum makes its exponential one, so this is nonzero.
        for probability in &mut probabilities {
            *probability /= sum;
        }
        let mut draw: f64 = self.rng.random_range(0.0f64..1.0f64);
        for (index, probability) in probabilities.iter().copied().enumerate() {
            if draw < probability {
                return token_index(index);
            }
            draw -= probability;
        }

        // Floating-point subtraction can leave a tiny residual after every
        // probability was consumed. Select the final unmasked token, not an
        // arbitrary vocabulary position.
        for (index, probability) in probabilities.iter().copied().enumerate().rev() {
            if probability > 0.0 {
                return token_index(index);
            }
        }
        AllLogitsMaskedSnafu.fail()
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    #[test]
    fn greedy_picks_argmax() -> Result<()> {
        let mut s = GreedySampler;
        assert_eq!(s.sample(&[0.1, 0.3, 0.2])?, 1);
        assert_eq!(s.sample(&[5.0, 1.0, 1.0, 1.0])?, 0);
        Ok(())
    }

    #[test]
    fn multinomial_is_reproducible_with_seed() -> Result<()> {
        let r1 = SmallRng::seed_from_u64(42);
        let r2 = SmallRng::seed_from_u64(42);
        let mut s1 = MultinomialSampler::new(r1);
        let mut s2 = MultinomialSampler::new(r2);
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        for _ in 0..8 {
            assert_eq!(s1.sample(&logits)?, s2.sample(&logits)?);
        }
        Ok(())
    }

    #[test]
    fn multinomial_picks_dominant_mass_with_peaked_dist() -> Result<()> {
        let r = SmallRng::seed_from_u64(7);
        let mut s = MultinomialSampler::new(r);
        // One token absolutely dominates.
        let logits = vec![10.0, 0.0, 0.0, 0.0];
        let mut hits = 0;
        for _ in 0..100 {
            if s.sample(&logits)? == 0 {
                hits += 1;
            }
        }
        // Expectation ~> 99%; floor at 90 so the test is robust.
        assert!(hits > 90);
        Ok(())
    }

    #[test]
    fn samplers_reject_invalid_logits() {
        let mut greedy = GreedySampler;
        let mut multinomial = MultinomialSampler::new(SmallRng::seed_from_u64(7));
        for logits in [
            Vec::new(),
            vec![f32::NAN],
            vec![f32::INFINITY],
            vec![f32::NEG_INFINITY, f32::NEG_INFINITY],
        ] {
            assert!(greedy.sample(&logits).is_err());
            assert!(multinomial.sample(&logits).is_err());
        }
    }

    #[test]
    fn samplers_preserve_negative_infinity_masks() -> Result<()> {
        let logits = [f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY];
        let mut greedy = GreedySampler;
        let mut multinomial = MultinomialSampler::new(SmallRng::seed_from_u64(7));
        assert_eq!(greedy.sample(&logits)?, 1);
        for _ in 0..8 {
            assert_eq!(multinomial.sample(&logits)?, 1);
        }
        Ok(())
    }
}
