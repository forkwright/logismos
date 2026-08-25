//! Samplers. Given a (possibly processed) logits slice, emit a token id.

use rand::Rng;

use crate::sampler_trait::Sampler;

/// Argmax sampler. No RNG.
#[derive(Debug, Clone, Copy, Default)]
pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        crate::greedy(logits)
    }
}

/// Multinomial sampler. Draws from softmax(logits). RNG is owned by
/// the caller and supplied at construction so reproducible streams
/// stay trivial (seed a `StdRng`, pass it in).
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
    fn sample(&mut self, logits: &[f32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }
        // Convert to probabilities, mask -inf to 0.
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = logits
            .iter()
            .map(|&l| if l.is_finite() { (l - max).exp() } else { 0.0 })
            .collect();
        let sum: f32 = exps.iter().sum();
        if sum <= 0.0 {
            return crate::greedy(logits);
        }
        for e in &mut exps {
            *e /= sum;
        }
        let mut u: f32 = self.rng.random_range(0.0f32..1.0f32);
        for (i, &p) in exps.iter().enumerate() {
            if u < p {
                return u32::try_from(i).unwrap_or(u32::MAX);
            }
            u -= p;
        }
        u32::try_from(exps.len().saturating_sub(1)).unwrap_or(u32::MAX)
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let mut s = GreedySampler;
        assert_eq!(s.sample(&[0.1, 0.3, 0.2]), 1);
        assert_eq!(s.sample(&[5.0, 1.0, 1.0, 1.0]), 0);
    }

    #[test]
    fn multinomial_is_reproducible_with_seed() {
        let r1 = SmallRng::seed_from_u64(42);
        let r2 = SmallRng::seed_from_u64(42);
        let mut s1 = MultinomialSampler::new(r1);
        let mut s2 = MultinomialSampler::new(r2);
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        for _ in 0..8 {
            assert_eq!(s1.sample(&logits), s2.sample(&logits));
        }
    }

    #[test]
    fn multinomial_picks_dominant_mass_with_peaked_dist() {
        let r = SmallRng::seed_from_u64(7);
        let mut s = MultinomialSampler::new(r);
        // One token absolutely dominates.
        let logits = vec![10.0, 0.0, 0.0, 0.0];
        let mut hits = 0;
        for _ in 0..100 {
            if s.sample(&logits) == 0 {
                hits += 1;
            }
        }
        // Expectation ~> 99%; floor at 90 so the test is robust.
        assert!(hits > 90);
    }
}
