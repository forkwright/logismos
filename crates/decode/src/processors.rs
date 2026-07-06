//! Logit processors — mutate a `&mut [f32]` logits vector in place.
//!
//! Every processor is a small self-contained transform. Chains
//! (`DecodeChain`) own the ordering; processors themselves are
//! order-agnostic individually.

use crate::chain::TokenContext;
use crate::processor_trait::LogitProcessor;

/// Divide every logit by `temperature`. `temperature == 0.0` collapses
/// to argmax (handled by `GreedySampler`). Temperatures below a
/// positive epsilon floor are clamped — a zero-pass here would produce
/// `inf` logits and poison downstream math.
#[derive(Debug, Clone, Copy)]
pub struct TemperatureScale(pub f32);

impl LogitProcessor for TemperatureScale {
    fn process(&mut self, logits: &mut [f32], _ctx: &TokenContext<'_>) {
        let t = self.0.max(1e-6);
        if (t - 1.0).abs() < 1e-9 {
            return;
        }
        for l in logits.iter_mut() {
            *l /= t;
        }
    }
}

/// Keep the `k` highest-scoring tokens; mask the rest to `-inf`.
#[derive(Debug, Clone, Copy)]
pub struct TopK(pub usize);

impl LogitProcessor for TopK {
    fn process(&mut self, logits: &mut [f32], _ctx: &TokenContext<'_>) {
        let k = self.0;
        if k == 0 || k >= logits.len() {
            return;
        }
        // Partial-sort: find the kth-largest score, mask everything below.
        let mut sorted: Vec<f32> = logits.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
        // k >= 1 (the k==0 branch returned) and k < logits.len(),
        // so the (k-1)th sorted entry exists. If the invariant ever
        // breaks, fail open (leave logits untouched).
        let Some(&threshold) = sorted.get(k.saturating_sub(1)) else {
            return;
        };
        for l in logits.iter_mut() {
            if *l < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    }
}

/// Nucleus sampling — keep the smallest prefix whose cumulative
/// softmax probability ≥ `p`. Rest mask to `-inf`.
#[derive(Debug, Clone, Copy)]
pub struct TopP(pub f32);

impl LogitProcessor for TopP {
    fn process(&mut self, logits: &mut [f32], _ctx: &TokenContext<'_>) {
        let p = self.0.clamp(0.0, 1.0);
        if p >= 1.0 || logits.is_empty() {
            return;
        }
        let probs = softmax(logits);
        // Sort indices by descending prob. Indices are all in-bounds
        // for `probs` by construction; `.get` keeps the invariant
        // explicit.
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_by(|&a, &b| {
            let pb = probs.get(b).copied().unwrap_or(0.0);
            let pa = probs.get(a).copied().unwrap_or(0.0);
            pb.partial_cmp(&pa).unwrap_or(core::cmp::Ordering::Equal)
        });
        let mut cum = 0.0f32;
        let mut keep = vec![false; logits.len()];
        for &i in &idx {
            if let Some(slot) = keep.get_mut(i) {
                *slot = true;
            }
            cum += probs.get(i).copied().unwrap_or(0.0);
            if cum >= p {
                break;
            }
        }
        for (i, l) in logits.iter_mut().enumerate() {
            if !keep.get(i).copied().unwrap_or(true) {
                *l = f32::NEG_INFINITY;
            }
        }
    }
}

/// Min-P filter — mask tokens whose probability is below
/// `min_p × max_probability`.
#[derive(Debug, Clone, Copy)]
pub struct MinP(pub f32);

impl LogitProcessor for MinP {
    fn process(&mut self, logits: &mut [f32], _ctx: &TokenContext<'_>) {
        let p = self.0.clamp(0.0, 1.0);
        if p <= 0.0 || logits.is_empty() {
            return;
        }
        let probs = softmax(logits);
        let max_p = probs.iter().copied().fold(0.0f32, f32::max);
        let threshold = p * max_p;
        for (i, l) in logits.iter_mut().enumerate() {
            let pi = probs.get(i).copied().unwrap_or(0.0);
            if pi < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    }
}

/// Downweight recently seen tokens. `penalty > 1.0` suppresses;
/// `penalty < 1.0` boosts. See Keskar et al. 2019 CTRL §4.1.
#[derive(Debug, Clone)]
pub struct RepetitionPenalty {
    /// Recent token ids to penalise.
    pub tokens: Vec<u32>,
    /// Divide the logit by `penalty` when `logit > 0`; multiply when
    /// `logit < 0`. Mirrors HF Transformers semantics.
    pub penalty: f32,
}

impl LogitProcessor for RepetitionPenalty {
    fn process(&mut self, logits: &mut [f32], _ctx: &TokenContext<'_>) {
        if (self.penalty - 1.0).abs() < 1e-9 {
            return;
        }
        let penalty = self.penalty;
        for &t in &self.tokens {
            let Ok(i) = usize::try_from(t) else {
                continue;
            };
            let Some(l) = logits.get_mut(i) else {
                continue;
            };
            if *l > 0.0 {
                *l /= penalty;
            } else {
                *l *= penalty;
            }
        }
    }
}

/// Typical-sampling stub. No-op in Phase 2; the full impl lands in
/// Phase 7 per the PLAN (Meister et al. 2022 typical-p sampling).
#[derive(Debug, Clone, Copy)]
pub struct TypicalSampling(pub f32);

impl LogitProcessor for TypicalSampling {
    fn process(&mut self, _logits: &mut [f32], _ctx: &TokenContext<'_>) {
        // Intentional no-op — Phase-7 hook.
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for e in &mut exps {
            *e /= sum;
        }
    }
    exps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> TokenContext<'static> {
        TokenContext {
            prev_tokens: &[],
            step: 0,
        }
    }

    #[test]
    fn temperature_divides() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let mut t = TemperatureScale(2.0);
        t.process(&mut logits, &ctx());
        assert_eq!(logits, vec![0.5, 1.0, 1.5]);
    }

    #[test]
    fn temperature_one_is_identity() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let mut t = TemperatureScale(1.0);
        t.process(&mut logits, &ctx());
        assert_eq!(logits, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn top_k_masks_below_kth() {
        let mut logits = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let mut p = TopK(2);
        p.process(&mut logits, &ctx());
        // Top 2 are 5.0 and 4.0. Others -> -inf.
        assert_eq!(logits[3], 5.0);
        assert_eq!(logits[4], 4.0);
        assert!(logits[0].is_infinite() && logits[0] < 0.0);
        assert!(logits[1].is_infinite() && logits[1] < 0.0);
        assert!(logits[2].is_infinite() && logits[2] < 0.0);
    }

    #[test]
    fn top_p_nucleus_keeps_dominant_prefix() {
        // Highly peaked distribution: logit 10.0 dominates.
        let mut logits = vec![10.0, 0.0, 0.0, 0.0];
        let mut p = TopP(0.5);
        p.process(&mut logits, &ctx());
        assert_eq!(logits[0], 10.0);
        for v in &logits[1..] {
            assert!(v.is_infinite() && *v < 0.0);
        }
    }

    #[test]
    fn min_p_masks_tails() {
        // logit 5 dominates → its prob is near 1. With min_p = 0.5,
        // only indices within 0.5× max probability survive.
        let mut logits = vec![5.0, 0.0, 0.0, 0.0];
        let mut p = MinP(0.5);
        p.process(&mut logits, &ctx());
        assert_eq!(logits[0], 5.0);
        for v in &logits[1..] {
            assert!(v.is_infinite() && *v < 0.0);
        }
    }

    #[test]
    fn repetition_penalty_suppresses_positive() {
        let mut logits = vec![2.0, 4.0, 1.0];
        let mut r = RepetitionPenalty {
            tokens: vec![1],
            penalty: 2.0,
        };
        r.process(&mut logits, &ctx());
        assert_eq!(logits[1], 2.0);
    }

    #[test]
    fn repetition_penalty_boosts_negative() {
        let mut logits = vec![-2.0, -4.0, 1.0];
        let mut r = RepetitionPenalty {
            tokens: vec![1],
            penalty: 2.0,
        };
        r.process(&mut logits, &ctx());
        assert_eq!(logits[1], -8.0);
    }

    #[test]
    fn typical_sampling_is_noop_in_phase_2() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let mut p = TypicalSampling(0.95);
        p.process(&mut logits, &ctx());
        assert_eq!(logits, vec![1.0, 2.0, 3.0]);
    }
}
