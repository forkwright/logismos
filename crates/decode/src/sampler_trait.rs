//! Sampler trait boundary.

/// Core contract. Samplers own their RNG state internally or through
/// their caller; the trait stays RNG-agnostic so greedy + multinomial
/// share the same shape.
pub trait Sampler: Send {
    /// Pick a token id from `logits`. Implementations assume the
    /// logits have already been processed (masked, scaled, ...).
    fn sample(&mut self, logits: &[f32]) -> u32;
}
