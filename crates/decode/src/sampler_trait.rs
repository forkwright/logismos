//! Sampler trait boundary.

use crate::Result;

/// Core contract. Samplers own their RNG state internally or through their
/// caller; the trait stays RNG-agnostic so greedy + multinomial share the same
/// shape.
pub trait Sampler: Send {
    /// Pick a token id from `logits`. Implementations assume the
    /// logits have already been processed (masked, scaled, ...), but must
    /// still validate them because samplers are independently callable.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] instead of substituting a sentinel token when
    /// no valid candidate exists.
    fn sample(&mut self, logits: &[f32]) -> Result<u32>;
}
