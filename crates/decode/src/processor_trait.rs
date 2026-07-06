//! Logit processor trait boundary.

use crate::chain::TokenContext;

/// Core contract. A processor reads the decoding context and
/// mutates the logits vector in place.
pub trait LogitProcessor: Send {
    /// Apply this processor to `logits`. Vector length stays constant
    /// (= vocab size); entries may be set to `f32::NEG_INFINITY` to
    /// forbid a token.
    fn process(&mut self, logits: &mut [f32], context: &TokenContext<'_>);
}
