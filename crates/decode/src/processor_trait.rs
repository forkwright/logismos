//! Logit processor trait boundary.

use crate::Result;
use crate::chain::TokenContext;

/// Core contract. A processor reads the decoding context and
/// mutates the logits vector in place.
pub trait LogitProcessor: Send {
    /// Apply this processor to `logits`.
    ///
    /// Vector length stays constant (= vocabulary size); entries may be set to
    /// `f32::NEG_INFINITY` to forbid a token. Implementations must reject
    /// invalid inputs rather than choosing a fallback token.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] when the input, processor configuration, or
    /// resulting logits violate the decode contract.
    ///
    /// # Transactionality
    ///
    /// This trait does not promise rollback for arbitrary implementations. A
    /// chain reports the error and callers must not treat the logits as a
    /// committed decode result after failure.
    fn process(&mut self, logits: &mut [f32], context: &TokenContext<'_>) -> Result<()>;
}
