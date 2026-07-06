//! Format-agnostic tensor archive trait.

use crate::{Result, TensorView};

/// Format-agnostic tensor-archive trait.
pub trait WeightProvider {
    /// Return a view over the tensor named `name`.
    fn get(&self, name: &str) -> Result<TensorView<'_>>;
    /// Enumerate every tensor name in archive order.
    fn names(&self) -> Vec<String>;
    /// Number of tensors in the archive.
    fn len(&self) -> usize;
    /// True when the archive holds zero tensors.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
