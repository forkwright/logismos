//! Checked geometry for sequence-major packed recurrent prefill.
//!
//! The descriptor deliberately has no arbitrary mask input: each admitted
//! sequence contributes one contiguous, nonempty valid prefix to the packed
//! rows. This makes holes and mask/length disagreement unrepresentable.

use snafu::ResultExt;

use crate::error::{PackedPrefillAllocationSnafu, Result, UnsupportedShapeSnafu};

const KERNEL: &str = "packed_prefill";

/// Checked sequence-major packed prefill geometry.
///
/// Rows for every sequence are contiguous and chronological. Committed
/// offsets are operation inputs used to prove the supplied chunk remains
/// inside its caller-owned context; this type grants neither context nor
/// scheduling authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedPrefillPlan {
    sequences: Vec<PackedSequence>,
    total_tokens: usize,
    max_context: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PackedSequence {
    length: usize,
    committed_offset: usize,
    start: usize,
}

impl PackedPrefillPlan {
    /// Validate sequence lengths, committed offsets, and their packed layout.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::UnsupportedShape`] when sequences are absent,
    /// lengths and offsets disagree, a valid prefix is empty, a checked sum
    /// overflows, or a chunk exceeds its supplied context bound.
    pub fn new(lengths: &[usize], committed_offsets: &[usize], max_context: usize) -> Result<Self> {
        if lengths.is_empty() {
            return unsupported("at least one sequence is required");
        }
        if max_context == 0 {
            return unsupported("context bound must be positive");
        }
        if lengths.len() != committed_offsets.len() {
            return unsupported("sequence lengths and committed offsets must have equal counts");
        }

        let mut sequences = Vec::new();
        sequences
            .try_reserve_exact(lengths.len())
            .context(PackedPrefillAllocationSnafu {
                allocation: "packed sequences",
                entries: lengths.len(),
            })?;
        let mut total_tokens = 0_usize;
        for (index, (&length, &offset)) in lengths.iter().zip(committed_offsets).enumerate() {
            if length == 0 {
                return unsupported(format!("sequence {index} has an empty valid prefix"));
            }
            let context_end = offset.checked_add(length).ok_or_else(|| {
                shape_error(format!(
                    "sequence {index} committed offset plus length overflows usize"
                ))
            })?;
            if context_end > max_context {
                return unsupported(format!(
                    "sequence {index} context end {context_end} exceeds bound {max_context}"
                ));
            }
            sequences.push(PackedSequence {
                length,
                committed_offset: offset,
                start: total_tokens,
            });
            total_tokens = total_tokens.checked_add(length).ok_or_else(|| {
                shape_error("packed valid-prefix lengths overflow usize".to_owned())
            })?;
        }
        if total_tokens == 0 {
            return unsupported("packed input must contain at least one valid row");
        }
        std::alloc::Layout::array::<f32>(total_tokens).map_err(|_| {
            shape_error(format!(
                "packed token count {total_tokens} exceeds the Rust layout domain"
            ))
        })?;

        Ok(Self {
            sequences,
            total_tokens,
            max_context,
        })
    }

    /// Return the number of independently staged sequences.
    #[must_use]
    pub const fn sequence_count(&self) -> usize {
        self.sequences.len()
    }

    /// Return the total number of valid sequence-major rows.
    #[must_use]
    pub const fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Return the caller-provided context bound used at admission.
    #[must_use]
    pub const fn max_context(&self) -> usize {
        self.max_context
    }

    /// Return one sequence's nonzero valid-prefix length.
    #[must_use]
    pub fn sequence_length(&self, sequence: usize) -> Option<usize> {
        self.sequences.get(sequence).map(|sequence| sequence.length)
    }

    /// Return one sequence's committed context offset.
    #[must_use]
    pub fn committed_offset(&self, sequence: usize) -> Option<usize> {
        self.sequences
            .get(sequence)
            .map(|sequence| sequence.committed_offset)
    }

    /// Return one sequence's checked packed-row start.
    #[must_use]
    pub fn packed_start(&self, sequence: usize) -> Option<usize> {
        self.sequences.get(sequence).map(|sequence| sequence.start)
    }

    /// Return one sequence's checked packed-row range.
    #[must_use]
    pub fn packed_range(&self, sequence: usize) -> Option<core::ops::Range<usize>> {
        let start = self.packed_start(sequence)?;
        let length = self.sequence_length(sequence)?;
        start.checked_add(length).map(|end| start..end)
    }
}

fn unsupported<T>(msg: impl Into<String>) -> Result<T> {
    Err(shape_error(msg.into()))
}

fn shape_error(msg: String) -> crate::Error {
    UnsupportedShapeSnafu {
        kernel: KERNEL,
        msg,
    }
    .build()
}

#[cfg(test)]
mod tests {
    use super::PackedPrefillPlan;

    #[test]
    fn sequence_major_starts_and_context_ends_are_checked() -> Result<(), String> {
        let plan = PackedPrefillPlan::new(&[2, 1, 3], &[4, 0, 7], 10)
            .map_err(|error| error.to_string())?;

        assert_eq!(plan.sequence_count(), 3);
        assert_eq!(plan.total_tokens(), 6);
        assert_eq!(plan.packed_range(0), Some(0..2));
        assert_eq!(plan.packed_range(1), Some(2..3));
        assert_eq!(plan.packed_range(2), Some(3..6));
        assert_eq!(plan.committed_offset(2), Some(7));
        Ok(())
    }

    #[test]
    fn invalid_prefixes_and_context_ends_refuse() {
        assert!(PackedPrefillPlan::new(&[], &[], 1).is_err());
        assert!(PackedPrefillPlan::new(&[1], &[0], 0).is_err());
        assert!(PackedPrefillPlan::new(&[1], &[], 1).is_err());
        assert!(PackedPrefillPlan::new(&[0], &[0], 1).is_err());
        assert!(PackedPrefillPlan::new(&[2], &[0], 1).is_err());
        assert!(PackedPrefillPlan::new(&[1], &[usize::MAX], usize::MAX).is_err());
        assert!(PackedPrefillPlan::new(&[usize::MAX, 1], &[0, 0], usize::MAX).is_err());
    }
}
