//! Executor-owned logical CPU allocation requirements for bounded Qwen3 profiles.

use loader::gguf::ArtifactDigest;
use snafu::ResultExt;

use crate::Result;
use crate::error::{Qwen3CpuSnafu, Qwen3ExecutionSnafu};
use crate::matrix::CheckedMatrix;

/// Checked logical CPU backing envelope for one Qwen3 executor.
///
/// The reported fields count exact requested `Vec<f32>` capacities composed
/// into a conservative phase upper bound. They exclude allocator capacity and
/// metadata, stack values, non-`f32` structures, process RSS, tokenizer and
/// template storage, GPU memory, and immutable serialized GGUF backing.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Qwen3CpuRequirements {
    artifact_digest: ArtifactDigest,
    serialized_backing_bytes: u64,
    max_context: usize,
    workspace_upper_bound_bytes: u64,
    returned_output_bytes: u64,
    logical_f32_upper_bound_bytes: u64,
}

impl Qwen3CpuRequirements {
    pub(crate) fn embedding(
        artifact_digest: ArtifactDigest,
        serialized_backing_bytes: u64,
        max_context: usize,
        shape: Qwen3AllocationShape,
    ) -> Result<Self> {
        Self::from_shape(
            artifact_digest,
            serialized_backing_bytes,
            max_context,
            shape.embedding_workspace_upper_bound()?,
            shape.returned_hidden,
        )
    }

    pub(crate) fn rank(
        artifact_digest: ArtifactDigest,
        serialized_backing_bytes: u64,
        max_context: usize,
        shape: Qwen3AllocationShape,
    ) -> Result<Self> {
        let body_envelope = checked_add(
            shape.embedding_workspace_upper_bound()?,
            shape.returned_hidden,
            "rank body complete logical envelope",
        )?;
        let rank_projection = checked_add(
            shape.returned_hidden,
            shape.rank_head_projection,
            "rank terminal hidden handoff and classifier projection",
        )?;
        Self::from_shape(
            artifact_digest,
            serialized_backing_bytes,
            max_context,
            body_envelope.max(rank_projection),
            0,
        )
    }

    fn from_shape(
        artifact_digest: ArtifactDigest,
        serialized_backing_bytes: u64,
        max_context: usize,
        workspace_upper_bound: usize,
        returned_output: usize,
    ) -> Result<Self> {
        let logical_f32_upper_bound = checked_add(
            workspace_upper_bound,
            returned_output,
            "Qwen3 logical f32 upper bound",
        )?;
        Ok(Self {
            artifact_digest,
            serialized_backing_bytes,
            max_context,
            workspace_upper_bound_bytes: f32_bytes(workspace_upper_bound)?,
            returned_output_bytes: f32_bytes(returned_output)?,
            logical_f32_upper_bound_bytes: f32_bytes(logical_f32_upper_bound)?,
        })
    }

    /// Return the digest identifying the verified serialized bytes bound here.
    #[must_use]
    pub const fn artifact_digest(self) -> ArtifactDigest {
        self.artifact_digest
    }

    /// Return the immutable verified GGUF backing length, excluded from logical f32 backing.
    #[must_use]
    pub const fn serialized_backing_bytes(self) -> u64 {
        self.serialized_backing_bytes
    }

    /// Return the verified caller context bound used for this envelope.
    #[must_use]
    pub const fn max_context(self) -> usize {
        self.max_context
    }

    /// Return the conservative upper bound for transient `Vec<f32>` backing.
    #[must_use]
    pub const fn workspace_upper_bound_bytes(self) -> u64 {
        self.workspace_upper_bound_bytes
    }

    /// Return returned heap-backed f32 output bytes.
    ///
    /// Embedding execution returns one hidden `Vec<f32>`; rank returns a
    /// stack `[f32; 2]` and therefore reports zero here.
    #[must_use]
    pub const fn returned_output_bytes(self) -> u64 {
        self.returned_output_bytes
    }

    /// Return workspace plus returned heap-backed f32 output bytes.
    #[must_use]
    pub const fn logical_f32_upper_bound_bytes(self) -> u64 {
        self.logical_f32_upper_bound_bytes
    }
}

/// One private shape authority for Qwen3 body execution and its requirements report.
#[derive(Clone, Copy)]
pub(crate) struct Qwen3AllocationShape {
    pub(crate) tokens: usize,
    pub(crate) hidden: usize,
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) hidden_rows: usize,
    pub(crate) decoded_embedding_row: usize,
    pub(crate) attention_norm: usize,
    pub(crate) query_norm: usize,
    pub(crate) key_norm: usize,
    pub(crate) key_cache: usize,
    pub(crate) value_cache: usize,
    pub(crate) attention_residual: usize,
    pub(crate) attention_row_norm: usize,
    pub(crate) query_projection: usize,
    pub(crate) key_projection: usize,
    pub(crate) value_projection: usize,
    pub(crate) query_renorm: usize,
    pub(crate) key_renorm: usize,
    pub(crate) causal_attention_output: usize,
    pub(crate) causal_attention_scores: usize,
    pub(crate) causal_attention_exponents: usize,
    pub(crate) attention_output_projection: usize,
    pub(crate) ffn_norm: usize,
    pub(crate) ffn_residual: usize,
    pub(crate) ffn_row_norm: usize,
    pub(crate) gate_projection: usize,
    pub(crate) silu_output: usize,
    pub(crate) up_projection: usize,
    pub(crate) hadamard_output: usize,
    pub(crate) down_projection: usize,
    pub(crate) final_norm: usize,
    pub(crate) final_rms_norm: usize,
    pub(crate) returned_hidden: usize,
    pub(crate) rank_head_projection: usize,
}

impl Qwen3AllocationShape {
    #[expect(
        clippy::too_many_arguments,
        reason = "the verified Qwen3 layout is the sole source of each allocation dimension"
    )]
    pub(crate) fn new(
        tokens: usize,
        hidden: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        q_width: usize,
        kv_width: usize,
        feed_forward: usize,
        rank_labels: usize,
    ) -> Result<Self> {
        let hidden_rows =
            kernels::cpu_f32::rms_norm_output_elements(tokens, hidden).context(Qwen3CpuSnafu)?;
        let key_cache = checked_product(tokens, kv_width, "Qwen3 causal key cache")?;
        let attention_residual = hidden_rows;
        let ffn_residual = hidden_rows;
        let gate_projection = CheckedMatrix::projection_output_elements(feed_forward);
        let silu_output = kernels::cpu_f32::unary_output_elements(gate_projection);
        let hadamard_output = kernels::cpu_f32::binary_output_elements(silu_output);
        let shape = Self {
            tokens,
            hidden,
            heads,
            kv_heads,
            head_dim,
            hidden_rows,
            decoded_embedding_row: CheckedMatrix::decoded_row_elements(hidden),
            attention_norm: hidden,
            query_norm: head_dim,
            key_norm: head_dim,
            key_cache,
            value_cache: key_cache,
            attention_residual,
            attention_row_norm: kernels::cpu_f32::rms_norm_output_elements(1, hidden)
                .context(Qwen3CpuSnafu)?,
            query_projection: CheckedMatrix::projection_output_elements(q_width),
            key_projection: CheckedMatrix::projection_output_elements(kv_width),
            value_projection: CheckedMatrix::projection_output_elements(kv_width),
            query_renorm: kernels::cpu_f32::rms_norm_output_elements(heads, head_dim)
                .context(Qwen3CpuSnafu)?,
            key_renorm: kernels::cpu_f32::rms_norm_output_elements(kv_heads, head_dim)
                .context(Qwen3CpuSnafu)?,
            causal_attention_output: q_width,
            causal_attention_scores: tokens,
            causal_attention_exponents: tokens,
            attention_output_projection: CheckedMatrix::projection_output_elements(hidden),
            ffn_norm: hidden,
            ffn_residual,
            ffn_row_norm: kernels::cpu_f32::rms_norm_output_elements(1, hidden)
                .context(Qwen3CpuSnafu)?,
            gate_projection,
            silu_output,
            up_projection: CheckedMatrix::projection_output_elements(feed_forward),
            hadamard_output,
            down_projection: CheckedMatrix::projection_output_elements(hidden),
            final_norm: hidden,
            final_rms_norm: kernels::cpu_f32::rms_norm_output_elements(tokens, hidden)
                .context(Qwen3CpuSnafu)?,
            returned_hidden: hidden,
            rank_head_projection: CheckedMatrix::projection_output_elements(rank_labels),
        };
        shape.validate_f32_vec_capacities()?;
        Ok(shape)
    }

    pub(crate) fn embedding_workspace_upper_bound(self) -> Result<usize> {
        let embedding = self.embedding_lookup_elements()?;
        let attention = self.attention_phase_elements()?;
        let ffn = self.ffn_phase_elements()?;
        let finalization = self.finalization_workspace_elements()?;
        Ok([embedding, attention, ffn, finalization]
            .into_iter()
            .max()
            .unwrap_or(0))
    }

    pub(crate) fn causal_prefix_elements(self, tokens: usize) -> Result<usize> {
        if tokens == 0 || tokens > self.tokens {
            return Qwen3ExecutionSnafu {
                requested: tokens,
                rule: "causal attention prefix must fit the checked Qwen3 token shape",
            }
            .fail();
        }
        Ok(tokens)
    }

    fn embedding_lookup_elements(self) -> Result<usize> {
        checked_add(
            self.hidden_rows,
            self.decoded_embedding_row,
            "Qwen3 embedding lookup workspace",
        )
    }

    fn finalization_workspace_elements(self) -> Result<usize> {
        checked_add(
            checked_add(
                self.hidden_rows,
                self.final_norm,
                "Qwen3 final norm workspace",
            )?,
            self.final_rms_norm,
            "Qwen3 final RMS workspace",
        )
    }

    fn attention_phase_elements(self) -> Result<usize> {
        let token_local = self.attention_token_workspace()?;
        sum(
            &[
                self.hidden_rows,
                self.key_cache,
                self.value_cache,
                self.attention_residual,
                self.attention_norm,
                self.query_norm,
                self.key_norm,
                token_local,
            ],
            "Qwen3 attention block workspace",
        )
    }

    fn attention_token_workspace(self) -> Result<usize> {
        let query_renorm = sum(
            &[
                self.attention_row_norm,
                self.query_projection,
                self.query_renorm,
                self.key_projection,
                self.value_projection,
            ],
            "Qwen3 query RMS workspace",
        )?;
        let key_renorm = sum(
            &[
                self.attention_row_norm,
                self.query_renorm,
                self.key_projection,
                self.key_renorm,
                self.value_projection,
            ],
            "Qwen3 key RMS workspace",
        )?;
        let attention = sum(
            &[
                self.attention_row_norm,
                self.query_renorm,
                self.key_renorm,
                self.value_projection,
                self.causal_attention_output,
                self.causal_attention_scores,
                self.causal_attention_exponents,
            ],
            "Qwen3 causal attention token workspace",
        )?;
        let output_projection = sum(
            &[
                self.attention_row_norm,
                self.query_renorm,
                self.key_renorm,
                self.value_projection,
                self.causal_attention_output,
                self.attention_output_projection,
            ],
            "Qwen3 attention output projection workspace",
        )?;
        Ok([query_renorm, key_renorm, attention, output_projection]
            .into_iter()
            .max()
            .unwrap_or(0))
    }

    fn ffn_phase_elements(self) -> Result<usize> {
        let gate_activation = sum(
            &[self.ffn_row_norm, self.gate_projection, self.silu_output],
            "Qwen3 FFN gate activation workspace",
        )?;
        let token_local = sum(
            &[
                self.ffn_row_norm,
                self.silu_output,
                self.up_projection,
                self.hadamard_output,
                self.down_projection,
            ],
            "Qwen3 FFN token workspace",
        )?;
        let token_local = gate_activation.max(token_local);
        sum(
            &[
                self.hidden_rows,
                self.key_cache,
                self.value_cache,
                self.attention_residual,
                self.attention_norm,
                self.query_norm,
                self.key_norm,
                self.ffn_norm,
                self.ffn_residual,
                token_local,
            ],
            "Qwen3 FFN block workspace",
        )
    }

    fn validate_f32_vec_capacities(self) -> Result<()> {
        for (elements, rule) in [
            (self.hidden_rows, "Qwen3 token hidden rows"),
            (self.decoded_embedding_row, "Qwen3 decoded embedding row"),
            (self.attention_norm, "Qwen3 attention norm"),
            (self.query_norm, "Qwen3 query norm"),
            (self.key_norm, "Qwen3 key norm"),
            (self.key_cache, "Qwen3 causal key cache"),
            (self.value_cache, "Qwen3 causal value cache"),
            (self.attention_residual, "Qwen3 attention residual"),
            (self.attention_row_norm, "Qwen3 attention row norm"),
            (self.query_projection, "Qwen3 query projection"),
            (self.key_projection, "Qwen3 key projection"),
            (self.value_projection, "Qwen3 value projection"),
            (self.query_renorm, "Qwen3 query RMS output"),
            (self.key_renorm, "Qwen3 key RMS output"),
            (
                self.causal_attention_output,
                "Qwen3 causal attention output",
            ),
            (
                self.causal_attention_scores,
                "Qwen3 causal attention scores",
            ),
            (
                self.causal_attention_exponents,
                "Qwen3 causal attention exponentials",
            ),
            (
                self.attention_output_projection,
                "Qwen3 attention output projection",
            ),
            (self.ffn_norm, "Qwen3 FFN norm"),
            (self.ffn_residual, "Qwen3 FFN residual"),
            (self.ffn_row_norm, "Qwen3 FFN row norm"),
            (self.gate_projection, "Qwen3 FFN gate projection"),
            (self.silu_output, "Qwen3 FFN SiLU output"),
            (self.up_projection, "Qwen3 FFN up projection"),
            (self.hadamard_output, "Qwen3 SwiGLU output"),
            (self.down_projection, "Qwen3 FFN down projection"),
            (self.final_norm, "Qwen3 final norm"),
            (self.final_rms_norm, "Qwen3 final RMS norm"),
            (self.returned_hidden, "Qwen3 returned hidden row"),
            (self.rank_head_projection, "Qwen3 rank head projection"),
        ] {
            checked_f32_vec_capacity(elements, rule)?;
        }
        Ok(())
    }
}

fn checked_product(left: usize, right: usize, rule: &'static str) -> Result<usize> {
    left.checked_mul(right).ok_or_else(|| {
        Qwen3ExecutionSnafu {
            requested: left,
            rule,
        }
        .build()
    })
}

fn checked_add(left: usize, right: usize, rule: &'static str) -> Result<usize> {
    left.checked_add(right).ok_or_else(|| {
        Qwen3ExecutionSnafu {
            requested: left,
            rule,
        }
        .build()
    })
}

fn sum(elements: &[usize], rule: &'static str) -> Result<usize> {
    elements
        .iter()
        .copied()
        .try_fold(0usize, |total, element| checked_add(total, element, rule))
}

fn f32_bytes(elements: usize) -> Result<u64> {
    let elements = u64::try_from(elements).map_err(|_| {
        Qwen3ExecutionSnafu {
            requested: usize::MAX,
            rule: "Qwen3 logical f32 element count exceeds u64",
        }
        .build()
    })?;
    elements
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .ok_or_else(|| {
            Qwen3ExecutionSnafu {
                requested: usize::MAX,
                rule: "Qwen3 logical f32 byte count overflowed",
            }
            .build()
        })
}

fn checked_f32_vec_capacity(elements: usize, rule: &'static str) -> Result<()> {
    std::alloc::Layout::array::<f32>(elements).map_err(|_| {
        Qwen3ExecutionSnafu {
            requested: elements,
            rule,
        }
        .build()
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Qwen3AllocationShape;

    #[test]
    fn shape_derives_checked_phase_capacities_from_one_geometry() -> std::result::Result<(), String>
    {
        let shape = Qwen3AllocationShape::new(4, 3, 2, 1, 2, 4, 2, 5, 2)
            .map_err(|error| error.to_string())?;
        if shape.hidden_rows != 12
            || shape.key_cache != 8
            || shape.ffn_residual != 12
            || shape.rank_head_projection != 2
        {
            return Err("Qwen3 allocation shape did not retain its derived capacities".to_string());
        }
        if shape
            .embedding_workspace_upper_bound()
            .map_err(|error| error.to_string())?
            < 14
        {
            return Err("Qwen3 allocation envelope lost embedding lookup overlap".to_string());
        }
        Ok(())
    }

    #[test]
    fn shape_refuses_overflow_before_any_allocation() {
        assert!(Qwen3AllocationShape::new(usize::MAX, 2, 1, 1, 1, 1, 1, 1, 2).is_err());
    }
}
