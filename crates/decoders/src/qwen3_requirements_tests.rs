//! Independent live-owner witnesses for Qwen3 logical CPU requirements.

use loader::gguf::{ArtifactDigest, Sha256Digest};

use super::{Qwen3AllocationShape, Qwen3CpuRequirements, checked_add, checked_f32_vec_capacity};
use crate::matrix::CheckedMatrix;

type TestResult = std::result::Result<(), String>;

const RANK_LABELS: usize = 2;

#[derive(Clone, Copy)]
struct IndependentGeometry {
    tokens: usize,
    hidden: usize,
    heads: usize,
    key_value_heads: usize,
    head: usize,
    query: usize,
    key_value: usize,
    feed_forward: usize,
}

impl IndependentGeometry {
    fn allocation_shape(self) -> std::result::Result<Qwen3AllocationShape, String> {
        Qwen3AllocationShape::new(
            self.tokens,
            self.hidden,
            self.heads,
            self.key_value_heads,
            self.head,
            self.query,
            self.key_value,
            self.feed_forward,
            RANK_LABELS,
        )
        .map_err(|error| error.to_string())
    }

    const fn hidden_rows(self) -> usize {
        self.tokens * self.hidden
    }

    const fn key_value_cache(self) -> usize {
        self.tokens * self.key_value
    }

    const fn embedding_lookup(self) -> usize {
        self.hidden_rows() + self.hidden
    }

    const fn attention_persistent(self) -> usize {
        2 * self.hidden_rows() + 2 * self.key_value_cache() + self.hidden + 2 * self.head
    }

    const fn attention_token_candidates(self) -> [usize; 4] {
        [
            self.hidden + 2 * self.query + 2 * self.key_value,
            self.hidden + self.query + 3 * self.key_value,
            self.hidden + 2 * self.query + 2 * self.key_value + 2 * self.tokens,
            2 * self.hidden + 2 * self.query + 2 * self.key_value,
        ]
    }

    fn attention_token(self) -> usize {
        self.attention_token_candidates()
            .into_iter()
            .max()
            .unwrap_or(0)
    }

    fn attention_phase(self) -> usize {
        self.attention_persistent() + self.attention_token()
    }

    const fn ffn_persistent(self) -> usize {
        3 * self.hidden_rows() + 2 * self.key_value_cache() + 2 * self.hidden + 2 * self.head
    }

    const fn ffn_token(self) -> usize {
        2 * self.hidden + 3 * self.feed_forward
    }

    const fn ffn_phase(self) -> usize {
        self.ffn_persistent() + self.ffn_token()
    }

    const fn finalization_workspace(self) -> usize {
        2 * self.hidden_rows() + self.hidden
    }

    fn embedding_workspace(self) -> usize {
        [
            self.embedding_lookup(),
            self.attention_phase(),
            self.ffn_phase(),
            self.finalization_workspace(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0)
    }
}

const ASYMMETRIC: IndependentGeometry = IndependentGeometry {
    tokens: 17,
    hidden: 7,
    heads: 6,
    key_value_heads: 2,
    head: 5,
    query: 30,
    key_value: 10,
    feed_forward: 19,
};

#[test]
fn asymmetric_live_owner_inventory_reconciles_every_named_phase() -> TestResult {
    let shape = ASYMMETRIC.allocation_shape()?;

    assert_eq!(ASYMMETRIC.hidden_rows(), 119, "token-major hidden rows");
    assert_eq!(ASYMMETRIC.key_value_cache(), 170, "one KV cache");
    assert_eq!(
        ASYMMETRIC.attention_token_candidates(),
        [87, 67, 121, 94],
        "query replacement, key replacement, causal attention, and output projection live sets",
    );
    assert_eq!(
        shape
            .embedding_lookup_elements()
            .map_err(|error| error.to_string())?,
        126,
        "the pre-reserved hidden buffer overlaps one decoded embedding row",
    );
    assert_eq!(
        shape
            .attention_token_workspace()
            .map_err(|error| error.to_string())?,
        121,
        "scores and exponentials make causal attention the token-local maximum",
    );
    assert_eq!(
        shape
            .attention_phase_elements()
            .map_err(|error| error.to_string())?,
        716,
        "attention includes hidden rows, both caches, residual accumulation, norms, and token locals",
    );
    assert_eq!(
        shape
            .ffn_phase_elements()
            .map_err(|error| error.to_string())?,
        792,
        "FFN includes lexically retained attention owners and all three feed-forward vectors",
    );
    assert_eq!(
        shape
            .finalization_workspace_elements()
            .map_err(|error| error.to_string())?,
        245,
        "finalization excludes only the separately reported returned hidden row",
    );
    assert_eq!(
        shape
            .embedding_workspace_upper_bound()
            .map_err(|error| error.to_string())?,
        ASYMMETRIC.embedding_workspace(),
        "the maximum must select from independently derived source-live phases",
    );
    Ok(())
}

#[test]
fn attention_output_copy_can_dominate_scores_and_replacement_temporaries() -> TestResult {
    let geometry = IndependentGeometry {
        tokens: 2,
        hidden: 31,
        heads: 3,
        key_value_heads: 1,
        head: 5,
        query: 15,
        key_value: 5,
        feed_forward: 9,
    };
    let shape = geometry.allocation_shape()?;

    assert_eq!(
        geometry.attention_token_candidates(),
        [71, 61, 75, 102],
        "the returned merged query and hidden projection coexist at the extend call",
    );
    assert_eq!(
        shape
            .attention_token_workspace()
            .map_err(|error| error.to_string())?,
        102,
        "the attention output-copy live set must not be replaced by the causal-score phase",
    );
    assert_eq!(
        shape
            .attention_phase_elements()
            .map_err(|error| error.to_string())?,
        287,
        "the output-copy local maximum composes with the persistent attention owners",
    );
    Ok(())
}

#[test]
fn query_and_key_replacement_live_sets_are_individually_observable() -> TestResult {
    let mut query_replacement = Qwen3AllocationShape::new(1, 1, 1, 1, 1, 1, 1, 1, RANK_LABELS)
        .map_err(|error| error.to_string())?;
    query_replacement.attention_row_norm = 2;
    query_replacement.query_projection = 101;
    query_replacement.query_renorm = 3;
    query_replacement.key_projection = 5;
    query_replacement.key_renorm = 11;
    query_replacement.value_projection = 7;
    query_replacement.causal_attention_output = 13;
    query_replacement.causal_attention_scores = 17;
    query_replacement.causal_attention_exponents = 19;
    query_replacement.attention_output_projection = 23;

    assert_eq!(
        query_replacement
            .attention_token_workspace()
            .map_err(|error| error.to_string())?,
        118,
        "the projected query must overlap its replacement RMS output",
    );

    let mut key_replacement = query_replacement;
    key_replacement.query_projection = 3;
    key_replacement.query_renorm = 5;
    key_replacement.key_projection = 101;
    key_replacement.key_renorm = 97;
    assert_eq!(
        key_replacement
            .attention_token_workspace()
            .map_err(|error| error.to_string())?,
        212,
        "the projected key must overlap both the retained query RMS output and its replacement",
    );
    Ok(())
}

#[test]
fn ffn_phase_retains_attention_buffers_until_the_block_scope_ends() -> TestResult {
    let geometry = IndependentGeometry {
        tokens: 3,
        hidden: 7,
        heads: 6,
        key_value_heads: 2,
        head: 5,
        query: 30,
        key_value: 10,
        feed_forward: 101,
    };
    let shape = geometry.allocation_shape()?;

    assert_eq!(
        geometry.ffn_persistent(),
        147,
        "lexically retained block owners"
    );
    assert_eq!(
        geometry.ffn_token(),
        317,
        "normalized row, activated/up/fused FFN values, and down projection",
    );
    assert_eq!(
        shape
            .ffn_phase_elements()
            .map_err(|error| error.to_string())?,
        464,
        "Rust borrow shortening must not be mistaken for early Vec drops",
    );
    assert_eq!(
        shape
            .embedding_workspace_upper_bound()
            .map_err(|error| error.to_string())?,
        464,
        "the demanding FFN phase must determine this geometry's envelope",
    );
    Ok(())
}

#[test]
fn shape_sizes_reconcile_with_the_lower_allocation_owners() -> TestResult {
    let shape = ASYMMETRIC.allocation_shape()?;
    let hidden_rows =
        kernels::cpu_f32::rms_norm_output_elements(ASYMMETRIC.tokens, ASYMMETRIC.hidden)
            .map_err(|error| error.to_string())?;
    let query_renorm =
        kernels::cpu_f32::rms_norm_output_elements(ASYMMETRIC.heads, ASYMMETRIC.head)
            .map_err(|error| error.to_string())?;
    let key_renorm =
        kernels::cpu_f32::rms_norm_output_elements(ASYMMETRIC.key_value_heads, ASYMMETRIC.head)
            .map_err(|error| error.to_string())?;

    assert_eq!(
        shape.hidden_rows, hidden_rows,
        "RMSNorm owns token-row sizing"
    );
    assert_eq!(
        shape.decoded_embedding_row,
        CheckedMatrix::decoded_row_elements(ASYMMETRIC.hidden),
        "checked matrix row decoding owns embedding-row sizing",
    );
    assert_eq!(
        shape.query_projection,
        CheckedMatrix::projection_output_elements(ASYMMETRIC.query),
        "checked matrix owns query projection sizing",
    );
    assert_eq!(
        shape.key_projection,
        CheckedMatrix::projection_output_elements(ASYMMETRIC.key_value),
        "checked matrix owns key projection sizing",
    );
    assert_eq!(
        shape.value_projection,
        CheckedMatrix::projection_output_elements(ASYMMETRIC.key_value),
        "checked matrix owns value projection sizing",
    );
    assert_eq!(
        shape.query_renorm, query_renorm,
        "RMSNorm owns query renormalization sizing"
    );
    assert_eq!(
        shape.key_renorm, key_renorm,
        "RMSNorm owns key renormalization sizing"
    );
    assert_eq!(
        shape.gate_projection,
        CheckedMatrix::projection_output_elements(ASYMMETRIC.feed_forward),
        "checked matrix owns gate projection sizing",
    );
    assert_eq!(
        shape.silu_output,
        kernels::cpu_f32::unary_output_elements(shape.gate_projection),
        "SiLU owns activated-gate sizing",
    );
    assert_eq!(
        shape.hadamard_output,
        kernels::cpu_f32::binary_output_elements(shape.silu_output),
        "Hadamard owns fused-activation sizing",
    );
    assert_eq!(
        shape.down_projection,
        CheckedMatrix::projection_output_elements(ASYMMETRIC.hidden),
        "checked matrix owns down projection sizing",
    );
    assert_eq!(
        shape.rank_head_projection,
        CheckedMatrix::projection_output_elements(RANK_LABELS),
        "checked matrix owns transient rank-head sizing",
    );
    Ok(())
}

#[test]
fn embedding_and_rank_reports_distinguish_heap_return_ownership() -> TestResult {
    let shape = ASYMMETRIC.allocation_shape()?;
    let digest = ArtifactDigest::Sha256(Sha256Digest::from_bytes([0x5a; 32]));
    let serialized_backing_bytes = 1_234_567;
    let embedding = Qwen3CpuRequirements::embedding(
        digest,
        serialized_backing_bytes,
        ASYMMETRIC.tokens,
        &shape,
    )
    .map_err(|error| error.to_string())?;
    let rank =
        Qwen3CpuRequirements::rank(digest, serialized_backing_bytes, ASYMMETRIC.tokens, &shape)
            .map_err(|error| error.to_string())?;

    assert_eq!(
        embedding.artifact_digest(),
        digest,
        "embedding artifact identity"
    );
    assert_eq!(rank.artifact_digest(), digest, "rank artifact identity");
    assert_eq!(
        embedding.serialized_backing_bytes(),
        serialized_backing_bytes,
        "serialized backing remains separate from logical f32 backing",
    );
    assert_eq!(embedding.max_context(), 17, "embedding admitted context");
    assert_eq!(
        embedding.workspace_upper_bound_bytes(),
        3_168,
        "embedding workspace bytes"
    );
    assert_eq!(
        embedding.returned_output_bytes(),
        28,
        "one returned hidden Vec owns seven f32 values",
    );
    assert_eq!(
        embedding.logical_f32_upper_bound_bytes(),
        3_196,
        "embedding workspace and returned Vec backing reconcile",
    );
    assert_eq!(rank.max_context(), 17, "rank admitted context");
    assert_eq!(
        rank.workspace_upper_bound_bytes(),
        3_196,
        "rank promotes the body-returned hidden Vec into transient workspace",
    );
    assert_eq!(
        rank.returned_output_bytes(),
        0,
        "the returned two-logit array is stack storage, not heap backing",
    );
    assert_eq!(
        rank.logical_f32_upper_bound_bytes(),
        3_196,
        "rank has no returned heap addend",
    );
    Ok(())
}

#[test]
fn individual_f32_vec_capacity_is_checked_at_the_isize_boundary() -> TestResult {
    let max_bytes = usize::try_from(isize::MAX).map_err(|error| error.to_string())?;
    let max_elements = max_bytes / std::mem::size_of::<f32>();
    checked_f32_vec_capacity(max_elements, "boundary allocation")
        .map_err(|error| error.to_string())?;

    let over_limit = max_elements
        .checked_add(1)
        .ok_or_else(|| "test f32 capacity boundary overflowed usize".to_string())?;
    let direct_error = checked_f32_vec_capacity(over_limit, "boundary allocation")
        .err()
        .ok_or_else(|| "an individually impossible f32 Vec capacity was accepted".to_string())?;
    assert!(
        matches!(direct_error, crate::Error::Qwen3Execution { .. }),
        "the individual Vec capacity guard must preserve a typed Qwen3 refusal",
    );

    let hidden = max_elements / 2 + 1;
    let shape_error = Qwen3AllocationShape::new(2, hidden, 1, 1, 1, 1, 1, 1, RANK_LABELS)
        .err()
        .ok_or_else(|| {
            "a Qwen3 shape containing an impossible hidden Vec was accepted".to_string()
        })?;
    assert!(
        matches!(shape_error, crate::Error::Qwen3Execution { .. }),
        "shape admission must apply the same individual Vec capacity guard",
    );
    Ok(())
}

#[test]
fn aggregate_live_bytes_may_exceed_one_vec_limit_without_being_a_vec_capacity() -> TestResult {
    let max_bytes = usize::try_from(isize::MAX).map_err(|error| error.to_string())?;
    let max_elements = max_bytes / std::mem::size_of::<f32>();
    let axis = max_elements / 8;
    let geometry = IndependentGeometry {
        tokens: 1,
        hidden: axis,
        heads: 1,
        key_value_heads: 1,
        head: axis,
        query: axis,
        key_value: axis,
        feed_forward: axis,
    };
    let shape = geometry.allocation_shape()?;
    let requirements =
        Qwen3CpuRequirements::embedding(ArtifactDigest::NotComputed, 0, geometry.tokens, &shape)
            .map_err(|error| error.to_string())?;
    let expected_elements = axis
        .checked_mul(15)
        .ok_or_else(|| "test aggregate element oracle overflowed usize".to_string())?;
    let f32_bytes = u64::try_from(std::mem::size_of::<f32>()).map_err(|error| error.to_string())?;
    let expected_bytes = u64::try_from(expected_elements)
        .map_err(|error| error.to_string())?
        .checked_mul(f32_bytes)
        .ok_or_else(|| "test aggregate byte oracle overflowed u64".to_string())?;
    let single_vec_limit = u64::try_from(isize::MAX).map_err(|error| error.to_string())?;

    assert!(axis > 0, "the target must admit at least one f32 element");
    assert_eq!(
        requirements.logical_f32_upper_bound_bytes(),
        expected_bytes,
        "fourteen FFN-workspace axes plus one returned-hidden axis",
    );
    assert!(
        expected_bytes > single_vec_limit,
        "the witness must exceed one Vec's byte domain while each constituent Vec remains valid",
    );
    Ok(())
}

#[test]
fn overflowing_owner_products_and_live_sums_fail_without_allocating() -> TestResult {
    let rms_error = Qwen3AllocationShape::new(usize::MAX, 2, 1, 1, 1, 1, 1, 1, RANK_LABELS)
        .err()
        .ok_or_else(|| "overflowing RMS-owned hidden rows were accepted".to_string())?;
    assert!(
        matches!(
            rms_error,
            crate::Error::Qwen3Cpu {
                source: kernels::Error::RmsNormSizeOverflow { .. },
                ..
            }
        ),
        "RMS-owned size arithmetic must retain its lower-owner error chain",
    );

    let key_cache_tokens = usize::MAX / 2 + 1;
    let key_cache_error =
        Qwen3AllocationShape::new(key_cache_tokens, 1, 1, 1, 1, 1, 2, 1, RANK_LABELS)
            .err()
            .ok_or_else(|| "overflowing Qwen3-owned KV cache was accepted".to_string())?;
    assert!(
        matches!(key_cache_error, crate::Error::Qwen3Execution { .. }),
        "Qwen3-owned multiplication must produce a typed Qwen3 refusal",
    );

    let live_sum_error = checked_add(usize::MAX, 1, "test live-owner sum")
        .err()
        .ok_or_else(|| "overflowing live-owner sum was accepted".to_string())?;
    assert!(
        matches!(live_sum_error, crate::Error::Qwen3Execution { .. }),
        "live-owner addition must produce a typed Qwen3 refusal",
    );

    let max_bytes = usize::try_from(isize::MAX).map_err(|error| error.to_string())?;
    let max_elements = max_bytes / std::mem::size_of::<f32>();
    let large_axis = (max_elements / 4)
        .checked_mul(3)
        .ok_or_else(|| "test live-owner axis overflowed usize".to_string())?;
    let large_shape = IndependentGeometry {
        tokens: 1,
        hidden: large_axis,
        heads: 1,
        key_value_heads: 1,
        head: large_axis,
        query: large_axis,
        key_value: large_axis,
        feed_forward: large_axis,
    }
    .allocation_shape()?;
    let aggregate_error = large_shape
        .embedding_workspace_upper_bound()
        .err()
        .ok_or_else(|| "overflowing aggregate live-owner sum was accepted".to_string())?;
    assert!(
        matches!(aggregate_error, crate::Error::Qwen3Execution { .. }),
        "aggregate workspace arithmetic must fail before any allocation",
    );
    Ok(())
}
