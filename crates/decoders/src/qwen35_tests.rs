use std::{collections::BTreeMap, fs, num::NonZeroU64};

use loader::gguf::{
    ArtifactByteLimit, ObservedArtifact, Sha256Digest, VerifiedArtifact, observe_gguf_with_sha256,
};
use sha2::{Digest, Sha256};
use tempfile::tempdir;
use test_fixtures::{RawGguf, RawMetadata, RawMetadataValue, RawTensor, serialize_raw_gguf};

use super::*;
use crate::Qwen35Weights;

const TEST_ALIGNMENT: u32 = 32;
const TEST_HIDDEN: u64 = 3;
const TEST_FEED_FORWARD: u64 = 5;
const TEST_HEADS: u64 = 2;
const TEST_KEY_VALUE_HEADS: u64 = 1;
const TEST_HEAD_WIDTH: u64 = 2;
const TEST_CONV_KERNEL: u64 = 2;
const TEST_INNER: u64 = 8;
const TEST_STATE: u64 = 2;
const TEST_TIME_STEP_RANK: u64 = 4;
const TEST_GROUP_COUNT: u64 = 2;
const TEST_VOCABULARY: u64 = 5;
const TEST_MAIN_BLOCKS: u64 = 4;
const TEST_FULL_ATTENTION_INTERVAL: u64 = 4;
const TEST_F32_BYTES: u64 = 4;
const TEST_F32_TYPE_ID: u32 = 0;
const TEST_Q8_0_TYPE_ID: u32 = 8;
const TEST_Q4_K_TYPE_ID: u32 = 12;
const TEST_Q5_K_TYPE_ID: u32 = 13;
const TEST_Q6_K_TYPE_ID: u32 = 14;
const TEST_IQ4_NL_TYPE_ID: u32 = 20;
const TEST_IQ4_XS_TYPE_ID: u32 = 23;
const TEST_Q_WIDTH: u64 = 8;
const TEST_FULL_ATTENTION_OUTPUT_WIDTH: u64 = 4;
const TEST_SSM_CONV_WIDTH: u64 = 16;
const TEST_NEXTN_PROJECTION_WIDTH: u64 = 6;
const TEST_PROJECTION_INPUT_WIDTH: u64 = 64;
const TEST_PROJECTION_OUTPUT_WIDTH: usize = 3;
const TEST_Q8_SCALE_ONE_BITS: u16 = 0x3c00;
const TEST_Q8_SCALE_NONFINITE_BITS: u16 = 0x7c00;
const TEST_ALTERNATING_PERIOD: usize = 2;
const TEST_RMS_EPSILON: f32 = 0.001;
const CANONICAL_HEADS: u64 = 4;
const CANONICAL_KEY_VALUE_HEADS: u64 = 2;
const CANONICAL_HEAD_WIDTH: u64 = 128;
const CANONICAL_CONTEXT: usize = 4;
const CANONICAL_ROPE_SECTIONS: [i32; 4] = [11, 11, 10, 0];
const MIXED_HIDDEN: u64 = 256;
const MIXED_FEED_FORWARD: u64 = 256;
const MIXED_INNER: u64 = 256;
const MIXED_CONTEXT: usize = 4;
const ORACLE_ABSOLUTE_TOLERANCE: f64 = 1.0e-3;
const ORACLE_RELATIVE_TOLERANCE: f64 = 1.0e-4;
const ORACLE_Q8_VALUES: usize = 32;
const ORACLE_Q8_BLOCK_BYTES: usize = 34;
const ORACLE_K_VALUES: usize = 256;
const ORACLE_K_GROUP_VALUES: usize = 32;
const ORACLE_K_SCALE_BYTES: usize = 12;
const ORACLE_Q4_BLOCK_BYTES: usize = 144;
const ORACLE_Q5_BLOCK_BYTES: usize = 176;
const ORACLE_Q6_BLOCK_BYTES: usize = 210;
const ORACLE_IQ4_NL_VALUES: usize = 32;
const ORACLE_IQ4_NL_BLOCK_BYTES: usize = 18;
const ORACLE_IQ4_XS_VALUES: usize = 256;
const ORACLE_IQ4_XS_BLOCK_BYTES: usize = 136;

// WHY: This is the operator-approved exact-sixteen interoperability exception.
// Provenance: ggml-org/llama.cpp@6a1a922d269908a29cbd4b49c27e6a8e7fd10fae,
// ggml/src/ggml-common.h:1120-1122. The oracle uses no upstream decoder expression.
const ORACLE_IQ4_RECONSTRUCTION: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

#[derive(Clone)]
enum MetadataEntry {
    U32(&'static str, u32),
    F32(&'static str, f32),
    String(&'static str, String),
    StringArray(&'static str, Vec<String>),
    I32Array(&'static str, Vec<i32>),
}

impl MetadataEntry {
    fn key(&self) -> &'static str {
        match self {
            Self::U32(key, _)
            | Self::F32(key, _)
            | Self::String(key, _)
            | Self::StringArray(key, _)
            | Self::I32Array(key, _) => key,
        }
    }
}

#[derive(Clone)]
struct FixtureTensor {
    name: String,
    dims: Vec<u64>,
    ggml_type: u32,
    payload: Vec<u8>,
}

#[derive(Clone)]
pub(crate) struct Fixture {
    metadata: Vec<MetadataEntry>,
    tensors: Vec<FixtureTensor>,
}

#[test]
fn accepts_small_all_f32_observation_with_one_nextn_block() -> std::result::Result<(), String> {
    let artifact = observe_fixture(&fixture(1)?)?;
    let profile =
        Qwen35StructuralProfile::try_from_observed(&artifact).map_err(|error| error.to_string())?;

    assert_eq!(
        profile.stored_block_count(),
        5,
        "stored count includes the terminal NextN block"
    );
    assert_eq!(
        profile.main_block_count(),
        TEST_MAIN_BLOCKS,
        "main count excludes the terminal NextN block"
    );
    assert_eq!(
        profile.nextn_block_count(),
        1,
        "fixture carries one terminal NextN block"
    );
    assert!(
        std::ptr::eq(profile.observed(), &raw const artifact),
        "profile must retain the opaque observation borrow"
    );
    Ok(())
}

#[test]
fn accepts_small_all_f32_observation_without_nextn() -> std::result::Result<(), String> {
    let artifact = observe_fixture(&fixture(0)?)?;
    let profile =
        Qwen35StructuralProfile::try_from_observed(&artifact).map_err(|error| error.to_string())?;

    assert_eq!(
        profile.stored_block_count(),
        TEST_MAIN_BLOCKS,
        "without NextN all stored blocks are main blocks"
    );
    assert_eq!(
        profile.main_block_count(),
        TEST_MAIN_BLOCKS,
        "main count remains the four-block cadence fixture"
    );
    assert_eq!(
        profile.nextn_block_count(),
        0,
        "fixture explicitly has no NextN block"
    );
    Ok(())
}

#[test]
fn keeps_execution_epsilon_outside_structural_admission() -> std::result::Result<(), String> {
    let mut fixture = fixture(1)?;
    fixture
        .metadata
        .retain(|entry| entry.key() != LAYERNORM_RMS_EPSILON_KEY);
    let artifact = observe_fixture(&fixture)?;
    let structural = Qwen35StructuralProfile::try_from_observed(&artifact);
    assert!(
        structural.is_ok(),
        "a structurally recognizable artifact need not carry a field used only by recurrent execution"
    );

    let epsilon = recurrent_layernorm_rms_epsilon(artifact.metadata());
    assert!(
        matches!(
            epsilon,
            Err(crate::Error::MissingMetadata {
                key: LAYERNORM_RMS_EPSILON_KEY,
                ..
            })
        ),
        "recurrent execution must refuse rather than default a missing RMS epsilon"
    );
    Ok(())
}

#[test]
fn executes_a_nonzero_recurrent_trunk_with_artifact_owned_state() -> std::result::Result<(), String>
{
    let mut fixture = fixture(1)?;
    for name in [
        "blk.0.attn_norm.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ssm_alpha.weight",
        "blk.0.ssm_beta.weight",
        "blk.0.ssm_conv1d.weight",
        "blk.0.ssm_norm.weight",
        "blk.0.ssm_out.weight",
    ] {
        set_f32_repeated(&mut fixture, name, 1.0)?;
    }
    set_f32_repeated(&mut fixture, "blk.0.ssm_a", -1.0)?;
    let payload = verify_fixture(&fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    let mut execution = weights
        .recurrent_execution(0)
        .map_err(|error| error.to_string())?;
    let input = [1.0_f32, -2.0, 3.0];

    let first = execution.step(&input).map_err(|error| error.to_string())?;
    let second = execution.step(&input).map_err(|error| error.to_string())?;

    assert!(
        first.iter().all(|value| value.is_finite()) && first.iter().any(|value| *value != 0.0),
        "nonzero verified parameters must produce a finite nonzero recurrent output"
    );
    assert_ne!(
        first, second,
        "the second token pass must observe the execution object's retained recurrent or convolution state"
    );
    Ok(())
}

#[test]
fn executes_verified_token_ids_through_hybrid_main_blocks_transactionally()
-> std::result::Result<(), String> {
    let mut fixture = fixture(1)?;
    let names = fixture
        .tensors
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect::<Vec<_>>();
    for name in names {
        set_f32_repeated(&mut fixture, &name, 0.125)?;
    }
    set_f32_repeated(&mut fixture, "blk.0.ssm_a", -1.0)?;
    set_f32_repeated(&mut fixture, "blk.1.ssm_a", -1.0)?;
    set_f32_repeated(&mut fixture, "blk.2.ssm_a", -1.0)?;
    let payload = verify_fixture(&fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;

    let mut batched = weights.execution(3).map_err(|error| error.to_string())?;
    let batched_logits = batched.step(&[1, 2]).map_err(|error| error.to_string())?;
    let mut sequential = weights.execution(3).map_err(|error| error.to_string())?;
    let mut sequential_logits = sequential.step(&[1]).map_err(|error| error.to_string())?;
    sequential_logits.extend(sequential.step(&[2]).map_err(|error| error.to_string())?);

    assert_eq!(
        batched_logits, sequential_logits,
        "one batched call must preserve token state order"
    );
    assert_eq!(batched_logits.len(), 2 * test_dimension(TEST_VOCABULARY)?);
    assert!(batched_logits.iter().all(|value| value.is_finite()));
    assert_ne!(
        &batched_logits[..test_dimension(TEST_VOCABULARY)?],
        &batched_logits[test_dimension(TEST_VOCABULARY)?..],
        "nonzero text positions and retained hybrid state must affect the next token logits"
    );
    Ok(())
}

#[test]
fn canonical_hybrid_execution_matches_independent_f64_oracle_and_rolls_back()
-> std::result::Result<(), String> {
    let fixture = canonical_hybrid_fixture()?;
    let mut oracle = CanonicalHybridOracle::from_fixture(&fixture)?;
    let expected_batch = oracle.step(&[1, 2])?;
    let expected_continuation = oracle.step(&[3])?;
    let mut no_attention = CanonicalHybridOracle::from_fixture(&fixture)?.without_attention();
    let no_attention_logits = no_attention.step(&[1, 2])?;
    let mut no_ffn = CanonicalHybridOracle::from_fixture(&fixture)?.without_ffn();
    let no_ffn_logits = no_ffn.step(&[1, 2])?;
    let mut adjacent_pairs = CanonicalHybridOracle::from_fixture(&fixture)?.with_adjacent_pairs();
    let adjacent_pair_logits = adjacent_pairs.step(&[1, 2])?;
    assert_oracle_difference(
        &expected_batch,
        &no_attention_logits,
        "attention residual path",
    )?;
    assert_oracle_difference(&expected_batch, &no_ffn_logits, "SwiGLU residual path")?;
    assert_oracle_difference(
        &expected_batch,
        &adjacent_pair_logits,
        "half-split IMRoPE layout",
    )?;

    let payload = verify_fixture(&fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    let mut batched = weights
        .execution(CANONICAL_CONTEXT)
        .map_err(|error| error.to_string())?;
    let actual_batch = batched.step(&[1, 2]).map_err(|error| error.to_string())?;
    assert_f32_matches_f64(
        &actual_batch,
        &expected_batch,
        "canonical recurrent/full-attention logits",
    )?;
    let actual_continuation = batched.step(&[3]).map_err(|error| error.to_string())?;
    assert_f32_matches_f64(
        &actual_continuation,
        &expected_continuation,
        "canonical hybrid continuation state",
    )?;

    let mut sequential = weights
        .execution(CANONICAL_CONTEXT)
        .map_err(|error| error.to_string())?;
    let mut sequential_logits = sequential.step(&[1]).map_err(|error| error.to_string())?;
    sequential_logits.extend(sequential.step(&[2]).map_err(|error| error.to_string())?);
    assert_eq!(
        actual_batch, sequential_logits,
        "multi-token execution must be exactly token-serial, including hybrid state"
    );

    let mut rollback = weights
        .execution(CANONICAL_CONTEXT)
        .map_err(|error| error.to_string())?;
    let refusal = rollback.step(&[1, u32::MAX]);
    assert!(
        refusal.is_err(),
        "an invalid second token must refuse the whole call"
    );
    let retry = rollback.step(&[1]).map_err(|error| error.to_string())?;
    let mut pristine = weights
        .execution(CANONICAL_CONTEXT)
        .map_err(|error| error.to_string())?;
    let pristine_first = pristine.step(&[1]).map_err(|error| error.to_string())?;
    assert_eq!(
        retry, pristine_first,
        "a refused call after a fully executed first hybrid token must not commit any state"
    );
    Ok(())
}

#[test]
fn canonical_hybrid_execution_honors_a_shorter_rope_rotation_domain()
-> std::result::Result<(), String> {
    let fixture = canonical_hybrid_fixture_with_n_rot(Some(64))?;
    let mut oracle = CanonicalHybridOracle::from_fixture(&fixture)?;
    let expected = oracle.step(&[1, 2])?;
    let mut full_width = CanonicalHybridOracle::from_fixture(&fixture)?.with_full_rotation();
    let full_width_logits = full_width.step(&[1, 2])?;
    assert_oracle_difference(
        &expected,
        &full_width_logits,
        "n_rot=64 IMRoPE tail preservation",
    )?;
    let payload = verify_fixture(&fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    let mut session = weights
        .execution(CANONICAL_CONTEXT)
        .map_err(|error| error.to_string())?;
    let actual = session.step(&[1, 2]).map_err(|error| error.to_string())?;
    assert_f32_matches_f64(&actual, &expected, "D=128 n_rot=64 canonical IMRoPE logits")
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the acceptance test keeps every required semantic and format falsifier visible"
)]
fn mixed_quantized_hybrid_execution_matches_independent_f64_oracle_and_rolls_back()
-> std::result::Result<(), String> {
    let fixture = mixed_quantized_hybrid_fixture()?;
    let mut oracle = CanonicalHybridOracle::from_fixture(&fixture)?;
    let expected_batch = oracle.step(&[1, 2])?;
    let expected_continuation = oracle.step(&[3])?;
    let mut expected_path = expected_batch.clone();
    expected_path.extend_from_slice(&expected_continuation);
    assert_oracle_variant(
        &fixture,
        &expected_path,
        &[1, 2, 3],
        "recurrent-attention cadence blocks",
        CanonicalHybridOracle::without_recurrent_attention,
    )?;
    assert_oracle_variant(
        &fixture,
        &expected_path,
        &[1, 2, 3],
        "full-attention cadence block",
        CanonicalHybridOracle::without_full_attention,
    )?;
    assert_oracle_variant(
        &fixture,
        &expected_path,
        &[1, 2, 3],
        "SwiGLU residual path",
        CanonicalHybridOracle::without_ffn,
    )?;
    assert_oracle_variant(
        &fixture,
        &expected_path,
        &[1, 2, 3],
        "partial IMRoPE rotation domain",
        CanonicalHybridOracle::with_full_rotation,
    )?;
    assert_oracle_variant(
        &fixture,
        &expected_path,
        &[1, 2, 3],
        "per-head interleaved Q/gate ordering",
        CanonicalHybridOracle::with_contiguous_q_gate_halves,
    )?;
    assert_oracle_variant(
        &fixture,
        &expected_path,
        &[1, 2, 3],
        "contiguous GQA head mapping",
        CanonicalHybridOracle::with_modulo_gqa,
    )?;
    assert_oracle_variant(
        &fixture,
        &expected_path,
        &[1, 2, 3],
        "token embedding row orientation",
        CanonicalHybridOracle::with_reversed_embedding_rows,
    )?;
    assert_oracle_variant(
        &fixture,
        &expected_path,
        &[1, 2, 3],
        "vocabulary output row orientation",
        CanonicalHybridOracle::with_reversed_output_rows,
    )?;
    for (name, format) in [
        (TOKEN_EMBEDDING_TENSOR, "Q8_0"),
        ("blk.0.attn_qkv.weight", "Q4_K"),
        ("blk.0.attn_gate.weight", "Q5_K"),
        ("blk.1.attn_qkv.weight", "F32"),
        ("blk.3.attn_v.weight", "Q6_K"),
        ("blk.3.attn_output.weight", "IQ4_NL"),
        (OUTPUT_TENSOR, "IQ4_XS"),
    ] {
        let mut without_path = fixture.clone();
        zero_serialized_tensor(&mut without_path, name)?;
        let mut alternate = CanonicalHybridOracle::from_fixture(&without_path)?;
        let alternate_logits = alternate.step(&[1, 2, 3])?;
        assert_oracle_difference(
            &expected_path,
            &alternate_logits,
            &format!("{format} tensor `{name}`"),
        )?;
    }

    let payload = verify_fixture(&fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    let mut batched = weights
        .execution(MIXED_CONTEXT)
        .map_err(|error| error.to_string())?;
    let actual_batch = batched.step(&[1, 2]).map_err(|error| error.to_string())?;
    assert_f32_matches_f64(
        &actual_batch,
        &expected_batch,
        "mixed-quantized recurrent/full-attention logits",
    )?;
    let actual_continuation = batched.step(&[3]).map_err(|error| error.to_string())?;
    assert_f32_matches_f64(
        &actual_continuation,
        &expected_continuation,
        "mixed-quantized continuation logits",
    )?;

    let mut sequential = weights.execution(3).map_err(|error| error.to_string())?;
    let mut sequential_logits = sequential.step(&[1]).map_err(|error| error.to_string())?;
    sequential_logits.extend(sequential.step(&[2]).map_err(|error| error.to_string())?);
    assert_eq!(
        actual_batch, sequential_logits,
        "mixed quant rows must retain token-order state"
    );
    assert!(
        actual_batch.iter().all(|value| value.is_finite())
            && actual_batch.iter().any(|value| *value != 0.0),
        "every mixed quantized path must reach finite nonzero logits"
    );
    assert!(
        actual_batch
            .chunks_exact(test_dimension(TEST_VOCABULARY)?)
            .all(|row| {
                row.windows(2)
                    .any(|pair| (pair[0] - pair[1]).abs() > f32::EPSILON)
            }),
        "distinct IQ4_XS output rows must produce non-degenerate logits"
    );

    let mut rollback = weights
        .execution(MIXED_CONTEXT)
        .map_err(|error| error.to_string())?;
    assert!(
        rollback.step(&[1, u32::MAX]).is_err(),
        "an invalid second token must refuse the whole mixed-quantized call"
    );
    let retry = rollback.step(&[1]).map_err(|error| error.to_string())?;
    let mut pristine_oracle = CanonicalHybridOracle::from_fixture(&fixture)?;
    let expected_first = pristine_oracle.step(&[1])?;
    assert_f32_matches_f64(&retry, &expected_first, "mixed-quantized rollback retry")?;
    Ok(())
}

#[test]
fn canonical_full_block_oracle_falsifies_native_layer_mutations() -> std::result::Result<(), String>
{
    let fixture = canonical_hybrid_fixture_with_context(16)?;
    let mut canonical = CanonicalHybridOracle::from_fixture(&fixture)?;
    let width = canonical.hidden_width();
    let mut without_ffn = CanonicalHybridOracle::from_fixture(&fixture)?.without_ffn();
    let mut without_attention =
        CanonicalHybridOracle::from_fixture(&fixture)?.without_full_attention();
    let mut contiguous =
        CanonicalHybridOracle::from_fixture(&fixture)?.with_contiguous_q_gate_halves();
    let mut modulo = CanonicalHybridOracle::from_fixture(&fixture)?.with_modulo_gqa();
    let mut adjacent = CanonicalHybridOracle::from_fixture(&fixture)?.with_adjacent_pairs();
    let full_block = 3;
    let mut expected_path = Vec::new();
    let mut variants = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for position in 0..9 {
        let input = (0..width)
            .map(|index| [0.25_f64, -0.5, 0.75][(position + index) % 3])
            .collect::<Vec<_>>();
        let expected = canonical.full_block_step(full_block, &input)?;
        expected_path.extend(expected);
        variants[0].extend(without_ffn.full_block_step(full_block, &input)?);
        variants[1].extend(without_attention.full_block_step(full_block, &input)?);
        variants[2].extend(contiguous.full_block_step(full_block, &input)?);
        variants[3].extend(modulo.full_block_step(full_block, &input)?);
        variants[4].extend(adjacent.full_block_step(full_block, &input)?);
    }
    for (label, alternate) in ["FFN", "full attention", "Q/gate", "GQA", "rotary"]
        .into_iter()
        .zip(variants)
    {
        assert_oracle_difference(&expected_path, &alternate, label)?;
    }
    Ok(())
}

#[test]
fn execution_uses_source_defined_rope_defaults_and_refuses_effective_scaling()
-> std::result::Result<(), String> {
    let mut missing_scaling = fixture(1)?;
    missing_scaling
        .metadata
        .retain(|entry| entry.key() != "qwen35.rope.scaling.type");
    let payload = verify_fixture(&missing_scaling)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    assert!(
        weights.execution(1).is_ok(),
        "an absent optional scaling key has source-defined unscaled semantics"
    );

    let mut scaled = fixture(1)?;
    replace_metadata(
        &mut scaled,
        MetadataEntry::String("qwen35.rope.scaling.type", "yarn".to_string()),
    )?;
    let payload = verify_fixture(&scaled)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    assert!(
        weights.execution(1).is_err(),
        "unsupported scaling must refuse rather than silently become unscaled"
    );
    Ok(())
}

#[test]
fn recurrent_step_does_not_commit_state_when_late_output_row_refuses()
-> std::result::Result<(), String> {
    let mut fixture = fixture(1)?;
    for name in [
        "blk.0.attn_norm.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ssm_alpha.weight",
        "blk.0.ssm_beta.weight",
        "blk.0.ssm_conv1d.weight",
        "blk.0.ssm_norm.weight",
        "blk.0.ssm_out.weight",
    ] {
        set_f32_repeated(&mut fixture, name, 1.0)?;
    }
    set_f32_repeated(&mut fixture, "blk.0.ssm_a", -1.0)?;
    set_f32_value(
        &mut fixture,
        "blk.0.ssm_out.weight",
        usize::from(8_u8),
        f32::NAN,
    )?;
    let payload = verify_fixture(&fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    let mut execution = weights
        .recurrent_execution(0)
        .map_err(|error| error.to_string())?;

    let error = execution.step(&[1.0_f32, -2.0, 3.0]);
    assert!(error.is_err(), "a late non-finite output row must refuse");
    assert!(
        execution.state_for_test().iter().all(|value| *value == 0.0),
        "the output projection failed after local recurrence, so caller-owned state must remain unchanged"
    );
    Ok(())
}

#[test]
fn recurrent_trunk_matches_independent_asymmetric_f64_oracle() -> std::result::Result<(), String> {
    let mut fixture = fixture(1)?;
    configure_asymmetric_recurrent_payload(&mut fixture)?;
    let payload = verify_fixture(&fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    let inputs = [1.0_f32, -2.0, 0.5, -1.5, 0.25, 2.0];
    let (expected_output, expected_state) = recurrent_oracle(&inputs)?;

    let mut batched = weights
        .recurrent_execution(0)
        .map_err(|error| error.to_string())?;
    let actual_output = batched.step(&inputs).map_err(|error| error.to_string())?;
    assert_f32_matches_f64(&actual_output, &expected_output, "batched recurrent output")?;
    assert_f32_matches_f64(
        batched.state_for_test(),
        &expected_state,
        "batched recurrent state",
    )?;

    let mut sequential = weights
        .recurrent_execution(0)
        .map_err(|error| error.to_string())?;
    let mut sequential_output = sequential
        .step(&inputs[..usize::from(3_u8)])
        .map_err(|error| error.to_string())?;
    sequential_output.extend(
        sequential
            .step(&inputs[usize::from(3_u8)..])
            .map_err(|error| error.to_string())?,
    );
    assert_eq!(
        actual_output, sequential_output,
        "a two-token step must preserve the same state transition order as two one-token steps"
    );
    Ok(())
}

#[test]
fn rejects_wrong_full_attention_query_shape() -> std::result::Result<(), String> {
    let mut fixture = fixture(1)?;
    mutate_tensor_shape(
        &mut fixture,
        "blk.3.attn_q.weight",
        vec![TEST_HIDDEN, TEST_HEAD_WIDTH],
    )?;
    let artifact = observe_fixture(&fixture)?;

    let error = preflight_error(&artifact)?;
    assert!(
        error.to_string().contains("blk.3.attn_q.weight"),
        "shape refusal must identify the mutated Q role"
    );
    Ok(())
}

#[test]
fn rejects_ungroupable_recurrent_value_heads() -> std::result::Result<(), String> {
    let mut fixture = fixture(1)?;
    set_u32(&mut fixture, SSM_INNER_SIZE_KEY, 6)?;
    set_u32(&mut fixture, SSM_TIME_STEP_RANK_KEY, 3)?;
    let artifact = observe_fixture(&fixture)?;

    let error = preflight_error(&artifact)?;
    assert!(
        error
            .to_string()
            .contains("ssm.time_step_rank must be divisible by ssm.group_count"),
        "Qwen tiled V-head layouts require a whole number of values per key-head group"
    );
    Ok(())
}

#[test]
fn rejects_wrong_full_attention_cadence() -> std::result::Result<(), String> {
    let mut fixture = fixture(1)?;
    set_u32(&mut fixture, FULL_ATTENTION_INTERVAL_KEY, 3)?;
    let artifact = observe_fixture(&fixture)?;

    let error = preflight_error(&artifact)?;
    assert!(
        error.to_string().contains("unclassified tensor"),
        "cadence mismatch must refuse the unexpected recurrent role"
    );
    Ok(())
}

#[test]
fn rejects_missing_nextn_role() -> std::result::Result<(), String> {
    let mut fixture = fixture(1)?;
    fixture
        .tensors
        .retain(|tensor| tensor.name != "blk.4.nextn.eh_proj.weight");
    let artifact = observe_fixture(&fixture)?;

    let error = preflight_error(&artifact)?;
    assert!(
        error.to_string().contains("tensor count"),
        "role inventory count must reject a missing terminal role"
    );
    Ok(())
}

#[test]
fn rejects_extra_unclassified_role() -> std::result::Result<(), String> {
    let mut fixture = fixture(1)?;
    rename_tensor(
        &mut fixture,
        "blk.4.nextn.hnorm.weight",
        "blk.4.nextn.extra.weight",
    )?;
    let artifact = observe_fixture(&fixture)?;

    let error = preflight_error(&artifact)?;
    assert!(
        error.to_string().contains("unclassified tensor"),
        "extra role must be named in the refusal"
    );
    Ok(())
}

#[test]
fn rejects_inconsistent_ssm_dimensions() -> std::result::Result<(), String> {
    let mut fixture = fixture(1)?;
    set_u32(&mut fixture, SSM_INNER_SIZE_KEY, 12)?;
    let artifact = observe_fixture(&fixture)?;

    let error = preflight_error(&artifact)?;
    assert!(
        error.to_string().contains(SSM_STATE_SIZE_KEY),
        "SSM relation refusal must identify state-size metadata"
    );
    Ok(())
}

#[test]
fn rejects_non_u32_shape_metadata() -> std::result::Result<(), String> {
    let mut fixture = fixture(1)?;
    replace_metadata(&mut fixture, MetadataEntry::F32(EMBEDDING_LENGTH_KEY, 2.0))?;
    let artifact = observe_fixture(&fixture)?;

    let error = preflight_error(&artifact)?;
    assert!(
        error.to_string().contains("must be u32"),
        "typed-metadata refusal must preserve the expected type"
    );
    Ok(())
}

#[test]
fn rejects_checked_projection_overflow() -> std::result::Result<(), String> {
    let mut fixture = fixture(1)?;
    set_u32(&mut fixture, HEAD_COUNT_KEY, u32::MAX)?;
    set_u32(&mut fixture, KEY_VALUE_HEAD_COUNT_KEY, u32::MAX)?;
    set_u32(&mut fixture, KEY_LENGTH_KEY, u32::MAX)?;
    set_u32(&mut fixture, VALUE_LENGTH_KEY, u32::MAX)?;
    let artifact = observe_fixture(&fixture)?;

    let error = preflight_error(&artifact)?;
    assert!(
        error.to_string().contains("arithmetic overflow"),
        "overflow must remain a typed structural refusal"
    );
    Ok(())
}

#[test]
fn rejects_huge_block_count_before_inventory_allocation() -> std::result::Result<(), String> {
    let mut fixture = fixture(1)?;
    set_u32(&mut fixture, BLOCK_COUNT_KEY, u32::MAX)?;
    let artifact = observe_fixture(&fixture)?;

    let error = preflight_error(&artifact)?;
    assert!(
        error.to_string().contains("tensor count"),
        "inventory count must refuse before constructing an enormous role map"
    );
    Ok(())
}

#[test]
fn rejects_more_than_one_nextn_block() -> std::result::Result<(), String> {
    let mut fixture = fixture(1)?;
    set_u32(&mut fixture, NEXTN_PREDICT_LAYERS_KEY, 2)?;
    let artifact = observe_fixture(&fixture)?;

    let error = preflight_error(&artifact)?;
    assert!(
        error.to_string().contains("zero or one NextN block"),
        "bounded profile must refuse a second NextN block"
    );
    Ok(())
}

#[test]
fn rejects_explicit_recurrent_layer_override() -> std::result::Result<(), String> {
    let mut fixture = fixture(1)?;
    fixture
        .metadata
        .push(MetadataEntry::U32(RECURRENT_LAYERS_KEY, 0));
    let artifact = observe_fixture(&fixture)?;

    let error = preflight_error(&artifact)?;
    assert!(
        error.to_string().contains(RECURRENT_LAYERS_KEY),
        "interval-only profile must refuse an explicit recurrent-layer override"
    );
    Ok(())
}

#[test]
fn loader_refuses_duplicate_tensors_before_profile_construction() -> std::result::Result<(), String>
{
    let mut fixture = fixture(1)?;
    let duplicate = fixture
        .tensors
        .first()
        .cloned()
        .ok_or_else(|| "fixture must contain a global tensor".to_string())?;
    fixture.tensors.push(duplicate);

    let Err(error) = observe_fixture(&fixture) else {
        return Err("opaque observation must reject duplicate descriptors".to_string());
    };
    assert!(
        error.contains("duplicate tensor name"),
        "loader boundary must reject duplicates before a profile exists"
    );
    Ok(())
}

#[test]
fn projects_verified_multiblock_q8_rows() -> std::result::Result<(), String> {
    let mut fixture = fixture_with_feed_forward(1, TEST_PROJECTION_INPUT_WIDTH)?;
    set_q8_payload(
        &mut fixture,
        "blk.0.ffn_down.weight",
        projection_payload(false),
    )?;
    let payload = verify_fixture(&fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    let activations = ordered_projection_activations()?;

    let output = weights
        .project("blk.0.ffn_down.weight", &activations)
        .map_err(|error| error.to_string())?;

    assert_eq!(
        output,
        vec![2080.0, -496.0, 1536.0],
        "three nonzero Q8_0 rows across two blocks each must affect the projection"
    );
    assert_eq!(
        output.len(),
        TEST_PROJECTION_OUTPUT_WIDTH,
        "the second GGUF dimension is the number of contiguous output rows"
    );
    Ok(())
}

#[test]
fn projection_rejects_wrong_name_rank_and_width_but_executes_iq4_nl()
-> std::result::Result<(), String> {
    let mut fixture = fixture_with_feed_forward(1, TEST_PROJECTION_INPUT_WIDTH)?;
    set_q8_payload(
        &mut fixture,
        "blk.0.ffn_down.weight",
        projection_payload(false),
    )?;
    let payload = verify_fixture(&fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    let activations = vec![
        1.0;
        usize::try_from(TEST_PROJECTION_INPUT_WIDTH).map_err(|error| {
            format!("test projection width must fit usize: {error}")
        })?
    ];

    let wrong_name = weights.project("blk.0.not_a_role.weight", &activations);
    assert!(
        matches!(wrong_name, Err(crate::Error::PayloadTensor { .. })),
        "unknown tensor names must not create a projection"
    );

    let wrong_rank = weights.project(OUTPUT_NORM_TENSOR, &activations);
    assert!(
        matches!(wrong_rank, Err(crate::Error::ProjectionRank { .. })),
        "recognized rank-one tensors must not be treated as matrices"
    );

    let mut iq4_fixture = fixture_with_feed_forward(1, TEST_PROJECTION_INPUT_WIDTH)?;
    set_iq4_nl_payload(
        &mut iq4_fixture,
        "blk.0.ffn_down.weight",
        iq4_nl_projection_payload(),
    )?;
    let iq4_payload = verify_fixture(&iq4_fixture)?;
    let iq4_weights =
        Qwen35Weights::try_from_verified(&iq4_payload).map_err(|error| error.to_string())?;
    let iq4_output = iq4_weights
        .project("blk.0.ffn_down.weight", &activations)
        .map_err(|error| error.to_string())?;
    assert_eq!(
        iq4_output,
        vec![448.0, -4_064.0, 1_808.0],
        "IQ4_NL projection must decode all nonzero synthetic rows before dotting activations"
    );

    let wrong_width = weights.project("blk.0.ffn_down.weight", &activations[..63]);
    assert!(
        matches!(wrong_width, Err(crate::Error::ProjectionInputWidth { .. })),
        "activation width must exactly match the first GGUF matrix dimension"
    );
    Ok(())
}

#[test]
fn projection_drops_local_output_when_a_late_q8_row_is_nonfinite() -> std::result::Result<(), String>
{
    let mut fixture = fixture_with_feed_forward(1, TEST_PROJECTION_INPUT_WIDTH)?;
    set_q8_payload(
        &mut fixture,
        "blk.0.ffn_down.weight",
        projection_payload(true),
    )?;
    let payload = verify_fixture(&fixture)?;
    let weights = Qwen35Weights::try_from_verified(&payload).map_err(|error| error.to_string())?;
    let activations = vec![
        1.0;
        usize::try_from(TEST_PROJECTION_INPUT_WIDTH).map_err(|error| {
            format!("test projection width must fit usize: {error}")
        })?
    ];

    let result = weights.project("blk.0.ffn_down.weight", &activations);
    assert!(
        matches!(result, Err(crate::Error::ProjectionRow { row: 1, .. })),
        "the second output row's non-finite serialized scale must refuse without returning row zero"
    );
    Ok(())
}

fn fixture(nextn_block_count: u64) -> std::result::Result<Fixture, String> {
    fixture_with_feed_forward(nextn_block_count, TEST_FEED_FORWARD)
}

fn fixture_with_feed_forward(
    nextn_block_count: u64,
    feed_forward: u64,
) -> std::result::Result<Fixture, String> {
    let stored_block_count = TEST_MAIN_BLOCKS
        .checked_add(nextn_block_count)
        .ok_or_else(|| "test stored block count overflowed".to_string())?;
    let mut tensors = Vec::new();
    add_tensor(
        &mut tensors,
        TOKEN_EMBEDDING_TENSOR,
        vec![TEST_HIDDEN, TEST_VOCABULARY],
    );
    add_tensor(&mut tensors, OUTPUT_NORM_TENSOR, vec![TEST_HIDDEN]);
    add_tensor(
        &mut tensors,
        OUTPUT_TENSOR,
        vec![TEST_HIDDEN, TEST_VOCABULARY],
    );
    for block_index in 0..TEST_MAIN_BLOCKS {
        if (block_index + 1).is_multiple_of(TEST_FULL_ATTENTION_INTERVAL) {
            add_full_attention_block(&mut tensors, block_index, feed_forward);
        } else {
            add_recurrent_block(&mut tensors, block_index, feed_forward);
        }
    }
    if nextn_block_count == 1 {
        add_nextn_block(&mut tensors, TEST_MAIN_BLOCKS, feed_forward);
    }

    Ok(Fixture {
        metadata: vec![
            MetadataEntry::String(ARCHITECTURE_KEY, ARCHITECTURE_VALUE.to_string()),
            MetadataEntry::U32(BLOCK_COUNT_KEY, to_u32(stored_block_count)?),
            MetadataEntry::U32(NEXTN_PREDICT_LAYERS_KEY, to_u32(nextn_block_count)?),
            MetadataEntry::U32(
                FULL_ATTENTION_INTERVAL_KEY,
                to_u32(TEST_FULL_ATTENTION_INTERVAL)?,
            ),
            MetadataEntry::U32(EMBEDDING_LENGTH_KEY, to_u32(TEST_HIDDEN)?),
            MetadataEntry::U32(FEED_FORWARD_LENGTH_KEY, to_u32(feed_forward)?),
            MetadataEntry::U32(HEAD_COUNT_KEY, to_u32(TEST_HEADS)?),
            MetadataEntry::U32(KEY_VALUE_HEAD_COUNT_KEY, to_u32(TEST_KEY_VALUE_HEADS)?),
            MetadataEntry::U32(KEY_LENGTH_KEY, to_u32(TEST_HEAD_WIDTH)?),
            MetadataEntry::U32(VALUE_LENGTH_KEY, to_u32(TEST_HEAD_WIDTH)?),
            MetadataEntry::U32(SSM_CONV_KERNEL_KEY, to_u32(TEST_CONV_KERNEL)?),
            MetadataEntry::U32(SSM_INNER_SIZE_KEY, to_u32(TEST_INNER)?),
            MetadataEntry::U32(SSM_STATE_SIZE_KEY, to_u32(TEST_STATE)?),
            MetadataEntry::U32(SSM_TIME_STEP_RANK_KEY, to_u32(TEST_TIME_STEP_RANK)?),
            MetadataEntry::U32(SSM_GROUP_COUNT_KEY, to_u32(TEST_GROUP_COUNT)?),
            MetadataEntry::F32(LAYERNORM_RMS_EPSILON_KEY, TEST_RMS_EPSILON),
            MetadataEntry::StringArray(
                TOKENS_KEY,
                (0..TEST_VOCABULARY)
                    .map(|index| format!("token-{index}"))
                    .collect(),
            ),
            MetadataEntry::U32("general.alignment", TEST_ALIGNMENT),
            MetadataEntry::U32("qwen35.context_length", 8),
            MetadataEntry::I32Array("qwen35.rope.dimension_sections", vec![1, 0, 0, 0]),
            MetadataEntry::F32("qwen35.rope.freq_base", 10_000.0),
            MetadataEntry::String("qwen35.rope.scaling.type", "none".to_string()),
        ],
        tensors,
    })
}

fn add_full_attention_block(tensors: &mut Vec<FixtureTensor>, block_index: u64, feed_forward: u64) {
    add_tensor(
        tensors,
        &block_tensor_name(block_index, ATTN_K_ROLE),
        vec![TEST_HIDDEN, TEST_HEAD_WIDTH],
    );
    add_tensor(
        tensors,
        &block_tensor_name(block_index, ATTN_K_NORM_ROLE),
        vec![TEST_HEAD_WIDTH],
    );
    add_tensor(
        tensors,
        &block_tensor_name(block_index, ATTN_NORM_ROLE),
        vec![TEST_HIDDEN],
    );
    add_tensor(
        tensors,
        &block_tensor_name(block_index, ATTN_OUTPUT_ROLE),
        vec![TEST_FULL_ATTENTION_OUTPUT_WIDTH, TEST_HIDDEN],
    );
    add_tensor(
        tensors,
        &block_tensor_name(block_index, ATTN_Q_ROLE),
        vec![TEST_HIDDEN, TEST_Q_WIDTH],
    );
    add_tensor(
        tensors,
        &block_tensor_name(block_index, ATTN_Q_NORM_ROLE),
        vec![TEST_HEAD_WIDTH],
    );
    add_tensor(
        tensors,
        &block_tensor_name(block_index, ATTN_V_ROLE),
        vec![TEST_HIDDEN, TEST_HEAD_WIDTH],
    );
    add_ffn_tensors(tensors, block_index, feed_forward);
    add_tensor(
        tensors,
        &block_tensor_name(block_index, POST_ATTENTION_NORM_ROLE),
        vec![TEST_HIDDEN],
    );
}

fn add_recurrent_block(tensors: &mut Vec<FixtureTensor>, block_index: u64, feed_forward: u64) {
    add_tensor(
        tensors,
        &block_tensor_name(block_index, ATTN_GATE_ROLE),
        vec![TEST_HIDDEN, TEST_INNER],
    );
    add_tensor(
        tensors,
        &block_tensor_name(block_index, ATTN_NORM_ROLE),
        vec![TEST_HIDDEN],
    );
    add_tensor(
        tensors,
        &block_tensor_name(block_index, ATTN_QKV_ROLE),
        vec![TEST_HIDDEN, TEST_SSM_CONV_WIDTH],
    );
    add_ffn_tensors(tensors, block_index, feed_forward);
    add_tensor(
        tensors,
        &block_tensor_name(block_index, POST_ATTENTION_NORM_ROLE),
        vec![TEST_HIDDEN],
    );
    add_tensor(
        tensors,
        &block_tensor_name(block_index, SSM_A_ROLE),
        vec![TEST_TIME_STEP_RANK],
    );
    add_tensor(
        tensors,
        &block_tensor_name(block_index, SSM_ALPHA_ROLE),
        vec![TEST_HIDDEN, TEST_TIME_STEP_RANK],
    );
    add_tensor(
        tensors,
        &block_tensor_name(block_index, SSM_BETA_ROLE),
        vec![TEST_HIDDEN, TEST_TIME_STEP_RANK],
    );
    add_tensor(
        tensors,
        &block_tensor_name(block_index, SSM_CONV1D_ROLE),
        vec![TEST_CONV_KERNEL, TEST_SSM_CONV_WIDTH],
    );
    add_tensor(
        tensors,
        &block_tensor_name(block_index, SSM_DT_ROLE),
        vec![TEST_TIME_STEP_RANK],
    );
    add_tensor(
        tensors,
        &block_tensor_name(block_index, SSM_NORM_ROLE),
        vec![TEST_STATE],
    );
    add_tensor(
        tensors,
        &block_tensor_name(block_index, SSM_OUT_ROLE),
        vec![TEST_INNER, TEST_HIDDEN],
    );
}

fn add_nextn_block(tensors: &mut Vec<FixtureTensor>, block_index: u64, feed_forward: u64) {
    add_full_attention_block(tensors, block_index, feed_forward);
    add_tensor(
        tensors,
        &block_tensor_name(block_index, NEXTN_EH_PROJ_ROLE),
        vec![TEST_NEXTN_PROJECTION_WIDTH, TEST_HIDDEN],
    );
    add_tensor(
        tensors,
        &block_tensor_name(block_index, NEXTN_ENORM_ROLE),
        vec![TEST_HIDDEN],
    );
    add_tensor(
        tensors,
        &block_tensor_name(block_index, NEXTN_HNORM_ROLE),
        vec![TEST_HIDDEN],
    );
    add_tensor(
        tensors,
        &block_tensor_name(block_index, NEXTN_SHARED_HEAD_NORM_ROLE),
        vec![TEST_HIDDEN],
    );
}

fn add_ffn_tensors(tensors: &mut Vec<FixtureTensor>, block_index: u64, feed_forward: u64) {
    add_tensor(
        tensors,
        &block_tensor_name(block_index, FFN_DOWN_ROLE),
        vec![feed_forward, TEST_HIDDEN],
    );
    add_tensor(
        tensors,
        &block_tensor_name(block_index, FFN_GATE_ROLE),
        vec![TEST_HIDDEN, feed_forward],
    );
    add_tensor(
        tensors,
        &block_tensor_name(block_index, FFN_UP_ROLE),
        vec![TEST_HIDDEN, feed_forward],
    );
}

fn add_tensor(tensors: &mut Vec<FixtureTensor>, name: &str, dims: Vec<u64>) {
    tensors.push(FixtureTensor {
        name: name.to_string(),
        dims,
        ggml_type: TEST_F32_TYPE_ID,
        payload: Vec::new(),
    });
}

fn set_q8_payload(
    fixture: &mut Fixture,
    name: &str,
    payload: Vec<u8>,
) -> std::result::Result<(), String> {
    let Some(tensor) = fixture
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == name)
    else {
        return Err(format!("fixture tensor `{name}` was not found"));
    };
    tensor.ggml_type = TEST_Q8_0_TYPE_ID;
    tensor.payload = payload;
    Ok(())
}

fn set_quant_payload(
    fixture: &mut Fixture,
    name: &str,
    ggml_type: u32,
    payload: Vec<u8>,
) -> std::result::Result<(), String> {
    let Some(tensor) = fixture
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == name)
    else {
        return Err(format!("fixture tensor `{name}` was not found"));
    };
    tensor.ggml_type = ggml_type;
    tensor.payload = payload;
    Ok(())
}

fn zero_serialized_tensor(fixture: &mut Fixture, name: &str) -> std::result::Result<(), String> {
    let Some(tensor) = fixture
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == name)
    else {
        return Err(format!("fixture tensor `{name}` was not found"));
    };
    tensor.payload.fill(0);
    Ok(())
}

fn set_iq4_nl_payload(
    fixture: &mut Fixture,
    name: &str,
    payload: Vec<u8>,
) -> std::result::Result<(), String> {
    let Some(tensor) = fixture
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == name)
    else {
        return Err(format!("fixture tensor `{name}` was not found"));
    };
    tensor.ggml_type = TEST_IQ4_NL_TYPE_ID;
    tensor.payload = payload;
    Ok(())
}

fn set_f32_repeated(
    fixture: &mut Fixture,
    name: &str,
    value: f32,
) -> std::result::Result<(), String> {
    let Some(tensor) = fixture
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == name)
    else {
        return Err(format!("fixture tensor `{name}` was not found"));
    };
    if tensor.ggml_type != TEST_F32_TYPE_ID {
        return Err(format!("fixture tensor `{name}` must use F32"));
    }
    let value_count = tensor
        .dims
        .iter()
        .copied()
        .try_fold(1_u64, |count, dimension| {
            count
                .checked_mul(dimension)
                .ok_or_else(|| format!("fixture tensor `{name}` value count overflowed"))
        })?;
    let value_count = usize::try_from(value_count)
        .map_err(|error| format!("fixture tensor `{name}` value count exceeds usize: {error}"))?;
    tensor.payload.clear();
    tensor.payload.reserve(
        value_count
            .checked_mul(usize::from(4_u8))
            .ok_or_else(|| format!("fixture tensor `{name}` payload byte count overflowed"))?,
    );
    for _ in 0..value_count {
        tensor.payload.extend(value.to_le_bytes());
    }
    Ok(())
}

pub(crate) fn set_f32_value(
    fixture: &mut Fixture,
    name: &str,
    value_index: usize,
    value: f32,
) -> std::result::Result<(), String> {
    let Some(tensor) = fixture
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == name)
    else {
        return Err(format!("fixture tensor `{name}` was not found"));
    };
    let byte_start = value_index
        .checked_mul(usize::from(4_u8))
        .ok_or_else(|| format!("fixture tensor `{name}` value offset overflowed"))?;
    let byte_end = byte_start
        .checked_add(usize::from(4_u8))
        .ok_or_else(|| format!("fixture tensor `{name}` value end overflowed"))?;
    let Some(slot) = tensor.payload.get_mut(byte_start..byte_end) else {
        return Err(format!(
            "fixture tensor `{name}` has no value {value_index}"
        ));
    };
    slot.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn configure_asymmetric_recurrent_payload(
    fixture: &mut Fixture,
) -> std::result::Result<(), String> {
    set_f32_values(
        &mut *fixture,
        "blk.0.attn_norm.weight",
        vec![1.0, -0.75, 0.5],
    )?;
    set_f32_values(
        &mut *fixture,
        "blk.0.attn_qkv.weight",
        asymmetric_values(usize::from(48_u8), 0),
    )?;
    set_f32_values(
        &mut *fixture,
        "blk.0.attn_gate.weight",
        asymmetric_values(usize::from(24_u8), 1),
    )?;
    set_f32_values(
        &mut *fixture,
        "blk.0.ssm_alpha.weight",
        asymmetric_values(usize::from(12_u8), 2),
    )?;
    set_f32_values(
        &mut *fixture,
        "blk.0.ssm_beta.weight",
        asymmetric_values(usize::from(12_u8), 3),
    )?;
    set_f32_values(
        &mut *fixture,
        "blk.0.ssm_conv1d.weight",
        asymmetric_values(usize::from(32_u8), 4),
    )?;
    set_f32_values(&mut *fixture, "blk.0.ssm_a", vec![-0.4, -0.9, -0.6, -0.8])?;
    set_f32_values(
        &mut *fixture,
        "blk.0.ssm_dt.bias",
        vec![0.15, -0.1, 0.3, 0.05],
    )?;
    set_f32_values(&mut *fixture, "blk.0.ssm_norm.weight", vec![0.7, -1.1])?;
    set_f32_values(
        &mut *fixture,
        "blk.0.ssm_out.weight",
        asymmetric_values(usize::from(24_u8), 5),
    )?;
    Ok(())
}

fn asymmetric_values(value_count: usize, phase: usize) -> Vec<f32> {
    const VALUES: [f32; 7] = [-0.75, -0.25, 0.125, 0.375, 0.625, -0.5, 0.875];
    (0..value_count)
        .map(|index| VALUES[(index + phase) % VALUES.len()])
        .collect()
}

pub(crate) fn canonical_hybrid_fixture() -> std::result::Result<Fixture, String> {
    canonical_hybrid_fixture_with_n_rot(None)
}

pub(crate) fn canonical_hybrid_fixture_with_context(
    context: usize,
) -> std::result::Result<Fixture, String> {
    canonical_hybrid_fixture_with_context_and_rotary(context, None)
}

pub(crate) fn canonical_hybrid_fixture_with_context_and_rotary(
    context: usize,
    n_rot: Option<u64>,
) -> std::result::Result<Fixture, String> {
    let mut fixture = canonical_hybrid_fixture_with_n_rot(n_rot)?;
    set_u32(
        &mut fixture,
        "qwen35.context_length",
        u32::try_from(context).map_err(|error| error.to_string())?,
    )?;
    Ok(fixture)
}

#[expect(
    clippy::too_many_lines,
    reason = "the mixed fixture names every hybrid tensor shape and quantized path explicitly"
)]
pub(crate) fn mixed_quantized_hybrid_fixture() -> std::result::Result<Fixture, String> {
    let mut fixture = fixture(0)?;
    set_u32(&mut fixture, EMBEDDING_LENGTH_KEY, to_u32(MIXED_HIDDEN)?)?;
    set_u32(
        &mut fixture,
        FEED_FORWARD_LENGTH_KEY,
        to_u32(MIXED_FEED_FORWARD)?,
    )?;
    set_u32(&mut fixture, SSM_INNER_SIZE_KEY, to_u32(MIXED_INNER)?)?;
    set_u32(&mut fixture, SSM_STATE_SIZE_KEY, 64)?;
    set_u32(&mut fixture, HEAD_COUNT_KEY, 4)?;
    set_u32(&mut fixture, KEY_VALUE_HEAD_COUNT_KEY, 2)?;
    set_u32(&mut fixture, KEY_LENGTH_KEY, 128)?;
    set_u32(&mut fixture, VALUE_LENGTH_KEY, 128)?;
    mutate_tensor_shape(
        &mut fixture,
        TOKEN_EMBEDDING_TENSOR,
        vec![MIXED_HIDDEN, TEST_VOCABULARY],
    )?;
    mutate_tensor_shape(&mut fixture, OUTPUT_NORM_TENSOR, vec![MIXED_HIDDEN])?;
    mutate_tensor_shape(
        &mut fixture,
        OUTPUT_TENSOR,
        vec![MIXED_HIDDEN, TEST_VOCABULARY],
    )?;
    for block in 0..3_u8 {
        let prefix = format!("blk.{block}");
        for role in [ATTN_NORM_ROLE, POST_ATTENTION_NORM_ROLE] {
            mutate_tensor_shape(
                &mut fixture,
                &format!("{prefix}.{role}"),
                vec![MIXED_HIDDEN],
            )?;
        }
        mutate_tensor_shape(
            &mut fixture,
            &format!("{prefix}.{ATTN_GATE_ROLE}"),
            vec![MIXED_HIDDEN, MIXED_INNER],
        )?;
        mutate_tensor_shape(
            &mut fixture,
            &format!("{prefix}.{ATTN_QKV_ROLE}"),
            vec![MIXED_HIDDEN, MIXED_INNER + 256],
        )?;
        for role in [FFN_DOWN_ROLE, FFN_GATE_ROLE, FFN_UP_ROLE, SSM_OUT_ROLE] {
            mutate_tensor_shape(
                &mut fixture,
                &format!("{prefix}.{role}"),
                vec![MIXED_HIDDEN, MIXED_HIDDEN],
            )?;
        }
        for role in [SSM_ALPHA_ROLE, SSM_BETA_ROLE] {
            mutate_tensor_shape(
                &mut fixture,
                &format!("{prefix}.{role}"),
                vec![MIXED_HIDDEN, TEST_TIME_STEP_RANK],
            )?;
        }
        mutate_tensor_shape(
            &mut fixture,
            &format!("{prefix}.{SSM_CONV1D_ROLE}"),
            vec![TEST_CONV_KERNEL, MIXED_INNER + 256],
        )?;
        mutate_tensor_shape(
            &mut fixture,
            &format!("{prefix}.{SSM_NORM_ROLE}"),
            vec![MIXED_INNER / TEST_TIME_STEP_RANK],
        )?;
    }
    for role in [ATTN_NORM_ROLE, POST_ATTENTION_NORM_ROLE] {
        mutate_tensor_shape(&mut fixture, &format!("blk.3.{role}"), vec![MIXED_HIDDEN])?;
    }
    for role in [FFN_DOWN_ROLE, FFN_GATE_ROLE, FFN_UP_ROLE] {
        mutate_tensor_shape(
            &mut fixture,
            &format!("blk.3.{role}"),
            vec![MIXED_HIDDEN, MIXED_HIDDEN],
        )?;
    }
    mutate_tensor_shape(&mut fixture, "blk.3.attn_k.weight", vec![MIXED_HIDDEN, 256])?;
    mutate_tensor_shape(&mut fixture, "blk.3.attn_v.weight", vec![MIXED_HIDDEN, 256])?;
    mutate_tensor_shape(
        &mut fixture,
        "blk.3.attn_q.weight",
        vec![MIXED_HIDDEN, 1024],
    )?;
    mutate_tensor_shape(
        &mut fixture,
        "blk.3.attn_output.weight",
        vec![512, MIXED_HIDDEN],
    )?;
    mutate_tensor_shape(&mut fixture, "blk.3.attn_k_norm.weight", vec![128])?;
    mutate_tensor_shape(&mut fixture, "blk.3.attn_q_norm.weight", vec![128])?;
    fixture
        .metadata
        .push(MetadataEntry::U32("qwen35.rope.dimension_count", 64));
    replace_metadata(
        &mut fixture,
        MetadataEntry::I32Array(
            "qwen35.rope.dimension_sections",
            CANONICAL_ROPE_SECTIONS.to_vec(),
        ),
    )?;
    replace_metadata(
        &mut fixture,
        MetadataEntry::U32(
            "qwen35.context_length",
            u32::try_from(MIXED_CONTEXT).map_err(|error| error.to_string())?,
        ),
    )?;

    let parameters = fixture
        .tensors
        .iter()
        .map(|tensor| {
            Ok((
                tensor.name.clone(),
                mixed_f32_values(&tensor.name, &tensor.dims)?,
            ))
        })
        .collect::<std::result::Result<Vec<_>, String>>()?;
    for (name, values) in parameters {
        set_f32_values(&mut fixture, &name, values)?;
    }
    set_quant_payload(
        &mut fixture,
        TOKEN_EMBEDDING_TENSOR,
        TEST_Q8_0_TYPE_ID,
        mixed_q8_payload(test_dimension(TEST_VOCABULARY)?, 8)?,
    )?;
    set_quant_payload(
        &mut fixture,
        "blk.0.attn_qkv.weight",
        TEST_Q4_K_TYPE_ID,
        mixed_q4_payload(test_dimension(MIXED_INNER + 256)?)?,
    )?;
    set_quant_payload(
        &mut fixture,
        "blk.0.attn_gate.weight",
        TEST_Q5_K_TYPE_ID,
        mixed_q5_payload(test_dimension(MIXED_INNER)?)?,
    )?;
    set_quant_payload(
        &mut fixture,
        "blk.3.attn_v.weight",
        TEST_Q6_K_TYPE_ID,
        mixed_q6_payload(256)?,
    )?;
    set_quant_payload(
        &mut fixture,
        "blk.3.attn_output.weight",
        TEST_IQ4_NL_TYPE_ID,
        mixed_iq4_nl_payload(256, 16)?,
    )?;
    set_quant_payload(
        &mut fixture,
        OUTPUT_TENSOR,
        TEST_IQ4_XS_TYPE_ID,
        mixed_iq4_xs_payload(test_dimension(TEST_VOCABULARY)?)?,
    )?;
    Ok(fixture)
}

#[expect(
    clippy::too_many_lines,
    reason = "the isolated GGUF fixture names each architecture-owned tensor explicitly"
)]
fn canonical_hybrid_fixture_with_n_rot(n_rot: Option<u64>) -> std::result::Result<Fixture, String> {
    let mut fixture = fixture(0)?;
    set_u32(&mut fixture, HEAD_COUNT_KEY, to_u32(CANONICAL_HEADS)?)?;
    set_u32(
        &mut fixture,
        KEY_VALUE_HEAD_COUNT_KEY,
        to_u32(CANONICAL_KEY_VALUE_HEADS)?,
    )?;
    set_u32(&mut fixture, KEY_LENGTH_KEY, to_u32(CANONICAL_HEAD_WIDTH)?)?;
    set_u32(
        &mut fixture,
        VALUE_LENGTH_KEY,
        to_u32(CANONICAL_HEAD_WIDTH)?,
    )?;
    set_u32(
        &mut fixture,
        "qwen35.context_length",
        u32::try_from(CANONICAL_CONTEXT).map_err(|error| error.to_string())?,
    )?;
    replace_metadata(
        &mut fixture,
        MetadataEntry::I32Array(
            "qwen35.rope.dimension_sections",
            CANONICAL_ROPE_SECTIONS.to_vec(),
        ),
    )?;
    if let Some(n_rot) = n_rot {
        fixture.metadata.push(MetadataEntry::U32(
            "qwen35.rope.dimension_count",
            to_u32(n_rot)?,
        ));
    }
    mutate_tensor_shape(
        &mut fixture,
        "blk.3.attn_k.weight",
        vec![
            TEST_HIDDEN,
            CANONICAL_KEY_VALUE_HEADS * CANONICAL_HEAD_WIDTH,
        ],
    )?;
    mutate_tensor_shape(
        &mut fixture,
        "blk.3.attn_k_norm.weight",
        vec![CANONICAL_HEAD_WIDTH],
    )?;
    mutate_tensor_shape(
        &mut fixture,
        "blk.3.attn_q.weight",
        vec![TEST_HIDDEN, CANONICAL_HEADS * CANONICAL_HEAD_WIDTH * 2],
    )?;
    mutate_tensor_shape(
        &mut fixture,
        "blk.3.attn_q_norm.weight",
        vec![CANONICAL_HEAD_WIDTH],
    )?;
    mutate_tensor_shape(
        &mut fixture,
        "blk.3.attn_v.weight",
        vec![
            TEST_HIDDEN,
            CANONICAL_KEY_VALUE_HEADS * CANONICAL_HEAD_WIDTH,
        ],
    )?;
    mutate_tensor_shape(
        &mut fixture,
        "blk.3.attn_output.weight",
        vec![CANONICAL_HEADS * CANONICAL_HEAD_WIDTH, TEST_HIDDEN],
    )?;

    let parameters = fixture
        .tensors
        .iter()
        .map(|tensor| {
            let count = tensor
                .dims
                .iter()
                .try_fold(1_u64, |total, dimension| total.checked_mul(*dimension))
                .ok_or_else(|| format!("canonical tensor `{}` size overflowed", tensor.name))?;
            let count = usize::try_from(count).map_err(|error| error.to_string())?;
            let phase = tensor
                .name
                .bytes()
                .fold(0_usize, |total, byte| total + usize::from(byte))
                % 7;
            Ok((tensor.name.clone(), asymmetric_values(count, phase)))
        })
        .collect::<std::result::Result<Vec<_>, String>>()?;
    for (name, values) in parameters {
        set_f32_values(&mut fixture, &name, values)?;
    }
    for block in 0..3_u8 {
        set_f32_values(
            &mut fixture,
            &format!("blk.{block}.ssm_a"),
            vec![-0.4, -0.9, -0.6, -0.8],
        )?;
    }
    Ok(fixture)
}

#[expect(
    clippy::struct_excessive_bools,
    reason = "independent booleans deliberately isolate each wrong semantic for falsification"
)]
pub(crate) struct CanonicalHybridOracle {
    tensors: BTreeMap<String, OracleTensor>,
    layout: OracleLayout,
    layers: Vec<OracleLayerState>,
    position: usize,
    include_recurrent_attention: bool,
    include_full_attention: bool,
    include_ffn: bool,
    adjacent_pairs: bool,
    contiguous_q_gate_halves: bool,
    modulo_gqa: bool,
    reverse_embedding_rows: bool,
    reverse_output_rows: bool,
}

pub(crate) type OracleStateSnapshot = (
    usize,
    Vec<(Vec<f64>, Vec<f64>)>,
    Vec<(usize, Vec<f64>, Vec<f64>)>,
);

struct OracleTensor {
    dims: Vec<usize>,
    values: Vec<f64>,
}

#[derive(Clone, Copy)]
struct OracleLayout {
    hidden: usize,
    feed_forward: usize,
    heads: usize,
    kv_heads: usize,
    key: usize,
    n_rot: usize,
    vocabulary: usize,
    main_blocks: usize,
    full_interval: usize,
    epsilon: f64,
    rope_base: f64,
    rope_sections: [usize; 4],
    conv_kernel: usize,
    inner: usize,
    key_dim: usize,
    value_heads: usize,
    key_heads: usize,
    value_dim: usize,
    key_width: usize,
    conv_width: usize,
}

enum OracleLayerState {
    Recurrent(OracleRecurrentState),
    Full(OracleFullState),
}

struct OracleRecurrentState {
    convolution: Vec<f64>,
    gdn: Vec<f64>,
}

struct OracleFullState {
    keys: Vec<f64>,
    values: Vec<f64>,
    tokens: usize,
}

impl CanonicalHybridOracle {
    pub(crate) fn from_fixture(fixture: &Fixture) -> std::result::Result<Self, String> {
        let layout = OracleLayout::from_fixture(fixture)?;
        let mut tensors = BTreeMap::new();
        for tensor in &fixture.tensors {
            tensors.insert(tensor.name.clone(), decode_oracle_tensor(tensor)?);
        }
        let history_width = layout
            .conv_kernel
            .checked_sub(1)
            .ok_or_else(|| "canonical convolution kernel must be nonzero".to_string())?;
        let convolution_len = layout
            .conv_width
            .checked_mul(history_width)
            .ok_or_else(|| "canonical convolution history size overflowed".to_string())?;
        let gdn_len = layout
            .value_heads
            .checked_mul(layout.key_dim)
            .and_then(|width| width.checked_mul(layout.value_dim))
            .ok_or_else(|| "canonical recurrent state size overflowed".to_string())?;
        let layers = (0..layout.main_blocks)
            .map(|block| {
                if (block + 1).is_multiple_of(layout.full_interval) {
                    OracleLayerState::Full(OracleFullState {
                        keys: Vec::new(),
                        values: Vec::new(),
                        tokens: 0,
                    })
                } else {
                    OracleLayerState::Recurrent(OracleRecurrentState {
                        convolution: vec![0.0; convolution_len],
                        gdn: vec![0.0; gdn_len],
                    })
                }
            })
            .collect();
        Ok(Self {
            tensors,
            layout,
            layers,
            position: 0,
            include_recurrent_attention: true,
            include_full_attention: true,
            include_ffn: true,
            adjacent_pairs: false,
            contiguous_q_gate_halves: false,
            modulo_gqa: false,
            reverse_embedding_rows: false,
            reverse_output_rows: false,
        })
    }

    pub(crate) const fn hidden_width(&self) -> usize {
        self.layout.hidden
    }

    fn without_attention(mut self) -> Self {
        self.include_recurrent_attention = false;
        self.include_full_attention = false;
        self
    }

    fn without_recurrent_attention(mut self) -> Self {
        self.include_recurrent_attention = false;
        self
    }

    fn without_full_attention(mut self) -> Self {
        self.include_full_attention = false;
        self
    }

    fn without_ffn(mut self) -> Self {
        self.include_ffn = false;
        self
    }

    fn with_adjacent_pairs(mut self) -> Self {
        self.adjacent_pairs = true;
        self
    }

    fn with_full_rotation(mut self) -> Self {
        self.layout.n_rot = self.layout.key;
        self
    }

    fn with_contiguous_q_gate_halves(mut self) -> Self {
        self.contiguous_q_gate_halves = true;
        self
    }

    fn with_modulo_gqa(mut self) -> Self {
        self.modulo_gqa = true;
        self
    }

    fn with_reversed_embedding_rows(mut self) -> Self {
        self.reverse_embedding_rows = true;
        self
    }

    fn with_reversed_output_rows(mut self) -> Self {
        self.reverse_output_rows = true;
        self
    }

    fn tensor(&self, name: &str) -> std::result::Result<&OracleTensor, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("canonical oracle is missing `{name}`"))
    }

    fn vector(&self, name: &str, width: usize) -> std::result::Result<&[f64], String> {
        let tensor = self.tensor(name)?;
        if tensor.dims != [width] {
            return Err(format!(
                "canonical oracle tensor `{name}` has dims {:?}, expected [{width}]",
                tensor.dims
            ));
        }
        Ok(&tensor.values)
    }

    fn project(
        &self,
        name: &str,
        input: &[f64],
        input_width: usize,
        output_width: usize,
    ) -> std::result::Result<Vec<f64>, String> {
        let tensor = self.tensor(name)?;
        if tensor.dims != [input_width, output_width] {
            return Err(format!(
                "canonical oracle tensor `{name}` has dims {:?}, expected [{input_width}, {output_width}]",
                tensor.dims
            ));
        }
        Ok(oracle_project(
            input,
            &tensor.values,
            input_width,
            output_width,
        ))
    }

    pub(crate) fn step(&mut self, tokens: &[u32]) -> std::result::Result<Vec<f64>, String> {
        let mut logits = Vec::new();
        for token in tokens {
            let mut token = usize::try_from(*token).map_err(|error| error.to_string())?;
            if self.reverse_embedding_rows && token < self.layout.vocabulary {
                token = self.layout.vocabulary - token - 1;
            }
            let embedding = self.tensor(TOKEN_EMBEDDING_TENSOR)?;
            if embedding.dims != [self.layout.hidden, self.layout.vocabulary] {
                return Err("canonical token embedding shape differs from metadata".to_string());
            }
            let start = token
                .checked_mul(self.layout.hidden)
                .ok_or_else(|| "canonical token offset overflowed".to_string())?;
            let mut hidden = embedding
                .values
                .get(start..start + self.layout.hidden)
                .ok_or_else(|| "canonical oracle token must be in vocabulary".to_string())?
                .to_vec();
            for block in 0..self.layout.main_blocks {
                hidden = self.block(block, &hidden)?;
            }
            let normalized = oracle_rms_with_epsilon(
                &hidden,
                self.vector(OUTPUT_NORM_TENSOR, self.layout.hidden)?,
                1,
                self.layout.hidden,
                self.layout.epsilon,
            )?;
            let mut token_logits = self.project(
                OUTPUT_TENSOR,
                &normalized,
                self.layout.hidden,
                self.layout.vocabulary,
            )?;
            if self.reverse_output_rows {
                token_logits.reverse();
            }
            logits.extend(token_logits);
            self.position = self
                .position
                .checked_add(1)
                .ok_or_else(|| "canonical position overflowed".to_string())?;
        }
        Ok(logits)
    }

    pub(crate) fn full_block_step(
        &mut self,
        block: usize,
        input: &[f64],
    ) -> std::result::Result<Vec<f64>, String> {
        if block >= self.layout.main_blocks
            || input.len() != self.layout.hidden
            || !(block + 1).is_multiple_of(self.layout.full_interval)
        {
            return Err(
                "canonical full-block input is outside the main-block hidden shape".to_string(),
            );
        }
        let output = self.block(block, input)?;
        self.position = self
            .position
            .checked_add(1)
            .ok_or_else(|| "canonical position overflowed".to_string())?;
        Ok(output)
    }

    fn block(&mut self, block: usize, input: &[f64]) -> std::result::Result<Vec<f64>, String> {
        let mut hidden = input.to_vec();
        let full = (block + 1).is_multiple_of(self.layout.full_interval);
        let mut attention = if full {
            self.full_attention(block, &hidden)?
        } else {
            self.recurrent_attention(block, &hidden)?
        };
        if (full && !self.include_full_attention) || (!full && !self.include_recurrent_attention) {
            attention.fill(0.0);
        }
        add_f64(&mut hidden, &attention)?;
        let normalized = oracle_rms_with_epsilon(
            &hidden,
            self.vector(
                &format!("blk.{block}.post_attention_norm.weight"),
                self.layout.hidden,
            )?,
            1,
            self.layout.hidden,
            self.layout.epsilon,
        )?;
        let mut ffn = self.ffn(block, &normalized)?;
        if !self.include_ffn {
            ffn.fill(0.0);
        }
        add_f64(&mut hidden, &ffn)?;
        Ok(hidden)
    }

    fn ffn(&self, block: usize, input: &[f64]) -> std::result::Result<Vec<f64>, String> {
        let gate = self.project(
            &format!("blk.{block}.ffn_gate.weight"),
            input,
            self.layout.hidden,
            self.layout.feed_forward,
        )?;
        let up = self.project(
            &format!("blk.{block}.ffn_up.weight"),
            input,
            self.layout.hidden,
            self.layout.feed_forward,
        )?;
        let fused = gate
            .iter()
            .zip(up)
            .map(|(gate, up)| (gate / (1.0 + (-gate).exp())) * up)
            .collect::<Vec<_>>();
        self.project(
            &format!("blk.{block}.ffn_down.weight"),
            &fused,
            self.layout.feed_forward,
            self.layout.hidden,
        )
    }

    #[expect(
        clippy::cast_precision_loss,
        clippy::needless_range_loop,
        clippy::too_many_lines,
        reason = "the scalar oracle deliberately spells out the pinned recurrent state transition"
    )]
    fn recurrent_attention(
        &mut self,
        block: usize,
        input: &[f64],
    ) -> std::result::Result<Vec<f64>, String> {
        let layout = self.layout;
        let normalized = oracle_rms_with_epsilon(
            input,
            self.vector(&format!("blk.{block}.attn_norm.weight"), layout.hidden)?,
            1,
            layout.hidden,
            layout.epsilon,
        )?;
        let qkv = self.project(
            &format!("blk.{block}.attn_qkv.weight"),
            &normalized,
            layout.hidden,
            layout.conv_width,
        )?;
        let z = self.project(
            &format!("blk.{block}.attn_gate.weight"),
            &normalized,
            layout.hidden,
            layout.inner,
        )?;
        let alpha = self.project(
            &format!("blk.{block}.ssm_alpha.weight"),
            &normalized,
            layout.hidden,
            layout.value_heads,
        )?;
        let beta = self.project(
            &format!("blk.{block}.ssm_beta.weight"),
            &normalized,
            layout.hidden,
            layout.value_heads,
        )?;
        let convolution_weights = self.tensor(&format!("blk.{block}.ssm_conv1d.weight"))?;
        if convolution_weights.dims != [layout.conv_kernel, layout.conv_width] {
            return Err(format!("canonical block {block} convolution shape differs"));
        }
        let convolution_weights = convolution_weights.values.clone();
        let a = self
            .vector(&format!("blk.{block}.ssm_a"), layout.value_heads)?
            .to_vec();
        let dt = self
            .vector(&format!("blk.{block}.ssm_dt.bias"), layout.value_heads)?
            .to_vec();
        let norm = self
            .vector(&format!("blk.{block}.ssm_norm.weight"), layout.value_dim)?
            .to_vec();
        let output_weight = self
            .tensor(&format!("blk.{block}.ssm_out.weight"))?
            .values
            .clone();
        let state = self
            .layers
            .get_mut(block)
            .and_then(|layer| match layer {
                OracleLayerState::Recurrent(state) => Some(state),
                OracleLayerState::Full(_) => None,
            })
            .ok_or_else(|| format!("canonical recurrent block {block} is absent"))?;
        let history_width = layout.conv_kernel.saturating_sub(1);
        let mut convolved = Vec::with_capacity(layout.conv_width);
        for channel in 0..layout.conv_width {
            let history_start = channel * history_width;
            let weight_start = channel * layout.conv_kernel;
            let mut sum = 0.0;
            for tap in 0..layout.conv_kernel {
                let sample = if tap < history_width {
                    state.convolution[history_start + tap]
                } else {
                    qkv[channel]
                };
                sum += sample * convolution_weights[weight_start + tap];
            }
            convolved.push(sum);
            if history_width != 0 {
                let history = &mut state.convolution[history_start..history_start + history_width];
                history.rotate_left(1);
                history[history_width - 1] = qkv[channel];
            }
        }
        let convolved = convolved
            .into_iter()
            .map(|value| value / (1.0 + (-value).exp()))
            .collect::<Vec<_>>();
        let mut output = vec![0.0; layout.inner];
        for value_head in 0..layout.value_heads {
            let key_head = value_head % layout.key_heads;
            let q = oracle_l2_with_epsilon(
                &convolved[key_head * layout.key_dim..(key_head + 1) * layout.key_dim],
                layout.epsilon,
            );
            let k = oracle_l2_with_epsilon(
                &convolved[layout.key_width + key_head * layout.key_dim
                    ..layout.key_width + (key_head + 1) * layout.key_dim],
                layout.epsilon,
            );
            let values = &convolved[2 * layout.key_width + value_head * layout.value_dim
                ..2 * layout.key_width + (value_head + 1) * layout.value_dim];
            let beta = 1.0 / (1.0 + (-beta[value_head]).exp());
            let alpha_dt = alpha[value_head] + dt[value_head];
            let gate = a[value_head] * (alpha_dt.max(0.0) + (-alpha_dt.abs()).exp().ln_1p());
            let decay = gate.exp();
            for key in 0..layout.key_dim {
                for value in 0..layout.value_dim {
                    state.gdn[(value_head * layout.key_dim + key) * layout.value_dim + value] *=
                        decay;
                }
            }
            let mut delta = vec![0.0; layout.value_dim];
            for value in 0..layout.value_dim {
                let prior = (0..layout.key_dim)
                    .map(|key| {
                        state.gdn[(value_head * layout.key_dim + key) * layout.value_dim + value]
                            * k[key]
                    })
                    .sum::<f64>();
                delta[value] = beta * (values[value] - prior);
            }
            for key in 0..layout.key_dim {
                for value in 0..layout.value_dim {
                    state.gdn[(value_head * layout.key_dim + key) * layout.value_dim + value] +=
                        k[key] * delta[value];
                }
            }
            for value in 0..layout.value_dim {
                output[value_head * layout.value_dim + value] = (0..layout.key_dim)
                    .map(|key| {
                        state.gdn[(value_head * layout.key_dim + key) * layout.value_dim + value]
                            * q[key]
                            / (layout.key_dim as f64).sqrt()
                    })
                    .sum();
            }
        }
        let output = oracle_rms_with_epsilon(
            &output,
            &norm,
            layout.value_heads,
            layout.value_dim,
            layout.epsilon,
        )?;
        let gated = output
            .iter()
            .zip(z)
            .map(|(output, z)| output * (z / (1.0 + (-z).exp())))
            .collect::<Vec<_>>();
        Ok(oracle_project(
            &gated,
            &output_weight,
            layout.inner,
            layout.hidden,
        ))
    }

    #[expect(
        clippy::cast_precision_loss,
        clippy::too_many_lines,
        reason = "the scalar oracle deliberately keeps pinned full-attention order visible"
    )]
    fn full_attention(
        &mut self,
        block: usize,
        input: &[f64],
    ) -> std::result::Result<Vec<f64>, String> {
        let layout = self.layout;
        let normalized = oracle_rms_with_epsilon(
            input,
            self.vector(&format!("blk.{block}.attn_norm.weight"), layout.hidden)?,
            1,
            layout.hidden,
            layout.epsilon,
        )?;
        let q_gate = self.project(
            &format!("blk.{block}.attn_q.weight"),
            &normalized,
            layout.hidden,
            layout.heads * layout.key * 2,
        )?;
        let (query, gate) = if self.contiguous_q_gate_halves {
            let middle = layout.heads * layout.key;
            (q_gate[..middle].to_vec(), q_gate[middle..].to_vec())
        } else {
            let mut query = Vec::with_capacity(layout.heads * layout.key);
            let mut gate = Vec::with_capacity(layout.heads * layout.key);
            for head in 0..layout.heads {
                let base = head * layout.key * 2;
                query.extend_from_slice(&q_gate[base..base + layout.key]);
                gate.extend_from_slice(&q_gate[base + layout.key..base + 2 * layout.key]);
            }
            (query, gate)
        };
        let mut query = oracle_rms_with_epsilon(
            query.as_slice(),
            self.vector(&format!("blk.{block}.attn_q_norm.weight"), layout.key)?,
            layout.heads,
            layout.key,
            layout.epsilon,
        )?;
        let key = self.project(
            &format!("blk.{block}.attn_k.weight"),
            &normalized,
            layout.hidden,
            layout.kv_heads * layout.key,
        )?;
        let mut key = oracle_rms_with_epsilon(
            &key,
            self.vector(&format!("blk.{block}.attn_k_norm.weight"), layout.key)?,
            layout.kv_heads,
            layout.key,
            layout.epsilon,
        )?;
        let value = self.project(
            &format!("blk.{block}.attn_v.weight"),
            &normalized,
            layout.hidden,
            layout.kv_heads * layout.key,
        )?;
        canonical_mrope(
            &mut query,
            layout.heads,
            layout.key,
            layout.n_rot,
            self.position,
            layout.rope_base,
            layout.rope_sections,
            self.adjacent_pairs,
        )?;
        canonical_mrope(
            &mut key,
            layout.kv_heads,
            layout.key,
            layout.n_rot,
            self.position,
            layout.rope_base,
            layout.rope_sections,
            self.adjacent_pairs,
        )?;
        let state = self
            .layers
            .get_mut(block)
            .and_then(|layer| match layer {
                OracleLayerState::Recurrent(_) => None,
                OracleLayerState::Full(state) => Some(state),
            })
            .ok_or_else(|| format!("canonical full-attention block {block} is absent"))?;
        state.keys.extend_from_slice(&key);
        state.values.extend_from_slice(&value);
        state.tokens += 1;
        let kv_width = layout.kv_heads * layout.key;
        let mut merged = Vec::with_capacity(layout.heads * layout.key);
        for head in 0..layout.heads {
            let kv_head = if self.modulo_gqa {
                head % layout.kv_heads
            } else {
                head / (layout.heads / layout.kv_heads)
            };
            let query = &query[head * layout.key..(head + 1) * layout.key];
            let scores = (0..state.tokens)
                .map(|token| {
                    let start = token * kv_width + kv_head * layout.key;
                    query
                        .iter()
                        .zip(&state.keys[start..start + layout.key])
                        .map(|(left, right)| left * right)
                        .sum::<f64>()
                        / (layout.key as f64).sqrt()
                })
                .collect::<Vec<_>>();
            let probabilities = oracle_softmax(&scores)?;
            for lane in 0..layout.key {
                let attended = probabilities
                    .iter()
                    .enumerate()
                    .map(|(token, probability)| {
                        probability * state.values[token * kv_width + kv_head * layout.key + lane]
                    })
                    .sum::<f64>();
                let gate = gate[head * layout.key + lane];
                merged.push(attended / (1.0 + (-gate).exp()));
            }
        }
        Ok(oracle_project(
            &merged,
            &self
                .tensor(&format!("blk.{block}.attn_output.weight"))?
                .values,
            layout.heads * layout.key,
            layout.hidden,
        ))
    }

    pub(crate) fn state_for_test(&self) -> OracleStateSnapshot {
        (
            self.position,
            self.layers
                .iter()
                .filter_map(|layer| match layer {
                    OracleLayerState::Recurrent(state) => {
                        Some((state.convolution.clone(), state.gdn.clone()))
                    }
                    OracleLayerState::Full(_) => None,
                })
                .collect(),
            self.layers
                .iter()
                .filter_map(|layer| match layer {
                    OracleLayerState::Recurrent(_) => None,
                    OracleLayerState::Full(state) => {
                        Some((state.tokens, state.keys.clone(), state.values.clone()))
                    }
                })
                .collect(),
        )
    }
}

impl OracleLayout {
    #[expect(
        clippy::too_many_lines,
        reason = "the independent oracle validates all metadata-derived dimensions in one boundary"
    )]
    fn from_fixture(fixture: &Fixture) -> std::result::Result<Self, String> {
        let hidden = fixture_u32(fixture, EMBEDDING_LENGTH_KEY)?;
        let feed_forward = fixture_u32(fixture, FEED_FORWARD_LENGTH_KEY)?;
        let heads = fixture_u32(fixture, HEAD_COUNT_KEY)?;
        let kv_heads = fixture_u32(fixture, KEY_VALUE_HEAD_COUNT_KEY)?;
        let key = fixture_u32(fixture, KEY_LENGTH_KEY)?;
        let vocabulary = fixture
            .metadata
            .iter()
            .find_map(|entry| match entry {
                MetadataEntry::StringArray(key, values) if *key == TOKENS_KEY => Some(values.len()),
                _ => None,
            })
            .ok_or_else(|| "canonical oracle vocabulary metadata is absent".to_string())?;
        let main_blocks = fixture_u32(fixture, BLOCK_COUNT_KEY)?
            .checked_sub(fixture_u32(fixture, NEXTN_PREDICT_LAYERS_KEY)?)
            .ok_or_else(|| "canonical oracle main block count underflowed".to_string())?;
        let full_interval = fixture_u32(fixture, FULL_ATTENTION_INTERVAL_KEY)?;
        let conv_kernel = fixture_u32(fixture, SSM_CONV_KERNEL_KEY)?;
        let inner = fixture_u32(fixture, SSM_INNER_SIZE_KEY)?;
        let key_dim = fixture_u32(fixture, SSM_STATE_SIZE_KEY)?;
        let value_heads = fixture_u32(fixture, SSM_TIME_STEP_RANK_KEY)?;
        let key_heads = fixture_u32(fixture, SSM_GROUP_COUNT_KEY)?;
        if hidden == 0
            || feed_forward == 0
            || heads == 0
            || kv_heads == 0
            || key == 0
            || main_blocks == 0
            || full_interval == 0
            || conv_kernel == 0
            || inner == 0
            || key_dim == 0
            || value_heads == 0
            || key_heads == 0
            || !heads.is_multiple_of(kv_heads)
            || !value_heads.is_multiple_of(key_heads)
            || !inner.is_multiple_of(value_heads)
        {
            return Err("canonical oracle metadata dimensions are not executable".to_string());
        }
        let value_dim = inner
            .checked_div(value_heads)
            .ok_or_else(|| "canonical oracle value-head count is zero".to_string())?;
        let key_width = key_heads
            .checked_mul(key_dim)
            .ok_or_else(|| "canonical oracle key width overflowed".to_string())?;
        let conv_width = key_width
            .checked_mul(2)
            .and_then(|width| width.checked_add(inner))
            .ok_or_else(|| "canonical oracle convolution width overflowed".to_string())?;
        let n_rot = fixture_optional_u32(fixture, "qwen35.rope.dimension_count")?.unwrap_or(key);
        let sections = fixture_i32_array(fixture, "qwen35.rope.dimension_sections")?;
        let rope_sections = <[usize; 4]>::try_from(
            sections
                .iter()
                .map(|section| usize::try_from(*section).map_err(|error| error.to_string()))
                .collect::<std::result::Result<Vec<_>, _>>()?,
        )
        .map_err(|sections| {
            format!("canonical oracle needs four RoPE sections, got {sections:?}")
        })?;
        Ok(Self {
            hidden,
            feed_forward,
            heads,
            kv_heads,
            key,
            n_rot,
            vocabulary,
            main_blocks,
            full_interval,
            epsilon: f64::from(fixture_f32(fixture, LAYERNORM_RMS_EPSILON_KEY)?),
            rope_base: f64::from(fixture_f32(fixture, "qwen35.rope.freq_base")?),
            rope_sections,
            conv_kernel,
            inner,
            key_dim,
            value_heads,
            key_heads,
            value_dim,
            key_width,
            conv_width,
        })
    }
}

fn fixture_u32(fixture: &Fixture, key: &str) -> std::result::Result<usize, String> {
    fixture_optional_u32(fixture, key)?
        .ok_or_else(|| format!("canonical oracle metadata `{key}` is absent"))
}

fn fixture_optional_u32(
    fixture: &Fixture,
    key: &str,
) -> std::result::Result<Option<usize>, String> {
    fixture
        .metadata
        .iter()
        .find_map(|entry| match entry {
            MetadataEntry::U32(entry_key, value) if *entry_key == key => Some(*value),
            _ => None,
        })
        .map(usize::try_from)
        .transpose()
        .map_err(|error| format!("canonical oracle metadata `{key}` exceeds usize: {error}"))
}

fn fixture_f32(fixture: &Fixture, key: &str) -> std::result::Result<f32, String> {
    fixture
        .metadata
        .iter()
        .find_map(|entry| match entry {
            MetadataEntry::F32(entry_key, value) if *entry_key == key => Some(*value),
            _ => None,
        })
        .ok_or_else(|| format!("canonical oracle metadata `{key}` is absent"))
}

fn fixture_i32_array<'fixture>(
    fixture: &'fixture Fixture,
    key: &str,
) -> std::result::Result<&'fixture [i32], String> {
    fixture
        .metadata
        .iter()
        .find_map(|entry| match entry {
            MetadataEntry::I32Array(entry_key, values) if *entry_key == key => {
                Some(values.as_slice())
            }
            _ => None,
        })
        .ok_or_else(|| format!("canonical oracle metadata `{key}` is absent"))
}

fn decode_oracle_tensor(tensor: &FixtureTensor) -> std::result::Result<OracleTensor, String> {
    let dims = tensor
        .dims
        .iter()
        .map(|dimension| usize::try_from(*dimension).map_err(|error| error.to_string()))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let value_count = dims.iter().try_fold(1_usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| format!("canonical tensor `{}` size overflowed", tensor.name))
    })?;
    let expected_bytes =
        usize::try_from(tensor_bytes(tensor)?).map_err(|error| error.to_string())?;
    let payload = if tensor.payload.is_empty() {
        vec![0_u8; expected_bytes]
    } else {
        tensor.payload.clone()
    };
    if payload.len() != expected_bytes {
        return Err(format!(
            "canonical tensor `{}` has {} bytes, expected {expected_bytes}",
            tensor.name,
            payload.len()
        ));
    }
    let values = match tensor.ggml_type {
        TEST_F32_TYPE_ID => decode_oracle_f32(&payload)?,
        TEST_Q8_0_TYPE_ID => decode_oracle_q8(&payload)?,
        TEST_Q4_K_TYPE_ID => decode_oracle_q4_k(&payload)?,
        TEST_Q5_K_TYPE_ID => decode_oracle_q5_k(&payload)?,
        TEST_Q6_K_TYPE_ID => decode_oracle_q6_k(&payload)?,
        TEST_IQ4_NL_TYPE_ID => decode_oracle_iq4_nl(&payload)?,
        TEST_IQ4_XS_TYPE_ID => decode_oracle_iq4_xs(&payload)?,
        other => {
            return Err(format!(
                "canonical oracle does not decode GGML type {other}"
            ));
        }
    };
    if values.len() != value_count {
        return Err(format!(
            "canonical tensor `{}` decoded {} values, expected {value_count}",
            tensor.name,
            values.len()
        ));
    }
    Ok(OracleTensor { dims, values })
}

fn decode_oracle_f32(bytes: &[u8]) -> std::result::Result<Vec<f64>, String> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err("canonical F32 payload is not scalar-aligned".to_string());
    }
    Ok(chunks
        .map(|value| f64::from(f32::from_le_bytes([value[0], value[1], value[2], value[3]])))
        .collect())
}

fn decode_oracle_q8(bytes: &[u8]) -> std::result::Result<Vec<f64>, String> {
    let blocks = bytes.chunks_exact(ORACLE_Q8_BLOCK_BYTES);
    if !blocks.remainder().is_empty() {
        return Err("canonical Q8_0 payload is not block-aligned".to_string());
    }
    let mut values = Vec::with_capacity(blocks.len() * ORACLE_Q8_VALUES);
    for block in blocks {
        let scale = oracle_f16(u16::from_le_bytes([block[0], block[1]]))?;
        values.extend(
            block[2..]
                .iter()
                .map(|value| scale * f64::from(i8::from_le_bytes([*value]))),
        );
    }
    Ok(values)
}

fn decode_oracle_q4_k(bytes: &[u8]) -> std::result::Result<Vec<f64>, String> {
    let blocks = bytes.chunks_exact(ORACLE_Q4_BLOCK_BYTES);
    if !blocks.remainder().is_empty() {
        return Err("canonical Q4_K payload is not block-aligned".to_string());
    }
    let mut values = Vec::with_capacity(blocks.len() * ORACLE_K_VALUES);
    for block in blocks {
        let scale = oracle_f16(u16::from_le_bytes([block[0], block[1]]))?;
        let minimum = oracle_f16(u16::from_le_bytes([block[2], block[3]]))?;
        let scales: &[u8; ORACLE_K_SCALE_BYTES] = block[4..16]
            .try_into()
            .map_err(|error| format!("canonical Q4_K scale header differs: {error}"))?;
        let quantized = &block[16..];
        for pair in 0..4 {
            let (low_scale, low_minimum) = oracle_k_scale_min(scales, pair * 2);
            let (high_scale, high_minimum) = oracle_k_scale_min(scales, pair * 2 + 1);
            for lane in 0..ORACLE_K_GROUP_VALUES {
                let packed = quantized[pair * ORACLE_K_GROUP_VALUES + lane];
                values.push(
                    scale * f64::from(low_scale) * f64::from(packed & 0x0f)
                        - minimum * f64::from(low_minimum),
                );
            }
            for lane in 0..ORACLE_K_GROUP_VALUES {
                let packed = quantized[pair * ORACLE_K_GROUP_VALUES + lane];
                values.push(
                    scale * f64::from(high_scale) * f64::from(packed >> 4)
                        - minimum * f64::from(high_minimum),
                );
            }
        }
    }
    Ok(values)
}

fn decode_oracle_q5_k(bytes: &[u8]) -> std::result::Result<Vec<f64>, String> {
    let blocks = bytes.chunks_exact(ORACLE_Q5_BLOCK_BYTES);
    if !blocks.remainder().is_empty() {
        return Err("canonical Q5_K payload is not block-aligned".to_string());
    }
    let mut values = Vec::with_capacity(blocks.len() * ORACLE_K_VALUES);
    for block in blocks {
        let scale = oracle_f16(u16::from_le_bytes([block[0], block[1]]))?;
        let minimum = oracle_f16(u16::from_le_bytes([block[2], block[3]]))?;
        let scales: &[u8; ORACLE_K_SCALE_BYTES] = block[4..16]
            .try_into()
            .map_err(|error| format!("canonical Q5_K scale header differs: {error}"))?;
        let high_bits = &block[16..48];
        let quantized = &block[48..];
        for pair in 0..4 {
            let (low_scale, low_minimum) = oracle_k_scale_min(scales, pair * 2);
            let (high_scale, high_minimum) = oracle_k_scale_min(scales, pair * 2 + 1);
            let low_bit = 1_u8 << (pair * 2);
            let high_bit = low_bit << 1;
            for lane in 0..ORACLE_K_GROUP_VALUES {
                let packed = quantized[pair * ORACLE_K_GROUP_VALUES + lane];
                let fifth = if high_bits[lane] & low_bit == 0 {
                    0
                } else {
                    16
                };
                let code = (packed & 0x0f) + fifth;
                values.push(
                    scale * f64::from(low_scale) * f64::from(code)
                        - minimum * f64::from(low_minimum),
                );
            }
            for lane in 0..ORACLE_K_GROUP_VALUES {
                let packed = quantized[pair * ORACLE_K_GROUP_VALUES + lane];
                let fifth = if high_bits[lane] & high_bit == 0 {
                    0
                } else {
                    16
                };
                let code = (packed >> 4) + fifth;
                values.push(
                    scale * f64::from(high_scale) * f64::from(code)
                        - minimum * f64::from(high_minimum),
                );
            }
        }
    }
    Ok(values)
}

fn decode_oracle_q6_k(bytes: &[u8]) -> std::result::Result<Vec<f64>, String> {
    let blocks = bytes.chunks_exact(ORACLE_Q6_BLOCK_BYTES);
    if !blocks.remainder().is_empty() {
        return Err("canonical Q6_K payload is not block-aligned".to_string());
    }
    let mut values = Vec::with_capacity(blocks.len() * ORACLE_K_VALUES);
    for block in blocks {
        let low = &block[..128];
        let high = &block[128..192];
        let scales = &block[192..208];
        let super_scale = oracle_f16(u16::from_le_bytes([block[208], block[209]]))?;
        for half_block in 0..2 {
            for quarter in 0..4 {
                for lane in 0..32 {
                    let low_byte = low[half_block * 64 + (quarter % 2) * 32 + lane];
                    let lower = if quarter < 2 {
                        low_byte & 0x0f
                    } else {
                        low_byte >> 4
                    };
                    let upper = (high[half_block * 32 + lane] >> (quarter * 2)) & 0x03;
                    let code = i16::from((upper << 4) | lower) - 32;
                    let scale_index = half_block * 8 + quarter * 2 + lane / 16;
                    let group_scale = i8::from_le_bytes([scales[scale_index]]);
                    values.push(super_scale * f64::from(group_scale) * f64::from(code));
                }
            }
        }
    }
    Ok(values)
}

fn decode_oracle_iq4_nl(bytes: &[u8]) -> std::result::Result<Vec<f64>, String> {
    let blocks = bytes.chunks_exact(ORACLE_IQ4_NL_BLOCK_BYTES);
    if !blocks.remainder().is_empty() {
        return Err("canonical IQ4_NL payload is not block-aligned".to_string());
    }
    let mut values = Vec::with_capacity(blocks.len() * ORACLE_IQ4_NL_VALUES);
    for block in blocks {
        let scale = oracle_f16(u16::from_le_bytes([block[0], block[1]]))?;
        for packed in &block[2..] {
            values.push(scale * f64::from(ORACLE_IQ4_RECONSTRUCTION[usize::from(*packed & 0x0f)]));
        }
        for packed in &block[2..] {
            values.push(scale * f64::from(ORACLE_IQ4_RECONSTRUCTION[usize::from(*packed >> 4)]));
        }
    }
    Ok(values)
}

fn decode_oracle_iq4_xs(bytes: &[u8]) -> std::result::Result<Vec<f64>, String> {
    let blocks = bytes.chunks_exact(ORACLE_IQ4_XS_BLOCK_BYTES);
    if !blocks.remainder().is_empty() {
        return Err("canonical IQ4_XS payload is not block-aligned".to_string());
    }
    let mut values = Vec::with_capacity(blocks.len() * ORACLE_IQ4_XS_VALUES);
    for block in blocks {
        let block_scale = oracle_f16(u16::from_le_bytes([block[0], block[1]]))?;
        let scale_high = u16::from_le_bytes([block[2], block[3]]);
        let scale_low = &block[4..8];
        let quantized = &block[8..];
        for group in 0..8_usize {
            let low = if group.is_multiple_of(2) {
                scale_low[group / 2] & 0x0f
            } else {
                scale_low[group / 2] >> 4
            };
            let high = u8::try_from((scale_high >> (group * 2)) & 0x03)
                .map_err(|error| error.to_string())?;
            let group_scale = i16::from(low | (high << 4)) - 32;
            let group_quantized = &quantized[group * 16..group * 16 + 16];
            for packed in group_quantized {
                values.push(
                    block_scale
                        * f64::from(group_scale)
                        * f64::from(ORACLE_IQ4_RECONSTRUCTION[usize::from(*packed & 0x0f)]),
                );
            }
            for packed in group_quantized {
                values.push(
                    block_scale
                        * f64::from(group_scale)
                        * f64::from(ORACLE_IQ4_RECONSTRUCTION[usize::from(*packed >> 4)]),
                );
            }
        }
    }
    Ok(values)
}

fn oracle_k_scale_min(scales: &[u8; ORACLE_K_SCALE_BYTES], group: usize) -> (u8, u8) {
    if group < 4 {
        return (scales[group] & 0x3f, scales[group + 4] & 0x3f);
    }
    let packed = scales[group + 4];
    (
        (packed & 0x0f) | ((scales[group - 4] >> 6) << 4),
        (packed >> 4) | ((scales[group] >> 6) << 4),
    )
}

fn oracle_f16(bits: u16) -> std::result::Result<f64, String> {
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x03ff;
    if exponent == 0x1f {
        return Err(format!(
            "canonical oracle encountered non-finite fp16 0x{bits:04x}"
        ));
    }
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    if exponent == 0 {
        return Ok(sign * f64::from(mantissa) * 2.0_f64.powi(-24));
    }
    Ok(sign * (1.0 + f64::from(mantissa) / 1024.0) * 2.0_f64.powi(i32::from(exponent) - 15))
}

#[expect(
    clippy::cast_precision_loss,
    clippy::too_many_arguments,
    reason = "the f64 oracle intentionally converts checked small fixture indexes to angles"
)]
fn canonical_mrope(
    values: &mut [f64],
    rows: usize,
    width: usize,
    n_rot: usize,
    position: usize,
    rope_base: f64,
    sections: [usize; 4],
    adjacent_pairs: bool,
) -> std::result::Result<(), String> {
    let section_total = sections
        .iter()
        .try_fold(0_usize, |total, section| total.checked_add(*section))
        .ok_or_else(|| "canonical IMRoPE section total overflowed".to_string())?;
    if values.len() != rows * width
        || n_rot == 0
        || !n_rot.is_multiple_of(2)
        || n_rot > width
        || section_total == 0
    {
        return Err(
            "canonical IMRoPE witness width does not match its serialized layout".to_string(),
        );
    }
    let positions = [position as f64, position as f64, position as f64, 0.0];
    for row in values.chunks_exact_mut(width) {
        for pair in 0..(n_rot / 2) {
            let sector = pair % section_total;
            let axis = if sector % 3 == 1 && sector < 3 * sections[1] {
                1
            } else if sector % 3 == 2 && sector < 3 * sections[2] {
                2
            } else if sector.is_multiple_of(3) && sector < 3 * sections[0] {
                0
            } else {
                3
            };
            let angle = positions[axis] / rope_base.powf((2 * pair) as f64 / n_rot as f64);
            let (sine, cosine) = angle.sin_cos();
            let (left, right) = if adjacent_pairs {
                (pair * 2, pair * 2 + 1)
            } else {
                (pair, pair + n_rot / 2)
            };
            let left_value = row[left];
            let right_value = row[right];
            row[left] = left_value * cosine - right_value * sine;
            row[right] = left_value * sine + right_value * cosine;
        }
    }
    Ok(())
}

fn oracle_softmax(scores: &[f64]) -> std::result::Result<Vec<f64>, String> {
    let maximum = scores
        .iter()
        .copied()
        .reduce(f64::max)
        .ok_or_else(|| "canonical attention has no causal scores".to_string())?;
    let exponentials = scores
        .iter()
        .map(|score| (score - maximum).exp())
        .collect::<Vec<_>>();
    let total = exponentials.iter().sum::<f64>();
    Ok(exponentials
        .into_iter()
        .map(|value| value / total)
        .collect())
}

fn add_f64(left: &mut [f64], right: &[f64]) -> std::result::Result<(), String> {
    if left.len() != right.len() {
        return Err("canonical residual widths differ".to_string());
    }
    for (left, right) in left.iter_mut().zip(right) {
        *left += right;
    }
    Ok(())
}

fn assert_oracle_difference(
    expected: &[f64],
    alternate: &[f64],
    path: &str,
) -> std::result::Result<(), String> {
    if expected.len() != alternate.len() {
        return Err(format!(
            "canonical {path} falsifier length differs: expected {}, alternate {}",
            expected.len(),
            alternate.len()
        ));
    }
    let mut greatest = 0.0_f64;
    let mut distinguishes = false;
    for (index, (expected, alternate)) in expected.iter().zip(alternate).enumerate() {
        if !expected.is_finite() || !alternate.is_finite() {
            return Err(format!(
                "canonical {path} falsifier is non-finite at {index}: expected {expected}, alternate {alternate}"
            ));
        }
        let delta = (expected - alternate).abs();
        let tolerance = oracle_tolerance(*expected, *alternate);
        greatest = greatest.max(delta);
        distinguishes |= delta > tolerance;
    }
    if !distinguishes {
        return Err(format!(
            "canonical fixture is insensitive to the required {path}: greatest logit delta {greatest} does not exceed the production-oracle tolerance"
        ));
    }
    Ok(())
}

fn assert_oracle_variant(
    fixture: &Fixture,
    expected: &[f64],
    tokens: &[u32],
    path: &str,
    configure: fn(CanonicalHybridOracle) -> CanonicalHybridOracle,
) -> std::result::Result<(), String> {
    let mut alternate = configure(CanonicalHybridOracle::from_fixture(fixture)?);
    let alternate_logits = alternate.step(tokens)?;
    assert_oracle_difference(expected, &alternate_logits, path)
}

fn set_f32_values(
    fixture: &mut Fixture,
    name: &str,
    values: Vec<f32>,
) -> std::result::Result<(), String> {
    let Some(tensor) = fixture
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == name)
    else {
        return Err(format!("fixture tensor `{name}` was not found"));
    };
    if tensor.ggml_type != TEST_F32_TYPE_ID {
        return Err(format!("fixture tensor `{name}` must use F32"));
    }
    let expected = tensor
        .dims
        .iter()
        .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| format!("fixture tensor `{name}` value count overflowed"))?;
    let expected = usize::try_from(expected)
        .map_err(|error| format!("fixture tensor `{name}` value count exceeds usize: {error}"))?;
    if values.len() != expected {
        return Err(format!(
            "fixture tensor `{name}` needs {expected} F32 values, got {}",
            values.len()
        ));
    }
    tensor.payload.clear();
    for value in values {
        tensor.payload.extend(value.to_le_bytes());
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct OracleShape {
    hidden: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_width: usize,
}

fn recurrent_oracle(inputs: &[f32]) -> std::result::Result<(Vec<f64>, Vec<f64>), String> {
    let shape = OracleShape {
        hidden: test_dimension(TEST_HIDDEN)?,
        key_heads: test_dimension(TEST_GROUP_COUNT)?,
        value_heads: test_dimension(TEST_TIME_STEP_RANK)?,
        key_dim: test_dimension(TEST_STATE)?,
        value_dim: test_dimension(TEST_STATE)?,
        conv_width: test_dimension(TEST_SSM_CONV_WIDTH)?,
    };
    let token_count = inputs
        .len()
        .checked_div(shape.hidden)
        .ok_or_else(|| "oracle hidden width is zero".to_string())?;
    let attention_norm = [1.0_f64, -0.75, 0.5];
    let normalized = oracle_rms(
        &inputs.iter().copied().map(f64::from).collect::<Vec<_>>(),
        &attention_norm,
        token_count,
        shape.hidden,
    )?;
    let qkv = oracle_project(
        &normalized,
        &asymmetric_values(48, 0),
        shape.hidden,
        shape.conv_width,
    );
    let z = oracle_project(
        &normalized,
        &asymmetric_values(24, 1),
        shape.hidden,
        shape.value_heads * shape.value_dim,
    );
    let alpha = oracle_project(
        &normalized,
        &asymmetric_values(12, 2),
        shape.hidden,
        shape.value_heads,
    );
    let beta = oracle_project(
        &normalized,
        &asymmetric_values(12, 3),
        shape.hidden,
        shape.value_heads,
    );
    let conv = oracle_conv(
        &qkv,
        &asymmetric_values(32, 4),
        token_count,
        shape.conv_width,
    );
    let convolved = conv
        .iter()
        .map(|value| value / (1.0 + (-value).exp()))
        .collect::<Vec<_>>();
    let (recurrent_output, state) =
        oracle_recurrence(&convolved, &alpha, &beta, token_count, shape)?;
    let output_norm = oracle_rms(
        &recurrent_output,
        &[0.7, -1.1],
        token_count * shape.value_heads,
        shape.value_dim,
    )?;
    let gated = output_norm
        .iter()
        .zip(z)
        .map(|(output, gate)| output * (gate / (1.0 + (-gate).exp())))
        .collect::<Vec<_>>();
    Ok((
        oracle_project(
            &gated,
            &asymmetric_values(24, 5),
            shape.value_heads * shape.value_dim,
            shape.hidden,
        ),
        state,
    ))
}

fn oracle_project<W>(
    input: &[f64],
    weights: &[W],
    input_width: usize,
    output_width: usize,
) -> Vec<f64>
where
    W: Copy + Into<f64>,
{
    input
        .chunks_exact(input_width)
        .flat_map(|row| {
            weights
                .chunks_exact(input_width)
                .take(output_width)
                .map(move |weight| {
                    row.iter()
                        .zip(weight)
                        .map(|(input, weight)| input * (*weight).into())
                        .sum()
                })
        })
        .collect()
}

fn oracle_rms(
    input: &[f64],
    weight: &[f64],
    rows: usize,
    width: usize,
) -> std::result::Result<Vec<f64>, String> {
    oracle_rms_with_epsilon(input, weight, rows, width, f64::from(TEST_RMS_EPSILON))
}

fn oracle_rms_with_epsilon(
    input: &[f64],
    weight: &[f64],
    rows: usize,
    width: usize,
    epsilon: f64,
) -> std::result::Result<Vec<f64>, String> {
    let width_f64 = f64::from(u32::try_from(width).map_err(|error| error.to_string())?);
    Ok(input
        .chunks_exact(width)
        .take(rows)
        .flat_map(|row| {
            let mean_square = row.iter().map(|value| value * value).sum::<f64>() / width_f64;
            let inverse = (mean_square + epsilon).sqrt().recip();
            row.iter()
                .zip(weight)
                .map(move |(value, weight)| value * inverse * weight)
        })
        .collect())
}

fn oracle_conv(input: &[f64], weights: &[f32], token_count: usize, channels: usize) -> Vec<f64> {
    let mut output = Vec::with_capacity(token_count * channels);
    for token in 0..token_count {
        for channel in 0..channels {
            let current = input[token * channels + channel];
            let previous = if token == 0 {
                0.0
            } else {
                input[(token - 1) * channels + channel]
            };
            let weight = &weights[channel * usize::from(2_u8)..][..usize::from(2_u8)];
            output.push(previous * f64::from(weight[0]) + current * f64::from(weight[1]));
        }
    }
    output
}

fn oracle_recurrence(
    convolved: &[f64],
    alpha: &[f64],
    beta: &[f64],
    token_count: usize,
    shape: OracleShape,
) -> std::result::Result<(Vec<f64>, Vec<f64>), String> {
    let mut state = vec![0.0; shape.value_heads * shape.key_dim * shape.value_dim];
    let mut output = vec![0.0; token_count * shape.value_heads * shape.value_dim];
    let a = [-0.4_f64, -0.9, -0.6, -0.8];
    let dt = [0.15_f64, -0.1, 0.3, 0.05];
    let scale = f64::from(u32::try_from(shape.key_dim).map_err(|error| error.to_string())?)
        .sqrt()
        .recip();
    for token in 0..token_count {
        let channels = &convolved[token * shape.conv_width..(token + 1) * shape.conv_width];
        for value_head in 0..shape.value_heads {
            let key_head = value_head % shape.key_heads;
            let q = oracle_l2(&channels[key_head * shape.key_dim..(key_head + 1) * shape.key_dim]);
            let k_offset = shape.key_heads * shape.key_dim;
            let k = oracle_l2(
                &channels[k_offset + key_head * shape.key_dim
                    ..k_offset + (key_head + 1) * shape.key_dim],
            );
            let v_offset = 2 * shape.key_heads * shape.key_dim + value_head * shape.value_dim;
            let v = &channels[v_offset..v_offset + shape.value_dim];
            let beta_value = 1.0 / (1.0 + (-beta[token * shape.value_heads + value_head]).exp());
            let alpha_dt = alpha[token * shape.value_heads + value_head] + dt[value_head];
            let gate = a[value_head] * (alpha_dt.max(0.0) + (-alpha_dt.abs()).exp().ln_1p());
            let decay = gate.exp();
            for key in 0..shape.key_dim {
                for value in 0..shape.value_dim {
                    let state_index = (value_head * shape.key_dim + key) * shape.value_dim + value;
                    state[state_index] *= decay;
                }
            }
            let mut delta = vec![0.0; shape.value_dim];
            for value in 0..shape.value_dim {
                let state_projection = (0..shape.key_dim)
                    .map(|key| {
                        state[(value_head * shape.key_dim + key) * shape.value_dim + value] * k[key]
                    })
                    .sum::<f64>();
                delta[value] = beta_value * (v[value] - state_projection);
            }
            for (key, key_value) in k.iter().copied().enumerate() {
                for (value, delta_value) in delta.iter().copied().enumerate() {
                    let state_index = (value_head * shape.key_dim + key) * shape.value_dim + value;
                    state[state_index] += key_value * delta_value;
                }
            }
            for value in 0..shape.value_dim {
                output[(token * shape.value_heads + value_head) * shape.value_dim + value] = (0
                    ..shape.key_dim)
                    .map(|key| {
                        state[(value_head * shape.key_dim + key) * shape.value_dim + value]
                            * q[key]
                            * scale
                    })
                    .sum();
            }
        }
    }
    Ok((output, state))
}

fn oracle_l2(values: &[f64]) -> Vec<f64> {
    oracle_l2_with_epsilon(values, f64::from(TEST_RMS_EPSILON))
}

fn oracle_l2_with_epsilon(values: &[f64], epsilon: f64) -> Vec<f64> {
    let denominator = values
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
        .max(epsilon);
    values.iter().map(|value| value / denominator).collect()
}

fn test_dimension(value: u64) -> std::result::Result<usize, String> {
    usize::try_from(value).map_err(|error| error.to_string())
}

pub(crate) fn assert_f32_matches_f64(
    actual: &[f32],
    expected: &[f64],
    subject: &str,
) -> std::result::Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{subject} length differs: actual {}, expected {}",
            actual.len(),
            expected.len()
        ));
    }
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let actual = f64::from(*actual);
        if !actual.is_finite() || !expected.is_finite() {
            return Err(format!(
                "{subject} is non-finite at {index}: actual {actual}, expected {expected}"
            ));
        }
        let delta = (actual - expected).abs();
        let tolerance = oracle_tolerance(actual, *expected);
        if delta > tolerance {
            return Err(format!(
                "{subject} differs at {index}: actual {actual}, expected {expected}, delta {delta}, tolerance {tolerance}"
            ));
        }
    }
    Ok(())
}

fn oracle_tolerance(left: f64, right: f64) -> f64 {
    ORACLE_ABSOLUTE_TOLERANCE + ORACLE_RELATIVE_TOLERANCE * left.abs().max(right.abs())
}

fn projection_payload(late_nonfinite_row: bool) -> Vec<u8> {
    let positive = [1_i8; quant::q8_0::Q8_0_VALUES_PER_BLOCK];
    let doubled = [2_i8; quant::q8_0::Q8_0_VALUES_PER_BLOCK];
    let negative = [-1_i8; quant::q8_0::Q8_0_VALUES_PER_BLOCK];
    let alternating = std::array::from_fn(|index| {
        if index.is_multiple_of(TEST_ALTERNATING_PERIOD) {
            1
        } else {
            -1
        }
    });
    let mut payload = Vec::new();
    append_q8_block(&mut payload, TEST_Q8_SCALE_ONE_BITS, positive);
    append_q8_block(&mut payload, TEST_Q8_SCALE_ONE_BITS, positive);
    append_q8_block(
        &mut payload,
        if late_nonfinite_row {
            TEST_Q8_SCALE_NONFINITE_BITS
        } else {
            TEST_Q8_SCALE_ONE_BITS
        },
        doubled,
    );
    append_q8_block(&mut payload, TEST_Q8_SCALE_ONE_BITS, negative);
    append_q8_block(&mut payload, TEST_Q8_SCALE_ONE_BITS, alternating);
    append_q8_block(&mut payload, TEST_Q8_SCALE_ONE_BITS, positive);
    payload
}

fn iq4_nl_projection_payload() -> Vec<u8> {
    let mut payload = Vec::new();
    append_iq4_nl_block(&mut payload, 0x3c00, 0x88);
    append_iq4_nl_block(&mut payload, 0x3c00, 0x99);
    append_iq4_nl_block(&mut payload, 0x3800, 0x00);
    append_iq4_nl_block(&mut payload, 0x3800, 0x00);
    append_iq4_nl_block(&mut payload, 0x3400, 0xff);
    append_iq4_nl_block(&mut payload, 0x3400, 0xff);
    payload
}

fn append_iq4_nl_block(payload: &mut Vec<u8>, scale_bits: u16, packed_codepoints: u8) {
    payload.extend_from_slice(&scale_bits.to_le_bytes());
    payload.extend([packed_codepoints; quant::iq4_nl::IQ4_NL_QUANT_BYTES]);
}

fn append_q8_block(
    payload: &mut Vec<u8>,
    scale_bits: u16,
    values: [i8; quant::q8_0::Q8_0_VALUES_PER_BLOCK],
) {
    payload.extend_from_slice(&scale_bits.to_le_bytes());
    payload.extend(values.map(|value| value.to_le_bytes()[0]));
}

#[expect(
    clippy::too_many_lines,
    reason = "the asymmetric literal generator handles vector, convolution, and matrix layouts"
)]
fn mixed_f32_values(name: &str, dims: &[u64]) -> std::result::Result<Vec<f32>, String> {
    let dims = dims
        .iter()
        .map(|dimension| usize::try_from(*dimension).map_err(|error| error.to_string()))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let phase = name
        .bytes()
        .fold(0_usize, |sum, byte| sum + usize::from(byte))
        % 17;
    if let [width] = dims.as_slice() {
        let mut values = Vec::with_capacity(*width);
        for index in 0..*width {
            let value = if name.ends_with("ssm_a") {
                -0.3125
                    - f32::from(
                        u8::try_from((index + phase) % 4).map_err(|error| error.to_string())?,
                    ) * 0.0625
            } else if name.ends_with("ssm_dt.bias") {
                -0.046_875
                    + f32::from(
                        u8::try_from((index + phase) % 5).map_err(|error| error.to_string())?,
                    ) * 0.031_25
            } else {
                0.75 + f32::from(
                    u8::try_from((index + phase) % 7).map_err(|error| error.to_string())?,
                ) * 0.046_875
            };
            values.push(value);
        }
        return Ok(values);
    }
    let [input_width, output_width] = dims.as_slice() else {
        return Err(format!(
            "mixed F32 tensor `{name}` must have rank one or two"
        ));
    };
    if name.ends_with("ssm_conv1d.weight") {
        if *input_width != 2 {
            return Err(format!("mixed convolution `{name}` must have width two"));
        }
        let mut values = Vec::with_capacity(*output_width * 2);
        for channel in 0..*output_width {
            let adjustment =
                f32::from(u8::try_from((channel + phase) % 5).map_err(|error| error.to_string())?)
                    * 0.015_625;
            values.extend([0.1875 + adjustment, 0.4375 - adjustment]);
        }
        return Ok(values);
    }
    let capacity = input_width
        .checked_mul(*output_width)
        .ok_or_else(|| format!("mixed F32 tensor `{name}` size overflowed"))?;
    let mut values = Vec::with_capacity(capacity);
    let amplitude = if name == "blk.1.ssm_out.weight" {
        8.0
    } else {
        1.0
    };
    for row in 0..*output_width {
        let primary = (row * 17 + phase * 7) % input_width;
        let secondary = (row * 29 + phase * 11 + 1) % input_width;
        let tertiary = (row * 43 + phase * 13 + 2) % input_width;
        let sign = if (row + phase).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        for column in 0..*input_width {
            let mut value = 0.0;
            if column == primary {
                value += sign
                    * (0.156_25
                        + f32::from(u8::try_from(row % 4).map_err(|error| error.to_string())?)
                            * 0.015_625);
            }
            if column == secondary {
                value -= sign
                    * (0.109_375
                        + f32::from(
                            u8::try_from((row + phase) % 3).map_err(|error| error.to_string())?,
                        ) * 0.015_625);
            }
            if column == tertiary {
                value += sign * 0.078_125;
            }
            values.push(value * amplitude);
        }
    }
    Ok(values)
}

fn mixed_q8_payload(rows: usize, blocks_per_row: usize) -> std::result::Result<Vec<u8>, String> {
    let block_count = rows
        .checked_mul(blocks_per_row)
        .ok_or_else(|| "mixed Q8_0 block count overflowed".to_string())?;
    let mut bytes = Vec::with_capacity(block_count * ORACLE_Q8_BLOCK_BYTES);
    for row in 0..rows {
        for block in 0..blocks_per_row {
            let scale_bits = 0x2000_u16
                + u16::try_from((row + block) % 4).map_err(|error| error.to_string())? * 0x0080;
            bytes.extend_from_slice(&scale_bits.to_le_bytes());
            for lane in 0..ORACLE_Q8_VALUES {
                let raw = i16::try_from((row * 37 + block * 19 + lane * 7) % 15)
                    .map_err(|error| error.to_string())?
                    - 7;
                let value = if raw == 0 {
                    if (row + block + lane).is_multiple_of(2) {
                        1
                    } else {
                        -1
                    }
                } else {
                    i8::try_from(raw).map_err(|error| error.to_string())?
                };
                bytes.push(value.to_le_bytes()[0]);
            }
        }
    }
    Ok(bytes)
}

fn mixed_q4_payload(rows: usize) -> std::result::Result<Vec<u8>, String> {
    let mut bytes = Vec::with_capacity(rows * ORACLE_Q4_BLOCK_BYTES);
    for row in 0..rows {
        let mut block = [0_u8; ORACLE_Q4_BLOCK_BYTES];
        block[..2].copy_from_slice(&0x1400_u16.to_le_bytes());
        block[2..4].copy_from_slice(&0x1400_u16.to_le_bytes());
        let (scales, minimums) = mixed_k_scale_values(row, 0)?;
        block[4..16].copy_from_slice(&pack_k_scale_min(scales, minimums));
        for pair in 0..4 {
            for lane in 0..ORACLE_K_GROUP_VALUES {
                let low = u8::try_from((row * 5 + pair * 11 + lane * 3) % 16)
                    .map_err(|error| error.to_string())?;
                let high = u8::try_from((row * 7 + pair * 13 + lane * 5 + 3) % 16)
                    .map_err(|error| error.to_string())?;
                block[16 + pair * ORACLE_K_GROUP_VALUES + lane] = low | (high << 4);
            }
        }
        bytes.extend_from_slice(&block);
    }
    Ok(bytes)
}

fn mixed_q5_payload(rows: usize) -> std::result::Result<Vec<u8>, String> {
    let mut bytes = Vec::with_capacity(rows * ORACLE_Q5_BLOCK_BYTES);
    for row in 0..rows {
        let mut block = [0_u8; ORACLE_Q5_BLOCK_BYTES];
        block[..2].copy_from_slice(&0x1000_u16.to_le_bytes());
        block[2..4].copy_from_slice(&0x1000_u16.to_le_bytes());
        let (scales, minimums) = mixed_k_scale_values(row, 9)?;
        block[4..16].copy_from_slice(&pack_k_scale_min(scales, minimums));
        for group in 0..8 {
            let pair = group / 2;
            for lane in 0..ORACLE_K_GROUP_VALUES {
                let code = u8::try_from((row * 11 + group * 17 + lane * 7 + 5) % 32)
                    .map_err(|error| error.to_string())?;
                let quant_index = 48 + pair * ORACLE_K_GROUP_VALUES + lane;
                if group.is_multiple_of(2) {
                    block[quant_index] |= code & 0x0f;
                } else {
                    block[quant_index] |= (code & 0x0f) << 4;
                }
                block[16 + lane] |= ((code >> 4) & 1) << group;
            }
        }
        bytes.extend_from_slice(&block);
    }
    Ok(bytes)
}

fn mixed_q6_payload(rows: usize) -> std::result::Result<Vec<u8>, String> {
    const SCALES: [i8; 8] = [-4, -2, -1, 1, 2, 3, -3, 4];
    let mut bytes = Vec::with_capacity(rows * ORACLE_Q6_BLOCK_BYTES);
    for row in 0..rows {
        let mut block = [0_u8; ORACLE_Q6_BLOCK_BYTES];
        for scale in 0..16 {
            block[192 + scale] = SCALES[(row + scale) % SCALES.len()].to_le_bytes()[0];
        }
        block[208..].copy_from_slice(&0x0c00_u16.to_le_bytes());
        for half_block in 0..2 {
            for quarter in 0..4 {
                for lane in 0..32 {
                    let signed =
                        i16::try_from((row * 13 + half_block * 23 + quarter * 11 + lane * 5) % 49)
                            .map_err(|error| error.to_string())?
                            - 24;
                    let code = u8::try_from(signed + 32).map_err(|error| error.to_string())?;
                    let low_index = half_block * 64 + (quarter % 2) * 32 + lane;
                    if quarter < 2 {
                        block[low_index] |= code & 0x0f;
                    } else {
                        block[low_index] |= (code & 0x0f) << 4;
                    }
                    block[128 + half_block * 32 + lane] |= (code >> 4) << (quarter * 2);
                }
            }
        }
        bytes.extend_from_slice(&block);
    }
    Ok(bytes)
}

fn mixed_iq4_nl_payload(
    rows: usize,
    blocks_per_row: usize,
) -> std::result::Result<Vec<u8>, String> {
    let block_count = rows
        .checked_mul(blocks_per_row)
        .ok_or_else(|| "mixed IQ4_NL block count overflowed".to_string())?;
    let mut bytes = Vec::with_capacity(block_count * ORACLE_IQ4_NL_BLOCK_BYTES);
    for row in 0..rows {
        for block in 0..blocks_per_row {
            let scale_bits = 0x1800_u16
                + u16::try_from((row + block) % 4).map_err(|error| error.to_string())? * 0x0080;
            bytes.extend_from_slice(&scale_bits.to_le_bytes());
            for lane in 0..16 {
                let low = u8::try_from((row * 3 + block * 5 + lane * 7) % 16)
                    .map_err(|error| error.to_string())?;
                let high = u8::try_from((row * 11 + block * 13 + lane * 3 + 1) % 16)
                    .map_err(|error| error.to_string())?;
                bytes.push(low | (high << 4));
            }
        }
    }
    Ok(bytes)
}

fn mixed_iq4_xs_payload(rows: usize) -> std::result::Result<Vec<u8>, String> {
    const GROUP_SCALES: [i16; 8] = [-7, -4, -2, -1, 2, 3, 5, 7];
    let mut bytes = Vec::with_capacity(rows * ORACLE_IQ4_XS_BLOCK_BYTES);
    for row in 0..rows {
        let mut block = [0_u8; ORACLE_IQ4_XS_BLOCK_BYTES];
        block[..2].copy_from_slice(
            &(0x0040_u16 + u16::try_from(row % 4).map_err(|error| error.to_string())? * 0x0010)
                .to_le_bytes(),
        );
        let mut high = 0_u16;
        for group in 0..8 {
            let encoded = u8::try_from(GROUP_SCALES[(row + group) % GROUP_SCALES.len()] + 32)
                .map_err(|error| error.to_string())?;
            if group.is_multiple_of(2) {
                block[4 + group / 2] |= encoded & 0x0f;
            } else {
                block[4 + group / 2] |= (encoded & 0x0f) << 4;
            }
            high |= u16::from(encoded >> 4) << (group * 2);
            for lane in 0..16 {
                let low = u8::try_from((row * 5 + group * 7 + lane * 3) % 16)
                    .map_err(|error| error.to_string())?;
                let high_code = u8::try_from((row * 11 + group * 3 + lane * 5 + 2) % 16)
                    .map_err(|error| error.to_string())?;
                block[8 + group * 16 + lane] = low | (high_code << 4);
            }
        }
        // WHY: GGML IQ4_XS serializes the two high-scale bytes before four
        // low-scale bytes; keeping this literal order catches header reversal.
        block[2..4].copy_from_slice(&high.to_le_bytes());
        bytes.extend_from_slice(&block);
    }
    Ok(bytes)
}

fn mixed_k_scale_values(
    row: usize,
    phase: usize,
) -> std::result::Result<([u8; 8], [u8; 8]), String> {
    let mut scales = [0_u8; 8];
    let mut minimums = [0_u8; 8];
    for group in 0..8 {
        scales[group] = 2 + u8::try_from((row * 3 + group * 5 + phase) % 11)
            .map_err(|error| error.to_string())?;
        minimums[group] = 12
            + u8::try_from((row * 7 + group * 9 + phase) % 44)
                .map_err(|error| error.to_string())?;
    }
    Ok((scales, minimums))
}

fn pack_k_scale_min(scales: [u8; 8], minimums: [u8; 8]) -> [u8; 12] {
    let mut packed = [0_u8; 12];
    for group in 0..4 {
        packed[group] = (scales[group] & 0x3f) | ((scales[group + 4] >> 4) << 6);
        packed[group + 4] = (minimums[group] & 0x3f) | ((minimums[group + 4] >> 4) << 6);
        packed[group + 8] = (scales[group + 4] & 0x0f) | (minimums[group + 4] << 4);
    }
    packed
}

fn ordered_projection_activations() -> std::result::Result<Vec<f32>, String> {
    let count = u8::try_from(TEST_PROJECTION_INPUT_WIDTH)
        .map_err(|error| format!("test projection width must fit u8: {error}"))?;
    Ok((1..=count).map(f32::from).collect())
}

fn set_u32(
    fixture: &mut Fixture,
    key: &'static str,
    value: u32,
) -> std::result::Result<(), String> {
    replace_metadata(fixture, MetadataEntry::U32(key, value))
}

fn replace_metadata(
    fixture: &mut Fixture,
    replacement: MetadataEntry,
) -> std::result::Result<(), String> {
    if let Some(entry) = fixture
        .metadata
        .iter_mut()
        .find(|entry| entry.key() == replacement.key())
    {
        *entry = replacement;
        return Ok(());
    }
    Err(format!(
        "fixture metadata key `{}` was not found",
        replacement.key()
    ))
}

fn mutate_tensor_shape(
    fixture: &mut Fixture,
    name: &str,
    dims: Vec<u64>,
) -> std::result::Result<(), String> {
    if let Some(tensor) = fixture
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == name)
    {
        tensor.dims = dims;
        return Ok(());
    }
    Err(format!("fixture tensor `{name}` was not found"))
}

fn rename_tensor(fixture: &mut Fixture, from: &str, to: &str) -> std::result::Result<(), String> {
    if let Some(tensor) = fixture
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == from)
    {
        tensor.name = to.to_string();
        return Ok(());
    }
    Err(format!("fixture tensor `{from}` was not found"))
}

fn observe_fixture(fixture: &Fixture) -> std::result::Result<ObservedArtifact, String> {
    let directory = tempdir().map_err(|error| error.to_string())?;
    let path = directory.path().join("qwen35-structure.gguf");
    fs::write(&path, fixture_bytes(fixture)?).map_err(|error| error.to_string())?;
    observe_gguf_with_sha256(&path).map_err(|error| error.to_string())
}

pub(crate) fn verify_fixture(fixture: &Fixture) -> std::result::Result<VerifiedArtifact, String> {
    let bytes = fixture_bytes(fixture)?;
    let expected = Sha256Digest::from_bytes(Sha256::digest(&bytes).into());
    let byte_limit = u64::try_from(bytes.len())
        .map_err(|error| format!("test fixture length must fit u64: {error}"))?;
    let byte_limit = NonZeroU64::new(byte_limit)
        .ok_or_else(|| "test fixture must contain GGUF header bytes".to_string())?;
    let directory = tempdir().map_err(|error| error.to_string())?;
    let path = directory.path().join("qwen35-payload.gguf");
    fs::write(&path, bytes).map_err(|error| error.to_string())?;
    VerifiedArtifact::load(&path, expected, ArtifactByteLimit::new(byte_limit))
        .map_err(|error| error.to_string())
}

fn preflight_error(artifact: &ObservedArtifact) -> std::result::Result<crate::Error, String> {
    match Qwen35StructuralProfile::try_from_observed(artifact) {
        Ok(_) => {
            Err("structural preflight unexpectedly accepted the malformed fixture".to_string())
        }
        Err(error) => Ok(error),
    }
}

fn fixture_bytes(fixture: &Fixture) -> std::result::Result<Vec<u8>, String> {
    let metadata = fixture
        .metadata
        .iter()
        .map(|entry| RawMetadata {
            key: entry.key().to_string(),
            value: match entry {
                MetadataEntry::U32(_, value) => RawMetadataValue::U32(*value),
                MetadataEntry::F32(_, value) => RawMetadataValue::F32(*value),
                MetadataEntry::String(_, value) => RawMetadataValue::String(value.clone()),
                MetadataEntry::StringArray(_, values) => {
                    RawMetadataValue::StringArray(values.clone())
                }
                MetadataEntry::I32Array(_, values) => RawMetadataValue::I32Array(values.clone()),
            },
        })
        .collect();
    let mut tensors = Vec::with_capacity(fixture.tensors.len());
    for tensor in &fixture.tensors {
        let byte_count = usize::try_from(tensor_bytes(tensor)?)
            .map_err(|_| "test tensor payload length exceeds usize".to_string())?;
        let payload = if tensor.payload.is_empty() {
            vec![0_u8; byte_count]
        } else if tensor.payload.len() == byte_count {
            tensor.payload.clone()
        } else {
            return Err(format!(
                "test tensor `{}` payload must be {byte_count} bytes, got {}",
                tensor.name,
                tensor.payload.len()
            ));
        };
        tensors.push(RawTensor {
            name: tensor.name.clone(),
            dims: tensor.dims.clone(),
            format: tensor.ggml_type,
            payload,
        });
    }
    serialize_raw_gguf(&RawGguf { metadata, tensors })
        .map(|fixture| fixture.bytes)
        .map_err(|error| error.to_string())
}

fn tensor_bytes(tensor: &FixtureTensor) -> std::result::Result<u64, String> {
    let logical_elements =
        tensor
            .dims
            .iter()
            .copied()
            .try_fold(1u64, |element_count, dimension| {
                element_count
                    .checked_mul(dimension)
                    .ok_or_else(|| "test tensor element count overflowed".to_string())
            })?;
    match tensor.ggml_type {
        TEST_F32_TYPE_ID => logical_elements
            .checked_mul(TEST_F32_BYTES)
            .ok_or_else(|| "test F32 tensor byte count overflowed".to_string()),
        TEST_Q8_0_TYPE_ID => {
            let values_per_block = to_u64(quant::q8_0::Q8_0_VALUES_PER_BLOCK)?;
            if !logical_elements.is_multiple_of(values_per_block) {
                return Err("test Q8_0 tensor elements must occupy complete blocks".to_string());
            }
            let block_bytes = to_u64(quant::q8_0::Q8_0_BLOCK_BYTES)?;
            (logical_elements / values_per_block)
                .checked_mul(block_bytes)
                .ok_or_else(|| "test Q8_0 tensor byte count overflowed".to_string())
        }
        TEST_Q4_K_TYPE_ID | TEST_Q5_K_TYPE_ID | TEST_Q6_K_TYPE_ID | TEST_IQ4_NL_TYPE_ID
        | TEST_IQ4_XS_TYPE_ID => {
            let (values_per_block, bytes_per_block) = match tensor.ggml_type {
                TEST_Q4_K_TYPE_ID => (256_u64, 144_u64),
                TEST_Q5_K_TYPE_ID => (256_u64, 176_u64),
                TEST_Q6_K_TYPE_ID => (256_u64, 210_u64),
                TEST_IQ4_NL_TYPE_ID => (32_u64, 18_u64),
                TEST_IQ4_XS_TYPE_ID => (256_u64, 136_u64),
                _ => return Err("test fixture quantized type dispatch drifted".to_string()),
            };
            if !logical_elements.is_multiple_of(values_per_block) {
                return Err(
                    "test block-quant tensor elements must occupy complete blocks".to_string(),
                );
            }
            (logical_elements / values_per_block)
                .checked_mul(bytes_per_block)
                .ok_or_else(|| "test block-quant tensor byte count overflowed".to_string())
        }
        other => Err(format!("test fixture does not support GGML type {other}")),
    }
}

fn to_u32(value: u64) -> std::result::Result<u32, String> {
    u32::try_from(value).map_err(|_| "test value exceeds u32".to_string())
}

fn to_u64(value: usize) -> std::result::Result<u64, String> {
    u64::try_from(value).map_err(|_| "test length exceeds u64".to_string())
}
