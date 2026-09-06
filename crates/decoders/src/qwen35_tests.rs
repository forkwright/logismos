use std::{collections::BTreeMap, fs, num::NonZeroU64};

use loader::gguf::{
    ArtifactByteLimit, ObservedArtifact, Sha256Digest, VerifiedArtifact, observe_gguf_with_sha256,
};
use sha2::{Digest, Sha256};
use tempfile::tempdir;

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
const TEST_IQ4_NL_TYPE_ID: u32 = 20;
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

pub(crate) struct CanonicalHybridOracle {
    tensors: BTreeMap<String, Vec<f64>>,
    recurrent: Vec<OracleRecurrentState>,
    keys: Vec<Vec<f64>>,
    values: Vec<Vec<f64>>,
    position: usize,
    rope_width: usize,
    include_attention: bool,
    include_ffn: bool,
    adjacent_pairs: bool,
}

type OracleStateSnapshot = (usize, Vec<(Vec<f64>, Vec<f64>)>, Vec<f64>, Vec<f64>);

struct OracleRecurrentState {
    convolution: Vec<f64>,
    gdn: Vec<f64>,
}

impl CanonicalHybridOracle {
    pub(crate) fn from_fixture(fixture: &Fixture) -> std::result::Result<Self, String> {
        let mut tensors = BTreeMap::new();
        for tensor in &fixture.tensors {
            if tensor.ggml_type != TEST_F32_TYPE_ID {
                return Err(format!(
                    "canonical oracle requires F32 tensor `{}`",
                    tensor.name
                ));
            }
            let values = tensor
                .payload
                .chunks_exact(4)
                .map(|bytes| {
                    f64::from(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                })
                .collect::<Vec<_>>();
            tensors.insert(tensor.name.clone(), values);
        }
        let rope_width = fixture
            .metadata
            .iter()
            .find_map(|entry| match entry {
                MetadataEntry::U32("qwen35.rope.dimension_count", value) => Some(*value),
                _ => None,
            })
            .map(usize::try_from)
            .transpose()
            .map_err(|error| error.to_string())?
            .unwrap_or(test_dimension(CANONICAL_HEAD_WIDTH)?);
        Ok(Self {
            tensors,
            recurrent: (0..3)
                .map(|_| OracleRecurrentState {
                    convolution: vec![0.0; test_dimension(TEST_SSM_CONV_WIDTH).unwrap_or(0)],
                    gdn: vec![0.0; 16],
                })
                .collect(),
            keys: Vec::new(),
            values: Vec::new(),
            position: 0,
            rope_width,
            include_attention: true,
            include_ffn: true,
            adjacent_pairs: false,
        })
    }

    fn without_attention(mut self) -> Self {
        self.include_attention = false;
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
        self.rope_width = 128;
        self
    }

    fn tensor(&self, name: &str) -> std::result::Result<&[f64], String> {
        self.tensors
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| format!("canonical oracle is missing `{name}`"))
    }

    pub(crate) fn step(&mut self, tokens: &[u32]) -> std::result::Result<Vec<f64>, String> {
        let mut logits = Vec::new();
        for token in tokens {
            let token = usize::try_from(*token).map_err(|error| error.to_string())?;
            let embedding = self.tensor(TOKEN_EMBEDDING_TENSOR)?;
            let hidden_width = test_dimension(TEST_HIDDEN)?;
            let start = token
                .checked_mul(hidden_width)
                .ok_or_else(|| "canonical token offset overflowed".to_string())?;
            let mut hidden = embedding
                .get(start..start + hidden_width)
                .ok_or_else(|| "canonical oracle token must be in vocabulary".to_string())?
                .to_vec();
            for block in 0..4 {
                let residual = hidden.clone();
                let mut attention = if block == 3 {
                    self.full_attention(&hidden)?
                } else {
                    self.recurrent_attention(block, &hidden)?
                };
                if !self.include_attention {
                    attention.fill(0.0);
                }
                add_f64(&mut hidden, &attention)?;
                let normalized = oracle_rms(
                    &hidden,
                    self.tensor(&format!("blk.{block}.post_attention_norm.weight"))?,
                    1,
                    hidden_width,
                )?;
                let mut ffn = self.ffn(block, &normalized)?;
                if !self.include_ffn {
                    ffn.fill(0.0);
                }
                hidden = residual;
                add_f64(&mut hidden, &attention)?;
                add_f64(&mut hidden, &ffn)?;
            }
            let normalized =
                oracle_rms(&hidden, self.tensor(OUTPUT_NORM_TENSOR)?, 1, hidden_width)?;
            logits.extend(oracle_project(
                &normalized,
                self.tensor(OUTPUT_TENSOR)?,
                hidden_width,
                test_dimension(TEST_VOCABULARY)?,
            ));
            self.position += 1;
        }
        Ok(logits)
    }

    fn ffn(&self, block: usize, input: &[f64]) -> std::result::Result<Vec<f64>, String> {
        let hidden = test_dimension(TEST_HIDDEN)?;
        let feed_forward = test_dimension(TEST_FEED_FORWARD)?;
        let gate = oracle_project(
            input,
            self.tensor(&format!("blk.{block}.ffn_gate.weight"))?,
            hidden,
            feed_forward,
        );
        let up = oracle_project(
            input,
            self.tensor(&format!("blk.{block}.ffn_up.weight"))?,
            hidden,
            feed_forward,
        );
        let fused = gate
            .iter()
            .zip(up)
            .map(|(gate, up)| (gate / (1.0 + (-gate).exp())) * up)
            .collect::<Vec<_>>();
        Ok(oracle_project(
            &fused,
            self.tensor(&format!("blk.{block}.ffn_down.weight"))?,
            feed_forward,
            hidden,
        ))
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
        let hidden = test_dimension(TEST_HIDDEN)?;
        let inner = test_dimension(TEST_INNER)?;
        let state_width = test_dimension(TEST_STATE)?;
        let value_heads = test_dimension(TEST_TIME_STEP_RANK)?;
        let key_heads = test_dimension(TEST_GROUP_COUNT)?;
        let conv_width = test_dimension(TEST_SSM_CONV_WIDTH)?;
        let normalized = oracle_rms(
            input,
            self.tensor(&format!("blk.{block}.attn_norm.weight"))?,
            1,
            hidden,
        )?;
        let qkv = oracle_project(
            &normalized,
            self.tensor(&format!("blk.{block}.attn_qkv.weight"))?,
            hidden,
            conv_width,
        );
        let z = oracle_project(
            &normalized,
            self.tensor(&format!("blk.{block}.attn_gate.weight"))?,
            hidden,
            inner,
        );
        let alpha = oracle_project(
            &normalized,
            self.tensor(&format!("blk.{block}.ssm_alpha.weight"))?,
            hidden,
            value_heads,
        );
        let beta = oracle_project(
            &normalized,
            self.tensor(&format!("blk.{block}.ssm_beta.weight"))?,
            hidden,
            value_heads,
        );
        let convolution_weights = self
            .tensor(&format!("blk.{block}.ssm_conv1d.weight"))?
            .to_vec();
        let a = self.tensor(&format!("blk.{block}.ssm_a"))?.to_vec();
        let dt = self.tensor(&format!("blk.{block}.ssm_dt.bias"))?.to_vec();
        let norm = self
            .tensor(&format!("blk.{block}.ssm_norm.weight"))?
            .to_vec();
        let output_weight = self
            .tensor(&format!("blk.{block}.ssm_out.weight"))?
            .to_vec();
        let state = self
            .recurrent
            .get_mut(block)
            .ok_or_else(|| "canonical recurrent block is absent".to_string())?;
        let mut convolved = Vec::with_capacity(conv_width);
        for channel in 0..conv_width {
            let weights = &convolution_weights[channel * 2..channel * 2 + 2];
            convolved.push(state.convolution[channel] * weights[0] + qkv[channel] * weights[1]);
        }
        state.convolution.copy_from_slice(&qkv);
        let convolved = convolved
            .into_iter()
            .map(|value| value / (1.0 + (-value).exp()))
            .collect::<Vec<_>>();
        let key_width = key_heads * state_width;
        let mut output = vec![0.0; inner];
        for value_head in 0..value_heads {
            let key_head = value_head % key_heads;
            let q = oracle_l2(&convolved[key_head * state_width..(key_head + 1) * state_width]);
            let k = oracle_l2(
                &convolved
                    [key_width + key_head * state_width..key_width + (key_head + 1) * state_width],
            );
            let values = &convolved[2 * key_width + value_head * state_width
                ..2 * key_width + (value_head + 1) * state_width];
            let beta = 1.0 / (1.0 + (-beta[value_head]).exp());
            let alpha_dt = alpha[value_head] + dt[value_head];
            let gate = a[value_head] * (alpha_dt.max(0.0) + (-alpha_dt.abs()).exp().ln_1p());
            let decay = gate.exp();
            for key in 0..state_width {
                for value in 0..state_width {
                    state.gdn[(value_head * state_width + key) * state_width + value] *= decay;
                }
            }
            let mut delta = vec![0.0; state_width];
            for value in 0..state_width {
                let prior = (0..state_width)
                    .map(|key| {
                        state.gdn[(value_head * state_width + key) * state_width + value] * k[key]
                    })
                    .sum::<f64>();
                delta[value] = beta * (values[value] - prior);
            }
            for key in 0..state_width {
                for value in 0..state_width {
                    state.gdn[(value_head * state_width + key) * state_width + value] +=
                        k[key] * delta[value];
                }
            }
            for value in 0..state_width {
                output[value_head * state_width + value] = (0..state_width)
                    .map(|key| {
                        state.gdn[(value_head * state_width + key) * state_width + value] * q[key]
                            / (state_width as f64).sqrt()
                    })
                    .sum();
            }
        }
        let output = oracle_rms(&output, &norm, value_heads, state_width)?;
        let gated = output
            .iter()
            .zip(z)
            .map(|(output, z)| output * (z / (1.0 + (-z).exp())))
            .collect::<Vec<_>>();
        Ok(oracle_project(&gated, &output_weight, inner, hidden))
    }

    #[expect(
        clippy::cast_precision_loss,
        clippy::too_many_lines,
        reason = "the scalar oracle deliberately keeps pinned full-attention order visible"
    )]
    fn full_attention(&mut self, input: &[f64]) -> std::result::Result<Vec<f64>, String> {
        let hidden = test_dimension(TEST_HIDDEN)?;
        let heads = test_dimension(CANONICAL_HEADS)?;
        let kv_heads = test_dimension(CANONICAL_KEY_VALUE_HEADS)?;
        let width = test_dimension(CANONICAL_HEAD_WIDTH)?;
        let normalized = oracle_rms(input, self.tensor("blk.3.attn_norm.weight")?, 1, hidden)?;
        let q_gate = oracle_project(
            &normalized,
            self.tensor("blk.3.attn_q.weight")?,
            hidden,
            heads * width * 2,
        );
        let mut query = Vec::with_capacity(heads * width);
        let mut gate = Vec::with_capacity(heads * width);
        for head in 0..heads {
            let base = head * width * 2;
            query.extend_from_slice(&q_gate[base..base + width]);
            gate.extend_from_slice(&q_gate[base + width..base + 2 * width]);
        }
        let mut query = oracle_rms(
            query.as_slice(),
            self.tensor("blk.3.attn_q_norm.weight")?,
            heads,
            width,
        )?;
        let key = oracle_project(
            &normalized,
            self.tensor("blk.3.attn_k.weight")?,
            hidden,
            kv_heads * width,
        );
        let mut key = oracle_rms(
            &key,
            self.tensor("blk.3.attn_k_norm.weight")?,
            kv_heads,
            width,
        )?;
        let value = oracle_project(
            &normalized,
            self.tensor("blk.3.attn_v.weight")?,
            hidden,
            kv_heads * width,
        );
        canonical_mrope(
            &mut query,
            heads,
            width,
            self.rope_width,
            self.position,
            self.adjacent_pairs,
        )?;
        canonical_mrope(
            &mut key,
            kv_heads,
            width,
            self.rope_width,
            self.position,
            self.adjacent_pairs,
        )?;
        self.keys.push(key);
        self.values.push(value);
        let mut merged = Vec::with_capacity(heads * width);
        for head in 0..heads {
            let kv_head = head / (heads / kv_heads);
            let query = &query[head * width..(head + 1) * width];
            let scores = self
                .keys
                .iter()
                .map(|key| {
                    query
                        .iter()
                        .zip(&key[kv_head * width..(kv_head + 1) * width])
                        .map(|(left, right)| left * right)
                        .sum::<f64>()
                        / (width as f64).sqrt()
                })
                .collect::<Vec<_>>();
            let probabilities = oracle_softmax(&scores)?;
            for lane in 0..width {
                let attended = probabilities
                    .iter()
                    .zip(&self.values)
                    .map(|(probability, value)| probability * value[kv_head * width + lane])
                    .sum::<f64>();
                let gate = gate[head * width + lane];
                merged.push(attended / (1.0 + (-gate).exp()));
            }
        }
        Ok(oracle_project(
            &merged,
            self.tensor("blk.3.attn_output.weight")?,
            heads * width,
            hidden,
        ))
    }

    pub(crate) fn state_for_test(&self) -> OracleStateSnapshot {
        (
            self.position,
            self.recurrent
                .iter()
                .map(|state| (state.convolution.clone(), state.gdn.clone()))
                .collect(),
            self.keys.iter().flatten().copied().collect(),
            self.values.iter().flatten().copied().collect(),
        )
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "the f64 oracle intentionally converts checked small fixture indexes to angles"
)]
fn canonical_mrope(
    values: &mut [f64],
    rows: usize,
    width: usize,
    n_rot: usize,
    position: usize,
    adjacent_pairs: bool,
) -> std::result::Result<(), String> {
    if width != test_dimension(CANONICAL_HEAD_WIDTH)?
        || values.len() != rows * width
        || n_rot == 0
        || !n_rot.is_multiple_of(2)
        || n_rot > width
    {
        return Err(
            "canonical IMRoPE witness width does not match its serialized layout".to_string(),
        );
    }
    let positions = [position as f64, position as f64, position as f64, 0.0];
    for row in values.chunks_exact_mut(width) {
        for pair in 0..(n_rot / 2) {
            let sector = pair % 32;
            let axis = if sector % 3 == 1 && sector < 3 * 11 {
                1
            } else if sector % 3 == 2 && sector < 3 * 10 {
                2
            } else if sector.is_multiple_of(3) && sector < 3 * 11 {
                0
            } else {
                3
            };
            let angle = positions[axis] / 10_000.0_f64.powf((2 * pair) as f64 / n_rot as f64);
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
    let greatest = expected
        .iter()
        .zip(alternate)
        .map(|(expected, alternate)| (expected - alternate).abs())
        .fold(0.0_f64, f64::max);
    if greatest <= 2.0e-5 {
        return Err(format!(
            "canonical fixture is insensitive to the required {path}: greatest logit delta {greatest}"
        ));
    }
    Ok(())
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
    let width_f64 = f64::from(u32::try_from(width).map_err(|error| error.to_string())?);
    Ok(input
        .chunks_exact(width)
        .take(rows)
        .flat_map(|row| {
            let mean_square = row.iter().map(|value| value * value).sum::<f64>() / width_f64;
            let inverse = (mean_square + f64::from(TEST_RMS_EPSILON)).sqrt().recip();
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
    let denominator = values
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
        .max(f64::from(TEST_RMS_EPSILON));
    values.iter().map(|value| value / denominator).collect()
}

fn test_dimension(value: u64) -> std::result::Result<usize, String> {
    usize::try_from(value).map_err(|error| error.to_string())
}

fn assert_f32_matches_f64(
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
        if (f64::from(*actual) - expected).abs() > 2.0e-5 {
            return Err(format!(
                "{subject} differs at {index}: actual {actual}, expected {expected}"
            ));
        }
    }
    Ok(())
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
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&to_u64(fixture.tensors.len())?.to_le_bytes());
    bytes.extend_from_slice(&to_u64(fixture.metadata.len())?.to_le_bytes());
    for entry in &fixture.metadata {
        append_metadata(&mut bytes, entry)?;
    }

    let mut offsets = Vec::with_capacity(fixture.tensors.len());
    let mut next_offset = 0u64;
    for tensor in &fixture.tensors {
        next_offset = align_up(next_offset, u64::from(TEST_ALIGNMENT))?;
        offsets.push(next_offset);
        next_offset = next_offset
            .checked_add(tensor_bytes(tensor)?)
            .ok_or_else(|| "test tensor offsets overflowed".to_string())?;
    }
    for (tensor, offset) in fixture.tensors.iter().zip(offsets) {
        append_string(&mut bytes, &tensor.name)?;
        bytes.extend_from_slice(&to_u32(to_u64(tensor.dims.len())?)?.to_le_bytes());
        for dimension in &tensor.dims {
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
        bytes.extend_from_slice(&tensor.ggml_type.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
    }
    pad_to_alignment(&mut bytes, u64::from(TEST_ALIGNMENT))?;
    for tensor in &fixture.tensors {
        pad_to_alignment(&mut bytes, u64::from(TEST_ALIGNMENT))?;
        let byte_count = usize::try_from(tensor_bytes(tensor)?)
            .map_err(|_| "test tensor payload length exceeds usize".to_string())?;
        if tensor.payload.is_empty() {
            bytes.extend(std::iter::repeat_n(0u8, byte_count));
        } else if tensor.payload.len() == byte_count {
            bytes.extend_from_slice(&tensor.payload);
        } else {
            return Err(format!(
                "test tensor `{}` payload must be {byte_count} bytes, got {}",
                tensor.name,
                tensor.payload.len()
            ));
        }
    }
    Ok(bytes)
}

fn append_metadata(bytes: &mut Vec<u8>, entry: &MetadataEntry) -> std::result::Result<(), String> {
    append_string(bytes, entry.key())?;
    match entry {
        MetadataEntry::U32(_, value) => {
            bytes.extend_from_slice(&4u32.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        MetadataEntry::F32(_, value) => {
            bytes.extend_from_slice(&6u32.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        MetadataEntry::String(_, value) => {
            bytes.extend_from_slice(&8u32.to_le_bytes());
            append_string(bytes, value)?;
        }
        MetadataEntry::StringArray(_, values) => {
            bytes.extend_from_slice(&9u32.to_le_bytes());
            bytes.extend_from_slice(&8u32.to_le_bytes());
            bytes.extend_from_slice(&to_u64(values.len())?.to_le_bytes());
            for value in values {
                append_string(bytes, value)?;
            }
        }
        MetadataEntry::I32Array(_, values) => {
            bytes.extend_from_slice(&9u32.to_le_bytes());
            bytes.extend_from_slice(&5u32.to_le_bytes());
            bytes.extend_from_slice(&to_u64(values.len())?.to_le_bytes());
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    Ok(())
}

fn append_string(bytes: &mut Vec<u8>, value: &str) -> std::result::Result<(), String> {
    bytes.extend_from_slice(&to_u64(value.len())?.to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
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
        TEST_IQ4_NL_TYPE_ID => {
            const VALUES_PER_BLOCK: u64 = 32;
            const BYTES_PER_BLOCK: u64 = 18;
            if !logical_elements.is_multiple_of(VALUES_PER_BLOCK) {
                return Err("test IQ4NL tensor elements must occupy complete blocks".to_string());
            }
            (logical_elements / VALUES_PER_BLOCK)
                .checked_mul(BYTES_PER_BLOCK)
                .ok_or_else(|| "test IQ4NL tensor byte count overflowed".to_string())
        }
        other => Err(format!("test fixture does not support GGML type {other}")),
    }
}

fn align_up(value: u64, alignment: u64) -> std::result::Result<u64, String> {
    let adjustment = alignment
        .checked_sub(1)
        .ok_or_else(|| "test alignment must be non-zero".to_string())?;
    value
        .checked_add(adjustment)
        .map(|rounded| rounded / alignment * alignment)
        .ok_or_else(|| "test alignment rounding overflowed".to_string())
}

fn pad_to_alignment(bytes: &mut Vec<u8>, alignment: u64) -> std::result::Result<(), String> {
    let length = to_u64(bytes.len())?;
    let padding = align_up(length, alignment)?
        .checked_sub(length)
        .ok_or_else(|| "test alignment padding underflowed".to_string())?;
    let padding = usize::try_from(padding).map_err(|_| "test padding exceeds usize".to_string())?;
    bytes.extend(std::iter::repeat_n(0u8, padding));
    Ok(())
}

fn to_u32(value: u64) -> std::result::Result<u32, String> {
    u32::try_from(value).map_err(|_| "test value exceeds u32".to_string())
}

fn to_u64(value: usize) -> std::result::Result<u64, String> {
    u64::try_from(value).map_err(|_| "test length exceeds u64".to_string())
}
