use std::{fs, num::NonZeroU64};

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
const TEST_INNER: u64 = 6;
const TEST_STATE: u64 = 2;
const TEST_TIME_STEP_RANK: u64 = 3;
const TEST_GROUP_COUNT: u64 = 2;
const TEST_VOCABULARY: u64 = 5;
const TEST_MAIN_BLOCKS: u64 = 4;
const TEST_FULL_ATTENTION_INTERVAL: u64 = 4;
const TEST_F32_BYTES: u64 = 4;
const TEST_F32_TYPE_ID: u32 = 0;
const TEST_Q8_0_TYPE_ID: u32 = 8;
const TEST_Q_WIDTH: u64 = 8;
const TEST_FULL_ATTENTION_OUTPUT_WIDTH: u64 = 4;
const TEST_SSM_CONV_WIDTH: u64 = 14;
const TEST_NEXTN_PROJECTION_WIDTH: u64 = 6;
const TEST_PROJECTION_INPUT_WIDTH: u64 = 64;
const TEST_PROJECTION_OUTPUT_WIDTH: usize = 3;
const TEST_Q8_SCALE_ONE_BITS: u16 = 0x3c00;
const TEST_Q8_SCALE_NONFINITE_BITS: u16 = 0x7c00;
const TEST_ALTERNATING_PERIOD: usize = 2;

#[derive(Clone)]
enum MetadataEntry {
    U32(&'static str, u32),
    F32(&'static str, f32),
    String(&'static str, String),
    StringArray(&'static str, Vec<String>),
}

impl MetadataEntry {
    fn key(&self) -> &'static str {
        match self {
            Self::U32(key, _)
            | Self::F32(key, _)
            | Self::String(key, _)
            | Self::StringArray(key, _) => key,
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
struct Fixture {
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
    set_u32(&mut fixture, SSM_INNER_SIZE_KEY, 3)?;
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
        .project_q8_0("blk.0.ffn_down.weight", &activations)
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
fn projection_rejects_wrong_name_rank_dtype_and_width() -> std::result::Result<(), String> {
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

    let wrong_name = weights.project_q8_0("blk.0.not_a_role.weight", &activations);
    assert!(
        matches!(wrong_name, Err(crate::Error::PayloadTensor { .. })),
        "unknown tensor names must not create a projection"
    );

    let wrong_rank = weights.project_q8_0(OUTPUT_NORM_TENSOR, &activations);
    assert!(
        matches!(wrong_rank, Err(crate::Error::ProjectionRank { .. })),
        "recognized rank-one tensors must not be treated as matrices"
    );

    let wrong_dtype = weights.project_q8_0(OUTPUT_TENSOR, &activations);
    assert!(
        matches!(wrong_dtype, Err(crate::Error::ProjectionDtype { .. })),
        "rank-two non-Q8 tensors must not be decoded by the Q8 path"
    );

    let wrong_width = weights.project_q8_0("blk.0.ffn_down.weight", &activations[..63]);
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

    let result = weights.project_q8_0("blk.0.ffn_down.weight", &activations);
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
            MetadataEntry::StringArray(
                TOKENS_KEY,
                (0..TEST_VOCABULARY)
                    .map(|index| format!("token-{index}"))
                    .collect(),
            ),
            MetadataEntry::U32("general.alignment", TEST_ALIGNMENT),
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

fn verify_fixture(fixture: &Fixture) -> std::result::Result<VerifiedArtifact, String> {
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
