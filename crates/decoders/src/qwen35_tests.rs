use std::fs;

use loader::gguf::{ObservedArtifact, observe_gguf_with_sha256};
use tempfile::tempdir;

use super::*;

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
const TEST_Q_WIDTH: u64 = 8;
const TEST_FULL_ATTENTION_OUTPUT_WIDTH: u64 = 4;
const TEST_SSM_CONV_WIDTH: u64 = 14;
const TEST_NEXTN_PROJECTION_WIDTH: u64 = 6;

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

fn fixture(nextn_block_count: u64) -> std::result::Result<Fixture, String> {
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
            add_full_attention_block(&mut tensors, block_index);
        } else {
            add_recurrent_block(&mut tensors, block_index);
        }
    }
    if nextn_block_count == 1 {
        add_nextn_block(&mut tensors, TEST_MAIN_BLOCKS);
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
            MetadataEntry::U32(FEED_FORWARD_LENGTH_KEY, to_u32(TEST_FEED_FORWARD)?),
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

fn add_full_attention_block(tensors: &mut Vec<FixtureTensor>, block_index: u64) {
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
    add_ffn_tensors(tensors, block_index);
    add_tensor(
        tensors,
        &block_tensor_name(block_index, POST_ATTENTION_NORM_ROLE),
        vec![TEST_HIDDEN],
    );
}

fn add_recurrent_block(tensors: &mut Vec<FixtureTensor>, block_index: u64) {
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
    add_ffn_tensors(tensors, block_index);
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

fn add_nextn_block(tensors: &mut Vec<FixtureTensor>, block_index: u64) {
    add_full_attention_block(tensors, block_index);
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

fn add_ffn_tensors(tensors: &mut Vec<FixtureTensor>, block_index: u64) {
    add_tensor(
        tensors,
        &block_tensor_name(block_index, FFN_DOWN_ROLE),
        vec![TEST_FEED_FORWARD, TEST_HIDDEN],
    );
    add_tensor(
        tensors,
        &block_tensor_name(block_index, FFN_GATE_ROLE),
        vec![TEST_HIDDEN, TEST_FEED_FORWARD],
    );
    add_tensor(
        tensors,
        &block_tensor_name(block_index, FFN_UP_ROLE),
        vec![TEST_HIDDEN, TEST_FEED_FORWARD],
    );
}

fn add_tensor(tensors: &mut Vec<FixtureTensor>, name: &str, dims: Vec<u64>) {
    tensors.push(FixtureTensor {
        name: name.to_string(),
        dims,
    });
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
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
    }
    pad_to_alignment(&mut bytes, u64::from(TEST_ALIGNMENT))?;
    for tensor in &fixture.tensors {
        pad_to_alignment(&mut bytes, u64::from(TEST_ALIGNMENT))?;
        let byte_count = usize::try_from(tensor_bytes(tensor)?)
            .map_err(|_| "test tensor payload length exceeds usize".to_string())?;
        bytes.extend(std::iter::repeat_n(0u8, byte_count));
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
    tensor
        .dims
        .iter()
        .copied()
        .try_fold(1u64, |element_count, dimension| {
            element_count
                .checked_mul(dimension)
                .ok_or_else(|| "test tensor element count overflowed".to_string())
        })?
        .checked_mul(TEST_F32_BYTES)
        .ok_or_else(|| "test tensor byte count overflowed".to_string())
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
