//! Synthetic end-to-end witnesses for the bounded Qwen3 rerank adapter.

use std::error::Error as StdError;
use std::num::NonZeroU64;

use loader::gguf::{ArtifactByteLimit, Sha256Digest, VerifiedArtifact};
use sha2::{Digest, Sha256};
use templates::TemplateLimits;
use test_fixtures::{RawGguf, RawMetadata, RawMetadataValue, RawTensor, serialize_raw_gguf};
use tokenize::{
    Error as TokenizerError, TokenizerByteLimit, TokenizerDigest, TokenizerIdentity,
    VerifiedTokenizer,
};

use super::{Qwen3Reranker, Qwen3RerankerLimits, multiply_bytes, signed_relevance_score};
use crate::{Error, RerankBatch, RerankItem, Reranker};

type TestResult<T> = std::result::Result<T, Box<dyn StdError>>;

const HIDDEN: usize = 2;
const FEED_FORWARD: usize = 4;
const CONTEXT: usize = 8;
const EPSILON: f32 = 0.001;
const F32: u32 = 0;
const POSITIVE_DOCUMENT: &str = "docpos";
const NEGATIVE_DOCUMENT: &str = "docneg";
const INSTRUCTION: &str = "instruct";
const POSITIVE_QUERY: &str = "positive";
const NEGATIVE_QUERY: &str = "negative";
const TEMPLATE: &str = "{% if messages[0].role == \"system\" and messages[1].role == \"query\" and messages[2].role == \"document\" %}system {{ messages[0].content }} query {{ messages[1].content }} document {% if messages[0].content == \"different\" and messages[1].content == \"positive\" and messages[2].content == \"docneg\" %}docpos{% else %}{{ messages[2].content }}{% endif %}{% else %}docpos{% endif %}";

#[test]
fn signed_score_is_raw_yes_minus_no_and_refuses_f32_overflow() -> TestResult<()> {
    let positive = signed_relevance_score(0, [3.5, -1.25])?;
    let negative = signed_relevance_score(1, [-1.25, 3.5])?;
    if positive.to_bits() != 4.75_f32.to_bits() || negative.to_bits() != (-4.75_f32).to_bits() {
        return Err("signed rerank score must remain raw yes minus no logits".into());
    }
    assert!(matches!(
        signed_relevance_score(2, [f32::MAX, -f32::MAX]),
        Err(Error::Qwen3NonFiniteScore { index: 2, .. })
    ));
    Ok(())
}

#[test]
fn dyn_reranker_returns_indexed_signed_scores_from_artifact_template() -> TestResult<()> {
    let raw = raw_rank_fixture()?;
    let (_directory, artifact) = verified_artifact(&raw)?;
    let reranker = reranker(&artifact, limits(128, 6, 2)?)?;
    let interface: &dyn Reranker = &reranker;
    let batch = batch(vec![
        (POSITIVE_QUERY, POSITIVE_DOCUMENT),
        (NEGATIVE_QUERY, NEGATIVE_DOCUMENT),
    ]);

    let predictions = interface.predict(batch)?;
    if predictions.len() != 2 || predictions.keys().copied().collect::<Vec<_>>() != [0, 1] {
        return Err("reranker did not preserve both input indices".into());
    }
    assert_score(
        predictions
            .get(&0)
            .ok_or("positive prediction is missing")?,
        hand_score([1.0, 0.0])?,
        "positive artifact-framed pair",
    )?;
    assert_score(
        predictions
            .get(&1)
            .ok_or("negative prediction is missing")?,
        hand_score([0.0, 1.0])?,
        "negative artifact-framed pair",
    )?;
    Ok(())
}

#[test]
fn explicit_instruction_and_typed_roles_control_the_final_document_token() -> TestResult<()> {
    let raw = raw_rank_fixture()?;
    let (_directory, artifact) = verified_artifact(&raw)?;
    let default = reranker(&artifact, limits(128, 6, 1)?)?;
    let custom = reranker_with_instruction(&artifact, limits(128, 6, 1)?, "different")?;
    let input = batch(vec![(POSITIVE_QUERY, NEGATIVE_DOCUMENT)]);
    let default_result = default.predict(input.clone())?;
    let custom_result = custom.predict(input)?;
    assert_score(
        default_result
            .get(&0)
            .ok_or("default framed prediction is missing")?,
        hand_score([0.0, 1.0])?,
        "default template document token",
    )?;
    assert_score(
        custom_result
            .get(&0)
            .ok_or("custom framed prediction is missing")?,
        hand_score([1.0, 0.0])?,
        "custom system/query/document template branch",
    )?;
    if default_result == custom_result {
        return Err("custom setup and typed role template branch did not alter the score".into());
    }
    Ok(())
}

#[test]
fn independent_byte_token_and_batch_bounds_refuse_without_truncation() -> TestResult<()> {
    let raw = raw_rank_fixture()?;
    let (_directory, artifact) = verified_artifact(&raw)?;
    let bytes = INSTRUCTION.len() + POSITIVE_QUERY.len() + POSITIVE_DOCUMENT.len();
    let byte_limited = reranker(&artifact, limits(bytes - 1, 6, 2)?)?;
    assert!(matches!(
        byte_limited.predict(batch(vec![(POSITIVE_QUERY, POSITIVE_DOCUMENT)])),
        Err(Error::Qwen3InputBytesTooLong { actual, limit, .. }) if actual == bytes && limit == bytes - 1
    ));
    let multibyte_limit = INSTRUCTION.len() + "é".chars().count() + POSITIVE_DOCUMENT.len();
    assert!(matches!(
        reranker(&artifact, limits(multibyte_limit, 6, 2)?)?.predict(batch(vec![("é", POSITIVE_DOCUMENT)])),
        Err(Error::Qwen3InputBytesTooLong { actual, limit, .. })
            if actual == multibyte_limit + 1 && limit == multibyte_limit
    ));

    let token_limited = reranker(&artifact, limits(128, 5, 2)?)?;
    assert!(matches!(
        token_limited.predict(batch(vec![(POSITIVE_QUERY, POSITIVE_DOCUMENT)])),
        Err(Error::Qwen3InputTokensTooLong {
            actual: 6,
            limit: 5,
            ..
        })
    ));

    let batch_limited = reranker(&artifact, limits(128, 6, 1)?)?;
    let oversized_blank = RerankBatch {
        items: vec![item(" ", " "), item(" ", " ")],
    };
    assert!(matches!(
        batch_limited.predict(oversized_blank),
        Err(Error::Qwen3BatchTooLarge {
            actual: 2,
            limit: 1,
            ..
        })
    ));
    assert!(matches!(
        reranker(&artifact, limits(128, 6, 1)?)?.predict(batch(vec![(" ", POSITIVE_DOCUMENT)])),
        Err(Error::EmptyQuery { index: 0, .. })
    ));
    Ok(())
}

#[test]
fn malformed_artifact_metadata_and_tokenizer_cannot_bypass_admission() -> TestResult<()> {
    assert_metadata_refusal(enable_automatic_bos, "tokenizer.ggml.add_bos_token")?;
    assert_metadata_refusal(remove_chat_template, "tokenizer.chat_template")?;

    let mut raw = raw_rank_fixture()?;
    reverse_artifact_vocabulary(&mut raw)?;
    let (_directory, artifact) = verified_artifact(&raw)?;
    let error = reranker(&artifact, limits(128, 6, 1)?)
        .err()
        .ok_or("reversed artifact vocabulary unexpectedly passed admission")?;
    let error = error
        .downcast_ref::<Error>()
        .ok_or("artifact vocabulary admission failure was not a rerank error")?;
    if !matches!(
        error,
        Error::Qwen3Tokenizer {
            source: TokenizerError::VocabularyMismatch { id: 0, .. },
            ..
        }
    ) || StdError::source(error).is_none()
    {
        return Err("artifact vocabulary refusal lost its typed source chain".into());
    }

    let mut raw = raw_rank_fixture()?;
    invalid_chat_template(&mut raw)?;
    let (_directory, artifact) = verified_artifact(&raw)?;
    let error = reranker(&artifact, limits(128, 6, 1)?)
        .err()
        .ok_or("invalid artifact template unexpectedly passed admission")?;
    let error = error
        .downcast_ref::<Error>()
        .ok_or("artifact template admission failure was not a rerank error")?;
    if !matches!(error, Error::Qwen3Template { .. }) || StdError::source(error).is_none() {
        return Err("template admission failure lost its typed source chain".into());
    }

    let raw = raw_rank_fixture()?;
    let (_directory, artifact) = verified_artifact(&raw)?;
    let tokenizer = tokenizer_with_swapped_vocabulary()?;
    let error = Qwen3Reranker::from_verified_cpu(
        &artifact,
        tokenizer,
        limits(128, 6, 1)?,
        INSTRUCTION.to_string(),
    )
    .err()
    .ok_or("mismatched tokenizer unexpectedly passed admission")?;
    if !matches!(
        &error,
        Error::Qwen3Tokenizer {
            source: TokenizerError::VocabularyMismatch { id: 9, .. },
            ..
        }
    ) || StdError::source(&error).is_none()
    {
        return Err("tokenizer admission failure lost its typed source chain".into());
    }
    Ok(())
}

#[test]
fn configured_tokenizer_padding_or_truncation_is_refused_before_prediction() -> TestResult<()> {
    let raw = raw_rank_fixture()?;
    let (_directory, artifact) = verified_artifact(&raw)?;
    let truncation = tokenizer_json().replace(
        "\"truncation\":null",
        "\"truncation\":{\"direction\":\"Right\",\"max_length\":5,\"strategy\":\"LongestFirst\",\"stride\":0}",
    );
    let truncation_error = Qwen3Reranker::from_verified_cpu(
        &artifact,
        verified_tokenizer(&truncation)?,
        limits(128, 6, 1)?,
        INSTRUCTION.to_string(),
    )
    .err()
    .ok_or("configured truncation unexpectedly passed rerank setup")?;
    if !matches!(
        &truncation_error,
        Error::Qwen3Tokenizer {
            source: TokenizerError::ConfiguredTruncation { .. },
            ..
        }
    ) || StdError::source(&truncation_error).is_none()
    {
        return Err("configured truncation refusal lost its typed source chain".into());
    }

    let padding = tokenizer_json().replace(
        "\"padding\":null",
        "\"padding\":{\"strategy\":{\"Fixed\":4},\"direction\":\"Right\",\"pad_to_multiple_of\":null,\"pad_id\":0,\"pad_type_id\":0,\"pad_token\":\"[UNK]\"}",
    );
    let padding_error = Qwen3Reranker::from_verified_cpu(
        &artifact,
        verified_tokenizer(&padding)?,
        limits(128, 6, 1)?,
        INSTRUCTION.to_string(),
    )
    .err()
    .ok_or("configured padding unexpectedly passed rerank setup")?;
    if !matches!(
        &padding_error,
        Error::Qwen3Tokenizer {
            source: TokenizerError::ConfiguredPadding { .. },
            ..
        }
    ) || StdError::source(&padding_error).is_none()
    {
        return Err("configured padding refusal lost its typed source chain".into());
    }
    Ok(())
}

#[test]
fn late_head_refusal_returns_an_error_and_pristine_retry_remains_usable() -> TestResult<()> {
    let mut malformed = raw_rank_fixture()?;
    set_f32_prefix(&mut malformed, "cls.output.weight", f32::NAN)?;
    let (_bad_directory, bad_artifact) = verified_artifact(&malformed)?;
    let bad = reranker(&bad_artifact, limits(128, 6, 1)?)?;
    let error = bad
        .predict(batch(vec![(POSITIVE_QUERY, POSITIVE_DOCUMENT)]))
        .err()
        .ok_or("late nonfinite rank head unexpectedly returned predictions")?;
    if !matches!(&error, Error::Qwen3Decoder { .. }) || StdError::source(&error).is_none() {
        return Err("late decoder refusal lost its typed source chain".into());
    }

    let raw = raw_rank_fixture()?;
    let (_directory, artifact) = verified_artifact(&raw)?;
    let pristine = reranker(&artifact, limits(128, 6, 1)?)?;
    let retry = pristine.predict(batch(vec![(POSITIVE_QUERY, POSITIVE_DOCUMENT)]))?;
    assert_score(
        retry
            .get(&0)
            .ok_or("pristine retry prediction is missing")?,
        hand_score([1.0, 0.0])?,
        "pristine retry",
    )
}

#[test]
fn cpu_requirements_compose_sequential_scalar_score_rows() -> TestResult<()> {
    let raw = raw_rank_fixture()?;
    let (_directory, artifact) = verified_artifact(&raw)?;
    let single_model = reranker(&artifact, limits(128, 6, 1)?)?;
    let single = single_model.cpu_requirements()?;
    let batch_model = reranker(&artifact, limits(128, 6, 3)?)?;
    let batch_requirements = batch_model.cpu_requirements()?;
    let decoder = single.decoder_cpu_requirements();
    let score_bytes = u64::try_from(std::mem::size_of::<f32>())?;
    assert_eq!(single.max_batch_items(), 1);
    assert_eq!(single.returned_output_bytes(), score_bytes);
    assert_eq!(
        single.logical_f32_upper_bound_bytes(),
        score_bytes.max(decoder.workspace_upper_bound_bytes())
    );
    assert_eq!(batch_requirements.max_batch_items(), 3);
    assert_eq!(
        batch_requirements.returned_output_bytes(),
        score_bytes
            .checked_mul(3)
            .ok_or("synthetic rerank output multiplication overflowed")?
    );
    assert_eq!(
        batch_requirements.logical_f32_upper_bound_bytes(),
        score_bytes
            .checked_mul(3)
            .ok_or("synthetic rerank output multiplication overflowed")?
            .max(
                score_bytes
                    .checked_mul(2)
                    .and_then(|prior| prior.checked_add(decoder.workspace_upper_bound_bytes()))
                    .ok_or("synthetic rerank peak multiplication overflowed")?,
            )
    );
    if decoder.returned_output_bytes() != 0 {
        return Err("rank decoder must classify terminal hidden output as workspace".into());
    }
    let predictions = batch_model.predict(batch(vec![
        (POSITIVE_QUERY, POSITIVE_DOCUMENT),
        (NEGATIVE_QUERY, NEGATIVE_DOCUMENT),
        (POSITIVE_QUERY, NEGATIVE_DOCUMENT),
    ]))?;
    let mut actual_output_bytes = 0_u64;
    for scores in predictions.values() {
        let row_bytes = u64::try_from(scores.len())?
            .checked_mul(score_bytes)
            .ok_or("synthetic rerank returned row overflowed")?;
        actual_output_bytes = actual_output_bytes
            .checked_add(row_bytes)
            .ok_or("synthetic rerank returned output sum overflowed")?;
    }
    assert_eq!(
        actual_output_bytes,
        batch_requirements.returned_output_bytes()
    );
    Ok(())
}

#[cfg(target_pointer_width = "64")]
#[test]
fn cpu_requirements_refuse_overflowing_scalar_batch_output() -> TestResult<()> {
    let raw = raw_rank_fixture()?;
    let (_directory, artifact) = verified_artifact(&raw)?;
    let model = reranker(&artifact, limits(128, 6, usize::MAX)?)?;
    assert!(matches!(
        model.cpu_requirements(),
        Err(Error::Qwen3RequirementsOverflow { .. })
    ));
    Ok(())
}

#[test]
fn cpu_requirements_refuse_portable_scalar_multiplication_overflow() -> TestResult<()> {
    assert!(matches!(
        multiply_bytes(u64::MAX, 2, "synthetic rerank overflow"),
        Err(Error::Qwen3RequirementsOverflow { .. })
    ));
    Ok(())
}

#[test]
fn later_item_refusal_discards_partial_batch_and_allows_same_instance_retry() -> TestResult<()> {
    let raw = raw_rank_fixture()?;
    let (_directory, artifact) = verified_artifact(&raw)?;
    let reranker = reranker(&artifact, limits(128, 6, 2)?)?;
    let error = reranker
        .predict(batch(vec![
            (POSITIVE_QUERY, POSITIVE_DOCUMENT),
            (POSITIVE_QUERY, "docpos docpos"),
        ]))
        .err()
        .ok_or("late oversized item unexpectedly returned a partial prediction map")?;
    if !matches!(
        error,
        Error::Qwen3InputTokensTooLong {
            index: 1,
            actual: 7,
            limit: 6,
            ..
        }
    ) {
        return Err("late oversized item did not retain its precise indexed refusal".into());
    }

    let retry = reranker.predict(batch(vec![(POSITIVE_QUERY, POSITIVE_DOCUMENT)]))?;
    if retry.keys().copied().collect::<Vec<_>>() != [0] {
        return Err("same reranker instance did not return a pristine retry map".into());
    }
    assert_score(
        retry
            .get(&0)
            .ok_or("same-instance retry prediction is missing")?,
        hand_score([1.0, 0.0])?,
        "same-instance retry",
    )
}

fn reranker(
    artifact: &VerifiedArtifact,
    limits: Qwen3RerankerLimits,
) -> TestResult<Qwen3Reranker<'_>> {
    reranker_with_instruction(artifact, limits, INSTRUCTION)
}

fn reranker_with_instruction<'artifact>(
    artifact: &'artifact VerifiedArtifact,
    limits: Qwen3RerankerLimits,
    instruction: &str,
) -> TestResult<Qwen3Reranker<'artifact>> {
    Ok(Qwen3Reranker::from_verified_cpu(
        artifact,
        tokenizer()?,
        limits,
        instruction.to_string(),
    )?)
}

fn limits(
    max_pair_bytes: usize,
    max_tokens: usize,
    max_batch_items: usize,
) -> TestResult<Qwen3RerankerLimits> {
    Ok(Qwen3RerankerLimits {
        max_pair_bytes,
        max_tokens,
        max_batch_items,
        template: TemplateLimits::new(1_024, 1_024, 10_000, 16)?,
    })
}

fn batch(pairs: Vec<(&str, &str)>) -> RerankBatch {
    RerankBatch {
        items: pairs
            .into_iter()
            .map(|(query, document)| item(query, document))
            .collect(),
    }
}

fn item(query: &str, document: &str) -> RerankItem {
    RerankItem {
        query: query.to_string(),
        document: document.to_string(),
    }
}

fn assert_score(actual: &[f32], expected: f64, label: &str) -> TestResult<()> {
    let [actual] = actual else {
        return Err(format!("{label} did not return exactly one score").into());
    };
    let actual = f64::from(*actual);
    let tolerance = 1.0e-5 + 1.0e-5 * actual.abs().max(expected.abs());
    if !actual.is_finite() || (actual - expected).abs() > tolerance {
        return Err(format!(
            "{label} score {actual} did not match independent expected {expected}"
        )
        .into());
    }
    Ok(())
}

fn hand_score(final_embedding: [f64; HIDDEN]) -> TestResult<f64> {
    let mean_square = final_embedding
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        / f64::from(u32::try_from(HIDDEN)?);
    let scale = (mean_square + f64::from(EPSILON)).sqrt().recip();
    let normalized = [final_embedding[0] * scale, final_embedding[1] * scale];
    let yes = 2.0 * normalized[0] - normalized[1];
    let no = -normalized[0] + 2.0 * normalized[1];
    Ok(yes - no)
}

fn tokenizer() -> TestResult<VerifiedTokenizer> {
    verified_tokenizer(&tokenizer_json())
}

fn tokenizer_with_swapped_vocabulary() -> TestResult<VerifiedTokenizer> {
    verified_tokenizer(
        &tokenizer_json().replace("\"docpos\":9,\"docneg\":10", "\"docpos\":10,\"docneg\":9"),
    )
}

fn verified_tokenizer(source: &str) -> TestResult<VerifiedTokenizer> {
    let bytes = source.as_bytes();
    let digest = TokenizerDigest::from_bytes(Sha256::digest(bytes).into());
    Ok(VerifiedTokenizer::from_bytes(
        bytes,
        TokenizerIdentity::new(bytes.len(), digest),
        TokenizerByteLimit::try_new(bytes.len())?,
    )?)
}

fn tokenizer_json() -> String {
    r#"{
      "version":"1.0", "truncation":null, "padding":null,
      "added_tokens":[
        {"id":1,"content":"<|im_start|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
        {"id":2,"content":"<|im_end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
      ],
      "normalizer":null, "pre_tokenizer":{"type":"Whitespace"}, "post_processor":null, "decoder":null,
      "model":{"type":"WordLevel","vocab":{"[UNK]":0,"<|im_start|>":1,"<|im_end|>":2,"system":3,"query":4,"document":5,"instruct":6,"positive":7,"negative":8,"docpos":9,"docneg":10,"different":11},"unk_token":"[UNK]"}
    }"#
    .to_string()
}

fn verified_artifact(raw: &RawGguf) -> TestResult<(tempfile::TempDir, VerifiedArtifact)> {
    let fixture = serialize_raw_gguf(raw)?;
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("synthetic-qwen3-rank.gguf");
    std::fs::write(&path, &fixture.bytes)?;
    let limit = NonZeroU64::new(fixture.byte_len).ok_or("serialized fixture is empty")?;
    Ok((
        directory,
        VerifiedArtifact::load(
            &path,
            Sha256Digest::from_bytes(fixture.sha256),
            ArtifactByteLimit::new(limit),
        )?,
    ))
}

#[expect(
    clippy::too_many_lines,
    reason = "keep the complete original synthetic model inventory together for oracle review"
)]
fn raw_rank_fixture() -> TestResult<RawGguf> {
    let tokens = vocabulary();
    let mut embedding = vec![0.0; tokens.len() * HIDDEN];
    set_row(
        &mut embedding,
        token_id(&tokens, POSITIVE_DOCUMENT)?,
        [1.0, 0.0],
    )?;
    set_row(
        &mut embedding,
        token_id(&tokens, NEGATIVE_DOCUMENT)?,
        [0.0, 1.0],
    )?;
    let zeros = |width: usize| vec![0.0; width];
    Ok(RawGguf {
        metadata: vec![
            metadata_string("general.architecture", "qwen3"),
            metadata_u32("qwen3.block_count", 1),
            metadata_u32("qwen3.context_length", u32::try_from(CONTEXT)?),
            metadata_u32("qwen3.embedding_length", u32::try_from(HIDDEN)?),
            metadata_u32("qwen3.feed_forward_length", u32::try_from(FEED_FORWARD)?),
            metadata_u32("qwen3.attention.head_count", 1),
            metadata_u32("qwen3.attention.head_count_kv", 1),
            metadata_u32("qwen3.attention.key_length", u32::try_from(HIDDEN)?),
            metadata_u32("qwen3.attention.value_length", u32::try_from(HIDDEN)?),
            metadata_f32("qwen3.attention.layer_norm_rms_epsilon", EPSILON),
            metadata_bool("qwen3.attention.causal", true),
            metadata_u32("qwen3.rope.dimension_count", u32::try_from(HIDDEN)?),
            metadata_f32("qwen3.rope.freq_base", 10_000.0),
            metadata_u32("qwen3.pooling_type", 4),
            RawMetadata {
                key: "qwen3.classifier.output_labels".to_string(),
                value: RawMetadataValue::StringArray(vec!["yes".to_string(), "no".to_string()]),
            },
            RawMetadata {
                key: "tokenizer.ggml.tokens".to_string(),
                value: RawMetadataValue::StringArray(tokens.clone()),
            },
            metadata_string("tokenizer.chat_template", TEMPLATE),
            metadata_bool("tokenizer.ggml.add_bos_token", false),
            metadata_bool("tokenizer.ggml.add_eos_token", false),
        ],
        tensors: vec![
            f32_tensor("token_embd.weight", &[HIDDEN, tokens.len()], &embedding)?,
            f32_tensor("output_norm.weight", &[HIDDEN], &[1.0, 1.0])?,
            f32_tensor("blk.0.attn_norm.weight", &[HIDDEN], &[1.0, 1.0])?,
            f32_tensor("blk.0.attn_q_norm.weight", &[HIDDEN], &[1.0, 1.0])?,
            f32_tensor("blk.0.attn_k_norm.weight", &[HIDDEN], &[1.0, 1.0])?,
            f32_tensor("blk.0.ffn_norm.weight", &[HIDDEN], &[1.0, 1.0])?,
            f32_tensor(
                "blk.0.attn_q.weight",
                &[HIDDEN, HIDDEN],
                &zeros(HIDDEN * HIDDEN),
            )?,
            f32_tensor(
                "blk.0.attn_k.weight",
                &[HIDDEN, HIDDEN],
                &zeros(HIDDEN * HIDDEN),
            )?,
            f32_tensor(
                "blk.0.attn_v.weight",
                &[HIDDEN, HIDDEN],
                &zeros(HIDDEN * HIDDEN),
            )?,
            f32_tensor(
                "blk.0.attn_output.weight",
                &[HIDDEN, HIDDEN],
                &zeros(HIDDEN * HIDDEN),
            )?,
            f32_tensor(
                "blk.0.ffn_gate.weight",
                &[HIDDEN, FEED_FORWARD],
                &zeros(HIDDEN * FEED_FORWARD),
            )?,
            f32_tensor(
                "blk.0.ffn_up.weight",
                &[HIDDEN, FEED_FORWARD],
                &zeros(HIDDEN * FEED_FORWARD),
            )?,
            f32_tensor(
                "blk.0.ffn_down.weight",
                &[FEED_FORWARD, HIDDEN],
                &zeros(FEED_FORWARD * HIDDEN),
            )?,
            f32_tensor("cls.output.weight", &[HIDDEN, 2], &[2.0, -1.0, -1.0, 2.0])?,
        ],
    })
}

fn vocabulary() -> Vec<String> {
    [
        "[UNK]",
        "<|im_start|>",
        "<|im_end|>",
        "system",
        "query",
        "document",
        INSTRUCTION,
        POSITIVE_QUERY,
        NEGATIVE_QUERY,
        POSITIVE_DOCUMENT,
        NEGATIVE_DOCUMENT,
        "different",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn token_id(tokens: &[String], token: &str) -> TestResult<usize> {
    tokens
        .iter()
        .position(|candidate| candidate == token)
        .ok_or_else(|| format!("synthetic vocabulary does not contain `{token}`").into())
}

fn set_row(values: &mut [f32], row: usize, replacement: [f32; HIDDEN]) -> TestResult<()> {
    let start = row
        .checked_mul(HIDDEN)
        .ok_or("embedding row start overflow")?;
    let target = values
        .get_mut(start..start + HIDDEN)
        .ok_or("embedding row is outside synthetic matrix")?;
    target.copy_from_slice(&replacement);
    Ok(())
}

fn f32_tensor(name: &str, dimensions: &[usize], values: &[f32]) -> TestResult<RawTensor> {
    let expected = dimensions.iter().try_fold(1_usize, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or("tensor element count overflow")
    })?;
    if values.len() != expected {
        return Err(format!(
            "synthetic tensor `{name}` has {} values, expected {expected}",
            values.len()
        )
        .into());
    }
    let mut payload = Vec::with_capacity(expected * std::mem::size_of::<f32>());
    for value in values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    Ok(RawTensor {
        name: name.to_string(),
        dims: dimensions
            .iter()
            .map(|dimension| u64::try_from(*dimension))
            .collect::<std::result::Result<Vec<_>, _>>()?,
        format: F32,
        payload,
    })
}

fn metadata_string(key: &str, value: &str) -> RawMetadata {
    RawMetadata {
        key: key.to_string(),
        value: RawMetadataValue::String(value.to_string()),
    }
}

fn metadata_u32(key: &str, value: u32) -> RawMetadata {
    RawMetadata {
        key: key.to_string(),
        value: RawMetadataValue::U32(value),
    }
}

fn metadata_f32(key: &str, value: f32) -> RawMetadata {
    RawMetadata {
        key: key.to_string(),
        value: RawMetadataValue::F32(value),
    }
}

fn metadata_bool(key: &str, value: bool) -> RawMetadata {
    RawMetadata {
        key: key.to_string(),
        value: RawMetadataValue::Bool(value),
    }
}

fn enable_automatic_bos(raw: &mut RawGguf) -> TestResult<()> {
    replace_metadata(
        raw,
        "tokenizer.ggml.add_bos_token",
        RawMetadataValue::Bool(true),
    )
}

fn assert_metadata_refusal(
    mutate: fn(&mut RawGguf) -> TestResult<()>,
    key: &'static str,
) -> TestResult<()> {
    let mut raw = raw_rank_fixture()?;
    mutate(&mut raw)?;
    let (_directory, artifact) = verified_artifact(&raw)?;
    let error = reranker(&artifact, limits(128, 6, 1)?)
        .err()
        .ok_or("malformed metadata unexpectedly passed rerank admission")?;
    let error = error
        .downcast_ref::<Error>()
        .ok_or("malformed metadata admission failure was not a rerank error")?;
    if !matches!(error, Error::Qwen3Metadata { key: actual, .. } if *actual == key) {
        return Err(format!("metadata key `{key}` was not precisely reported").into());
    }
    Ok(())
}

fn remove_chat_template(raw: &mut RawGguf) -> TestResult<()> {
    let position = raw
        .metadata
        .iter()
        .position(|entry| entry.key == "tokenizer.chat_template")
        .ok_or("synthetic template metadata is missing")?;
    raw.metadata.remove(position);
    Ok(())
}

fn invalid_chat_template(raw: &mut RawGguf) -> TestResult<()> {
    replace_metadata(
        raw,
        "tokenizer.chat_template",
        RawMetadataValue::String("{% if".to_string()),
    )
}

fn reverse_artifact_vocabulary(raw: &mut RawGguf) -> TestResult<()> {
    replace_metadata(
        raw,
        "tokenizer.ggml.tokens",
        RawMetadataValue::StringArray(vocabulary().into_iter().rev().collect()),
    )
}

fn replace_metadata(raw: &mut RawGguf, key: &str, value: RawMetadataValue) -> TestResult<()> {
    let entry = raw
        .metadata
        .iter_mut()
        .find(|entry| entry.key == key)
        .ok_or_else(|| format!("synthetic metadata `{key}` is missing"))?;
    entry.value = value;
    Ok(())
}

fn set_f32_prefix(raw: &mut RawGguf, name: &str, value: f32) -> TestResult<()> {
    let tensor = raw
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| format!("synthetic tensor `{name}` is missing"))?;
    tensor
        .payload
        .get_mut(..4)
        .ok_or("synthetic tensor has no first F32 lane")?
        .copy_from_slice(&value.to_le_bytes());
    Ok(())
}
