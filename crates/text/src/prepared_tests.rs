use std::cell::Cell;
use std::collections::VecDeque;
use std::mem::size_of;

use decoders::Qwen35LogitSelection;
use sha2::{Digest, Sha256};
use test_fixtures::build_qwen35_fixture;
use tokenize::{TokenizerDigest, TokenizerIdentity};

use super::{
    Cancellation, Error, FinishReason, GenerationDriver, GenerationRequest, NeverCancelled,
    TextMessage, TextRole,
};
use crate::error::{CancelledSnafu, InvalidConfigurationSnafu};
use crate::tests::{
    TestResult, fixture_config, load_fixture, mutated_artifact, pipeline_result,
    pipeline_with_tokenizer, set_f32_row, test_limits, tokenizer_json, verified_artifact,
};

const TOKENS: [&str; 5] = ["[UNK]", "<bos>", "<eos>", "hello", "assistant"];
const CONTENT_TEMPLATE: &str = "{{ messages[0].content }}";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TokenizationCancellationState {
    BeforeRendering,
    BeforeTokenization,
    Cancelled,
}

struct CancelWhenTokenizationStarts {
    state: Cell<TokenizationCancellationState>,
}

impl CancelWhenTokenizationStarts {
    fn new() -> Self {
        Self {
            state: Cell::new(TokenizationCancellationState::BeforeRendering),
        }
    }

    fn has_flipped(&self) -> bool {
        self.state.get() == TokenizationCancellationState::Cancelled
    }
}

impl Cancellation for CancelWhenTokenizationStarts {
    fn is_cancelled(&self) -> bool {
        match self.state.get() {
            TokenizationCancellationState::BeforeRendering => {
                self.state
                    .set(TokenizationCancellationState::BeforeTokenization);
                false
            }
            TokenizationCancellationState::BeforeTokenization => {
                self.state.set(TokenizationCancellationState::Cancelled);
                false
            }
            TokenizationCancellationState::Cancelled => true,
        }
    }
}

fn text_error<T>(result: super::Result<T>) -> TestResult<Error> {
    match result {
        Err(error) => Ok(error),
        Ok(_) => Err(std::io::Error::other("text operation unexpectedly succeeded").into()),
    }
}

fn tokenizer_identity(tokenizer_json: &str) -> TokenizerIdentity {
    let digest = TokenizerDigest::from_bytes(Sha256::digest(tokenizer_json.as_bytes()).into());
    TokenizerIdentity::new(tokenizer_json.len(), digest)
}

enum FakeStep {
    Logits(Vec<f32>),
    Failure,
}

struct FakeDriver {
    steps: VecDeque<FakeStep>,
    calls: Vec<Vec<u32>>,
    checks_each_prompt_token: bool,
}

impl FakeDriver {
    fn with_steps(steps: impl IntoIterator<Item = FakeStep>) -> Self {
        Self {
            steps: steps.into_iter().collect(),
            calls: Vec::new(),
            checks_each_prompt_token: false,
        }
    }

    fn with_prompt_token_checks(steps: impl IntoIterator<Item = FakeStep>) -> Self {
        Self {
            checks_each_prompt_token: true,
            ..Self::with_steps(steps)
        }
    }
}

impl GenerationDriver for FakeDriver {
    fn step(
        &mut self,
        token_ids: &[u32],
        cancellation: &dyn Cancellation,
    ) -> super::Result<Vec<f32>> {
        self.calls.push(token_ids.to_vec());
        if self.checks_each_prompt_token {
            for _ in token_ids {
                if cancellation.is_cancelled() {
                    return CancelledSnafu {
                        boundary: "fake native prefill token",
                    }
                    .fail();
                }
            }
        }
        match self.steps.pop_front() {
            Some(FakeStep::Logits(logits)) => Ok(logits),
            Some(FakeStep::Failure) => InvalidConfigurationSnafu {
                rule: "fake generation driver failed",
            }
            .fail(),
            None => InvalidConfigurationSnafu {
                rule: "fake generation driver has no programmed step",
            }
            .fail(),
        }
    }
}

struct CancelOnCheck {
    check: Cell<usize>,
    cancel_at: usize,
}

impl CancelOnCheck {
    fn new(cancel_at: usize) -> Self {
        Self {
            check: Cell::new(0),
            cancel_at,
        }
    }

    fn checks(&self) -> usize {
        self.check.get()
    }
}

impl Cancellation for CancelOnCheck {
    fn is_cancelled(&self) -> bool {
        let check = self.check.get() + 1;
        self.check.set(check);
        check >= self.cancel_at
    }
}

fn logits_for(token_id: usize) -> Vec<f32> {
    let mut logits = vec![0.0; TOKENS.len()];
    logits[token_id] = 1.0;
    logits
}

#[test]
fn shared_driver_receives_prompt_then_selected_decode_ids_and_retains_output() -> TestResult<()> {
    let tokenizer_json = tokenizer_json();
    let config = fixture_config(&TOKENS, 3, false, false, CONTENT_TEMPLATE);
    let fixture = build_qwen35_fixture(&config)?;
    let (_directory, artifact) = load_fixture(&fixture)?;
    let pipeline = pipeline_with_tokenizer(&artifact, &tokenizer_json)?;
    let messages = [TextMessage::new(TextRole::User, "assistant")];
    let prepared =
        pipeline.prepare(GenerationRequest::new(&messages, 2, false), &NeverCancelled)?;
    let mut driver = FakeDriver::with_steps([
        FakeStep::Logits(logits_for(3)),
        FakeStep::Logits(logits_for(4)),
    ]);

    let generation = prepared.generate_with_driver(&mut driver, &NeverCancelled)?;

    assert_eq!(driver.calls, [vec![4], vec![3]]);
    assert_eq!(generation.token_ids(), [3, 4]);
    assert_eq!(generation.finish_reason(), FinishReason::Length);
    assert_eq!(generation.text(), "hello assistant");
    assert!(
        generation.text.capacity() >= 128,
        "the returned string must be the pre-acquired retained output owner"
    );
    assert!(
        generation.token_ids.capacity() >= 2,
        "the returned IDs must be the pre-acquired retained ID owner"
    );
    Ok(())
}

#[test]
fn output_storage_is_acquired_before_the_first_driver_step() -> TestResult<()> {
    let tokenizer_json = tokenizer_json();
    let config = fixture_config(&TOKENS, 3, false, false, CONTENT_TEMPLATE);
    let fixture = build_qwen35_fixture(&config)?;
    let (_directory, artifact) = load_fixture(&fixture)?;
    let mut limits = test_limits(tokenizer_json.len())?;
    limits.output_bytes = usize::MAX
        .checked_sub(size_of::<u32>())
        .ok_or_else(|| std::io::Error::other("usize cannot hold one token ID"))?;
    let pipeline = pipeline_result(&artifact, &tokenizer_json, limits)?;
    let messages = [TextMessage::new(TextRole::User, "hello")];
    let prepared =
        pipeline.prepare(GenerationRequest::new(&messages, 1, false), &NeverCancelled)?;
    let mut driver = FakeDriver::with_steps([FakeStep::Logits(logits_for(3))]);

    let error = text_error(prepared.generate_with_driver(&mut driver, &NeverCancelled))?;

    assert!(
        matches!(
            error,
            Error::Allocation {
                target: "decoded output bytes",
                ..
            }
        ),
        "the impossible retained output reservation must remain a typed allocation failure"
    );
    assert!(
        driver.calls.is_empty(),
        "no driver step may begin before every output and scratch owner is acquired"
    );
    Ok(())
}

#[test]
fn prepared_storage_plan_separates_retained_and_scratch_extents() -> TestResult<()> {
    let tokenizer_json = tokenizer_json();
    let config = fixture_config(&TOKENS, 3, false, false, CONTENT_TEMPLATE);
    let fixture = build_qwen35_fixture(&config)?;
    let (_directory, artifact) = load_fixture(&fixture)?;
    let limits = test_limits(tokenizer_json.len())?;
    let pipeline = pipeline_result(&artifact, &tokenizer_json, limits)?;
    let messages = [TextMessage::new(TextRole::User, "hello")];
    let prepared =
        pipeline.prepare(GenerationRequest::new(&messages, 2, false), &NeverCancelled)?;
    let generated_id_bytes = 2usize
        .checked_mul(size_of::<u32>())
        .ok_or_else(|| std::io::Error::other("generated ID byte count overflowed"))?;
    let retained_bytes = limits
        .output_bytes
        .checked_add(generated_id_bytes)
        .ok_or_else(|| std::io::Error::other("retained byte count overflowed"))?;

    assert_eq!(
        prepared.output_storage_plan.requested_retained_bytes(),
        retained_bytes,
        "retained accounting must contain only the moved output and ID owners"
    );
    assert!(
        prepared.output_storage_plan.requested_scratch_bytes() > 0,
        "transform arenas and indexes must remain separately scratch-owned"
    );
    Ok(())
}

#[test]
fn injected_driver_decode_limit_failure_publishes_no_generation() -> TestResult<()> {
    let tokenizer_json = tokenizer_json();
    let config = fixture_config(&TOKENS, 3, false, false, CONTENT_TEMPLATE);
    let fixture = build_qwen35_fixture(&config)?;
    let (_directory, artifact) = load_fixture(&fixture)?;
    let mut limits = test_limits(tokenizer_json.len())?;
    limits.output_bytes = 4;
    let pipeline = pipeline_result(&artifact, &tokenizer_json, limits)?;
    let messages = [TextMessage::new(TextRole::User, "hello")];
    let prepared =
        pipeline.prepare(GenerationRequest::new(&messages, 1, false), &NeverCancelled)?;
    let mut driver = FakeDriver::with_steps([FakeStep::Logits(logits_for(3))]);

    let error = text_error(prepared.generate_with_driver(&mut driver, &NeverCancelled))?;

    assert!(
        matches!(
            error,
            Error::LimitExceeded {
                field: "decoded output bytes",
                actual: 5,
                limit: 4,
                ..
            }
        ),
        "bounded collective decoding must preserve the established text limit error"
    );
    assert_eq!(
        driver.calls,
        [vec![3]],
        "the failure must occur after selection but before publishing a generation"
    );
    Ok(())
}

#[test]
fn driver_can_observe_cancellation_between_native_prefill_tokens() -> TestResult<()> {
    let tokenizer_json = tokenizer_json();
    let config = fixture_config(&TOKENS, 3, true, true, CONTENT_TEMPLATE);
    let fixture = build_qwen35_fixture(&config)?;
    let (_directory, artifact) = load_fixture(&fixture)?;
    let pipeline = pipeline_with_tokenizer(&artifact, &tokenizer_json)?;
    let messages = [TextMessage::new(TextRole::User, "hello")];
    let prepared =
        pipeline.prepare(GenerationRequest::new(&messages, 1, false), &NeverCancelled)?;
    let cancellation = CancelOnCheck::new(4);
    let mut driver = FakeDriver::with_prompt_token_checks([]);

    let error = text_error(prepared.generate_with_driver(&mut driver, &cancellation))?;

    assert!(
        matches!(
            error,
            Error::Cancelled {
                boundary: "fake native prefill token",
                ..
            }
        ),
        "a per-token adapter cancellation must remain a typed pipeline cancellation"
    );
    assert_eq!(driver.calls, [vec![1, 3, 2]]);
    assert_eq!(cancellation.checks(), 4);
    Ok(())
}

#[test]
fn driver_failure_after_a_selected_token_returns_no_output() -> TestResult<()> {
    let tokenizer_json = tokenizer_json();
    let config = fixture_config(&TOKENS, 3, false, false, CONTENT_TEMPLATE);
    let fixture = build_qwen35_fixture(&config)?;
    let (_directory, artifact) = load_fixture(&fixture)?;
    let pipeline = pipeline_with_tokenizer(&artifact, &tokenizer_json)?;
    let messages = [TextMessage::new(TextRole::User, "hello")];
    let prepared =
        pipeline.prepare(GenerationRequest::new(&messages, 2, false), &NeverCancelled)?;
    let mut driver = FakeDriver::with_steps([FakeStep::Logits(logits_for(3)), FakeStep::Failure]);

    let error = text_error(prepared.generate_with_driver(&mut driver, &NeverCancelled))?;

    assert!(
        matches!(
            error,
            Error::InvalidConfiguration {
                rule: "fake generation driver failed",
                ..
            }
        ),
        "the exact typed backend error must be returned without an output value"
    );
    assert_eq!(driver.calls, [vec![3], vec![3]]);
    Ok(())
}

#[test]
fn preparation_profile_owner_accepts_clones_and_refuses_an_equal_independent_profile()
-> TestResult<()> {
    let tokenizer_json = tokenizer_json();
    let config = fixture_config(&TOKENS, 3, false, false, CONTENT_TEMPLATE);
    let fixture = build_qwen35_fixture(&config)?;
    let (_directory, artifact) = load_fixture(&fixture)?;
    let first = pipeline_with_tokenizer(&artifact, &tokenizer_json)?;
    let clone = first.clone();
    let independent = pipeline_with_tokenizer(&artifact, &tokenizer_json)?;
    let messages = [TextMessage::new(TextRole::User, "hello")];
    let prepared = first.prepare(GenerationRequest::new(&messages, 1, false), &NeverCancelled)?;

    assert!(clone.owns_preparation(&prepared));
    assert!(first.owns_preparation(&prepared));
    assert!(
        !independent.owns_preparation(&prepared),
        "equal tokenizer/artifact bytes must not substitute for the retained profile owner"
    );
    drop(first);
    assert!(
        clone.owns_preparation(&prepared),
        "a clone must retain the same profile after the original wrapper drops"
    );
    Ok(())
}

#[test]
fn distinct_preparations_bind_exact_renderings_and_final_prompt_ids() -> TestResult<()> {
    let tokenizer_json = tokenizer_json();
    let config = fixture_config(&TOKENS, 3, true, true, CONTENT_TEMPLATE);
    let fixture = build_qwen35_fixture(&config)?;
    let (_directory, artifact) = load_fixture(&fixture)?;
    let pipeline = pipeline_with_tokenizer(&artifact, &tokenizer_json)?;
    let hello_messages = [TextMessage::new(TextRole::User, "hello")];
    let assistant_messages = [TextMessage::new(TextRole::User, "assistant")];

    let hello = pipeline.prepare(
        GenerationRequest::new(&hello_messages, 1, false),
        &NeverCancelled,
    )?;
    let assistant = pipeline.prepare(
        GenerationRequest::new(&assistant_messages, 1, false),
        &NeverCancelled,
    )?;

    assert_eq!(hello.rendered_prompt(), "hello");
    assert_eq!(hello.prompt_token_ids(), [1, 3, 2]);
    assert_eq!(assistant.rendered_prompt(), "assistant");
    assert_eq!(assistant.prompt_token_ids(), [1, 4, 2]);
    assert_ne!(hello.prompt_token_ids(), assistant.prompt_token_ids());
    Ok(())
}

#[test]
fn prepared_report_binds_both_identities_and_exact_request_axes() -> TestResult<()> {
    let tokenizer_json = tokenizer_json();
    let expected_tokenizer = tokenizer_identity(&tokenizer_json);
    let (_directory, artifact) = verified_artifact(3, true, false)?;
    let inspection = artifact.observation().inspection();
    let expected_artifact = inspection.digest;
    let expected_backing_bytes = inspection.file_len;
    let pipeline = pipeline_with_tokenizer(&artifact, &tokenizer_json)?;
    let messages = [TextMessage::new(TextRole::User, "hello")];
    let prepared =
        pipeline.prepare(GenerationRequest::new(&messages, 2, false), &NeverCancelled)?;
    let prompt_tokens = prepared.prompt_token_ids().len();
    let requirements = prepared.decoder_cpu_requirements();

    assert_eq!(prepared.tokenizer_identity(), expected_tokenizer);
    assert_eq!(requirements.artifact_digest(), expected_artifact);
    assert_eq!(
        requirements.serialized_backing_bytes(),
        expected_backing_bytes
    );
    assert_eq!(requirements.selection(), Qwen35LogitSelection::LastToken);
    assert_eq!(requirements.max_step_tokens(), prompt_tokens);
    assert_eq!(
        requirements.max_context(),
        prompt_tokens + prepared.max_output_tokens()
    );
    Ok(())
}

#[test]
fn prepared_request_retains_profile_after_pipeline_and_artifact_drop() -> TestResult<()> {
    let (prepared, expected_tokenizer, expected_artifact) = {
        let tokenizer_json = tokenizer_json();
        let expected_tokenizer = tokenizer_identity(&tokenizer_json);
        let config = fixture_config(&TOKENS, 3, true, true, CONTENT_TEMPLATE);
        let fixture = build_qwen35_fixture(&config)?;
        let (directory, artifact) = load_fixture(&fixture)?;
        let expected_artifact = artifact.observation().inspection().digest;
        let pipeline = pipeline_with_tokenizer(&artifact, &tokenizer_json)?;
        let messages = [TextMessage::new(TextRole::User, "hello")];
        let prepared =
            pipeline.prepare(GenerationRequest::new(&messages, 1, false), &NeverCancelled)?;
        drop(pipeline);
        drop(artifact);
        drop(directory);
        (prepared, expected_tokenizer, expected_artifact)
    };

    assert_eq!(prepared.rendered_prompt(), "hello");
    assert_eq!(prepared.prompt_token_ids(), [1, 3, 2]);
    assert_eq!(prepared.tokenizer_identity(), expected_tokenizer);
    assert_eq!(
        prepared.decoder_cpu_requirements().artifact_digest(),
        expected_artifact
    );
    assert_eq!(prepared.context_tokens(), 4);
    assert_eq!(prepared.max_output_tokens(), 1);
    assert_eq!(
        prepared
            .verified_weights()
            .execution_plan(4, 3, Qwen35LogitSelection::LastToken)?
            .cpu_requirements()
            .artifact_digest(),
        expected_artifact,
        "the prepared owner must retain its exact verified weights"
    );
    let generation = prepared.generate(&NeverCancelled)?;
    assert_eq!(generation.token_ids(), [3]);
    assert_eq!(generation.text(), "hello");
    Ok(())
}

#[test]
fn configured_ceiling_above_artifact_admits_only_requests_the_artifact_can_execute()
-> TestResult<()> {
    let tokenizer_json = tokenizer_json();
    let (_directory, artifact) = verified_artifact(3, true, false)?;
    let mut limits = test_limits(tokenizer_json.len())?;
    limits.context_tokens = 9;
    limits.output_tokens = 7;
    let pipeline = pipeline_result(&artifact, &tokenizer_json, limits)?;
    let messages = [TextMessage::new(TextRole::User, "hello")];

    let fitting = pipeline.prepare(GenerationRequest::new(&messages, 6, false), &NeverCancelled)?;
    assert_eq!(fitting.decoder_cpu_requirements().max_context(), 8);

    let error =
        text_error(pipeline.prepare(GenerationRequest::new(&messages, 7, false), &NeverCancelled))?;
    assert!(
        matches!(
            error,
            Error::Decoder {
                source: decoders::Error::ExecutionContext {
                    requested: 9,
                    rule: "caller context must not exceed artifact context_length",
                    ..
                },
                ..
            }
        ),
        "an artifact-bound decoder error must reject only the actually oversized request"
    );
    Ok(())
}

#[test]
fn malformed_execution_payload_prepares_inertly_and_fails_only_when_consumed() -> TestResult<()> {
    let tokenizer_json = tokenizer_json();
    let config = fixture_config(&TOKENS, 3, false, false, "hello");
    let (_directory, artifact) = mutated_artifact(&config, |raw| {
        set_f32_row(raw, "token_embd.weight", 3, &[f32::NAN, 1.0, 1.0])
    })?;
    let pipeline = pipeline_with_tokenizer(&artifact, &tokenizer_json)?;
    let messages = [TextMessage::new(TextRole::User, "hello")];

    let prepared =
        pipeline.prepare(GenerationRequest::new(&messages, 1, false), &NeverCancelled)?;
    assert_eq!(prepared.prompt_token_ids(), [3]);
    let error = text_error(prepared.generate(&NeverCancelled))?;
    assert!(
        matches!(
            &error,
            Error::Decoder {
                source: decoders::Error::ProjectionRow { name, row, .. },
                ..
            } if name == "token_embd.weight" && *row == 3
        ),
        "malformed execution data must remain inert until the preparation is consumed: {error:?}"
    );
    Ok(())
}

#[test]
fn cancellation_that_flips_as_tokenization_starts_blocks_consumption_and_retry_is_fresh()
-> TestResult<()> {
    let (_directory, artifact) = verified_artifact(3, true, false)?;
    let tokenizer_json = tokenizer_json();
    let pipeline = pipeline_with_tokenizer(&artifact, &tokenizer_json)?;
    let messages = [TextMessage::new(TextRole::User, "hello")];
    let cancellation = CancelWhenTokenizationStarts::new();

    let prepared = pipeline.prepare(GenerationRequest::new(&messages, 1, false), &cancellation)?;
    assert!(
        cancellation.has_flipped(),
        "cancellation must become visible while tokenizer work is non-preemptible"
    );
    let error = text_error(prepared.generate(&cancellation))?;
    assert!(
        matches!(
            error,
            Error::Cancelled {
                boundary: "decoder session construction",
                ..
            }
        ),
        "an inert preparation must not publish output after cancellation becomes visible"
    );

    let retry = pipeline.generate(GenerationRequest::new(&messages, 1, false), &NeverCancelled)?;
    assert_eq!(retry.token_ids(), [3]);
    assert_eq!(retry.text(), "hello");
    Ok(())
}
