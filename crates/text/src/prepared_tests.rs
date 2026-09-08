use std::cell::Cell;

use decoders::Qwen35LogitSelection;
use sha2::{Digest, Sha256};
use test_fixtures::build_qwen35_fixture;
use tokenize::{TokenizerDigest, TokenizerIdentity};

use super::{Cancellation, Error, GenerationRequest, NeverCancelled, TextMessage, TextRole};
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
            error,
            Error::Decoder {
                source: decoders::Error::ExecutionArithmetic { .. },
                ..
            }
        ),
        "malformed execution data must remain inert until the preparation is consumed"
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
