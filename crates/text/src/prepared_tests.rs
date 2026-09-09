use std::cell::Cell;
use std::collections::VecDeque;
use std::fmt::{Display, Formatter};
use std::mem::size_of;
use std::rc::Rc;

use decoders::Qwen35LogitSelection;
use sha2::{Digest, Sha256};
use test_fixtures::build_qwen35_fixture;
use tokenize::{TokenizerDigest, TokenizerIdentity};

use super::{
    Cancellation, Error, FinishReason, GenerationDriver, GenerationLogits, GenerationRequest,
    LegacyGenerationLogits, NeverCancelled, PreparedGeneration, RecycledGenerationDriver,
    RecycledGenerationError, RecycledLogitsPlan, RecycledLogitsStorage, TextMessage, TextRole,
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

fn recycled_error<T, DriverError>(
    result: std::result::Result<T, RecycledGenerationError<DriverError>>,
) -> TestResult<RecycledGenerationError<DriverError>>
where
    DriverError: std::error::Error,
{
    match result {
        Err(error) => Ok(error),
        Ok(_) => Err(std::io::Error::other("recycled generation unexpectedly succeeded").into()),
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

struct FakeRecycledDriver {
    steps: VecDeque<FakeStep>,
    calls: Vec<Vec<u32>>,
    row_pointers: Vec<*const f32>,
    checks_each_prompt_token: bool,
}

impl FakeRecycledDriver {
    fn with_steps(steps: impl IntoIterator<Item = FakeStep>) -> Self {
        Self {
            steps: steps.into_iter().collect(),
            calls: Vec::new(),
            row_pointers: Vec::new(),
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

impl RecycledGenerationDriver for FakeRecycledDriver {
    type Error = Error;

    fn step_into(
        &mut self,
        token_ids: &[u32],
        logits: &mut [f32],
        cancellation: &dyn Cancellation,
    ) -> super::Result<()> {
        self.calls.push(token_ids.to_vec());
        self.row_pointers.push(logits.as_ptr());
        if self.checks_each_prompt_token {
            for _ in token_ids {
                if cancellation.is_cancelled() {
                    return CancelledSnafu {
                        boundary: "fake recycled native prefill token",
                    }
                    .fail();
                }
            }
        }
        match self.steps.pop_front() {
            Some(FakeStep::Logits(row)) if row.len() == logits.len() => {
                logits.copy_from_slice(&row);
                Ok(())
            }
            Some(FakeStep::Logits(_)) => InvalidConfigurationSnafu {
                rule: "fake recycled driver row has the wrong width",
            }
            .fail(),
            Some(FakeStep::Failure) => InvalidConfigurationSnafu {
                rule: "fake recycled generation driver failed",
            }
            .fail(),
            None => InvalidConfigurationSnafu {
                rule: "fake recycled generation driver has no programmed step",
            }
            .fail(),
        }
    }
}

#[derive(Debug)]
struct CustodyDriverError {
    custody: Rc<usize>,
}

impl Display for CustodyDriverError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("fake driver retained resource custody")
    }
}

impl std::error::Error for CustodyDriverError {}

struct CustodyFailureDriver {
    failure: Option<CustodyDriverError>,
}

impl RecycledGenerationDriver for CustodyFailureDriver {
    type Error = CustodyDriverError;

    fn step_into(
        &mut self,
        _token_ids: &[u32],
        _logits: &mut [f32],
        _cancellation: &dyn Cancellation,
    ) -> std::result::Result<(), Self::Error> {
        match self.failure.take() {
            Some(error) => Err(error),
            None => Ok(()),
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

fn acquire_logits(prepared: &PreparedGeneration) -> super::Result<RecycledLogitsStorage> {
    prepared.recycled_logits_plan().acquire()
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
fn legacy_adapter_borrows_the_driver_vec_without_a_second_row() -> TestResult<()> {
    struct PointerDriver<'pointer> {
        returned_pointer: &'pointer Cell<*const f32>,
    }

    impl GenerationDriver for PointerDriver<'_> {
        fn step(
            &mut self,
            _token_ids: &[u32],
            _cancellation: &dyn Cancellation,
        ) -> super::Result<Vec<f32>> {
            let row = logits_for(3);
            self.returned_pointer.set(row.as_ptr());
            Ok(row)
        }
    }

    let returned_pointer = Cell::new(std::ptr::null());
    let mut driver = PointerDriver {
        returned_pointer: &returned_pointer,
    };
    let mut adapter = LegacyGenerationLogits::new(&mut driver);
    let row = adapter.step(&[3], &NeverCancelled)?;

    assert_eq!(
        row.as_ptr(),
        returned_pointer.get(),
        "the loop must borrow the exact Vec allocation returned by the legacy driver"
    );
    Ok(())
}

#[test]
fn recycled_driver_preserves_output_and_reuses_one_row_across_steps() -> TestResult<()> {
    let tokenizer_json = tokenizer_json();
    let config = fixture_config(&TOKENS, 3, false, false, CONTENT_TEMPLATE);
    let fixture = build_qwen35_fixture(&config)?;
    let (_directory, artifact) = load_fixture(&fixture)?;
    let pipeline = pipeline_with_tokenizer(&artifact, &tokenizer_json)?;
    let messages = [TextMessage::new(TextRole::User, "assistant")];
    let prepared =
        pipeline.prepare(GenerationRequest::new(&messages, 2, false), &NeverCancelled)?;
    let logits_storage = acquire_logits(&prepared)?;
    let mut driver = FakeRecycledDriver::with_steps([
        FakeStep::Logits(logits_for(3)),
        FakeStep::Logits(logits_for(4)),
    ]);

    let generation =
        prepared.generate_with_recycled_driver(&mut driver, logits_storage, &NeverCancelled)?;

    assert_eq!(
        driver.calls,
        [vec![4], vec![3]],
        "the recycled port must receive the prompt followed by one selected continuation ID"
    );
    assert_eq!(
        driver.row_pointers.len(),
        2,
        "the two generation steps must each receive the acquired row"
    );
    assert!(
        driver
            .row_pointers
            .windows(2)
            .all(|pair| pair.first() == pair.get(1)),
        "every recycled step must receive the same allocation identity"
    );
    assert_eq!(
        generation.token_ids(),
        [3, 4],
        "the recycled port must retain both independently expected token IDs"
    );
    assert_eq!(
        generation.finish_reason(),
        FinishReason::Length,
        "the recycled port must preserve length completion"
    );
    assert_eq!(
        generation.text(),
        "hello assistant",
        "the recycled port must preserve collective output decoding"
    );
    Ok(())
}

#[test]
fn recycled_driver_error_moves_non_send_custody_without_losing_identity() -> TestResult<()> {
    let tokenizer_json = tokenizer_json();
    let config = fixture_config(&TOKENS, 3, false, false, CONTENT_TEMPLATE);
    let fixture = build_qwen35_fixture(&config)?;
    let (_directory, artifact) = load_fixture(&fixture)?;
    let pipeline = pipeline_with_tokenizer(&artifact, &tokenizer_json)?;
    let messages = [TextMessage::new(TextRole::User, "hello")];
    let prepared =
        pipeline.prepare(GenerationRequest::new(&messages, 1, false), &NeverCancelled)?;
    let logits_storage = acquire_logits(&prepared)?;
    let custody = Rc::new(41usize);
    let custody_identity = Rc::as_ptr(&custody);
    let mut driver = CustodyFailureDriver {
        failure: Some(CustodyDriverError { custody }),
    };

    let error = recycled_error(prepared.generate_with_recycled_driver(
        &mut driver,
        logits_storage,
        &NeverCancelled,
    ))?;

    match error {
        RecycledGenerationError::Driver { source } => {
            assert_eq!(
                Rc::as_ptr(&source.custody),
                custody_identity,
                "the driver failure must retain the exact non-Send custody allocation"
            );
            assert_eq!(
                *source.custody, 41,
                "the driver failure must preserve its owned custody value"
            );
        }
        RecycledGenerationError::Pipeline { source } => {
            return Err(std::io::Error::other(format!(
                "driver custody was flattened into a pipeline error: {source}"
            ))
            .into());
        }
    }
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
fn recycled_port_acquires_output_storage_before_the_first_driver_step() -> TestResult<()> {
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
    let logits_storage = acquire_logits(&prepared)?;
    let mut driver = FakeRecycledDriver::with_steps([FakeStep::Logits(logits_for(3))]);

    let error = recycled_error(prepared.generate_with_recycled_driver(
        &mut driver,
        logits_storage,
        &NeverCancelled,
    ))?;

    assert!(
        matches!(
            error,
            RecycledGenerationError::Pipeline {
                source: Error::Allocation {
                    target: "decoded output bytes",
                    ..
                }
            }
        ),
        "the recycled port must preserve output pre-acquisition failure"
    );
    assert!(
        driver.calls.is_empty(),
        "no recycled driver step may begin before collective storage acquisition"
    );
    Ok(())
}

#[test]
fn legacy_driver_wrong_shape_remains_a_typed_pipeline_error() -> TestResult<()> {
    let tokenizer_json = tokenizer_json();
    let config = fixture_config(&TOKENS, 3, false, false, CONTENT_TEMPLATE);
    let fixture = build_qwen35_fixture(&config)?;
    let (_directory, artifact) = load_fixture(&fixture)?;
    let pipeline = pipeline_with_tokenizer(&artifact, &tokenizer_json)?;
    let messages = [TextMessage::new(TextRole::User, "hello")];
    let prepared =
        pipeline.prepare(GenerationRequest::new(&messages, 1, false), &NeverCancelled)?;
    let mut driver = FakeDriver::with_steps([FakeStep::Logits(vec![0.0; TOKENS.len() - 1])]);

    let error = text_error(prepared.generate_with_driver(&mut driver, &NeverCancelled))?;

    assert!(
        matches!(
            error,
            Error::LogitShape {
                actual,
                expected,
                ..
            } if actual == TOKENS.len() - 1 && expected == TOKENS.len()
        ),
        "the unchanged legacy port must reject a returned row with the wrong width"
    );
    assert_eq!(
        driver.calls,
        [vec![3]],
        "shape refusal must occur after exactly one legacy driver invocation"
    );
    Ok(())
}

#[test]
fn recycled_storage_width_mismatch_refuses_before_driver_invocation() -> TestResult<()> {
    const WIDE_TOKENS: [&str; 6] = ["[UNK]", "<bos>", "<eos>", "hello", "assistant", "extra"];

    let tokenizer_json = tokenizer_json();
    let wide_tokenizer_json =
        tokenizer_json.replace("\"assistant\":4}", "\"assistant\":4,\"extra\":5}");
    let wide_config = fixture_config(&WIDE_TOKENS, 3, false, false, CONTENT_TEMPLATE);
    let wide_fixture = build_qwen35_fixture(&wide_config)?;
    let (_wide_directory, wide_artifact) = load_fixture(&wide_fixture)?;
    let wide_pipeline = pipeline_with_tokenizer(&wide_artifact, &wide_tokenizer_json)?;
    let messages = [TextMessage::new(TextRole::User, "hello")];
    let wide_prepared =
        wide_pipeline.prepare(GenerationRequest::new(&messages, 1, false), &NeverCancelled)?;
    let mismatched_storage = acquire_logits(&wide_prepared)?;

    let config = fixture_config(&TOKENS, 3, false, false, CONTENT_TEMPLATE);
    let fixture = build_qwen35_fixture(&config)?;
    let (_directory, artifact) = load_fixture(&fixture)?;
    let pipeline = pipeline_with_tokenizer(&artifact, &tokenizer_json)?;
    let prepared =
        pipeline.prepare(GenerationRequest::new(&messages, 1, false), &NeverCancelled)?;
    let mut driver = FakeRecycledDriver::with_steps([]);

    let validation = text_error(
        prepared
            .recycled_logits_plan()
            .validate_storage(&mismatched_storage),
    )?;
    assert!(
        matches!(
            validation,
            Error::RecycledLogitsStorageMismatch {
                actual: 6,
                expected: 5,
                ..
            }
        ),
        "the public plan check must reject substituted storage before session construction"
    );

    let error = recycled_error(prepared.generate_with_recycled_driver(
        &mut driver,
        mismatched_storage,
        &NeverCancelled,
    ))?;

    assert!(
        matches!(
            error,
            RecycledGenerationError::Pipeline {
                source: Error::RecycledLogitsStorageMismatch {
                    actual: 6,
                    expected: 5,
                    ..
                }
            }
        ),
        "storage from another vocabulary width must remain a typed pipeline refusal"
    );
    assert!(
        driver.calls.is_empty(),
        "storage mismatch must be rejected before invoking the recycled driver"
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
        prepared.output_storage_plan().requested_retained_bytes(),
        retained_bytes,
        "retained accounting must contain only the moved output and ID owners"
    );
    assert!(
        prepared.output_storage_plan().requested_scratch_bytes() > 0,
        "transform arenas and indexes must remain separately scratch-owned"
    );
    assert_eq!(
        prepared.recycled_logits_plan().vocabulary_width(),
        TOKENS.len(),
        "the recycled row width must come from the exact verified tokenizer"
    );
    assert_eq!(
        prepared.recycled_logits_plan().requested_bytes(),
        TOKENS.len() * size_of::<f32>(),
        "recycled row accounting must expose the checked logical f32 extent"
    );
    Ok(())
}

#[test]
fn recycled_logits_plan_checks_extent_and_reports_allocation_refusal() -> TestResult<()> {
    let overflow = text_error(RecycledLogitsPlan::new(usize::MAX))?;
    assert!(
        matches!(
            overflow,
            Error::RecycledLogitsExtentOverflow {
                vocabulary_width: usize::MAX,
                ..
            }
        ),
        "an unrepresentable f32 extent must fail while deriving the plan"
    );

    let largest_representable_width = usize::MAX / size_of::<f32>();
    let plan = RecycledLogitsPlan::new(largest_representable_width)?;
    let allocation = text_error(plan.acquire())?;
    assert!(
        matches!(
            allocation,
            Error::Allocation {
                target: "recycled logits row",
                ..
            }
        ),
        "an allocator refusal must remain a typed fallible acquisition error"
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
fn recycled_driver_observes_the_same_internal_prefill_cancellation() -> TestResult<()> {
    let tokenizer_json = tokenizer_json();
    let config = fixture_config(&TOKENS, 3, true, true, CONTENT_TEMPLATE);
    let fixture = build_qwen35_fixture(&config)?;
    let (_directory, artifact) = load_fixture(&fixture)?;
    let pipeline = pipeline_with_tokenizer(&artifact, &tokenizer_json)?;
    let messages = [TextMessage::new(TextRole::User, "hello")];
    let prepared =
        pipeline.prepare(GenerationRequest::new(&messages, 1, false), &NeverCancelled)?;
    let logits_storage = acquire_logits(&prepared)?;
    let cancellation = CancelOnCheck::new(4);
    let mut driver = FakeRecycledDriver::with_prompt_token_checks([]);

    let error = recycled_error(prepared.generate_with_recycled_driver(
        &mut driver,
        logits_storage,
        &cancellation,
    ))?;

    assert!(
        matches!(
            error,
            RecycledGenerationError::Driver {
                source: Error::Cancelled {
                    boundary: "fake recycled native prefill token",
                    ..
                }
            }
        ),
        "internal recycled-driver cancellation must retain its driver error boundary"
    );
    assert_eq!(
        driver.calls,
        [vec![1, 3, 2]],
        "the recycled driver must receive the exact prepared prompt"
    );
    assert_eq!(
        cancellation.checks(),
        4,
        "the recycled port must preserve the legacy cancellation observation order"
    );
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
    let execution_profile = first.execution_profile();
    let expected_context_ceiling = test_limits(tokenizer_json.len())?.context_tokens;

    assert!(clone.owns_preparation(&prepared));
    assert!(first.owns_preparation(&prepared));
    assert!(
        !independent.owns_preparation(&prepared),
        "equal tokenizer/artifact bytes must not substitute for the retained profile owner"
    );
    assert!(
        std::ptr::eq(execution_profile.weights(), prepared.verified_weights()),
        "the generic execution profile must borrow the preparation's exact retained weights"
    );
    assert_eq!(
        execution_profile.context_ceiling(),
        expected_context_ceiling,
        "the execution profile must expose the configured context ceiling"
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
