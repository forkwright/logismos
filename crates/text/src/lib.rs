//! # text
//!
//! A bounded, artifact-bound native text request pipeline.
//!
//! [`TextPipeline`] binds one digest-verified GGUF payload, its embedded chat
//! template, and one explicitly receipt-verified `tokenizer.json`. It accepts
//! only text messages, renders through a capability-free MiniJinja environment,
//! then executes greedy CPU generation in a fresh session for each request.
//!
//! A failed pipeline call returns neither a session nor partial generated text.
//! Its private execution is dropped, but a completed underlying decoder step is
//! not rolled back before that drop.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::doc_markdown
)]

pub mod error;

use decoders::{Qwen35CpuRequirements, Qwen35ExecutionPlan, Qwen35LogitSelection, Qwen35Weights};
use loader::gguf::{MetaValue, VerifiedArtifact};
use serde::Serialize;
use snafu::{IntoError, ResultExt};
use templates::{BoundedTemplate, TemplateLimits};
use tokenize::{TokenizerByteLimit, TokenizerIdentity, VerifiedTokenizer};

use crate::error::{
    AllocationSnafu, CancelledSnafu, DecodeSnafu, DecoderSnafu, EmptyPromptSnafu,
    InvalidConfigurationSnafu, LimitExceededSnafu, LogitShapeSnafu, MetadataSnafu,
    RenderedUtf8Snafu, SpecialTokenPolicySnafu, TemplateRendererSnafu, TemplateSnafu,
    TokenizerSnafu, VocabularyMismatchSnafu,
};

pub use crate::error::{Error, Result};

const CHAT_TEMPLATE_KEY: &str = "tokenizer.chat_template";
const TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const BOS_TOKEN_ID_KEY: &str = "tokenizer.ggml.bos_token_id";
const EOS_TOKEN_ID_KEY: &str = "tokenizer.ggml.eos_token_id";
const ADD_BOS_TOKEN_KEY: &str = "tokenizer.ggml.add_bos_token";
const ADD_EOS_TOKEN_KEY: &str = "tokenizer.ggml.add_eos_token";
const EOT_TOKEN_ID_KEY: &str = "tokenizer.ggml.eot_token_id";
const EOM_TOKEN_ID_KEY: &str = "tokenizer.ggml.eom_token_id";

/// One trusted tokenizer companion supplied alongside a verified model.
///
/// The identity must come from configuration controlled by the service or
/// invoker. It is a content receipt, not publisher authentication; a service
/// must allowlist model/tokenizer pairs rather than accept these facts from an
/// untrusted request.
#[derive(Clone, Copy, Debug)]
pub struct TokenizerCompanion<'bytes> {
    bytes: &'bytes [u8],
    identity: TokenizerIdentity,
}

impl<'bytes> TokenizerCompanion<'bytes> {
    /// Bind raw tokenizer bytes to their independently recorded identity.
    #[must_use]
    pub const fn new(bytes: &'bytes [u8], identity: TokenizerIdentity) -> Self {
        Self { bytes, identity }
    }
}

/// Static upper bounds for one artifact-bound pipeline.
///
/// These bounds cover request input and retained output. MiniJinja temporary
/// allocations, tokenizer parse/encode/decode allocations, and output decoding
/// are deliberately outside the decoder execution memory report; this is not a
/// hostile-template or total-process-memory sandbox.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PipelineLimits {
    /// Maximum accepted tokenizer companion bytes before hashing or parsing.
    pub tokenizer_bytes: TokenizerByteLimit,
    /// Maximum embedded template UTF-8 byte length.
    pub template_bytes: usize,
    /// Maximum number of text messages in one request.
    pub messages: usize,
    /// Maximum UTF-8 bytes in a single message body.
    pub message_bytes: usize,
    /// Maximum total UTF-8 bytes across supplied message bodies.
    pub prompt_bytes: usize,
    /// Maximum rendered-chat UTF-8 bytes retained before tokenization.
    pub rendered_bytes: usize,
    /// Maximum prompt plus requested generated token IDs.
    pub context_tokens: usize,
    /// Maximum output token IDs requested by one call.
    pub output_tokens: usize,
    /// Maximum decoded UTF-8 output bytes retained for a successful call.
    pub output_bytes: usize,
    /// MiniJinja instruction budget for each render.
    pub template_fuel: u64,
    /// MiniJinja recursion limit for each render.
    pub template_recursion: usize,
}

impl PipelineLimits {
    fn validate(self) -> Result<TemplateLimits> {
        if [
            self.messages,
            self.message_bytes,
            self.prompt_bytes,
            self.context_tokens,
            self.output_tokens,
            self.output_bytes,
        ]
        .contains(&0)
        {
            return InvalidConfigurationSnafu {
                rule: "static limits must be non-zero and template fuel must fit isize",
            }
            .fail();
        }
        template_limits(self).map_err(|_| {
            InvalidConfigurationSnafu {
                rule: "static limits must be non-zero and template fuel must fit isize",
            }
            .build()
        })
    }
}

/// Text-only chat role accepted by the initial pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum TextRole {
    /// System instruction text.
    System,
    /// User-authored text.
    User,
    /// Prior assistant text.
    Assistant,
}

/// One text-only message supplied to an artifact chat template.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TextMessage {
    role: TextRole,
    content: String,
    tool_calls: Vec<()>,
    reasoning_content: Option<String>,
}

impl TextMessage {
    /// Construct one supported text message.
    #[must_use]
    pub fn new(role: TextRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            reasoning_content: None,
        }
    }

    /// Return the supported message role.
    #[must_use]
    pub const fn role(&self) -> TextRole {
        self.role
    }

    /// Borrow the message text.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }
}

/// One bounded non-streaming generation request.
#[derive(Clone, Copy, Debug)]
pub struct GenerationRequest<'messages> {
    messages: &'messages [TextMessage],
    max_output_tokens: usize,
    enable_thinking: bool,
}

impl<'messages> GenerationRequest<'messages> {
    /// Construct a request with a caller-selected output-token cap.
    #[must_use]
    pub const fn new(
        messages: &'messages [TextMessage],
        max_output_tokens: usize,
        enable_thinking: bool,
    ) -> Self {
        Self {
            messages,
            max_output_tokens,
            enable_thinking,
        }
    }
}

/// Why a successful non-streaming generation finished.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum FinishReason {
    /// An artifact-validated Qwen end-of-generation ID was selected and omitted.
    EndOfSequence,
    /// The request's output-token cap was reached.
    Length,
}

/// Complete generation returned only after successful collective decoding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Generation {
    text: String,
    token_ids: Vec<u32>,
    finish_reason: FinishReason,
}

/// One immutable prepared text request awaiting a single decoder session.
///
/// Preparation binds the artifact-derived chat rendering, final prompt IDs,
/// request-specific execution plan, and output cap. It does not allocate a
/// decoder session or execute model operations. The decoder requirement view
/// counts only decoder-owned logical `f32` storage; it excludes retained prompt
/// IDs, rendered text, template/tokenizer storage, process RSS, GPU memory, and
/// physical reservations.
///
/// WHY: qualification callers must inspect the exact artifact-bound request
/// without gaining a mutable prompt or a second execution path.
pub struct PreparedGeneration<'pipeline, 'artifact> {
    pipeline: &'pipeline TextPipeline<'artifact>,
    plan: Qwen35ExecutionPlan<'pipeline, 'artifact>,
    rendered_prompt: String,
    prompt_token_ids: Vec<u32>,
    max_output_tokens: usize,
}

impl<'pipeline, 'artifact> PreparedGeneration<'pipeline, 'artifact> {
    /// Borrow the exact template rendering bound to this prepared request.
    ///
    /// WHY: qualification evidence must be the rendering that will be decoded.
    #[must_use]
    pub fn rendered_prompt(&self) -> &str {
        &self.rendered_prompt
    }

    /// Borrow the final model prompt IDs, including artifact-selected specials.
    ///
    /// WHY: callers need token-level evidence without permission to alter it.
    #[must_use]
    pub fn prompt_token_ids(&self) -> &[u32] {
        &self.prompt_token_ids
    }

    /// Return the verified identity of the tokenizer bound to this request.
    ///
    /// WHY: prompt IDs only have meaning with their verified tokenizer bytes.
    #[must_use]
    pub const fn tokenizer_identity(&self) -> TokenizerIdentity {
        self.pipeline.tokenizer.identity()
    }

    /// Return the request-specific decoder-owned logical CPU requirements.
    ///
    /// WHY: callers can assess the exact decoder allocation shape before execution.
    #[must_use]
    pub const fn decoder_cpu_requirements(&self) -> Qwen35CpuRequirements {
        self.plan.cpu_requirements()
    }

    /// Return the prepared request's checked generated-token cap.
    ///
    /// WHY: the cap is part of the plan's context admission and execution contract.
    #[must_use]
    pub const fn max_output_tokens(&self) -> usize {
        self.max_output_tokens
    }

    /// Construct one decoder session and generate from this exact prepared prompt.
    ///
    /// This consumes the prepared request so callers cannot alter or reuse its
    /// prompt IDs with another pipeline or execution plan. A cancellation or
    /// later failure returns no partial text.
    ///
    /// WHY: consuming the preparation preserves the inspected prompt, decoder
    /// plan, and execution as one inseparable request.
    ///
    /// # Errors
    ///
    /// Returns a typed cancellation, allocation, decoder, tokenizer, or output
    /// limit error. No partial response is published on failure.
    pub fn generate(self, cancellation: &dyn Cancellation) -> Result<Generation> {
        let Self {
            pipeline,
            plan,
            rendered_prompt,
            prompt_token_ids,
            max_output_tokens,
        } = self;
        drop(rendered_prompt);
        check_cancelled(cancellation, "decoder session construction")?;
        let mut execution = plan.execution().context(DecoderSnafu)?;
        check_cancelled(cancellation, "prompt decoder step")?;
        let mut logits = execution.step(&prompt_token_ids).context(DecoderSnafu)?;
        let mut generated = Vec::new();
        generated
            .try_reserve_exact(max_output_tokens)
            .context(AllocationSnafu {
                target: "generated token IDs",
            })?;
        let finish_reason = loop {
            check_cancelled(cancellation, "greedy selection")?;
            let next = greedy_last_logits(&logits, pipeline.tokenizer.tokenizer().vocab_size())?;
            if pipeline.special_tokens.stop_ids.contains(&next) {
                break FinishReason::EndOfSequence;
            }
            generated.push(next);
            if generated.len() == max_output_tokens {
                break FinishReason::Length;
            }
            check_cancelled(cancellation, "next decoder step")?;
            // Selection is complete; do not retain the old vocabulary row while
            // the decoder allocates the next step's workspace and output.
            drop(logits);
            logits = execution.step(&[next]).context(DecoderSnafu)?;
        };
        check_cancelled(cancellation, "collective output decoding")?;
        let text = pipeline
            .tokenizer
            .tokenizer()
            .decode(&generated, false)
            .context(TokenizerSnafu)?;
        check_limit(
            "decoded output bytes",
            text.len(),
            pipeline.limits.output_bytes,
        )?;
        check_cancelled(cancellation, "publishing completed response")?;
        Ok(Generation {
            text,
            token_ids: generated,
            finish_reason,
        })
    }
}

impl Generation {
    /// Borrow collective decoded generated text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Borrow generated non-end-of-generation token IDs in order.
    #[must_use]
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    /// Return why generation completed.
    #[must_use]
    pub const fn finish_reason(&self) -> FinishReason {
        self.finish_reason
    }
}

/// Cancellation observation at defined request boundaries.
pub trait Cancellation {
    /// Return whether the caller requests cancellation before the next boundary.
    fn is_cancelled(&self) -> bool;
}

/// Cancellation source that never cancels.
#[derive(Clone, Copy, Debug, Default)]
pub struct NeverCancelled;

impl Cancellation for NeverCancelled {
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// Validated artifact special-token policy.
#[derive(Clone, Debug, Eq, PartialEq)]
struct SpecialTokenPolicy {
    bos_id: Option<u32>,
    eos_id: u32,
    add_bos: bool,
    add_eos: bool,
    stop_ids: Vec<u32>,
}

/// Artifact-bound text pipeline.
pub struct TextPipeline<'artifact> {
    weights: Qwen35Weights<'artifact>,
    tokenizer: VerifiedTokenizer,
    template: BoundedTemplate<'artifact>,
    special_tokens: SpecialTokenPolicy,
    limits: PipelineLimits,
}

impl<'artifact> TextPipeline<'artifact> {
    /// Verify one tokenizer companion and bind it to one verified GGUF artifact.
    ///
    /// The GGUF is the sole source of template, vocabulary, special IDs, and
    /// add-special policy. This constructor performs no sibling-file discovery.
    pub fn new(
        artifact: &'artifact VerifiedArtifact,
        companion: TokenizerCompanion<'_>,
        limits: PipelineLimits,
    ) -> Result<Self> {
        let template_limits = limits.validate()?;
        let tokenizer = VerifiedTokenizer::from_bytes(
            companion.bytes,
            companion.identity,
            limits.tokenizer_bytes,
        )
        .context(TokenizerSnafu)?;
        tokenizer
            .verify_unpadded_untruncated()
            .context(TokenizerSnafu)?;
        let metadata = artifact.observation().metadata();
        let template = metadata_string(metadata, CHAT_TEMPLATE_KEY)?;
        check_limit("template bytes", template.len(), limits.template_bytes)?;
        let special_tokens = verify_vocabulary(metadata, &tokenizer)?;
        let template =
            BoundedTemplate::new(template, template_limits).map_err(map_template_error)?;
        let weights = Qwen35Weights::try_from_verified(artifact).context(DecoderSnafu)?;
        Ok(Self {
            weights,
            tokenizer,
            template,
            special_tokens,
            limits,
        })
    }

    /// Prepare and generate greedily in one fresh private decoder session.
    ///
    /// A cancellation or later failure returns no partial text. Any completed
    /// decoder `step` remains committed only inside the private session that is
    /// immediately dropped; this method does not offer decoder rollback.
    /// Cancellation is cooperative at the named boundaries: template rendering,
    /// tokenizer work, and one complete decoder step are not preempted midway.
    ///
    /// WHY: the legacy one-call API must remain behaviorally identical while
    /// sharing the inspectable preparation boundary.
    ///
    /// # Errors
    ///
    /// Propagates preparation and generation errors; no partial response is
    /// published on cancellation or failure.
    pub fn generate(
        &self,
        request: GenerationRequest<'_>,
        cancellation: &dyn Cancellation,
    ) -> Result<Generation> {
        self.prepare(request, cancellation)?.generate(cancellation)
    }

    /// Render and tokenize one request into an immutable decoder execution plan.
    ///
    /// This performs no decoder-session allocation or model operation. Its plan
    /// uses the exact prompt-plus-output context and prompt-step width rather
    /// than the pipeline's configured ceilings.
    ///
    /// WHY: callers can qualify a concrete artifact-bound request before any
    /// mutable decoder state or model computation exists.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, validation, rendering, tokenizer, limit, or
    /// decoder-plan-admission errors.
    pub fn prepare(
        &self,
        request: GenerationRequest<'_>,
        cancellation: &dyn Cancellation,
    ) -> Result<PreparedGeneration<'_, 'artifact>> {
        check_cancelled(cancellation, "template rendering")?;
        self.validate_request(&request)?;
        let rendered = self.render(&request)?;
        check_cancelled(cancellation, "tokenization")?;
        let prompt = self.encode_prompt(&rendered)?;
        if prompt.is_empty() {
            return EmptyPromptSnafu.fail();
        }
        let requested_context = prompt
            .len()
            .checked_add(request.max_output_tokens)
            .ok_or_else(|| {
                InvalidConfigurationSnafu {
                    rule: "prompt plus output-token request overflowed usize",
                }
                .build()
            })?;
        check_limit(
            "prompt plus output tokens",
            requested_context,
            self.limits.context_tokens,
        )?;
        let plan = self
            .weights
            .execution_plan(
                requested_context,
                prompt.len(),
                Qwen35LogitSelection::LastToken,
            )
            .context(DecoderSnafu)?;
        Ok(PreparedGeneration {
            pipeline: self,
            plan,
            rendered_prompt: rendered,
            prompt_token_ids: prompt,
            max_output_tokens: request.max_output_tokens,
        })
    }

    fn validate_request(&self, request: &GenerationRequest<'_>) -> Result<()> {
        check_limit("messages", request.messages.len(), self.limits.messages)?;
        if request.messages.is_empty() {
            return InvalidConfigurationSnafu {
                rule: "generation request must contain at least one text message",
            }
            .fail();
        }
        if !request
            .messages
            .iter()
            .any(|message| matches!(message.role, TextRole::User))
        {
            return InvalidConfigurationSnafu {
                rule: "generation request must contain at least one user text message",
            }
            .fail();
        }
        check_limit(
            "requested output tokens",
            request.max_output_tokens,
            self.limits.output_tokens,
        )?;
        if request.max_output_tokens == 0 {
            return InvalidConfigurationSnafu {
                rule: "generation request output-token limit must be non-zero",
            }
            .fail();
        }
        let mut total_bytes = 0usize;
        for (index, message) in request.messages.iter().enumerate() {
            if matches!(message.role, TextRole::System) && index != 0 {
                return InvalidConfigurationSnafu {
                    rule: "a system text message is supported only at the start of a request",
                }
                .fail();
            }
            check_limit(
                "message bytes",
                message.content.len(),
                self.limits.message_bytes,
            )?;
            total_bytes = total_bytes
                .checked_add(message.content.len())
                .ok_or_else(|| {
                    InvalidConfigurationSnafu {
                        rule: "request message bytes overflowed usize",
                    }
                    .build()
                })?;
        }
        check_limit(
            "request prompt bytes",
            total_bytes,
            self.limits.prompt_bytes,
        )
    }

    fn render(&self, request: &GenerationRequest<'_>) -> Result<String> {
        self.template
            .render(RenderContext {
                messages: request.messages,
                add_generation_prompt: true,
                enable_thinking: request.enable_thinking,
                tools: Vec::<()>::new(),
            })
            .map_err(map_template_error)
    }

    fn encode_prompt(&self, rendered: &str) -> Result<Vec<u32>> {
        let mut prompt = self
            .tokenizer
            .tokenizer()
            .encode(rendered, false)
            .context(TokenizerSnafu)?;
        prompt.try_reserve(2).context(AllocationSnafu {
            target: "prompt special-token prefix/suffix",
        })?;
        if self.special_tokens.add_bos {
            let bos_id = self.special_tokens.bos_id.ok_or_else(|| {
                SpecialTokenPolicySnafu {
                    rule: "add_bos requires a declared BOS token ID",
                }
                .build()
            })?;
            prompt.insert(0, bos_id);
        }
        if self.special_tokens.add_eos {
            prompt.push(self.special_tokens.eos_id);
        }
        Ok(prompt)
    }
}

#[derive(Serialize)]
struct RenderContext<'messages> {
    messages: &'messages [TextMessage],
    add_generation_prompt: bool,
    enable_thinking: bool,
    tools: Vec<()>,
}

fn template_limits(limits: PipelineLimits) -> templates::Result<TemplateLimits> {
    TemplateLimits::new(
        limits.template_bytes,
        limits.rendered_bytes,
        limits.template_fuel,
        limits.template_recursion,
    )
}

fn map_template_error(error: templates::Error) -> Error {
    match error {
        templates::Error::Allocation { target, source, .. } => {
            AllocationSnafu { target }.into_error(source)
        }
        templates::Error::Template { source, .. } => TemplateSnafu.into_error(source),
        templates::Error::RenderedUtf8 { source, .. } => RenderedUtf8Snafu.into_error(source),
        templates::Error::LimitExceeded {
            field,
            actual,
            limit,
            ..
        } => LimitExceededSnafu {
            field,
            actual,
            limit,
        }
        .build(),
        templates::Error::InvalidConfiguration { rule, .. } => {
            InvalidConfigurationSnafu { rule }.build()
        }
        _ => TemplateRendererSnafu.into_error(error),
    }
}

fn metadata_string<'metadata>(
    metadata: &'metadata std::collections::HashMap<String, MetaValue>,
    key: &'static str,
) -> Result<&'metadata str> {
    match metadata.get(key) {
        Some(MetaValue::String(value)) => Ok(value),
        _ => MetadataSnafu { key }.fail(),
    }
}

fn metadata_u32(
    metadata: &std::collections::HashMap<String, MetaValue>,
    key: &'static str,
) -> Result<u32> {
    match metadata.get(key) {
        Some(MetaValue::U32(value)) => Ok(*value),
        _ => MetadataSnafu { key }.fail(),
    }
}

fn metadata_bool_or_false(
    metadata: &std::collections::HashMap<String, MetaValue>,
    key: &'static str,
) -> Result<bool> {
    match metadata.get(key) {
        None => Ok(false),
        Some(MetaValue::Bool(value)) => Ok(*value),
        _ => MetadataSnafu { key }.fail(),
    }
}

fn metadata_optional_u32(
    metadata: &std::collections::HashMap<String, MetaValue>,
    key: &'static str,
) -> Result<Option<u32>> {
    match metadata.get(key) {
        None => Ok(None),
        Some(MetaValue::U32(value)) => Ok(Some(*value)),
        _ => MetadataSnafu { key }.fail(),
    }
}

fn verify_vocabulary(
    metadata: &std::collections::HashMap<String, MetaValue>,
    tokenizer: &VerifiedTokenizer,
) -> Result<SpecialTokenPolicy> {
    let beginning_token = metadata_optional_u32(metadata, BOS_TOKEN_ID_KEY)?;
    let sequence_end = metadata_u32(metadata, EOS_TOKEN_ID_KEY)?;
    let prepend_beginning = metadata_bool_or_false(metadata, ADD_BOS_TOKEN_KEY)?;
    let append_sequence_end = metadata_bool_or_false(metadata, ADD_EOS_TOKEN_KEY)?;
    let turn_end = metadata_optional_u32(metadata, EOT_TOKEN_ID_KEY)?;
    let message_end = metadata_optional_u32(metadata, EOM_TOKEN_ID_KEY)?;
    let values = match metadata.get(TOKENS_KEY) {
        Some(MetaValue::Array(array)) => array.values(),
        _ => return MetadataSnafu { key: TOKENS_KEY }.fail(),
    };
    verify_exact_vocabulary(values, tokenizer)?;
    if prepend_beginning && beginning_token.is_none() {
        return SpecialTokenPolicySnafu {
            rule: "add_bos requires a declared BOS token ID",
        }
        .fail();
    }
    let mut stop_ids = vec![sequence_end];
    for identifier in [turn_end, message_end].into_iter().flatten() {
        if !stop_ids.contains(&identifier) {
            stop_ids.push(identifier);
        }
    }
    verify_declared_special_tokens(
        values,
        tokenizer,
        [beginning_token, Some(sequence_end), turn_end, message_end],
    )?;
    add_recognized_qwen_stops(tokenizer, values.len(), &mut stop_ids)?;
    Ok(SpecialTokenPolicy {
        bos_id: beginning_token,
        eos_id: sequence_end,
        add_bos: prepend_beginning,
        add_eos: append_sequence_end,
        stop_ids,
    })
}

fn verify_exact_vocabulary(values: &[MetaValue], tokenizer: &VerifiedTokenizer) -> Result<()> {
    for value in values {
        if !matches!(value, MetaValue::String(_)) {
            return MetadataSnafu { key: TOKENS_KEY }.fail();
        }
    }
    match tokenizer.verify_exact_vocabulary(
        values.len(),
        values.iter().filter_map(|value| match value {
            MetaValue::String(value) => Some(value.as_str()),
            _ => None,
        }),
    ) {
        Ok(()) => Ok(()),
        Err(tokenize::Error::VocabularyLengthMismatch { .. }) => InvalidConfigurationSnafu {
            rule: "artifact token table and tokenizer vocabulary length differ",
        }
        .fail(),
        Err(tokenize::Error::VocabularyMismatch { id, .. }) => {
            VocabularyMismatchSnafu { id }.fail()
        }
        Err(error) => Err(error).context(TokenizerSnafu),
    }
}

fn verify_declared_special_tokens(
    values: &[MetaValue],
    tokenizer: &VerifiedTokenizer,
    identifiers: [Option<u32>; 4],
) -> Result<()> {
    for identifier in identifiers.into_iter().flatten() {
        verify_special_token(values, tokenizer, identifier)?;
    }
    Ok(())
}

fn verify_special_token(
    values: &[MetaValue],
    tokenizer: &VerifiedTokenizer,
    identifier: u32,
) -> Result<()> {
    match tokenizer.verify_declared_special_id(values.len(), identifier) {
        Ok(()) => Ok(()),
        Err(
            tokenize::Error::SpecialTokenIdOutOfRange { .. }
            | tokenize::Error::SpecialTokenMissing { .. },
        ) => SpecialTokenPolicySnafu {
            rule: "declared special token ID is absent from exact vocabulary",
        }
        .fail(),
        Err(tokenize::Error::SpecialTokenNotMarked { .. }) => SpecialTokenPolicySnafu {
            rule: "declared special token is not marked special by tokenizer.json",
        }
        .fail(),
        Err(tokenize::Error::SpecialTokenEncodingMismatch { .. }) => SpecialTokenPolicySnafu {
            rule: "declared special token does not encode to exactly its artifact ID",
        }
        .fail(),
        Err(error) => Err(error).context(TokenizerSnafu),
    }
}

fn add_recognized_qwen_stops(
    tokenizer: &VerifiedTokenizer,
    vocabulary_size: usize,
    stop_ids: &mut Vec<u32>,
) -> Result<()> {
    for spelling in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(identifier) = tokenizer.tokenizer().token_to_id(spelling) {
            match tokenizer.verify_declared_special_id(vocabulary_size, identifier) {
                Ok(()) => {}
                Err(tokenize::Error::SpecialTokenNotMarked { .. }) => {
                    return SpecialTokenPolicySnafu {
                        rule: "a recognized Qwen EOG spelling must be tokenizer-special",
                    }
                    .fail();
                }
                Err(tokenize::Error::SpecialTokenEncodingMismatch { .. }) => {
                    return SpecialTokenPolicySnafu {
                        rule: "a recognized Qwen EOG spelling must encode to exactly its token ID",
                    }
                    .fail();
                }
                Err(error) => return Err(error).context(TokenizerSnafu),
            }
            if !stop_ids.contains(&identifier) {
                stop_ids.push(identifier);
            }
        }
    }
    Ok(())
}

fn greedy_last_logits(logits: &[f32], vocabulary: usize) -> Result<u32> {
    if logits.len() != vocabulary {
        return LogitShapeSnafu {
            actual: logits.len(),
            expected: vocabulary,
        }
        .fail();
    }
    let index = decode::greedy(logits).context(DecodeSnafu)?;
    if usize::try_from(index)
        .ok()
        .as_ref()
        .is_none_or(|index| *index >= vocabulary)
    {
        return LogitShapeSnafu {
            actual: usize::try_from(index).unwrap_or(usize::MAX),
            expected: vocabulary,
        }
        .fail();
    }
    Ok(index)
}

fn check_limit(field: &'static str, actual: usize, limit: usize) -> Result<()> {
    if actual > limit {
        return LimitExceededSnafu {
            field,
            actual,
            limit,
        }
        .fail();
    }
    Ok(())
}

fn check_cancelled(cancellation: &dyn Cancellation, boundary: &'static str) -> Result<()> {
    if cancellation.is_cancelled() {
        return CancelledSnafu { boundary }.fail();
    }
    Ok(())
}

#[cfg(test)]
const CRATE_NAME: &str = "text";

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::error::Error as StdError;
    use std::num::{NonZeroU64, NonZeroUsize};

    use loader::gguf::{ArtifactByteLimit, Sha256Digest};
    use sha2::{Digest, Sha256};
    use test_fixtures::{
        Qwen35FixtureConfig, RawGguf, RawMetadata, RawMetadataValue, SyntheticGguf,
        build_qwen35_fixture, raw_qwen35_fixture, serialize_raw_gguf,
    };
    use tokenize::TokenizerDigest;

    use super::*;

    const TOKENS: [&str; 5] = ["[UNK]", "<bos>", "<eos>", "hello", "assistant"];
    const QWEN_END_TOKENS: [&str; 9] = [
        "[UNK]",
        "<bos>",
        "<eos>",
        "hello",
        "assistant",
        "<eot>",
        "<eom>",
        "<|im_end|>",
        "<|endoftext|>",
    ];
    const BYTE_FALLBACK_TOKENS: [&str; 6] =
        ["[UNK]", "<bos>", "<eos>", "hello", "<0xC3>", "<0xA9>"];
    const STARTSWITH_TEMPLATE: &str =
        "{% if messages[0].content.startswith('h') %}hello{% endif %}";
    pub(super) type TestResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

    pub(super) fn test_limits(
        tokenizer_bytes: usize,
    ) -> std::result::Result<PipelineLimits, std::io::Error> {
        let tokenizer_bytes = NonZeroUsize::new(tokenizer_bytes).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "empty tokenizer fixture")
        })?;
        Ok(PipelineLimits {
            tokenizer_bytes: TokenizerByteLimit::new(tokenizer_bytes),
            template_bytes: 4_096,
            messages: 4,
            message_bytes: 128,
            prompt_bytes: 256,
            rendered_bytes: 256,
            context_tokens: 8,
            output_tokens: 2,
            output_bytes: 128,
            template_fuel: 10_000,
            template_recursion: 16,
        })
    }

    pub(super) fn tokenizer_json() -> String {
        r#"{
          "version":"1.0", "truncation":null, "padding":null,
          "added_tokens":[
            {"id":1,"content":"<bos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
            {"id":2,"content":"<eos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
          ],
          "normalizer":null, "pre_tokenizer":{"type":"Whitespace"},
          "post_processor":null, "decoder":null,
          "model":{"type":"WordLevel","vocab":{"[UNK]":0,"<bos>":1,"<eos>":2,"hello":3,"assistant":4},"unk_token":"[UNK]"}
        }"#.to_owned()
    }

    fn qwen_end_tokenizer_json() -> String {
        r#"{
          "version":"1.0", "truncation":null, "padding":null,
          "added_tokens":[
            {"id":1,"content":"<bos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
            {"id":2,"content":"<eos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
            {"id":5,"content":"<eot>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
            {"id":6,"content":"<eom>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
            {"id":7,"content":"<|im_end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
            {"id":8,"content":"<|endoftext|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
          ],
          "normalizer":null, "pre_tokenizer":{"type":"Whitespace"},
          "post_processor":null, "decoder":null,
          "model":{"type":"WordLevel","vocab":{"[UNK]":0,"<bos>":1,"<eos>":2,"hello":3,"assistant":4,"<eot>":5,"<eom>":6,"<|im_end|>":7,"<|endoftext|>":8},"unk_token":"[UNK]"}
        }"#.to_owned()
    }

    fn byte_fallback_tokenizer_json() -> String {
        r#"{
          "version":"1.0", "truncation":null, "padding":null,
          "added_tokens":[
            {"id":1,"content":"<bos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
            {"id":2,"content":"<eos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
          ],
          "normalizer":null, "pre_tokenizer":{"type":"Whitespace"},
          "post_processor":null, "decoder":{"type":"ByteFallback"},
          "model":{"type":"WordLevel","vocab":{"[UNK]":0,"<bos>":1,"<eos>":2,"hello":3,"<0xC3>":4,"<0xA9>":5},"unk_token":"[UNK]"}
        }"#.to_owned()
    }

    pub(super) fn fixture_config(
        tokens: &[&str],
        greedy_token_id: u32,
        prepend_beginning: bool,
        append_ending: bool,
        chat_template: &str,
    ) -> Qwen35FixtureConfig {
        Qwen35FixtureConfig {
            tokens: tokens.iter().map(|token| (*token).to_owned()).collect(),
            bos_token_id: 1,
            eos_token_id: 2,
            add_bos: prepend_beginning,
            add_eos: append_ending,
            chat_template: chat_template.to_owned(),
            greedy_token_id,
        }
    }

    pub(super) fn load_fixture(
        fixture: &SyntheticGguf,
    ) -> TestResult<(tempfile::TempDir, VerifiedArtifact)> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("synthetic.gguf");
        std::fs::write(&path, &fixture.bytes)?;
        let limit = NonZeroU64::new(fixture.byte_len)
            .ok_or_else(|| std::io::Error::other("synthetic fixture has zero byte length"))?;
        let artifact = VerifiedArtifact::load(
            &path,
            Sha256Digest::from_bytes(fixture.sha256),
            ArtifactByteLimit::new(limit),
        )?;
        Ok((directory, artifact))
    }

    pub(super) fn mutated_artifact(
        config: &Qwen35FixtureConfig,
        mutate: impl FnOnce(&mut RawGguf) -> TestResult<()>,
    ) -> TestResult<(tempfile::TempDir, VerifiedArtifact)> {
        let mut raw = raw_qwen35_fixture(config)?;
        mutate(&mut raw)?;
        let fixture = serialize_raw_gguf(&raw)?;
        load_fixture(&fixture)
    }

    fn remove_metadata(raw: &mut RawGguf, key: &str) -> TestResult<()> {
        let position = raw
            .metadata
            .iter()
            .position(|entry| entry.key == key)
            .ok_or_else(|| std::io::Error::other(format!("fixture metadata `{key}` is absent")))?;
        raw.metadata.remove(position);
        Ok(())
    }

    fn push_u32_metadata(raw: &mut RawGguf, key: &str, value: u32) {
        raw.metadata.push(RawMetadata {
            key: key.to_owned(),
            value: RawMetadataValue::U32(value),
        });
    }

    fn f32_tensor_mut<'raw>(raw: &'raw mut RawGguf, name: &str) -> TestResult<&'raw mut Vec<u8>> {
        let tensor = raw
            .tensors
            .iter_mut()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| std::io::Error::other(format!("fixture tensor `{name}` is absent")))?;
        if tensor.format != 0 {
            return Err(
                std::io::Error::other(format!("fixture tensor `{name}` is not F32")).into(),
            );
        }
        Ok(&mut tensor.payload)
    }

    pub(super) fn set_f32_row(
        raw: &mut RawGguf,
        name: &str,
        row: usize,
        values: &[f32],
    ) -> TestResult<()> {
        let tensor = raw
            .tensors
            .iter_mut()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| std::io::Error::other(format!("fixture tensor `{name}` is absent")))?;
        let width = usize::try_from(
            *tensor
                .dims
                .first()
                .ok_or_else(|| std::io::Error::other("fixture tensor has no row width"))?,
        )?;
        let rows = usize::try_from(
            *tensor
                .dims
                .get(1)
                .ok_or_else(|| std::io::Error::other("fixture tensor has no row count"))?,
        )?;
        if tensor.format != 0 || values.len() != width || row >= rows {
            return Err(std::io::Error::other(format!(
                "fixture tensor `{name}` cannot accept row {row} with width {}",
                values.len()
            ))
            .into());
        }
        let byte_start = row
            .checked_mul(width)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| std::io::Error::other("fixture row start overflowed"))?;
        let byte_len = width
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| std::io::Error::other("fixture row length overflowed"))?;
        let byte_end = byte_start
            .checked_add(byte_len)
            .ok_or_else(|| std::io::Error::other("fixture row end overflowed"))?;
        let row_bytes = tensor
            .payload
            .get_mut(byte_start..byte_end)
            .ok_or_else(|| std::io::Error::other("fixture tensor payload ended before row"))?;
        for (cell, value) in row_bytes.chunks_exact_mut(4).zip(values) {
            cell.copy_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }

    pub(super) fn verified_artifact(
        greedy_token_id: u32,
        prepend_beginning: bool,
        append_ending: bool,
    ) -> TestResult<(tempfile::TempDir, VerifiedArtifact)> {
        let config = fixture_config(
            &TOKENS,
            greedy_token_id,
            prepend_beginning,
            append_ending,
            STARTSWITH_TEMPLATE,
        );
        let fixture = build_qwen35_fixture(&config)?;
        load_fixture(&fixture)
    }

    fn pipeline_for(artifact: &VerifiedArtifact) -> TestResult<TextPipeline<'_>> {
        let tokenizer_json = tokenizer_json();
        pipeline_with_tokenizer(artifact, &tokenizer_json)
    }

    pub(super) fn pipeline_with_tokenizer<'artifact>(
        artifact: &'artifact VerifiedArtifact,
        tokenizer_json: &str,
    ) -> TestResult<TextPipeline<'artifact>> {
        let limits = test_limits(tokenizer_json.len())?;
        Ok(pipeline_result(artifact, tokenizer_json, limits)?)
    }

    pub(super) fn pipeline_result<'artifact>(
        artifact: &'artifact VerifiedArtifact,
        tokenizer_json: &str,
        limits: PipelineLimits,
    ) -> Result<TextPipeline<'artifact>> {
        let digest = TokenizerDigest::from_bytes(Sha256::digest(tokenizer_json.as_bytes()).into());
        TextPipeline::new(
            artifact,
            TokenizerCompanion::new(
                tokenizer_json.as_bytes(),
                TokenizerIdentity::new(tokenizer_json.len(), digest),
            ),
            limits,
        )
    }

    fn prompt_ids(pipeline: &TextPipeline<'_>, messages: &[TextMessage]) -> TestResult<Vec<u32>> {
        let request = GenerationRequest::new(messages, 1, false);
        pipeline.validate_request(&request)?;
        let rendered = pipeline.render(&request)?;
        Ok(pipeline.encode_prompt(&rendered)?)
    }

    fn text_error<T>(result: Result<T>) -> TestResult<Error> {
        match result {
            Err(error) => Ok(error),
            Ok(_) => Err(std::io::Error::other("text operation unexpectedly succeeded").into()),
        }
    }

    fn expect_limit<T>(
        result: Result<T>,
        expected_field: &'static str,
        expected_actual: usize,
        expected_limit: usize,
    ) -> TestResult<()> {
        match result {
            Err(Error::LimitExceeded {
                field,
                actual,
                limit,
                ..
            }) => {
                assert_eq!(field, expected_field, "wrong bounded field");
                assert_eq!(actual, expected_actual, "wrong observed bound value");
                assert_eq!(limit, expected_limit, "wrong configured bound value");
                Ok(())
            }
            Err(error) => Err(std::io::Error::other(format!(
                "expected limit error, received `{error}`"
            ))
            .into()),
            Ok(_) => Err(std::io::Error::other("bounded operation unexpectedly succeeded").into()),
        }
    }

    fn expect_invalid_rule<T>(result: Result<T>, expected: &'static str) -> TestResult<()> {
        match result {
            Err(Error::InvalidConfiguration { rule, .. }) => {
                assert_eq!(rule, expected, "wrong invalid-configuration rule");
                Ok(())
            }
            Err(error) => Err(std::io::Error::other(format!(
                "expected invalid configuration, received `{error}`"
            ))
            .into()),
            Ok(_) => Err(std::io::Error::other("invalid operation unexpectedly succeeded").into()),
        }
    }

    fn require_source<'error>(
        error: &'error Error,
    ) -> TestResult<&'error (dyn StdError + 'static)> {
        StdError::source(error)
            .ok_or_else(|| std::io::Error::other(format!("`{error}` has no error source")).into())
    }

    struct CancelOnCheck {
        target: usize,
        checks: Cell<usize>,
    }

    impl CancelOnCheck {
        fn new(target: usize) -> Self {
            Self {
                target,
                checks: Cell::new(0),
            }
        }

        fn checks(&self) -> usize {
            self.checks.get()
        }
    }

    impl Cancellation for CancelOnCheck {
        fn is_cancelled(&self) -> bool {
            let check = self.checks.get();
            self.checks.set(check + 1);
            check == self.target
        }
    }
    #[test]
    fn crate_identity_matches_role() {
        assert_eq!(env!("CARGO_PKG_NAME"), CRATE_NAME);
    }
    #[test]
    fn bos_and_eos_flags_define_exact_prompt_without_silent_deduplication() -> TestResult<()> {
        let tokenizer_json = tokenizer_json();
        let messages = [TextMessage::new(TextRole::User, "hello")];
        let cases: &[(bool, bool, &[u32])] = &[
            (false, false, &[3]),
            (true, false, &[1, 3]),
            (false, true, &[3, 2]),
            (true, true, &[1, 3, 2]),
        ];
        for &(prepend_beginning, append_ending, expected) in cases {
            let config = fixture_config(&TOKENS, 3, prepend_beginning, append_ending, "hello");
            let fixture = build_qwen35_fixture(&config)?;
            let (_directory, artifact) = load_fixture(&fixture)?;
            let pipeline = pipeline_with_tokenizer(&artifact, &tokenizer_json)?;
            assert_eq!(
                prompt_ids(&pipeline, &messages)?,
                expected,
                "prepend_beginning={prepend_beginning}, append_ending={append_ending} produced the wrong prompt"
            );
        }

        let duplicate_config = fixture_config(&TOKENS, 3, true, true, "<bos> hello <eos>");
        let fixture = build_qwen35_fixture(&duplicate_config)?;
        let (_directory, artifact) = load_fixture(&fixture)?;
        let pipeline = pipeline_with_tokenizer(&artifact, &tokenizer_json)?;
        assert_eq!(
            prompt_ids(&pipeline, &messages)?,
            [1, 1, 3, 2, 2],
            "artifact add flags must apply after encode(false), even when the template already emitted the IDs"
        );
        Ok(())
    }

    #[test]
    fn missing_bos_and_add_flags_follow_source_policy_while_eos_remains_required() -> TestResult<()>
    {
        let tokenizer_json = tokenizer_json();
        let messages = [TextMessage::new(TextRole::User, "hello")];
        let no_add_config = fixture_config(&TOKENS, 3, false, false, "hello");
        let (_directory, artifact) =
            mutated_artifact(&no_add_config, |raw| remove_metadata(raw, BOS_TOKEN_ID_KEY))?;
        let pipeline = pipeline_with_tokenizer(&artifact, &tokenizer_json)?;
        assert_eq!(
            prompt_ids(&pipeline, &messages)?,
            [3],
            "an unused absent BOS ID must not be invented or rejected"
        );

        let (_directory, artifact) = mutated_artifact(&no_add_config, |raw| {
            remove_metadata(raw, ADD_BOS_TOKEN_KEY)?;
            remove_metadata(raw, ADD_EOS_TOKEN_KEY)
        })?;
        let pipeline = pipeline_with_tokenizer(&artifact, &tokenizer_json)?;
        assert_eq!(
            prompt_ids(&pipeline, &messages)?,
            [3],
            "missing add flags must use the source false default"
        );

        let add_bos_config = fixture_config(&TOKENS, 3, true, false, "hello");
        let (_directory, artifact) = mutated_artifact(&add_bos_config, |raw| {
            remove_metadata(raw, BOS_TOKEN_ID_KEY)
        })?;
        let limits = test_limits(tokenizer_json.len())?;
        let error = text_error(pipeline_result(&artifact, &tokenizer_json, limits))?;
        assert!(
            matches!(
                error,
                Error::SpecialTokenPolicy {
                    rule: "add_bos requires a declared BOS token ID",
                    ..
                }
            ),
            "add_bos without a BOS ID must fail with its exact policy rule"
        );

        let (_directory, artifact) =
            mutated_artifact(&no_add_config, |raw| remove_metadata(raw, EOS_TOKEN_ID_KEY))?;
        let limits = test_limits(tokenizer_json.len())?;
        let error = text_error(pipeline_result(&artifact, &tokenizer_json, limits))?;
        assert!(
            matches!(
                error,
                Error::Metadata {
                    key: EOS_TOKEN_ID_KEY,
                    ..
                }
            ),
            "EOS metadata is required even when add_eos is false"
        );
        Ok(())
    }

    #[test]
    fn explicit_and_source_recognized_qwen_end_ids_stop_without_publication() -> TestResult<()> {
        let tokenizer_json = qwen_end_tokenizer_json();
        let messages = [TextMessage::new(TextRole::User, "hello")];
        let cases = [
            (5, Some(5), Some(6)),
            (6, Some(5), Some(6)),
            (1, Some(1), Some(1)),
            (7, None, None),
            (8, None, None),
        ];
        for (greedy_token_id, eot_id, eom_id) in cases {
            let config = fixture_config(&QWEN_END_TOKENS, greedy_token_id, false, false, "hello");
            let (_directory, artifact) = mutated_artifact(&config, |raw| {
                if let Some(id) = eot_id {
                    push_u32_metadata(raw, EOT_TOKEN_ID_KEY, id);
                }
                if let Some(id) = eom_id {
                    push_u32_metadata(raw, EOM_TOKEN_ID_KEY, id);
                }
                Ok(())
            })?;
            let pipeline = pipeline_with_tokenizer(&artifact, &tokenizer_json)?;
            let generation =
                pipeline.generate(GenerationRequest::new(&messages, 1, false), &NeverCancelled)?;
            assert_eq!(
                generation.finish_reason(),
                FinishReason::EndOfSequence,
                "candidate ID {greedy_token_id} was not treated as a checked Qwen stop ID"
            );
            assert!(
                generation.token_ids().is_empty(),
                "stop ID {greedy_token_id} must not enter retained output"
            );
            assert!(
                generation.text().is_empty(),
                "stop ID {greedy_token_id} must not enter decoded output"
            );
        }
        Ok(())
    }

    #[test]
    fn optional_and_recognized_end_ids_require_checked_ids_and_special_markers() -> TestResult<()> {
        let tokenizer_json = qwen_end_tokenizer_json();
        let config = fixture_config(&QWEN_END_TOKENS, 3, false, false, "hello");

        let (_directory, artifact) = mutated_artifact(&config, |raw| {
            push_u32_metadata(raw, EOT_TOKEN_ID_KEY, 99);
            Ok(())
        })?;
        let limits = test_limits(tokenizer_json.len())?;
        let error = text_error(pipeline_result(&artifact, &tokenizer_json, limits))?;
        assert!(
            matches!(
                error,
                Error::SpecialTokenPolicy {
                    rule: "declared special token ID is absent from exact vocabulary",
                    ..
                }
            ),
            "out-of-range EOT metadata must fail before generation"
        );

        let (_directory, artifact) = mutated_artifact(&config, |raw| {
            push_u32_metadata(raw, EOT_TOKEN_ID_KEY, 5);
            Ok(())
        })?;
        let ordinary_eot = tokenizer_json.replace(
            "\"id\":5,\"content\":\"<eot>\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":true",
            "\"id\":5,\"content\":\"<eot>\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":false",
        );
        let limits = test_limits(ordinary_eot.len())?;
        let error = text_error(pipeline_result(&artifact, &ordinary_eot, limits))?;
        assert!(
            matches!(
                error,
                Error::SpecialTokenPolicy {
                    rule: "declared special token is not marked special by tokenizer.json",
                    ..
                }
            ),
            "explicit EOT must carry the tokenizer special marker"
        );

        let fixture = build_qwen35_fixture(&config)?;
        let (_directory, artifact) = load_fixture(&fixture)?;
        let ordinary_im_end = tokenizer_json.replace(
            "\"id\":7,\"content\":\"<|im_end|>\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":true",
            "\"id\":7,\"content\":\"<|im_end|>\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":false",
        );
        let limits = test_limits(ordinary_im_end.len())?;
        let error = text_error(pipeline_result(&artifact, &ordinary_im_end, limits))?;
        assert!(
            matches!(
                error,
                Error::SpecialTokenPolicy {
                    rule: "a recognized Qwen EOG spelling must be tokenizer-special",
                    ..
                }
            ),
            "source-recognized Qwen endings must carry the tokenizer special marker"
        );
        Ok(())
    }

    #[test]
    fn tokenizer_template_and_message_input_caps_report_exact_dimensions() -> TestResult<()> {
        let tokenizer_json = tokenizer_json();
        let config = fixture_config(&TOKENS, 3, false, false, STARTSWITH_TEMPLATE);
        let fixture = build_qwen35_fixture(&config)?;
        let (_directory, artifact) = load_fixture(&fixture)?;

        let mut tokenizer_limits = test_limits(tokenizer_json.len())?;
        tokenizer_limits.tokenizer_bytes = TokenizerByteLimit::try_new(tokenizer_json.len() - 1)?;
        let error = text_error(pipeline_result(
            &artifact,
            &tokenizer_json,
            tokenizer_limits,
        ))?;
        match error {
            Error::Tokenizer {
                source: tokenize::Error::ByteLimitExceeded { limit, actual, .. },
                ..
            } => {
                assert_eq!(actual, tokenizer_json.len(), "wrong tokenizer byte count");
                assert_eq!(limit, tokenizer_json.len() - 1, "wrong tokenizer cap");
            }
            error => {
                return Err(std::io::Error::other(format!(
                    "expected tokenizer byte cap, received `{error}`"
                ))
                .into());
            }
        }

        let mut template_limits = test_limits(tokenizer_json.len())?;
        template_limits.template_bytes = STARTSWITH_TEMPLATE.len() - 1;
        expect_limit(
            pipeline_result(&artifact, &tokenizer_json, template_limits),
            "template bytes",
            STARTSWITH_TEMPLATE.len(),
            STARTSWITH_TEMPLATE.len() - 1,
        )?;

        let mut message_count_limits = test_limits(tokenizer_json.len())?;
        message_count_limits.messages = 1;
        let pipeline = pipeline_result(&artifact, &tokenizer_json, message_count_limits)?;
        let two_messages = [
            TextMessage::new(TextRole::User, "hello"),
            TextMessage::new(TextRole::Assistant, "hello"),
        ];
        expect_limit(
            pipeline.generate(
                GenerationRequest::new(&two_messages, 1, false),
                &NeverCancelled,
            ),
            "messages",
            2,
            1,
        )?;

        let mut message_byte_limits = test_limits(tokenizer_json.len())?;
        message_byte_limits.message_bytes = 5;
        let pipeline = pipeline_result(&artifact, &tokenizer_json, message_byte_limits)?;
        let multibyte_message = [TextMessage::new(TextRole::User, "ééé")];
        expect_limit(
            pipeline.generate(
                GenerationRequest::new(&multibyte_message, 1, false),
                &NeverCancelled,
            ),
            "message bytes",
            6,
            5,
        )?;

        let mut prompt_byte_limits = test_limits(tokenizer_json.len())?;
        prompt_byte_limits.prompt_bytes = 5;
        let pipeline = pipeline_result(&artifact, &tokenizer_json, prompt_byte_limits)?;
        let six_prompt_bytes = [
            TextMessage::new(TextRole::User, "hey"),
            TextMessage::new(TextRole::Assistant, "yes"),
        ];
        expect_limit(
            pipeline.generate(
                GenerationRequest::new(&six_prompt_bytes, 1, false),
                &NeverCancelled,
            ),
            "request prompt bytes",
            6,
            5,
        )?;
        Ok(())
    }

    #[test]
    fn configured_tokenizer_padding_and_truncation_are_refused_in_native_setup() -> TestResult<()> {
        let config = fixture_config(&TOKENS, 3, false, false, "hello");
        let fixture = build_qwen35_fixture(&config)?;
        let (_directory, artifact) = load_fixture(&fixture)?;
        let ordinary = tokenizer_json();
        let cases = [
            (
                ordinary.replace(
                    "\"truncation\":null",
                    "\"truncation\":{\"direction\":\"Right\",\"max_length\":1,\"strategy\":\"LongestFirst\",\"stride\":0}",
                ),
                "truncation",
            ),
            (
                ordinary.replace(
                    "\"padding\":null",
                    "\"padding\":{\"strategy\":{\"Fixed\":4},\"direction\":\"Right\",\"pad_to_multiple_of\":null,\"pad_id\":0,\"pad_type_id\":0,\"pad_token\":\"[UNK]\"}",
                ),
                "padding",
            ),
        ];
        for (configured, setting) in cases {
            let error = text_error(pipeline_result(
                &artifact,
                &configured,
                test_limits(configured.len())?,
            ))?;
            match (setting, error) {
                (
                    "truncation",
                    Error::Tokenizer {
                        source: tokenize::Error::ConfiguredTruncation { .. },
                        ..
                    },
                )
                | (
                    "padding",
                    Error::Tokenizer {
                        source: tokenize::Error::ConfiguredPadding { .. },
                        ..
                    },
                ) => {}
                (_, error) => {
                    return Err(std::io::Error::other(format!(
                        "native text setup accepted or misreported configured tokenizer {setting}: {error}"
                    ))
                    .into());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn context_output_token_and_decoded_byte_caps_report_exact_dimensions() -> TestResult<()> {
        let tokenizer_json = tokenizer_json();
        let messages = [TextMessage::new(TextRole::User, "hello")];
        let (_directory, artifact) = verified_artifact(3, true, false)?;

        let mut context_limits = test_limits(tokenizer_json.len())?;
        context_limits.context_tokens = 2;
        let pipeline = pipeline_result(&artifact, &tokenizer_json, context_limits)?;
        expect_limit(
            pipeline.generate(GenerationRequest::new(&messages, 1, false), &NeverCancelled),
            "prompt plus output tokens",
            3,
            2,
        )?;

        let mut output_token_limits = test_limits(tokenizer_json.len())?;
        output_token_limits.output_tokens = 1;
        let pipeline = pipeline_result(&artifact, &tokenizer_json, output_token_limits)?;
        expect_limit(
            pipeline.generate(GenerationRequest::new(&messages, 2, false), &NeverCancelled),
            "requested output tokens",
            2,
            1,
        )?;

        let mut output_byte_limits = test_limits(tokenizer_json.len())?;
        output_byte_limits.output_bytes = 4;
        let pipeline = pipeline_result(&artifact, &tokenizer_json, output_byte_limits)?;
        expect_limit(
            pipeline.generate(GenerationRequest::new(&messages, 1, false), &NeverCancelled),
            "decoded output bytes",
            5,
            4,
        )?;
        Ok(())
    }

    #[test]
    fn malformed_request_shapes_and_empty_rendered_prompts_are_refused() -> TestResult<()> {
        let (_directory, artifact) = verified_artifact(3, false, false)?;
        let pipeline = pipeline_for(&artifact)?;
        let empty_messages: [TextMessage; 0] = [];
        expect_invalid_rule(
            pipeline.generate(
                GenerationRequest::new(&empty_messages, 1, false),
                &NeverCancelled,
            ),
            "generation request must contain at least one text message",
        )?;

        let no_user = [
            TextMessage::new(TextRole::System, "hello"),
            TextMessage::new(TextRole::Assistant, "hello"),
        ];
        expect_invalid_rule(
            pipeline.generate(GenerationRequest::new(&no_user, 1, false), &NeverCancelled),
            "generation request must contain at least one user text message",
        )?;

        let noninitial_system = [
            TextMessage::new(TextRole::User, "hello"),
            TextMessage::new(TextRole::System, "hello"),
        ];
        expect_invalid_rule(
            pipeline.generate(
                GenerationRequest::new(&noninitial_system, 1, false),
                &NeverCancelled,
            ),
            "a system text message is supported only at the start of a request",
        )?;

        let messages = [TextMessage::new(TextRole::User, "hello")];
        expect_invalid_rule(
            pipeline.generate(GenerationRequest::new(&messages, 0, false), &NeverCancelled),
            "generation request output-token limit must be non-zero",
        )?;

        let empty_render = [TextMessage::new(TextRole::User, "bye")];
        let error = text_error(pipeline.generate(
            GenerationRequest::new(&empty_render, 1, false),
            &NeverCancelled,
        ))?;
        assert!(
            matches!(error, Error::EmptyPrompt { .. }),
            "a template that encodes no IDs must fail as EmptyPrompt"
        );
        Ok(())
    }

    #[test]
    fn generated_byte_fallback_tokens_are_decoded_as_one_collective_sequence() -> TestResult<()> {
        let tokenizer_json = byte_fallback_tokenizer_json();
        let config = fixture_config(&BYTE_FALLBACK_TOKENS, 4, false, false, "hello");
        let (_directory, artifact) = mutated_artifact(&config, |raw| {
            f32_tensor_mut(raw, "output.weight")?.fill(0);
            set_f32_row(raw, "token_embd.weight", 3, &[1.0, 0.0, 0.0])?;
            set_f32_row(raw, "token_embd.weight", 4, &[0.0, 1.0, 0.0])?;
            set_f32_row(raw, "output.weight", 4, &[1.0, 0.0, 0.0])?;
            set_f32_row(raw, "output.weight", 5, &[0.0, 1.0, 0.0])
        })?;
        let pipeline = pipeline_with_tokenizer(&artifact, &tokenizer_json)?;
        let tokenizer = pipeline.tokenizer.tokenizer();
        let first_piece = tokenizer.decode(&[4], false)?;
        let second_piece = tokenizer.decode(&[5], false)?;
        assert_eq!(first_piece, "�", "the first byte is not standalone UTF-8");
        assert_eq!(second_piece, "�", "the second byte is not standalone UTF-8");
        assert_eq!(
            format!("{first_piece}{second_piece}"),
            "��",
            "the fixture must distinguish per-token concatenation"
        );

        let messages = [TextMessage::new(TextRole::User, "hello")];
        let generation =
            pipeline.generate(GenerationRequest::new(&messages, 2, false), &NeverCancelled)?;
        assert_eq!(
            generation.token_ids(),
            [4, 5],
            "the synthetic decoder must emit both UTF-8 byte tokens in order"
        );
        assert_eq!(
            generation.text(),
            "é",
            "the pipeline must decode the complete generated token sequence collectively"
        );
        assert_eq!(
            generation.finish_reason(),
            FinishReason::Length,
            "the two-token request should finish at its exact length cap"
        );
        Ok(())
    }

    #[test]
    fn template_and_decode_failures_retain_typed_sources() -> TestResult<()> {
        let tokenizer_json = tokenizer_json();
        let config = fixture_config(&TOKENS, 3, false, false, "{% if");
        let fixture = build_qwen35_fixture(&config)?;
        let (_directory, artifact) = load_fixture(&fixture)?;
        let template_error = text_error(pipeline_result(
            &artifact,
            &tokenizer_json,
            test_limits(tokenizer_json.len())?,
        ))?;
        assert!(
            require_source(&template_error)?.is::<minijinja::Error>(),
            "template wrapper must retain minijinja::Error"
        );

        let decode_error = text_error(greedy_last_logits(&[f32::NAN], 1))?;
        assert!(
            require_source(&decode_error)?.is::<decode::Error>(),
            "greedy wrapper must retain decode::Error"
        );

        Ok(())
    }

    #[test]
    fn tokenizer_failures_retain_typed_sources() -> TestResult<()> {
        let config = fixture_config(&TOKENS, 3, false, false, "hello");
        let fixture = build_qwen35_fixture(&config)?;
        let (_directory, artifact) = load_fixture(&fixture)?;

        let invalid_tokenizer = "{";
        let limits = test_limits(invalid_tokenizer.len())?;
        let tokenizer_error = text_error(pipeline_result(&artifact, invalid_tokenizer, limits))?;
        assert!(
            require_source(&tokenizer_error)?.is::<tokenize::Error>(),
            "tokenizer wrapper must retain tokenize::Error"
        );

        Ok(())
    }

    #[test]
    fn verified_gguf_template_tokenizer_and_greedy_generation_are_bound() -> TestResult<()> {
        let (_directory, artifact) = verified_artifact(3, true, false)?;
        let pipeline = pipeline_for(&artifact)?;
        let messages = [TextMessage::new(TextRole::User, "hello")];
        let generation =
            pipeline.generate(GenerationRequest::new(&messages, 1, false), &NeverCancelled)?;
        assert_eq!(generation.text(), "hello");
        assert_eq!(generation.token_ids(), &[3]);
        assert_eq!(generation.finish_reason(), FinishReason::Length);
        Ok(())
    }

    #[test]
    fn prepared_generation_binds_direct_generation_to_the_same_prompt_and_result() -> TestResult<()>
    {
        let (_directory, artifact) = verified_artifact(3, true, false)?;
        let prepared_pipeline = pipeline_for(&artifact)?;
        let direct_pipeline = pipeline_for(&artifact)?;
        let messages = [TextMessage::new(TextRole::User, "hello")];
        let request = GenerationRequest::new(&messages, 1, false);
        let prepared = prepared_pipeline.prepare(request, &NeverCancelled)?;
        assert_eq!(
            prepared.rendered_prompt(),
            "hello",
            "preparation must expose the artifact-rendered prompt"
        );
        assert_eq!(
            prepared.prompt_token_ids(),
            [1, 3],
            "preparation must expose the exact special-token-aware prompt IDs"
        );
        assert_eq!(
            prepared.tokenizer_identity(),
            direct_pipeline.tokenizer.identity(),
            "prepared requests must retain their verified tokenizer identity"
        );
        assert_eq!(
            prepared.max_output_tokens(),
            1,
            "preparation must retain the checked request output cap"
        );
        let prepared_generation = prepared.generate(&NeverCancelled)?;
        let direct_generation = direct_pipeline.generate(request, &NeverCancelled)?;
        assert_eq!(
            prepared_generation, direct_generation,
            "direct generation must consume the same preparation path"
        );
        Ok(())
    }

    #[test]
    fn prepared_generation_uses_request_specific_decoder_bounds() -> TestResult<()> {
        let (_directory, artifact) = verified_artifact(3, true, false)?;
        let pipeline = pipeline_for(&artifact)?;
        let messages = [TextMessage::new(TextRole::User, "hello")];
        let prepared =
            pipeline.prepare(GenerationRequest::new(&messages, 1, false), &NeverCancelled)?;
        let requirements = prepared.decoder_cpu_requirements();
        assert_eq!(
            requirements.max_context(),
            3,
            "request context must be prompt plus requested output, not the ceiling"
        );
        assert_eq!(
            requirements.max_step_tokens(),
            2,
            "the initial last-token decoder step must use the prompt width"
        );
        assert!(
            requirements.max_context() < pipeline.limits.context_tokens,
            "the prepared context should remain narrower than the configured ceiling"
        );
        assert!(
            requirements.max_step_tokens() < pipeline.limits.context_tokens,
            "the prepared step should remain narrower than the configured ceiling"
        );
        Ok(())
    }

    #[test]
    fn prepared_generation_checks_cancellation_before_decoder_session_construction()
    -> TestResult<()> {
        let (_directory, artifact) = verified_artifact(3, true, false)?;
        let pipeline = pipeline_for(&artifact)?;
        let messages = [TextMessage::new(TextRole::User, "hello")];
        let prepared =
            pipeline.prepare(GenerationRequest::new(&messages, 1, false), &NeverCancelled)?;
        let cancelled = CancelOnCheck::new(0);
        let error = text_error(prepared.generate(&cancelled))?;
        assert!(
            matches!(
                error,
                Error::Cancelled {
                    boundary: "decoder session construction",
                    ..
                }
            ),
            "prepared generation must check cancellation before session allocation"
        );
        assert_eq!(
            cancelled.checks(),
            1,
            "the new boundary must be the first observation after preparation"
        );
        Ok(())
    }

    #[test]
    fn eos_stops_before_collective_decoding() -> TestResult<()> {
        let (_directory, artifact) = verified_artifact(2, false, false)?;
        let pipeline = pipeline_for(&artifact)?;
        let messages = [TextMessage::new(TextRole::User, "hello")];
        let generation =
            pipeline.generate(GenerationRequest::new(&messages, 2, false), &NeverCancelled)?;
        assert_eq!(generation.finish_reason(), FinishReason::EndOfSequence);
        assert!(generation.token_ids().is_empty());
        assert!(generation.text().is_empty());
        Ok(())
    }

    #[test]
    fn tokenizer_id_or_special_marker_mismatch_is_refused() -> TestResult<()> {
        let (_directory, artifact) = verified_artifact(3, false, false)?;
        let swapped_ids =
            tokenizer_json().replace("\"hello\":3,\"assistant\":4", "\"hello\":4,\"assistant\":3");
        assert!(
            matches!(pipeline_with_tokenizer(&artifact, &swapped_ids), Err(error) if matches!(error.downcast_ref::<Error>(), Some(Error::VocabularyMismatch { .. })))
        );
        let ordinary_eos = tokenizer_json().replace(
            "\"id\":2,\"content\":\"<eos>\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":true",
            "\"id\":2,\"content\":\"<eos>\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":false",
        );
        assert!(
            matches!(pipeline_with_tokenizer(&artifact, &ordinary_eos), Err(error) if matches!(error.downcast_ref::<Error>(), Some(Error::SpecialTokenPolicy { .. })))
        );
        Ok(())
    }

    #[test]
    fn cancellation_after_prompt_step_returns_no_response_or_shared_state() -> TestResult<()> {
        let (_directory, artifact) = verified_artifact(3, true, false)?;
        let pipeline = pipeline_for(&artifact)?;
        let pristine = pipeline_for(&artifact)?;
        let messages = [TextMessage::new(TextRole::User, "hello")];
        let expected =
            pristine.generate(GenerationRequest::new(&messages, 1, false), &NeverCancelled)?;
        let cancelled = CancelOnCheck::new(4);
        let error =
            text_error(pipeline.generate(GenerationRequest::new(&messages, 1, false), &cancelled))?;
        assert!(
            matches!(
                error,
                Error::Cancelled {
                    boundary: "greedy selection",
                    ..
                }
            ),
            "check 4 must observe cancellation after the complete prompt decoder step"
        );
        assert_eq!(
            cancelled.checks(),
            5,
            "the session-construction checkpoint must precede the prompt decoder step"
        );
        let retry =
            pipeline.generate(GenerationRequest::new(&messages, 1, false), &NeverCancelled)?;
        assert_eq!(
            retry, expected,
            "a fresh request after prompt-step cancellation must equal a pristine execution"
        );
        Ok(())
    }

    #[test]
    fn already_cancelled_request_stops_before_validation_or_rendering() -> TestResult<()> {
        let (_directory, artifact) = verified_artifact(3, true, false)?;
        let pipeline = pipeline_for(&artifact)?;
        let messages = [TextMessage::new(TextRole::User, "hello")];
        let cancelled = CancelOnCheck::new(0);
        let error =
            text_error(pipeline.generate(GenerationRequest::new(&messages, 1, false), &cancelled))?;
        assert!(
            matches!(
                error,
                Error::Cancelled {
                    boundary: "template rendering",
                    ..
                }
            ),
            "an already-cancelled request must stop at the first boundary"
        );
        assert_eq!(
            cancelled.checks(),
            1,
            "already-cancelled work must make exactly one observation"
        );
        Ok(())
    }

    #[test]
    fn cancellation_after_collective_decode_prevents_publish_and_retry_is_pristine()
    -> TestResult<()> {
        let (_directory, artifact) = verified_artifact(3, true, false)?;
        let pipeline = pipeline_for(&artifact)?;
        let pristine = pipeline_for(&artifact)?;
        let messages = [TextMessage::new(TextRole::User, "hello")];
        let expected =
            pristine.generate(GenerationRequest::new(&messages, 1, false), &NeverCancelled)?;
        let cancelled = CancelOnCheck::new(6);
        let error =
            text_error(pipeline.generate(GenerationRequest::new(&messages, 1, false), &cancelled))?;
        assert!(
            matches!(
                error,
                Error::Cancelled {
                    boundary: "publishing completed response",
                    ..
                }
            ),
            "check 5 must cancel only after collective decoding completes"
        );
        assert_eq!(
            cancelled.checks(),
            7,
            "publication cancellation must follow the collective-decode check"
        );
        let retry =
            pipeline.generate(GenerationRequest::new(&messages, 1, false), &NeverCancelled)?;
        assert_eq!(
            retry, expected,
            "a fresh request after publication cancellation must equal pristine execution"
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "prepared_tests.rs"]
mod prepared_tests;
