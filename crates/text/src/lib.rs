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

use std::io::{self, Write};

use decoders::{Qwen35LogitSelection, Qwen35Weights};
use loader::gguf::{MetaValue, VerifiedArtifact};
use minijinja::{Environment, UndefinedBehavior, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use serde::Serialize;
use tokenize::{TokenizerByteLimit, TokenizerIdentity, VerifiedTokenizer};

use crate::error::{
    AllocationSnafu, CancelledSnafu, DecoderSnafu, EmptyPromptSnafu, InvalidConfigurationSnafu,
    LimitExceededSnafu, MetadataSnafu, SpecialTokenPolicySnafu, TemplateSnafu, TokenizerSnafu,
    VocabularyMismatchSnafu,
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
    fn validate(self) -> Result<()> {
        if [
            self.template_bytes,
            self.messages,
            self.message_bytes,
            self.prompt_bytes,
            self.rendered_bytes,
            self.context_tokens,
            self.output_tokens,
            self.output_bytes,
            self.template_recursion,
        ]
        .contains(&0)
            || self.template_fuel == 0
            || isize::try_from(self.template_fuel).is_err()
        {
            return InvalidConfigurationSnafu {
                rule: "static limits must be non-zero and template fuel must fit isize",
            }
            .fail();
        }
        Ok(())
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
#[derive(Debug)]
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
    bos_id: u32,
    eos_id: u32,
    add_bos: bool,
    add_eos: bool,
    stop_ids: Vec<u32>,
}

/// Artifact-bound text pipeline.
pub struct TextPipeline<'artifact> {
    weights: Qwen35Weights<'artifact>,
    tokenizer: VerifiedTokenizer,
    environment: Environment<'artifact>,
    template: &'artifact str,
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
        limits.validate()?;
        let tokenizer = VerifiedTokenizer::from_bytes(
            companion.bytes,
            companion.identity,
            limits.tokenizer_bytes,
        )
        .map_err(|error| {
            TokenizerSnafu {
                message: error.to_string(),
            }
            .build()
        })?;
        let metadata = artifact.observation().metadata();
        let template = metadata_string(metadata, CHAT_TEMPLATE_KEY)?;
        check_limit("template bytes", template.len(), limits.template_bytes)?;
        let special_tokens = verify_vocabulary(
            metadata,
            tokenizer.tokenizer(),
            metadata_u32(metadata, BOS_TOKEN_ID_KEY)?,
            metadata_u32(metadata, EOS_TOKEN_ID_KEY)?,
            metadata_bool_or_false(metadata, ADD_BOS_TOKEN_KEY)?,
            metadata_bool_or_false(metadata, ADD_EOS_TOKEN_KEY)?,
            metadata_optional_u32(metadata, EOT_TOKEN_ID_KEY)?,
            metadata_optional_u32(metadata, EOM_TOKEN_ID_KEY)?,
        )?;
        let environment = compile_template(template, limits)?;
        let weights = Qwen35Weights::try_from_verified(artifact).map_err(|error| {
            DecoderSnafu {
                message: error.to_string(),
            }
            .build()
        })?;
        Ok(Self {
            weights,
            tokenizer,
            environment,
            template,
            special_tokens,
            limits,
        })
    }

    /// Generate greedily in one fresh private decoder session.
    ///
    /// A cancellation or later failure returns no partial text. Any completed
    /// decoder `step` remains committed only inside the private session that is
    /// immediately dropped; this method does not offer decoder rollback.
    pub fn generate(
        &self,
        request: GenerationRequest<'_>,
        cancellation: &dyn Cancellation,
    ) -> Result<Generation> {
        check_cancelled(cancellation, "template rendering")?;
        self.validate_request(&request)?;
        let rendered = self.render(&request)?;
        check_cancelled(cancellation, "tokenization")?;
        let mut prompt = self
            .tokenizer
            .tokenizer()
            .encode(&rendered, false)
            .map_err(|error| {
                TokenizerSnafu {
                    message: error.to_string(),
                }
                .build()
            })?;
        prompt.try_reserve(2).map_err(|_| {
            AllocationSnafu {
                target: "prompt special-token prefix/suffix",
            }
            .build()
        })?;
        if self.special_tokens.add_bos {
            prompt.insert(0, self.special_tokens.bos_id);
        }
        if self.special_tokens.add_eos {
            prompt.push(self.special_tokens.eos_id);
        }
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
        let vocabulary = self.tokenizer.tokenizer().vocab_size();
        let plan = self
            .weights
            .execution_plan(
                self.limits.context_tokens,
                self.limits.context_tokens,
                Qwen35LogitSelection::LastToken,
            )
            .map_err(|error| {
                DecoderSnafu {
                    message: error.to_string(),
                }
                .build()
            })?;
        let mut execution = plan.execution().map_err(|error| {
            DecoderSnafu {
                message: error.to_string(),
            }
            .build()
        })?;
        check_cancelled(cancellation, "prompt decoder step")?;
        let mut logits = execution.step(&prompt).map_err(|error| {
            DecoderSnafu {
                message: error.to_string(),
            }
            .build()
        })?;
        let mut generated = Vec::new();
        generated
            .try_reserve_exact(request.max_output_tokens)
            .map_err(|_| {
                AllocationSnafu {
                    target: "generated token IDs",
                }
                .build()
            })?;
        let finish_reason = loop {
            check_cancelled(cancellation, "greedy selection")?;
            let next = greedy_last_logits(&logits, vocabulary)?;
            if self.special_tokens.stop_ids.contains(&next) {
                break FinishReason::EndOfSequence;
            }
            generated.push(next);
            if generated.len() == request.max_output_tokens {
                break FinishReason::Length;
            }
            check_cancelled(cancellation, "next decoder step")?;
            logits = execution.step(&[next]).map_err(|error| {
                DecoderSnafu {
                    message: error.to_string(),
                }
                .build()
            })?;
        };
        check_cancelled(cancellation, "collective output decoding")?;
        let text = self
            .tokenizer
            .tokenizer()
            .decode(&generated, false)
            .map_err(|error| {
                TokenizerSnafu {
                    message: error.to_string(),
                }
                .build()
            })?;
        check_limit("decoded output bytes", text.len(), self.limits.output_bytes)?;
        Ok(Generation {
            text,
            token_ids: generated,
            finish_reason,
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
        let template = self
            .environment
            .template_from_str(&self.template)
            .map_err(|error| {
                TemplateSnafu {
                    message: error.to_string(),
                }
                .build()
            })?;
        let mut output = ByteCappedWriter::new(self.limits.rendered_bytes);
        let result = template.render_captured_to(
            context!(messages => request.messages, add_generation_prompt => true, enable_thinking => request.enable_thinking, tools => Vec::<()>::new()),
            &mut output,
        );
        if output.exceeded {
            return LimitExceededSnafu {
                field: "rendered template bytes",
                actual: output.attempted,
                limit: self.limits.rendered_bytes,
            }
            .fail();
        }
        result.map_err(|error| {
            TemplateSnafu {
                message: error.to_string(),
            }
            .build()
        })?;
        output.into_string()
    }
}

fn compile_template<'template>(
    template: &'template str,
    limits: PipelineLimits,
) -> Result<Environment<'template>> {
    let mut environment = Environment::new();
    environment.set_undefined_behavior(UndefinedBehavior::Strict);
    environment.set_unknown_method_callback(unknown_method_callback);
    environment.set_fuel(Some(limits.template_fuel));
    environment.set_recursion_limit(limits.template_recursion);
    if environment.recursion_limit() != limits.template_recursion {
        return InvalidConfigurationSnafu {
            rule: "requested template recursion limit is not supported by this runtime",
        }
        .fail();
    }
    environment.template_from_str(template).map_err(|error| {
        TemplateSnafu {
            message: error.to_string(),
        }
        .build()
    })?;
    Ok(environment)
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
    tokenizer: &tokenize::Tokenizer,
    bos_id: u32,
    eos_id: u32,
    add_bos: bool,
    add_eos: bool,
    eot_id: Option<u32>,
    eom_id: Option<u32>,
) -> Result<SpecialTokenPolicy> {
    let values = match metadata.get(TOKENS_KEY) {
        Some(MetaValue::Array(array)) => array.values(),
        _ => return MetadataSnafu { key: TOKENS_KEY }.fail(),
    };
    if values.len() != tokenizer.vocab_size() {
        return InvalidConfigurationSnafu {
            rule: "artifact token table and tokenizer vocabulary length differ",
        }
        .fail();
    }
    for (index, value) in values.iter().enumerate() {
        let id = u32::try_from(index).map_err(|_| {
            InvalidConfigurationSnafu {
                rule: "artifact token ID exceeds u32",
            }
            .build()
        })?;
        let MetaValue::String(expected) = value else {
            return MetadataSnafu { key: TOKENS_KEY }.fail();
        };
        if tokenizer.id_to_token(id).as_deref() != Some(expected)
            || tokenizer.token_to_id(expected) != Some(id)
        {
            return VocabularyMismatchSnafu { id }.fail();
        }
    }
    let mut stop_ids = vec![eos_id];
    for id in [bos_id, eos_id].into_iter().chain(eot_id).chain(eom_id) {
        if values
            .get(usize::try_from(id).map_err(|_| {
                SpecialTokenPolicySnafu {
                    rule: "special token ID does not fit usize",
                }
                .build()
            })?)
            .is_none()
            || tokenizer.id_to_token(id).is_none()
        {
            return SpecialTokenPolicySnafu {
                rule: "declared special token ID is absent from exact vocabulary",
            }
            .fail();
        }
        if id != bos_id && !stop_ids.contains(&id) {
            stop_ids.push(id);
        }
        let token = tokenizer.id_to_token(id).ok_or_else(|| {
            SpecialTokenPolicySnafu {
                rule: "declared special token has no tokenizer string",
            }
            .build()
        })?;
        let encoded = tokenizer.encode(&token, false).map_err(|error| {
            TokenizerSnafu {
                message: error.to_string(),
            }
            .build()
        })?;
        if encoded.as_slice() != [id] {
            return SpecialTokenPolicySnafu {
                rule: "declared special token does not encode to exactly its artifact ID",
            }
            .fail();
        }
    }
    for spelling in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(id) = tokenizer.token_to_id(spelling) {
            if !tokenizer.is_special_token(id) {
                return SpecialTokenPolicySnafu {
                    rule: "a recognized Qwen EOG spelling must be tokenizer-special",
                }
                .fail();
            }
            let encoded = tokenizer.encode(spelling, false).map_err(|error| {
                TokenizerSnafu {
                    message: error.to_string(),
                }
                .build()
            })?;
            if encoded.as_slice() != [id] {
                return SpecialTokenPolicySnafu {
                    rule: "a recognized Qwen EOG spelling must encode to exactly its token ID",
                }
                .fail();
            }
            if !stop_ids.contains(&id) {
                stop_ids.push(id);
            }
        }
    }
    Ok(SpecialTokenPolicy {
        bos_id,
        eos_id,
        add_bos,
        add_eos,
        stop_ids,
    })
}

fn greedy_last_logits(logits: &[f32], vocabulary: usize) -> Result<u32> {
    if logits.len() != vocabulary {
        return DecoderSnafu {
            message: "decoder returned logits with an unexpected shape".to_owned(),
        }
        .fail();
    }
    let index = decode::greedy(logits).map_err(|error| {
        DecoderSnafu {
            message: error.to_string(),
        }
        .build()
    })?;
    if usize::try_from(index)
        .ok()
        .filter(|index| *index < vocabulary)
        .is_none()
    {
        return DecoderSnafu {
            message: "greedy sampler selected an out-of-range token ID".to_owned(),
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

struct ByteCappedWriter {
    output: Vec<u8>,
    maximum: usize,
    attempted: usize,
    exceeded: bool,
}

impl ByteCappedWriter {
    fn new(maximum: usize) -> Self {
        Self {
            output: Vec::new(),
            maximum,
            attempted: 0,
            exceeded: false,
        }
    }
    fn into_string(self) -> Result<String> {
        String::from_utf8(self.output).map_err(|error| {
            TemplateSnafu {
                message: error.to_string(),
            }
            .build()
        })
    }
}

impl Write for ByteCappedWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let attempted = self.output.len().checked_add(buffer.len());
        self.attempted = attempted.unwrap_or(usize::MAX);
        if attempted.is_none_or(|value| value > self.maximum) {
            self.exceeded = true;
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "output limit reached",
            ));
        }
        self.output.extend_from_slice(buffer);
        Ok(buffer.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
const CRATE_NAME: &str = "text";

#[cfg(test)]
mod tests {
    use std::num::{NonZeroU64, NonZeroUsize};

    use loader::gguf::{ArtifactByteLimit, Sha256Digest};
    use sha2::{Digest, Sha256};
    use test_fixtures::{Qwen35FixtureConfig, build_qwen35_fixture};
    use tokenize::TokenizerDigest;

    use super::*;

    const TOKENS: [&str; 5] = ["[UNK]", "<bos>", "<eos>", "hello", "assistant"];

    fn test_limits(tokenizer_bytes: usize) -> std::result::Result<PipelineLimits, std::io::Error> {
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

    fn tokenizer_json() -> String {
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

    fn verified_artifact(
        greedy_token_id: u32,
        add_bos: bool,
        add_eos: bool,
    ) -> Result<(tempfile::TempDir, VerifiedArtifact)> {
        let fixture = build_qwen35_fixture(&Qwen35FixtureConfig {
            tokens: TOKENS.iter().map(|token| (*token).to_owned()).collect(),
            bos_token_id: 1,
            eos_token_id: 2,
            add_bos,
            add_eos,
            chat_template: "{% if messages[0].content.startswith('h') %}hello{% endif %}"
                .to_owned(),
            greedy_token_id,
        })
        .map_err(|message| TemplateSnafu { message }.build())?;
        let directory = tempfile::tempdir().map_err(|error| {
            TemplateSnafu {
                message: error.to_string(),
            }
            .build()
        })?;
        let path = directory.path().join("synthetic.gguf");
        std::fs::write(&path, &fixture.bytes).map_err(|error| {
            TemplateSnafu {
                message: error.to_string(),
            }
            .build()
        })?;
        let limit = NonZeroU64::new(fixture.byte_len).ok_or_else(|| {
            InvalidConfigurationSnafu {
                rule: "synthetic fixture must have a non-zero byte length",
            }
            .build()
        })?;
        let artifact = VerifiedArtifact::load(
            &path,
            Sha256Digest::from_bytes(fixture.sha256),
            ArtifactByteLimit::new(limit),
        )
        .map_err(|error| {
            DecoderSnafu {
                message: error.to_string(),
            }
            .build()
        })?;
        Ok((directory, artifact))
    }

    fn pipeline_for(artifact: &VerifiedArtifact) -> Result<TextPipeline<'_>> {
        let tokenizer_json = tokenizer_json();
        let digest = TokenizerDigest::from_bytes(Sha256::digest(tokenizer_json.as_bytes()).into());
        TextPipeline::new(
            artifact,
            TokenizerCompanion::new(
                tokenizer_json.as_bytes(),
                TokenizerIdentity::new(tokenizer_json.len(), digest),
            ),
            test_limits(tokenizer_json.len()).map_err(|error| {
                TemplateSnafu {
                    message: error.to_string(),
                }
                .build()
            })?,
        )
    }
    #[test]
    fn crate_identity_matches_role() {
        assert_eq!(env!("CARGO_PKG_NAME"), CRATE_NAME);
    }
    #[test]
    fn capped_writer_refuses_overflow_without_partial_chunk() {
        let mut writer = ByteCappedWriter::new(3);
        assert_eq!(writer.write(b"ok").ok(), Some(2));
        assert!(writer.write(b"no").is_err());
        assert_eq!(writer.output, b"ok");
    }

    #[test]
    fn template_limits_refuse_fuel_overflow_and_silent_recursion_cap() -> Result<()> {
        let mut limits = test_limits(1).map_err(|error| {
            TemplateSnafu {
                message: error.to_string(),
            }
            .build()
        })?;
        limits.template_fuel = u64::MAX;
        assert!(matches!(
            limits.validate(),
            Err(Error::InvalidConfiguration { .. })
        ));
        limits.template_fuel = 1;
        limits.template_recursion = usize::MAX;
        assert!(matches!(
            compile_template("hello", limits),
            Err(Error::InvalidConfiguration { .. })
        ));
        Ok(())
    }

    #[test]
    fn template_resolution_has_no_registered_or_host_loader_authority() -> Result<()> {
        let environment = compile_template(
            "hello",
            test_limits(1).map_err(|error| {
                TemplateSnafu {
                    message: error.to_string(),
                }
                .build()
            })?,
        )?;
        for source in [
            "{% include 'missing' %}",
            "{% set target = 'missing' %}{% include target %}",
            "{% import 'missing' as imported %}",
            "{% set target = 'missing' %}{% import target as imported %}",
            "{% extends 'missing' %}{% block body %}x{% endblock %}",
            "{% extends '<string>' %}{% block body %}x{% endblock %}",
        ] {
            let template = environment.template_from_str(source).map_err(|error| {
                TemplateSnafu {
                    message: error.to_string(),
                }
                .build()
            })?;
            assert!(
                template.render(()).is_err(),
                "source unexpectedly resolved: {source}"
            );
        }
        let ignored = environment
            .template_from_str("a{% include 'missing' ignore missing %}b")
            .map_err(|error| {
                TemplateSnafu {
                    message: error.to_string(),
                }
                .build()
            })?
            .render(())
            .map_err(|error| {
                TemplateSnafu {
                    message: error.to_string(),
                }
                .build()
            })?;
        assert_eq!(ignored, "ab");
        Ok(())
    }

    #[test]
    fn verified_gguf_template_tokenizer_and_greedy_generation_are_bound() -> Result<()> {
        let tokenizer_json = tokenizer_json();
        let tokenizer_digest =
            TokenizerDigest::from_bytes(Sha256::digest(tokenizer_json.as_bytes()).into());
        let fixture = build_qwen35_fixture(&Qwen35FixtureConfig {
            tokens: TOKENS.iter().map(|token| (*token).to_owned()).collect(),
            bos_token_id: 1,
            eos_token_id: 2,
            add_bos: true,
            add_eos: false,
            chat_template: "{% if messages[0].content.startswith('h') %}hello{% endif %}"
                .to_owned(),
            greedy_token_id: 3,
        })
        .map_err(|message| TemplateSnafu { message }.build())?;
        let directory = tempfile::tempdir().map_err(|error| {
            TemplateSnafu {
                message: error.to_string(),
            }
            .build()
        })?;
        let path = directory.path().join("synthetic.gguf");
        std::fs::write(&path, &fixture.bytes).map_err(|error| {
            TemplateSnafu {
                message: error.to_string(),
            }
            .build()
        })?;
        let limit = NonZeroU64::new(fixture.byte_len).ok_or_else(|| {
            InvalidConfigurationSnafu {
                rule: "synthetic fixture must have a non-zero byte length",
            }
            .build()
        })?;
        let artifact = VerifiedArtifact::load(
            &path,
            Sha256Digest::from_bytes(fixture.sha256),
            ArtifactByteLimit::new(limit),
        )
        .map_err(|error| {
            DecoderSnafu {
                message: error.to_string(),
            }
            .build()
        })?;
        let pipeline = TextPipeline::new(
            &artifact,
            TokenizerCompanion::new(
                tokenizer_json.as_bytes(),
                TokenizerIdentity::new(tokenizer_json.len(), tokenizer_digest),
            ),
            test_limits(tokenizer_json.len()).map_err(|error| {
                TemplateSnafu {
                    message: error.to_string(),
                }
                .build()
            })?,
        )?;
        let messages = [TextMessage::new(TextRole::User, "hello")];
        let generation =
            pipeline.generate(GenerationRequest::new(&messages, 1, false), &NeverCancelled)?;
        assert_eq!(generation.text(), "hello");
        assert_eq!(generation.token_ids(), &[3]);
        assert_eq!(generation.finish_reason(), FinishReason::Length);
        Ok(())
    }

    #[test]
    fn eos_stops_before_collective_decoding() -> Result<()> {
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
    fn cancellation_after_prompt_step_returns_no_response_or_shared_state() -> Result<()> {
        struct CancelOnThirdCheck(std::cell::Cell<usize>);
        impl Cancellation for CancelOnThirdCheck {
            fn is_cancelled(&self) -> bool {
                let check = self.0.get();
                self.0.set(check + 1);
                check == 2
            }
        }
        let (_directory, artifact) = verified_artifact(3, true, false)?;
        let pipeline = pipeline_for(&artifact)?;
        let messages = [TextMessage::new(TextRole::User, "hello")];
        let cancelled = CancelOnThirdCheck(std::cell::Cell::new(0));
        assert!(matches!(
            pipeline.generate(GenerationRequest::new(&messages, 1, false), &cancelled),
            Err(Error::Cancelled { .. })
        ));
        let generation =
            pipeline.generate(GenerationRequest::new(&messages, 1, false), &NeverCancelled)?;
        assert_eq!(generation.token_ids(), &[3]);
        Ok(())
    }
}
