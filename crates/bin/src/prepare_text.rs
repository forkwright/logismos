//! Typed, preparation-only CLI adapter for one verified native text request.

use std::{
    ffi::OsString,
    fmt,
    fs::File,
    io::{Read, Take},
    num::NonZeroU64,
    path::{Path, PathBuf},
};

use decoders::{Qwen35CpuRequirements, Qwen35LogitSelection};
use loader::gguf::{ArtifactByteLimit, ArtifactDigest, Sha256Digest, VerifiedArtifact};
use serde::{Deserialize, Serialize};
use text::{
    GenerationRequest, NeverCancelled, PipelineLimits, TextMessage, TextPipeline, TextRole,
    TokenizerCompanion,
};
use tokenize::{TokenizerByteLimit, TokenizerDigest, TokenizerIdentity};

use crate::MAX_PLAN_INPUT_BYTES;

pub(crate) const PREPARE_TEXT_USAGE: &str = concat!(
    "usage: logismos prepare-text --model <path> --model-sha256 <hex> --model-bytes <bytes> ",
    "--tokenizer <path> --tokenizer-sha256 <hex> --tokenizer-bytes <bytes> ",
    "--request-json <json> (arguments must appear in this order)"
);

/// Parse and prepare one exact artifact-bound text request without execution.
pub(crate) fn command(
    mut arguments: impl Iterator<Item = OsString>,
) -> Result<PreparedTextOutcome, PrepareTextError> {
    let arguments = CommandArguments::parse(&mut arguments)?;
    let request = parse_request(&arguments.request_json)?;
    let model_limit = ArtifactByteLimit::new(arguments.model_bytes);
    let artifact = VerifiedArtifact::load(
        &arguments.model_path,
        Sha256Digest::from_bytes(arguments.model_digest),
        model_limit,
    )
    .map_err(|source| PrepareTextError::Model { source })?;
    let observed_model_bytes = artifact.observation().inspection().file_len;
    if observed_model_bytes != arguments.model_bytes.get() {
        return Err(PrepareTextError::ModelLengthMismatch);
    }

    let tokenizer_bytes = read_tokenizer(&arguments.tokenizer_path, arguments.tokenizer_bytes)?;
    let tokenizer_identity = TokenizerIdentity::new(
        arguments.tokenizer_bytes,
        TokenizerDigest::from_bytes(arguments.tokenizer_digest),
    );
    let PrepareTextRequest {
        messages: request_messages,
        max_output_tokens,
        enable_thinking,
        limits: request_limits,
    } = request;
    let limits = request_limits.into_pipeline_limits(
        TokenizerByteLimit::try_new(arguments.tokenizer_bytes)
            .map_err(|source| PrepareTextError::TokenizerConfiguration { source })?,
    );
    let pipeline = TextPipeline::new(
        &artifact,
        TokenizerCompanion::new(&tokenizer_bytes, tokenizer_identity),
        limits,
    )
    .map_err(|source| PrepareTextError::Pipeline { source })?;
    let messages = into_text_messages(request_messages)?;
    let prepared = pipeline
        .prepare(
            GenerationRequest::new(&messages, max_output_tokens, enable_thinking),
            &NeverCancelled,
        )
        .map_err(|source| PrepareTextError::Pipeline { source })?;
    PreparedTextOutcome::from_prepared(&prepared)
}

#[derive(Debug)]
pub(crate) enum PrepareTextError {
    InvalidArguments,
    InvalidRequest {
        source: serde_json::Error,
    },
    RequestExceedsByteLimit,
    IntegerOverflow,
    TokenizerOpen {
        source: std::io::Error,
    },
    TokenizerRead {
        source: std::io::Error,
    },
    TokenizerAllocation {
        source: std::collections::TryReserveError,
    },
    TokenizerInputExceedsExpected,
    TokenizerConfiguration {
        source: tokenize::Error,
    },
    Model {
        source: loader::Error,
    },
    ModelLengthMismatch,
    Pipeline {
        source: text::Error,
    },
    ReceiptAllocation {
        source: std::collections::TryReserveError,
    },
    InternalIdentity,
}

impl PrepareTextError {
    /// Stable receipt category without rendering a source that may contain a path or prompt.
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Self::InvalidArguments | Self::IntegerOverflow => "invalid_arguments",
            Self::InvalidRequest { .. } | Self::RequestExceedsByteLimit => "invalid_request",
            Self::TokenizerOpen { .. } | Self::TokenizerRead { .. } => "unreadable_tokenizer",
            Self::TokenizerAllocation { .. } | Self::ReceiptAllocation { .. } => "allocation",
            Self::TokenizerInputExceedsExpected => "tokenizer_identity_mismatch",
            Self::TokenizerConfiguration { source } => tokenizer_error_kind(source),
            Self::Model { source } => model_error_kind(source),
            Self::ModelLengthMismatch => "model_identity_mismatch",
            Self::Pipeline { source } => pipeline_error_kind(source),
            Self::InternalIdentity => "internal",
        }
    }
}

impl fmt::Display for PrepareTextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.kind())
    }
}

impl std::error::Error for PrepareTextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidRequest { source } => Some(source),
            Self::TokenizerOpen { source } | Self::TokenizerRead { source } => Some(source),
            Self::TokenizerAllocation { source } | Self::ReceiptAllocation { source } => {
                Some(source)
            }
            Self::TokenizerConfiguration { source } => Some(source),
            Self::Model { source } => Some(source),
            Self::Pipeline { source } => Some(source),
            Self::InvalidArguments
            | Self::RequestExceedsByteLimit
            | Self::IntegerOverflow
            | Self::TokenizerInputExceedsExpected
            | Self::ModelLengthMismatch
            | Self::InternalIdentity => None,
        }
    }
}

fn model_error_kind(error: &loader::Error) -> &'static str {
    match error {
        loader::Error::ArtifactDigestMismatch { .. } => "model_identity_mismatch",
        loader::Error::ArtifactExceedsByteLimit { .. } => "model_byte_limit",
        loader::Error::Io { .. } | loader::Error::ArtifactInputNotRegular { .. } => {
            "unreadable_model"
        }
        loader::Error::MmapStale { .. } => "concurrent_mutation",
        loader::Error::ArtifactBackingAllocation { .. } => "allocation",
        loader::Error::Gguf { .. } | loader::Error::UnknownGgmlType { .. } => "invalid_gguf",
        _ => "model_refused",
    }
}

fn tokenizer_error_kind(error: &tokenize::Error) -> &'static str {
    match error {
        tokenize::Error::DigestMismatch { .. } | tokenize::Error::ByteLengthMismatch { .. } => {
            "tokenizer_identity_mismatch"
        }
        tokenize::Error::ConfiguredPadding { .. }
        | tokenize::Error::ConfiguredTruncation { .. } => "tokenizer_configuration",
        tokenize::Error::ByteLimitExceeded { .. } | tokenize::Error::InvalidByteLimit { .. } => {
            "tokenizer_byte_limit"
        }
        _ => "tokenizer_refused",
    }
}

fn pipeline_error_kind(error: &text::Error) -> &'static str {
    match error {
        text::Error::Tokenizer { source, .. } => tokenizer_error_kind(source),
        text::Error::Decoder { .. } => "decoder_plan_refused",
        text::Error::Allocation { .. } => "allocation",
        text::Error::LimitExceeded { .. } | text::Error::InvalidConfiguration { .. } => {
            "request_limit"
        }
        text::Error::Template { .. }
        | text::Error::TemplateRenderer { .. }
        | text::Error::RenderedUtf8 { .. }
        | text::Error::EmptyPrompt { .. } => "template_refused",
        text::Error::VocabularyMismatch { .. }
        | text::Error::Metadata { .. }
        | text::Error::SpecialTokenPolicy { .. } => "artifact_tokenizer_mismatch",
        _ => "text_refused",
    }
}

struct CommandArguments {
    model_path: PathBuf,
    model_digest: [u8; 32],
    model_bytes: NonZeroU64,
    tokenizer_path: PathBuf,
    tokenizer_digest: [u8; 32],
    tokenizer_bytes: usize,
    request_json: String,
}

impl CommandArguments {
    fn parse(arguments: &mut impl Iterator<Item = OsString>) -> Result<Self, PrepareTextError> {
        let model_path = PathBuf::from(required_value(arguments, "--model")?);
        let model_digest = parse_digest(required_value(arguments, "--model-sha256")?)?;
        let model_bytes = parse_nonzero_u64(required_value(arguments, "--model-bytes")?)?;
        let tokenizer_path = PathBuf::from(required_value(arguments, "--tokenizer")?);
        let tokenizer_digest = parse_digest(required_value(arguments, "--tokenizer-sha256")?)?;
        let tokenizer_bytes = usize::try_from(
            parse_nonzero_u64(required_value(arguments, "--tokenizer-bytes")?)?.get(),
        )
        .map_err(|_| PrepareTextError::IntegerOverflow)?;
        let request_json = required_value(arguments, "--request-json")?
            .into_string()
            .map_err(|_| PrepareTextError::InvalidArguments)?;
        if request_json.len() > MAX_PLAN_INPUT_BYTES {
            return Err(PrepareTextError::RequestExceedsByteLimit);
        }
        if arguments.next().is_some() {
            return Err(PrepareTextError::InvalidArguments);
        }
        Ok(Self {
            model_path,
            model_digest,
            model_bytes,
            tokenizer_path,
            tokenizer_digest,
            tokenizer_bytes,
            request_json,
        })
    }
}

fn required_value(
    arguments: &mut impl Iterator<Item = OsString>,
    expected_flag: &str,
) -> Result<OsString, PrepareTextError> {
    let Some(flag) = arguments.next() else {
        return Err(PrepareTextError::InvalidArguments);
    };
    if flag.to_str() != Some(expected_flag) {
        return Err(PrepareTextError::InvalidArguments);
    }
    arguments.next().ok_or(PrepareTextError::InvalidArguments)
}

fn parse_nonzero_u64(value: OsString) -> Result<NonZeroU64, PrepareTextError> {
    let value = value
        .into_string()
        .map_err(|_| PrepareTextError::InvalidArguments)?;
    let parsed = value
        .parse::<u64>()
        .map_err(|_| PrepareTextError::InvalidArguments)?;
    NonZeroU64::new(parsed).ok_or(PrepareTextError::InvalidArguments)
}

fn parse_digest(value: OsString) -> Result<[u8; 32], PrepareTextError> {
    let value = value
        .into_string()
        .map_err(|_| PrepareTextError::InvalidArguments)?;
    let bytes = value.as_bytes();
    if bytes.len() != 64 {
        return Err(PrepareTextError::InvalidArguments);
    }
    let mut digest = [0_u8; 32];
    for (index, pair) in bytes.chunks_exact(2).enumerate() {
        let high = hex_nibble(pair[0]).ok_or(PrepareTextError::InvalidArguments)?;
        let low = hex_nibble(pair[1]).ok_or(PrepareTextError::InvalidArguments)?;
        digest[index] = (high << 4) | low;
    }
    Ok(digest)
}

const fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

fn read_tokenizer(path: &Path, expected: usize) -> Result<Vec<u8>, PrepareTextError> {
    let bounded = expected
        .checked_add(1)
        .ok_or(PrepareTextError::IntegerOverflow)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(bounded)
        .map_err(|source| PrepareTextError::TokenizerAllocation { source })?;
    let file = File::open(path).map_err(|source| PrepareTextError::TokenizerOpen { source })?;
    let bounded = u64::try_from(bounded).map_err(|_| PrepareTextError::IntegerOverflow)?;
    let mut limited: Take<File> = file.take(bounded);
    limited
        .read_to_end(&mut bytes)
        .map_err(|source| PrepareTextError::TokenizerRead { source })?;
    if bytes.len() > expected {
        return Err(PrepareTextError::TokenizerInputExceedsExpected);
    }
    Ok(bytes)
}

fn parse_request(input: &str) -> Result<PrepareTextRequest, PrepareTextError> {
    serde_json::from_str(input).map_err(|source| PrepareTextError::InvalidRequest { source })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PrepareTextRequest {
    messages: Vec<PrepareTextMessage>,
    max_output_tokens: usize,
    enable_thinking: bool,
    limits: PrepareTextLimits,
}

fn into_text_messages(
    request_messages: Vec<PrepareTextMessage>,
) -> Result<Vec<TextMessage>, PrepareTextError> {
    let mut messages = Vec::new();
    messages
        .try_reserve_exact(request_messages.len())
        .map_err(|source| PrepareTextError::ReceiptAllocation { source })?;
    for message in request_messages {
        messages.push(TextMessage::new(
            message.role.into_text_role(),
            message.content,
        ));
    }
    Ok(messages)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PrepareTextMessage {
    role: PrepareTextRole,
    content: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum PrepareTextRole {
    System,
    User,
    Assistant,
}

impl PrepareTextRole {
    const fn into_text_role(self) -> TextRole {
        match self {
            Self::System => TextRole::System,
            Self::User => TextRole::User,
            Self::Assistant => TextRole::Assistant,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PrepareTextLimits {
    template_bytes: usize,
    messages: usize,
    message_bytes: usize,
    prompt_bytes: usize,
    rendered_bytes: usize,
    context_tokens: usize,
    output_tokens: usize,
    output_bytes: usize,
    template_fuel: u64,
    template_recursion: usize,
}

impl PrepareTextLimits {
    const fn into_pipeline_limits(self, tokenizer_bytes: TokenizerByteLimit) -> PipelineLimits {
        PipelineLimits {
            tokenizer_bytes,
            template_bytes: self.template_bytes,
            messages: self.messages,
            message_bytes: self.message_bytes,
            prompt_bytes: self.prompt_bytes,
            rendered_bytes: self.rendered_bytes,
            context_tokens: self.context_tokens,
            output_tokens: self.output_tokens,
            output_bytes: self.output_bytes,
            template_fuel: self.template_fuel,
            template_recursion: self.template_recursion,
        }
    }
}

/// Bounded JSON evidence from preparation alone, never decoder execution.
#[derive(Serialize)]
pub(crate) struct PreparedTextOutcome {
    schema_version: u32,
    outcome: &'static str,
    model: VerifiedIdentityReceipt,
    tokenizer: VerifiedIdentityReceipt,
    rendered_prompt: String,
    prompt_token_ids: Vec<u32>,
    max_output_tokens: usize,
    decoder_cpu_requirements: DecoderCpuRequirementsReceipt,
}

impl PreparedTextOutcome {
    fn from_prepared(
        prepared: &text::PreparedGeneration<'_, '_>,
    ) -> Result<Self, PrepareTextError> {
        let requirements = prepared.decoder_cpu_requirements();
        let model = VerifiedIdentityReceipt {
            algorithm: "sha256",
            sha256: artifact_digest_hex(requirements.artifact_digest())?,
            bytes: requirements.serialized_backing_bytes(),
        };
        let tokenizer_identity = prepared.tokenizer_identity();
        let tokenizer = VerifiedIdentityReceipt {
            algorithm: "sha256",
            sha256: hex_digest(tokenizer_identity.digest().as_bytes())?,
            bytes: u64::try_from(tokenizer_identity.byte_length())
                .map_err(|_| PrepareTextError::InternalIdentity)?,
        };
        Ok(Self {
            schema_version: 1,
            outcome: "text_prepared",
            model,
            tokenizer,
            rendered_prompt: copy_string(prepared.rendered_prompt())?,
            prompt_token_ids: copy_ids(prepared.prompt_token_ids())?,
            max_output_tokens: prepared.max_output_tokens(),
            decoder_cpu_requirements: DecoderCpuRequirementsReceipt::from_requirements(
                requirements,
            )?,
        })
    }
}

#[derive(Serialize)]
struct VerifiedIdentityReceipt {
    algorithm: &'static str,
    sha256: String,
    bytes: u64,
}

#[derive(Serialize)]
struct DecoderCpuRequirementsReceipt {
    logical_f32_scope: &'static str,
    artifact_sha256: String,
    serialized_backing_bytes: u64,
    retained_bytes: u64,
    transaction_copy_bytes: u64,
    workspace_upper_bound_bytes: u64,
    returned_logits_bytes: u64,
    logical_f32_upper_bound_bytes: u64,
    max_context: usize,
    max_step_tokens: usize,
    selection: &'static str,
}

impl DecoderCpuRequirementsReceipt {
    fn from_requirements(requirements: Qwen35CpuRequirements) -> Result<Self, PrepareTextError> {
        let selection = match requirements.selection() {
            Qwen35LogitSelection::LastToken => "last_token",
            Qwen35LogitSelection::AllTokens => "all_tokens",
            _ => return Err(PrepareTextError::InternalIdentity),
        };
        Ok(Self {
            logical_f32_scope: "decoder-only; excludes artifact backing, prompt IDs, rendered text, template, tokenizer, RSS, and GPU memory",
            artifact_sha256: artifact_digest_hex(requirements.artifact_digest())?,
            serialized_backing_bytes: requirements.serialized_backing_bytes(),
            retained_bytes: requirements.retained_bytes(),
            transaction_copy_bytes: requirements.transaction_copy_bytes(),
            workspace_upper_bound_bytes: requirements.workspace_upper_bound_bytes(),
            returned_logits_bytes: requirements.returned_logits_bytes(),
            logical_f32_upper_bound_bytes: requirements.logical_f32_upper_bound_bytes(),
            max_context: requirements.max_context(),
            max_step_tokens: requirements.max_step_tokens(),
            selection,
        })
    }
}

fn artifact_digest_hex(digest: ArtifactDigest) -> Result<String, PrepareTextError> {
    match digest {
        ArtifactDigest::Sha256(digest) => hex_digest(digest.as_bytes()),
        _ => Err(PrepareTextError::InternalIdentity),
    }
}

fn hex_digest(bytes: &[u8; 32]) -> Result<String, PrepareTextError> {
    let mut output = String::new();
    output
        .try_reserve_exact(64)
        .map_err(|source| PrepareTextError::ReceiptAllocation { source })?;
    for byte in bytes {
        use std::fmt::Write;
        write!(output, "{byte:02x}").map_err(|_| PrepareTextError::InternalIdentity)?;
    }
    Ok(output)
}

fn copy_string(value: &str) -> Result<String, PrepareTextError> {
    let mut copy = String::new();
    copy.try_reserve_exact(value.len())
        .map_err(|source| PrepareTextError::ReceiptAllocation { source })?;
    copy.push_str(value);
    Ok(copy)
}

fn copy_ids(value: &[u32]) -> Result<Vec<u32>, PrepareTextError> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(value.len())
        .map_err(|source| PrepareTextError::ReceiptAllocation { source })?;
    copy.extend_from_slice(value);
    Ok(copy)
}
