use std::mem::size_of;

use serde::Deserialize;
use snafu::{IntoError, ResultExt};
use tokenizers::decoders::DecoderWrapper;
use tokenizers::pre_tokenizers::metaspace::PrependScheme;
use tokenizers::utils::SysRegex;

use crate::error::{
    DecodePlanOverflowSnafu, DecodeStorageAllocationSnafu, DecodeStorageExhaustedSnafu,
    DecodeTokenCapacityExceededSnafu, DecodeUtf8InvariantSnafu, DecodedByteLimitExceededSnafu,
    DecoderProgramSnafu,
};
use crate::{Error, Result};

const REPLACEMENT_CHARACTER: &str = "�";
const BYTE_LEVEL_DIRECT_ASCII_START: u32 = 33;
const BYTE_LEVEL_DIRECT_ASCII_END: u32 = 126;
const BYTE_LEVEL_DIRECT_LATIN_START: u32 = 161;
const BYTE_LEVEL_DIRECT_LATIN_GAP: u32 = 173;
const BYTE_LEVEL_DIRECT_LATIN_END: u32 = 255;
const BYTE_LEVEL_DIRECT_LATIN_FIRST_END: u32 = 172;
const BYTE_LEVEL_DIRECT_LATIN_SECOND_START: u32 = 174;
const BYTE_LEVEL_FIRST_EXTRA_START: u32 = 256;
const BYTE_LEVEL_FIRST_EXTRA_END: u32 = 288;
const BYTE_LEVEL_SECOND_EXTRA_START: u32 = 289;
const BYTE_LEVEL_SECOND_EXTRA_END: u32 = 322;
const BYTE_LEVEL_LAST_EXTRA: u32 = 323;
const BYTE_LEVEL_SECOND_OFFSET: u32 = 162;

const WORDPIECE_CLEANUP: [(&str, &str); 10] = [
    (" .", "."),
    (" ?", "?"),
    (" !", "!"),
    (" ,", ","),
    (" ' ", "'"),
    (" n't", "n't"),
    (" 'm", "'m"),
    (" do not", " don't"),
    (" 's", "'s"),
    (" 've", "'ve"),
];
const WORDPIECE_CLEANUP_LAST: (&str, &str) = (" 're", "'re");

/// Checked requested storage for one collective token sequence.
///
/// WHY: generation must acquire every owned container before model execution,
/// while keeping native regex and allocator overhead as separate evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeStoragePlan {
    token_capacity: usize,
    output_capacity: usize,
    arena_capacity: usize,
    span_capacity: usize,
    raw_capacity: usize,
    maximum_decoded_bytes: usize,
    requested_retained_bytes: usize,
    requested_scratch_bytes: usize,
}

impl DecodeStoragePlan {
    /// Return requested bytes that remain in a successful text/ID result.
    #[must_use]
    pub const fn requested_retained_bytes(self) -> usize {
        self.requested_retained_bytes
    }

    /// Return requested bytes dropped with the transform scratch after decode.
    #[must_use]
    pub const fn requested_scratch_bytes(self) -> usize {
        self.requested_scratch_bytes
    }

    /// Return the decoder-derived maximum before the caller's byte limit.
    #[must_use]
    pub const fn maximum_decoded_bytes(self) -> usize {
        self.maximum_decoded_bytes
    }

    /// Reserve every owned collective-decode container.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DecodeStorageAllocation`] before decoder execution when
    /// the standard allocator cannot satisfy any requested container.
    pub fn acquire(self) -> Result<DecodeStorage> {
        let token_ids = reserved_vec(self.token_capacity, "generated token IDs")?;
        let output = reserved_string(self.output_capacity, "decoded output bytes")?;
        let first = TokenBuffer::acquire(self.arena_capacity, self.span_capacity)?;
        let second = TokenBuffer::acquire(self.arena_capacity, self.span_capacity)?;
        let raw = reserved_vec(self.raw_capacity, "collective raw-byte scratch")?;
        Ok(DecodeStorage {
            token_limit: self.token_capacity,
            output_limit: self.output_capacity,
            raw_limit: self.raw_capacity,
            token_ids,
            output,
            first,
            second,
            raw,
        })
    }
}

/// Acquired request-local storage for generated IDs and collective decoding.
///
/// Values can enter only through checked methods, so execution never grows an
/// owned container after this storage has been acquired.
pub struct DecodeStorage {
    token_limit: usize,
    output_limit: usize,
    raw_limit: usize,
    token_ids: Vec<u32>,
    output: String,
    first: TokenBuffer,
    second: TokenBuffer,
    raw: Vec<u8>,
}

impl DecodeStorage {
    /// Append one generated non-stop token ID within the acquired capacity.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DecodeTokenCapacityExceeded`] without writing when the
    /// acquired generated-ID capacity is already full.
    pub fn push_token_id(&mut self, token_id: u32) -> Result<()> {
        let actual = checked_add(self.token_ids.len(), 1, "generated token count")?;
        if actual > self.token_limit || actual > self.token_ids.capacity() {
            return DecodeTokenCapacityExceededSnafu {
                actual,
                capacity: self.token_limit,
            }
            .fail();
        }
        self.token_ids.push(token_id);
        Ok(())
    }

    /// Borrow generated token IDs already accepted by this storage.
    #[must_use]
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }
}

pub(crate) struct CollectiveDecoder {
    vocabulary: Vocabulary,
    stages: Vec<Stage>,
    join_with_spaces: bool,
}

impl CollectiveDecoder {
    pub(crate) fn compile(tokenizer: &tokenizers::Tokenizer) -> Result<Self> {
        let vocabulary = Vocabulary::compile(tokenizer)?;
        let mut stages = Vec::new();
        let join_with_spaces = tokenizer.get_decoder().is_none();
        if let Some(decoder) = tokenizer.get_decoder() {
            append_decoder(decoder, &mut stages)?;
        }
        Ok(Self {
            vocabulary,
            stages,
            join_with_spaces,
        })
    }

    pub(crate) fn storage_plan(
        &self,
        token_capacity: usize,
        output_capacity: usize,
    ) -> Result<DecodeStoragePlan> {
        let initial_bytes = checked_mul(
            token_capacity,
            self.vocabulary.max_spelling_bytes,
            "initial canonical spelling bytes",
        )?;
        let mut shape = Shape::new(initial_bytes, token_capacity);
        let mut arena_capacity = shape.bytes;
        let mut span_capacity = shape.spans;
        let mut raw_capacity = 0;
        for stage in &self.stages {
            if stage.uses_raw_bytes() {
                raw_capacity = raw_capacity.max(shape.bytes);
            }
            shape = stage.output_bound(shape)?;
            arena_capacity = arena_capacity.max(shape.bytes);
            span_capacity = span_capacity.max(shape.spans);
        }
        let maximum_decoded_bytes = self.final_byte_bound(shape)?;
        make_storage_plan(
            token_capacity,
            output_capacity,
            arena_capacity,
            span_capacity,
            raw_capacity,
            maximum_decoded_bytes,
        )
    }

    pub(crate) fn owned_resident_bytes(&self) -> Result<usize> {
        let vocabulary_index = checked_mul(
            self.vocabulary.entries.capacity(),
            size_of::<VocabularyEntry>(),
            "resident sparse vocabulary index bytes",
        )?;
        let stage_vector = checked_mul(
            self.stages.capacity(),
            size_of::<Stage>(),
            "resident decoder stage-vector bytes",
        )?;
        let mut total = checked_add(
            self.vocabulary.slab.capacity(),
            vocabulary_index,
            "resident vocabulary container bytes",
        )?;
        total = checked_add(total, stage_vector, "resident decoder program bytes")?;
        for stage in &self.stages {
            total = checked_add(
                total,
                stage.owned_text_capacity()?,
                "resident decoder text bytes",
            )?;
        }
        Ok(total)
    }

    pub(crate) fn decode(
        &self,
        storage: DecodeStorage,
        skip_special_tokens: bool,
    ) -> Result<(String, Vec<u32>)> {
        let DecodeStorage {
            token_ids,
            mut output,
            mut first,
            mut second,
            mut raw,
            output_limit,
            raw_limit,
            token_limit: _,
        } = storage;
        self.vocabulary
            .seed(&token_ids, skip_special_tokens, &mut first)?;
        for stage in &self.stages {
            second.clear();
            stage.apply(&first, &mut second, &mut raw, raw_limit)?;
            std::mem::swap(&mut first, &mut second);
        }
        write_output(&first, self.join_with_spaces, &mut output, output_limit)?;
        drop(raw);
        drop(second);
        drop(first);
        Ok((output, token_ids))
    }

    fn final_byte_bound(&self, shape: Shape) -> Result<usize> {
        if self.join_with_spaces {
            checked_add(
                shape.bytes,
                shape.spans.saturating_sub(1),
                "space-joined decoded bytes",
            )
        } else {
            Ok(shape.bytes)
        }
    }
}

#[derive(Clone, Copy)]
struct Shape {
    bytes: usize,
    spans: usize,
}

impl Shape {
    const fn new(bytes: usize, spans: usize) -> Self {
        Self { bytes, spans }
    }
}

struct Vocabulary {
    slab: String,
    entries: Vec<VocabularyEntry>,
    max_spelling_bytes: usize,
}

impl Vocabulary {
    fn compile(tokenizer: &tokenizers::Tokenizer) -> Result<Self> {
        let vocabulary = tokenizer.get_vocab(true);
        let mut identifiers = reserved_vec(vocabulary.len(), "sparse vocabulary identifiers")?;
        identifiers.extend(vocabulary.into_values());
        identifiers.sort_unstable();
        identifiers.dedup();
        let added_tokens = tokenizer.get_added_tokens_decoder();
        let mut slab = String::new();
        let mut entries = reserved_vec(identifiers.len(), "sparse vocabulary spans")?;
        let mut max_spelling_bytes = 0;
        for identifier in identifiers {
            let Some(spelling) = tokenizer.id_to_token(identifier) else {
                continue;
            };
            slab.try_reserve_exact(spelling.len())
                .context(DecodeStorageAllocationSnafu {
                    target: "canonical vocabulary spelling slab",
                })?;
            let start = slab.len();
            slab.push_str(&spelling);
            let end = slab.len();
            let special = added_tokens
                .get(&identifier)
                .is_some_and(|token| token.special);
            entries.push(VocabularyEntry {
                identifier,
                span: TokenSpan { start, end },
                special,
            });
            max_spelling_bytes = max_spelling_bytes.max(spelling.len());
        }
        Ok(Self {
            slab,
            entries,
            max_spelling_bytes,
        })
    }

    fn seed(
        &self,
        identifiers: &[u32],
        skip_special_tokens: bool,
        target: &mut TokenBuffer,
    ) -> Result<()> {
        target.clear();
        for identifier in identifiers {
            let Ok(index) = self
                .entries
                .binary_search_by_key(identifier, |entry| entry.identifier)
            else {
                continue;
            };
            let Some(entry) = self.entries.get(index) else {
                continue;
            };
            if skip_special_tokens && entry.special {
                continue;
            }
            let spelling = span_text(self.slab.as_bytes(), entry.span)?;
            target.push_str(spelling)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct VocabularyEntry {
    identifier: u32,
    span: TokenSpan,
    special: bool,
}

#[derive(Clone, Copy)]
struct TokenSpan {
    start: usize,
    end: usize,
}

struct TokenBuffer {
    bytes: Vec<u8>,
    spans: Vec<TokenSpan>,
    byte_limit: usize,
    span_limit: usize,
}

impl TokenBuffer {
    fn acquire(byte_limit: usize, span_limit: usize) -> Result<Self> {
        Ok(Self {
            bytes: reserved_vec(byte_limit, "collective transform byte arena")?,
            spans: reserved_vec(span_limit, "collective transform span index")?,
            byte_limit,
            span_limit,
        })
    }

    fn clear(&mut self) {
        self.bytes.clear();
        self.spans.clear();
    }

    fn push_str(&mut self, text: &str) -> Result<()> {
        let start = self.begin_token()?;
        self.append_str(text)?;
        self.finish_token(start)
    }

    fn begin_token(&self) -> Result<usize> {
        let needed = checked_add(self.spans.len(), 1, "collective transform span count")?;
        ensure_storage(
            needed,
            self.span_limit.min(self.spans.capacity()),
            "span-index",
        )?;
        Ok(self.bytes.len())
    }

    fn append_str(&mut self, text: &str) -> Result<()> {
        let needed = checked_add(
            self.bytes.len(),
            text.len(),
            "collective transform byte count",
        )?;
        ensure_storage(
            needed,
            self.byte_limit.min(self.bytes.capacity()),
            "byte-arena",
        )?;
        self.bytes.extend_from_slice(text.as_bytes());
        Ok(())
    }

    fn append_char(&mut self, character: char) -> Result<()> {
        let mut encoded = [0; 4];
        self.append_str(character.encode_utf8(&mut encoded))
    }

    fn finish_token(&mut self, start: usize) -> Result<()> {
        let needed = checked_add(self.spans.len(), 1, "collective transform span count")?;
        ensure_storage(
            needed,
            self.span_limit.min(self.spans.capacity()),
            "span-index",
        )?;
        self.spans.push(TokenSpan {
            start,
            end: self.bytes.len(),
        });
        Ok(())
    }

    fn token(&self, span: TokenSpan) -> Result<&str> {
        span_text(&self.bytes, span)
    }
}

enum Stage {
    Bpe {
        suffix: String,
    },
    ByteLevel,
    WordPiecePrefix {
        prefix: String,
    },
    Metaspace {
        replacement: char,
        trim_first: bool,
    },
    CtcDeduplicate,
    FilterEmpty,
    ReplaceLiteral {
        pattern: String,
        content: String,
        empty_input_matches: bool,
    },
    ReplaceRegex {
        regex: SysRegex,
        content: String,
    },
    Fuse,
    Strip {
        content: char,
        start: usize,
        stop: usize,
    },
    ByteFallback,
}

impl Stage {
    fn output_bound(&self, input: Shape) -> Result<Shape> {
        let bytes = match self {
            Self::Bpe { suffix } if suffix.is_empty() => checked_add(
                checked_mul(input.bytes, 2, "empty-suffix BPE bytes")?,
                input.spans,
                "empty-suffix BPE boundaries",
            )?,
            Self::ByteLevel => checked_mul(input.bytes, 3, "lossy byte-level UTF-8 bytes")?,
            Self::WordPiecePrefix { .. } => checked_add(
                input.bytes,
                input.spans.saturating_sub(1),
                "WordPiece separator bytes",
            )?,
            Self::ReplaceLiteral {
                pattern, content, ..
            } => literal_replacement_bound(input, pattern, content)?,
            Self::ReplaceRegex { content, .. } => regex_replacement_bound(input, content)?,
            _ => input.bytes,
        };
        let spans = match self {
            Self::ByteLevel | Self::Fuse => 1,
            _ => input.spans,
        };
        Ok(Shape::new(bytes, spans))
    }

    const fn uses_raw_bytes(&self) -> bool {
        matches!(self, Self::ByteLevel | Self::ByteFallback)
    }

    fn owned_text_capacity(&self) -> Result<usize> {
        match self {
            Self::Bpe { suffix } => Ok(suffix.capacity()),
            Self::WordPiecePrefix { prefix } => Ok(prefix.capacity()),
            Self::ReplaceLiteral {
                pattern, content, ..
            } => checked_add(
                pattern.capacity(),
                content.capacity(),
                "resident literal replacement text bytes",
            ),
            Self::ReplaceRegex { content, .. } => Ok(content.capacity()),
            _ => Ok(0),
        }
    }

    fn apply(
        &self,
        source: &TokenBuffer,
        target: &mut TokenBuffer,
        raw: &mut Vec<u8>,
        raw_limit: usize,
    ) -> Result<()> {
        match self {
            Self::Bpe { suffix } => apply_bpe(source, target, suffix),
            Self::ByteLevel => apply_byte_level(source, target, raw, raw_limit),
            Self::WordPiecePrefix { prefix } => apply_wordpiece(source, target, prefix),
            Self::Metaspace {
                replacement,
                trim_first,
            } => apply_metaspace(source, target, *replacement, *trim_first),
            Self::CtcDeduplicate => apply_deduplicate(source, target),
            Self::FilterEmpty => apply_filter_empty(source, target),
            Self::ReplaceLiteral {
                pattern,
                content,
                empty_input_matches,
            } => apply_literal_replace(source, target, pattern, content, *empty_input_matches),
            Self::ReplaceRegex { regex, content } => {
                apply_regex_replace(source, target, regex, content)
            }
            Self::Fuse => apply_fuse(source, target),
            Self::Strip {
                content,
                start,
                stop,
            } => apply_strip(source, target, *content, *start, *stop),
            Self::ByteFallback => apply_byte_fallback(source, target, raw, raw_limit),
        }
    }
}

#[derive(Deserialize)]
struct SerializedReplace {
    pattern: SerializedPattern,
    content: String,
}

#[derive(Deserialize)]
enum SerializedPattern {
    String(String),
    Regex(String),
}

fn append_decoder(decoder: &DecoderWrapper, stages: &mut Vec<Stage>) -> Result<()> {
    match decoder {
        DecoderWrapper::BPE(decoder) => push_stage(
            stages,
            Stage::Bpe {
                suffix: copy_string(&decoder.suffix)?,
            },
        ),
        DecoderWrapper::ByteLevel(_) => push_stage(stages, Stage::ByteLevel),
        DecoderWrapper::WordPiece(decoder) => append_wordpiece(decoder, stages),
        DecoderWrapper::Metaspace(decoder) => push_stage(
            stages,
            Stage::Metaspace {
                replacement: decoder.get_replacement(),
                trim_first: decoder.get_prepend_scheme() != PrependScheme::Never,
            },
        ),
        DecoderWrapper::CTC(decoder) => append_ctc(decoder, stages),
        DecoderWrapper::Sequence(sequence) => {
            for nested in sequence.get_decoders() {
                append_decoder(nested, stages)?;
            }
            Ok(())
        }
        DecoderWrapper::Replace(decoder) => append_replace(decoder, stages),
        DecoderWrapper::Fuse(_) => push_stage(stages, Stage::Fuse),
        DecoderWrapper::Strip(decoder) => push_stage(
            stages,
            Stage::Strip {
                content: decoder.content,
                start: decoder.start,
                stop: decoder.stop,
            },
        ),
        DecoderWrapper::ByteFallback(_) => push_stage(stages, Stage::ByteFallback),
    }
}

fn append_wordpiece(
    decoder: &tokenizers::decoders::wordpiece::WordPiece,
    stages: &mut Vec<Stage>,
) -> Result<()> {
    push_stage(
        stages,
        Stage::WordPiecePrefix {
            prefix: copy_string(&decoder.prefix)?,
        },
    )?;
    if decoder.cleanup {
        append_cleanup(stages)?;
    }
    Ok(())
}

fn append_ctc(decoder: &tokenizers::decoders::ctc::CTC, stages: &mut Vec<Stage>) -> Result<()> {
    push_stage(stages, Stage::CtcDeduplicate)?;
    push_literal_stage(stages, &decoder.pad_token, "")?;
    if decoder.cleanup {
        append_cleanup(stages)?;
        push_literal_stage(stages, &decoder.word_delimiter_token, " ")?;
    }
    push_stage(stages, Stage::FilterEmpty)
}

fn append_cleanup(stages: &mut Vec<Stage>) -> Result<()> {
    for (pattern, content) in WORDPIECE_CLEANUP {
        push_literal_stage(stages, pattern, content)?;
    }
    push_literal_stage(stages, WORDPIECE_CLEANUP_LAST.0, WORDPIECE_CLEANUP_LAST.1)
}

fn append_replace(
    decoder: &tokenizers::normalizers::replace::Replace,
    stages: &mut Vec<Stage>,
) -> Result<()> {
    let serialized =
        serde_json::to_value(decoder).map_err(|error| decoder_program_error(error.to_string()))?;
    let config: SerializedReplace = serde_json::from_value(serialized)
        .map_err(|error| decoder_program_error(error.to_string()))?;
    match config.pattern {
        SerializedPattern::String(pattern) => push_stage(
            stages,
            Stage::ReplaceLiteral {
                pattern,
                content: config.content,
                empty_input_matches: false,
            },
        ),
        SerializedPattern::Regex(source) => {
            let regex =
                SysRegex::new(&source).map_err(|error| decoder_program_error(error.to_string()))?;
            push_stage(
                stages,
                Stage::ReplaceRegex {
                    regex,
                    content: config.content,
                },
            )
        }
    }
}

fn push_literal_stage(stages: &mut Vec<Stage>, pattern: &str, content: &str) -> Result<()> {
    push_stage(
        stages,
        Stage::ReplaceLiteral {
            pattern: copy_string(pattern)?,
            content: copy_string(content)?,
            empty_input_matches: true,
        },
    )
}

fn push_stage(stages: &mut Vec<Stage>, stage: Stage) -> Result<()> {
    stages
        .try_reserve(1)
        .context(DecodeStorageAllocationSnafu {
            target: "compiled collective decoder stages",
        })?;
    stages.push(stage);
    Ok(())
}

fn copy_string(source: &str) -> Result<String> {
    let mut owned = reserved_string(source.len(), "compiled decoder text")?;
    owned.push_str(source);
    Ok(owned)
}

fn make_storage_plan(
    token_capacity: usize,
    output_capacity: usize,
    arena_capacity: usize,
    span_capacity: usize,
    raw_capacity: usize,
    maximum_decoded_bytes: usize,
) -> Result<DecodeStoragePlan> {
    let identifier_bytes = checked_mul(token_capacity, size_of::<u32>(), "retained ID bytes")?;
    let requested_retained_bytes = checked_add(
        output_capacity,
        identifier_bytes,
        "retained output and ID bytes",
    )?;
    let arena_bytes = checked_mul(arena_capacity, 2, "paired transform arenas")?;
    let span_units = checked_mul(span_capacity, size_of::<TokenSpan>(), "span-index bytes")?;
    let span_bytes = checked_mul(span_units, 2, "paired span indexes")?;
    let requested_scratch_bytes = checked_add(
        checked_add(arena_bytes, span_bytes, "transform arena and index bytes")?,
        raw_capacity,
        "transform and raw scratch bytes",
    )?;
    Ok(DecodeStoragePlan {
        token_capacity,
        output_capacity,
        arena_capacity,
        span_capacity,
        raw_capacity,
        maximum_decoded_bytes,
        requested_retained_bytes,
        requested_scratch_bytes,
    })
}

fn literal_replacement_bound(input: Shape, pattern: &str, content: &str) -> Result<usize> {
    if pattern.is_empty() {
        let boundaries = checked_add(input.bytes, input.spans, "empty-pattern boundaries")?;
        let inserted = checked_mul(boundaries, content.len(), "empty-pattern replacement bytes")?;
        return checked_add(input.bytes, inserted, "empty-pattern decoded bytes");
    }
    let expansion = content.len().saturating_sub(pattern.len());
    if expansion == 0 {
        return Ok(input.bytes);
    }
    let matches = input.bytes / pattern.len();
    checked_add(
        input.bytes,
        checked_mul(matches, expansion, "literal replacement expansion")?,
        "literal replacement bytes",
    )
}

fn regex_replacement_bound(input: Shape, content: &str) -> Result<usize> {
    let matches = checked_add(input.bytes, input.spans, "regex match count")?;
    let replacements = checked_mul(matches, content.len(), "regex replacement bytes")?;
    checked_add(input.bytes, replacements, "regex decoded bytes")
}

fn apply_bpe(source: &TokenBuffer, target: &mut TokenBuffer, suffix: &str) -> Result<()> {
    let last = source.spans.len().saturating_sub(1);
    for (index, span) in source.spans.iter().copied().enumerate() {
        let token = source.token(span)?;
        let replacement = if index == last { "" } else { " " };
        replace_literal_token(token, suffix, replacement, target)?;
    }
    Ok(())
}

fn apply_wordpiece(source: &TokenBuffer, target: &mut TokenBuffer, prefix: &str) -> Result<()> {
    for (index, span) in source.spans.iter().copied().enumerate() {
        let token = source.token(span)?;
        if index == 0 {
            target.push_str(token)?;
        } else if let Some(continuation) = token.strip_prefix(prefix) {
            target.push_str(continuation)?;
        } else {
            let start = target.begin_token()?;
            target.append_str(" ")?;
            target.append_str(token)?;
            target.finish_token(start)?;
        }
    }
    Ok(())
}

fn apply_metaspace(
    source: &TokenBuffer,
    target: &mut TokenBuffer,
    replacement: char,
    trim_first: bool,
) -> Result<()> {
    for (index, span) in source.spans.iter().copied().enumerate() {
        let token = source.token(span)?;
        let start = target.begin_token()?;
        for character in token.chars() {
            if character == replacement {
                if index != 0 || !trim_first {
                    target.append_str(" ")?;
                }
            } else {
                target.append_char(character)?;
            }
        }
        target.finish_token(start)?;
    }
    Ok(())
}

fn apply_deduplicate(source: &TokenBuffer, target: &mut TokenBuffer) -> Result<()> {
    let mut previous = None;
    for span in source.spans.iter().copied() {
        let token = source.token(span)?;
        if previous != Some(token) {
            target.push_str(token)?;
        }
        previous = Some(token);
    }
    Ok(())
}

fn apply_filter_empty(source: &TokenBuffer, target: &mut TokenBuffer) -> Result<()> {
    for span in source.spans.iter().copied() {
        let token = source.token(span)?;
        if !token.is_empty() {
            target.push_str(token)?;
        }
    }
    Ok(())
}

fn apply_literal_replace(
    source: &TokenBuffer,
    target: &mut TokenBuffer,
    pattern: &str,
    content: &str,
    empty_input_matches: bool,
) -> Result<()> {
    for span in source.spans.iter().copied() {
        let token = source.token(span)?;
        if token.is_empty() && !empty_input_matches {
            target.push_str("")?;
        } else {
            replace_literal_token(token, pattern, content, target)?;
        }
    }
    Ok(())
}

fn replace_literal_token(
    token: &str,
    pattern: &str,
    content: &str,
    target: &mut TokenBuffer,
) -> Result<()> {
    let token_start = target.begin_token()?;
    let mut previous = 0;
    for (start, matched) in token.match_indices(pattern) {
        target.append_str(text_range(token, previous, start)?)?;
        target.append_str(content)?;
        previous = checked_add(start, matched.len(), "literal match end")?;
    }
    target.append_str(text_range(token, previous, token.len())?)?;
    target.finish_token(token_start)
}

fn apply_regex_replace(
    source: &TokenBuffer,
    target: &mut TokenBuffer,
    regex: &SysRegex,
    content: &str,
) -> Result<()> {
    for span in source.spans.iter().copied() {
        let token = source.token(span)?;
        if token.is_empty() {
            target.push_str("")?;
            continue;
        }
        let token_start = target.begin_token()?;
        let mut previous = 0;
        for (start, end) in regex.find_iter(token) {
            target.append_str(text_range(token, previous, start)?)?;
            target.append_str(content)?;
            previous = end;
        }
        target.append_str(text_range(token, previous, token.len())?)?;
        target.finish_token(token_start)?;
    }
    Ok(())
}

fn apply_fuse(source: &TokenBuffer, target: &mut TokenBuffer) -> Result<()> {
    let start = target.begin_token()?;
    for span in source.spans.iter().copied() {
        target.append_str(source.token(span)?)?;
    }
    target.finish_token(start)
}

fn apply_strip(
    source: &TokenBuffer,
    target: &mut TokenBuffer,
    content: char,
    start_limit: usize,
    stop_limit: usize,
) -> Result<()> {
    for span in source.spans.iter().copied() {
        let token = source.token(span)?;
        let start = strip_start(token, content, start_limit);
        let stop = strip_stop(token, content, stop_limit);
        let bounded_start = start.min(stop);
        target.push_str(text_range(token, bounded_start, stop)?)?;
    }
    Ok(())
}

fn strip_start(token: &str, content: char, limit: usize) -> usize {
    let mut boundary = 0;
    for (removed, (offset, character)) in token.char_indices().enumerate() {
        if removed == limit || character != content {
            break;
        }
        boundary = offset + character.len_utf8();
    }
    boundary
}

fn strip_stop(token: &str, content: char, limit: usize) -> usize {
    let mut boundary = token.len();
    for (removed, (offset, character)) in token.char_indices().rev().enumerate() {
        if removed == limit || character != content {
            break;
        }
        boundary = offset;
    }
    boundary
}

fn apply_byte_fallback(
    source: &TokenBuffer,
    target: &mut TokenBuffer,
    raw: &mut Vec<u8>,
    raw_limit: usize,
) -> Result<()> {
    raw.clear();
    for span in source.spans.iter().copied() {
        let token = source.token(span)?;
        if let Some(byte) = fallback_byte(token) {
            push_raw(raw, byte, raw_limit)?;
        } else {
            flush_fallback(raw, target)?;
            target.push_str(token)?;
        }
    }
    flush_fallback(raw, target)
}

fn fallback_byte(token: &str) -> Option<u8> {
    if token.len() != 6 || !token.starts_with("<0x") || !token.ends_with('>') {
        return None;
    }
    u8::from_str_radix(token.get(3..5)?, 16).ok()
}

fn flush_fallback(raw: &mut Vec<u8>, target: &mut TokenBuffer) -> Result<()> {
    if raw.is_empty() {
        return Ok(());
    }
    match std::str::from_utf8(raw) {
        Ok(decoded) => target.push_str(decoded)?,
        Err(_) => {
            for _ in 0..raw.len() {
                target.push_str(REPLACEMENT_CHARACTER)?;
            }
        }
    }
    raw.clear();
    Ok(())
}

fn apply_byte_level(
    source: &TokenBuffer,
    target: &mut TokenBuffer,
    raw: &mut Vec<u8>,
    raw_limit: usize,
) -> Result<()> {
    raw.clear();
    for span in source.spans.iter().copied() {
        let token = source.token(span)?;
        if token
            .chars()
            .all(|character| byte_level_byte(character).is_some())
        {
            for character in token.chars() {
                let byte = byte_level_byte(character).ok_or_else(|| {
                    DecoderProgramSnafu {
                        message: "byte-level character map changed during one token".to_owned(),
                    }
                    .build()
                })?;
                push_raw(raw, byte, raw_limit)?;
            }
        } else {
            append_raw(raw, token.as_bytes(), raw_limit)?;
        }
    }
    write_lossy_utf8(raw, target)
}

fn byte_level_byte(character: char) -> Option<u8> {
    let codepoint = u32::from(character);
    match codepoint {
        BYTE_LEVEL_DIRECT_ASCII_START..=BYTE_LEVEL_DIRECT_ASCII_END
        | BYTE_LEVEL_DIRECT_LATIN_START..=BYTE_LEVEL_DIRECT_LATIN_FIRST_END
        | BYTE_LEVEL_DIRECT_LATIN_SECOND_START..=BYTE_LEVEL_DIRECT_LATIN_END => {
            u8::try_from(codepoint).ok()
        }
        BYTE_LEVEL_FIRST_EXTRA_START..=BYTE_LEVEL_FIRST_EXTRA_END => {
            u8::try_from(codepoint.checked_sub(BYTE_LEVEL_FIRST_EXTRA_START)?).ok()
        }
        BYTE_LEVEL_SECOND_EXTRA_START..=BYTE_LEVEL_SECOND_EXTRA_END => {
            u8::try_from(codepoint.checked_sub(BYTE_LEVEL_SECOND_OFFSET)?).ok()
        }
        BYTE_LEVEL_LAST_EXTRA => Some(u8::try_from(BYTE_LEVEL_DIRECT_LATIN_GAP).ok()?),
        _ => None,
    }
}

fn write_lossy_utf8(raw: &[u8], target: &mut TokenBuffer) -> Result<()> {
    let token_start = target.begin_token()?;
    let mut remaining = raw;
    while !remaining.is_empty() {
        match std::str::from_utf8(remaining) {
            Ok(valid) => {
                target.append_str(valid)?;
                remaining = &[];
            }
            Err(error) => {
                let valid_bytes = remaining
                    .get(..error.valid_up_to())
                    .ok_or_else(|| utf8_error(raw))?;
                let valid = std::str::from_utf8(valid_bytes).context(DecodeUtf8InvariantSnafu)?;
                target.append_str(valid)?;
                target.append_str(REPLACEMENT_CHARACTER)?;
                remaining = remaining_after_error(remaining, error)?;
            }
        }
    }
    target.finish_token(token_start)
}

fn remaining_after_error(remaining: &[u8], error: std::str::Utf8Error) -> Result<&[u8]> {
    let Some(error_length) = error.error_len() else {
        return Ok(&[]);
    };
    let consumed = checked_add(error.valid_up_to(), error_length, "lossy UTF-8 progress")?;
    remaining
        .get(consumed..)
        .ok_or_else(|| utf8_error(remaining))
}

fn write_output(
    source: &TokenBuffer,
    join_with_spaces: bool,
    output: &mut String,
    output_limit: usize,
) -> Result<()> {
    let separators = if join_with_spaces {
        source.spans.len().saturating_sub(1)
    } else {
        0
    };
    let actual = checked_add(source.bytes.len(), separators, "final decoded output bytes")?;
    if actual > output_limit {
        return DecodedByteLimitExceededSnafu {
            limit: output_limit,
            actual,
        }
        .fail();
    }
    ensure_storage(actual, output.capacity(), "retained-output")?;
    output.clear();
    for (index, span) in source.spans.iter().copied().enumerate() {
        if join_with_spaces && index != 0 {
            output.push(' ');
        }
        output.push_str(source.token(span)?);
    }
    Ok(())
}

fn push_raw(raw: &mut Vec<u8>, byte: u8, raw_limit: usize) -> Result<()> {
    let needed = checked_add(raw.len(), 1, "raw-byte scratch length")?;
    ensure_storage(needed, raw_limit.min(raw.capacity()), "raw-byte")?;
    raw.push(byte);
    Ok(())
}

fn append_raw(raw: &mut Vec<u8>, bytes: &[u8], raw_limit: usize) -> Result<()> {
    let needed = checked_add(raw.len(), bytes.len(), "raw-byte scratch length")?;
    ensure_storage(needed, raw_limit.min(raw.capacity()), "raw-byte")?;
    raw.extend_from_slice(bytes);
    Ok(())
}

fn span_text(bytes: &[u8], span: TokenSpan) -> Result<&str> {
    let selected = bytes
        .get(span.start..span.end)
        .ok_or_else(|| utf8_error(bytes))?;
    std::str::from_utf8(selected).context(DecodeUtf8InvariantSnafu)
}

fn text_range(text: &str, start: usize, end: usize) -> Result<&str> {
    text.get(start..end)
        .ok_or_else(|| utf8_error(text.as_bytes()))
}

fn utf8_error(bytes: &[u8]) -> Error {
    match std::str::from_utf8(bytes) {
        Err(source) => DecodeUtf8InvariantSnafu.into_error(source),
        Ok(_) => decoder_program_error("collective decoder span boundaries are invalid".to_owned()),
    }
}

fn ensure_storage(needed: usize, capacity: usize, target: &'static str) -> Result<()> {
    if needed > capacity {
        return DecodeStorageExhaustedSnafu {
            target,
            needed,
            capacity,
        }
        .fail();
    }
    Ok(())
}

fn checked_add(left: usize, right: usize, target: &'static str) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| DecodePlanOverflowSnafu { target }.build())
}

fn checked_mul(left: usize, right: usize, target: &'static str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| DecodePlanOverflowSnafu { target }.build())
}

fn reserved_vec<T>(capacity: usize, target: &'static str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .context(DecodeStorageAllocationSnafu { target })?;
    Ok(values)
}

fn reserved_string(capacity: usize, target: &'static str) -> Result<String> {
    let mut value = String::new();
    value
        .try_reserve_exact(capacity)
        .context(DecodeStorageAllocationSnafu { target })?;
    Ok(value)
}

fn decoder_program_error(message: String) -> Error {
    DecoderProgramSnafu { message }.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoded_byte_limit_is_checked_before_the_output_buffer_is_written() -> Result<()> {
        let mut source = TokenBuffer::acquire(5, 1)?;
        source.push_str("hello")?;
        let mut output = reserved_string(4, "test retained output")?;

        let result = write_output(&source, false, &mut output, 4);

        assert!(
            matches!(
                result,
                Err(Error::DecodedByteLimitExceeded {
                    limit: 4,
                    actual: 5,
                    ..
                })
            ),
            "the exact completed byte length must be refused"
        );
        assert!(
            output.is_empty(),
            "the retained output buffer must remain unwritten after refusal"
        );
        Ok(())
    }

    #[test]
    fn requested_retained_and_scratch_extents_name_distinct_owners() -> Result<()> {
        let plan = make_storage_plan(2, 7, 11, 2, 5, 12)?;
        let identifier_bytes = 2usize.checked_mul(size_of::<u32>()).ok_or_else(|| {
            DecodePlanOverflowSnafu {
                target: "test identifier bytes",
            }
            .build()
        })?;
        let expected_retained = 7usize.checked_add(identifier_bytes).ok_or_else(|| {
            DecodePlanOverflowSnafu {
                target: "test retained bytes",
            }
            .build()
        })?;

        assert_eq!(
            plan.requested_retained_bytes(),
            expected_retained,
            "retained accounting must contain only output and generated IDs"
        );
        assert_eq!(
            plan.maximum_decoded_bytes(),
            12,
            "the semantic decoded maximum must remain distinct from output capacity"
        );
        assert!(
            plan.requested_scratch_bytes() > plan.requested_retained_bytes(),
            "paired transform arenas and indexes must remain scratch-owned"
        );
        Ok(())
    }
}
