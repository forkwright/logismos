use std::mem::size_of;

use serde_json::{Map, Value, json};

use super::{Error, Result, Tokenizer};

type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

fn tokenizer_json(tokens: &[&str], decoder: &Value) -> TestResult<Vec<u8>> {
    let mut vocabulary = Map::new();
    for (identifier, spelling) in tokens.iter().enumerate() {
        vocabulary.insert((*spelling).to_owned(), json!(identifier));
    }
    let tokenizer = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": decoder,
        "model": {
            "type": "WordLevel",
            "vocab": vocabulary,
            "unk_token": tokens.first().copied().unwrap_or("[UNK]")
        }
    });
    Ok(serde_json::to_vec(&tokenizer)?)
}

fn bounded_decode(
    tokenizer: &Tokenizer,
    identifiers: &[u32],
    output_limit: usize,
    skip_special_tokens: bool,
) -> Result<(String, Vec<u32>)> {
    let plan = tokenizer.decode_storage_plan(identifiers.len(), output_limit)?;
    let mut storage = plan.acquire()?;
    for identifier in identifiers {
        storage.push_token_id(*identifier)?;
    }
    tokenizer.decode_with_storage(storage, skip_special_tokens)
}

fn assert_decoder(
    tokens: &[&str],
    decoder: &Value,
    identifiers: &[u32],
    expected: &str,
) -> TestResult {
    let bytes = tokenizer_json(tokens, decoder)?;
    let tokenizer = Tokenizer::from_bytes(&bytes)?;
    let upstream = tokenizer.decode(identifiers, false)?;
    let (bounded, retained) = bounded_decode(&tokenizer, identifiers, expected.len(), false)?;

    assert_eq!(
        upstream, expected,
        "the hand-authored semantic expectation must match pinned upstream"
    );
    assert_eq!(
        bounded, expected,
        "the bounded implementation must match the independent expectation"
    );
    assert_eq!(
        bounded, upstream,
        "the bounded implementation must remain differential with pinned upstream"
    );
    assert_eq!(
        retained, identifiers,
        "collective decoding must return the originally retained generated IDs"
    );
    Ok(())
}

#[test]
fn null_decoder_uses_upstream_space_joining() -> TestResult {
    assert_decoder(
        &["[UNK]", "hello", "world"],
        &Value::Null,
        &[1, 2],
        "hello world",
    )
}

#[test]
fn bpe_preserves_suffix_and_empty_suffix_semantics() -> TestResult {
    assert_decoder(
        &["[UNK]", "hello</w>", "world</w>"],
        &json!({"type": "BPEDecoder", "suffix": "</w>"}),
        &[1, 2],
        "hello world",
    )?;
    assert_decoder(
        &["[UNK]", "a", "b"],
        &json!({"type": "BPEDecoder", "suffix": ""}),
        &[1, 2],
        " a b",
    )
}

#[test]
fn byte_level_decodes_split_utf8_only_as_one_sequence() -> TestResult {
    assert_decoder(
        &["[UNK]", "Ã", "©"],
        &json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}),
        &[1, 2],
        "é",
    )
}

#[test]
fn wordpiece_preserves_prefix_and_cleanup_order() -> TestResult {
    assert_decoder(
        &["[UNK]", "hello", "##s", "!"],
        &json!({"type": "WordPiece", "prefix": "##", "cleanup": true}),
        &[1, 2, 3],
        "hellos!",
    )?;
    assert_decoder(
        &["[UNK]", "a", "b"],
        &json!({"type": "WordPiece", "prefix": "", "cleanup": false}),
        &[1, 2],
        "ab",
    )
}

#[test]
fn metaspace_preserves_prepend_scheme_semantics() -> TestResult {
    assert_decoder(
        &["[UNK]", "▁hello", "▁world"],
        &json!({"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true}),
        &[1, 2],
        "hello world",
    )?;
    assert_decoder(
        &["[UNK]", "▁hello", "▁world"],
        &json!({"type": "Metaspace", "replacement": "▁", "prepend_scheme": "never", "split": true}),
        &[1, 2],
        " hello world",
    )
}

#[test]
fn ctc_preserves_dedup_pad_delimiter_and_empty_delimiter_order() -> TestResult {
    assert_decoder(
        &["[UNK]", "<pad>", "h", "|", "i"],
        &json!({"type": "CTC", "pad_token": "<pad>", "word_delimiter_token": "|", "cleanup": true}),
        &[1, 1, 2, 2, 3, 3, 4],
        "h i",
    )?;
    assert_decoder(
        &["[UNK]", ""],
        &json!({"type": "CTC", "pad_token": "<pad>", "word_delimiter_token": "", "cleanup": true}),
        &[1],
        " ",
    )
}

#[test]
fn replace_preserves_literal_regex_empty_and_zero_width_semantics() -> TestResult {
    assert_decoder(
        &["[UNK]", "a_b"],
        &json!({"type": "Replace", "pattern": {"String": "_"}, "content": " "}),
        &[1],
        "a b",
    )?;
    assert_decoder(
        &["[UNK]", "é"],
        &json!({"type": "Replace", "pattern": {"String": ""}, "content": "."}),
        &[1],
        ".é.",
    )?;
    assert_decoder(
        &["[UNK]", ""],
        &json!({"type": "Replace", "pattern": {"String": ""}, "content": "."}),
        &[1],
        "",
    )?;
    assert_decoder(
        &["[UNK]", "a   b"],
        &json!({"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": "_"}),
        &[1],
        "a_b",
    )?;
    assert_decoder(
        &["[UNK]", "ab"],
        &json!({"type": "Replace", "pattern": {"Regex": "(?=b)"}, "content": "_"}),
        &[1],
        "a_b",
    )
}

#[test]
fn fuse_and_strip_preserve_token_boundaries_and_unicode() -> TestResult {
    assert_decoder(
        &["[UNK]", "hello", "world"],
        &json!({"type": "Fuse"}),
        &[1, 2],
        "helloworld",
    )?;
    assert_decoder(
        &["[UNK]", "üühelloüü"],
        &json!({"type": "Strip", "content": "ü", "start": 1, "stop": 2}),
        &[1],
        "ühello",
    )
}

#[test]
fn byte_fallback_preserves_valid_and_final_invalid_runs() -> TestResult {
    assert_decoder(
        &["[UNK]", "<0xC3>", "<0xA9>"],
        &json!({"type": "ByteFallback"}),
        &[1, 2],
        "é",
    )?;
    assert_decoder(
        &["[UNK]", "<0xE5>", "<0x8F>"],
        &json!({"type": "ByteFallback"}),
        &[1, 2],
        "��",
    )
}

#[test]
fn nested_sequences_flatten_without_changing_stage_semantics() -> TestResult {
    let decoder = json!({
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "_"}, "content": " "},
            {"type": "Sequence", "decoders": [
                {"type": "Strip", "content": "x", "start": 1, "stop": 1},
                {"type": "Fuse"}
            ]},
            {"type": "Replace", "pattern": {"Regex": " +"}, "content": "-"}
        ]
    });
    assert_decoder(
        &["[UNK]", "x_hi_x", "xthere_x"],
        &decoder,
        &[1, 2],
        "-hi-there-",
    )
}

#[test]
fn expanding_then_shrinking_sequence_uses_scratch_not_output_headroom() -> TestResult {
    let decoder = json!({
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": ""}, "content": "xyz"},
            {"type": "Replace", "pattern": {"String": "xyz"}, "content": ""}
        ]
    });
    let bytes = tokenizer_json(&["[UNK]", "a"], &decoder)?;
    let tokenizer = Tokenizer::from_bytes(&bytes)?;
    let plan = tokenizer.decode_storage_plan(1, 1)?;

    assert!(
        plan.maximum_decoded_bytes() >= 7,
        "the plan must retain the expanding intermediate's checked maximum"
    );
    assert!(
        plan.requested_scratch_bytes() > plan.requested_retained_bytes(),
        "intermediate growth must be owned by scratch instead of output headroom"
    );
    let (decoded_output, _) = bounded_decode(&tokenizer, &[1], 1, false)?;
    assert_eq!(
        decoded_output, "a",
        "late shrinking must succeed at the exact retained output cap"
    );
    assert_eq!(
        tokenizer.decode(&[1], false)?,
        decoded_output,
        "late shrinking must remain differential with pinned upstream"
    );
    Ok(())
}

#[test]
fn unknown_and_special_ids_preserve_upstream_filtering() -> TestResult {
    let tokenizer = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 2, "content": "<special>", "single_word": false, "lstrip": false,
             "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "a": 1,
                  "<special>": 2, "b": 3}, "unk_token": "[UNK]"}
    });
    let bytes = serde_json::to_vec(&tokenizer)?;
    let tokenizer = Tokenizer::from_bytes(&bytes)?;
    let identifiers = [1, u32::MAX, 2, 3];

    for (skip_special_tokens, expected) in [(false, "a <special> b"), (true, "a b")] {
        let upstream = tokenizer.decode(&identifiers, skip_special_tokens)?;
        let (bounded, retained) = bounded_decode(
            &tokenizer,
            &identifiers,
            expected.len(),
            skip_special_tokens,
        )?;
        assert_eq!(
            upstream, expected,
            "unexpected upstream filtering semantics"
        );
        assert_eq!(
            bounded, upstream,
            "bounded filtering diverged from upstream"
        );
        assert_eq!(
            retained, identifiers,
            "filtering must not mutate retained generated IDs"
        );
    }
    Ok(())
}

#[test]
fn sparse_high_ids_do_not_create_a_dense_max_id_table() -> TestResult {
    const HIGH_ID: u32 = 1_000_000_000;
    let tokenizer = json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [], "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "high": HIGH_ID},
                  "unk_token": "[UNK]"}
    });
    let bytes = serde_json::to_vec(&tokenizer)?;
    let tokenizer = Tokenizer::from_bytes(&bytes)?;
    let (decoded, _) = bounded_decode(&tokenizer, &[HIGH_ID], 4, false)?;

    assert_eq!(
        decoded, "high",
        "the sparse high ID must resolve canonically"
    );
    assert_eq!(
        tokenizer.decode(&[HIGH_ID], false)?,
        decoded,
        "sparse lookup must remain differential with pinned upstream"
    );
    Ok(())
}

#[test]
fn plan_arithmetic_overflow_is_rejected_before_acquisition() -> TestResult {
    let bytes = tokenizer_json(&["[UNK]", "hello"], &Value::Null)?;
    let tokenizer = Tokenizer::from_bytes(&bytes)?;
    let result = tokenizer.decode_storage_plan(usize::MAX, 1);

    assert!(
        matches!(result, Err(Error::DecodePlanOverflow { .. })),
        "checked vocabulary multiplication must reject usize overflow"
    );
    Ok(())
}

#[test]
fn persistent_decoder_containers_are_separate_from_request_storage() -> TestResult {
    let bytes = tokenizer_json(
        &["[UNK]", "a_b"],
        &json!({"type": "Replace", "pattern": {"String": "_"}, "content": " "}),
    )?;
    let tokenizer = Tokenizer::from_bytes(&bytes)?;
    let plan = tokenizer.decode_storage_plan(1, 3)?;

    assert!(
        tokenizer.collective_decoder_resident_bytes()? > 0,
        "packed vocabulary and decoder program must remain attributable residency"
    );
    assert_eq!(
        plan.requested_retained_bytes(),
        3 + size_of::<u32>(),
        "per-request retained bytes must not absorb persistent tokenizer residency"
    );
    Ok(())
}

#[test]
fn impossible_storage_request_fails_fallibly() -> TestResult {
    let bytes = tokenizer_json(&["[UNK]", "a"], &Value::Null)?;
    let tokenizer = Tokenizer::from_bytes(&bytes)?;
    let output_capacity = usize::MAX
        .checked_sub(size_of::<u32>())
        .ok_or_else(|| std::io::Error::other("usize cannot hold one token ID"))?;
    let plan = tokenizer.decode_storage_plan(1, output_capacity)?;

    assert!(
        matches!(plan.acquire(), Err(Error::DecodeStorageAllocation { .. })),
        "an impossible retained allocation must return its fallible allocation error"
    );
    Ok(())
}

#[test]
fn decoded_byte_cap_reports_exact_length_before_publication() -> TestResult {
    let bytes = tokenizer_json(&["[UNK]", "hello"], &Value::Null)?;
    let tokenizer = Tokenizer::from_bytes(&bytes)?;
    let result = bounded_decode(&tokenizer, &[1], 4, false);

    assert!(
        matches!(
            result,
            Err(Error::DecodedByteLimitExceeded {
                limit: 4,
                actual: 5,
                ..
            })
        ),
        "the completed output length must be refused before a result is returned"
    );
    Ok(())
}
