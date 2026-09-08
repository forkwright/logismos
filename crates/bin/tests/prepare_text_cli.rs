//! Process contracts for native preparation without decoder execution.

#![expect(
    clippy::expect_used,
    reason = "process-contract assertions use expect() to give focused test failures"
)]

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use sha2::{Digest, Sha256};
use tempfile::TempDir;
use test_fixtures::{
    Qwen35FixtureConfig, SyntheticGguf, build_qwen35_fixture, raw_qwen35_fixture,
    serialize_raw_gguf,
};

const TOKENS: [&str; 5] = ["[UNK]", "<bos>", "<eos>", "hello", "assistant"];
const BOS_TOKEN_ID: u32 = 1;
const EOS_TOKEN_ID: u32 = 2;
const F32_BYTES: usize = size_of::<f32>();
const HELLO_TOKEN_ID: u32 = 3;

fn command() -> Command {
    Command::new(env!("CARGO_BIN_EXE_logismos"))
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
    }"#
    .to_owned()
}

fn fixture_config(template: &str) -> Qwen35FixtureConfig {
    Qwen35FixtureConfig {
        tokens: TOKENS.iter().map(|token| (*token).to_owned()).collect(),
        bos_token_id: BOS_TOKEN_ID,
        eos_token_id: EOS_TOKEN_ID,
        add_bos: true,
        add_eos: false,
        chat_template: template.to_owned(),
        greedy_token_id: HELLO_TOKEN_ID,
    }
}

struct FixtureInput {
    _directory: TempDir,
    model_path: PathBuf,
    model_digest: String,
    model_bytes: u64,
    tokenizer_path: PathBuf,
    tokenizer_digest: String,
    tokenizer_bytes: usize,
}

fn write_fixture(fixture: SyntheticGguf, tokenizer: String) -> FixtureInput {
    let directory = tempfile::tempdir().expect("temporary directory must be created");
    let model_path = directory.path().join("model.gguf");
    fs::write(&model_path, fixture.bytes).expect("synthetic GGUF must be written");
    let tokenizer_path = directory.path().join("tokenizer.json");
    fs::write(&tokenizer_path, tokenizer.as_bytes()).expect("synthetic tokenizer must be written");
    let tokenizer_sha256: [u8; 32] = Sha256::digest(tokenizer.as_bytes()).into();
    FixtureInput {
        _directory: directory,
        model_path,
        model_digest: hex_digest(&fixture.sha256),
        model_bytes: fixture.byte_len,
        tokenizer_path,
        tokenizer_digest: hex_digest(&tokenizer_sha256),
        tokenizer_bytes: tokenizer.len(),
    }
}

fn standard_fixture(template: &str) -> FixtureInput {
    write_fixture(
        build_qwen35_fixture(&fixture_config(template)).expect("fixture must serialize"),
        tokenizer_json(),
    )
}

fn nan_embedding_fixture() -> FixtureInput {
    let mut raw = raw_qwen35_fixture(&fixture_config("hello")).expect("raw fixture must build");
    let embedding = raw
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == "token_embd.weight")
        .expect("raw fixture must include token embedding");
    let hidden = usize::try_from(embedding.dims[0]).expect("hidden width must fit usize");
    let hello_row = usize::try_from(HELLO_TOKEN_ID).expect("hello token id must fit usize");
    let row_start = hello_row
        .checked_mul(hidden)
        .and_then(|elements| elements.checked_mul(F32_BYTES))
        .expect("consumed embedding row offset must fit usize");
    let row_end = row_start
        .checked_add(F32_BYTES)
        .expect("consumed embedding row end must fit usize");
    let poisoned = embedding
        .payload
        .get_mut(row_start..row_end)
        .expect("consumed embedding row must fit its declared payload");
    poisoned.copy_from_slice(&f32::NAN.to_le_bytes());
    write_fixture(
        serialize_raw_gguf(&raw).expect("mutated fixture must serialize"),
        tokenizer_json(),
    )
}

fn request_json(context_tokens: usize) -> String {
    format!(
        concat!(
            "{{\"messages\":[{{\"role\":\"user\",\"content\":\"hello\"}}],",
            "\"max_output_tokens\":1,\"enable_thinking\":false,\"limits\":{{",
            "\"template_bytes\":4096,\"messages\":4,\"message_bytes\":128,",
            "\"prompt_bytes\":256,\"rendered_bytes\":256,\"context_tokens\":{context_tokens},",
            "\"output_tokens\":2,\"output_bytes\":128,\"template_fuel\":10000,",
            "\"template_recursion\":16}}}}"
        ),
        context_tokens = context_tokens
    )
}

fn run(
    fixture: &FixtureInput,
    model_digest: &str,
    model_bytes: u64,
    tokenizer_digest: &str,
    tokenizer_bytes: usize,
    request: &str,
) -> Output {
    command()
        .args([
            "prepare-text",
            "--model",
            as_utf8_path(&fixture.model_path),
            "--model-sha256",
            model_digest,
            "--model-bytes",
            &model_bytes.to_string(),
            "--tokenizer",
            as_utf8_path(&fixture.tokenizer_path),
            "--tokenizer-sha256",
            tokenizer_digest,
            "--tokenizer-bytes",
            &tokenizer_bytes.to_string(),
            "--request-json",
            request,
        ])
        .output()
        .expect("prepare-text binary must run")
}

fn as_utf8_path(path: &Path) -> &str {
    path.to_str().expect("temporary path must be UTF-8")
}

fn json(output: &Output) -> serde_json::Value {
    serde_json::from_slice(&output.stdout).expect("prepare-text output must be JSON")
}

fn assert_error(output: &Output, kind: &str) {
    assert_eq!(
        output.status.code(),
        Some(2),
        "prepare-text refusals must use the typed command error exit"
    );
    let receipt = json(output);
    assert_eq!(receipt["schema_version"], 1);
    assert_eq!(receipt["outcome"], "error");
    assert_eq!(receipt["command"], "prepare-text");
    assert_eq!(receipt["kind"], kind);
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    let mut output = String::new();
    for byte in bytes {
        use std::fmt::Write;
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[test]
fn prepare_text_reports_exact_rendering_ids_identities_and_plan() {
    let fixture = standard_fixture("hello");
    let request = request_json(8);
    let output = run(
        &fixture,
        &fixture.model_digest,
        fixture.model_bytes,
        &fixture.tokenizer_digest,
        fixture.tokenizer_bytes,
        &request,
    );
    assert_eq!(
        output.status.code(),
        Some(0),
        "preparation-only fixture must succeed"
    );
    let receipt = json(&output);
    assert_eq!(receipt["schema_version"], 1);
    assert_eq!(receipt["outcome"], "text_prepared");
    assert_eq!(receipt["model"]["sha256"], fixture.model_digest);
    assert_eq!(receipt["model"]["bytes"], fixture.model_bytes);
    assert_eq!(receipt["tokenizer"]["sha256"], fixture.tokenizer_digest);
    assert_eq!(
        receipt["tokenizer"]["bytes"],
        u64::try_from(fixture.tokenizer_bytes).expect("fixture length must fit u64")
    );
    assert_eq!(receipt["rendered_prompt"], "hello");
    assert_eq!(receipt["prompt_token_ids"], serde_json::json!([1, 3]));
    assert_eq!(receipt["max_output_tokens"], 1);
    assert_eq!(
        receipt["decoder_cpu_requirements"]["selection"],
        "last_token"
    );
    assert_eq!(
        receipt["decoder_cpu_requirements"]["logical_f32_scope"],
        "decoder-only; excludes artifact backing, prompt IDs, rendered text, template, tokenizer, RSS, and GPU memory"
    );
    assert_eq!(receipt["decoder_cpu_requirements"]["max_context"], 3);
    assert_eq!(receipt["decoder_cpu_requirements"]["max_step_tokens"], 2);
}

#[test]
fn prepare_text_refuses_model_and_tokenizer_identity_or_length_mismatches() {
    let fixture = standard_fixture("hello");
    let request = request_json(8);
    let wrong_digest = "00".repeat(32);
    assert_error(
        &run(
            &fixture,
            &wrong_digest,
            fixture.model_bytes,
            &fixture.tokenizer_digest,
            fixture.tokenizer_bytes,
            &request,
        ),
        "model_identity_mismatch",
    );
    assert_error(
        &run(
            &fixture,
            &fixture.model_digest,
            fixture.model_bytes - 1,
            &fixture.tokenizer_digest,
            fixture.tokenizer_bytes,
            &request,
        ),
        "model_byte_limit",
    );
    assert_error(
        &run(
            &fixture,
            &fixture.model_digest,
            fixture.model_bytes + 1,
            &fixture.tokenizer_digest,
            fixture.tokenizer_bytes,
            &request,
        ),
        "model_identity_mismatch",
    );
    assert_error(
        &run(
            &fixture,
            &fixture.model_digest,
            fixture.model_bytes,
            &wrong_digest,
            fixture.tokenizer_bytes,
            &request,
        ),
        "tokenizer_identity_mismatch",
    );
    assert_error(
        &run(
            &fixture,
            &fixture.model_digest,
            fixture.model_bytes,
            &fixture.tokenizer_digest,
            fixture.tokenizer_bytes - 1,
            &request,
        ),
        "tokenizer_identity_mismatch",
    );
    assert_error(
        &run(
            &fixture,
            &fixture.model_digest,
            fixture.model_bytes,
            &fixture.tokenizer_digest,
            fixture.tokenizer_bytes + 1,
            &request,
        ),
        "tokenizer_identity_mismatch",
    );
}

#[test]
fn prepare_text_strictly_refuses_unknown_request_fields_roles_and_configured_tokenizers() {
    let fixture = standard_fixture("hello");
    let request = request_json(8);
    let unknown_field = request.replacen(
        "\"max_output_tokens\"",
        "\"unknown\":0,\"max_output_tokens\"",
        1,
    );
    assert_error(
        &run(
            &fixture,
            &fixture.model_digest,
            fixture.model_bytes,
            &fixture.tokenizer_digest,
            fixture.tokenizer_bytes,
            &unknown_field,
        ),
        "invalid_request",
    );
    let unknown_role = request.replace("\"role\":\"user\"", "\"role\":\"tool\"");
    assert_error(
        &run(
            &fixture,
            &fixture.model_digest,
            fixture.model_bytes,
            &fixture.tokenizer_digest,
            fixture.tokenizer_bytes,
            &unknown_role,
        ),
        "invalid_request",
    );
    let zero_message_limit = request.replace("\"messages\":4", "\"messages\":0");
    assert_error(
        &run(
            &fixture,
            &fixture.model_digest,
            fixture.model_bytes,
            &fixture.tokenizer_digest,
            fixture.tokenizer_bytes,
            &zero_message_limit,
        ),
        "request_limit",
    );

    let tokenizer = tokenizer_json().replace(
        "\"padding\":null",
        "\"padding\":{\"strategy\":\"BatchLongest\",\"direction\":\"Right\",\"pad_to_multiple_of\":null,\"pad_id\":0,\"pad_type_id\":0,\"pad_token\":\"[UNK]\"}",
    );
    let configured = write_fixture(
        build_qwen35_fixture(&fixture_config("hello")).expect("fixture must serialize"),
        tokenizer,
    );
    assert_error(
        &run(
            &configured,
            &configured.model_digest,
            configured.model_bytes,
            &configured.tokenizer_digest,
            configured.tokenizer_bytes,
            &request,
        ),
        "tokenizer_configuration",
    );
}

#[test]
fn prepare_text_refuses_actual_artifact_context_and_never_executes_nan_weights() {
    let oversized = standard_fixture("hello hello hello hello hello hello hello hello");
    let oversized_request = request_json(10);
    assert_error(
        &run(
            &oversized,
            &oversized.model_digest,
            oversized.model_bytes,
            &oversized.tokenizer_digest,
            oversized.tokenizer_bytes,
            &oversized_request,
        ),
        "decoder_plan_refused",
    );

    let inert_nan = nan_embedding_fixture();
    let request = request_json(8);
    let output = run(
        &inert_nan,
        &inert_nan.model_digest,
        inert_nan.model_bytes,
        &inert_nan.tokenizer_digest,
        inert_nan.tokenizer_bytes,
        &request,
    );
    assert_eq!(
        output.status.code(),
        Some(0),
        "preparation must not consume malformed numerical payloads"
    );
    assert_eq!(
        json(&output)["prompt_token_ids"],
        serde_json::json!([BOS_TOKEN_ID, HELLO_TOKEN_ID]),
        "preparation must retain the exact prompt whose consumed hello row is poisoned"
    );
}

#[test]
fn prepare_text_errors_do_not_disclose_input_paths() {
    let fixture = standard_fixture("hello");
    let absent = fixture._directory.path().join("not-present.gguf");
    let request = request_json(8);
    let output = command()
        .args([
            "prepare-text",
            "--model",
            as_utf8_path(&absent),
            "--model-sha256",
            &fixture.model_digest,
            "--model-bytes",
            &fixture.model_bytes.to_string(),
            "--tokenizer",
            as_utf8_path(&fixture.tokenizer_path),
            "--tokenizer-sha256",
            &fixture.tokenizer_digest,
            "--tokenizer-bytes",
            &fixture.tokenizer_bytes.to_string(),
            "--request-json",
            &request,
        ])
        .output()
        .expect("prepare-text binary must run");
    assert_error(&output, "unreadable_model");
    let output_text = String::from_utf8(output.stdout).expect("receipt must be UTF-8");
    assert!(
        !output_text.contains(as_utf8_path(&absent)),
        "structured error receipt must not disclose the model path"
    );
    assert!(
        output.stderr.is_empty(),
        "prepare-text failures must not render source errors to stderr"
    );
}
