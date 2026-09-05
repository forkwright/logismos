//! Process-level contracts for the CPU-only `logismos inspect` CLI.

#![expect(
    clippy::expect_used,
    reason = "process-contract assertions use expect() to give focused test failures"
)]

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use tempfile::TempDir;

const KNOWN_FIXTURE_SHA256: &str =
    "623d94e17734e71bc68433a1f9121ae9b59f4aabc33fdf74f7b5cc62b61c3980";

fn command() -> Command {
    Command::new(env!("CARGO_BIN_EXE_logismos"))
}

fn run(arguments: &[&str]) -> Output {
    command()
        .args(arguments)
        .output()
        .expect("inspect binary must run")
}

fn append_string(bytes: &mut Vec<u8>, value: &str) {
    let length = u64::try_from(value.len()).expect("fixture string length must fit u64");
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn fixture_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&2u64.to_le_bytes());

    append_string(&mut bytes, "answer");
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&42u32.to_le_bytes());
    append_string(&mut bytes, "general.alignment");
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&32u32.to_le_bytes());

    append_string(&mut bytes, "one");
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&3u64.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    let padding = (32 - bytes.len() % 32) % 32;
    bytes.extend(std::iter::repeat_n(0u8, padding));
    for value in [1.0_f32, 2.0, 3.0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn write_fixture(directory: &TempDir, filename: &str, contents: &[u8]) -> PathBuf {
    let path = directory.path().join(filename);
    fs::write(&path, contents).expect("fixture must be written");
    path
}

fn as_utf8_path(path: &Path) -> &str {
    path.to_str().expect("temporary path must be UTF-8")
}

fn json(output: &Output) -> serde_json::Value {
    serde_json::from_slice(&output.stdout).expect("inspect output must be JSON")
}

fn assert_typed_error(output: &Output, kind: &str) {
    assert_eq!(output.status.code(), Some(2), "inspect failure exits two");
    assert!(
        output.stderr.is_empty(),
        "inspection errors are stdout JSON only"
    );
    let receipt = json(output);
    assert_eq!(receipt["schema_version"], 1);
    assert_eq!(receipt["outcome"], "error");
    assert_eq!(receipt["command"], "inspect");
    assert_eq!(receipt["kind"], kind);
}

#[test]
fn inspect_emits_a_bounded_path_free_receipt_for_gguf_bytes() {
    let directory = tempfile::tempdir().expect("temporary directory must be created");
    let path = write_fixture(&directory, "not-a-model-extension.data", &fixture_bytes());
    let output = run(&["inspect", "--input", as_utf8_path(&path)]);

    assert!(output.status.success(), "valid GGUF inspection succeeds");
    assert!(output.stderr.is_empty(), "success must not write stderr");
    let receipt = json(&output);
    assert_eq!(receipt["schema_version"], 1);
    assert_eq!(receipt["outcome"], "inspection");
    assert_eq!(receipt["format"], "gguf-v3");
    assert_eq!(receipt["computed_digest"]["algorithm"], "sha256");
    assert_eq!(receipt["computed_digest"]["hex"], KNOWN_FIXTURE_SHA256);
    assert_eq!(receipt["tensor_count"], 1);
    assert_eq!(receipt["type_census"][0]["ggml_type"], "F32");
    assert_eq!(receipt["type_census"][0]["serialized_bytes"], 12);
    let output_text = String::from_utf8(output.stdout).expect("JSON must be UTF-8");
    assert!(
        !output_text.contains(as_utf8_path(&path)),
        "machine receipt must not leak the host input path"
    );
}

#[test]
fn inspect_argument_errors_are_typed_and_exit_two() {
    for arguments in [
        ["inspect"].as_slice(),
        ["inspect", "--input", "-"].as_slice(),
        ["inspect", "--unknown", "fixture.gguf"].as_slice(),
    ] {
        assert_typed_error(&run(arguments), "invalid_arguments");
    }
}

#[test]
fn inspect_fails_closed_for_malformed_and_over_limit_metadata() {
    let directory = tempfile::tempdir().expect("temporary directory must be created");
    let truncated = write_fixture(&directory, "misleading.gguf", b"GGUF");
    let truncated_output = run(&["inspect", "--input", as_utf8_path(&truncated)]);
    assert_typed_error(&truncated_output, "invalid_gguf");

    let mut over_limit = Vec::new();
    over_limit.extend_from_slice(b"GGUF");
    over_limit.extend_from_slice(&3u32.to_le_bytes());
    over_limit.extend_from_slice(&0u64.to_le_bytes());
    over_limit.extend_from_slice(&100_001u64.to_le_bytes());
    let bounded = write_fixture(&directory, "metadata-count.data", &over_limit);
    let bounded_output = run(&["inspect", "--input", as_utf8_path(&bounded)]);
    assert_typed_error(&bounded_output, "invalid_gguf");

    let mut unsupported_type = Vec::new();
    unsupported_type.extend_from_slice(b"GGUF");
    unsupported_type.extend_from_slice(&3u32.to_le_bytes());
    unsupported_type.extend_from_slice(&1u64.to_le_bytes());
    unsupported_type.extend_from_slice(&0u64.to_le_bytes());
    append_string(&mut unsupported_type, "unknown-type");
    unsupported_type.extend_from_slice(&1u32.to_le_bytes());
    unsupported_type.extend_from_slice(&1u64.to_le_bytes());
    unsupported_type.extend_from_slice(&999u32.to_le_bytes());
    unsupported_type.extend_from_slice(&0u64.to_le_bytes());
    let unsupported = write_fixture(&directory, "unsupported-type.data", &unsupported_type);
    let unsupported_output = run(&["inspect", "--input", as_utf8_path(&unsupported)]);
    assert_typed_error(&unsupported_output, "invalid_gguf");

    let missing = directory.path().join("does-not-exist.gguf");
    let missing_output = run(&["inspect", "--input", as_utf8_path(&missing)]);
    assert_typed_error(&missing_output, "unreadable_input");
    let missing_json = String::from_utf8(missing_output.stdout).expect("JSON must be UTF-8");
    assert!(
        !missing_json.contains(as_utf8_path(&missing)),
        "error receipt must not leak the host input path"
    );
}
