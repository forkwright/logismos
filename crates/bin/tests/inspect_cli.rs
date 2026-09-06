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

fn v1_receipt() -> String {
    let mut receipt = String::from(concat!(
        "{\"schema_version\":1,\"outcome\":\"inspection\",\"format\":\"gguf-v3\",",
        "\"computed_digest\":{\"algorithm\":\"sha256\",\"hex\":\""
    ));
    receipt.push_str(KNOWN_FIXTURE_SHA256);
    receipt.push_str(concat!(
        "\"},\"file_bytes\":140,\"tensor_count\":1,\"model\":{",
        "\"architecture\":null,\"name\":null,\"file_type\":null,",
        "\"quantization_version\":null},\"type_census\":[{\"ggml_type\":\"F32\",",
        "\"tensor_count\":1,\"logical_elements\":3,\"serialized_bytes\":12}]}\n"
    ));
    receipt
}

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

fn append_metadata_header(bytes: &mut Vec<u8>, key: &str, type_id: u32) {
    append_string(bytes, key);
    bytes.extend_from_slice(&type_id.to_le_bytes());
}

fn full_metadata_fixture_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&13u64.to_le_bytes());

    append_metadata_header(&mut bytes, "z.u8", 0);
    bytes.push(u8::MAX);
    append_metadata_header(&mut bytes, "y.i8", 1);
    bytes.extend_from_slice(&(-1i8).to_le_bytes());
    append_metadata_header(&mut bytes, "x.u16", 2);
    bytes.extend_from_slice(&u16::MAX.to_le_bytes());
    append_metadata_header(&mut bytes, "w.i16", 3);
    bytes.extend_from_slice(&(-2i16).to_le_bytes());
    append_metadata_header(&mut bytes, "v.u32", 4);
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    append_metadata_header(&mut bytes, "u.i32", 5);
    bytes.extend_from_slice(&(-3i32).to_le_bytes());
    append_metadata_header(&mut bytes, "t.f32", 6);
    bytes.extend_from_slice(&0x8000_0000u32.to_le_bytes());
    append_metadata_header(&mut bytes, "s.bool", 7);
    bytes.push(1);
    append_metadata_header(&mut bytes, "r.string", 8);
    append_string(&mut bytes, "deliberate metadata output");
    append_metadata_header(&mut bytes, "q.array", 9);
    bytes.extend_from_slice(&6u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&0x7fc0_1234u32.to_le_bytes());
    append_metadata_header(&mut bytes, "p.u64", 10);
    bytes.extend_from_slice(&u64::MAX.to_le_bytes());
    append_metadata_header(&mut bytes, "o.i64", 11);
    bytes.extend_from_slice(&i64::MIN.to_le_bytes());
    append_metadata_header(&mut bytes, "a.f64", 12);
    bytes.extend_from_slice(&0x7ff8_0000_0000_1234u64.to_le_bytes());

    append_string(&mut bytes, "one");
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    let padding = (32 - bytes.len() % 32) % 32;
    bytes.extend(std::iter::repeat_n(0u8, padding));
    bytes.extend_from_slice(&0f32.to_le_bytes());
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

fn metadata_value<'a>(report: &'a serde_json::Value, key: &str) -> &'a serde_json::Value {
    report["metadata"]
        .as_array()
        .and_then(|entries| entries.iter().find(|entry| entry["key"] == key))
        .map(|entry| &entry["value"])
        .expect("metadata report must contain its declared key")
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
    assert_eq!(
        output_text,
        v1_receipt(),
        "no-flag receipt remains v1-exact"
    );
    assert!(
        !output_text.contains(as_utf8_path(&path)),
        "machine receipt must not leak the host input path"
    );
}

#[test]
fn inspect_metadata_emits_deterministic_exactly_typed_values() {
    let directory = tempfile::tempdir().expect("temporary directory must be created");
    let path = write_fixture(
        &directory,
        "all-metadata-types.gguf",
        &full_metadata_fixture_bytes(),
    );
    let no_flag = run(&["inspect", "--input", as_utf8_path(&path)]);
    assert!(no_flag.status.success(), "ordinary inspection succeeds");
    let output = run(&["inspect", "--input", as_utf8_path(&path), "--metadata"]);

    assert!(output.status.success(), "metadata inspection succeeds");
    assert!(
        output.stderr.is_empty(),
        "metadata success must not write stderr"
    );
    let output_text = String::from_utf8(output.stdout).expect("JSON must be UTF-8");
    assert!(
        !output_text.contains(as_utf8_path(&path)),
        "metadata report must not leak the host input path"
    );
    let report: serde_json::Value =
        serde_json::from_str(&output_text).expect("metadata output must be JSON");
    assert_eq!(report["schema_version"], 1);
    assert_eq!(report["outcome"], "inspection_metadata");
    assert_eq!(report["format"], "gguf-v3");
    assert_eq!(
        report["inspection"],
        json(&no_flag),
        "metadata envelope embeds the ordinary inspection receipt"
    );

    let keys: Vec<_> = report["metadata"]
        .as_array()
        .expect("metadata must be an array")
        .iter()
        .map(|entry| entry["key"].as_str().expect("metadata key must be text"))
        .collect();
    assert_eq!(
        keys,
        [
            "a.f64", "o.i64", "p.u64", "q.array", "r.string", "s.bool", "t.f32", "u.i32", "v.u32",
            "w.i16", "x.u16", "y.i8", "z.u8",
        ],
        "metadata entries must sort by key rather than HashMap iteration"
    );
    assert_eq!(metadata_value(&report, "z.u8")["type"], "u8");
    assert_eq!(metadata_value(&report, "y.i8")["type"], "i8");
    assert_eq!(metadata_value(&report, "x.u16")["type"], "u16");
    assert_eq!(metadata_value(&report, "w.i16")["type"], "i16");
    assert_eq!(metadata_value(&report, "v.u32")["type"], "u32");
    assert_eq!(metadata_value(&report, "u.i32")["type"], "i32");
    assert_eq!(metadata_value(&report, "t.f32")["type"], "f32");
    assert_eq!(metadata_value(&report, "s.bool")["type"], "bool");
    assert_eq!(metadata_value(&report, "r.string")["type"], "string");
    assert_eq!(metadata_value(&report, "q.array")["type"], "array");
    assert_eq!(metadata_value(&report, "p.u64")["type"], "u64");
    assert_eq!(metadata_value(&report, "o.i64")["type"], "i64");
    assert_eq!(metadata_value(&report, "a.f64")["type"], "f64");
    assert_eq!(metadata_value(&report, "z.u8")["value"], u8::MAX);
    assert_eq!(metadata_value(&report, "y.i8")["value"], -1);
    assert_eq!(metadata_value(&report, "x.u16")["value"], u16::MAX);
    assert_eq!(metadata_value(&report, "w.i16")["value"], -2);
    assert_eq!(metadata_value(&report, "v.u32")["value"], u32::MAX);
    assert_eq!(metadata_value(&report, "u.i32")["value"], -3);
    assert_eq!(metadata_value(&report, "s.bool")["value"], true);
    assert_eq!(
        metadata_value(&report, "r.string")["value"],
        "deliberate metadata output"
    );
    assert_eq!(metadata_value(&report, "p.u64")["value"], u64::MAX);
    assert_eq!(metadata_value(&report, "o.i64")["value"], i64::MIN);
    assert_eq!(metadata_value(&report, "t.f32")["bits"], "80000000");
    assert_eq!(metadata_value(&report, "a.f64")["bits"], "7ff8000000001234");
    assert!(
        metadata_value(&report, "t.f32").get("value").is_none(),
        "negative zero is represented by IEEE bits rather than a lossy JSON number"
    );
    let array_values = metadata_value(&report, "q.array")["values"]
        .as_array()
        .expect("array metadata must expose tagged values");
    assert_eq!(array_values[0]["type"], "f32");
    assert_eq!(array_values[0]["bits"], "7fc01234");
}

#[test]
fn inspect_argument_errors_are_typed_and_exit_two() {
    for arguments in [
        ["inspect"].as_slice(),
        ["inspect", "--input", "-"].as_slice(),
        ["inspect", "--unknown", "fixture.gguf"].as_slice(),
        ["inspect", "--metadata", "--input", "fixture.gguf"].as_slice(),
        ["inspect", "--input", "--metadata"].as_slice(),
        [
            "inspect",
            "--input",
            "fixture.gguf",
            "--metadata",
            "--metadata",
        ]
        .as_slice(),
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
