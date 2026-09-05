//! Process-level contracts for the `logismos plan` CLI.

use std::{
    io::Write,
    process::{Command, Output, Stdio},
};

use tempfile::NamedTempFile;

const SUCCESS_INPUT: &str = include_str!("fixtures/plan-success.json");
const REFUSAL_INPUT: &str = include_str!("fixtures/plan-refusal.json");

fn command() -> Command {
    Command::new(env!("CARGO_BIN_EXE_logismos"))
}

fn run_with_stdin(input: &str, arguments: &[&str]) -> Output {
    let mut process = command()
        .args(arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("test binary must spawn");
    let mut stdin = process.stdin.take().expect("piped stdin must be available");
    stdin
        .write_all(input.as_bytes())
        .expect("test input must write to child stdin");
    drop(stdin);
    process.wait_with_output().expect("test binary must exit")
}

fn assert_success(output: &Output) {
    assert!(output.status.success(), "plan must succeed: {output:?}");
    assert!(output.stderr.is_empty(), "success must not write stderr");
    let result: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("success output must be JSON");
    assert_eq!(
        result["outcome"], "plan",
        "success must emit a plan outcome"
    );
}

fn write_fixture(contents: &str) -> NamedTempFile {
    let mut fixture = NamedTempFile::new().expect("temporary fixture must be created");
    fixture
        .write_all(contents.as_bytes())
        .expect("temporary fixture must be written");
    fixture
}

#[test]
fn plan_accepts_stdin_file_and_explicit_dash_deterministically() {
    let stdin = run_with_stdin(SUCCESS_INPUT, &["plan"]);
    assert_success(&stdin);

    let fixture = write_fixture(SUCCESS_INPUT);
    let path = fixture
        .path()
        .to_str()
        .expect("temporary path must be UTF-8");
    let from_file = command()
        .args(["plan", "--input", path])
        .output()
        .expect("test binary must run with file input");
    assert_success(&from_file);

    let explicit_dash = run_with_stdin(SUCCESS_INPUT, &["plan", "--input", "-"]);
    assert_success(&explicit_dash);
    assert_eq!(
        stdin.stdout, from_file.stdout,
        "file input must be deterministic"
    );
    assert_eq!(
        stdin.stdout, explicit_dash.stdout,
        "explicit stdin must be deterministic"
    );
}

#[test]
fn refusal_is_json_exit_one_and_has_no_partial_plan() {
    let output = run_with_stdin(REFUSAL_INPUT, &["plan"]);
    assert_eq!(output.status.code(), Some(1), "refusal must exit one");
    assert!(output.stderr.is_empty(), "refusal must be stdout JSON only");
    let result: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("refusal output must be JSON");
    assert_eq!(result["outcome"], "refusal", "refusal must be typed");
    assert_eq!(
        result["refusal"]["kind"], "capacity_exhausted",
        "transactional refusal must preserve the capacity reason"
    );
    assert!(
        result.get("admitted_placements").is_none(),
        "transactional refusal must not expose a partial plan"
    );
}

#[test]
fn usage_and_io_failures_exit_two() {
    let usage = command().output().expect("test binary must run");
    assert_eq!(usage.status.code(), Some(2), "usage failure must exit two");
    assert_eq!(usage.stderr, b"usage: logismos plan [--input <path>|-]\n");

    let parse = command()
        .args(["plan", "--input"])
        .output()
        .expect("test binary must run");
    assert_eq!(parse.status.code(), Some(2), "parse failure must exit two");
    assert_eq!(
        parse.stderr, usage.stderr,
        "parse and usage failures share one usage message"
    );

    let input = command()
        .args([
            "plan",
            "--input",
            "/definitely-not-a-logismos-plan-input.json",
        ])
        .output()
        .expect("test binary must run");
    assert_eq!(input.status.code(), Some(2), "input failure must exit two");
    assert_eq!(input.stdout, b"", "input failure must not emit JSON");
}
