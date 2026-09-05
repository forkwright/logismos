//! # logismos binary
//!
//! The `plan` and `inspect` subcommands are CPU-only: they neither initialise
//! HIP nor start a service. The minimal `cargo build -p bin` dependency graph
//! also excludes HIP; broader workspace feature unification can produce a
//! different linkage graph without changing that command behavior.

use std::{
    env,
    ffi::OsString,
    fmt,
    fs::File,
    io::{self, Read, Write},
    path::Path,
    process::ExitCode,
};

use serde::Serialize;

const USAGE: &str = "usage: logismos plan [--input <path>|-]";
const INSPECTION_USAGE: &str = "usage: logismos inspect --input <path>";
const INSPECTION_SCHEMA_VERSION: u32 = 1;
/// Maximum accepted placement-contract size, preventing unbounded CLI input allocation.
const MAX_PLAN_INPUT_BYTES: usize = 4 * 1024 * 1024;

fn main() -> ExitCode {
    match run() {
        Ok(outcome) => write_outcome(&outcome),
        Err(CliError::Inspection(error)) => write_inspection_error(error),
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<CommandOutcome, CliError> {
    let mut arguments = env::args_os();
    let _program = arguments.next();
    match arguments.next().as_deref() {
        Some(command) if command == "plan" => plan_command(arguments).map(CommandOutcome::Plan),
        Some(command) if command == "inspect" => {
            inspect_command(arguments).map(CommandOutcome::Inspection)
        }
        _ => Err(CliError::Usage),
    }
}

enum CommandOutcome {
    Plan(placement::PlanOutcome),
    Inspection(InspectionReceipt),
}

fn plan_command(
    mut arguments: impl Iterator<Item = OsString>,
) -> Result<placement::PlanOutcome, CliError> {
    let input = match arguments.next().as_deref() {
        None => read_stdin()?,
        Some(argument) if argument == "-" => {
            if arguments.next().is_some() {
                return Err(CliError::Usage);
            }
            read_stdin()?
        }
        Some(argument) if argument == "--input" => {
            let Some(path) = arguments.next() else {
                return Err(CliError::Usage);
            };
            if arguments.next().is_some() {
                return Err(CliError::Usage);
            }
            if path == "-" {
                read_stdin()?
            } else {
                read_file(&path)?
            }
        }
        Some(_) => return Err(CliError::Usage),
    };
    Ok(placement::plan_json(&input))
}

fn inspect_command(
    mut arguments: impl Iterator<Item = OsString>,
) -> Result<InspectionReceipt, CliError> {
    let Some(flag) = arguments.next() else {
        return Err(CliError::Inspection(InspectionError::InvalidArguments));
    };
    if flag != "--input" {
        return Err(CliError::Inspection(InspectionError::InvalidArguments));
    }
    let Some(path) = arguments.next() else {
        return Err(CliError::Inspection(InspectionError::InvalidArguments));
    };
    if path == "-" || arguments.next().is_some() {
        return Err(CliError::Inspection(InspectionError::InvalidArguments));
    }
    let inspection = loader::gguf::inspect_gguf_with_sha256(Path::new(&path))
        .map_err(|error| map_inspection_error(&error))?;
    InspectionReceipt::from_inspection(inspection).map_err(CliError::Inspection)
}

fn read_stdin() -> Result<String, CliError> {
    let stdin = io::stdin();
    read_input(&mut stdin.lock())
}

fn read_file(path: &OsString) -> Result<String, CliError> {
    let mut file =
        File::open(path).map_err(|_| CliError::Input("unable to read placement input"))?;
    read_input(&mut file)
}

fn read_input(reader: &mut impl Read) -> Result<String, CliError> {
    let bounded_length = MAX_PLAN_INPUT_BYTES
        .checked_add(1)
        .ok_or(CliError::Input("unable to bound placement input"))?;
    let bounded_length = u64::try_from(bounded_length)
        .map_err(|_| CliError::Input("unable to bound placement input"))?;
    let mut bytes = Vec::new();
    reader
        .take(bounded_length)
        .read_to_end(&mut bytes)
        .map_err(|_| CliError::Input("unable to read placement input"))?;
    if bytes.len() > MAX_PLAN_INPUT_BYTES {
        return Err(CliError::Input("placement input exceeds the byte limit"));
    }
    String::from_utf8(bytes).map_err(|_| CliError::Input("placement input must be UTF-8"))
}

fn write_outcome(outcome: &CommandOutcome) -> ExitCode {
    match outcome {
        CommandOutcome::Plan(outcome) => write_plan_outcome(outcome),
        CommandOutcome::Inspection(receipt) => write_inspection_receipt(receipt),
    }
}

fn write_plan_outcome(outcome: &placement::PlanOutcome) -> ExitCode {
    let exit_code = if matches!(outcome, placement::PlanOutcome::Plan { .. }) {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    };
    let Ok(output) = serde_json::to_string(outcome) else {
        eprintln!("unable to serialize placement result");
        return ExitCode::from(2);
    };
    if writeln!(io::stdout().lock(), "{output}").is_err() {
        eprintln!("unable to write placement result");
        return ExitCode::from(2);
    }
    exit_code
}

fn write_inspection_receipt(receipt: &InspectionReceipt) -> ExitCode {
    write_inspection_json(
        receipt,
        "unable to serialize inspection receipt",
        ExitCode::SUCCESS,
    )
}

fn write_inspection_error(error: InspectionError) -> ExitCode {
    let receipt = InspectionErrorReceipt {
        schema_version: INSPECTION_SCHEMA_VERSION,
        outcome: "error",
        command: "inspect",
        kind: error.kind(),
    };
    write_inspection_json(
        &receipt,
        "unable to serialize inspection error",
        ExitCode::from(2),
    )
}

fn write_inspection_json(
    value: &impl Serialize,
    serialization_error: &'static str,
    exit_code: ExitCode,
) -> ExitCode {
    let Ok(output) = serde_json::to_string(value) else {
        eprintln!("{serialization_error}");
        return ExitCode::from(2);
    };
    if writeln!(io::stdout().lock(), "{output}").is_err() {
        eprintln!("unable to write inspection result");
        return ExitCode::from(2);
    }
    exit_code
}

fn map_inspection_error(error: &loader::Error) -> CliError {
    let kind = match error {
        loader::Error::Io { .. } => InspectionError::UnreadableInput,
        loader::Error::MmapStale { .. } => InspectionError::ConcurrentMutation,
        loader::Error::Gguf { .. } | loader::Error::Msg { .. } | _ => InspectionError::InvalidGguf,
    };
    CliError::Inspection(kind)
}

#[derive(Debug, Clone, Copy)]
enum InspectionError {
    InvalidArguments,
    UnreadableInput,
    InvalidGguf,
    ConcurrentMutation,
    Internal,
}

impl InspectionError {
    const fn kind(self) -> &'static str {
        match self {
            Self::InvalidArguments => "invalid_arguments",
            Self::UnreadableInput => "unreadable_input",
            Self::InvalidGguf => "invalid_gguf",
            Self::ConcurrentMutation => "concurrent_mutation",
            Self::Internal => "internal",
        }
    }
}

#[derive(Serialize)]
struct InspectionReceipt {
    schema_version: u32,
    outcome: &'static str,
    format: &'static str,
    computed_digest: ComputedDigest,
    file_bytes: u64,
    tensor_count: u64,
    model: InspectionModel,
    type_census: Vec<InspectionTypeCensus>,
}

impl InspectionReceipt {
    fn from_inspection(inspection: loader::gguf::Inspection) -> Result<Self, InspectionError> {
        let loader::gguf::ArtifactDigest::Sha256(digest) = inspection.digest else {
            return Err(InspectionError::Internal);
        };
        let tensor_count =
            u64::try_from(inspection.tensors.len()).map_err(|_| InspectionError::Internal)?;
        let type_census = inspection
            .type_census
            .into_iter()
            .map(|entry| InspectionTypeCensus {
                ggml_type: format!("{:?}", entry.ggml_type),
                tensor_count: entry.tensor_count,
                logical_elements: entry.logical_elements,
                serialized_bytes: entry.byte_len,
            })
            .collect();
        Ok(Self {
            schema_version: INSPECTION_SCHEMA_VERSION,
            outcome: "inspection",
            format: "gguf-v3",
            computed_digest: ComputedDigest {
                algorithm: "sha256",
                hex: digest.to_string(),
            },
            file_bytes: inspection.file_len,
            tensor_count,
            model: InspectionModel {
                architecture: inspection.model.architecture,
                name: inspection.model.name,
                file_type: inspection.model.file_type,
                quantization_version: inspection.model.quantization_version,
            },
            type_census,
        })
    }
}

#[derive(Serialize)]
struct ComputedDigest {
    algorithm: &'static str,
    hex: String,
}

#[derive(Serialize)]
struct InspectionModel {
    architecture: Option<String>,
    name: Option<String>,
    file_type: Option<u32>,
    quantization_version: Option<u32>,
}

#[derive(Serialize)]
struct InspectionTypeCensus {
    ggml_type: String,
    tensor_count: u64,
    logical_elements: u64,
    serialized_bytes: u64,
}

#[derive(Serialize)]
struct InspectionErrorReceipt {
    schema_version: u32,
    outcome: &'static str,
    command: &'static str,
    kind: &'static str,
}

#[derive(Debug)]
enum CliError {
    Usage,
    Input(&'static str),
    Inspection(InspectionError),
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage => formatter.write_str(USAGE),
            Self::Input(message) => formatter.write_str(message),
            Self::Inspection(_) => formatter.write_str(INSPECTION_USAGE),
        }
    }
}
