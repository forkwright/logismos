//! # logismos binary
//!
//! The `plan` and `inspect` subcommands are CPU-only: they neither initialise
//! HIP nor start a service. The minimal `cargo build -p bin` dependency graph
//! also excludes HIP; broader workspace feature unification can produce a
//! different linkage graph without changing that command behavior.

use std::{
    collections::HashMap,
    env,
    ffi::OsString,
    fmt,
    fs::File,
    io::{self, Read, Write},
    path::Path,
    process::ExitCode,
};

use serde::{
    Serialize, Serializer,
    ser::{SerializeSeq, SerializeStruct},
};

const USAGE: &str = "usage: logismos plan [--input <path>|-]";
const INSPECTION_USAGE: &str = "usage: logismos inspect --input <path> [--metadata]";
const INSPECTION_SCHEMA_VERSION: u32 = 1;
const INSPECTION_METADATA_SCHEMA_VERSION: u32 = 1;
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
            inspect_command(arguments).map(|outcome| CommandOutcome::Inspection(Box::new(outcome)))
        }
        _ => Err(CliError::Usage),
    }
}

enum CommandOutcome {
    Plan(placement::PlanOutcome),
    Inspection(Box<InspectionOutcome>),
}

enum InspectionOutcome {
    Receipt(InspectionReceipt),
    Metadata {
        receipt: InspectionReceipt,
        observed: Box<loader::gguf::ObservedArtifact>,
    },
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
) -> Result<InspectionOutcome, CliError> {
    let Some(flag) = arguments.next() else {
        return Err(CliError::Inspection(InspectionError::InvalidArguments));
    };
    if flag != "--input" {
        return Err(CliError::Inspection(InspectionError::InvalidArguments));
    }
    let Some(path) = arguments.next() else {
        return Err(CliError::Inspection(InspectionError::InvalidArguments));
    };
    if path == "-" || path == "--metadata" {
        return Err(CliError::Inspection(InspectionError::InvalidArguments));
    }
    let metadata_requested = match arguments.next().as_deref() {
        None => false,
        Some(flag) if flag == "--metadata" && arguments.next().is_none() => true,
        Some(_) => return Err(CliError::Inspection(InspectionError::InvalidArguments)),
    };
    let observed = loader::gguf::observe_gguf_with_sha256(Path::new(&path))
        .map_err(|error| map_inspection_error(&error))?;
    let receipt =
        InspectionReceipt::from_inspection(observed.inspection()).map_err(CliError::Inspection)?;
    Ok(if metadata_requested {
        InspectionOutcome::Metadata {
            receipt,
            observed: Box::new(observed),
        }
    } else {
        InspectionOutcome::Receipt(receipt)
    })
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
        CommandOutcome::Inspection(outcome) => write_inspection_outcome(outcome),
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

fn write_inspection_outcome(outcome: &InspectionOutcome) -> ExitCode {
    match outcome {
        InspectionOutcome::Receipt(receipt) => write_inspection_receipt(receipt),
        InspectionOutcome::Metadata { receipt, observed } => {
            write_inspection_metadata_report(receipt, observed)
        }
    }
}

fn write_inspection_metadata_report(
    receipt: &InspectionReceipt,
    observed: &loader::gguf::ObservedArtifact,
) -> ExitCode {
    let report = InspectionMetadataReport {
        schema_version: INSPECTION_METADATA_SCHEMA_VERSION,
        outcome: "inspection_metadata",
        format: "gguf-v3",
        inspection: receipt,
        metadata: MetadataEntries(observed.metadata()),
        tensors: TensorEntries(&observed.inspection().tensors),
    };
    let stdout = io::stdout();
    let mut writer = stdout.lock();
    if serde_json::to_writer(&mut writer, &report).is_err() || writeln!(writer).is_err() {
        eprintln!("unable to serialize inspection metadata report");
        return ExitCode::from(2);
    }
    ExitCode::SUCCESS
}

fn write_inspection_error(error: InspectionError) -> ExitCode {
    let diagnostic = error.diagnostic();
    let receipt = InspectionErrorReceipt {
        schema_version: INSPECTION_SCHEMA_VERSION,
        outcome: "error",
        command: "inspect",
        kind: error.kind(),
    };
    let exit_code = write_inspection_json(
        &receipt,
        "unable to serialize inspection error",
        ExitCode::from(2),
    );
    if let Some(diagnostic) = diagnostic {
        eprintln!("{diagnostic}");
    }
    exit_code
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
        loader::Error::UnknownGgmlType {
            type_id, offset, ..
        } => InspectionError::UnknownGgmlType {
            type_id: *type_id,
            offset: *offset,
        },
        loader::Error::Gguf { .. } | loader::Error::Msg { .. } | _ => InspectionError::InvalidGguf,
    };
    CliError::Inspection(kind)
}

#[derive(Debug, Clone, Copy)]
enum InspectionError {
    InvalidArguments,
    UnreadableInput,
    InvalidGguf,
    UnknownGgmlType { type_id: u32, offset: u64 },
    ConcurrentMutation,
    Internal,
}

impl InspectionError {
    const fn kind(self) -> &'static str {
        match self {
            Self::InvalidArguments => "invalid_arguments",
            Self::UnreadableInput => "unreadable_input",
            Self::InvalidGguf | Self::UnknownGgmlType { .. } => "invalid_gguf",
            Self::ConcurrentMutation => "concurrent_mutation",
            Self::Internal => "internal",
        }
    }

    const fn diagnostic(self) -> Option<UnknownGgmlTypeDiagnostic> {
        match self {
            Self::UnknownGgmlType { type_id, offset } => {
                Some(UnknownGgmlTypeDiagnostic { type_id, offset })
            }
            Self::InvalidArguments
            | Self::UnreadableInput
            | Self::InvalidGguf
            | Self::ConcurrentMutation
            | Self::Internal => None,
        }
    }
}

struct UnknownGgmlTypeDiagnostic {
    type_id: u32,
    offset: u64,
}

impl fmt::Display for UnknownGgmlTypeDiagnostic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "GGUF inspection refused unknown GGML storage type id {} at descriptor offset {}",
            self.type_id, self.offset
        )
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
    fn from_inspection(inspection: &loader::gguf::Inspection) -> Result<Self, InspectionError> {
        let loader::gguf::ArtifactDigest::Sha256(digest) = inspection.digest else {
            return Err(InspectionError::Internal);
        };
        let tensor_count =
            u64::try_from(inspection.tensors.len()).map_err(|_| InspectionError::Internal)?;
        let type_census = inspection
            .type_census
            .iter()
            .map(|entry| InspectionTypeCensus {
                ggml_type: GgmlTypeName(entry.ggml_type),
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
                architecture: inspection.model.architecture.clone(),
                name: inspection.model.name.clone(),
                file_type: inspection.model.file_type,
                quantization_version: inspection.model.quantization_version,
            },
            type_census,
        })
    }
}

/// Explicit metadata-report contract for an observed GGUF artifact.
///
/// This report is diagnostic output only. It exposes parsed file metadata but
/// is not a model-admission authority, source-provenance assertion, or runtime
/// support claim. Its tensor entries preserve descriptor source order and
/// validated serialized extents; they do not decode tensor payloads.
#[derive(Serialize)]
struct InspectionMetadataReport<'a> {
    schema_version: u32,
    outcome: &'static str,
    format: &'static str,
    inspection: &'a InspectionReceipt,
    metadata: MetadataEntries<'a>,
    tensors: TensorEntries<'a>,
}

struct MetadataEntries<'a>(&'a HashMap<String, loader::gguf::MetaValue>);

impl Serialize for MetadataEntries<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // WHY: `HashMap` iteration is deliberately nondeterministic; the
        // loader's parser already bounds this collection, so sorting borrowed
        // entries gives a stable report without copying metadata values.
        let mut entries: Vec<_> = self.0.iter().collect();
        entries.sort_unstable_by(|left, right| left.0.cmp(right.0));
        let mut sequence = serializer.serialize_seq(Some(entries.len()))?;
        for (key, value) in entries {
            sequence.serialize_element(&MetadataEntry {
                key,
                value: TaggedMetadataValue(value),
            })?;
        }
        sequence.end()
    }
}

#[derive(Serialize)]
struct MetadataEntry<'a> {
    key: &'a str,
    value: TaggedMetadataValue<'a>,
}

struct TaggedMetadataValue<'a>(&'a loader::gguf::MetaValue);

impl Serialize for TaggedMetadataValue<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use loader::gguf::MetaValue;

        let type_name = self.0.value_type().tag();
        match self.0 {
            MetaValue::U8(value) => serialize_tagged_value(serializer, type_name, "value", value),
            MetaValue::I8(value) => serialize_tagged_value(serializer, type_name, "value", value),
            MetaValue::U16(value) => serialize_tagged_value(serializer, type_name, "value", value),
            MetaValue::I16(value) => serialize_tagged_value(serializer, type_name, "value", value),
            MetaValue::U32(value) => serialize_tagged_value(serializer, type_name, "value", value),
            MetaValue::I32(value) => serialize_tagged_value(serializer, type_name, "value", value),
            MetaValue::U64(value) => serialize_tagged_value(serializer, type_name, "value", value),
            MetaValue::I64(value) => serialize_tagged_value(serializer, type_name, "value", value),
            MetaValue::F32(value) => {
                let bits = format!("{:08x}", value.to_bits());
                serialize_tagged_value(serializer, type_name, "bits", &bits)
            }
            MetaValue::F64(value) => {
                let bits = format!("{:016x}", value.to_bits());
                serialize_tagged_value(serializer, type_name, "bits", &bits)
            }
            MetaValue::Bool(value) => serialize_tagged_value(serializer, type_name, "value", value),
            MetaValue::String(value) => {
                serialize_tagged_value(serializer, type_name, "value", value)
            }
            MetaValue::Array(array) => serialize_tagged_array(
                serializer,
                type_name,
                array.element_type().tag(),
                &MetadataArray(array.values()),
            ),
            _ => Err(serde::ser::Error::custom(
                "unsupported GGUF metadata variant in metadata report",
            )),
        }
    }
}

fn serialize_tagged_array<S>(
    serializer: S,
    type_name: &'static str,
    element_type: &'static str,
    values: &MetadataArray<'_>,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut tagged = serializer.serialize_struct("TaggedMetadataValue", 3)?;
    tagged.serialize_field("type", type_name)?;
    tagged.serialize_field("element_type", element_type)?;
    tagged.serialize_field("values", values)?;
    tagged.end()
}

fn serialize_tagged_value<S, T>(
    serializer: S,
    type_name: &'static str,
    value_name: &'static str,
    value: &T,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    T: Serialize + ?Sized,
{
    let mut tagged = serializer.serialize_struct("TaggedMetadataValue", 2)?;
    tagged.serialize_field("type", type_name)?;
    tagged.serialize_field(value_name, value)?;
    tagged.end()
}

struct MetadataArray<'a>(&'a [loader::gguf::MetaValue]);

impl Serialize for MetadataArray<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for value in self.0 {
            sequence.serialize_element(&TaggedMetadataValue(value))?;
        }
        sequence.end()
    }
}

struct TensorEntries<'a>(&'a [loader::gguf::InspectedTensor]);

impl Serialize for TensorEntries<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for tensor in self.0 {
            sequence.serialize_element(&TensorEntry {
                name: &tensor.name,
                dims: &tensor.dims,
                ggml_type: GgmlTypeName(tensor.ggml_type),
                logical_elements: tensor.logical_elements,
                file_offset: tensor.file_offset,
                serialized_bytes: tensor.byte_len,
            })?;
        }
        sequence.end()
    }
}

#[derive(Serialize)]
struct TensorEntry<'a> {
    name: &'a str,
    dims: &'a [u64],
    ggml_type: GgmlTypeName,
    logical_elements: u64,
    file_offset: u64,
    serialized_bytes: u64,
}

#[derive(Clone, Copy)]
struct GgmlTypeName(loader::gguf::GgmlType);

impl Serialize for GgmlTypeName {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_str(&format_args!("{:?}", self.0))
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
    ggml_type: GgmlTypeName,
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
