//! # logismos binary
//!
//! The `plan` subcommand reads one bounded placement contract from standard
//! input or one JSON file. It is CPU-only: it neither initialises HIP nor
//! starts a service.

use std::{
    env,
    ffi::OsString,
    fmt,
    fs::File,
    io::{self, Read, Write},
    process::ExitCode,
};

const USAGE: &str = "usage: logismos plan [--input <path>|-]";
/// Maximum accepted placement-contract size, preventing unbounded CLI input allocation.
const MAX_PLAN_INPUT_BYTES: usize = 4 * 1024 * 1024;

fn main() -> ExitCode {
    match run() {
        Ok(outcome) => write_outcome(outcome),
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<placement::PlanOutcome, CliError> {
    let mut arguments = env::args_os();
    let _program = arguments.next();
    match arguments.next().as_deref() {
        Some(command) if command == "plan" => plan_command(arguments),
        _ => Err(CliError::Usage),
    }
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

fn write_outcome(outcome: placement::PlanOutcome) -> ExitCode {
    let exit_code = match outcome {
        placement::PlanOutcome::Plan { .. } => ExitCode::SUCCESS,
        placement::PlanOutcome::Refusal { .. } => ExitCode::from(1),
        _ => ExitCode::from(1),
    };
    let output = match serde_json::to_string(&outcome) {
        Ok(output) => output,
        Err(_) => {
            eprintln!("unable to serialize placement result");
            return ExitCode::from(2);
        }
    };
    if writeln!(io::stdout().lock(), "{output}").is_err() {
        eprintln!("unable to write placement result");
        return ExitCode::from(2);
    }
    exit_code
}

#[derive(Debug)]
enum CliError {
    Usage,
    Input(&'static str),
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage => formatter.write_str(USAGE),
            Self::Input(message) => formatter.write_str(message),
        }
    }
}
