//! # logismos binary
//!
//! Single binary entrypoint. The currently available `plan` subcommand reads
//! a placement contract from standard input or one JSON file and writes a
//! pure planning result. It does not initialise HIP or start a service.

use std::{
    env, fs,
    io::{self, Read},
    process::ExitCode,
};

fn main() -> ExitCode {
    match run() {
        Ok(outcome) => {
            println!("{outcome}");
            ExitCode::SUCCESS
        }
        Err(usage) => {
            eprintln!("{usage}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<String, &'static str> {
    let mut arguments = env::args_os();
    let _program = arguments.next();
    match arguments.next().as_deref() {
        Some(command) if command == "plan" => plan_command(arguments),
        _ => Err("usage: logismos plan [--input <path>|-]"),
    }
}

fn plan_command(
    mut arguments: impl Iterator<Item = std::ffi::OsString>,
) -> Result<String, &'static str> {
    let input = match arguments.next().as_deref() {
        None => read_stdin().map_err(|_| "unable to read placement input")?,
        Some(argument) if argument == "-" => {
            if arguments.next().is_some() {
                return Err("usage: logismos plan [--input <path>|-]");
            }
            read_stdin().map_err(|_| "unable to read placement input")?
        }
        Some(argument) if argument == "--input" => {
            let Some(path) = arguments.next() else {
                return Err("usage: logismos plan [--input <path>|-]");
            };
            if arguments.next().is_some() {
                return Err("usage: logismos plan [--input <path>|-]");
            }
            if path == "-" {
                read_stdin().map_err(|_| "unable to read placement input")?
            } else {
                fs::read_to_string(path).map_err(|_| "unable to read placement input")?
            }
        }
        Some(_) => return Err("usage: logismos plan [--input <path>|-]"),
    };
    serde_json::to_string(&placement::plan_json(&input))
        .map_err(|_| "unable to serialize placement result")
}

fn read_stdin() -> io::Result<String> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    Ok(input)
}
