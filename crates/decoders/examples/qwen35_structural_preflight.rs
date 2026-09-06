//! Path-free Qwen3.5 structural-preflight witness.
//!
//! Run this example only through `scripts/gpu-denied-runner.sh --ro-input-file`.
//! `LOGISMOS_GPU_DENIED_INPUT` names the runner's synthetic read-only bind; an
//! environment variable alone does not isolate or authorize a host artifact.
//! This witness reports structural observation facts only. It does not decode
//! payloads, establish provenance, execute a model, or estimate residency.

use std::{env, ffi::OsString, path::PathBuf, process::ExitCode};

use decoders::Qwen35StructuralProfile;
use loader::gguf::{ArtifactDigest, ObservedArtifact, observe_gguf_with_sha256};

const INPUT_ENVIRONMENT: &str = "LOGISMOS_GPU_DENIED_INPUT";
const REPORT_SCHEMA_VERSION: u64 = 1;
const MISSING_INPUT_EXIT: u8 = 64;
const REFUSAL_EXIT: u8 = 2;

fn main() -> ExitCode {
    let Some(input) = input_path(env::var_os(INPUT_ENVIRONMENT)) else {
        eprintln!("qwen35 structural preflight requires the reviewed read-only input binding");
        return ExitCode::from(MISSING_INPUT_EXIT);
    };
    let Ok(observed) = observe_gguf_with_sha256(&input) else {
        eprintln!("qwen35 structural observation refused the supplied input");
        return ExitCode::from(REFUSAL_EXIT);
    };
    let Ok(profile) = Qwen35StructuralProfile::try_from_observed(&observed) else {
        eprintln!("qwen35 structural preflight refused the observed artifact");
        return ExitCode::from(REFUSAL_EXIT);
    };
    let Some(report) = report(&observed, &profile) else {
        eprintln!("qwen35 structural observation did not retain a SHA-256 digest");
        return ExitCode::from(REFUSAL_EXIT);
    };
    println!("{report}");
    ExitCode::SUCCESS
}

fn input_path(value: Option<OsString>) -> Option<PathBuf> {
    value.map(PathBuf::from)
}

fn report(observed: &ObservedArtifact, profile: &Qwen35StructuralProfile<'_>) -> Option<String> {
    let inspection = observed.inspection();
    let ArtifactDigest::Sha256(digest) = inspection.digest else {
        return None;
    };
    Some(format!(
        concat!(
            "{{\"schema_version\":{},",
            "\"outcome\":\"structural_preflight\",",
            "\"digest\":{{\"algorithm\":\"sha256\",\"hex\":\"{}\"}},",
            "\"file_bytes\":{},\"tensor_count\":{},",
            "\"stored_block_count\":{},\"main_block_count\":{},",
            "\"nextn_block_count\":{}}}"
        ),
        REPORT_SCHEMA_VERSION,
        digest,
        inspection.file_len,
        inspection.tensors.len(),
        profile.stored_block_count(),
        profile.main_block_count(),
        profile.nextn_block_count(),
    ))
}

#[cfg(test)]
mod tests {
    use super::input_path;

    #[test]
    fn missing_input_binding_is_refused() {
        assert!(
            input_path(None).is_none(),
            "missing runner input binding must not fall back to a host path"
        );
    }
}
