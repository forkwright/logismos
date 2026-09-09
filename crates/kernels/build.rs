//! `kernels` build script.
//!
//! Compiles every `.hip` source under `src/**/hip/*.hip` plus the
//! matching `_launcher.cpp` shim, and links them together into a
//! single static archive that Rust consumes via
//! `cargo:rustc-link-lib=static=logismos_kernels`.
//!
//! `LOGISMOS_HIP_BUILD=required` compiles those sources with `hipcc`
//! (ROCm ≥ 6.4) and fails if the compiler is absent. `cpu-only` builds
//! the CPU references only; the launchers return [`Error::NoGpuBuild`].

#![expect(
    clippy::doc_markdown,
    reason = "build-script docs use ROCm and cargo cfg spelling that trip doc_markdown"
)]

use std::env;
use std::path::{Path, PathBuf};
// kanon:ignore RUST/no-direct-process-command -- a build script runs before the workspace is built, so no project process wrapper is linkable here
use std::process::Command;

const HIP_BUILD_MODE_ENV: &str = "LOGISMOS_HIP_BUILD";
const HIP_BUILD_REQUIRED: &str = "required";
const HIP_BUILD_CPU_ONLY: &str = "cpu-only";
const HIP_TARGET: &str = include_str!("../../contracts/gpu-target.txt").trim_ascii();

enum HipBuildMode {
    Required,
    CpuOnly,
}

struct HipBuildConfiguration {
    mode: HipBuildMode,
    explicitly_set: bool,
}

fn main() -> Result<(), String> {
    println!("cargo:rustc-check-cfg=cfg(logismos_no_gpu_kernels)");
    println!("cargo:rerun-if-changed=build.rs");
    // Re-run on any .hip or .cpp change under src/.
    for entry in walk_sources(&PathBuf::from("src")) {
        println!("cargo:rerun-if-changed={}", entry.display());
    }
    println!("cargo:rerun-if-env-changed=HIPCC");
    println!("cargo:rerun-if-env-changed={HIP_BUILD_MODE_ENV}");
    println!("cargo:rerun-if-env-changed=LOGISMOS_SKIP_HIP_BUILD");
    println!("cargo:rerun-if-changed=../../contracts/gpu-target.txt");

    if env::var("LOGISMOS_SKIP_HIP_BUILD").is_ok() {
        return Err(
            "LOGISMOS_SKIP_HIP_BUILD is retired; set LOGISMOS_HIP_BUILD=cpu-only instead"
                .to_string(),
        );
    }

    let hip_build = hip_build_mode()?;
    if !gpu_feature_enabled() {
        if hip_build.explicitly_set && matches!(hip_build.mode, HipBuildMode::Required) {
            return Err(
                "kernels/gpu is disabled but LOGISMOS_HIP_BUILD=required explicitly requests HIP compilation; enable the gpu feature or set LOGISMOS_HIP_BUILD=cpu-only"
                    .to_string(),
            );
        }
        return Ok(());
    }

    let out_dir = match env::var("OUT_DIR") {
        Ok(directory) => PathBuf::from(directory),
        Err(error) => return Err(format!("OUT_DIR is set by cargo: {error}")),
    };
    if matches!(hip_build.mode, HipBuildMode::CpuOnly) {
        println!("cargo:warning=HIP kernel compile disabled (LOGISMOS_HIP_BUILD=cpu-only)");
        println!("cargo:rustc-cfg=logismos_no_gpu_kernels");
        return Ok(());
    }

    let hipcc = env::var("HIPCC").unwrap_or_else(|_| "hipcc".to_string());
    if which(&hipcc).is_none() {
        return Err(format!(
            "hipcc not found on PATH while {HIP_BUILD_MODE_ENV}={HIP_BUILD_REQUIRED}; \
             set HIPCC=/path/to/hipcc or select {HIP_BUILD_CPU_ONLY}"
        ));
    }

    let hip_sources = walk_with_ext(&PathBuf::from("src"), "hip");
    let cpp_sources = walk_with_ext(&PathBuf::from("src"), "cpp");

    if hip_sources.is_empty() && cpp_sources.is_empty() {
        return Err(format!(
            "no HIP/CPP sources under src/ while {HIP_BUILD_MODE_ENV}={HIP_BUILD_REQUIRED}; \
             only {HIP_BUILD_CPU_ONLY} may produce no kernel archive"
        ));
    }

    write_row_format_header(&out_dir)?;
    compile_sources(&hipcc, &out_dir, &hip_sources, &cpp_sources)?;

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=logismos_kernels");
    // The HIP device runtime lives in the `amdhip64` shared object
    // already pulled in by hipcore; we rely on that.
    // hipcc links stdc++; re-export for the final binary.
    println!("cargo:rustc-link-lib=dylib=stdc++");

    Ok(())
}

fn write_row_format_header(out_dir: &Path) -> Result<(), String> {
    let values_per_block = quant::q8_0::Q8_0_VALUES_PER_BLOCK;
    let scale_bytes = quant::q8_0::Q8_0_SCALE_BYTES;
    let value_bytes = quant::q8_0::Q8_0_VALUE_BYTES;
    let block_bytes = quant::q8_0::Q8_0_BLOCK_BYTES;
    let derived_block_bytes = scale_bytes.checked_add(value_bytes).ok_or_else(|| {
        "quant Q8_0 header fields overflow while deriving block bytes".to_string()
    })?;
    if block_bytes != derived_block_bytes
        || value_bytes != values_per_block
        || scale_bytes != std::mem::size_of::<u16>()
    {
        return Err("quant Q8_0 constants violate their declared layout relation".to_string());
    }
    let q4_values = quant::Q4_K_VALUES_PER_BLOCK;
    let q4_prefix = quant::q4_k::Q4_K_PREFIX_BYTES;
    let q4_scales = quant::q4_k::Q4_K_SCALE_BYTES;
    let q4_quant = quant::q4_k::Q4_K_QUANT_BYTES;
    let q4_bytes = quant::Q4_K_BLOCK_BYTES;
    let q5_values = quant::Q5_K_VALUES_PER_BLOCK;
    let q5_prefix = quant::q5_k::Q5_K_PREFIX_BYTES;
    let q5_scales = quant::q5_k::Q5_K_SCALE_BYTES;
    let q5_high = quant::q5_k::Q5_K_HIGH_BITS_BYTES;
    let q5_quant = quant::q5_k::Q5_K_QUANT_BYTES;
    let q5_bytes = quant::Q5_K_BLOCK_BYTES;
    let q6_values = quant::Q6_K_VALUES_PER_BLOCK;
    let q6_low = quant::q6_k::Q6_K_LOW_BITS_BYTES;
    let q6_high = quant::q6_k::Q6_K_HIGH_BITS_BYTES;
    let q6_scales = quant::q6_k::Q6_K_SCALE_BYTES;
    let q6_super_scale = quant::q6_k::Q6_K_SUPER_SCALE_BYTES;
    let q6_bytes = quant::Q6_K_BLOCK_BYTES;
    let iq4_nl_values = quant::IQ4_NL_VALUES_PER_BLOCK;
    let iq4_nl_scale = quant::iq4_nl::IQ4_NL_SCALE_BYTES;
    let iq4_nl_quant = quant::iq4_nl::IQ4_NL_QUANT_BYTES;
    let iq4_nl_bytes = quant::IQ4_NL_BLOCK_BYTES;
    let iq4_xs_values = quant::IQ4_XS_VALUES_PER_BLOCK;
    let iq4_xs_scale = quant::iq4_xs::IQ4_XS_SCALE_BYTES;
    let iq4_xs_scale_low = quant::iq4_xs::IQ4_XS_SCALE_LOW_BYTES;
    let iq4_xs_scale_high = quant::iq4_xs::IQ4_XS_SCALE_HIGH_BYTES;
    let iq4_xs_quant = quant::iq4_xs::IQ4_XS_QUANT_BYTES;
    let iq4_xs_bytes = quant::IQ4_XS_BLOCK_BYTES;
    let iq4_values = quant::IQ4_RECONSTRUCTION_VALUES
        .iter()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let q4_derived = checked_sum(&[q4_prefix, q4_scales, q4_quant], "Q4_K")?;
    let q5_derived = checked_sum(&[q5_prefix, q5_scales, q5_high, q5_quant], "Q5_K")?;
    let q6_derived = checked_sum(&[q6_low, q6_high, q6_scales, q6_super_scale], "Q6_K")?;
    let iq4_nl_derived = checked_sum(&[iq4_nl_scale, iq4_nl_quant], "IQ4_NL")?;
    let iq4_xs_derived = checked_sum(
        &[iq4_xs_scale, iq4_xs_scale_low, iq4_xs_scale_high, iq4_xs_quant],
        "IQ4_XS",
    )?;
    if q4_bytes != q4_derived
        || q5_bytes != q5_derived
        || q6_bytes != q6_derived
        || iq4_nl_bytes != iq4_nl_derived
        || iq4_xs_bytes != iq4_xs_derived
    {
        return Err("quant constants violate an executable serialized-row layout relation".to_string());
    }
    let header = format!(
        "#pragma once\n\n#include <cstddef>\n#include <cstdint>\n\ninline constexpr std::size_t LOGISMOS_F32_VALUE_BYTES = {f32_bytes};\ninline constexpr std::size_t LOGISMOS_Q8_0_VALUES_PER_BLOCK = {values_per_block};\ninline constexpr std::size_t LOGISMOS_Q8_0_SCALE_BYTES = {scale_bytes};\ninline constexpr std::size_t LOGISMOS_Q8_0_VALUE_BYTES = {value_bytes};\ninline constexpr std::size_t LOGISMOS_Q8_0_BLOCK_BYTES = {block_bytes};\ninline constexpr std::size_t LOGISMOS_K_GROUP_VALUES = {k_group};\ninline constexpr std::size_t LOGISMOS_Q4_K_VALUES_PER_BLOCK = {q4_values};\ninline constexpr std::size_t LOGISMOS_Q4_K_PREFIX_BYTES = {q4_prefix};\ninline constexpr std::size_t LOGISMOS_Q4_K_SCALE_BYTES = {q4_scales};\ninline constexpr std::size_t LOGISMOS_Q4_K_BLOCK_BYTES = {q4_bytes};\ninline constexpr std::size_t LOGISMOS_Q5_K_VALUES_PER_BLOCK = {q5_values};\ninline constexpr std::size_t LOGISMOS_Q5_K_PREFIX_BYTES = {q5_prefix};\ninline constexpr std::size_t LOGISMOS_Q5_K_SCALE_BYTES = {q5_scales};\ninline constexpr std::size_t LOGISMOS_Q5_K_HIGH_BITS_BYTES = {q5_high};\ninline constexpr std::size_t LOGISMOS_Q5_K_BLOCK_BYTES = {q5_bytes};\ninline constexpr std::size_t LOGISMOS_Q6_K_VALUES_PER_BLOCK = {q6_values};\ninline constexpr std::size_t LOGISMOS_Q6_K_LOW_BITS_BYTES = {q6_low};\ninline constexpr std::size_t LOGISMOS_Q6_K_HIGH_BITS_BYTES = {q6_high};\ninline constexpr std::size_t LOGISMOS_Q6_K_SCALE_BYTES = {q6_scales};\ninline constexpr std::size_t LOGISMOS_Q6_K_BLOCK_BYTES = {q6_bytes};\ninline constexpr std::size_t LOGISMOS_IQ4_NL_VALUES_PER_BLOCK = {iq4_nl_values};\ninline constexpr std::size_t LOGISMOS_IQ4_NL_BLOCK_BYTES = {iq4_nl_bytes};\ninline constexpr std::size_t LOGISMOS_IQ4_XS_VALUES_PER_BLOCK = {iq4_xs_values};\ninline constexpr std::size_t LOGISMOS_IQ4_XS_BLOCK_BYTES = {iq4_xs_bytes};\ninline constexpr std::int8_t LOGISMOS_IQ4_RECONSTRUCTION_VALUES[16] = {{{iq4_values}}};\n"
        , f32_bytes = quant::f32_row::F32_ROW_VALUE_BYTES, k_group = quant::K_GROUP_VALUES
    );
    std::fs::write(out_dir.join("row_format.h"), header)
        .map_err(|error| format!("write generated serialized-row format header: {error}"))
}

fn checked_sum(fields: &[usize], format: &str) -> Result<usize, String> {
    fields.iter().try_fold(0usize, |total, field| {
        total
            .checked_add(*field)
            .ok_or_else(|| format!("quant {format} header fields overflow while deriving block bytes"))
    })
}

fn compile_sources(
    hipcc: &str,
    out_dir: &Path,
    hip_sources: &[PathBuf],
    cpp_sources: &[PathBuf],
) -> Result<(), String> {
    let mut obj_files = Vec::with_capacity(hip_sources.len() + cpp_sources.len());

    for src in hip_sources.iter().chain(cpp_sources) {
        let obj = out_dir.join(format!(
            "{}.o",
            src.file_name().and_then(|s| s.to_str()).unwrap_or("anon")
        ));
        // kanon:ignore RUST/no-direct-process-command -- invoking hipcc is the build script's purpose
        let mut command = Command::new(hipcc);
        command
            .args([
                &format!("--offload-arch={HIP_TARGET}"),
                "-O3",
                "-std=c++17",
                "-fPIC",
                // Pin wave32 on gfx11 for WMMA correctness (dossier 01
                // §3.3 + §7.4). RDNA3 defaults to wave32 anyway; this
                // makes the choice audit-visible.
                "-mno-wavefrontsize64",
                "-I",
            ])
            .arg(out_dir);
        if src.file_name().is_some_and(|name| {
            name == "row_gemv.hip" || name == "gdn_step.hip" || name == "causal_conv_step.hip"
        }) {
            // WHY: these correctness baselines retain separately rounded f32
            // operations. Scope no-fast-math and no contraction to their
            // sources rather than changing the rest of the HIP archive.
            command.args(["-fno-fast-math", "-ffp-contract=off"]);
        }
        // kanon:ignore RUST/no-direct-process-command -- invoking hipcc is the build script's purpose
        let status = match command.arg("-c").arg(src).arg("-o").arg(&obj).status() {
            Ok(s) => s,
            Err(e) => return Err(format!("invoke hipcc on {}: {e}", src.display())),
        };
        if !status.success() {
            return Err(format!(
                "hipcc failed compiling {} (status {status:?})",
                src.display()
            ));
        }
        obj_files.push(obj);
    }

    let archive = out_dir.join("liblogismos_kernels.a");
    remove_stale_archive(&archive);
    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());
    let status = match Command::new(&ar)
        .arg("rcs")
        .arg(&archive)
        .args(&obj_files)
        .status()
    {
        Ok(s) => s,
        Err(e) => return Err(format!("invoke ar at {ar}: {e}")),
    };
    if !status.success() {
        return Err(format!("ar failed (status {status:?})"));
    }

    Ok(())
}

fn gpu_feature_enabled() -> bool {
    env::var_os("CARGO_FEATURE_GPU").is_some()
}

fn hip_build_mode() -> Result<HipBuildConfiguration, String> {
    match env::var(HIP_BUILD_MODE_ENV) {
        Ok(value) if value == HIP_BUILD_REQUIRED => Ok(HipBuildConfiguration {
            mode: HipBuildMode::Required,
            explicitly_set: true,
        }),
        Ok(value) if value == HIP_BUILD_CPU_ONLY => Ok(HipBuildConfiguration {
            mode: HipBuildMode::CpuOnly,
            explicitly_set: true,
        }),
        Ok(value) => Err(format!(
            "invalid {HIP_BUILD_MODE_ENV}={value:?}; expected {HIP_BUILD_REQUIRED} or {HIP_BUILD_CPU_ONLY}"
        )),
        Err(env::VarError::NotPresent) => Ok(HipBuildConfiguration {
            mode: HipBuildMode::Required,
            explicitly_set: false,
        }),
        Err(error) => Err(format!("read {HIP_BUILD_MODE_ENV}: {error}")),
    }
}

/// Remove a stale static archive, logging (but not failing) on I/O error.
///
/// WHY: the archive is routinely replaced on every build; failure to remove
/// a prior copy is non-fatal because `ar rcs` overwrites. Surfacing the error
/// as `cargo:warning=...` lets an operator diagnose downstream `ar` failures
/// without silently discarding the root cause.
fn remove_stale_archive(archive: &Path) {
    if !archive.exists() {
        return;
    }
    if let Err(err) = std::fs::remove_file(archive) {
        println!(
            "cargo:warning=failed to remove stale archive {}: {err}",
            archive.display()
        );
    }
}

fn which(cmd: &str) -> Option<PathBuf> {
    if cmd.contains('/') {
        let p = PathBuf::from(cmd);
        return if p.is_file() { Some(p) } else { None };
    }
    let path = env::var_os("PATH")?;
    for dir in env::split_paths(&path) {
        let candidate = dir.join(cmd);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn walk_sources(root: &Path) -> Vec<PathBuf> {
    let mut hip = walk_with_ext(root, "hip");
    hip.extend(walk_with_ext(root, "cpp"));
    hip.extend(walk_with_ext(root, "h"));
    hip.extend(walk_with_ext(root, "hpp"));
    hip
}

fn walk_with_ext(root: &Path, ext: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk_rec(root, ext, &mut out);
    out
}

fn walk_rec(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in read.flatten() {
        let p = entry.path();
        if p.is_dir() {
            walk_rec(&p, ext, out);
        } else if p.extension().and_then(|e| e.to_str()) == Some(ext) {
            out.push(p);
        }
    }
}
