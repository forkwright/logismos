//! `kernels` build script.
//!
//! Compiles every `.hip` source under `src/**/hip/*.hip` plus the
//! matching `_launcher.cpp` shim, and links them together into a
//! single static archive that Rust consumes via
//! `cargo:rustc-link-lib=static=logismos_kernels`.
//!
//! Requires `hipcc` on PATH (ROCm ≥ 6.4). On a machine without
//! `hipcc` we fall through to a "no-GPU" build: only the CPU
//! references compile. That fallback exists so `cargo check` can run
//! on a developer machine without ROCm; nothing non-test actually
//! depends on this state at runtime (the launchers return
//! [`Error::NoGpuBuild`] when no kernel archive was linked).

#![expect(
    clippy::doc_markdown,
    reason = "build-script docs use ROCm and cargo cfg spelling that trip doc_markdown"
)]

use std::env;
use std::path::{Path, PathBuf};
// kanon:ignore RUST/no-direct-process-command -- a build script runs before the workspace is built, so no project process wrapper is linkable here
use std::process::Command;

fn main() -> Result<(), String> {
    println!("cargo:rustc-check-cfg=cfg(logismos_no_gpu_kernels)");
    println!("cargo:rerun-if-changed=build.rs");
    // Re-run on any .hip or .cpp change under src/.
    for entry in walk_sources(&PathBuf::from("src")) {
        println!("cargo:rerun-if-changed={}", entry.display());
    }
    println!("cargo:rerun-if-env-changed=HIPCC");
    println!("cargo:rerun-if-env-changed=LOGISMOS_SKIP_HIP_BUILD");

    if env::var("LOGISMOS_SKIP_HIP_BUILD").is_ok() {
        println!("cargo:warning=LOGISMOS_SKIP_HIP_BUILD set — skipping HIP kernel compile");
        return Ok(());
    }

    let hipcc = env::var("HIPCC").unwrap_or_else(|_| "hipcc".to_string());
    if which(&hipcc).is_none() {
        println!(
            "cargo:warning=hipcc not found on PATH; skipping HIP kernel compile. \
             Set HIPCC=/path/to/hipcc or install ROCm."
        );
        // Emit a placeholder cfg so Rust code can tell.
        println!("cargo:rustc-cfg=logismos_no_gpu_kernels");
        return Ok(());
    }

    let out_dir = match env::var("OUT_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(e) => return Err(format!("OUT_DIR is set by cargo: {e}")),
    };
    let hip_sources = walk_with_ext(&PathBuf::from("src"), "hip");
    let cpp_sources = walk_with_ext(&PathBuf::from("src"), "cpp");

    if hip_sources.is_empty() && cpp_sources.is_empty() {
        println!("cargo:warning=no HIP/CPP sources under src/; nothing to build");
        return Ok(());
    }

    let mut obj_files = Vec::with_capacity(hip_sources.len() + cpp_sources.len());

    for src in hip_sources.iter().chain(cpp_sources.iter()) {
        let obj = out_dir.join(format!(
            "{}.o",
            src.file_name().and_then(|s| s.to_str()).unwrap_or("anon")
        ));
        // kanon:ignore RUST/no-direct-process-command -- invoking hipcc is the build script's purpose
        let status = match Command::new(&hipcc)
            .args([
                "--offload-arch=gfx1100",
                "-O3",
                "-std=c++17",
                "-fPIC",
                // Pin wave32 on gfx11 for WMMA correctness (dossier 01
                // §3.3 + §7.4). RDNA3 defaults to wave32 anyway; this
                // makes the choice audit-visible.
                "-mno-wavefrontsize64",
                "-c",
            ])
            .arg(src)
            .arg("-o")
            .arg(&obj)
            .status()
        {
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

    // Bundle all objects into a static archive.
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

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=logismos_kernels");
    // The HIP device runtime lives in the `amdhip64` shared object
    // already pulled in by hipcore; we rely on that.
    // hipcc links stdc++; re-export for the final binary.
    println!("cargo:rustc-link-lib=dylib=stdc++");

    Ok(())
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
