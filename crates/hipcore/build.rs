//! `hipcore` build script.
//!
//! - Runs bindgen over the HIP runtime header.
//! - Emits cargo link directives for `amdhip64`.
//!
//! Layouts differ by distribution: Fedora puts the shared object in `/usr/lib64`, while
//! Debian and Ubuntu use the multiarch directory `/usr/lib/<triple>`. Both are searched, as
//! is `/opt/rocm` when present. The soname is matched by prefix because the version suffix
//! tracks the installed ROCm release. If the header or the runtime is missing, the build
//! fails immediately with a clear message — per project rule 7 (no silent fallbacks).

#![expect(
    clippy::doc_markdown,
    reason = "build-script docs use ROCm package and header names that trip doc_markdown"
)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() -> Result<(), String> {
    let hip_include = locate_hip_include()?;
    let wrapper = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("include/wrapper_runtime.h");

    // rerun-if-changed takes a path relative to the package root. Absolute
    // paths here get baked into cargo's build fingerprint; when cargo builds
    // the workspace inside a tempdir (e.g. archeion's CI sandbox at
    // /tmp/.tmpXXXXX/), that tempdir path is cached and later stats fail once
    // the tempdir evaporates. Closes forkwright/logismos#1.
    println!("cargo:rerun-if-changed=include/wrapper_runtime.h");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");

    // Link amdhip64 from the system path.
    let lib_dir = locate_hip_lib()?;
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=amdhip64");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap_or_else(|_| "target".into()));
    let out_path = out_dir.join("hip_bindings.rs");

    // WHY the suffix is detected rather than written down: hip_runtime_api.h macro-renames a few
    // symbols to an ABI-revision suffix (`#define hipDeviceProp_t hipDeviceProp_tR0600`) before
    // bindgen ever sees the unversioned name, so the generated bindings expose only the suffixed
    // ones and an alias is needed to restore the stable name a C caller gets for free.
    //
    // Hardcoding `R0600` pinned this crate to ROCm 6.0 in a way that failed misleadingly: on ROCm
    // 5.7 the macro does not exist, the build script still succeeded, and the crate then failed to
    // compile with unresolved-import errors that read as a code defect rather than a too-old SDK.
    // Reading the revision out of the installed header instead means the crate tracks whatever
    // ROCm is present — including revisions that do not exist yet — and needs no edit when AMD
    // bumps it again.
    let abi_aliases = detect_abi_aliases(&hip_include);

    let mut builder = bindgen::Builder::default()
        .header(wrapper.to_string_lossy())
        .clang_arg("-D__HIP_PLATFORM_AMD__=1")
        .clang_arg(format!("-I{}", hip_include.display()))
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++17")
        .layout_tests(false)
        .derive_default(true)
        .allowlist_function("hip.*")
        .allowlist_type("hip.*|HIP.*")
        .allowlist_var("hip.*|HIP.*")
        .rustified_enum("hipError_t")
        .rustified_enum("hipMemcpyKind")
        .rustified_enum("hipDeviceAttribute_t");

    for line in &abi_aliases {
        builder = builder.raw_line(line);
    }

    let bindings = builder
        .generate()
        .map_err(|e| format!("bindgen failed on HIP runtime header: {e}"))?;

    bindings
        .write_to_file(&out_path)
        .map_err(|e| format!("write HIP bindings to {}: {e}", out_path.display()))?;

    Ok(())
}

fn locate_hip_include() -> Result<PathBuf, String> {
    // Order: HIP_PATH env, ROCM_PATH env, Fedora system path, /opt/rocm.
    let candidates = hip_base_candidates()
        .into_iter()
        .map(|base| base.join("include"))
        .chain(std::iter::once(PathBuf::from("/usr/include")))
        .collect::<Vec<_>>();

    for c in &candidates {
        if c.join("hip/hip_runtime_api.h").is_file() {
            return Ok(c.clone());
        }
    }
    Err(format!(
        "hipcore: could not find hip/hip_runtime_api.h. Searched: {candidates:?}. \
         Set HIP_PATH or ROCM_PATH."
    ))
}

fn locate_hip_lib() -> Result<PathBuf, String> {
    let multiarch = multiarch_dir_name();
    let candidates: Vec<PathBuf> = hip_base_candidates()
        .into_iter()
        .flat_map(|base| {
            // WHY the multiarch directory: Debian and Ubuntu install shared objects under
            // lib/<triple>, so libamdhip64.so lands in /usr/lib/x86_64-linux-gnu and never in
            // /usr/lib. Searching only lib and lib64 reports the runtime as missing on every
            // Debian-family box while it is installed and loadable, which reads as "install
            // ROCm" to someone who already has.
            vec![
                base.join("lib"),
                base.join("lib64"),
                base.join("lib").join(&multiarch),
            ]
        })
        .chain([PathBuf::from("/usr/lib64"), PathBuf::from("/usr/lib")])
        .collect();

    for c in &candidates {
        if has_hip_runtime(c) {
            return Ok(c.clone());
        }
    }
    Err(format!(
        "hipcore: could not find libamdhip64.so. Searched: {candidates:?}. \
         Install ROCm runtime (Fedora: `hip-runtime-amd`)."
    ))
}

/// Aliases restoring the unversioned HIP symbol names, derived from the installed header.
///
/// Returns an empty list when the header does not macro-rename them, which is the case before
/// ROCm 6 — there the unversioned names are already what bindgen emits, and adding an alias to a
/// symbol that does not exist is what broke the build on ROCm 5.7.
fn detect_abi_aliases(include_dir: &Path) -> Vec<String> {
    let header = include_dir.join("hip/hip_runtime_api.h");
    let Ok(src) = fs::read_to_string(&header) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (macro_name, alias) in [
        ("hipDeviceProp_t", "pub type hipDeviceProp_t = {};"),
        (
            "hipGetDeviceProperties",
            "pub use {} as hipGetDeviceProperties;",
        ),
    ] {
        if let Some(target) = renamed_target(&src, macro_name) {
            out.push(alias.replace("{}", &target));
        }
    }
    out
}

/// The right-hand side of `#define <name> <name>R<digits>`, if the header defines one.
fn renamed_target(src: &str, name: &str) -> Option<String> {
    let needle = format!("#define {name} ");
    src.lines().find_map(|line| {
        let rest = line.trim().strip_prefix(&needle)?;
        let target = rest.split_whitespace().next()?;
        // Only accept the ABI-revision form, so an unrelated #define cannot inject a name.
        let suffix = target.strip_prefix(name)?.strip_prefix('R')?;
        (!suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()))
            .then(|| target.to_string())
    })
}

/// WHY a prefix scan instead of exact filenames: distributions ship the runtime under a
/// versioned soname, and Ubuntu 24.04 carries `libamdhip64.so.5`. Testing only `.so` and
/// `.so.6` misses a present, loadable library on any box that is not on ROCm 6 — the linker
/// resolves `-lamdhip64` through whichever soname is there, so the exact suffix is not the
/// build script's business.
fn has_hip_runtime(dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    entries.filter_map(Result::ok).any(|e| {
        e.file_name()
            .to_str()
            .is_some_and(|n| n.starts_with("libamdhip64.so"))
    })
}

/// Debian multiarch directory name for the build target, e.g. `x86_64-linux-gnu`.
///
/// WHY derived from TARGET rather than hardcoded: cargo's triple carries a vendor field
/// (`x86_64-unknown-linux-gnu`) that the multiarch path omits, and hardcoding one arch would
/// silently reintroduce this bug on aarch64.
fn multiarch_dir_name() -> String {
    let target = env::var("TARGET").unwrap_or_default();
    let parts: Vec<&str> = target.split('-').collect();
    if parts.len() >= 4 {
        format!("{}-{}-{}", parts[0], parts[2], parts[3])
    } else {
        target
    }
}

fn hip_base_candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(p) = env::var("HIP_PATH") {
        v.push(PathBuf::from(p));
    }
    if let Ok(p) = env::var("ROCM_PATH") {
        v.push(PathBuf::from(p));
    }
    v.push(PathBuf::from("/opt/rocm"));
    v.push(PathBuf::from("/usr"));
    v.retain(|p: &PathBuf| Path::new(p).exists());
    v
}
