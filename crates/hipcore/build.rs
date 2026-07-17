//! `hipcore` build script.
//!
//! - Runs bindgen over the HIP runtime header.
//! - Emits cargo link directives for `amdhip64`.
//!
//! ROCm 6.4.x on Fedora ships headers in `/usr/include/hip/` and the
//! shared object as `/usr/lib64/libamdhip64.so`. No `/opt/rocm` path
//! on this host. If either is missing, the build fails immediately
//! with a clear message — per project rule 7 (no silent fallbacks).

#![expect(
    clippy::doc_markdown,
    reason = "build-script docs use ROCm package and header names that trip doc_markdown"
)]

use std::env;
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

    let bindings = bindgen::Builder::default()
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
        .rustified_enum("hipDeviceAttribute_t")
        // hip_runtime_api.h macro-renames a handful of symbols to an ABI-revision
        // suffix (`#define hipDeviceProp_t hipDeviceProp_tR0600`, same for
        // hipGetDeviceProperties) before bindgen ever sees the unversioned name —
        // the preprocessor rewrites the typedef and every reference to it in the
        // header itself. C callers get the versioned symbol transparently by
        // re-including the header; bindgen has no equivalent, so the generated
        // bindings only expose the suffixed names. Alias them back so hipcore's
        // Rust call sites can use the stable, version-agnostic name like a C
        // caller would. Bump the suffix here if a future ROCm header revision
        // renames it again.
        .raw_line("pub type hipDeviceProp_t = hipDeviceProp_tR0600;")
        .raw_line("pub use hipGetDevicePropertiesR0600 as hipGetDeviceProperties;")
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
    let candidates: Vec<PathBuf> = hip_base_candidates()
        .into_iter()
        .flat_map(|base| vec![base.join("lib"), base.join("lib64")])
        .chain([PathBuf::from("/usr/lib64"), PathBuf::from("/usr/lib")])
        .collect();

    for c in &candidates {
        if c.join("libamdhip64.so").exists() || c.join("libamdhip64.so.6").exists() {
            return Ok(c.clone());
        }
    }
    Err(format!(
        "hipcore: could not find libamdhip64.so. Searched: {candidates:?}. \
         Install ROCm runtime (Fedora: `hip-runtime-amd`)."
    ))
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
