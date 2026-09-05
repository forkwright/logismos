//! CPU rebuild witness for the native raw-instruction fixture.

use std::path::PathBuf;
use std::process::Command;

const LLVM_MC: &str = "/usr/lib64/rocm/llvm/bin/llvm-mc";
const LLVM_OBJDUMP: &str = "/usr/lib64/rocm/llvm/bin/llvm-objdump";
const EXPECTED_ASSEMBLER: &str = "AOMP-18.0-12";
const EXPECTED_SOURCE_ID: &str = "18.0-12-ce1873ac686bb90ddec72bb99889a4e80e2de382";

#[test]
#[ignore = "requires the explicitly pinned local AOMP llvm-mc/llvm-objdump witness"]
#[allow(
    clippy::expect_used,
    reason = "the explicit witness test must name each unavailable external prerequisite"
)]
fn rebuilds_and_disassembles_gfx1100_copy_add_fixture() {
    let version = Command::new(LLVM_MC)
        .arg("--version")
        .output()
        .expect("ROCm LLVM assembler must be installed for this artifact witness");
    assert!(version.status.success(), "llvm-mc --version failed");
    let version_text = String::from_utf8_lossy(&version.stdout);
    assert!(version_text.contains(EXPECTED_ASSEMBLER));
    assert!(version_text.contains(EXPECTED_SOURCE_ID));

    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("wave32_copy_add.s");
    let output_directory = tempfile::tempdir().expect("temporary artifact directory");
    let object = output_directory.path().join("wave32_copy_add.o");

    let assembly = Command::new(LLVM_MC)
        .args([
            "-triple=amdgcn-amd-amdhsa",
            "-mcpu=gfx1100",
            "-filetype=obj",
            "-o",
        ])
        .arg(&object)
        .arg(&fixture)
        .output()
        .expect("llvm-mc must start");
    assert!(
        assembly.status.success(),
        "llvm-mc failed: {}",
        String::from_utf8_lossy(&assembly.stderr)
    );

    let disassembly = Command::new(LLVM_OBJDUMP)
        .arg("-d")
        .arg(&object)
        .output()
        .expect("llvm-objdump must start");
    assert!(
        disassembly.status.success(),
        "llvm-objdump failed: {}",
        String::from_utf8_lossy(&disassembly.stderr)
    );
    let text = String::from_utf8_lossy(&disassembly.stdout);
    assert!(text.contains("7E020300"));
    assert!(text.contains("4A040300"));
    assert!(text.contains("BFB00000"));
    assert!(text.contains("v_mov_b32_e32 v1, v0"));
    assert!(text.contains("v_add_nc_u32_e32 v2, v0, v1"));
    assert!(text.contains("s_endpgm"));
}
