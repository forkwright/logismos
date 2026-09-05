//! CPU rebuild witness for the native raw-instruction fixture.

use std::path::PathBuf;
use std::process::Command;

use emulation::elf::inspect_relocatable_text;

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

    let object_bytes = std::fs::read(&object).expect("rebuilt object must be readable");
    let relocatable = inspect_relocatable_text(&object_bytes)
        .expect("LLVM relocatable fixture meets the inspect-only contract");
    assert_eq!(
        relocatable.text(),
        [
            0x00, 0x03, 0x02, 0x7e, 0x00, 0x03, 0x04, 0x4a, 0x00, 0x00, 0xb0, 0xbf,
        ]
    );
    let execution = relocatable
        .into_wave32_program(vec![[0; 32]; 3], 3)
        .expect("admitted text meets raw dispatch bounds")
        .execute()
        .expect("admitted text uses only the implemented instruction forms");
    assert_eq!(execution.coverage().end_program_count(), 1);
}

#[test]
#[ignore = "requires the explicitly pinned local AOMP llvm-mc/llvm-objdump witness"]
#[allow(
    clippy::expect_used,
    reason = "the explicit witness names unavailable external prerequisites"
)]
fn rebuilds_disassembles_and_admits_gfx1100_wmma_fixture() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wave32_wmma.s");
    let directory = tempfile::tempdir().expect("temporary artifact directory");
    let object = directory.path().join("wave32_wmma.o");
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
    assert!(disassembly.status.success());
    let text = String::from_utf8_lossy(&disassembly.stdout);
    assert!(text.contains("CC404018"));
    assert!(text.contains("1C421100"));
    assert!(text.contains("v_wmma_f32_16x16x16_f16 v[24:31], v[0:7], v[8:15], v[16:23]"));
    let bytes = std::fs::read(&object).expect("object readable");
    let admitted = inspect_relocatable_text(&bytes).expect("relocatable WMMA object admitted");
    assert_eq!(
        &admitted.text()[..8],
        [0x18, 0x40, 0x40, 0xcc, 0x00, 0x11, 0x42, 0x1c]
    );
}
