// CPU-only artifact provenance:
// source: this file
// target: amdgcn-amd-amdhsa / gfx1100
// assembler: /usr/lib64/rocm/llvm/bin/llvm-mc, AOMP-18.0-12,
// Source ID 18.0-12-ce1873ac686bb90ddec72bb99889a4e80e2de382.
// The integration test rebuilds this file and disassembles it with llvm-objdump.

.text
.globl wave32_copy_add
.type wave32_copy_add,@function
wave32_copy_add:
  v_mov_b32_e32 v1, v0
  v_add_nc_u32_e32 v2, v0, v1
  s_endpgm
