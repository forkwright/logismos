// CPU-only LLVM witness; see llvm_artifact.rs for pinned AOMP rebuild and admission.
.text
.globl wave32_mul_f32
.type wave32_mul_f32,@function
wave32_mul_f32:
  v_mul_f32_e32 v11, v3, v11
  s_endpgm
