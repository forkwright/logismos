// CPU-only LLVM witness; see llvm_artifact.rs for pinned AOMP rebuild and admission.
.text
.globl wave32_wmma
.type wave32_wmma,@function
wave32_wmma:
  v_wmma_f32_16x16x16_f16 v[24:31], v[0:7], v[8:15], v[16:23]
  s_endpgm
