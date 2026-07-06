# Changelog

All notable changes to logismos are recorded here.

## Unreleased

### Added

- Phase 5 CPU reranker: `ModernBertCpuReranker` and `ModernBertEncoder` CPU fp32 forward pass (#47)
- Phase 6a GDN CPU reference: `gated_delta_rule::cpu` with `chunk_gated_delta_rule` and `fused_recurrent_gated_delta_rule` (#47)
- Phase 6b TurboQuant: `encode_turbo3_0_head` / `decode_turbo3_0_head` with Lloyd-Max codebook (#46)
- Phase 6a GDN API contract and preflight research doc (#45)
- Phase 5 GteReranker: `Reranker` trait, `GteReranker` scaffold, `ModernBertConfig` (#43)
- Phase 5 TurboQuant block contract: `Turbo3Block` type (#42)
- GDN ROCm preflight diff research doc (#40)
- ModernBERT preflight: tightened preflight validation (#41)
- ModernBERT reranker contract: `ModernBertPreflight`, full config deserialization (#37)
- Phase 3 test fixtures: Stella parity gates split into standalone fixture crate (#38)
- GDN AITER preflight research doc (#36)

### Fixed

- `hipcore::Device`: use `hipDeviceProp_t` / `hipGetDeviceProperties` (R0600 aliases removed for ROCm 6.x)
- Kernel result defaults: explicit handling in cfg-gated branches
- Matmul CPU helper visibility: narrowed to `pub(crate)`

## 0.0.0

Initial workspace bootstrap. Phase 0: crate scaffolding, forge CI baseline, license configuration.
