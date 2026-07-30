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

## [1.0.1](https://github.com/forkwright/logismos/compare/v1.0.0...v1.0.1) (2026-07-30)


### Bug Fixes

* **hipcore:** build on Debian-family distros and any installed ROCm revision ([#15](https://github.com/forkwright/logismos/issues/15)) ([070ca43](https://github.com/forkwright/logismos/commit/070ca4311bad86c5e3f7d7821306b6c113e920d3))

## 1.0.0 (2026-07-22)


### Bug Fixes

* **ci:** security workflow — audit-ignore parity + workspace wildcard-path allowance ([#1](https://github.com/forkwright/logismos/issues/1)) ([00c05e4](https://github.com/forkwright/logismos/commit/00c05e42e1b1fe6c67dc8e7a0c3012e2e3c4de3c))


### Documentation

* **repo:** sync kanon-generated context blocks into CLAUDE.md + README.md ([#2](https://github.com/forkwright/logismos/issues/2)) ([#2](https://github.com/forkwright/logismos/issues/2)) ([94e4e97](https://github.com/forkwright/logismos/commit/94e4e97dce6eee8f29098e792b2790b709778b0b))

## 0.0.0

Initial workspace bootstrap. Phase 0: crate scaffolding, forge CI baseline, license configuration.
