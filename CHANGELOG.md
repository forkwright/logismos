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

## [1.0.3](https://github.com/forkwright/logismos/compare/v1.0.2...v1.0.3) (2026-08-04)


### Bug Fixes

* **kernels:** guard mask_additive_in_place against div-by-zero ([#62](https://github.com/forkwright/logismos/issues/62)) ([b9ff95d](https://github.com/forkwright/logismos/commit/b9ff95dc85187696c9969a401f3c876893e2b116))

## [1.0.2](https://github.com/forkwright/logismos/compare/v1.0.1...v1.0.2) (2026-08-03)


### Bug Fixes

* **encoders:** correct ModernBERT global-attention schedule and layer-0 identity norm ([#22](https://github.com/forkwright/logismos/issues/22)) ([8347721](https://github.com/forkwright/logismos/commit/83477217d08736e3746f648463af6f87271867d3))
* **encoders:** give StellaLayer its own forward instead of inlining it in the encoder ([#19](https://github.com/forkwright/logismos/issues/19)) ([12bfd17](https://github.com/forkwright/logismos/commit/12bfd178ecb34a4c54c45d6a469d6a380c5a410d)), closes [#3](https://github.com/forkwright/logismos/issues/3)
* **gate-attestation:** bind Gate-Passed check to PR tip commit ([#23](https://github.com/forkwright/logismos/issues/23)) ([293392c](https://github.com/forkwright/logismos/commit/293392ccaefbad9f4cd3abd2b7297edb2fbbca2e)), closes [#2399](https://github.com/forkwright/logismos/issues/2399)
* remove indexing panics across kernels, quant, encoders and transformers ([#17](https://github.com/forkwright/logismos/issues/17)) ([5ed3886](https://github.com/forkwright/logismos/commit/5ed388699f0f8f6e5aca099888ed81894e9a9fa0)), closes [#3](https://github.com/forkwright/logismos/issues/3)

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
