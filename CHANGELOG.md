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

## [1.0.7](https://github.com/forkwright/logismos/compare/v1.0.6...v1.0.7) (2026-08-16)


### Bug Fixes

* **kernels:** enforce sgemm shape invariant in every build profile ([#100](https://github.com/forkwright/logismos/issues/100)) ([c00bbfc](https://github.com/forkwright/logismos/commit/c00bbfcf918f05ca45bd18dfaab401385f10a8b7))

## [1.0.6](https://github.com/forkwright/logismos/compare/v1.0.5...v1.0.6) (2026-08-16)


### Bug Fixes

* **decode,encoders:** mask NaN in TopK, reuse its buffer; name+test the stella tensor-count guard ([#84](https://github.com/forkwright/logismos/issues/84)) ([312bd81](https://github.com/forkwright/logismos/commit/312bd815ade1c9f3fedae6175f6f394ce9c2a040))
* **embed:** correct stella IO/prompt/shape/core-count defects; add tests ([#87](https://github.com/forkwright/logismos/issues/87)) ([dc35ba9](https://github.com/forkwright/logismos/commit/dc35ba95bb947336d13e0eac7c4323c12aaf18cb))
* **hipcore, praxis:** audit batch — device-context Drop, odd head_dim, stream/table churn ([#85](https://github.com/forkwright/logismos/issues/85)) ([b531436](https://github.com/forkwright/logismos/commit/b531436d605099f8d1edba43e141b2cef952a10e))
* **praxis:** reject mixed HIP/CPU device pairs in matmul and rms_norm ([#97](https://github.com/forkwright/logismos/issues/97)) ([5b4e92b](https://github.com/forkwright/logismos/commit/5b4e92be5911759c001f432b1f003d22de7770d4))
* **taxis,kernels:** close silent-degradation gaps in shape/kernel invariants ([#88](https://github.com/forkwright/logismos/issues/88)) ([e7f1295](https://github.com/forkwright/logismos/commit/e7f129507029ff8070615b3b069b1ed8a7dc88d6))


### Documentation

* correct the residual private claim and portable-path the kanon pointers ([#80](https://github.com/forkwright/logismos/issues/80)) ([5d0c4dd](https://github.com/forkwright/logismos/commit/5d0c4ddb10d96d2965b63e4682812c56f35f7e1e))
* **kanon-root:** complete [#65](https://github.com/forkwright/logismos/issues/65)'s portable-path fix, close the docs-sync gap ([#90](https://github.com/forkwright/logismos/issues/90)) ([f499910](https://github.com/forkwright/logismos/commit/f4999108040057ab0998143b54078d6640bd5c6d))

## [1.0.5](https://github.com/forkwright/logismos/compare/v1.0.4...v1.0.5) (2026-08-15)


### Documentation

* correct push-boundary claim now that logismos is public ([#77](https://github.com/forkwright/logismos/issues/77)) ([a03687b](https://github.com/forkwright/logismos/commit/a03687b1ad8664b72a86f1ca0cde511870d6eedc))

## [1.0.4](https://github.com/forkwright/logismos/compare/v1.0.3...v1.0.4) (2026-08-09)


### Bug Fixes

* **kernels:** guard softmax_last_dim against a fully-masked row's NaN ([#66](https://github.com/forkwright/logismos/issues/66)) ([e48f2ff](https://github.com/forkwright/logismos/commit/e48f2fff40fc6a486a2646c2dff344078da7722d)), closes [#30](https://github.com/forkwright/logismos/issues/30) [#41](https://github.com/forkwright/logismos/issues/41)
* **kernels:** make embed_lookup's out-of-range contract explicit and uniform ([#73](https://github.com/forkwright/logismos/issues/73)) ([06fa164](https://github.com/forkwright/logismos/commit/06fa1646a4675ee6bff97ca0d7300bcfb29366d0)), closes [#55](https://github.com/forkwright/logismos/issues/55)
* **loader:** reject unbounded GGUF allocations, array nesting, and dims overflow ([#67](https://github.com/forkwright/logismos/issues/67)) ([ee7fa09](https://github.com/forkwright/logismos/commit/ee7fa0994d1e3dd5fa4a39a0b6716d3b1943f46a)), closes [#34](https://github.com/forkwright/logismos/issues/34) [#35](https://github.com/forkwright/logismos/issues/35) [#36](https://github.com/forkwright/logismos/issues/36) [#37](https://github.com/forkwright/logismos/issues/37)
* **transformers:** make attention slicing and rope gather bounds-checked ([#74](https://github.com/forkwright/logismos/issues/74)) ([fc1cb88](https://github.com/forkwright/logismos/commit/fc1cb882ebaba57ae2d490c74bbb45cb54ce3d26)), closes [#61](https://github.com/forkwright/logismos/issues/61) [#28](https://github.com/forkwright/logismos/issues/28)


### Documentation

* correct max_tokens contract and document LOGISMOS_SKIP_HIP_BUILD ([#72](https://github.com/forkwright/logismos/issues/72)) ([87080f3](https://github.com/forkwright/logismos/commit/87080f375fe0562e96ed6267efd6ab72b8928984)), closes [#50](https://github.com/forkwright/logismos/issues/50) [#48](https://github.com/forkwright/logismos/issues/48)

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
