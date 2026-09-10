# Changelog

All notable changes to logismos are recorded here.

## [2.0.0](https://github.com/forkwright/logismos/compare/v1.0.11...v2.0.0) (2026-09-10)


### ⚠ BREAKING CHANGES

* **runtime:** supporting taxis/cache/kernel APIs now return typed failures where construction or numerical success was previously unchecked. `Shape`/`Layout`/`Tensor` use checked element counts; `DType` uses checked byte counts; `Tensor::from_cpu`, `Layout::from_parts`, `CacheLayout::new` and `FlatKvCache::new` are fallible. The old alternate `try_from_cpu` entry is consolidated, public cache fields are private, fp32 softmax and unit normalization return `Result`, and embedding normalization retains the shared typed kernel cause. In-tree consumers and tests are migrated.
* **text:** decode processors, samplers and chain methods now return typed errors; policy construction is validated. Empty, NaN, positive-infinity and fully masked logits no longer silently select token zero. The in-tree consumer is migrated. The stable core trait/error surface is unchanged.
* **runtime:** the direct Rust library target changes from `core` to `logismos_core`. Cargo package identity and public facade/traits/errors are unchanged. Direct consumers using the old library-target import must migrate; existing project-qualified dependency aliases remain valid.
* **runtime:** cpu_f32::rms_norm returns kernels::Result<Vec<f32>>; transformers::Error::Kernel exposes a typed source. The unreleased Qwen35Weights::project_q8_0 is replaced by descriptor-driven project. Stable core traits, HIP FFI, and tolerance defaults are unchanged.

### Features

* add agent-aware admission and artifact observations ([#154](https://github.com/forkwright/logismos/issues/154)) ([f558245](https://github.com/forkwright/logismos/commit/f558245960b65f1e96585170788aa0a54966caaf))
* add bounded native Qwen3 CPU reranking ([#165](https://github.com/forkwright/logismos/issues/165)) ([e9b6537](https://github.com/forkwright/logismos/commit/e9b65371f066f22591a3f79a0a024486b89372e0))
* add observation-bound model structure and q8 decoding ([#156](https://github.com/forkwright/logismos/issues/156)) ([f6a21d7](https://github.com/forkwright/logismos/commit/f6a21d707a683f03a3dcc337a1882d6d0f6c06cb))
* bind native host accounting and recycled logits ownership ([#191](https://github.com/forkwright/logismos/issues/191)) ([c75a1d9](https://github.com/forkwright/logismos/commit/c75a1d915671c94f5e547d3c61c6671efad5d68f))
* bind native q8 execution to verified artifacts ([#158](https://github.com/forkwright/logismos/issues/158)) ([109730d](https://github.com/forkwright/logismos/commit/109730d77c8791c70687a350553724ecf621c609))
* bind shared native residency and prepared request ownership ([#189](https://github.com/forkwright/logismos/issues/189)) ([e9a7238](https://github.com/forkwright/logismos/commit/e9a72383809a4006e564030fdae2207da64b6024))
* **cache:** integrate execution-private paged KV ([#183](https://github.com/forkwright/logismos/issues/183)) ([ea920d5](https://github.com/forkwright/logismos/commit/ea920d577b9739a5741b3027772bfa467a899fec))
* compose native gfx1100 full-attention layers ([#185](https://github.com/forkwright/logismos/issues/185)) ([6d602a3](https://github.com/forkwright/logismos/commit/6d602a394d186d66ea8dfeb0d8cb6c616ec1769b))
* **decoders:** execute bounded native model chunks ([#196](https://github.com/forkwright/logismos/issues/196)) ([4e28481](https://github.com/forkwright/logismos/commit/4e2848185e51c2b72b9a4c337d76bf8b5e81b7cd))
* **decoders:** publish CPU sequence batches atomically ([#198](https://github.com/forkwright/logismos/issues/198)) ([2d324f9](https://github.com/forkwright/logismos/commit/2d324f92330303ba9eeaa83c4b768550b677bbea))
* **decoders:** stage native single-sequence recurrent chunks ([#195](https://github.com/forkwright/logismos/issues/195)) ([be44c41](https://github.com/forkwright/logismos/commit/be44c41459d252d4fe5cac503d5da1167c35fc84))
* derive native Qwen3 CPU retrieval requirements ([#166](https://github.com/forkwright/logismos/issues/166)) ([2263ff4](https://github.com/forkwright/logismos/commit/2263ff4e7ba130cb32764f659c8d9eaa24488397))
* **embed:** execute bounded native Qwen3 CPU embeddings ([#164](https://github.com/forkwright/logismos/issues/164)) ([eab451b](https://github.com/forkwright/logismos/commit/eab451b89387be587ffff4d32a69ef884941e9f7)), closes [#150](https://github.com/forkwright/logismos/issues/150)
* establish agent-aware gfx1100 compute foundations ([a0f4250](https://github.com/forkwright/logismos/commit/a0f42509403290ce3d381e718c9ab776e5268074))
* execute native gfx1100 hybrid main models ([#187](https://github.com/forkwright/logismos/issues/187)) ([0c1e4a9](https://github.com/forkwright/logismos/commit/0c1e4a965f3bd57f331a365749e2e4850c08b91a))
* expose verified native text preparation ([#176](https://github.com/forkwright/logismos/issues/176)) ([aeb6211](https://github.com/forkwright/logismos/commit/aeb6211ade98cff2187fb8f62471bc9d96eab6e1))
* **hermeneus:** execute bounded native prompt chunks ([#197](https://github.com/forkwright/logismos/issues/197)) ([08cfa58](https://github.com/forkwright/logismos/commit/08cfa5899a6982c8fc69d1e2abc9f491d23324a4))
* inspect exact IQ4 artifacts and extend bounded emulation ([#155](https://github.com/forkwright/logismos/issues/155)) ([fee3c55](https://github.com/forkwright/logismos/commit/fee3c556a590203c92317d5cc927f7b3c6dbd5f1))
* **kernels:** add staged causal convolution HIP step ([#181](https://github.com/forkwright/logismos/issues/181)) ([022436f](https://github.com/forkwright/logismos/commit/022436fa0024ba09a4d3e7e2afc8c5a931b4b16c))
* **kernels:** add staged grouped GDN decode step ([#180](https://github.com/forkwright/logismos/issues/180)) ([4e7c2da](https://github.com/forkwright/logismos/commit/4e7c2da23a0b9c08c8388c7e13b90e21d33df2c2))
* **kernels:** compose checked packed CPU prefill ([#194](https://github.com/forkwright/logismos/issues/194)) ([e6dd128](https://github.com/forkwright/logismos/commit/e6dd128a742a64a8bb1fcc961615f0df90247964)), closes [#46](https://github.com/forkwright/logismos/issues/46) [#150](https://github.com/forkwright/logismos/issues/150)
* **kernels:** generalize native serialized-row GEMV ([#182](https://github.com/forkwright/logismos/issues/182)) ([8af8165](https://github.com/forkwright/logismos/commit/8af8165b98244b5ea7396d5c0d3c5a22bc613fd5))
* **kernels:** integrate checked paged decode attention ([#184](https://github.com/forkwright/logismos/issues/184)) ([52dde57](https://github.com/forkwright/logismos/commit/52dde571de649fdbf0d2baf55b3ebdff23df104a))
* own shared native text execution and teardown ([#192](https://github.com/forkwright/logismos/issues/192)) ([6eb1a12](https://github.com/forkwright/logismos/commit/6eb1a12c5520248f6f4c62fa9d798c8f1e575767))
* prepare native recurrent hybrid composition ([#186](https://github.com/forkwright/logismos/issues/186)) ([32fcf8b](https://github.com/forkwright/logismos/commit/32fcf8b55a1b5d71eac90f53b995d9a48a56785c))
* prepare native text requests and Q8 GPU projection ([#167](https://github.com/forkwright/logismos/issues/167)) ([bb1e746](https://github.com/forkwright/logismos/commit/bb1e7468c3397d2c74ec27f5798d69bb9df3bd81))
* preserve native and generated-output resource custody ([#190](https://github.com/forkwright/logismos/issues/190)) ([6d4156f](https://github.com/forkwright/logismos/commit/6d4156f3653886de57b18ab9bd048272d59255bb))
* refuse invalid native arithmetic before publication ([#188](https://github.com/forkwright/logismos/issues/188)) ([981ce29](https://github.com/forkwright/logismos/commit/981ce290cf0d8d48363f336b200a80ed90139e94))
* **runtime:** execute mixed-weight recurrent CPU paths ([#159](https://github.com/forkwright/logismos/issues/159)) ([ec69f3d](https://github.com/forkwright/logismos/commit/ec69f3dbe4e2d319d8c095a1a4767e986cd7c804))
* **runtime:** execute native hybrid CPU model paths ([#162](https://github.com/forkwright/logismos/issues/162)) ([e1b5203](https://github.com/forkwright/logismos/commit/e1b5203b0afbc09bdd48ce57d42ae7977693bbb3))
* **sched:** retire admissions without revoking peer workloads ([#179](https://github.com/forkwright/logismos/issues/179)) ([9b1d7bd](https://github.com/forkwright/logismos/commit/9b1d7bd4ba45338868246fc5ccaa8b2a5046fb44)), closes [#174](https://github.com/forkwright/logismos/issues/174)
* **text:** add bounded native CPU generation ([#163](https://github.com/forkwright/logismos/issues/163)) ([8fe4e6d](https://github.com/forkwright/logismos/commit/8fe4e6d8c64f0a7f7e6e851548c0a6d63a127768))


### Bug Fixes

* **ci:** repair release-checks caller permission ([041dee5](https://github.com/forkwright/logismos/commit/041dee511bbd3c1cef2f89ef56dfb5e675acce01))
* **runtime:** enforce checked tensor and numerical boundaries ([#193](https://github.com/forkwright/logismos/issues/193)) ([51b965d](https://github.com/forkwright/logismos/commit/51b965dde7e06170cf02c142540a31867a560a7c))


### Documentation

* point planning guidance at private corpus ([#157](https://github.com/forkwright/logismos/issues/157)) ([cb54330](https://github.com/forkwright/logismos/commit/cb543306046351a630348999152d397fb8304ea0))

## [1.0.11](https://github.com/forkwright/logismos/compare/v1.0.10...v1.0.11) (2026-09-03)


### Documentation

* Phase 4 unblocked — menos (W7900) returned to service ([#148](https://github.com/forkwright/logismos/issues/148)) ([3812fba](https://github.com/forkwright/logismos/commit/3812fbabc66c4f49b24e9f338048560ef9aed138))

## [1.0.10](https://github.com/forkwright/logismos/compare/v1.0.9...v1.0.10) (2026-08-26)


### Bug Fixes

* **ci:** name the sha's real version on every action pin ([#132](https://github.com/forkwright/logismos/issues/132)) ([84d0fbe](https://github.com/forkwright/logismos/commit/84d0fbe990237dbdea6f46fe8c2a203d4e95e1e1))
* **deps:** regenerate Cargo.lock for bindgen 0.72 and rand 0.9 ([#139](https://github.com/forkwright/logismos/issues/139)) ([51bd2c1](https://github.com/forkwright/logismos/commit/51bd2c119a19040699d3631c5fd107cbd10d22de))
* **license:** converge remaining metadata on Noncommercial, not Shield ([#138](https://github.com/forkwright/logismos/issues/138)) ([d32ab01](https://github.com/forkwright/logismos/commit/d32ab01574d55e3322f390680d2eb5d085f1ab49))


### Refactoring

* **scope:** retire training marker crates ([#137](https://github.com/forkwright/logismos/issues/137)) ([dbcb9aa](https://github.com/forkwright/logismos/commit/dbcb9aaba0d2975dcc89fc4bfd9e33091f42e1e4)), closes [#131](https://github.com/forkwright/logismos/issues/131)

## [1.0.9](https://github.com/forkwright/logismos/compare/v1.0.8...v1.0.9) (2026-08-21)


### Refactoring

* **errors:** migrate eleven crates from thiserror to snafu ([#111](https://github.com/forkwright/logismos/issues/111)) ([3131e13](https://github.com/forkwright/logismos/commit/3131e138d8b410e1f22d3475b7fdf9d592907094))

## [1.0.8](https://github.com/forkwright/logismos/compare/v1.0.7...v1.0.8) (2026-08-17)


### Bug Fixes

* **hipcore:** protect destination buffer in copy_from_host_async ([#110](https://github.com/forkwright/logismos/issues/110)) ([a6ed8c5](https://github.com/forkwright/logismos/commit/a6ed8c5edd6594566d028337b59e74102f0fa54e))


### Documentation

* **gate-attestation:** document hipcore's ROCm build requirement and CI fallback ([#78](https://github.com/forkwright/logismos/issues/78)) ([43abb58](https://github.com/forkwright/logismos/commit/43abb589ebeaa1e9400d78f9b1dd0f85fcfc01ef)), closes [#14](https://github.com/forkwright/logismos/issues/14)

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
