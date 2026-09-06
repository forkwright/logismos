# ARCHITECTURE.md

## Overview

Logismos is a Rust workspace for loading, quantization, inference, and serving. Crates form a
strict DAG with no cycles and explicitly justified tier edges. Each crate owns one responsibility.

The product direction is an agent-aware operating environment for local AI compute. Aletheia
supplies intent; Logismos owns inference semantics and execution within granted resources;
Arche/Tropos and systemd retain host modes and process supervision. The pure planning interface
must not initialize a GPU, start a process, or reserve physical memory merely by returning a plan.

## Tier model

| Tier | Role |
|------|------|
| T0 Foundation | HIP FFI, errors, stable model API, and pure resource contracts |
| T1 Infrastructure | Kernels, quantization, tokenization, loading, and caching |
| T2 Model families | Transformer operations and encoder/decoder implementations |
| T3 Pipelines | End-to-end inference pipelines |
| T4 Serving | Scheduling, admission/residency coordination, sampling, and provider adapters |
| T5 Entrypoint | Integration facade and binary |

The exact crate inventory derives from `cargo metadata --format-version 1 --no-deps --locked`; it
is intentionally not copied into this document. [`contracts/runtime-scope.toml`](contracts/runtime-scope.toml)
records the product boundary. The CI guard enforces its declared fields, locked workspace graph,
retired path/package/lock absence, and license coherence. Review still decides whether future code
semantically respects that boundary.

## Dependency rules

- Lower tiers never depend on higher tiers.
- `core` and `isa` have no Logismos-local dependencies. `isa` is
  pure parsing over the checked-in target token; it neither links nor probes HIP.
  The `core` package exposes the Rust library target `logismos_core` to avoid
  colliding with Rust's standard library; the facade remains `logismos::core`.
- `hipcore`, `placement`, and `emulation` depend on `isa` so target-architecture
  identity and suffix syntax have one implementation.
- `placement` and `sched` have no HIP/device-runtime dependency. `bin` consumes
  `placement` for `plan` and metadata-only `loader` for `inspect` without
  linking the device runtime.
- `decoders` consumes `loader` without its tensor adapter and consumes `quant`
  for explicit CPU row projection, without linking HIP. Its recurrent-attention
  path uses the standalone CPU `kernels` graph. Structural profiles remain
  distinct from payload-bound execution. The lower-level `quant` crate owns
  executable block and row geometry; inspection and projection reuse that owner.
- `emulation` is a CPU test aid, not a production device backend.
- `taxis` depends locally on `hipcore`.
- `kernels/gpu` enables the local `hipcore` and `taxis` dependencies and GPU
  launcher modules, including their nested parity references. Standalone
  `cpu_f32`, `gdn`, and `causal_conv` remain available without that feature;
  `transformers` selects that CPU-only graph, while `praxis` explicitly enables
  GPU launchers. Direct `kernels` users retain the default GPU feature. The
  crate does not depend on `core`.
- Cross-tier deps must be justified. Within-tier deps are code smell.

## Key invariants

- `INVARIANT: core stable API surface` - `crates/core` currently exports `EmbeddingModel` as its
  model trait. `Reranker` currently belongs to `crates/rerank`, alongside its implementations;
  architecture documentation must not project it into `core` before the code does. Breaking
  changes to the actual stable surface require a semver bump.
- `INVARIANT: CPU reference parity` - Every HIP kernel in `crates/kernels` has a CPU reference in the same module. Default tolerance is 1e-3.
- `INVARIANT: hardware-access boundary` - `crates/hipcore` owns unsafe device/runtime access.
  HIP is the production provider; the approved HSA/ROCr experiment remains behind this boundary.
  Neither pure planning nor test simulation depends on initializing that provider.
- `INVARIANT: no-silent-fallback` - When a GPU kernel is unavailable, crates return a typed error. They never silently fall back to CPU.

## Resource ownership

Stable device identities are distinct from visible ordinals. The configured base architecture,
not SKU names or a 48-GB threshold, determines current admission. Validated feature suffixes are
descriptive input, not an independent semantic-support qualification. Every allocation and resource
estimate belongs to one device; an optional absent device cannot disable an otherwise valid
single-device plan.

Artifact identity, execution-profile requests, memory estimates, host allowances, and observed
residency are separate facts. Repeated profiles reference one artifact identity. Byte arithmetic
is checked. A successful resource plan is an admission calculation, not a physical reservation,
an optimal placement claim, or permission to stop another workload. The CPU-only
`placement::ReservationLedger` and `sched::Scheduler` add process-local,
opaque-capability admission accounting after planning; they do not establish
unique host ownership, allocate a device, or establish physical residency.
Their detailed state protocol and proof limits belong to the `sched` rustdoc
and the operator-managed private planning corpus, not this overview.

The host-mode compiler consumes resolved inference contracts rather than maintaining its own
model-memory formula. Host inventories, external GPU consumers and operator policy stay private
and outside the inference runtime's authority.

## Native payload ownership

`loader::gguf::VerifiedArtifact` owns one immutable serialized backing, admitted
under an explicit byte limit and matched against a required SHA-256 expectation.
Its metadata and tensor borrows come from those same bytes. This content binding
does not establish publisher authenticity, a filesystem snapshot, or a total
host-memory reservation. Existing observation receipts remain reporting data.

`decoders::Qwen35Weights` binds the existing structural contract to that owner
and executes named F32, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_NL, and IQ4_XS matrix
projections through `quant`. Linear row decoding shares those checked block
decoders and geometry; other unsupported formats are explicit refusals.
`Qwen35RecurrentExecution` binds convolution history and GDN state to those
weights and one recurrent layer; an unsuccessful step commits neither state.
It adapts GGUF tiled heads to the generic grouped GDN contract explicitly.
Execution requires finite positive RMS epsilon; structural recognition does not.
The shared CPU RMSNorm returns typed errors for malformed inputs and non-finite
arithmetic instead of concealing overflow behind finite zero outputs.

This is a recurrent-attention path, not a complete decoder block or model:
residual/FFN composition, full attention, tokenizer/logits, NextN, artifact-level
quality, and admission integration remain separate requirements. Explicit CPU
execution is not a fallback for a GPU operation.

## cfg flags

- `logismos_no_gpu_kernels` - build path without compiled HIP kernels. Implemented GPU operations
  return `Error::NoGpuBuild`; genuinely unimplemented operations retain `Error::NotImplemented`.

## Glossary

See `_llm/glossary.md` for domain term definitions.
