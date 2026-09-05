# ARCHITECTURE.md

## Overview

Logismos is a Rust workspace for loading, quantization, inference, and serving. Crates form a
strict DAG - no cycles, no sideways dependencies. Each crate owns exactly one responsibility.

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
| T4 Serving | Scheduling, sampling, and provider adapters |
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
- `hipcore`, `placement`, and `emulation` depend on `isa` so target-architecture
  identity and suffix syntax have one implementation.
- `placement` has no HIP/device-runtime dependency; `bin` consumes it without
  linking the device runtime for the `plan` command.
- `emulation` is a CPU test aid, not a production device backend.
- `taxis` depends locally on `hipcore`.
- `kernels` depends locally on `hipcore` and `taxis`; it does not depend on `core`.
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
an optimal placement claim, or permission to stop another workload.

The host-mode compiler consumes resolved inference contracts rather than maintaining its own
model-memory formula. Host inventories, external GPU consumers and operator policy stay private
and outside the inference runtime's authority.

## cfg flags

- `logismos_no_gpu_kernels` - build path without compiled HIP kernels. Implemented GPU operations
  return `Error::NoGpuBuild`; genuinely unimplemented operations retain `Error::NotImplemented`.

## Glossary

See `_llm/glossary.md` for domain term definitions.
