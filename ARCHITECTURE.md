# ARCHITECTURE.md

## Overview

Logismos is a Rust workspace for loading, quantization, inference, and serving. Crates form a
strict DAG - no cycles, no sideways dependencies. Each crate owns exactly one responsibility.

## Tier model

| Tier | Role |
|------|------|
| T0 Foundation | HIP FFI, errors, and the stable API surface |
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
- `core` and `hipcore` have no Logismos-local dependencies.
- `taxis` depends locally on `hipcore`.
- `kernels` depends locally on `hipcore` and `taxis`; it does not depend on `core`.
- Cross-tier deps must be justified. Within-tier deps are code smell.

## Key invariants

- `INVARIANT: core stable API surface` - `crates/core` currently exports `EmbeddingModel` as its
  model trait. `Reranker` currently belongs to `crates/rerank`, alongside its implementations;
  architecture documentation must not project it into `core` before the code does. Breaking
  changes to the actual stable surface require a semver bump.
- `INVARIANT: CPU reference parity` - Every HIP kernel in `crates/kernels` has a CPU reference in the same module. Default tolerance is 1e-3.
- `INVARIANT: HIP FFI boundary` - `crates/hipcore` is the sole point of unsafe HIP FFI. All other crates access the GPU through `hipcore`.
- `INVARIANT: no-silent-fallback` - When a GPU kernel is unavailable, crates return a typed error. They never silently fall back to CPU.

## cfg flags

- `logismos_no_gpu_kernels` - build path without compiled HIP kernels. Implemented GPU operations
  return `Error::NoGpuBuild`; genuinely unimplemented operations retain `Error::NotImplemented`.

## Glossary

See `_llm/glossary.md` for domain term definitions.
