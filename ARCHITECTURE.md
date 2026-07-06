# ARCHITECTURE.md

## Overview

Logismos is a ~27-crate Rust workspace. Crates form a strict DAG - no cycles, no sideways dependencies. Each crate owns exactly one responsibility.

## Tier model

| Tier | Role | Crates |
|------|------|--------|
| T0 Foundation | HIP FFI, error types, stable API surface | `core`, `hipcore` |
| T1 Infrastructure | Kernels, quantization, tokenization, caching | `kernels`, `quant`, `tokenize`, `cache` |
| T2 Model families | Transformer ops, encoder/decoder impls | `transformers`, `encoders`, `decoders`, `loader` |
| T3 Pipelines | End-to-end model pipelines | `embed`, `rerank`, `tts`, `ekphrasis` |
| T4 Serving | Scheduling, sampling, provider adapters | `sched`, `sample`, `praxis`, `hermeneus` |
| T5 Entrypoint | Binary facade | `bin`, `logismos` |

## Dependency rules

- Lower tiers never depend on higher tiers.
- `core` has no logismos-internal deps; all other crates may depend on `core`.
- `hipcore` has no logismos-internal deps beyond `core`.
- `kernels` depends on `hipcore` and `core` only.
- Cross-tier deps must be justified. Within-tier deps are code smell.

## Key invariants

- `INVARIANT: core stable API surface` - `crates/core` exports the traits (`EmbeddingModel`, `Reranker`) that all pipeline crates depend on. Breaking changes require a semver bump.
- `INVARIANT: CPU reference parity` - Every HIP kernel in `crates/kernels` has a CPU reference in the same module. Default tolerance is 1e-3.
- `INVARIANT: HIP FFI boundary` - `crates/hipcore` is the sole point of unsafe HIP FFI. All other crates access the GPU through `hipcore`.
- `INVARIANT: no-silent-fallback` - When a GPU kernel is unavailable, crates return a typed error. They never silently fall back to CPU.

## cfg flags

- `logismos_no_gpu_kernels` - headless/no-HIP build path. All GPU kernel stubs return `Error::NotImplemented`. Used for CI without ROCm hardware.

## Glossary

See `_llm/glossary.md` for domain term definitions.
