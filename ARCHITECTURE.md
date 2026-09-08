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
| T1 Infrastructure | Kernels, quantization, tokenization, loading, caching, and CPU decode policy |
| T2 Model families | Transformer operations and encoder/decoder implementations |
| T3 Pipelines | End-to-end inference pipelines |
| T4 Serving | Scheduling, admission/residency coordination, and provider adapters |
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
  `placement` for `plan`, `loader` for `inspect`, and `text` for `prepare-text`
  without linking the device runtime.
- `decoders` consumes `loader` without its tensor adapter and consumes `quant`
  for explicit CPU row projection, without linking HIP. Its recurrent-attention
  path uses the standalone CPU `kernels` graph. Structural profiles remain
  distinct from payload-bound execution. The lower-level `quant` crate owns
  executable block and row geometry; inspection and projection reuse that owner.
- `emulation` is a CPU test aid, not a production device backend.
- `decode` owns checked logit processing and token selection over CPU slices.
  It has no tensor/device-runtime dependency; a pipeline can consume it without
  depending on a serving layer. Empty, NaN, positive-infinity and fully masked
  rows fail explicitly. Negative infinity is the masking representation, not
  an implicit fallback to token zero.
- `text` is a CPU-only pipeline over `decoders`, `tokenize` and `decode`, with
  `loader`'s verified-GGUF artifact surface, its optional tensor adapter
  disabled, and a restricted template substrate. It does
  not depend on scheduling, a provider adapter, or a device runtime.
- `test-fixtures` is dev-only shared synthetic GGUF support. It has no model
  execution dependency and is never a production dependency of a pipeline.
- `embed` consumes the CPU-only `decoders` and verified-tokenizer paths for
  native Qwen3 embeddings. Its default `stella` feature preserves the existing
  Stella API and tensor/encoder graph; direct consumers disable default
  features to exclude that accelerator-capable graph. Consumers still use the
  unchanged `core::EmbeddingModel` contract.
- `rerank` consumes the same CPU decoder/tokenizer graph and `templates` for
  native Qwen3 pair scoring. Its default `modernbert` feature preserves the
  existing encoder implementation; disabling default features removes that
  accelerator-capable graph without changing the `Reranker` contract.
- `templates` owns bounded, capability-free artifact-template rendering for
  `text` and `rerank`. It has no model, GGUF, tokenizer or device dependency;
  pipelines retain artifact binding, typed message roles and token policy.
- `tokenize` owns exact vocabulary/special-ID verification and refusal of
  configured tokenizer padding or truncation. Native text, embedding and
  reranking setup invoke that guard; ordinary tokenizer consumers retain their
  configured behavior. Disabling automatic special tokens alone does not
  disable padding or truncation.
- `taxis` depends locally on `hipcore`.
- `kernels/gpu` enables the local `hipcore` and `taxis` dependencies and GPU
  launcher modules, including their nested parity references. Standalone
  `cpu_f32`, `gdn`, and `causal_conv` remain available without that feature;
  `transformers` selects that CPU-only graph, while `praxis` explicitly enables
  GPU launchers. Direct `kernels` users retain the default GPU feature. The
  crate does not depend on `core`.
- `kernels::q8_0_gemv` depends on `quant` as the lower format owner. Its checked
  shape and CPU reference reuse Q8_0 row geometry and execution; its build
  dependency generates HIP layout constants from that same authority. This
  within-tier edge replaces duplicate format definitions and remains HIP-free
  when GPU features are disabled.
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

The standalone Q8_0 GEMV primitive accepts raw row-major quantized matrix bytes
and f32 activations/output. A checked opaque shape owns exact extents and ABI
bounds. The CPU reference delegates each row to `quant`; the HIP implementation
uses one sequential thread per row with source-specific floating-point controls.
Its unsafe asynchronous launcher requires valid device buffers, lifetimes,
nonaliasing and admitted finite arithmetic. CPU typed nonfinite refusals do not
imply device-result validation. Its GPU domain requires zero-or-normal scales,
operands and intermediates; CPU subnormal witnesses do not qualify GPU denormal
modes. This is a correctness-oriented primitive, not a
whole-model GPU path, performance result or hardware qualification.

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

`Qwen35Execution` composes the main recurrent/full-attention blocks, residuals,
dense FFN, final normalization and vocabulary projection from token IDs. One
ordered layer-state enum owns recurrent or KV history, never parallel optional
states. Its caller-supplied context bound is capped by artifact and signed-position
limits. Whole-call staging is fallible and commits only after all token logits
succeed. Its text-only interleaved RoPE supports checked full or partial rotary
dimensions; unsupported effective scaling fails explicitly.

`Qwen35ExecutionPlan` validates caller context and step bounds and chooses
all-token or last-token logits. Its precomputed `Qwen35CpuRequirements` derives
logical `f32` backing from allocation owners, not a separate estimator.
Causal-convolution and grouped-GDN plans are consumed by their kernels;
recurrent, full-attention, FFN and LM-head owners compose named allocation
phases. The report separates retained state, its transaction clone, transient
workspace upper bound, and returned logits. Serialized verified backing is
reported separately. Structure/stack storage, allocator overhead and capacity,
template/tokenizer allocations, process RSS, physical residency and GPU memory
are outside this report. It is neither an allocation guarantee nor device
admission input.

Independent synthetic f64 witnesses cover the composed mixed-quantized model,
retained recurrent/KV/position state, continuation and rollback. Deliberately
incorrect format/order/operator paths must be distinguishable at the same
comparison tolerance; batch-versus-sequential agreement alone is insufficient.

`text::TextPipeline` binds an explicitly selected tokenizer identity to the
verified model's vocabulary and special-token policy. The same model digest
commits the embedded template; no second template identity authority exists.
Its private template environment registers no named templates or loader and
renders only the admitted source. It accepts typed text messages and uses
checked greedy selection followed by collective sequence decoding. Requests
have independent execution state, bounded context/output and cooperative
cancellation checks; failures publish neither partial text nor resumable state.
Completed internal decoder steps are discarded with that private session, not
undone in a shared session. These limits do not bound total template/tokenizer
heap use or authenticate the selected model/tokenizer's publisher.

`TextPipeline::prepare` returns an opaque `PreparedGeneration` that owns the
rendered prompt and final token IDs, borrows the immutable pipeline, and retains
one `Qwen35ExecutionPlan`. Its context is checked prompt plus output tokens;
its maximum step is the nonempty prompt length. Configured limits are ceilings,
while the actual request must fit the artifact. Read-only getters expose these
exact inputs, tokenizer identity and decoder requirements. Preparation performs
no decoder-session allocation or model operation. Consuming generation drops the
rendered prompt and checks cancellation before allocating a fresh session;
ordinary generation delegates to this same path. Cancellation is cooperative,
so preparation may finish inertly if cancellation arrives during tokenization.
The decoder report excludes rendered text, u32 prompt/generated IDs, tokenizer
and decoded strings; it is not a whole-request estimate or admission grant.

The HIP-free binary's `prepare-text` adapter uses this same owner and never
constructs a decoder session. Trusted CLI arguments supply exact model and
tokenizer identity expectations, while a bounded strict JSON argument supplies
the request and explicit limits. The adapter bounds tokenizer input by its
expected serialized length and reuses the verified artifact's immutable
backing; it does not replace that backing with an inspection receipt. Structured
failures omit input paths. Successful receipts intentionally disclose rendering
and token IDs and are not an independent correctness oracle. Neither the
command nor the paired-file denied runner supplies a host resource reservation.

NextN, serving, exact-artifact quality, physical residency and admission
integration remain separate requirements.
Explicit CPU execution is not a fallback for a GPU operation.

## Native embedding ownership

`decoders::Qwen3Weights` binds a causal Qwen3 embedding profile to the same
verified-payload owner. Qwen3 and Qwen3.5 share private checked GGUF matrix
access, including its allocation geometry; model-specific attention and
positional contracts remain separate. Qwen3 admits the bounded F32/Q8_0
matrix profile, derives projection widths independently of hidden width, and
uses per-head Q/K RMS normalization followed by split-half RoPE. One stateless
CPU call executes all causal blocks and returns the final-RMS terminal hidden
row. Inputs are unpadded token sequences, not a padded batch or persistent KV
session. Extra output/classifier heads and unsupported metadata are refused.

`embed::qwen3::Qwen3EmbeddingModel` owns text/prefix/token policy and full-width
L2-normalized output through the stable embedding trait. `tokenize` owns the
format-neutral ordered vocabulary and declared-special compatibility checks;
it does not depend on GGUF, templates or model-specific policy. Each pipeline
binds its verified tokenizer to its own artifact metadata and applies its own
special-token policy. Embedding input uses no chat template or implicit query
instruction, and advertised dimensions do not imply Matryoshka qualification.
Synthetic family and pipeline tests do not establish deployed-artifact parity,
retrieval quality, reindex authority, serving or GPU qualification.

## Native reranking ownership

`decoders::Qwen3RankWeights` admits a distinct rank profile over the shared
private Qwen3 body. Rank pooling, exact `[yes, no]` labels and a two-row
`cls.output.weight` are required; embedding admission still refuses extra
heads. The decoder returns raw terminal-token classifier logits after final
RMS normalization. `rerank` owns their signed `yes - no` reduction and returns
one relevance logit per input index, not probabilities or sorted results.

The pipeline renders its verified artifact's embedded template using typed
system/query/document messages and an explicit setup instruction. It encodes
the whole render with automatic special tokens disabled. Separate byte,
token and batch bounds reject oversized inputs without truncation. The shared
`templates` owner retains strict undefined values, fuel, recursion and output
bounds, without external or named template resolution. Public batch structs
are revalidated at the prediction boundary; failure publishes no partial map.
These controls do not bound all tokenizer/template intermediates or total
process memory. Synthetic execution does not resolve exact-artifact conversion
provenance, template parity, retrieval quality, deployment or hardware gates.

## Native retrieval CPU requirements

`decoders::Qwen3CpuRequirements` binds a logical `f32` backing envelope to the
verified artifact and the executor's context bound. The checked private Qwen3
allocation shape is shared by execution and reporting. Its named phase sums
include temporary replacement allocations and buffers that remain live through
the FFN; sequential blocks do not multiply live workspace. Qwen3 execution is
stateless, so there is no retained-session or transaction-copy term.

Embedding reports separate the returned terminal hidden vector from workspace.
Rank reports include the terminal hidden vector and temporary classifier
projection, but the returned two-logit array occupies no heap backing. Concrete
embedding and reranking adapters compose their decoder report with the trusted
batch-item bound, accounting for previously returned vectors or scalar rows
while the next item executes. They do not multiply per-item workspace by the
batch count or change the stable consumer traits.

These are checked upper bounds on requested `Vec<f32>` backing, not allocator
capacity or physical reservations. Serialized GGUF bytes are identity-bound
and reported separately. Stack and structure storage, allocator overhead,
non-f32 buffers, tokenizer/template intermediates, process RSS and GPU memory
are excluded. A digest identifies bytes, not publisher authenticity. Neither
these reports nor the Qwen3.5 CPU report establish device requirements or
authorize admission and residency integration.

## cfg flags

- `logismos_no_gpu_kernels` - build path without compiled HIP kernels. Implemented GPU operations
  return `Error::NoGpuBuild`; genuinely unimplemented operations retain `Error::NotImplemented`.

## Glossary

See `_llm/glossary.md` for domain term definitions.
