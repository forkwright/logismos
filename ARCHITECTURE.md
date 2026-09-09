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
- `decoders` defaults to consuming `loader` without its tensor adapter and `quant`
  for explicit CPU row projection, without linking HIP. Its recurrent-attention
  and full-attention paths use the standalone CPU `kernels` graph; full-attention history consumes
  `cache` with its legacy tensor-backed flat feature disabled. Structural profiles remain
  distinct from payload-bound execution. The lower-level `quant` crate owns
  executable block and row geometry; inspection and projection reuse that owner.
  Opt-in `decoders/gpu` adds optional `hipcore`, `kernels/gpu` and `cache/gpu`
  for owned native one-block and main-model consumers; the GPU-capable facade selects it.
  Default CPU consumers and package-isolated CPU checks do not select it.
- `cache::paged` owns one logical KV ledger, CPU backing with borrowed row views
  and atomic append transactions, and optional separate native K/V/table backing.
  It does not own model semantics, shared-prefix identity,
  scheduling or host grants. Legacy flat tensor callers retain their default
  feature; their `KvCache` trait is not a shared-paging lifecycle contract.
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
  `cpu_f32`, `gdn`, `causal_conv`, and `attention` remain available without that feature;
  `transformers` selects that CPU-only graph, while `praxis` explicitly enables
  GPU launchers. Direct `kernels` users retain the default GPU feature. The
  crate does not depend on `core`.
- `kernels::row_gemv` depends on `quant` as the lower format owner. Its checked
  shape and CPU reference reuse serialized-row geometry and execution; its
  build dependency generates HIP layout and reconstruction constants from
  that same authority. This within-tier edge replaces duplicate format
  definitions and remains HIP-free when GPU features are disabled.
- `kernels::attention` owns checked single-query attention geometry, contiguous
  GQA mapping and operation workspace. Fallible borrowed-row access keeps it
  independent of `cache`; the decoder adapts its private paged history directly.
  The GPU-only native descriptor separately owns physical page geometry and
  launch admission. CPU page selection is not native device policy.
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

The standalone serialized-row GEMV primitive consumes `quant::RowFormat`, raw
row-major matrix bytes and f32 activations/output. Serialized F32 weights stay
f32; they do not pass through the separate fp16 WMMA operation. A checked opaque
shape owns exact extents and ABI bounds. The CPU reference delegates each row
to `quant`; private format-specific HIP entrypoints use one sequential thread
per row with source-specific floating-point controls and little-endian byte
decoding. Format identity remains the Rust enum, not a second numeric wire schema.
Its unsafe asynchronous launcher requires valid device buffers, lifetimes,
nonaliasing and admitted finite arithmetic. CPU typed nonfinite refusals do not
imply device-result validation. Its GPU domain requires zero-or-normal scales,
operands and intermediates; CPU subnormal witnesses do not qualify GPU denormal
modes. This is a correctness-oriented primitive, not a
whole-model GPU path, performance result or hardware qualification.

The grouped-GDN and causal-convolution native decode steps reuse their CPU
allocation plans as geometry owners. Their raw asynchronous launchers read
immutable prior state/history and write distinct staged results. They share a
private span owner with row GEMV for checked byte/f32 extents and writable
aliases. Empty convolution history has no memory footprint. Decoder transactions, persistent
residency and grant handling remain above these operations. Their precise
numerical domains and refusal rules live in the operation rustdoc; standalone
kernels do not establish native hybrid-model execution or device qualification.

Single-query native paged attention reads separate dense K/V arrays laid out as
`[physical_page][in_page_token][kv_head][head_width]`, with a caller-owned `u32`
logical-to-physical page table. Its descriptor admits explicit B8/B16/B32 pages,
checked allocation spans and launch dimensions. One wave32 block serves each
query head: lane-strided dot products use a fixed shuffle tree, followed by
token-ordered online softmax and staged output normalization. The native f32
order deliberately differs from the CPU operation's preserved materialized
score/softmax/value order. The raw unsafe asynchronous launcher requires valid
device table entries, finite normal-or-zero arithmetic, buffer lifetimes and a
nonaliasing staged output; it does not inspect device data or validate results.
CPU arithmetic witnesses and compiled code objects do not qualify GPU numerical
behavior. Both native decoder consumers use this kernel under their owned
completion boundary; admission and hardware qualification remain separate work.

`Qwen35NativeLayerPlan` borrows the exact verified weights it inspects and is
consumed when uploading a native full-attention-block session. Its demand comes
from seven serialized matrix allocations, four norm vectors, named workspace
buffers, per-step input/output and rotary controls, and the native KV/table
plans. These are requested device extents, not measured residency, allocator
overhead, a grant, or an allowance for results retained by callers.

The unsafe blocking session composes input RMSNorm, Q/gate/K/V projections,
per-head Q/K RMSNorm, partial text mRoPE, staged paged attention, sigmoid gating,
output projection/residual, post-attention norm and the full SwiGLU/residual
path. CPU and native control preparation share the allocation-free per-pair
mRoPE derivation. Raw f32 launchers retain caller-owned normal-or-zero
arithmetic preconditions. Both native session consumers instead use checked
variants throughout; a final output scan would miss invalid intermediates.

`kernels::numerical_status` owns one initialized, non-cloneable device status
per session. Rust owns its typed categories and byte extent; HIP constants
derive from that representation. Bitwise input classification precedes F16
scale conversion or F32 arithmetic, and explicit operation results accumulate
sticky subnormal/nonfinite flags without changing the numerical result or
skipping required shuffles. Only live reduction lanes perform reduction adds;
unused lane arithmetic cannot falsely refuse a valid result. Deliberate attention maximum initialization is
algorithm state, not a model operand. This checks Logismos operations and
math-call results, not hidden library temporaries. Source-local denorm controls
do not qualify emitted FP modes, math libraries or device behavior; the native
entry remains an unsafe qualified-environment boundary.

The native cache reuses the CPU's logical reservation/COW ledger but keeps
separate layer-major K/V allocations and a device table mirror. A completed
append parks its sole reservation inside the cache without publishing, ending
the borrow so the complete session can synchronize. Only after synchronization
and a successful status read may the cache publish and the session advance
position and return its owned device output.
The guard starts before the first kernel submission. Preflight refusal is
retryable; any submitted failure poisons the session without publishing its
logical state. Uncertain completion retains all resources, including immutable
weights and input. Drop retries synchronization and forgets the entire bundle
if completion remains uncertain. This is not device rollback, reset authority,
multi-token atomicity, or a safe serving API.

`Qwen35NativeExecutionPlan` binds the same verified artifact through native
embedding-row lookup, every main block in metadata order, final RMSNorm and the
distinct `output.weight` projection. It deliberately excludes an optional
terminal NextN extension from execution and device weight/state allocation;
presence of that extension is not a blanket refusal. Serialized row lookup and
GEMV share format-local reconstruction, retaining the established f32 dot order.

The native model owns one stream and one layer-indexed KV pool when full
attention is present. Each recurrent layer has distinct committed and staged
raw convolution history and GDN state. A single full-attention workspace, a
single recurrent workspace and a common residual/FFN workspace are reused in
stream order; two hidden rows alternate between main blocks. Recurrent Q/K
normalization tiles by source-head modulo into equal-head GDN; its activated
convolution V tail is borrowed rather than transposed or copied. Both block
kinds enter the shared finish only after their own output projection.

The existing resource guard covers the first embedding submission through
final logits. One append spans every full layer, interleaved with recurrent
work, and is prepared once. After synchronization, all fallible local checks
precede KV publication; recurrent state swaps and position advancement then
form an infallible tail. The public demand distinguishes actual immutable
uploads, shared workspaces, hidden/final/logit/control buffers, KV/table,
active/staged recurrent allocations and numerical status. It excludes
host/runtime overhead and arbitrarily retained returned logits. It is neither a resource grant nor
measured physical residency. Native numeric obligations remain explicitly
unsafe; compiler-checked, ignored device witnesses are not execution evidence.

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
ordered layer-state enum associates each main block with its recurrent state
or its full-attention layer in one execution-private paged KV pool. Its
caller-supplied context bound is capped by artifact and signed-position limits.
The paged transaction borrows the pool, retains immutable committed full pages,
and copy-on-writes only a partial tail. Attention reads borrowed logical rows
without rebuilding flat history. Recurrent state is staged separately; the
whole call publishes KV, recurrent state and position only after all selected
logits succeed and all-layer append completeness is validated. Its text-only
interleaved RoPE supports checked full or partial rotary
dimensions; unsupported effective scaling fails explicitly.

`Qwen35ExecutionPlan` validates caller context and step bounds and chooses
all-token or last-token logits. It retains the exact private CPU page plan
selected from 8/16/32-token candidates using allocation-owner costs. This is
not a qualified GPU alignment, shared-prefix or performance choice.
Its precomputed `Qwen35CpuRequirements` derives
logical `f32` backing from allocation owners, not a separate estimator.
Causal-convolution, grouped-GDN and single-query attention plans are consumed by their kernels;
recurrent, full-attention, FFN and LM-head owners compose named allocation
phases. The report separates retained state, separately allocated recurrent
transaction copies, transient workspace upper bound, and returned logits.
The complete preallocated KV pool, including page padding and its tail-copy
spare, belongs to retained backing; transaction accounting does not charge the
same pages twice. Serialized verified backing is
reported separately. Structure/stack storage, allocator overhead and capacity,
template/tokenizer allocations, process RSS, physical residency and GPU memory
are outside this report. It is neither an allocation guarantee nor device
admission input.

Private paging provides no cross-session sharing, prefix index, eviction,
device allocation or admission lease. Those require their actual consumers
and additional identity, lifecycle and qualification contracts.

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
