# Logismos

*λογισμός - reckoning, calculation. The step-by-step numerical mode of reasoning.*

An agent-aware operating environment for local AI compute: a Rust-native inference stack
targeting AMD gfx1100, with owned HIP/WMMA kernels and progressively owned execution policy.

**Status:** HIP primitives, Stella CPU golden-fixture parity, action-free placement,
process-local admission/residency coordination, and bounded instruction emulation exist.
GGUF inspection, digest-bound mixed-weight CPU projections, bounded hybrid CPU
text generation and native Qwen3 CPU embeddings/reranking are foundations, not serving or
hardware qualification. The W7900
is available; the RX 7900 XTX is a
planned second device and requires its own qualification. The experimental below-HIP
provider remains unimplemented.

The standalone [serialized-row GEMV](crates/kernels/src/row_gemv/mod.rs) primitive
shares checked format geometry with `quant` and supplies an explicit CPU
reference plus private HIP launchers for its executable row formats. Serialized
F32 weights remain f32 rather than being converted to fp16.
The native numerical profile remains zero-or-normal. Checked launchers record
subnormal/nonfinite inputs and explicit arithmetic intermediates in one sticky
status; raw launchers retain caller-owned numerical preconditions.
GPU compilation is not numerical or performance
qualification; the safe text/retrieval pipelines remain CPU-only.

The [single-query paged-attention](crates/kernels/src/attention/mod.rs) operation
is consumed by the CPU hybrid decoder through borrowed KV rows. Its checked plan
also owns the decoder's attention-workspace accounting. A separate native
descriptor and wave32 HIP kernel support explicit 8/16/32-token physical pages;
they are not a whole-model GPU executor or a qualified device-cache policy.

The opt-in `decoders/gpu` surface exposes an explicitly unsafe, blocking native
Qwen3.5-family main-model executor: serialized embedding lookup, every recurrent
and full-attention main block, final normalization and a distinct output head.
`Qwen35NativeExecutionPlan` binds exact verified weights and derives requested
device allocation extents. Auxiliary NextN tensors may be present but are not
uploaded or executed by this main autoregressive baseline. The narrower
`Qwen35NativeLayerPlan` remains available for single-block qualification.

One immutable native model owner shares uploaded weights across sessions.
Each session independently owns scratch, recurrent committed/staged state,
paged KV, controls, stream, numerical status and pending logits. It publishes
state only after whole-token synchronization and a clear status read.
Arithmetic failures poison the session without returning logits or publishing
KV, recurrent state or position; uncertain completion retains the entire
session bundle and its shared model reference. Requested demand separates
resident uploads, per-session state, token controls and returned logits;
arbitrarily retained outputs and qualified runtime overhead remain separate.
The status checks explicit operations, not hidden math-library temporaries, and
still requires a qualified denorm-preserving compiler/math/device profile.
The `logismos` facade selects this surface; direct CPU consumers keep it off.
This is a qualification boundary, not a safe GPU text pipeline, serving,
a resource grant, or evidence of W7900/XTX numerical or performance parity.

## Why

Aletheia knows what work needs doing; Logismos owns how inference uses the resources granted
to it. The intended loop connects workload intent, admission, model and state residency,
kernels, and measured execution costs. Ordinary clients retain an inference API; agent-aware
clients can supply richer intent without becoming GPU administrators.

Arche/Tropos and systemd retain host modes and process-lifecycle enforcement. Logismos does not
take over gaming, display ownership, firmware, or the fleet's development-work scheduler.

## Scope

- In: loading, quantization, inference, and serving; typed model profiles, resource planning,
  admission, residency, and device-local execution. gfx1100 is the architecture target, not a
  synonym for one SKU or a 48-GB minimum. A single supported device is a normal configuration.
- First serving target: Qwen3.8 hybrid text execution plus Qwen3 embedding and reranking
  continuity, qualified against exact artifacts. Independent multi-device services are planned;
  two devices do not form one allocation pool.
- Out: general model formation, training authority, and model release. Automatic fleet cutover,
  direct PCI takeover, firmware changes, and unqualified hardware/performance claims.

Upstream projects are reference corpora for original implementations. The bounded gfx1100
functional emulator is original; full-kernel coverage remains planned. HIP remains the
production substrate; an original HSA/ROCr provider is an explicitly scoped experiment,
not a production backend claim.

The [`placement`](crates/placement/src/lib.rs) ledger owns per-device reservation accounting;
[`sched`](crates/sched/src/lib.rs) coordinates admission, use, revocation, and confirmed release.
These are deterministic process-local contracts, not physical reservations or a running executor.
A service owner must use one controller per resource grant and drain or reconcile on restart.
Supplied memory estimates are not yet artifact-derived requirements or measured residency.
In-process requested-byte leases compete with v1 workload reservations in the
same per-device ledger. Dropping a lease does not release its accounting.

`logismos inspect --input PATH` retains its v1 digest/census receipt. Explicit `--metadata`
adds typed metadata (including empty-array element types and float bits) and source-order
tensor extents from the same observation. Inspection does not prove conversion provenance,
payload decoding, model support, or an atomic filesystem snapshot. The
[GPU-denied runner](docs/gpu-denied-runner.md) can admit one exact read-only input for this work,
or an explicit model/tokenizer pair for native text preparation.

[`decoders`](crates/decoders/src/lib.rs) derives a bounded Qwen3.5-family structural profile
from an opaque observation: exact typed metadata determines tensor roles and shapes, and
main decoder blocks remain distinct from an optional auxiliary NextN block. This does not
validate payloads or authorize execution. For payload access, `loader::gguf::VerifiedArtifact`
owns one immutable byte backing under an explicit size limit and requires a matching SHA-256
expectation. Clones share that backing and its observation without rereading the
source. `decoders::Qwen35Weights` retains that owner, binds it to the structural profile and executes
named F32/Q8_0/Q4_K/Q5_K/Q6_K/IQ4_NL/IQ4_XS matrix projections using
[`quant`](crates/quant/src/lib.rs). The same checked block decoders support linear
row decoding.
Its recurrent-attention executor owns layer-bound convolution/GDN state and commits state
only after a successful step. `Qwen35Weights::execution(max_context)` constructs a bounded
token-ID session over the main hybrid blocks, with causal grouped attention, full or partial
interleaved RoPE, recurrent state, residual/FFN composition and token-major vocabulary logits.
Each call commits all layer state only after every input token and output projection succeeds.
Full-attention history uses execution-private CPU pages: complete committed pages stay
immutable, partial tails are copied before writing, and attention reads paged rows directly.
Failed calls publish neither KV history nor recurrent state. This is not cross-session
prefix sharing, eviction or a GPU hybrid-model path. The separate native
one-block session shares the logical paged ledger but owns its own device
backing; it does not migrate this CPU session onto a GPU.
Other unsupported formats or execution configurations fail explicitly.
`execution_plan` additionally bounds tokens per step and selects all-token or last-token
logits; prefill for generation need not retain a vocabulary row for every prompt token.
Its `cpu_requirements()` reports artifact-bound logical `f32` backing: retained state
(including padded KV pages and the preallocated tail-copy spare), separately allocated
recurrent transaction copies, a conservative workspace upper bound, and returned logits. The allocation
owners consume the same named sizes. Serialized artifact bytes are separate; neither value is
an allocation guarantee, whole-process memory estimate, GPU requirement or physical reservation.
This CPU path does not provide NextN, serving, real-artifact quality or GPU
qualification; it does not authenticate a publisher or reserve device memory. The existing
mmap tensor adapter is separate.

[`text`](crates/text/src/lib.rs) composes this native CPU path with the artifact's embedded
chat template, an explicitly digest-selected tokenizer companion, and checked greedy decoding.
Typed text-only requests have byte, context and output bounds. Each request owns fresh execution
state and returns only a complete decoded result; cancellation or error exposes no partial text
or resumable state. Model/tokenizer identity expectations belong to trusted setup, not individual
untrusted requests, and are content binding rather than publisher authentication.

`TextPipeline::prepare` exposes the exact rendered prompt, final token IDs and
artifact-bound decoder requirements without creating a decoder session or running
the model. Its opaque result owns the immutable pipeline profile and bound
decoder plan, so it outlives the caller's setup handles without copying model
or tokenizer payloads. It is consumed for execution; ordinary `generate` uses
the same path. Context and prefill bounds derive from that request, not configured
ceilings. These reports cover decoder f32 backing only, not prompt/tokenizer/text
allocations, whole-request memory or admission.

`logismos prepare-text` exposes that same preparation path through a HIP-free
CLI. Trusted invocation selects both artifacts by path, expected SHA-256 and
exact byte length; a bounded strict JSON argument supplies the typed request
and pipeline limits. Its JSON receipt includes rendering, final token IDs,
verified identities and decoder-only CPU requirements. It performs no model
operation, but loading retains the full immutable model backing and tokenization
has its own allocations: use a separately granted host budget for large inputs.
Receipt agreement with an independent reference is a later qualification step;
the command alone establishes neither numerical parity nor model quality.

The private template environment has no loader or registered templates. Named imports, includes
and inheritance cannot resolve another source; missing includes explicitly marked optional are
no-ops. Fuel, recursion and rendered-output limits are operational bounds, not a hostile-template
or total-process-memory sandbox. Tokenizer parse/encode/decode intermediate allocations are not
bounded by the returned-output byte limit. Direct `text` and `decode` consumers do not link HIP;
CPU execution is explicit, never a fallback for a GPU operation.

[`embed::qwen3`](crates/embed/src/qwen3.rs) exposes native Qwen3 CPU embeddings
through the unchanged `core::EmbeddingModel` trait. Verified artifact and
tokenizer contents drive causal execution, last-token pooling and full-width
L2-normalized output. Query prefixes are explicit setup policy; embedding
input uses no chat template. Unsupported dimensions and malformed or oversized
requests fail rather than truncate. Independent synthetic proofs do not
establish exact deployed-artifact parity, retrieval quality or reindex authority.
Direct `embed` consumers disable default features for the HIP-free native path;
the default `stella` feature preserves the existing Stella API and dependencies.

[`rerank`](crates/rerank/src/lib.rs) implements native Qwen3 CPU pair scoring
through the existing `Reranker` contract. Its checked rank profile shares the
causal decoder body without weakening embedding admission. The verified
artifact supplies its template; setup supplies the instruction and independent
byte/token/batch limits. Each input index receives one raw `yes - no` relevance
logit, not a probability. Oversized requests fail without truncation.
Direct native consumers disable default features; the default `modernbert`
feature preserves the existing encoder implementation. Exact converted-model
provenance, template/tokenizer parity and retrieval quality remain unqualified.

Native Qwen3 executors and concrete embedding/reranking adapters expose
`cpu_requirements()` for allocation-owner-derived logical `f32` backing. Adapter
reports include sequential batch-result accumulation, not parallel workspace
multiplication. These are not whole-process or GPU budgets; see
[retrieval requirements ownership](ARCHITECTURE.md#native-retrieval-cpu-requirements)
for output semantics, artifact binding and exclusions.

[`templates`](crates/templates/src/lib.rs) owns bounded template rendering for
text generation and reranking. It permits no host callbacks, loader or named
template registration; output, recursion and fuel limits are operational
controls, not a total-memory sandbox.

[`contracts/runtime-scope.toml`](contracts/runtime-scope.toml) records this product boundary.
Bounded adaptation remains absent unless a named consumer contract supplies an output owner,
retention and revocation policy, and rollback. The repository guard validates those declared
requirements and concrete workspace/license invariants; semantic scope remains a review decision.

## Build configuration

`kernels/gpu` enables HIP launchers and is on by default for direct consumers.
With that feature, `crates/kernels/build.rs` compiles HIP sources with `hipcc`
for the target in `contracts/gpu-target.txt`; CPU-only iteration explicitly
selects `LOGISMOS_HIP_BUILD=cpu-only`. Without the feature, the standalone CPU
graph needs no HIP compiler or runtime. The precise feature/build-mode matrix
and its fail-closed witnesses are in the [runner documentation](docs/gpu-denied-runner.md).

Build mode is not an isolation boundary. Agent-led checks run through the
[GPU-denied runner](docs/gpu-denied-runner.md), which denies device access even
to dependencies and build scripts. For example:

```bash
scripts/gpu-denied-runner.sh -- /bin/sh -ceu \
  'LOGISMOS_HIP_BUILD=cpu-only cargo test --locked -p placement -p bin'
```

`crates/hipcore/build.rs` is different: it resolves the HIP runtime
header and links `amdhip64` unconditionally, even in CPU-only kernel mode.
`cargo check --workspace` therefore fails at `hipcore` on
any box without ROCm headers installed — including a fresh developer
machine. That failure is expected, not a defect
(forkwright/logismos#14); do not look for a local workaround.

It does not block CI or merging. `gate-attestation`
(`.github/workflows/gate-attestation.yml`, via `forkwright/.github`'s
`hybrid-gate.yml`) installs Ubuntu's `libamdhip64-dev` on the
GH-hosted runner before building. That universe-component package
ships the two headers `hipcore`'s wrapper includes, plus
`libamdhip64.so`. A PR with no local `Gate-Passed` trailer therefore
still gets a real `cargo check`/`clippy`/`nextest` pass across the
whole workspace, `hipcore` included. On a non-ROCm host, push without
a trailer and let that CI path attest the change.

Before formatting, the public workflow runs repository guards. The runtime-scope guard
self-tests positive and negative cases, requires locked Cargo metadata, rejects retired
path/package/lock identities, and derives license coherence from Cargo metadata plus the checked
`LICENSE` bytes. The document-contract guard checks authored crate navigation and fixture-generator
output inventory. The kanon-root SSOT guard rejects duplicate checkout-root instructions outside
`CLAUDE.md`. These guards do not claim to infer arbitrary program semantics.

What it does not prove: the GH-hosted runner has no AMD GPU. This path
proves the workspace compiles and links against real HIP headers/ABI —
it never executes a HIP kernel. A change touching `.hip` sources, or
`hipcore`/`kernels` FFI surface, still needs verification on real
hardware in a separately reserved operator qualification window before anyone
can trust it at runtime. Compiling real HIP code inside the GPU-denied runner
can catch compiler and code-object defects without allocating on a device;
neither that check nor functional emulation establishes hardware performance.

## Layout

The operator-managed private planning corpus is canonical for Logismos's vision, roadmap,
current state, naming decisions, research dossiers, and phase plans. It deliberately has no public
repository path or link. Ask the operator or approved planning service for the material that
governs a change. [ARCHITECTURE.md](ARCHITECTURE.md) records the implemented crate topology and
ownership; Kanon standards govern engineering practice.

Repo-local:

- [CLAUDE.md](CLAUDE.md) - working instructions for AI assistants.
- [AGENTS.md](AGENTS.md) - cross-tool bootstrap.
- [ARCHITECTURE.md](ARCHITECTURE.md) - implemented crate topology and dependency model.
- `crates/` - the workspace.
- `phases/03-stella/golden/` - runtime test fixtures for Phase 3 parity test + Stella throughput bench.

## License

PolyForm Noncommercial 1.0.0. See [LICENSE](LICENSE).

<!-- kanon:auto-start -->
## Repository Metadata

- Registry name: `logismos`
- Description: Kanon-managed forkwright repository `logismos`.
- Repository identity: `forkwright/logismos`
- Hosting: `github`
- Push authority: GitHub-primary - push and PR through GitHub
- Kanon prefix: `lo`
- Config source: `workflow/kanon.toml [projects.logismos]`
- Planning state: `projects/logismos/STATE.md`
- Last state update: `2026-09-05`

Run `kanon docs sync --check --repo logismos` to verify this generated
section and `kanon docs sync --apply --repo logismos` to refresh it.

## Blast zone

- Paths explicitly named by the rendered prompt, role, or template input.

## Acceptance verifier

```bash
kanon gate
```
<!-- kanon:auto-end -->
