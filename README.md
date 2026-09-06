# Logismos

*λογισμός - reckoning, calculation. The step-by-step numerical mode of reasoning.*

An agent-aware operating environment for local AI compute: a Rust-native inference stack
targeting AMD gfx1100, with owned HIP/WMMA kernels and progressively owned execution policy.

**Status:** HIP primitives, Stella CPU golden-fixture parity, action-free placement,
process-local admission/residency coordination, and bounded instruction emulation exist.
GGUF inspection, digest-bound mixed-weight CPU projections and a stateful recurrent-attention
path are foundations, not native decoder serving or hardware qualification. The W7900
is available; the RX 7900 XTX is a
planned second device and requires its own qualification. The experimental below-HIP
provider remains unimplemented.

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

`logismos inspect --input PATH` retains its v1 digest/census receipt. Explicit `--metadata`
adds typed metadata (including empty-array element types and float bits) and source-order
tensor extents from the same observation. Inspection does not prove conversion provenance,
payload decoding, model support, or an atomic filesystem snapshot. The
[GPU-denied runner](docs/gpu-denied-runner.md) can admit one exact read-only input for this work.

[`decoders`](crates/decoders/src/lib.rs) derives a bounded Qwen3.5-family structural profile
from an opaque observation: exact typed metadata determines tensor roles and shapes, and
main decoder blocks remain distinct from an optional auxiliary NextN block. This does not
validate payloads or authorize execution. For payload access, `loader::gguf::VerifiedArtifact`
owns one immutable byte backing under an explicit size limit and requires a matching SHA-256
expectation. `decoders::Qwen35Weights` binds that backing to the structural profile and executes
named F32/Q8_0/Q4_K/Q5_K/Q6_K/IQ4_NL/IQ4_XS matrix projections using
[`quant`](crates/quant/src/lib.rs). The same checked block decoders support linear
row decoding without repeated basis-vector projections.
Its recurrent-attention executor owns layer-bound convolution/GDN state and commits state
only after a successful step. Other unsupported formats fail explicitly. This explicit CPU path does not
implement full decoder blocks, tokenizer/logits, NextN, or a complete model; it does not
authenticate a publisher or reserve device memory. The existing mmap tensor adapter is separate.

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
