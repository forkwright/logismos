# Logismos

*λογισμός - reckoning, calculation. The step-by-step numerical mode of reasoning.*

An agent-aware operating environment for local AI compute: a Rust-native inference stack
targeting AMD gfx1100, with owned HIP/WMMA kernels and progressively owned execution policy.

**Status:** HIP primitives, kernels, and Stella CPU end-to-end golden-fixture parity exist.
Native decoder serving, agent-aware placement, instruction emulation, and the experimental
below-HIP provider are under development, not qualified capabilities. The W7900 is available;
the RX 7900 XTX is a planned second device and requires its own qualification.

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

Upstream projects are reference corpora for original implementations, including the planned
bounded gfx1100 functional emulator. HIP remains the production substrate; an original
HSA/ROCr provider is an explicitly scoped experiment, not a production backend claim.

[`contracts/runtime-scope.toml`](contracts/runtime-scope.toml) records this product boundary.
Bounded adaptation remains absent unless a named consumer contract supplies an output owner,
retention and revocation policy, and rollback. The repository guard validates those declared
requirements and concrete workspace/license invariants; semantic scope remains a review decision.

## Build configuration

`crates/kernels/build.rs` compiles HIP sources with `hipcc` for the target in
`contracts/gpu-target.txt`. The default and `LOGISMOS_HIP_BUILD=required`
both fail if the compiler is missing or compilation fails. CPU-only iteration
requires `LOGISMOS_HIP_BUILD=cpu-only`; the retired `LOGISMOS_SKIP_HIP_BUILD`
variable is an error. A CPU-only build has no GPU kernels, and HIP-backed ops
return `Error::NoGpuBuild` instead of running on-device.

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

Before formatting, the public workflow runs two cheap repository guards. The runtime-scope guard
self-tests positive and negative cases, requires locked Cargo metadata, rejects retired
path/package/lock identities, and derives license coherence from Cargo metadata plus the checked
`LICENSE` bytes. The kanon-root SSOT guard rejects duplicate checkout-root instructions outside
`CLAUDE.md`. Neither guard claims to infer arbitrary program semantics.

What it does not prove: the GH-hosted runner has no AMD GPU. This path
proves the workspace compiles and links against real HIP headers/ABI —
it never executes a HIP kernel. A change touching `.hip` sources, or
`hipcore`/`kernels` FFI surface, still needs verification on real
hardware in a separately reserved operator qualification window before anyone
can trust it at runtime. Compiling real HIP code inside the GPU-denied runner
can catch compiler and code-object defects without allocating on a device;
neither that check nor functional emulation establishes hardware performance.

## Layout

Planning canonical lives in kanon, under `projects/logismos/`; see [CLAUDE.md](CLAUDE.md) for how
to resolve the kanon checkout root on this box:

- `projects/logismos/vision.md` - what this is and what it is not.
- `projects/logismos/ROADMAP.md` - phased plan.
- `projects/logismos/STATE.md` - current state.
- `projects/logismos/gnomon.md` - naming discipline inherited from the ecosystem.
- `projects/logismos/research/` - research dossiers.
- `projects/logismos/phases/NN-*/PLAN.md` - per-phase implementation specs.

Repo-local:

- [CLAUDE.md](CLAUDE.md) - working instructions for AI assistants.
- [AGENTS.md](AGENTS.md) - cross-tool bootstrap.
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
