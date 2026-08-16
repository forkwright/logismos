# Logismos

*λογισμός - reckoning, calculation. The step-by-step numerical mode of reasoning.*

GPU inference stack for transformer embedding models, built ground-up on
HIP + hipBLASLt, targeting AMD gfx1100 (W7900).

**Status:** Phases 0-3 complete - Stella 1.5B v5 runs end-to-end on CPU with golden-fixture parity. Phase 4 (GPU cutover) is blocked on hardware availability: the AMD W7900 host is down for recovery, so GPU paths are unverified until it returns.

## Why

Kanon's knowledge substrate (mnemosyne) needs a GPU-accelerated embedder. Candle
has no ROCm backend. AMD deprecated ONNX Runtime's ROCm support. Rolling our own
keeps the hardware boundary owned in-repo. One transformer family, one GPU family,
written from the device upward.

## Scope

- In: tensor ops on HIP. Transformer inference. Stella 1.5B end-to-end. The
 `EmbeddingModel` contract mnemosyne consumes.
- Out: training. Non-AMD GPUs. Runtime graph optimization. Multi-GPU.

## Build configuration

`crates/kernels/build.rs` compiles HIP kernel sources via `hipcc`. When
`hipcc` is absent from `PATH` this falls back automatically to a
CPU-only, `logismos_no_gpu_kernels`-cfg'd build (this is the normal
path on a GH-hosted CI runner and any box without ROCm).

On a box that *has* ROCm installed but where a HIP kernel compile needs
to be skipped anyway - bisecting a `hipcc` regression, working around a
broken local ROCm install, or a fast CPU-only iteration loop - set
`LOGISMOS_SKIP_HIP_BUILD` to force the same CPU-only fallback the
`hipcc`-absent path takes:

```
LOGISMOS_SKIP_HIP_BUILD=1 cargo build
```

Cost: the resulting build has no GPU kernels - `kernels` compiles
against its CPU reference path only, and any HIP-backed op returns
`Error::NoGpuBuild` at runtime instead of running on-device.

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

PolyForm Shield 1.0.0 with AI-training prohibition. See [LICENSE](LICENSE).

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
- Last state update: `2026-08-03`

Run `kanon docs sync --check --repo logismos` to verify this generated
section and `kanon docs sync --apply --repo logismos` to refresh it.

## Blast zone

- Paths explicitly named by the rendered prompt, role, or template input.

## Acceptance verifier

```bash
kanon gate
```
<!-- kanon:auto-end -->
