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

## Layout

Planning canonical lives in kanon:

- `~/dev/kanon/projects/logismos/vision.md` - what this is and what it is not.
- `~/dev/kanon/projects/logismos/ROADMAP.md` - phased plan.
- `~/dev/kanon/projects/logismos/STATE.md` - current state.
- `~/dev/kanon/projects/logismos/gnomon.md` - naming discipline inherited from the ecosystem.
- `~/dev/kanon/projects/logismos/research/` - research dossiers.
- `~/dev/kanon/projects/logismos/phases/NN-*/PLAN.md` - per-phase implementation specs.

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
- Forge repo: `forkwright/logismos`
- Kanon prefix: `lo`
- Config source: `workflow/kanon.toml [projects.logismos]`
- Planning state: `projects/logismos/STATE.md`
- Last state update: `not recorded`

Run `kanon docs sync --check --repo logismos` to verify this generated
section and `kanon docs sync --apply --repo logismos` to refresh it.

## Blast zone

- Paths explicitly named by the rendered prompt, role, or template input.

## Acceptance verifier

```bash
kanon gate
```
<!-- kanon:auto-end -->
