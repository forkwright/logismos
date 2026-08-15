<!--
scope: logismos repo conventions (Rust+HIP inference runtime, ~27 peer crates, W7900 primary target)
defers_to: kanon standards at ~/dev/kanon/crates/basanos/standards/
tightens: no-external-ML-framework rule, HIP FFI as lowest acceptable layer, per-kernel CPU-reference-test policy
-->

# Logismos - operating instructions

Read `~/dev/kanon/projects/logismos/{vision,ROADMAP,STATE}.md` before
substantive work. Planning canonical lives in kanon. This repo holds code plus repo-local agent
docs (`CLAUDE.md`, `AGENTS.md`, `README.md`).

## What logismos is

A Rust-native local inference platform. Workspace of ~25 peer
crates across 6 tiers (foundation, infrastructure, model families,
training, serving, API). HIP + hipBLASLt foundation, W7900 (gfx1100)
primary target. Every neural workload in the ecosystem - embedding,
rerank, classification, NER, extraction, decoder LLMs (incl. MoE),
speculative decoding, voice (STT + TTS), image-model backend ops,
training - runs through logismos.

Consumers pick the minimum subset of crates they need. `core` exposes
the stable trait surface. Implementations live in dedicated crates.
Infrastructure lives in lower-tier crates that share cleanly.

## Meta-principle

> All inference software as a system of crates. Each with clear, scoped
> uses and responsibilities.
>
> Projects pick single / all / a combo as needed.
> DRY. Systems and meta-systems thinking everywhere.
>
> Abstraction as part of a hierarchy that is no-compromise at any level.
> Nothing repeated at a lower level that can be abstracted higher without loss.
> Build the right tool the first time with no corner cutting.

Translated:

1. Each crate is one clear responsibility. Repeated responsibility is
 a missing lower-tier crate.
2. Abstraction moves up, never sideways.
3. No lossy abstraction.
4. Consumers pick the minimum.

## Crate naming rule

**No prefixes.** Each crate owns a distinctive single-word name.
Pattern matches the ecosystem (basanos, archeion, stoa, mneme, nous,
hermeneus - not kanon-X or aletheia-X). Where a role corresponds to
an existing ecosystem-Greek name, logismos inherits that name and its
original home eventually consolidates onto ours.

Current crates (see `~/dev/kanon/projects/logismos/vision.md` for the role table):

Greek (earned by role): `hermeneus`, `ekphrasis`, `ichneutes`,
`melete`, `taxis`, `praxis`.

English mechanical: `hipcore`, `kernels`, `loader`, `tokenize`,
`quant`, `cache`, `sample`, `grammar`, `sched`, `transformers`,
`encoders`, `decoders`, `embed`, `rerank`, `tts`, `diffusion`,
`autograd`, `optim`, `data`, `mcp`, `bin`, `core`.

Top-level facade: `logismos`.

New Greek names must pass `~/dev/kanon/projects/logismos/gnomon.md`'s
L1-L4 gate. No haste to invent decoration.

## What "done" looks like

- Every kernel has a CPU reference test. Tolerance 1e-3 unless
 otherwise justified inline.
- Every phase ships a working verifiable artefact, not a partial diff.
- Every public type has the minimum surface the next phase needs.
 YAGNI is default. Convenience accessors follow real callers.
- No silent CPU fallbacks. GPU unavailable means precise error.

## Boundaries

- **Do not** add a dependency on another ML framework (no candle,
 torch, burn, tract, ort, llama.cpp, vLLM). FFI to ROCm is the lowest
 acceptable layer.
- **Do not** add backends speculatively.
- **Do not** push to a third-party upstream remote. `origin`
 (`forkwright/logismos`, public) takes normal pushes, gated by
 `gate-attestation`.
- **Do not** start kernel work before the research dossier for that
 subsystem closes its decisions.

## Standards

- No `unwrap` / `expect` / `panic` in library code outside tests.
- Public types `#[non_exhaustive]` where future extension is plausible.
- Comments answer WHY, not WHAT.
- `#![deny(unsafe_op_in_unsafe_fn)]` at crate root.
- HIP FFI is the only place unsafe is acceptable at scale. Each unsafe
 block carries a safety comment.

Rust gate is forge-primary via `.kanon-ci.toml` (ROCm 6.4 on menos
forge). Run `kanon lint --rust` locally before push until menos returns.

When `kanon lint` runs from this repo, zero open violations.

## Working with AI assistants

Operating principle, memory system, and global constraints come from
`~/.claude/CLAUDE.md`. Those rules apply here. Logismos-specific tightening:

- Agents must scope work to the active `~/dev/kanon/projects/logismos/phases/NN-*/PLAN.md`. No scope creep into future phases.
- A passing `cargo check` is not evidence of correctness. Every
 kernel carries a CPU reference test at 1e-3 tolerance.

<!-- kanon:auto-start -->
## Generated kanon context

- Registry name: `logismos`
- Forge repo: `forkwright/logismos`
- Kanon prefix: `lo`
- Config source: `workflow/kanon.toml [projects.logismos]`
- Standards source: `crates/basanos/standards/STANDARDS.md`
- MCP routing catalog: `workflow/AGENTS-mcp-tools.md`

Run `kanon docs sync --check --repo logismos` to verify this generated
section and `kanon docs sync --apply --repo logismos` to refresh it.

## Blast zone

- Paths explicitly named by the rendered prompt, role, or template input.

## Acceptance verifier

```bash
kanon gate
```
<!-- kanon:auto-end -->
