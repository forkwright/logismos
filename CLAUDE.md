<!--
scope: logismos repo conventions (Rust+HIP inference runtime, W7900 primary target)
defers_to: kanon standards at kanon's crates/basanos/standards/ (checkout-root resolution below)
tightens: no-external-ML-framework rule, HIP FFI as lowest acceptable layer, per-kernel CPU-reference-test policy
-->

# Logismos - operating instructions

Read kanon's `projects/logismos/{vision,ROADMAP,STATE}.md` before substantive work. Planning
canonical lives in kanon, a separate fleet-internal checkout — resolve its root per-box with the
MCP tool `mcp__kanon__config_location_get` (intent `kanon-repo`), which returns an already-expanded
path. The CLI form `kanon locate kanon-repo` prints its unexpanded `$KANON_ROOT` template rather
than resolving it (tracked as forkwright/kanon#3484) — do not treat that output as a filesystem path.
Without MCP access, ask the operator for the current checkout root; never hardcode one, it differs
per machine. This repo holds code plus repo-local agent docs (`CLAUDE.md`, `AGENTS.md`,
`README.md`).

## What logismos is

A Rust-native local inference platform on a HIP + hipBLASLt foundation, with W7900 (gfx1100) as
the primary target. Logismos owns loading, quantization, inference, and serving. Model formation,
general training, behavioral evaluation, and model release belong to the named producer; Logismos
owns runtime parity, correctness, compatibility, resource, and serving evidence.

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

The current crate inventory is derived from Cargo workspace metadata. Do not maintain a second
hand-written list here. See kanon's `projects/logismos/vision.md` for the role model, and run
`python3 scripts/check_runtime_scope.py` to verify its concrete structural guardrails. The check
does not replace semantic review of new behavior.

New Greek names must pass kanon's `projects/logismos/gnomon.md`'s
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
- **Do not** create general training or model-release authority. Bounded adaptation may enter only
  through a named consumer contract that defines its persistent output owner, retention and
  revocation, and rollback; execution authority does not imply release authority.
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

`kanon gate --stamp` cannot complete on a non-ROCm host: `hipcore`'s
build script fails hard without real HIP headers (forkwright/logismos#14),
so no `Gate-Passed` trailer is obtainable off menos. Do not chase one.
Push untrailered; the required `gate / gate` check falls through to
CI's `full-gate-build`, which installs real ROCm headers on the
GH-hosted runner and genuinely compiles the workspace — see README's
Build configuration section for exactly what that check does and does
not verify.

## Working with AI assistants

Operating principle, memory system, and global constraints come from
`~/.claude/CLAUDE.md`. Those rules apply here. Logismos-specific tightening:

- Agents must scope work to the active kanon `projects/logismos/phases/NN-*/PLAN.md`. No scope creep into future phases.
- A passing `cargo check` is not evidence of correctness. Every
 kernel carries a CPU reference test at 1e-3 tolerance.

<!-- kanon:auto-start -->
## Generated kanon context

- Registry name: `logismos`
- Repository identity: `forkwright/logismos`
- Hosting: `github`
- Push authority: GitHub-primary - push and PR through GitHub
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
