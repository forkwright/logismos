---
scope: logismos repo conventions (Rust-native agent-aware inference, gfx1100 target)
defers_to: kanon standards at kanon's crates/basanos/standards/ (see CLAUDE.md for how to resolve the kanon checkout root on this box)
tightens: original inference implementations, explicit hardware-access boundary, per-kernel CPU-reference-test policy
---

# AGENTS.md - Logismos

Cross-tool guide for AI coding agents (Claude Code, Kimi, Codex, Cursor, Copilot, etc.). Read [CLAUDE.md](CLAUDE.md) for operating instructions, including how to resolve the kanon checkout root on this box. Planning canonical lives in kanon: `projects/logismos/{vision,ROADMAP,STATE}.md`.

## Build / Test / Lint

```bash
cargo check -p <crate>                       # fast compile check
cargo test  -p <crate>                       # single crate tests
cargo clippy --workspace -- -D warnings      # lint (zero warnings under -D)
cargo test  --workspace                      # full suite
```

Use `CARGO_TARGET_DIR=/data/target` from `<workspace>/logismos`. Leave `CARGO_TARGET_DIR` unset inside `<workspace>/worktrees/logismos/<slug>/` so each worktree gets its own `<wt>/target`.

## Key patterns

- **No external ML frameworks.** HIP-first stack. Do not add dependencies on `candle`, `torch`, `burn`, `onnx`, or `llama.cpp`. Upstream runtime and emulator code is Read-only prior art; implement original code. The approved experimental HSA/ROCr provider preserves `amdgpu` and stays behind the hardware-access boundary.
- **Errors:** `snafu` with `.context()` and `Location` tracking. Agents must not call `unwrap()` in library code. Use `#[expect(lint, reason = "...")]` over `#[allow]`.
- **Time:** `jiff`. The fleet bans `chrono`.
- **Crate naming:** standalone single-word names (no `logismos-X` prefixes). Greek when the role earns it; English mechanical otherwise. See `CLAUDE.md` § "Crate naming rule".
- **Per-kernel CPU reference test.** Every GPU kernel has a CPU-side reference. Default tolerance is 1e-3 unless justified inline.
- **No silent CPU fallbacks.** GPU unavailable means precise error, never silent degradation.
- **Device-independent planning.** Capability checks target gfx1100, not a device name, ordinal, or fixed VRAM size. The W7900-only configuration remains supported; an absent optional XTX does not block it.
- **Agent-safe iteration.** Tests and compiler work use an OS-enforced GPU-denied runner. Hardware qualification is a separate operator-coordinated lane; do not probe devices, run GPU tests, change modes, or evict services as routine validation.
- **Visibility:** `pub(crate)` default. Use `pub` only on cross-crate API.
- **Test data:** synthetic identities (alice, bob, acme.corp). Never use real names, emails, or hosts.
- **Hosted at forkwright/logismos.** Changes land through the required public CI gate.

## Where to add things

| Task | Location |
|------|----------|
| New HIP kernel | `crates/kernels/src/<family>/` + `crates/hipcore` for FFI surface |
| Quantization scheme | `crates/quant/` |
| Sampler / decode policy | `crates/decode/` |
| New transformer family | `crates/transformers/src/<family>/` |
| Encoder model | `crates/encoders/` |
| Decoder model | `crates/decoders/` |
| Embedding model wiring | `crates/embed/` |
| Reranker contract + implementations | `crates/rerank/` |
| TTS pipeline component | `crates/tts/` |
| Speculative-decoding scheduler | `crates/sched/` |
| Embedding-model public trait (`EmbeddingModel`) | `crates/core/` |
| Provider adapter (HTTP/MCP) | `crates/hermeneus/` |
| STT pipeline | `crates/ekphrasis/` |
| Tokenizer | `crates/tokenize/` |

Crate architecture + dependency direction lives in kanon's `projects/logismos/vision.md` (role table) and `projects/logismos/STATE.md` (current state).

Logismos owns load, quantize, infer, and serve. It does not own general model formation, training,
or model release. [`contracts/runtime-scope.toml`](contracts/runtime-scope.toml) records that
boundary. Its guard checks the declared fields plus retired path/package/lock absence and license
coherence; it does not infer future code semantics. Do not recreate retired marker crates to imply
future authority.

## Standards

Universal engineering policy lives in kanon at `crates/basanos/standards/`. Read `STANDARDS.md` § Philosophy before writing code. Check `RUST.md` for language-specific rules. Logismos tightens kanon's universal rules with the original-implementation, per-kernel-CPU-test, and explicit hardware-boundary policies above.

## Boundaries

- **Always:** stay within the declared blast radius. Verify behavior with independent CPU references and the declared proof lane; compilation, emulation, and hardware qualification are different evidence.
- **Ask first:** changes to the public stable trait surface in `core` (downstream consumers depend), `hipcore` FFI shape, or kernel numerics tolerance defaults.
- **Never:** push to third-party upstream remotes. Never add ML-framework dependencies. Never introduce silent CPU fallback paths. Never bypass a capability's public contract to couple a consumer to implementation internals: embedding consumers use `core::EmbeddingModel`, while reranking consumers use `rerank::Reranker`.

## Verification

On a non-ROCm host, do not claim a local full-gate stamp. Run the non-build scope guard when
applicable, push without a trailer, and let the required public `gate / gate` workflow compile the
workspace with its documented HIP headers. GPU behavior still requires the real-hardware gate.

<!-- kanon:auto-start -->
<!--
scope: logismos repo cross-tool agent guide (Claude Code, Kimi, Codex, Cursor, Windsurf, Copilot)
generated_by: kanon docs sync
defers_to: CLAUDE.md for Claude Code-specific behavior; ~/menos-ops/CLAUDE.md for machine + service topology
tightens: repo-local MCP routing conventions; repo-local authoring conventions
-->

# logismos

Kanon-managed forkwright repository `logismos`.

## Commands

Run `kanon --help` for all kanon-managed workflow commands. Run project-local
build, test, and lint commands from this repository root.

- `kanon gate` - full local gate for kanon-managed PRs
- `kanon lint --fix` - deterministic standards fixes
- `kanon lint --explain <RULE>` - rule rationale and fix guidance
- `kanon pr open <head_ref> --title "..."` - open a forge PR
- `kanon pr merge <N> [--strategy squash|ff|rebase]` - merge after CI and gate checks
- `kanon docs sync --check --repo logismos` - verify derived bootstrap docs
- `kanon docs sync --apply --repo logismos` - regenerate derived bootstrap docs

For agent-native operations, prefer the `mcp__kanon__*` tool family. The canonical MCP routing catalog is not vendored in this repo; consult the kanon toolkit's `workflow/AGENTS-mcp-tools.md` for routing and fallback rules.

## Standards

Read `crates/basanos/standards/STANDARDS.md` § Philosophy before writing code. Key principles:
no workarounds, define once, reference everywhere, no shortcuts, no compromise on quality.
Rust work also reads `crates/basanos/standards/RUST.md` before editing Rust code.

## Rules

- Structured comment tags only: WHY, NOTE, WARNING, PERF, SAFETY, INVARIANT, TODO(#NNN), FIXME(#NNN)
- Conventional commits: `type(scope): description`
- Add `Gate-Passed: kanon 0.1.0` to validated commit bodies
- Never add `#[allow]` suppressions; use `#[expect(lint, reason = "...")]` only when justified
- Prefer MCP tools first; CLI commands are resilience fallbacks

## Architecture

- Registry name: `logismos`
- Repository identity: `forkwright/logismos`
- Hosting: `github`
- Push authority: GitHub-primary - push and PR through GitHub
- Kanon prefix: `lo`
- Config source: `workflow/kanon.toml [projects.logismos]`

## Boundaries

Always: run the applicable gate before pushing, stay inside the declared blast radius.
Ask first: workflow, service, credential, schema, or deployment changes.
Never: bypass CI, push to protected upstream refs, commit secrets, or suppress warnings.

## Blast zone

- Paths explicitly named by the rendered prompt, role, or template input.

## Acceptance verifier

```bash
kanon gate
```
<!-- kanon:auto-end -->
