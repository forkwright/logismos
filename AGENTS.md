---
scope: logismos repo conventions (Rust+HIP inference runtime, ~27 peer crates, W7900 primary target)
defers_to: kanon standards at kanon's crates/basanos/standards/ (see CLAUDE.md for how to resolve the kanon checkout root on this box)
tightens: no-external-ML-framework rule, HIP FFI as lowest acceptable layer, per-kernel CPU-reference-test policy
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

- **No external ML frameworks.** HIP-first stack. Do NOT add dependencies on `candle`, `torch`, `burn`, `onnx`, `llama.cpp`. HIP FFI via `hipcore` is the lowest acceptable layer.
- **Errors:** `snafu` with `.context()` and `Location` tracking. Agents must not call `unwrap()` in library code. Use `#[expect(lint, reason = "...")]` over `#[allow]`.
- **Time:** `jiff`. The fleet bans `chrono`.
- **Crate naming:** standalone single-word names (no `logismos-X` prefixes). Greek when the role earns it; English mechanical otherwise. See `CLAUDE.md` § "Crate naming rule".
- **Per-kernel CPU reference test.** Every GPU kernel has a CPU-side reference. Default tolerance is 1e-3 unless justified inline.
- **No silent CPU fallbacks.** GPU unavailable means precise error, never silent degradation.
- **Visibility:** `pub(crate)` default. Use `pub` only on cross-crate API.
- **Test data:** synthetic identities (alice, bob, acme.corp). Never use real names, emails, or hosts.
- **Hosted at forkwright/logismos.** Changes land via gated pushes; the gate runs before every push.

## Where to add things

| Task | Location |
|------|----------|
| New HIP kernel | `crates/kernels/src/<family>/` + `crates/hipcore` for FFI surface |
| Quantization scheme | `crates/quant/` |
| Sampler / decode policy | `crates/sample/` or `crates/decode/` |
| New transformer family | `crates/transformers/src/<family>/` |
| Encoder model | `crates/encoders/` |
| Decoder model | `crates/decoders/` |
| Embedding model wiring | `crates/embed/` |
| Reranker | `crates/rerank/` |
| TTS pipeline component | `crates/tts/` |
| Speculative-decoding scheduler | `crates/sched/` |
| Public stable trait | `crates/core/` (every consumer reads through this) |
| Provider adapter (HTTP/MCP) | `crates/hermeneus/` |
| STT pipeline | `crates/ekphrasis/` |
| Tokenizer | `crates/tokenize/` |

Crate architecture + dependency direction lives in kanon's `projects/logismos/vision.md` (role table) and `projects/logismos/STATE.md` (current state).

## Standards

Universal engineering policy lives in kanon at `crates/basanos/standards/`. Read `STANDARDS.md` § Philosophy before writing code. Check `RUST.md` for language-specific rules. Logismos tightens kanon's universal rules with the no-external-ML-framework, per-kernel-CPU-test, and HIP-FFI-floor policies above.

## Boundaries

- **Always:** stay within the declared blast radius. Verify behavioral changes with the relevant CPU reference test or hipBLASLt comparison.
- **Ask first:** changes to the public stable trait surface in `core` (downstream consumers depend), `hipcore` FFI shape, or kernel numerics tolerance defaults.
- **Never:** push to third-party upstream remotes. Never add ML-framework dependencies. Never introduce silent CPU fallback paths. Never bypass the `core` trait surface from a consumer crate.

## Gate trailer

Run `kanon gate --stamp` before pushing, then prefix every commit body with `Gate-Passed: kanon 0.1.0` once the stamp succeeds. Kanon holds trailer authority until logismos has its own dispatched CI.

<!-- kanon:auto-start -->
<!--
scope: logismos repo cross-tool agent guide (Claude Code, Kimi, Codex, Cursor, Windsurf, Copilot)
generated_by: kanon docs sync
defers_to: CLAUDE.md for Claude Code-specific behavior; ~/menos-ops/CLAUDE.md for machine + service topology
tightens: workflow/AGENTS-mcp-tools.md catalog routing; crates/basanos/standards/AGENT-DOCS.md authoring rules
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

For agent-native operations, prefer the `mcp__kanon__*` tool family. See
[workflow/AGENTS-mcp-tools.md](workflow/AGENTS-mcp-tools.md) for routing and fallback rules.

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
