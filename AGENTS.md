---
scope: logismos repo conventions (Rust+HIP inference runtime, ~27 peer crates, W7900 primary target)
defers_to: kanon standards at ~/dev/kanon/crates/basanos/standards/
tightens: no-external-ML-framework rule, HIP FFI as lowest acceptable layer, per-kernel CPU-reference-test policy
---

# AGENTS.md - Logismos

Cross-tool guide for AI coding agents (Claude Code, Kimi, Codex, Cursor, Copilot, etc.). Read [CLAUDE.md](CLAUDE.md) for operating instructions. Planning canonical lives in kanon: `~/dev/kanon/projects/logismos/{vision,ROADMAP,STATE}.md`.

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

Crate architecture + dependency direction lives in `~/dev/kanon/projects/logismos/vision.md` (role table) and `~/dev/kanon/projects/logismos/STATE.md` (current state).

## Standards

Universal engineering policy lives in kanon at `~/dev/kanon/crates/basanos/standards/`. Read `STANDARDS.md` § Philosophy before writing code. Check `RUST.md` for language-specific rules. Logismos tightens kanon's universal rules with the no-external-ML-framework, per-kernel-CPU-test, and HIP-FFI-floor policies above.

## Boundaries

- **Always:** stay within the declared blast radius. Verify behavioral changes with the relevant CPU reference test or hipBLASLt comparison.
- **Ask first:** changes to the public stable trait surface in `core` (downstream consumers depend), `hipcore` FFI shape, or kernel numerics tolerance defaults.
- **Never:** push to third-party upstream remotes. Never add ML-framework dependencies. Never introduce silent CPU fallback paths. Never bypass the `core` trait surface from a consumer crate.

## Gate trailer

Run `kanon gate --stamp` before pushing, then prefix every commit body with `Gate-Passed: kanon 0.1.0` once the stamp succeeds. Kanon holds trailer authority until logismos has its own dispatched CI.
