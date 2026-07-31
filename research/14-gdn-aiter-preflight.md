# Research 14  -  AITER-vs-FLA GDN preflight (issue #11, Option A)

> **Scope.** Option-A preflight only: audit the algorithmic reference surfaces
> before any HIP WMMA kernel work begins. Output is a readied implementation
> contract for Option B (CPU-reference + shape-contract PR) and Option C
> (full HIP port). No kernel code. No Triton runtime dependency.
>
> **Date:** 2026-05-26. **Target:** W7900 (gfx1100, wave32). **Issue:** #11.

**Pinned source revisions checked on 2026-05-26:**

| Source | Commit | Date | Note |
|---|---:|---|---|
| `fla-org/flash-linear-attention` | `19b5a3f411ecea6cdda62c6cc65cdae55ed2dec5` | 2026-05-25 | Adds current beta-sigmoid / negative-eigenvalue flags for GDN recurrent decode. |
| `ROCm/aiter` | `09cccff87d34db0b8a81eb871de8cf3f615a2627` | 2026-05-26 | Current AMD ROCm GDN Triton tree. |

---

## 1. FLA exact reference surfaces

Upstream `fla-org/flash-linear-attention` (MIT) is the algorithmic authority.
All line numbers below reference `main` as of 2026-05-26.

### 1.1 Chunk prefill  -  fused KKT + solve_tril

**File:** `fla/ops/gated_delta_rule/chunk_fwd.py`
**Kernel:** `chunk_gated_delta_rule_fwd_kkt_solve_kernel` (line 31-303)

What it does in one Triton kernel:
1. Loads `k` and `beta` (and optional gate `g`) in `BC = 16` sub-chunks.
2. Computes all 10 lower-triangular `[BC, BC]` blocks of `beta * K @ K^T`
   in registers (lines 77-139).
3. Applies gate scaling via `exp2(g[:,None] - g[None,:])` when `USE_G`
   (lines 147-164). Falls back to strict lower-triangular masking when
   `g is None`.
4. Scales diagonal blocks by `beta[:,None]` and off-diagonal blocks by
   `beta[:,None]` (lines 167-176).
5. Forward-substitutes on each diagonal block in-register to produce
   `(I + A_diag)^{-1}` (lines 179-212). Iteration runs `for i in range(2, min(BC, T - i_tcN))`.
6. Block-merges off-diagonal inverses via three nested `tl.dot` chains
family. The library uses Triton, not HIP WMMA, so callers cannot adopt it as a
   - `'tf32'` when `IS_TF32_SUPPORTED` is True (NVIDIA Ampere+)
   - `'ieee'` otherwise (gfx1100 path).
7. Stores the full `(I+A)^{-1}` to `A` output as a `[T, BT]` banded matrix.

**Surrounding plumbing:**
- `chunk_gated_delta_rule_fwd_intra` (line 306-384) calls the kernel above,
  then `recompute_w_u_fwd` from `wy_fast.py`.
- `FLA_CHUNK_SIZE = 64` is hard-coded in `fla/ops/utils.py` line 31.

### 1.2 Recurrent decode  -  token-by-token state update

**File:** `fla/ops/gated_delta_rule/fused_recurrent.py`
**Kernel:** `fused_recurrent_gated_delta_rule_fwd_kernel` (line 24-190)

What it does:
- One program per `(batchxv_head, V-tile)`; loops over `T` tokens.
- State `b_h` is `[BK, BV]` or `[BV, BK]` depending on `STATE_V_FIRST`.
- Per token loads `q`, `k`, `v`, `beta`, optional `g`/`gk`/`gv`.
- Optional in-kernel L2 norm on `q` and `k` (lines 79-82).
- The kernel scales `q` by `1/sqrtK` inside the loop (line 83).
- `beta` may be headwise (`[B,T,HV]`) or per-element (`[B,T,HV,V]`).
- Sigmoid on `beta` when `APPLY_BETA_SIGMOID`; `2*sigmoid` when
  `ALLOW_NEG_EIGVAL` (lines 86-91).
- Gate decay: `b_h *= exp(g)` with per-head `A_log` + `dt_bias` fusion when
  `USE_GATE_IN_KERNEL` (lines 93-100). This is the exact math from the
  canonical Phase 6a plan at
  `phases/06a-gdn-hybrid/PLAN.md` section 6.2 step 3.
- Per-v-head `gk`/`gv` apply elementwise exponent to state (lines 102-111).
- Delta update (lines 113-120):
  ```
  v = beta * (v - sum(h * k, axis=...))
  h += outer(v, k)   # or outer(k, v) depending on layout
  o = sum(h * q, axis=...)
  ```
- Stores `o` per token; optionally writes final state `ht`.

**Wrapper:** `fused_recurrent_gated_delta_rule_fwd` (line 192-261) sets
`BK = next_power_of_2(K)`, `BV = min(8, next_power_of_2(V))` when `gv is None`,
else `next_power_of_2(V)`. Grid is `(NV, N*HV)` with `num_warps=1`.

### 1.3 Other FLA files in the GDN call tree

| File | Role | Lines of interest |
|---|---|---|
| `fla/ops/gated_delta_rule/chunk.py` | Seven-kernel pipeline orchestrator: `chunk_gated_delta_rule_fwd` | 23-85 |
| `fla/ops/gated_delta_rule/wy_fast.py` | `recompute_w_u_fwd`  -  WY-factor recomputation after solve_tril | Whole file |
| `fla/ops/gated_delta_rule/gate.py` | Standalone gating kernels (non-fused path) | Whole file |
| `fla/ops/gated_delta_rule/naive.py` | PyTorch reference for unit testing | Whole file |
| `fla/layers/gated_deltanet.py` | `GatedDeltaNet` layer wrapper | 1-400+ |
| `fla/models/gated_deltanet/` | Model-level configs and forwarding | Whole dir |

### 1.4 Precision note: `IS_TF32_SUPPORTED` on gfx1100

`chunk_fwd.py` line 24-27:
```python
if IS_TF32_SUPPORTED:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('tf32')
else:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('ieee')
```

On gfx1100 `IS_TF32_SUPPORTED` is False; the `solve_tril` block-merge dots
fall back to IEEE fp32 accumulation. This is the baseline logismos must match
 -  no tf32 fast-math path exists on RDNA3.

---

## 2. AITER reference surfaces

AITER gives the closest AMD-authored ROCm porting prior for this algorithm
family. It is Triton-based, not HIP WMMA, so it cannot be adopted as a
runtime dependency per `AGENTS.md` section "No external ML frameworks." It is
used only as a correctness cross-check.

### 2.1 What AITER contains

Per the issue #11 triage comment, AITER retains:
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill.py`
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/decode.py`
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/utils.py`

AITER release notes (v0.1.13-rc2, 2026-04-10) explicitly list "gated delta
rule decode optimizations" as a new kernel added in the Silo bulk merge.

### 2.2 What AITER targets

- **Hardware:** CDNA3 (gfx942, MI300X/MI325X) and CDNA4 (gfx950, MI355X).
- **Wave size:** 64 (CDNA default). Triton auto-tuning configs are sized for
  wave64 occupancy.
- **Precision:** Same `ieee` fallback as FLA on AMD. No tf32 on CDNA3 either.

**Critical mismatch:** gfx1100 (RDNA3) is wave32 with 256 VGPRs per lane.
CDNA3 is wave64 with 256 *logical* VGPRs (128 physical per lane-equivalent).
AITER register-pressure decisions do not transfer directly to gfx1100.

---

## 3. AITER-vs-FLA algorithmic diff  -  resolved preflight

The side-by-side read found no alternate GDN recurrence in AITER. AITER keeps
the same delta update and output equations, but it changes the ROCm execution
decomposition and several interface defaults. The HIP WMMA port should treat
FLA as the algorithmic reference and AITER as the ROCm layout / precision prior.

### 3.1 Prefill decomposition and KKT / solve algebra

| Surface | FLA | AITER | Port decision |
|---|---|---|---|
| KKT + solve | `chunk_fwd.py` `chunk_gated_delta_rule_fwd_kkt_solve_kernel`, lines 38-315. One kernel computes KKT blocks, applies gate/beta scaling, solves diagonal blocks, merges off-diagonal inverse blocks, and stores solved `A`. | `prefill/fused_cumsum_kkt.py` lines 152-226 computes cumsum + raw KKT into `A_raw`; `prefill/fused_solve_tril_recompute.py` lines 33-333 solves `A_raw` and recomputes `w/u` in one later kernel. | Keep the FLA algebra and AITER launch split as the first HIP plan: write raw KKT once, then fuse solve + `w/u` recompute if VGPR pressure allows. |
| Diagonal solve | FLA lines 219-248 initialize `-A`, forward-substitute each 16-row diagonal block, then add identity. | AITER lines 91-115 do the same for four 16-row diagonal blocks in fused solve+recompute. | Algorithm is equivalent. |
| Off-diagonal merge | FLA lines 254-288 compute `Ai10`, `Ai21`, `Ai32`, `Ai20`, `Ai31`, `Ai30` with `SOLVE_TRIL_DOT_PRECISION`. | AITER lines 154-187 compute the same block dependencies as `b21`, `b32`, `b43`, `b31`, `b42`, `b41` with `DOT_PRECISION`. | Same dependency graph; line names differ because AITER numbers blocks 1-4 instead of 0-3. |
| Recompute `w/u` | FLA calls `recompute_w_u_fwd` after storing solved `A` (`chunk_fwd.py` lines 383-392). | AITER fuses recompute inside the solve kernel (`fused_solve_tril_recompute.py` lines 200-331). | AITER shows the useful ROCm optimization: avoid a solved-`A` global-memory round trip after the raw KKT write. |

### 3.2 Gate exponent representation

FLA converts gate cumulative sums to base-2 units in `chunk.py` lines 51-67
using `RCP_LN2`, then uses `exp2` in `chunk_fwd.py` lines 179-190 and
`wy_fast.py` lines 72-77 / 150 / 197. AITER keeps natural-log units and uses
`exp` in `prefill/fused_cumsum_kkt.py` lines 214-216,
`prefill/fused_solve_tril_recompute.py` lines 219-222, and
`prefill/chunk_o.py` lines 121-128.

Both are mathematically equivalent when the cumsum scale is paired with the
matching exponential. The HIP port must choose one representation at the module
boundary and keep it explicit. For gfx1100, natural-log `exp` matches the
Phase 6a math and AITER's ROCm path. Base-2 `exp2` is acceptable only if the
CPU reference applies the same `RCP_LN2` scaling before comparison.

### 3.3 Boundary masking and overflow behavior

FLA explicitly guards the gate-difference exponent against out-of-bounds
`0 - g_inbounds` overflow in `chunk_fwd.py` lines 172-190 by combining the
triangular mask with per-subchunk validity masks before `exp2`. AITER's scalar
reference path has `safe_exp` in `prefill/fused_cumsum_kkt.py` lines 10-12 and
uses it at lines 73-76, but the optimized path relies on `tl.where(m_A, ..., 0)`
around `exp` at lines 214-216.

The HIP port should preserve FLA's stricter validity-mask ordering. It is a
correctness guard, not only a performance detail, because IEEE `0 * inf` can
become NaN if an out-of-bounds gate value participates in the exponent before
masking.

### 3.4 Decode gate fusion and beta semantics

FLA recurrent decode now exposes `USE_GATE_IN_KERNEL`, `HAS_DT_BIAS`,
`APPLY_BETA_SIGMOID`, and `ALLOW_NEG_EIGVAL` in
`fused_recurrent.py` lines 19-65. The loop applies optional `sigmoid(beta)`
and optional `2*sigmoid(beta)` at lines 120-127, then fuses
`-exp(A_log) * softplus(g + dt_bias)` at lines 129-136.

AITER has two decode surfaces:

- Generic `decode/fused_recurrent.py` lines 46-191 matches the old generic
  recurrent kernel. It accepts precomputed `g` and `beta`. It does not expose
  `A_log`, `dt_bias`, beta sigmoid, negative eigenvalues, or V-first state.
- Qwen-style fused `decode/fused_sigmoid_gating_recurrent.py` lines 44-189
  hard-codes `g = -exp(A_log) * softplus(a + dt_bias)`, `beta = sigmoid(b)`,
  and V-first state update. It does not expose `ALLOW_NEG_EIGVAL`.

Port decision: implement the FLA flag surface in Rust, but specialize the first
HIP path for Qwen's required case: `use_gate_in_kernel = true`,
`use_beta_sigmoid_in_kernel = true`, `allow_neg_eigval = false`. That matches
AITER's optimized Qwen path without losing the public contract needed for FLA
parity tests.

### 3.5 Softplus approximation

FLA's AMD path selects `softplus_triton` in `fla/ops/utils/softplus.py`
lines 85-115: `where(x < 20, log(1 + exp(x)), x)`. AITER uses the same
thresholded formula with explicit `softplus_beta` and `softplus_threshold`
arguments in `decode/fused_sigmoid_gating_recurrent.py` lines 133-142 and
`prefill/fused_gdn_gating_prefill.py` lines 33-44; the AITER wrappers default
to beta `1.0` and threshold `20.0` at `prefill/fused_gdn_gating_prefill.py`
lines 50-57.

No ROCm-specific alternate softplus was found. The CPU reference should use
the thresholded fp32 formula, not an unguarded `log1p(exp(x))`, so large
positive `a + dt_bias` values match both source trees.

### 3.6 Chunk size, block size, and autotune assumptions

FLA fixes `BC = 16` in `chunk_fwd.py` line 360 and uses `BT = chunk_size`
with the default `64`. AITER's optimized prefill fixes the same effective
`BT = 64` default in `prefill/fused_cumsum_kkt.py` lines 229-237, with K
blocking autotuned over `BK in {32,64}` and AMD `num_stages in {2,3}` at
lines 140-150. AITER's fused solve+recompute wrapper fixes `BK = 64` and
`BV = 64` in `prefill/fused_solve_tril_recompute.py` lines 357-391.

No larger MI300-only chunk tile was found in the GDN prefill path. The W7900
contract can keep `BT = 64`, `BC = 16` as the correctness baseline and reserve
`BT = 32` only as a measured VGPR/LDS fallback.

### 3.7 Varlen handling

Both trees support `cu_seqlens` + prepared chunk indices for prefill. FLA
uses `prepare_chunk_indices` in `chunk_fwd.py` lines 362-364 and validates
flattened varlen batches in `chunk.py` lines 525-535. AITER does the same in
`prefill/fused_cumsum_kkt.py` lines 258-263 and
`prefill/fused_solve_tril_recompute.py` lines 363-368. AITER decode also
supports `cu_seqlens` in `decode/fused_recurrent.py` lines 109-115 and
`decode/fused_sigmoid_gating_recurrent.py` lines 78-87.

Port decision: keep `cu_seqlens` and `chunk_indices` in the Rust contract from
the first scaffold, even if the first HIP kernel only accepts equal-length
synthetic tests.

### 3.8 State layout

FLA's generic recurrent kernel supports both K-first `[K,V]` and V-first
`[V,K]` state layouts through `STATE_V_FIRST` in `fused_recurrent.py`
lines 96-110, 152-159, and 174-179. FLA's public wrapper documents V-first as
an optional `state_v_first` mode in `chunk.py` lines 440-445.

AITER's generic decode kernel uses K-first state (`decode/fused_recurrent.py`
lines 139-142 and 189-191). Its optimized Qwen decode kernel uses V-first
state (`decode/fused_sigmoid_gating_recurrent.py` lines 101-117 and
141-144). AITER prefill also carries an explicit V-first optimized entry point
in `prefill/chunk.py` lines 198-268.

Port decision: keep the Phase 6a `ssm_state` V-first shape
`[H_v, D_v, D_k]`. AITER's optimized Qwen path agrees with that choice, and
it makes the decode output tile contiguous in `V`.

### 3.9 Precision defaults and ROCm-specific behavior

FLA selects `SOLVE_TRIL_DOT_PRECISION = 'tf32'` only when
`IS_TF32_SUPPORTED`, else `'ieee'`, in `chunk_fwd.py` lines 18-21. AITER's
ROCm utility defaults `FLA_TRIL_PRECISION` to `'ieee'` in
`utils/solve_tril.py` lines 25-33, and `gated_delta_rule_utils.py` lines
488-490 set `TRITON_F32_DEFAULT=ieee` for AMD. AITER's fused recompute uses
`allow_tf32=False` for the `u` dot products in
`prefill/fused_solve_tril_recompute.py` lines 248-259.

Port decision: gfx1100 must use IEEE fp32 accumulation for solve/recompute
dots. There is no tf32 fast path to preserve on RDNA3.

---

## 4. Next implementation contract (Option B readiness)

This section is the hand-off spec from preflight (Option A) to the
CPU-reference PR (Option B) and onward to HIP (Option C).

### 4.1 Crate/module location

`crates/kernels/src/gated_delta_rule/` following the existing kernel shape:
```text
gated_delta_rule/
    mod.rs       -  Rust launcher + FFI declarations
    cpu.rs       -  fp32 CPU reference for every kernel stage
    hip/*.hip    -  device source (Option C)
    hip/*.cpp    -  extern "C" launcher shim (Option C)
```

### 4.2 Kernel stages and inputs/outputs

Port the seven-kernel pipeline faithfully (Option A from PLAN section 8.3).
Monolithic fusion is deferred to Phase 10.

| Stage | Rust launcher name | Inputs | Outputs | Tolerance (fp16) |
|---|---|---|---|---|
| 1 | `chunk_local_cumsum` | `g: [B,T,HV] fp32` | `g_cumsum: [B,T,HV] fp32` | 1e-5 |
| 2 | `chunk_scaled_dot_kkt_fwd` | `k: [B,T,H,K] fp16`, `beta: [B,T,HV] fp32`, `g: [B,T,HV] fp32` | `A: [B,T,HV,BT] fp16/32` | 1e-3 |
| 3 | `solve_tril` | `A: [B,T,HV,BT]` | `A_inv: [B,T,HV,BT]` | 1e-3 |
| 4 | `recompute_w_u_fwd` | `k, v, beta, A_inv, g_cumsum` | `w: [B,T,HV,K]`, `u: [B,T,HV,V]` | 1e-3 |
| 5 | `chunk_gated_delta_rule_fwd_h` | `k, w, u, g, S_start: [B,HV,V,K] fp16` | `h: [B,T,HV,V]`, `v_new: [B,T,HV,V]`, `S_end` | 1e-3 |
| 6 | `chunk_fwd_o` | `q, k, v_new, h, g, scale: f32` | `o: [B,T,HV,V] fp16` | 1e-3 |
| 7 | `l2norm_fwd` | `q, k: [B,T,H,K] fp16` | `q_norm, k_norm: same` | 1e-4 |
| 8 | `fused_recurrent_gated_delta_rule_packed_decode` | `mixed_qkv, a, b, A_log, dt_bias, scale, ssm_state` | `o: [num_tokens,HV,V]`, updated `ssm_state` | 1e-3 |

These are logical parity stages. The HIP path may fuse stages 2-4 following
AITER's ROCm split once each logical stage has a CPU reference and a parity
test.

**Decode-only Sprint 1** needs stages 8 + 7 only, with a reference-CPU
prefill seeding `ssm_state`.

**Key contract rules:**
- All `g`, `beta`, `A_log`, `dt_bias` are **fp32 mandatory** (PLAN section 6.4).
- `ssm_state` is fp16 stored, fp32 accumulated inside kernel.
- Scale is `1/sqrtD_k` (default `D_k = 128` -> `scale = 0.0883883`).
- `BT = 64`, `BC = 16` unless VGPR analysis forces `BT = 32`.

### 4.3 CPU reference expectations

Every stage must have a pure-Rust (or pure-NumPy golden-generating) fp32
reference. The CPU reference is the parity anchor. HIP is the test subject.

**Algorithm source:** Implement against the canonical Phase 6a plan at
`phases/06a-gdn-hybrid/PLAN.md` sections 6.2 and 6.3,
not against Triton source. Triton is the cross-check. The plan math is the
spec.

**Test shapes (synthetic):**
- `B = 1`, `T in {1, 64, 2048, 8192}`
- `H = 16`, `HV = 32`, `K = V = 128`
- Varlen: `B = 1`, `cu_seqlens = [0, 512, 1536, 2048]`

**Tolerance gates:**
- fp16: `|Delta| <= 1e-3` relative (default per `AGENTS.md` section  per-kernel CPU test).
- Q4_K_S: `|Delta| <= 5e-3` (PLAN section 12.2).

### 4.4 W7900 / gfx1100 parity gates

| Gate | Criterion |
|---|---|
| Wave size | wave32 only; `-mno-wavefrontsize64` in `build.rs` |
| VGPR budget | <= 256 per lane; static assertion in HIP source |
| LDS budget | <= 64 KB per workgroup; static assertion |
| WMMA precision | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` (fp32 accumulate) |
| No tf32 | Explicit `ieee` equivalent; no fast-math dot path |
| Occupancy floor | >= 1 wave/CU for heavy kernels; measure with `rocprof` |

**Register-pressure decision tree (decode):**
1. Try register-resident state `D_v x D_k = 16384` fp32 values.
   At wave32 = 512 VGPRs/lane -> **FAIL** (> 256).
2. Split `D_v` across 2 waves/v-head -> 256 VGPRs/lane -> **marginal**.
3. If (2) spills, stage via LDS in `16 x 128 = 2 KB` slabs.
4. If LDS staging drops decode throughput below 20 tok/s, drop chunk size
   `BT = 64 -> 32` (prefill impact <= 15 %).

### 4.5 No-Triton / no-external-runtime boundary

- **No Triton compiler in the build graph.** Triton is a research reference
  only. `cargo build` must not invoke `triton.compile` or depend on PyTorch.
- **No runtime Python.** The `kernels` crate builds HIP `.hip` + `.cpp`
  through `build.rs` calling `hipcc`. No JIT compilation at load time.
- **No AITER wheel dependency.** AITER is read-only reference. Its code is
  not imported, vendored, or code-generated into logismos.
- **Lowest acceptable layer:** HIP FFI via `hipcore` (`AGENTS.md` section 
  "No external ML frameworks").

---

## 5. Residual risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| AITER prefill fuses different logical stages than FLA | Medium | Medium (parity triage can become opaque) | Keep logical CPU references for stages 2-4 even if HIP fuses them later |
| Natural-log vs base-2 gate representation drifts between CPU and HIP | Medium | High (wrong decay) | Pin one representation in the Rust contract; if using `exp2`, scale cumsum by `RCP_LN2` in CPU and HIP |
| `BT = 64` exceeds LDS on gfx1100 for stage 5 | Low | High (prefill kernel fails to launch) | Keep `BT = 32` fallback ready; measure LDS in prototype |
| Decode VGPR spill even at 2 waves/v-head | Medium | Medium (20 tok/s target at risk) | LDS staging is acceptable; plan fallback to naive per-token CPU prefill seeding |
| FLA upstream changes algebra before logismos locks parity | Low | Medium (chase drift) | Pin FLA commit hash in `research/14-gdn-aiter-preflight.md`; upgrade is explicit Phase 10 task |
| `IS_TF32_SUPPORTED` behavior changes in ROCm 6.5+ | Low | Low | gfx1100 has no tf32; path is already `ieee` |

---

## 6. Acceptance criteria for this preflight (Option A done-when)

- [x] Exact FLA reference surfaces named with file paths and line numbers.
- [x] AITER reference surfaces named with known file paths and hardware target.
- [x] Side-by-side diff of AITER decode/prefill vs FLA resolved with pinned
  source revisions.
- [x] ROCm-specific behaviors documented: launch split, gate exponent units,
  boundary masking, V-first Qwen state layout, and IEEE fp32 precision.
- [x] Implementation contract defines inputs/outputs per kernel stage.
- [x] CPU-reference expectations specified (shapes, tolerances, algorithm source).
- [x] W7900/gfx1100 parity gates enumerated (wave32, VGPR, LDS, WMMA, no tf32).
- [x] No-Triton/no-external-runtime boundary explicit.

---

*End of preflight. Next step is Option B: land `gated_delta_rule` kernel
module with CPU reference functions, synthetic FLA-parity fixtures, and
launcher stubs returning `NoGpuBuild` until HIP exists.*
