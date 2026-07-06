# Phase 6a GDN Preflight Audit — AITER vs fla-org

> **Scope.** Algorithmic audit of `chunk_gated_delta_rule` preflight surfaces
> before HIP WMMA wave32 porting begins. Target: W7900 (gfx1100, wave32).
> **Issue:** #11.

## 1. Algorithm comparison: `chunk_gated_delta_rule_fwd_kkt_solve_kernel`

| Aspect | fla-org | AITER | Significance |
|---|---|---|---|
| **File** | `fla/ops/gated_delta_rule/chunk_fwd.py` | `aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill/fused_cumsum_kkt.py` + `fused_solve_tril_recompute.py` | AITER splits KKT accumulation and solve+recompute into two kernels; FLA fuses them in one. |
| **KKT block compute** | Lines [line TBD — verify at port time]. Computes `beta * K @ K^T` for all 10 lower-triangular `[BC, BC]` blocks in registers with `BC = 16`. | `fused_cumsum_kkt.py` lines [line TBD — verify at port time]. Same effective KKT algebra; cumsum applied before KKT write. | Algebra equivalent; decomposition differs. |
| **Gate scaling** | `exp2(g[:,None] - g[None,:])` when `USE_G` is True; strict lower-triangular masking when `g is None`. | Natural-log units using `exp(g_diff)`; `tl.where(m_A, ..., 0)` masking. | Mathematically equivalent if cumsum scale matches exponential base. HIP port must pin one representation. |
| **Diagonal solve** | Forward-substitute each 16-row diagonal block in-register to produce `(I + A_diag)^{-1}`. | Same algorithm, four 16-row diagonal blocks in fused solve+recompute kernel. | Equivalent. |
| **Off-diagonal merge** | `Ai10`, `Ai21`, `Ai32`, `Ai20`, `Ai31`, `Ai30` via `SOLVE_TRIL_DOT_PRECISION`. | `b21`, `b32`, `b43`, `b31`, `b42`, `b41` via `DOT_PRECISION`. | Same dependency graph; block numbering differs (AITER uses 1-4, FLA uses 0-3). |
| **Recompute `w/u`** | Called as separate `recompute_w_u_fwd` after storing solved `A`. | Fused inside solve kernel to avoid a global-memory round trip. | AITER optimization is ROCm-relevant; avoids extra device-memory pass. |
| **Precision flag** | `SOLVE_TRIL_DOT_PRECISION = 'tf32'` when `IS_TF32_SUPPORTED`, else `'ieee'`. | `FLA_TRIL_PRECISION = 'ieee'` default; `allow_tf32=False` for `u` dots. | No tf32 on AMD; both converge to IEEE fp32. |

## 2. ROCm-specific fixes in AITER not in fla-org

| Fix | Location | gfx1100 relevance | Note |
|---|---|---|---|
| **IEEE fp32 default for tril dots** | `utils/solve_tril.py` lines [line TBD — verify at port time]; `gated_delta_rule_utils.py` lines [line TBD — verify at port time] | **Applies.** gfx1100 has no tf32; mandatory fallback. | AITER explicitly sets `TRITON_F32_DEFAULT=ieee` for AMD. FLA relies on `IS_TF32_SUPPORTED` auto-detect. |
| **Fused solve + recompute** | `prefill/fused_solve_tril_recompute.py` | **Applicable with care.** AITER targets wave64 CDNA3; gfx1100 is wave32 with tighter VGPR budget. Fusion may need to split on RDNA3 if register pressure exceeds 256 VGPRs/lane. | Reduces global memory traffic; port should attempt fusion but measure VGPR pressure. |
| **Natural-log gate representation** | `prefill/fused_cumsum_kkt.py`, `decode/fused_sigmoid_gating_recurrent.py` | **Applies.** Matches `exp` instruction latency on gfx1100. | FLA uses base-2 `exp2` with `RCP_LN2` scaling. Either works if CPU reference uses the same scale. |
| **Qwen-style fused decode kernel** | `decode/fused_sigmoid_gating_recurrent.py` | **Applicable.** Hard-codes `g = -exp(A_log) * softplus(a + dt_bias)` and `beta = sigmoid(b)`. | FLA exposes these as optional flags (`USE_GATE_IN_KERNEL`, `APPLY_BETA_SIGMOID`). AITER's specialization is the ROCm-optimized path. |
| **Boundary masking via `tl.where`** | `prefill/fused_cumsum_kkt.py` lines [line TBD — verify at port time] | **Applies.** Prevents `0 * inf -> NaN` from out-of-bounds gate values. | FLA uses stricter validity-mask ordering before `exp2`; HIP port should prefer FLA's ordering for safety. |

## 3. gfx1100 gap analysis

AITER is authored for CDNA3 (gfx942, MI300X) with **wave64** occupancy tuning. gfx1100 (RDNA3, W7900) is **wave32** with 256 VGPRs per lane.

| Area | CDNA3 (AITER target) | gfx1100 (logismos target) | Adjustment needed |
|---|---|---|---|
| Wave size | 64 | 32 | All tile math must use wave32 WMMA (`_w32` suffix). |
| VGPR budget | 256 logical per lane-equivalent (128 physical) | 256 physical per lane | Tighter limit; large register-resident state (`D_v * D_k = 16384` fp32) may spill. |
| LDS budget | Up to 128 KB | Up to 64 KB | Stage 5 (`chunk_gated_delta_rule_fwd_h`) LDS usage must be checked. |
| WMMA accumulator | `__builtin_amdgcn_wmma_f32_16x16x16_f16` (wave64 variant) | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` | Explicit `_w32` suffix mandatory. |
| Occupancy | High on MI300X | Lower on W7900 | Decode kernel may need 2 waves/v-head or LDS staging to avoid VGPR spill. |

AITER's autotune configs (`num_stages in {2,3}`, `BK in {32,64}`) are sized for wave64 and may not be optimal for gfx1100. A separate autotune sweep on W7900 is required for the HIP port.

## 4. `IS_TF32_SUPPORTED` verdict for gfx1100

**Verdict: False.**

RDNA3 (gfx1100) does not implement TensorFloat-32 (tf32). The `IS_TF32_SUPPORTED` flag is False on all AMD GPUs.

- **Fallback path:** `SOLVE_TRIL_DOT_PRECISION = 'ieee'` (strict fp32 accumulation).
- **Impact:** Slightly lower effective throughput on the block-merge dot chains, but numerically safer.
- **Action for port:** The HIP WMMA kernel must not attempt a tf32 fast-math path. Use `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` with fp32 accumulate, which is the IEEE-equivalent behavior.

## 5. Decision recommendation

**Use fla-org as the algorithmic correctness reference, merged with AITER's ROCm-specific execution fixes.**

Rationale:
- fla-org defines the canonical delta-rule recurrence, gate flags (`USE_G`, `USE_GK`, `USE_GV`, `USE_EXP2`), and varlen handling.
- AITER contributes proven ROCm optimizations (IEEE precision default, fused solve+recompute, natural-log gate representation) that improve performance without changing the math.
- AITER's wave64 register-pressure decisions are **not** portable to gfx1100 and must be re-derived for wave32.

**Porting checklist derived from this audit:**
1. Implement IEEE fp32 accumulation for all solve/recompute dot products.
2. Pin natural-log gate representation at the Rust API boundary (or consistently use base-2 with `RCP_LN2` in both CPU reference and HIP).
3. Preserve FLA's strict boundary-mask ordering before exponentiation.
4. Target wave32 WMMA with VGPR <= 256 and LDS <= 64 KB.
5. Keep the flag surface (`use_g`, `use_gk`, `use_gv`, `use_exp2`, `is_varlen`) from FLA; specialize the first HIP path for Qwen's required case.

---

*End of Phase 6a preflight audit. Next step: land `gdn` kernel module with CPU reference stubs, test harness, and launcher API (Phase 6a skeleton).*
