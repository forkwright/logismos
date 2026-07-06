// WMMA GEMM kernel for gfx1100 (RDNA3, wave32).
//
// D = A @ B, row-major, fp16 in / fp32 accumulate / fp16 out.
//
// One wave32 warp per (16 × 16) output tile. The 32 lanes cooperate
// on 16×16 fragments via `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`.
//
// ## Fragment layouts (wave32, verified empirically against CPU ref)
//
// For `_f32_16x16x16_f16_w32`:
//
//   - `frag_a`: one *row* of the 16×16 A tile per lane.
//     `frag_a[i] = A_tile[lane % 16, i]`, `i ∈ [0,16)`. Row-major A
//     loads this with stride-1.
//
//   - `frag_b`: one *column* of the 16×16 B tile per lane.
//     `frag_b[i] = B_tile[i, lane % 16]`, `i ∈ [0,16)`. Row-major B
//     loads this with stride `ldb = N`.
//
//   - Accumulator: each lane L holds 8 fp32 values of column
//     `L % 16` of the D tile. Lane L elem `e` (0..8) corresponds to
//     `D_tile[2 * e + L / 16, L % 16]`. Lanes 0..15 hold even rows
//     (0, 2, …, 14); lanes 16..31 hold odd rows (1, 3, …, 15).
//
// cubecl-cpp's HIP manual-mma codegen inverts A vs B (lane holds a
// column of A and a row of B) because cubecl's canonical layout is A
// col-major + B row-major — equivalent to computing `A^T @ B^T`'s
// transpose. We keep A row-major + B row-major as the natural HF
// layout; that flips the per-lane fragment assignment.
//
// ## Phase 1 simplifications
//
// - A and B are both row-major (HF convention). We load a column of
//   A with a strided read (K-stride `lda = K`), and a row of B with a
//   dense read (`ldb = N`). That costs one extra strided read per
//   k-tile vs a transposed-A layout; tuning sweep moves this into
//   LDS later.
// - Shapes are assumed multiples of 16. Tail handling lives in the
//   Phase-1 tuning sweep.
// - One wave per output tile; no inter-tile reuse. Expected peak
//   around 20-30 TFLOPs (under the hand-rolled gate of 70 % / 86
//   TFLOPs). Phase-1 tuning is explicitly scoped to close the gap.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)

typedef _Float16 half16_t __attribute__((ext_vector_type(16)));
typedef float    float8_t __attribute__((ext_vector_type(8)));

extern "C" __global__ void logismos_matmul_wmma_fp16_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half*       __restrict__ d,
    int m,
    int n,
    int k)
{
    constexpr int TILE = 16;

    const int tile_row = blockIdx.y;    // tile-row index in D
    const int tile_col = blockIdx.x;    // tile-col index in D
    const int row_base = tile_row * TILE;
    const int col_base = tile_col * TILE;
    if (row_base >= m || col_base >= n) return;

    const int lane = threadIdx.x;       // 0..31
    const int laneMod16 = lane % 16;
    const int laneDiv16 = lane / 16;    // 0 or 1

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int k_base = 0; k_base < k; k_base += TILE) {
        // frag_a: row `laneMod16` of the 16x16 A-tile.
        // frag_b: column `laneMod16` of the 16x16 B-tile.
        half16_t frag_a;
        const int a_row = row_base + laneMod16;
        if (a_row < m) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int col = k_base + i;
                frag_a[i] = (col < k) ? (_Float16)a[a_row * k + col] : (_Float16)0.0f;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) frag_a[i] = (_Float16)0.0f;
        }

        half16_t frag_b;
        const int b_col = col_base + laneMod16;
        if (b_col < n) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int row = k_base + i;
                frag_b[i] = (row < k) ? (_Float16)b[row * n + b_col] : (_Float16)0.0f;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) frag_b[i] = (_Float16)0.0f;
        }

        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(frag_a, frag_b, acc);
    }

    // Store: lane L holds column (L%16); elem e holds row 2e + L/16.
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        int out_row = row_base + 2 * e + laneDiv16;
        int out_col = col_base + laneMod16;
        if (out_row < m && out_col < n) {
            d[out_row * n + out_col] = __float2half(acc[e]);
        }
    }
}

#else  // !gfx11

extern "C" __global__ void logismos_matmul_wmma_fp16_kernel(
    const __half* /*a*/, const __half* /*b*/, __half* /*d*/,
    int /*m*/, int /*n*/, int /*k*/) {}

#endif
