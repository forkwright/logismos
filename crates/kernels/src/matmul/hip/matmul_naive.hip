// Naive GEMM kernel — correctness anchor.
//
// D = A @ B, row-major, fp16 in / fp16 out, fp32 accumulate.
// One thread per output element. Tiny block. This is *not* fast;
// it exists to pin numerical behaviour before the WMMA path joins
// the pipeline.
//
// Shapes: A is (M, K), B is (K, N), D is (M, N). All row-major.

#include <cstdint>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

extern "C" __global__ void logismos_matmul_naive_fp16_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half*       __restrict__ d,
    int m,
    int n,
    int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;

    // WARNING: index arithmetic in int64_t, not int. `row*k` / `i*n` overflow
    // a 32-bit signed product for large dimensions (forkwright/logismos#32);
    // the launcher rejects any M/N/K whose element-index products would
    // exceed i32::MAX before this kernel ever launches
    // (`kernels::matmul::check_matmul_shape`), but the kernel's own
    // arithmetic must not depend solely on that upstream promise to stay
    // defined.
    const std::int64_t row64 = row;
    const std::int64_t col64 = col;
    float acc = 0.0f;
    for (int i = 0; i < k; ++i) {
        float a_ik = __half2float(a[row64 * k + i]);
        float b_ij = __half2float(b[static_cast<std::int64_t>(i) * n + col64]);
        acc += a_ik * b_ij;
    }
    d[row64 * n + col64] = __float2half(acc);
}
