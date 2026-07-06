// Naive GEMM kernel — correctness anchor.
//
// D = A @ B, row-major, fp16 in / fp16 out, fp32 accumulate.
// One thread per output element. Tiny block. This is *not* fast;
// it exists to pin numerical behaviour before the WMMA path joins
// the pipeline.
//
// Shapes: A is (M, K), B is (K, N), D is (M, N). All row-major.

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

    float acc = 0.0f;
    for (int i = 0; i < k; ++i) {
        float a_ik = __half2float(a[row * k + i]);
        float b_ij = __half2float(b[i * n + col]);
        acc += a_ik * b_ij;
    }
    d[row * n + col] = __float2half(acc);
}
