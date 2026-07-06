// Rust-callable launcher shims for the matmul kernels. The
// `hipLaunchKernelGGL` macro is C-only; wrapping kernel launches in
// `extern "C"` functions lets Rust link against a stable symbol.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

extern "C" __global__ void logismos_matmul_naive_fp16_kernel(
    const __half*, const __half*, __half*, int, int, int);

extern "C" __global__ void logismos_matmul_wmma_fp16_kernel(
    const __half*, const __half*, __half*, int, int, int);

extern "C" hipError_t logismos_launch_matmul_naive_fp16(
    const void* a_fp16,
    const void* b_fp16,
    void*       d_fp16,
    int m,
    int n,
    int k,
    hipStream_t stream)
{
    // Block 16x16, grid ceil(n/16) x ceil(m/16).
    dim3 block(16, 16, 1);
    dim3 grid((n + 15) / 16, (m + 15) / 16, 1);
    logismos_matmul_naive_fp16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(a_fp16),
        reinterpret_cast<const __half*>(b_fp16),
        reinterpret_cast<__half*>(d_fp16),
        m, n, k);
    return hipGetLastError();
}

extern "C" hipError_t logismos_launch_matmul_wmma_fp16(
    const void* a_fp16,
    const void* b_fp16,
    void*       d_fp16,
    int m,
    int n,
    int k,
    hipStream_t stream)
{
    // One wave32 warp per output tile of 16x16.
    dim3 block(32, 1, 1);
    dim3 grid((n + 15) / 16, (m + 15) / 16, 1);
    logismos_matmul_wmma_fp16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(a_fp16),
        reinterpret_cast<const __half*>(b_fp16),
        reinterpret_cast<__half*>(d_fp16),
        m, n, k);
    return hipGetLastError();
}
