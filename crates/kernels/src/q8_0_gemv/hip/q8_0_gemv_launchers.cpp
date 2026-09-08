// Rust-callable launch shim for the sequential Q8_0 GEMV correctness kernel.

#include <cstdint>

#include <hip/hip_runtime.h>

extern "C" __global__ void logismos_q8_0_gemv_f32_kernel(
    const std::uint8_t*, const float*, float*, int, int);

namespace {
constexpr unsigned int THREADS_PER_BLOCK = 256;
}

extern "C" hipError_t logismos_launch_q8_0_gemv_f32(
    const void* matrix_q8_0,
    const void* activations_f32,
    void* output_f32,
    int rows,
    int width,
    hipStream_t stream)
{
    const auto row_count = static_cast<unsigned long long>(rows);
    const auto grid_x = static_cast<unsigned int>(
        (row_count + THREADS_PER_BLOCK - 1U) / THREADS_PER_BLOCK);
    const dim3 block(THREADS_PER_BLOCK, 1, 1);
    const dim3 grid(grid_x, 1, 1);
    // SAFETY: the Rust shape checks extents and ABI dimensions; its unsafe
    // caller upholds device ownership, alignment, lifetimes and non-aliasing.
    logismos_q8_0_gemv_f32_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const std::uint8_t*>(matrix_q8_0),
        reinterpret_cast<const float*>(activations_f32),
        reinterpret_cast<float*>(output_f32),
        rows,
        width);
    return hipGetLastError();
}
