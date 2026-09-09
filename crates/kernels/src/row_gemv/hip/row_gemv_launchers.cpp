// Private launch shims for serialized-row GEMV correctness kernels.

#include <cstdint>

#include <hip/hip_runtime.h>

extern "C" __global__ void logismos_f32_row_gemv_f32_kernel(
    const std::uint8_t*, const float*, float*, int, int);
extern "C" __global__ void logismos_q8_0_row_gemv_f32_kernel(
    const std::uint8_t*, const float*, float*, int, int);
extern "C" __global__ void logismos_q4_k_row_gemv_f32_kernel(
    const std::uint8_t*, const float*, float*, int, int);
extern "C" __global__ void logismos_q5_k_row_gemv_f32_kernel(
    const std::uint8_t*, const float*, float*, int, int);
extern "C" __global__ void logismos_q6_k_row_gemv_f32_kernel(
    const std::uint8_t*, const float*, float*, int, int);
extern "C" __global__ void logismos_iq4_nl_row_gemv_f32_kernel(
    const std::uint8_t*, const float*, float*, int, int);
extern "C" __global__ void logismos_iq4_xs_row_gemv_f32_kernel(
    const std::uint8_t*, const float*, float*, int, int);

namespace {
constexpr unsigned int THREADS_PER_BLOCK = 256;
}

#define LOGISMOS_ROW_GEMV_LAUNCHER(NAME) \
extern "C" hipError_t logismos_launch_##NAME##_row_gemv_f32( \
    const void* matrix, const void* activations, void* output, int rows, int width, hipStream_t stream) \
{ \
    const auto row_count = static_cast<unsigned long long>(rows); \
    const auto grid_x = static_cast<unsigned int>((row_count + THREADS_PER_BLOCK - 1U) / THREADS_PER_BLOCK); \
    const dim3 block(THREADS_PER_BLOCK, 1, 1); \
    const dim3 grid(grid_x, 1, 1); \
    /* SAFETY: Rust validates exact device spans and the caller retains ownership through completion. */ \
    logismos_##NAME##_row_gemv_f32_kernel<<<grid, block, 0, stream>>>( \
        reinterpret_cast<const std::uint8_t*>(matrix), \
        reinterpret_cast<const float*>(activations), \
        reinterpret_cast<float*>(output), rows, width); \
    return hipGetLastError(); \
}

LOGISMOS_ROW_GEMV_LAUNCHER(f32)
LOGISMOS_ROW_GEMV_LAUNCHER(q8_0)
LOGISMOS_ROW_GEMV_LAUNCHER(q4_k)
LOGISMOS_ROW_GEMV_LAUNCHER(q5_k)
LOGISMOS_ROW_GEMV_LAUNCHER(q6_k)
LOGISMOS_ROW_GEMV_LAUNCHER(iq4_nl)
LOGISMOS_ROW_GEMV_LAUNCHER(iq4_xs)

#undef LOGISMOS_ROW_GEMV_LAUNCHER
