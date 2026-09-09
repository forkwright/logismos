// NOTE: private Rust-callable shim for the staged causal-convolution step.

#include <cstdint>

#include <hip/hip_runtime.h>

extern "C" __global__ void logismos_causal_conv_step_f32_kernel(
    const float*, const float*, const float*, float*, float*, std::uint32_t, std::uint32_t,
    std::uint32_t*);

namespace {
constexpr std::uint32_t THREADS_PER_BLOCK = 256U;
}

extern "C" hipError_t logismos_launch_causal_conv_step_f32_checked(
    const void* input_f32, const void* weights_f32, const void* history_in_f32,
    void* history_out_f32, void* output_f32, std::uint32_t channel_count,
    std::uint32_t width, void* numerical_status, hipStream_t stream)
{
    // WHY: `channel_count` is positive by the allocation-plan admission. Subtract
    // before division, so the ceil division cannot overflow at u32::MAX.
    const std::uint32_t grid_x = (channel_count - 1U) / THREADS_PER_BLOCK + 1U;
    const dim3 block(THREADS_PER_BLOCK, 1U, 1U);
    const dim3 grid(grid_x, 1U, 1U);
    // SAFETY: the Rust validation establishes extents and ABI widths; its
    // unsafe caller upholds device ownership, lifetimes, and concurrency.
    logismos_causal_conv_step_f32_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(input_f32),
        reinterpret_cast<const float*>(weights_f32),
        reinterpret_cast<const float*>(history_in_f32),
        reinterpret_cast<float*>(history_out_f32),
        reinterpret_cast<float*>(output_f32),
        channel_count,
        width,
        reinterpret_cast<std::uint32_t*>(numerical_status));
    return hipGetLastError();
}
