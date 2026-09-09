// Rust-callable launch shim for the staged causal-convolution step.

#include <cstdint>

#include <hip/hip_runtime.h>

extern "C" __global__ void logismos_causal_conv_step_f32_kernel(
    const float*, const float*, const float*, float*, float*, std::uint32_t, std::uint32_t);

extern "C" hipError_t logismos_launch_causal_conv_step_f32(
    const void* input_f32,
    const void* weights_f32,
    const void* history_in_f32,
    void* history_out_f32,
    void* output_f32,
    std::uint32_t channel_count,
    std::uint32_t width,
    hipStream_t stream)
{
    // One block and one thread per channel avoids unchecked launch geometry
    // arithmetic and leaves physical grid limits to HIP's reported launch status.
    const dim3 block(1U, 1U, 1U);
    const dim3 grid(channel_count, 1U, 1U);
    // SAFETY: the Rust validation establishes extents and ABI widths; its
    // unsafe caller upholds device ownership, lifetimes, and concurrency.
    logismos_causal_conv_step_f32_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(input_f32),
        reinterpret_cast<const float*>(weights_f32),
        reinterpret_cast<const float*>(history_in_f32),
        reinterpret_cast<float*>(history_out_f32),
        reinterpret_cast<float*>(output_f32),
        channel_count,
        width);
    return hipGetLastError();
}
