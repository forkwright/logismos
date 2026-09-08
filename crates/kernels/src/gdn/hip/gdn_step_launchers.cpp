// Rust-callable launcher for the grouped dense-f32 GDN staged step.

#include <cstdint>

#include <hip/hip_runtime.h>

extern "C" __global__ void logismos_gdn_grouped_step_f32_kernel(
    const float*, const float*, const float*, const float*, const float*,
    const float*, float*, float*, float, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t);

extern "C" hipError_t logismos_launch_gdn_grouped_step_f32(
    const void* q_f32,
    const void* k_f32,
    const void* v_f32,
    const void* beta_f32,
    const void* g_f32,
    const void* state_in_f32,
    void* state_out_f32,
    void* output_f32,
    float scale,
    std::uint32_t key_head_count,
    std::uint32_t value_head_count,
    std::uint32_t key_dim,
    std::uint32_t value_dim,
    hipStream_t stream)
{
    const dim3 block(value_dim, 1, 1);
    const dim3 grid(value_head_count, 1, 1);
    // SAFETY: the Rust launcher validates exact spans and ABI dimensions; its
    // unsafe caller upholds device ownership, lifetimes, and numerical domain.
    logismos_gdn_grouped_step_f32_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(q_f32),
        reinterpret_cast<const float*>(k_f32),
        reinterpret_cast<const float*>(v_f32),
        reinterpret_cast<const float*>(beta_f32),
        reinterpret_cast<const float*>(g_f32),
        reinterpret_cast<const float*>(state_in_f32),
        reinterpret_cast<float*>(state_out_f32),
        reinterpret_cast<float*>(output_f32),
        scale,
        key_head_count,
        value_head_count,
        key_dim,
        value_dim);
    return hipGetLastError();
}
