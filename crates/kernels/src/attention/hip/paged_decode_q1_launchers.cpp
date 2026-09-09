// Rust-callable launcher for Q=1 native paged attention.

#include <cstdint>

#include <hip/hip_runtime.h>

extern "C" __global__ void logismos_paged_decode_q1_f32_kernel(
    const float*, const float*, const float*, const std::uint32_t*, float*, float,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t);

namespace {
constexpr std::uint32_t WAVE_WIDTH = 32U;
}

extern "C" hipError_t logismos_launch_paged_decode_q1_f32(
    const void* query_f32,
    const void* keys_f32,
    const void* values_f32,
    const void* page_table_u32,
    void* output_f32,
    std::uint32_t visible_tokens,
    std::uint32_t query_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_width,
    std::uint32_t page_tokens,
    std::uint32_t physical_pages,
    float scale,
    hipStream_t stream)
{
    (void)physical_pages;
    const dim3 block(WAVE_WIDTH, 1U, 1U);
    const dim3 grid(query_heads, 1U, 1U);
    // SAFETY: Rust validates native descriptor extents and ABI values; the
    // unsafe caller owns table-value bounds, device lifetimes, and concurrency.
    logismos_paged_decode_q1_f32_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(query_f32),
        reinterpret_cast<const float*>(keys_f32),
        reinterpret_cast<const float*>(values_f32),
        reinterpret_cast<const std::uint32_t*>(page_table_u32),
        reinterpret_cast<float*>(output_f32),
        scale,
        visible_tokens,
        query_heads,
        kv_heads,
        head_width,
        page_tokens);
    return hipGetLastError();
}
