#include <cstdint>
#include <hip/hip_runtime.h>
extern "C" __global__ void logismos_paged_prefill_b1_f32_kernel(const float*, const float*, const float*, const std::uint32_t*, float*, float, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t*);
extern "C" hipError_t logismos_launch_paged_prefill_b1_f32(
    const void* query_f32, const void* keys_f32, const void* values_f32, const void* page_table_u32,
    void* output_f32, std::uint32_t tokens, std::uint32_t work_items, std::uint32_t offset, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_width, std::uint32_t page_tokens,
    std::uint32_t physical_pages, float scale, void* numerical_status, hipStream_t stream) {
    (void)physical_pages;
    const dim3 block(32U, 1U, 1U);
    const dim3 grid(work_items, 1U, 1U);
    logismos_paged_prefill_b1_f32_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(query_f32), reinterpret_cast<const float*>(keys_f32),
        reinterpret_cast<const float*>(values_f32), reinterpret_cast<const std::uint32_t*>(page_table_u32),
        reinterpret_cast<float*>(output_f32), scale, tokens, offset, query_heads, kv_heads,
        head_width, page_tokens, reinterpret_cast<std::uint32_t*>(numerical_status));
    return hipGetLastError();
}
