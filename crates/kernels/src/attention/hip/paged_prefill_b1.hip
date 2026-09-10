// B=1 causal multiquery paged-attention geometry wrapper.
#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "paged_attention_wave32.h"

extern "C" __global__ void logismos_paged_prefill_b1_f32_kernel(
    const float* __restrict__ query_f32,
    const float* __restrict__ keys_f32,
    const float* __restrict__ values_f32,
    const std::uint32_t* __restrict__ page_table_u32,
    float* __restrict__ output_f32,
    float scale,
    std::uint32_t tokens,
    std::uint32_t offset,
    std::uint32_t query_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_width,
    std::uint32_t page_tokens,
    std::uint32_t* numerical_status)
{
    const std::uint32_t work = blockIdx.x;
    const std::uint32_t token = work / query_heads;
    const std::uint32_t query_head = work % query_heads;
    const std::uint32_t lane = threadIdx.x;
    if (token >= tokens) {
        return;
    }
    const std::uint32_t visible_tokens = offset + token + 1U;
    const std::size_t row_stride =
        static_cast<std::size_t>(query_heads) * static_cast<std::size_t>(head_width);
    const std::size_t query_start =
        static_cast<std::size_t>(token) * row_stride
        + static_cast<std::size_t>(query_head) * static_cast<std::size_t>(head_width);

    logismos_paged_attention_wave32_f32(
        query_f32,
        keys_f32,
        values_f32,
        page_table_u32,
        output_f32,
        scale,
        visible_tokens,
        query_head,
        query_heads,
        kv_heads,
        head_width,
        page_tokens,
        query_start,
        lane,
        numerical_status);
}
