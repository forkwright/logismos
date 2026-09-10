// Q=1 paged-attention correctness kernel.
//
// One 32-lane wave owns one query head. It does not own cache state, model
// policy, or table construction; Rust validates only descriptor extents while
// the unsafe caller owns table values and device-residency obligations.

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "paged_attention_wave32.h"

extern "C" __global__ void logismos_paged_decode_q1_f32_kernel(
    const float* __restrict__ query_f32,
    const float* __restrict__ keys_f32,
    const float* __restrict__ values_f32,
    const std::uint32_t* __restrict__ page_table_u32,
    float* __restrict__ output_f32,
    float scale,
    std::uint32_t visible_tokens,
    std::uint32_t query_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_width,
    std::uint32_t page_tokens,
    std::uint32_t* numerical_status)
{
    const std::uint32_t query_head = blockIdx.x;
    const std::uint32_t lane = threadIdx.x;
    if (query_head >= query_heads) {
        return;
    }

    const std::size_t query_start =
        static_cast<std::size_t>(query_head) * static_cast<std::size_t>(head_width);
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
