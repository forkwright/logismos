// B=1 causal multiquery paged-attention correctness kernel.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>
#include "numerical_status.h"

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
extern "C" __global__ void logismos_paged_prefill_b1_f32_kernel(
    const float* __restrict__ query_f32, const float* __restrict__ keys_f32,
    const float* __restrict__ values_f32, const std::uint32_t* __restrict__ page_table_u32,
    float* __restrict__ output_f32, float scale, std::uint32_t tokens, std::uint32_t offset,
    std::uint32_t query_heads, std::uint32_t kv_heads, std::uint32_t head_width,
    std::uint32_t page_tokens, std::uint32_t* numerical_status) {
    constexpr std::uint32_t WAVE_WIDTH = 32U;
    const std::uint32_t work = blockIdx.x;
    const std::uint32_t token = work / query_heads;
    const std::uint32_t query_head = work % query_heads;
    const std::uint32_t lane = threadIdx.x;
    if (token >= tokens) return;
    const std::uint32_t visible_tokens = offset + token + 1U;
    const std::uint32_t heads_per_kv = query_heads / kv_heads;
    const std::uint32_t kv_head = query_head / heads_per_kv;
    const std::size_t width = static_cast<std::size_t>(head_width);
    const std::size_t lane_index = static_cast<std::size_t>(lane);
    const std::size_t row_stride = static_cast<std::size_t>(query_heads) * width;
    const std::size_t query_start = static_cast<std::size_t>(token) * row_stride + static_cast<std::size_t>(query_head) * width;
    const float checked_scale = logismos_status_input(scale, numerical_status);
    float running_maximum = -INFINITY;
    float running_normalizer = 0.0f;
    for (std::size_t column = lane_index; column < width; column += WAVE_WIDTH) output_f32[query_start + column] = 0.0f;
    for (std::uint32_t cache_token = 0; cache_token < visible_tokens; ++cache_token) {
        const std::uint32_t logical_page = cache_token / page_tokens;
        const std::uint32_t in_page_token = cache_token % page_tokens;
        const std::uint32_t physical_page = page_table_u32[logical_page];
        const std::size_t kv_start = ((static_cast<std::size_t>(physical_page) * page_tokens + in_page_token) * kv_heads + kv_head) * width;
        float partial_dot = 0.0f;
        for (std::size_t column = lane_index; column < width; column += WAVE_WIDTH) {
            const float query = logismos_status_input(query_f32[query_start + column], numerical_status);
            const float key = logismos_status_input(keys_f32[kv_start + column], numerical_status);
            partial_dot = logismos_status_add(partial_dot, logismos_status_mul(query, key, numerical_status), numerical_status);
        }
        for (std::uint32_t shuffle_offset = WAVE_WIDTH / 2U; shuffle_offset > 0U; shuffle_offset /= 2U) {
            const float shuffled = __shfl_down(partial_dot, shuffle_offset, WAVE_WIDTH);
            if (lane < shuffle_offset) partial_dot = logismos_status_add(partial_dot, shuffled, numerical_status);
        }
        float score = lane == 0U ? logismos_status_mul(partial_dot, checked_scale, numerical_status) : 0.0f;
        score = logismos_status_result(__shfl(score, 0, WAVE_WIDTH), numerical_status);
        if (cache_token != 0U) logismos_status_arithmetic_operand(running_maximum, numerical_status);
        const float next_maximum = logismos_status_result(fmaxf(running_maximum, score), numerical_status);
        const float prior_rescale = cache_token == 0U ? 0.0f : logismos_status_exp(logismos_status_sub(running_maximum, next_maximum, numerical_status), numerical_status);
        const float token_weight = logismos_status_exp(logismos_status_sub(score, next_maximum, numerical_status), numerical_status);
        const float next_normalizer = logismos_status_add(logismos_status_mul(prior_rescale, running_normalizer, numerical_status), token_weight, numerical_status);
        for (std::size_t column = lane_index; column < width; column += WAVE_WIDTH) {
            const float prior = logismos_status_arithmetic_operand(output_f32[query_start + column], numerical_status);
            const float value = logismos_status_input(values_f32[kv_start + column], numerical_status);
            output_f32[query_start + column] = logismos_status_add(logismos_status_mul(prior_rescale, prior, numerical_status), logismos_status_mul(token_weight, value, numerical_status), numerical_status);
        }
        running_maximum = next_maximum;
        running_normalizer = next_normalizer;
    }
    for (std::size_t column = lane_index; column < width; column += WAVE_WIDTH) {
        output_f32[query_start + column] = logismos_status_div(logismos_status_arithmetic_operand(output_f32[query_start + column], numerical_status), logismos_status_arithmetic_operand(running_normalizer, numerical_status), numerical_status);
    }
}
#pragma clang fp contract(on)
#pragma clang fp reassociate(on)
