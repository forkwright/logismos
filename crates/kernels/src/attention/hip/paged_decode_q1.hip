// Q=1 paged-attention correctness kernel.
//
// One 32-lane wave owns one query head. It does not own cache state, model
// policy, or table construction; Rust validates only descriptor extents while
// the unsafe caller owns table values and device-residency obligations.

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

// WHY: preserve the explicitly documented native multiply/add and online
// update order. This operation is mathematically softmax-equivalent but does
// not claim the CPU materialized-score rounding sequence.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
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
    std::uint32_t page_tokens)
{
    constexpr std::uint32_t WAVE_WIDTH = 32U;
    const std::uint32_t query_head = blockIdx.x;
    const std::uint32_t lane = threadIdx.x;
    if (query_head >= query_heads) {
        return;
    }

    const std::uint32_t heads_per_kv = query_heads / kv_heads;
    const std::uint32_t kv_head = query_head / heads_per_kv;
    const std::size_t width = static_cast<std::size_t>(head_width);
    const std::size_t lane_index = static_cast<std::size_t>(lane);
    const std::size_t query_start = static_cast<std::size_t>(query_head) * width;
    const std::size_t output_start = query_start;
    float running_maximum = -INFINITY;
    float running_normalizer = 0.0f;

    // INVARIANT: all lanes execute every shuffle, including lanes without an
    // output coordinate when `head_width < 32` or the width has a tail.
    // INVARIANT: `head_width` crosses the ABI as u32 but coordinates promote
    // before every stride. On the supported 64-bit device index domain, the
    // final `+ WAVE_WIDTH` therefore cannot wrap near u32::MAX.
    for (std::size_t column = lane_index; column < width; column += WAVE_WIDTH) {
        output_f32[output_start + column] = 0.0f;
    }

    for (std::uint32_t token = 0; token < visible_tokens; ++token) {
        const std::uint32_t logical_page = token / page_tokens;
        const std::uint32_t in_page_token = token % page_tokens;
        const std::uint32_t physical_page = page_table_u32[logical_page];
        const std::size_t page_base =
            (static_cast<std::size_t>(physical_page) * static_cast<std::size_t>(page_tokens)
                + static_cast<std::size_t>(in_page_token))
            * static_cast<std::size_t>(kv_heads) * width;
        const std::size_t kv_start = page_base + static_cast<std::size_t>(kv_head) * width;

        float partial_dot = 0.0f;
        for (std::size_t column = lane_index; column < width; column += WAVE_WIDTH) {
            const float product = query_f32[query_start + column] * keys_f32[kv_start + column];
            partial_dot = partial_dot + product;
        }
        // INVARIANT: the fixed tree is the native dot-reduction order.
        for (std::uint32_t offset = WAVE_WIDTH / 2U; offset > 0U; offset /= 2U) {
            partial_dot += __shfl_down(partial_dot, offset, WAVE_WIDTH);
        }
        float score = lane == 0U ? partial_dot * scale : 0.0f;
        score = __shfl(score, 0, WAVE_WIDTH);

        const float next_maximum = fmaxf(running_maximum, score);
        const float prior_rescale = isinf(running_maximum) ? 0.0f : expf(running_maximum - next_maximum);
        const float token_weight = expf(score - next_maximum);
        const float next_normalizer = prior_rescale * running_normalizer + token_weight;
        for (std::size_t column = lane_index; column < width; column += WAVE_WIDTH) {
            const std::size_t output_index = output_start + column;
            output_f32[output_index] = prior_rescale * output_f32[output_index]
                + token_weight * values_f32[kv_start + column];
        }
        running_maximum = next_maximum;
        running_normalizer = next_normalizer;
    }

    for (std::size_t column = lane_index; column < width; column += WAVE_WIDTH) {
        const std::size_t output_index = output_start + column;
        output_f32[output_index] = output_f32[output_index] / running_normalizer;
    }
}
#pragma clang fp contract(on)
#pragma clang fp reassociate(on)
