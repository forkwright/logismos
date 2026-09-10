// Grouped dense-f32 GDN staged correctness kernel with serial device time.
//
// One block owns one value head; one thread owns one value column. The key
// axis is serial so each column preserves the CPU recurrence's accumulation
// order. This is a staged primitive, not decoder or state-cache wiring.

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "numerical_status.h"

// WHY: the CPU recurrence performs separately rounded f32 multiply/add steps
// in key order. FMA contraction or reassociation would alter that reference
// sequence, so retain the controls only for this correctness-kernel source.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
extern "C" __global__ void logismos_gdn_grouped_step_f32_kernel(
    const float* __restrict__ q_f32,
    const float* __restrict__ k_f32,
    const float* __restrict__ v_f32,
    const float* __restrict__ beta_f32,
    const float* __restrict__ g_f32,
    const float* __restrict__ state_in_f32,
    float* __restrict__ state_out_f32,
    float* __restrict__ output_f32,
    float scale,
    std::uint32_t token_count,
    std::uint32_t key_head_count,
    std::uint32_t value_head_count,
    std::uint32_t key_dim,
    std::uint32_t value_dim,
    std::uint32_t* status)
{
    const std::uint32_t value_head = blockIdx.x;
    const std::uint32_t value_column = threadIdx.x;
    if (value_head >= value_head_count || value_column >= value_dim) {
        return;
    }

    const std::uint32_t heads_per_key = value_head_count / key_head_count;
    const std::uint32_t key_head = value_head / heads_per_key;
    const std::size_t key_width = static_cast<std::size_t>(key_dim);
    const std::size_t value_width = static_cast<std::size_t>(value_dim);
    const std::size_t head_state_width = key_width * value_width;
    const std::size_t state_head_offset = static_cast<std::size_t>(value_head) * head_state_width;
    const std::size_t token_width = static_cast<std::size_t>(token_count);
    for (std::size_t token = 0; token < token_width; ++token) {
        const std::size_t key_head_offset = (static_cast<std::size_t>(key_head) * token_width + token) * key_width;
        const std::size_t value_head_offset = (static_cast<std::size_t>(value_head) * token_width + token) * value_width;
        const float decay = logismos_status_exp(
            logismos_status_input(g_f32[static_cast<std::size_t>(value_head) * token_width + token], status), status);
        float projection = 0.0f;
        for (std::uint32_t key_index = 0; key_index < key_dim; ++key_index) {
            const std::size_t state_index = state_head_offset
                + static_cast<std::size_t>(key_index) * value_width + value_column;
            const float state = logismos_status_input(
                token == 0 ? state_in_f32[state_index] : state_out_f32[state_index], status);
            const float key = logismos_status_input(
                k_f32[key_head_offset + static_cast<std::size_t>(key_index)], status);
            const float decayed_state = logismos_status_mul(state, decay, status);
            projection = logismos_status_add(
                projection, logismos_status_mul(decayed_state, key, status), status);
        }
        const float delta = logismos_status_mul(
            logismos_status_input(beta_f32[static_cast<std::size_t>(value_head) * token_width + token], status),
            logismos_status_sub(
                logismos_status_input(v_f32[value_head_offset + value_column], status), projection, status),
            status);
        float accumulated_output = 0.0f;
        for (std::uint32_t key_index = 0; key_index < key_dim; ++key_index) {
            const std::size_t state_index = state_head_offset
                + static_cast<std::size_t>(key_index) * value_width + value_column;
            const float state = logismos_status_input(
                token == 0 ? state_in_f32[state_index] : state_out_f32[state_index], status);
            const float key = logismos_status_input(
                k_f32[key_head_offset + static_cast<std::size_t>(key_index)], status);
            const float next_state = logismos_status_add(
                logismos_status_mul(state, decay, status), logismos_status_mul(key, delta, status), status);
            state_out_f32[state_index] = logismos_status_result(next_state, status);
            const float query = logismos_status_input(
                q_f32[key_head_offset + static_cast<std::size_t>(key_index)], status);
            const float scaled = logismos_status_mul(
                logismos_status_mul(next_state, query, status), logismos_status_input(scale, status), status);
            accumulated_output = logismos_status_add(accumulated_output, scaled, status);
        }
        output_f32[value_head_offset + value_column] = logismos_status_result(accumulated_output, status);
    }
}
#pragma clang fp contract(on)
#pragma clang fp reassociate(on)
