// Grouped dense-f32 GDN one-token staged correctness kernel.
//
// One block owns one value head; one thread owns one value column. The key
// axis is serial so each column preserves the CPU recurrence's accumulation
// order. This is a staged primitive, not decoder or state-cache wiring.

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

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
    std::uint32_t key_head_count,
    std::uint32_t value_head_count,
    std::uint32_t key_dim,
    std::uint32_t value_dim)
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
    const std::size_t key_head_offset = static_cast<std::size_t>(key_head) * key_width;
    const std::size_t value_head_offset = static_cast<std::size_t>(value_head) * value_width;
    const std::size_t state_head_offset = static_cast<std::size_t>(value_head) * head_state_width;
    const float decay = expf(g_f32[value_head]);

    float projection = 0.0f;
    for (std::uint32_t key_index = 0; key_index < key_dim; ++key_index) {
        const std::size_t state_index = state_head_offset
            + static_cast<std::size_t>(key_index) * value_width + value_column;
        // SAFETY: the Rust launch contract validates the exact non-aliasing
        // device spans; this one-token profile bounds every derived index.
        const float decayed_state = state_in_f32[state_index] * decay;
        projection = projection + decayed_state
            * k_f32[key_head_offset + static_cast<std::size_t>(key_index)];
    }

    const float delta = beta_f32[value_head]
        * (v_f32[value_head_offset + value_column] - projection);
    float accumulated_output = 0.0f;
    for (std::uint32_t key_index = 0; key_index < key_dim; ++key_index) {
        const std::size_t state_index = state_head_offset
            + static_cast<std::size_t>(key_index) * value_width + value_column;
        const float next_state = state_in_f32[state_index] * decay
            + k_f32[key_head_offset + static_cast<std::size_t>(key_index)] * delta;
        state_out_f32[state_index] = next_state;
        accumulated_output = accumulated_output
            + (next_state * q_f32[key_head_offset + static_cast<std::size_t>(key_index)]) * scale;
    }
    output_f32[value_head_offset + value_column] = accumulated_output;
}
#pragma clang fp contract(on)
#pragma clang fp reassociate(on)
