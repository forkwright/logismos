// NOTE: dense-f32 causal-convolution correctness kernel with serial device time.
//
// One thread owns one channel and serializes oldest-to-newest tap products and
// additions. This is a staged primitive, not model or decoder integration.

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "numerical_status.h"

// WHY: `causal_conv_fwd` makes separately rounded f32 products and additions
// in oldest-to-newest order. Fast contraction or reassociation would change
// that correctness baseline, so the restriction is local to this source.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
extern "C" __global__ void logismos_causal_conv_step_f32_kernel(
    const float* __restrict__ input_f32,
    const float* __restrict__ weights_f32,
    const float* __restrict__ history_in_f32,
    float* __restrict__ history_out_f32,
    float* __restrict__ output_f32,
    std::uint32_t token_count,
    std::uint32_t channel_count,
    std::uint32_t width,
    std::uint32_t* status)
{
    const auto channel = static_cast<std::uint64_t>(blockIdx.x)
        * static_cast<std::uint64_t>(blockDim.x)
        + static_cast<std::uint64_t>(threadIdx.x);
    if (channel >= channel_count) {
        return;
    }

    const auto channel_index = static_cast<std::size_t>(channel);
    const auto width_index = static_cast<std::size_t>(width);
    const auto weight_start = channel_index * width_index;
    const auto history_width = width_index - 1U;
    const auto history_start = channel_index * history_width;
    for (std::size_t token = 0; token < static_cast<std::size_t>(token_count); ++token) {
        float accumulator = 0.0f;
        for (std::size_t tap = 0; tap < width_index; ++tap) {
            const std::size_t position = token + tap;
            const float sample = position < history_width
                ? logismos_status_input(history_in_f32[history_start + position], status)
                : logismos_status_input(
                    input_f32[(position - history_width) * static_cast<std::size_t>(channel_count) + channel_index],
                    status);
            const float product = logismos_status_mul(sample, logismos_status_input(weights_f32[weight_start + tap], status), status);
            accumulator = logismos_status_add(accumulator, product, status);
        }
        output_f32[token * static_cast<std::size_t>(channel_count) + channel_index] =
            logismos_status_result(accumulator, status);
    }
    for (std::size_t history_index = 0; history_index < history_width; ++history_index) {
        const std::size_t position = static_cast<std::size_t>(token_count) + history_index;
        const float next_history = position < history_width
            ? logismos_status_input(history_in_f32[history_start + position], status)
            : logismos_status_input(
                input_f32[(position - history_width) * static_cast<std::size_t>(channel_count) + channel_index],
                status);
        history_out_f32[history_start + history_index] = logismos_status_result(next_history, status);
    }
}
#pragma clang fp contract(on)
#pragma clang fp reassociate(on)
