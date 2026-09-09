// One-token dense-f32 causal-convolution correctness kernel.
//
// One thread owns one channel and serializes oldest-to-newest tap products and
// additions. This is a staged primitive, not model or decoder integration.

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

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
    std::uint32_t channel_count,
    std::uint32_t width)
{
    const auto channel = static_cast<std::uint32_t>(blockIdx.x);
    if (channel >= channel_count) {
        return;
    }

    const auto channel_index = static_cast<std::size_t>(channel);
    const auto width_index = static_cast<std::size_t>(width);
    const auto weight_start = channel_index * width_index;
    float accumulator = 0.0f;

    if (width == 1U) {
        accumulator = accumulator + input_f32[channel_index] * weights_f32[weight_start];
        output_f32[channel_index] = accumulator;
        return;
    }

    const auto history_width = width_index - 1U;
    const auto history_start = channel_index * history_width;
    // SAFETY: the Rust allocation-plan and span checks establish exact
    // non-aliasing device extents before this raw pointer arithmetic.
    for (std::size_t tap = 0; tap < history_width; ++tap) {
        const float product = history_in_f32[history_start + tap] * weights_f32[weight_start + tap];
        accumulator = accumulator + product;
    }
    const float current_product = input_f32[channel_index] * weights_f32[weight_start + history_width];
    accumulator = accumulator + current_product;
    output_f32[channel_index] = accumulator;

    for (std::size_t history_index = 0; history_index < history_width; ++history_index) {
        history_out_f32[history_start + history_index] = history_index + 1U < history_width
            ? history_in_f32[history_start + history_index + 1U]
            : input_f32[channel_index];
    }
}
#pragma clang fp contract(on)
#pragma clang fp reassociate(on)
