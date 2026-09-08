// Sequential Q8_0 GEMV correctness kernel.
//
// One thread computes one output row. The deliberately serial inner loop
// preserves the format owner's lane order; it is not a WMMA/performance path.

#include <cstddef>
#include <cstdint>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "q8_0_format.h"

// WHY: `quant::q8_0::row_dot_f32` performs separate f32 decode, product, and
// accumulation steps in serialized order. Fast contraction would turn the
// latter two into FMA and change that correctness baseline, so disable it for
// this kernel's source scope only.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
extern "C" __global__ void logismos_q8_0_gemv_f32_kernel(
    const std::uint8_t* __restrict__ matrix_q8_0,
    const float* __restrict__ activations_f32,
    float* __restrict__ output_f32,
    int rows,
    int width)
{
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const std::size_t row_index = static_cast<std::size_t>(row);
    const std::size_t width_index = static_cast<std::size_t>(width);
    const std::size_t block_count = width_index / LOGISMOS_Q8_0_VALUES_PER_BLOCK;
    // SAFETY: the Rust shape and launcher contracts establish these exact
    // device-buffer extents and non-aliasing before this raw pointer arithmetic.
    const std::uint8_t* row_bytes = matrix_q8_0 + row_index * block_count * LOGISMOS_Q8_0_BLOCK_BYTES;
    float accumulator = 0.0f;
    for (std::size_t block_index = 0; block_index < block_count; ++block_index) {
        const std::uint8_t* block = row_bytes + block_index * LOGISMOS_Q8_0_BLOCK_BYTES;
        std::uint16_t scale_bits = 0;
        for (std::size_t byte_index = 0; byte_index < LOGISMOS_Q8_0_SCALE_BYTES; ++byte_index) {
            scale_bits |= static_cast<std::uint16_t>(block[byte_index])
                << (byte_index * static_cast<std::size_t>(8));
        }
        const float scale = __half2float(__ushort_as_half(scale_bits));
        for (std::size_t lane = 0; lane < LOGISMOS_Q8_0_VALUES_PER_BLOCK; ++lane) {
            const auto signed_value = static_cast<std::int8_t>(
                block[LOGISMOS_Q8_0_SCALE_BYTES + lane]);
            const float decoded = scale * static_cast<float>(signed_value);
            const float product = decoded * activations_f32[
                block_index * LOGISMOS_Q8_0_VALUES_PER_BLOCK + lane];
            accumulator = accumulator + product;
        }
    }
    output_f32[row_index] = accumulator;
}
#pragma clang fp contract(on)
#pragma clang fp reassociate(on)
