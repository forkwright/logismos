// Serialized-row GEMV correctness kernels.

#include <cstddef>
#include <cstdint>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "row_format.h"

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

namespace {
constexpr std::size_t BITS_PER_BYTE = 8;
constexpr std::uint8_t LOW_NIBBLE_MASK = 0x0f;
constexpr std::uint8_t LOW_SIX_BITS_MASK = 0x3f;
constexpr std::uint8_t HIGH_TWO_BITS_SHIFT = 6;

__device__ std::uint16_t load_u16_le(const std::uint8_t* bytes)
{
    return static_cast<std::uint16_t>(bytes[0])
        | (static_cast<std::uint16_t>(bytes[1]) << BITS_PER_BYTE);
}

__device__ float load_f16_le(const std::uint8_t* bytes)
{
    return __half2float(__ushort_as_half(load_u16_le(bytes)));
}

__device__ float load_f32_le(const std::uint8_t* bytes)
{
    const std::uint32_t bits = static_cast<std::uint32_t>(bytes[0])
        | (static_cast<std::uint32_t>(bytes[1]) << BITS_PER_BYTE)
        | (static_cast<std::uint32_t>(bytes[2]) << (BITS_PER_BYTE * 2))
        | (static_cast<std::uint32_t>(bytes[3]) << (BITS_PER_BYTE * 3));
    return __uint_as_float(bits);
}

__device__ int signed_i8(std::uint8_t value)
{
    return value < 128U ? static_cast<int>(value) : static_cast<int>(value) - 256;
}

__device__ void k_scale_min(const std::uint8_t* scales, std::size_t group, std::uint8_t* scale, std::uint8_t* minimum)
{
    if (group < LOGISMOS_K_PAIR_COUNT) {
        *scale = scales[group] & LOW_SIX_BITS_MASK;
        *minimum = scales[group + LOGISMOS_K_PAIR_COUNT] & LOW_SIX_BITS_MASK;
        return;
    }
    const std::uint8_t packed = scales[group + LOGISMOS_K_PAIR_COUNT];
    *scale = static_cast<std::uint8_t>(
        (packed & LOW_NIBBLE_MASK)
        | ((scales[group - LOGISMOS_K_PAIR_COUNT] >> HIGH_TWO_BITS_SHIFT) << 4));
    *minimum = static_cast<std::uint8_t>(
        (packed >> 4)
        | ((scales[group] >> HIGH_TWO_BITS_SHIFT) << 4));
}

__device__ bool checked_row_index(int rows, std::size_t* row)
{
    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x)
        * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (candidate >= static_cast<std::size_t>(rows)) {
        return false;
    }
    *row = candidate;
    return true;
}

__device__ float dot_f32(const std::uint8_t* matrix, const float* activations, std::size_t row, std::size_t width)
{
    const std::uint8_t* row_bytes = matrix + row * width * LOGISMOS_F32_VALUE_BYTES;
    float accumulator = 0.0f;
    for (std::size_t lane = 0; lane < width; ++lane) {
        const float decoded = load_f32_le(row_bytes + lane * LOGISMOS_F32_VALUE_BYTES);
        const float product = decoded * activations[lane];
        accumulator = accumulator + product;
    }
    return accumulator;
}

__device__ float dot_q8_0(const std::uint8_t* matrix, const float* activations, std::size_t row, std::size_t width)
{
    const std::size_t blocks = width / LOGISMOS_Q8_0_VALUES_PER_BLOCK;
    const std::uint8_t* row_bytes = matrix + row * blocks * LOGISMOS_Q8_0_BLOCK_BYTES;
    float accumulator = 0.0f;
    for (std::size_t block_index = 0; block_index < blocks; ++block_index) {
        const std::uint8_t* block = row_bytes + block_index * LOGISMOS_Q8_0_BLOCK_BYTES;
        const float scale = load_f16_le(block);
        for (std::size_t lane = 0; lane < LOGISMOS_Q8_0_VALUES_PER_BLOCK; ++lane) {
            const float decoded = scale * static_cast<float>(signed_i8(block[LOGISMOS_Q8_0_SCALE_BYTES + lane]));
            const float product = decoded * activations[block_index * LOGISMOS_Q8_0_VALUES_PER_BLOCK + lane];
            accumulator = accumulator + product;
        }
    }
    return accumulator;
}

__device__ float dot_q4_k(const std::uint8_t* matrix, const float* activations, std::size_t row, std::size_t width)
{
    const std::size_t blocks = width / LOGISMOS_Q4_K_VALUES_PER_BLOCK;
    const std::uint8_t* row_bytes = matrix + row * blocks * LOGISMOS_Q4_K_BLOCK_BYTES;
    float accumulator = 0.0f;
    for (std::size_t block_index = 0; block_index < blocks; ++block_index) {
        const std::uint8_t* block = row_bytes + block_index * LOGISMOS_Q4_K_BLOCK_BYTES;
        const float super_scale = load_f16_le(block);
        const float super_minimum = load_f16_le(block + LOGISMOS_Q4_K_SUPER_MINIMUM_OFFSET);
        const std::uint8_t* scales = block + LOGISMOS_Q4_K_SCALE_OFFSET;
        const std::uint8_t* quantized = block + LOGISMOS_Q4_K_QUANT_OFFSET;
        for (std::size_t pair = 0; pair < LOGISMOS_K_PAIR_COUNT; ++pair) {
            std::uint8_t low_scale = 0;
            std::uint8_t low_minimum = 0;
            std::uint8_t high_scale = 0;
            std::uint8_t high_minimum = 0;
            k_scale_min(scales, pair * 2, &low_scale, &low_minimum);
            k_scale_min(scales, pair * 2 + 1, &high_scale, &high_minimum);
            const float low_scale_f32 = super_scale * static_cast<float>(low_scale);
            const float low_minimum_f32 = super_minimum * static_cast<float>(low_minimum);
            const float high_scale_f32 = super_scale * static_cast<float>(high_scale);
            const float high_minimum_f32 = super_minimum * static_cast<float>(high_minimum);
            for (std::size_t lane = 0; lane < LOGISMOS_K_GROUP_VALUES; ++lane) {
                const std::uint8_t packed = quantized[pair * LOGISMOS_K_GROUP_VALUES + lane];
                const float low_decoded = low_scale_f32 * static_cast<float>(packed & LOW_NIBBLE_MASK) - low_minimum_f32;
                const float low_product = low_decoded * activations[block_index * LOGISMOS_Q4_K_VALUES_PER_BLOCK + pair * LOGISMOS_K_GROUP_VALUES * 2 + lane];
                accumulator = accumulator + low_product;
            }
            for (std::size_t lane = 0; lane < LOGISMOS_K_GROUP_VALUES; ++lane) {
                const std::uint8_t packed = quantized[pair * LOGISMOS_K_GROUP_VALUES + lane];
                const float high_decoded = high_scale_f32 * static_cast<float>(packed >> 4) - high_minimum_f32;
                const float high_product = high_decoded * activations[block_index * LOGISMOS_Q4_K_VALUES_PER_BLOCK + pair * LOGISMOS_K_GROUP_VALUES * 2 + LOGISMOS_K_GROUP_VALUES + lane];
                accumulator = accumulator + high_product;
            }
        }
    }
    return accumulator;
}

__device__ float dot_q5_k(const std::uint8_t* matrix, const float* activations, std::size_t row, std::size_t width)
{
    const std::size_t blocks = width / LOGISMOS_Q5_K_VALUES_PER_BLOCK;
    const std::uint8_t* row_bytes = matrix + row * blocks * LOGISMOS_Q5_K_BLOCK_BYTES;
    float accumulator = 0.0f;
    for (std::size_t block_index = 0; block_index < blocks; ++block_index) {
        const std::uint8_t* block = row_bytes + block_index * LOGISMOS_Q5_K_BLOCK_BYTES;
        const float super_scale = load_f16_le(block);
        const float super_minimum = load_f16_le(block + LOGISMOS_Q5_K_SUPER_MINIMUM_OFFSET);
        const std::uint8_t* scales = block + LOGISMOS_Q5_K_SCALE_OFFSET;
        const std::uint8_t* high_bits = block + LOGISMOS_Q5_K_HIGH_BITS_OFFSET;
        const std::uint8_t* quantized = block + LOGISMOS_Q5_K_QUANT_OFFSET;
        for (std::size_t pair = 0; pair < LOGISMOS_K_PAIR_COUNT; ++pair) {
            std::uint8_t low_scale = 0;
            std::uint8_t low_minimum = 0;
            std::uint8_t high_scale = 0;
            std::uint8_t high_minimum = 0;
            k_scale_min(scales, pair * 2, &low_scale, &low_minimum);
            k_scale_min(scales, pair * 2 + 1, &high_scale, &high_minimum);
            const float low_scale_f32 = super_scale * static_cast<float>(low_scale);
            const float low_minimum_f32 = super_minimum * static_cast<float>(low_minimum);
            const float high_scale_f32 = super_scale * static_cast<float>(high_scale);
            const float high_minimum_f32 = super_minimum * static_cast<float>(high_minimum);
            const std::uint8_t low_bit = static_cast<std::uint8_t>(1U << (pair * 2));
            const std::uint8_t high_bit = static_cast<std::uint8_t>(low_bit << 1);
            for (std::size_t lane = 0; lane < LOGISMOS_K_GROUP_VALUES; ++lane) {
                const std::uint8_t packed = quantized[pair * LOGISMOS_K_GROUP_VALUES + lane];
                const std::uint8_t fifth = high_bits[lane];
                const std::uint8_t low = static_cast<std::uint8_t>((packed & LOW_NIBBLE_MASK) + ((fifth & low_bit) == 0 ? 0 : 16));
                const float low_decoded = low_scale_f32 * static_cast<float>(low) - low_minimum_f32;
                const float low_product = low_decoded * activations[block_index * LOGISMOS_Q5_K_VALUES_PER_BLOCK + pair * LOGISMOS_K_GROUP_VALUES * 2 + lane];
                accumulator = accumulator + low_product;
            }
            for (std::size_t lane = 0; lane < LOGISMOS_K_GROUP_VALUES; ++lane) {
                const std::uint8_t packed = quantized[pair * LOGISMOS_K_GROUP_VALUES + lane];
                const std::uint8_t fifth = high_bits[lane];
                const std::uint8_t high = static_cast<std::uint8_t>((packed >> 4) + ((fifth & high_bit) == 0 ? 0 : 16));
                const float high_decoded = high_scale_f32 * static_cast<float>(high) - high_minimum_f32;
                const float high_product = high_decoded * activations[block_index * LOGISMOS_Q5_K_VALUES_PER_BLOCK + pair * LOGISMOS_K_GROUP_VALUES * 2 + LOGISMOS_K_GROUP_VALUES + lane];
                accumulator = accumulator + high_product;
            }
        }
    }
    return accumulator;
}

__device__ float dot_q6_k(const std::uint8_t* matrix, const float* activations, std::size_t row, std::size_t width)
{
    const std::size_t blocks = width / LOGISMOS_Q6_K_VALUES_PER_BLOCK;
    const std::uint8_t* row_bytes = matrix + row * blocks * LOGISMOS_Q6_K_BLOCK_BYTES;
    float accumulator = 0.0f;
    for (std::size_t block_index = 0; block_index < blocks; ++block_index) {
        const std::uint8_t* block = row_bytes + block_index * LOGISMOS_Q6_K_BLOCK_BYTES;
        const std::uint8_t* low = block;
        const std::uint8_t* high = block + LOGISMOS_Q6_K_HIGH_BITS_OFFSET;
        const std::uint8_t* scales = block + LOGISMOS_Q6_K_SCALE_OFFSET;
        const float super_scale = load_f16_le(block + LOGISMOS_Q6_K_SUPER_SCALE_OFFSET);
        for (std::size_t half = 0; half < LOGISMOS_Q6_K_HALF_BLOCK_COUNT; ++half) {
            for (std::size_t quarter = 0; quarter < LOGISMOS_Q6_K_QUARTERS_PER_HALF_BLOCK; ++quarter) {
                for (std::size_t lane = 0; lane < LOGISMOS_Q6_K_VALUES_PER_QUARTER; ++lane) {
                    const std::uint8_t packed_high = high[half * LOGISMOS_Q6_K_HIGH_BYTES_PER_HALF + lane];
                    const std::uint8_t low_byte = low[half * LOGISMOS_Q6_K_LOW_BYTES_PER_HALF + (quarter % 2) * LOGISMOS_Q6_K_VALUES_PER_QUARTER + lane];
                    const std::uint8_t lower = quarter < 2 ? low_byte & LOW_NIBBLE_MASK : low_byte >> 4;
                    const std::uint8_t upper = static_cast<std::uint8_t>((packed_high >> (quarter * 2)) & 0x03);
                    const int quantized = static_cast<int>((upper << 4) | lower) - 32;
                    const std::size_t scale_index = half * LOGISMOS_Q6_K_SCALES_PER_HALF + quarter * 2 + lane / LOGISMOS_Q6_K_VALUES_PER_SCALE;
                    const float decoded = super_scale * static_cast<float>(signed_i8(scales[scale_index])) * static_cast<float>(quantized);
                    const std::size_t activation_index = block_index * LOGISMOS_Q6_K_VALUES_PER_BLOCK + half * LOGISMOS_Q6_K_VALUES_PER_HALF + quarter * LOGISMOS_Q6_K_VALUES_PER_QUARTER + lane;
                    const float product = decoded * activations[activation_index];
                    accumulator = accumulator + product;
                }
            }
        }
    }
    return accumulator;
}

__device__ float dot_iq4_nl(const std::uint8_t* matrix, const float* activations, std::size_t row, std::size_t width)
{
    const std::size_t blocks = width / LOGISMOS_IQ4_NL_VALUES_PER_BLOCK;
    const std::uint8_t* row_bytes = matrix + row * blocks * LOGISMOS_IQ4_NL_BLOCK_BYTES;
    float accumulator = 0.0f;
    for (std::size_t block_index = 0; block_index < blocks; ++block_index) {
        const std::uint8_t* block = row_bytes + block_index * LOGISMOS_IQ4_NL_BLOCK_BYTES;
        const float scale = load_f16_le(block);
        for (std::size_t lane = 0; lane < LOGISMOS_IQ4_NL_VALUES_PER_BLOCK / 2; ++lane) {
            const std::uint8_t packed = block[LOGISMOS_IQ4_NL_QUANT_OFFSET + lane];
            const float low_decoded = scale * static_cast<float>(LOGISMOS_IQ4_RECONSTRUCTION_VALUES[packed & LOW_NIBBLE_MASK]);
            const float low_product = low_decoded * activations[block_index * LOGISMOS_IQ4_NL_VALUES_PER_BLOCK + lane];
            accumulator = accumulator + low_product;
        }
        for (std::size_t lane = 0; lane < LOGISMOS_IQ4_NL_VALUES_PER_BLOCK / 2; ++lane) {
            const std::uint8_t packed = block[LOGISMOS_IQ4_NL_QUANT_OFFSET + lane];
            const float high_decoded = scale * static_cast<float>(LOGISMOS_IQ4_RECONSTRUCTION_VALUES[packed >> 4]);
            const float high_product = high_decoded * activations[block_index * LOGISMOS_IQ4_NL_VALUES_PER_BLOCK + LOGISMOS_IQ4_NL_VALUES_PER_BLOCK / 2 + lane];
            accumulator = accumulator + high_product;
        }
    }
    return accumulator;
}

__device__ float dot_iq4_xs(const std::uint8_t* matrix, const float* activations, std::size_t row, std::size_t width)
{
    const std::size_t blocks = width / LOGISMOS_IQ4_XS_VALUES_PER_BLOCK;
    const std::uint8_t* row_bytes = matrix + row * blocks * LOGISMOS_IQ4_XS_BLOCK_BYTES;
    float accumulator = 0.0f;
    for (std::size_t block_index = 0; block_index < blocks; ++block_index) {
        const std::uint8_t* block = row_bytes + block_index * LOGISMOS_IQ4_XS_BLOCK_BYTES;
        const float block_scale = load_f16_le(block);
        const std::uint16_t scale_high = load_u16_le(block + LOGISMOS_IQ4_XS_SCALE_HIGH_OFFSET);
        for (std::size_t group = 0; group < LOGISMOS_IQ4_XS_VALUES_PER_BLOCK / LOGISMOS_IQ4_XS_GROUP_VALUES; ++group) {
            const std::uint8_t scale_low_byte = block[LOGISMOS_IQ4_XS_SCALE_LOW_OFFSET + group / 2];
            const std::uint8_t low_bits = group % 2 == 0 ? scale_low_byte & LOW_NIBBLE_MASK : scale_low_byte >> 4;
            const std::uint8_t high_bits = static_cast<std::uint8_t>((scale_high >> (group * 2)) & 0x03);
            const int group_scale = static_cast<int>(low_bits | (high_bits << 4)) - 32;
            const float value_scale = block_scale * static_cast<float>(group_scale);
            for (std::size_t lane = 0; lane < LOGISMOS_IQ4_XS_GROUP_VALUES / 2; ++lane) {
                const std::uint8_t packed = block[LOGISMOS_IQ4_XS_QUANT_OFFSET + group * (LOGISMOS_IQ4_XS_GROUP_VALUES / 2) + lane];
                const float low_decoded = value_scale * static_cast<float>(LOGISMOS_IQ4_RECONSTRUCTION_VALUES[packed & LOW_NIBBLE_MASK]);
                const float low_product = low_decoded * activations[block_index * LOGISMOS_IQ4_XS_VALUES_PER_BLOCK + group * LOGISMOS_IQ4_XS_GROUP_VALUES + lane];
                accumulator = accumulator + low_product;
            }
            for (std::size_t lane = 0; lane < LOGISMOS_IQ4_XS_GROUP_VALUES / 2; ++lane) {
                const std::uint8_t packed = block[LOGISMOS_IQ4_XS_QUANT_OFFSET + group * (LOGISMOS_IQ4_XS_GROUP_VALUES / 2) + lane];
                const float high_decoded = value_scale * static_cast<float>(LOGISMOS_IQ4_RECONSTRUCTION_VALUES[packed >> 4]);
                const float high_product = high_decoded * activations[block_index * LOGISMOS_IQ4_XS_VALUES_PER_BLOCK + group * LOGISMOS_IQ4_XS_GROUP_VALUES + LOGISMOS_IQ4_XS_GROUP_VALUES / 2 + lane];
                accumulator = accumulator + high_product;
            }
        }
    }
    return accumulator;
}

}

extern "C" __global__ void logismos_f32_row_gemv_f32_kernel(const std::uint8_t* matrix, const float* activations, float* output, int rows, int width)
{
    std::size_t row = 0;
    if (checked_row_index(rows, &row)) {
        output[row] = dot_f32(matrix, activations, row, static_cast<std::size_t>(width));
    }
}

extern "C" __global__ void logismos_q8_0_row_gemv_f32_kernel(const std::uint8_t* matrix, const float* activations, float* output, int rows, int width)
{
    std::size_t row = 0;
    if (checked_row_index(rows, &row)) {
        output[row] = dot_q8_0(matrix, activations, row, static_cast<std::size_t>(width));
    }
}

extern "C" __global__ void logismos_q4_k_row_gemv_f32_kernel(const std::uint8_t* matrix, const float* activations, float* output, int rows, int width)
{
    std::size_t row = 0;
    if (checked_row_index(rows, &row)) {
        output[row] = dot_q4_k(matrix, activations, row, static_cast<std::size_t>(width));
    }
}

extern "C" __global__ void logismos_q5_k_row_gemv_f32_kernel(const std::uint8_t* matrix, const float* activations, float* output, int rows, int width)
{
    std::size_t row = 0;
    if (checked_row_index(rows, &row)) {
        output[row] = dot_q5_k(matrix, activations, row, static_cast<std::size_t>(width));
    }
}

extern "C" __global__ void logismos_q6_k_row_gemv_f32_kernel(const std::uint8_t* matrix, const float* activations, float* output, int rows, int width)
{
    std::size_t row = 0;
    if (checked_row_index(rows, &row)) {
        output[row] = dot_q6_k(matrix, activations, row, static_cast<std::size_t>(width));
    }
}

extern "C" __global__ void logismos_iq4_nl_row_gemv_f32_kernel(const std::uint8_t* matrix, const float* activations, float* output, int rows, int width)
{
    std::size_t row = 0;
    if (checked_row_index(rows, &row)) {
        output[row] = dot_iq4_nl(matrix, activations, row, static_cast<std::size_t>(width));
    }
}

extern "C" __global__ void logismos_iq4_xs_row_gemv_f32_kernel(const std::uint8_t* matrix, const float* activations, float* output, int rows, int width)
{
    std::size_t row = 0;
    if (checked_row_index(rows, &row)) {
        output[row] = dot_iq4_xs(matrix, activations, row, static_cast<std::size_t>(width));
    }
}

#pragma clang fp contract(on)
#pragma clang fp reassociate(on)
