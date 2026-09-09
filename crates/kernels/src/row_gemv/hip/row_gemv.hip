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

// WHY: GEMV and selected-row lookup must not drift into distinct serialized
// format decoders; both paths consume these generated-layout reconstruction owners.
__device__ float decode_f32_element(const std::uint8_t* row_bytes, std::size_t lane)
{
    return load_f32_le(row_bytes + lane * LOGISMOS_F32_VALUE_BYTES);
}

__device__ float decode_q8_0_element(const std::uint8_t* block, std::size_t lane)
{
    const float scale = load_f16_le(block);
    return scale * static_cast<float>(signed_i8(block[LOGISMOS_Q8_0_SCALE_BYTES + lane]));
}

__device__ float decode_q4_k_element(const std::uint8_t* block, std::size_t lane)
{
    const std::size_t group = lane / LOGISMOS_K_GROUP_VALUES;
    const std::size_t pair = group / 2;
    const std::size_t group_lane = lane % LOGISMOS_K_GROUP_VALUES;
    std::uint8_t scale = 0;
    std::uint8_t minimum = 0;
    k_scale_min(block + LOGISMOS_Q4_K_SCALE_OFFSET, group, &scale, &minimum);
    const std::uint8_t packed = block[
        LOGISMOS_Q4_K_QUANT_OFFSET + pair * LOGISMOS_K_GROUP_VALUES + group_lane];
    const std::uint8_t quantized = group % 2 == 0
        ? packed & LOW_NIBBLE_MASK
        : packed >> 4;
    const float value_scale = load_f16_le(block) * static_cast<float>(scale);
    const float value_minimum = load_f16_le(block + LOGISMOS_Q4_K_SUPER_MINIMUM_OFFSET)
        * static_cast<float>(minimum);
    return value_scale * static_cast<float>(quantized) - value_minimum;
}

__device__ float decode_q5_k_element(const std::uint8_t* block, std::size_t lane)
{
    const std::size_t group = lane / LOGISMOS_K_GROUP_VALUES;
    const std::size_t pair = group / 2;
    const std::size_t group_lane = lane % LOGISMOS_K_GROUP_VALUES;
    std::uint8_t scale = 0;
    std::uint8_t minimum = 0;
    k_scale_min(block + LOGISMOS_Q5_K_SCALE_OFFSET, group, &scale, &minimum);
    const std::uint8_t packed = block[
        LOGISMOS_Q5_K_QUANT_OFFSET + pair * LOGISMOS_K_GROUP_VALUES + group_lane];
    const std::uint8_t low = group % 2 == 0
        ? packed & LOW_NIBBLE_MASK
        : packed >> 4;
    const std::uint8_t fifth_bit = static_cast<std::uint8_t>(1U << group);
    const std::uint8_t high =
        (block[LOGISMOS_Q5_K_HIGH_BITS_OFFSET + group_lane] & fifth_bit) == 0 ? 0 : 16;
    const std::uint8_t quantized = static_cast<std::uint8_t>(low + high);
    const float value_scale = load_f16_le(block) * static_cast<float>(scale);
    const float value_minimum = load_f16_le(block + LOGISMOS_Q5_K_SUPER_MINIMUM_OFFSET)
        * static_cast<float>(minimum);
    return value_scale * static_cast<float>(quantized) - value_minimum;
}

__device__ float decode_q6_k_element(const std::uint8_t* block, std::size_t lane)
{
    const std::size_t half = lane / LOGISMOS_Q6_K_VALUES_PER_HALF;
    const std::size_t remainder = lane % LOGISMOS_Q6_K_VALUES_PER_HALF;
    const std::size_t quarter = remainder / LOGISMOS_Q6_K_VALUES_PER_QUARTER;
    const std::size_t quarter_lane = remainder % LOGISMOS_Q6_K_VALUES_PER_QUARTER;
    const std::uint8_t low_byte = block[
        half * LOGISMOS_Q6_K_LOW_BYTES_PER_HALF
        + (quarter % 2) * LOGISMOS_Q6_K_VALUES_PER_QUARTER
        + quarter_lane];
    const std::uint8_t lower = quarter < 2
        ? low_byte & LOW_NIBBLE_MASK
        : low_byte >> 4;
    const std::uint8_t upper = static_cast<std::uint8_t>(
        (block[
            LOGISMOS_Q6_K_HIGH_BITS_OFFSET
            + half * LOGISMOS_Q6_K_HIGH_BYTES_PER_HALF
            + quarter_lane]
            >> (quarter * 2))
        & 0x03);
    const int quantized = static_cast<int>((upper << 4) | lower) - 32;
    const std::size_t scale_index = half * LOGISMOS_Q6_K_SCALES_PER_HALF
        + quarter * 2
        + quarter_lane / LOGISMOS_Q6_K_VALUES_PER_SCALE;
    const float value_scale = load_f16_le(block + LOGISMOS_Q6_K_SUPER_SCALE_OFFSET)
        * static_cast<float>(signed_i8(block[LOGISMOS_Q6_K_SCALE_OFFSET + scale_index]));
    return value_scale * static_cast<float>(quantized);
}

__device__ float decode_iq4_nl_element(const std::uint8_t* block, std::size_t lane)
{
    const std::size_t half_width = LOGISMOS_IQ4_NL_VALUES_PER_BLOCK / 2;
    const std::uint8_t packed = block[LOGISMOS_IQ4_NL_QUANT_OFFSET + lane % half_width];
    const std::uint8_t index = lane < half_width
        ? packed & LOW_NIBBLE_MASK
        : packed >> 4;
    return load_f16_le(block)
        * static_cast<float>(LOGISMOS_IQ4_RECONSTRUCTION_VALUES[index]);
}

__device__ float decode_iq4_xs_element(const std::uint8_t* block, std::size_t lane)
{
    const std::size_t group = lane / LOGISMOS_IQ4_XS_GROUP_VALUES;
    const std::size_t group_lane = lane % LOGISMOS_IQ4_XS_GROUP_VALUES;
    const std::uint8_t scale_low_byte =
        block[LOGISMOS_IQ4_XS_SCALE_LOW_OFFSET + group / 2];
    const std::uint8_t low_bits = group % 2 == 0
        ? scale_low_byte & LOW_NIBBLE_MASK
        : scale_low_byte >> 4;
    const std::uint16_t scale_high =
        load_u16_le(block + LOGISMOS_IQ4_XS_SCALE_HIGH_OFFSET);
    const std::uint8_t high_bits =
        static_cast<std::uint8_t>((scale_high >> (group * 2)) & 0x03);
    const int group_scale = static_cast<int>(low_bits | (high_bits << 4)) - 32;
    const std::size_t half_width = LOGISMOS_IQ4_XS_GROUP_VALUES / 2;
    const std::uint8_t packed = block[
        LOGISMOS_IQ4_XS_QUANT_OFFSET + group * half_width + group_lane % half_width];
    const std::uint8_t index = group_lane < half_width
        ? packed & LOW_NIBBLE_MASK
        : packed >> 4;
    const float value_scale = load_f16_le(block) * static_cast<float>(group_scale);
    return value_scale * static_cast<float>(LOGISMOS_IQ4_RECONSTRUCTION_VALUES[index]);
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

__device__ bool checked_value_index(int width, std::size_t* lane)
{
    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x)
        * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (candidate >= static_cast<std::size_t>(width)) {
        return false;
    }
    *lane = candidate;
    return true;
}

__device__ const std::uint8_t* selected_block(
    const std::uint8_t* matrix,
    int row,
    int width,
    std::size_t lane,
    std::size_t values_per_block,
    std::size_t block_bytes)
{
    const std::size_t blocks = static_cast<std::size_t>(width) / values_per_block;
    const std::size_t row_offset = static_cast<std::size_t>(row) * blocks * block_bytes;
    const std::size_t block_offset = lane / values_per_block * block_bytes;
    return matrix + row_offset + block_offset;
}

__device__ float dot_f32(const std::uint8_t* matrix, const float* activations, std::size_t row, std::size_t width)
{
    const std::uint8_t* row_bytes = matrix + row * width * LOGISMOS_F32_VALUE_BYTES;
    float accumulator = 0.0f;
    for (std::size_t lane = 0; lane < width; ++lane) {
        const float decoded = decode_f32_element(row_bytes, lane);
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
        for (std::size_t lane = 0; lane < LOGISMOS_Q8_0_VALUES_PER_BLOCK; ++lane) {
            const float decoded = decode_q8_0_element(block, lane);
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
        for (std::size_t lane = 0; lane < LOGISMOS_Q4_K_VALUES_PER_BLOCK; ++lane) {
            const float decoded = decode_q4_k_element(block, lane);
            const float product = decoded * activations[block_index * LOGISMOS_Q4_K_VALUES_PER_BLOCK + lane];
            accumulator = accumulator + product;
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
        for (std::size_t lane = 0; lane < LOGISMOS_Q5_K_VALUES_PER_BLOCK; ++lane) {
            const float decoded = decode_q5_k_element(block, lane);
            const float product = decoded * activations[block_index * LOGISMOS_Q5_K_VALUES_PER_BLOCK + lane];
            accumulator = accumulator + product;
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
        for (std::size_t lane = 0; lane < LOGISMOS_Q6_K_VALUES_PER_BLOCK; ++lane) {
            const float decoded = decode_q6_k_element(block, lane);
            const float product = decoded * activations[block_index * LOGISMOS_Q6_K_VALUES_PER_BLOCK + lane];
            accumulator = accumulator + product;
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
        for (std::size_t lane = 0; lane < LOGISMOS_IQ4_NL_VALUES_PER_BLOCK; ++lane) {
            const float decoded = decode_iq4_nl_element(block, lane);
            const float product = decoded * activations[block_index * LOGISMOS_IQ4_NL_VALUES_PER_BLOCK + lane];
            accumulator = accumulator + product;
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
        for (std::size_t lane = 0; lane < LOGISMOS_IQ4_XS_VALUES_PER_BLOCK; ++lane) {
            const float decoded = decode_iq4_xs_element(block, lane);
            const float product = decoded * activations[block_index * LOGISMOS_IQ4_XS_VALUES_PER_BLOCK + lane];
            accumulator = accumulator + product;
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

extern "C" __global__ void logismos_f32_row_decode_f32_kernel(const std::uint8_t* matrix, float* output, int row, int width)
{
    std::size_t lane = 0;
    if (!checked_value_index(width, &lane)) {
        return;
    }
    const std::uint8_t* row_bytes = matrix
        + static_cast<std::size_t>(row) * static_cast<std::size_t>(width)
            * LOGISMOS_F32_VALUE_BYTES;
    output[lane] = decode_f32_element(row_bytes, lane);
}

extern "C" __global__ void logismos_q8_0_row_decode_f32_kernel(const std::uint8_t* matrix, float* output, int row, int width)
{
    std::size_t lane = 0;
    if (!checked_value_index(width, &lane)) {
        return;
    }
    const std::uint8_t* block = selected_block(
        matrix,
        row,
        width,
        lane,
        LOGISMOS_Q8_0_VALUES_PER_BLOCK,
        LOGISMOS_Q8_0_BLOCK_BYTES);
    output[lane] = decode_q8_0_element(block, lane % LOGISMOS_Q8_0_VALUES_PER_BLOCK);
}

extern "C" __global__ void logismos_q4_k_row_decode_f32_kernel(const std::uint8_t* matrix, float* output, int row, int width)
{
    std::size_t lane = 0;
    if (!checked_value_index(width, &lane)) {
        return;
    }
    const std::uint8_t* block = selected_block(
        matrix,
        row,
        width,
        lane,
        LOGISMOS_Q4_K_VALUES_PER_BLOCK,
        LOGISMOS_Q4_K_BLOCK_BYTES);
    output[lane] = decode_q4_k_element(block, lane % LOGISMOS_Q4_K_VALUES_PER_BLOCK);
}

extern "C" __global__ void logismos_q5_k_row_decode_f32_kernel(const std::uint8_t* matrix, float* output, int row, int width)
{
    std::size_t lane = 0;
    if (!checked_value_index(width, &lane)) {
        return;
    }
    const std::uint8_t* block = selected_block(
        matrix,
        row,
        width,
        lane,
        LOGISMOS_Q5_K_VALUES_PER_BLOCK,
        LOGISMOS_Q5_K_BLOCK_BYTES);
    output[lane] = decode_q5_k_element(block, lane % LOGISMOS_Q5_K_VALUES_PER_BLOCK);
}

extern "C" __global__ void logismos_q6_k_row_decode_f32_kernel(const std::uint8_t* matrix, float* output, int row, int width)
{
    std::size_t lane = 0;
    if (!checked_value_index(width, &lane)) {
        return;
    }
    const std::uint8_t* block = selected_block(
        matrix,
        row,
        width,
        lane,
        LOGISMOS_Q6_K_VALUES_PER_BLOCK,
        LOGISMOS_Q6_K_BLOCK_BYTES);
    output[lane] = decode_q6_k_element(block, lane % LOGISMOS_Q6_K_VALUES_PER_BLOCK);
}

extern "C" __global__ void logismos_iq4_nl_row_decode_f32_kernel(const std::uint8_t* matrix, float* output, int row, int width)
{
    std::size_t lane = 0;
    if (!checked_value_index(width, &lane)) {
        return;
    }
    const std::uint8_t* block = selected_block(
        matrix,
        row,
        width,
        lane,
        LOGISMOS_IQ4_NL_VALUES_PER_BLOCK,
        LOGISMOS_IQ4_NL_BLOCK_BYTES);
    output[lane] = decode_iq4_nl_element(block, lane % LOGISMOS_IQ4_NL_VALUES_PER_BLOCK);
}

extern "C" __global__ void logismos_iq4_xs_row_decode_f32_kernel(const std::uint8_t* matrix, float* output, int row, int width)
{
    std::size_t lane = 0;
    if (!checked_value_index(width, &lane)) {
        return;
    }
    const std::uint8_t* block = selected_block(
        matrix,
        row,
        width,
        lane,
        LOGISMOS_IQ4_XS_VALUES_PER_BLOCK,
        LOGISMOS_IQ4_XS_BLOCK_BYTES);
    output[lane] = decode_iq4_xs_element(block, lane % LOGISMOS_IQ4_XS_VALUES_PER_BLOCK);
}

namespace {
constexpr unsigned int ROW_DECODE_THREADS_PER_BLOCK = 256;

unsigned int row_decode_blocks(int width)
{
    const auto values = static_cast<unsigned long long>(width);
    return static_cast<unsigned int>(
        (values + ROW_DECODE_THREADS_PER_BLOCK - 1U) / ROW_DECODE_THREADS_PER_BLOCK);
}
}

#define LOGISMOS_ROW_DECODE_LAUNCHER(NAME) \
extern "C" hipError_t logismos_launch_##NAME##_row_decode_f32( \
    const void* matrix, void* output, int row, int width, hipStream_t stream) \
{ \
    const dim3 block(ROW_DECODE_THREADS_PER_BLOCK, 1, 1); \
    const dim3 grid(row_decode_blocks(width), 1, 1); \
    /* SAFETY: Rust validates exact device spans; the caller retains inputs and output through completion. */ \
    logismos_##NAME##_row_decode_f32_kernel<<<grid, block, 0, stream>>>( \
        reinterpret_cast<const std::uint8_t*>(matrix), \
        reinterpret_cast<float*>(output), row, width); \
    return hipGetLastError(); \
}

LOGISMOS_ROW_DECODE_LAUNCHER(f32)
LOGISMOS_ROW_DECODE_LAUNCHER(q8_0)
LOGISMOS_ROW_DECODE_LAUNCHER(q4_k)
LOGISMOS_ROW_DECODE_LAUNCHER(q5_k)
LOGISMOS_ROW_DECODE_LAUNCHER(q6_k)
LOGISMOS_ROW_DECODE_LAUNCHER(iq4_nl)
LOGISMOS_ROW_DECODE_LAUNCHER(iq4_xs)

#undef LOGISMOS_ROW_DECODE_LAUNCHER

#pragma clang fp contract(on)
#pragma clang fp reassociate(on)
