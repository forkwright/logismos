// Token-aware Qwen recurrent Q/K arrangement and L2 normalization.
//
// The grouped GDN primitive has a floor key-head mapping. Qwen recurrent
// semantics instead tile source heads in modulo order, so this producer emits
// explicitly tiled Q/K rows before GDN receives an equal-head configuration.

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "numerical_status.h"

// WHY: this is a correctness baseline for Qwen's separately rounded f32 L2
// arithmetic. Keep contraction and reassociation disabled only for this source.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

namespace {
constexpr std::uint32_t kWaveSize = 32;

__global__ void decoder_recurrent_qk_l2_f32_kernel(
    const float* __restrict__ convolved,
    float* __restrict__ query,
    float* __restrict__ key,
    const std::uint32_t token_count,
    const std::uint32_t convolved_row_elements,
    const std::uint32_t source_key_heads,
    const std::uint32_t key_width,
    const float epsilon,
    std::uint32_t* numerical_status) {
    const std::uint32_t value_head = blockIdx.x;
    const std::uint32_t token = blockIdx.y;
    const std::uint32_t lane = threadIdx.x;
    if (lane >= kWaveSize) {
        return;
    }
    const std::uint32_t source_head = value_head % source_key_heads;
    const std::size_t width = static_cast<std::size_t>(key_width);
    const std::size_t tokens = static_cast<std::size_t>(token_count);
    const std::size_t source_elements = static_cast<std::size_t>(source_key_heads) * width;
    const std::size_t row_elements = static_cast<std::size_t>(convolved_row_elements);
    const std::size_t source_start = static_cast<std::size_t>(token) * row_elements
        + static_cast<std::size_t>(source_head) * width;
    const std::size_t target_start = (static_cast<std::size_t>(value_head) * tokens
        + static_cast<std::size_t>(token)) * width;

    float query_sum = 0.0f;
    float key_sum = 0.0f;
    if (lane == 0) {
        for (std::size_t column = 0; column < width; ++column) {
            const float query_value = convolved[source_start + column];
            const float key_value = convolved[source_start + source_elements + column];
            logismos_status_input(query_value, numerical_status);
            logismos_status_input(key_value, numerical_status);
            query_sum = logismos_status_add(
                query_sum, logismos_status_mul(query_value, query_value, numerical_status), numerical_status);
            key_sum = logismos_status_add(
                key_sum, logismos_status_mul(key_value, key_value, numerical_status), numerical_status);
        }
    }
    query_sum = __shfl(query_sum, 0, kWaveSize);
    key_sum = __shfl(key_sum, 0, kWaveSize);
    logismos_status_input(epsilon, numerical_status);
    const float query_root = logismos_status_sqrt(query_sum, numerical_status);
    const float key_root = logismos_status_sqrt(key_sum, numerical_status);
    const float query_denominator = fmaxf(query_root, epsilon);
    const float key_denominator = fmaxf(key_root, epsilon);
    logismos_status_result(query_denominator, numerical_status);
    logismos_status_result(key_denominator, numerical_status);
    for (std::size_t column = lane; column < width; column += kWaveSize) {
        const float query_value = convolved[source_start + column];
        const float key_value = convolved[source_start + source_elements + column];
        logismos_status_input(query_value, numerical_status);
        logismos_status_input(key_value, numerical_status);
        query[target_start + column] = logismos_status_div(query_value, query_denominator, numerical_status);
        key[target_start + column] = logismos_status_div(key_value, key_denominator, numerical_status);
    }
}
}

extern "C" hipError_t logismos_launch_decoder_recurrent_qk_l2_f32(
    const void* convolved,
    void* query,
    void* key,
    const std::uint32_t token_count,
    const std::uint32_t convolved_row_elements,
    const std::uint32_t source_key_heads,
    const std::uint32_t value_heads,
    const std::uint32_t key_width,
    const float epsilon,
    void* numerical_status,
    hipStream_t stream) {
    decoder_recurrent_qk_l2_f32_kernel<<<dim3(value_heads, token_count), dim3(kWaveSize), 0, stream>>>(
        reinterpret_cast<const float*>(convolved),
        reinterpret_cast<float*>(query),
        reinterpret_cast<float*>(key),
        token_count,
        convolved_row_elements,
        source_key_heads,
        key_width,
        epsilon,
        reinterpret_cast<std::uint32_t*>(numerical_status));
    return hipGetLastError();
}

namespace {
constexpr std::uint32_t kLayoutThreads = 256U;

__global__ void decoder_recurrent_values_to_head_major_f32_kernel(
    const float* __restrict__ convolved,
    float* __restrict__ values,
    const std::uint32_t token_count,
    const std::uint32_t convolved_row_elements,
    const std::uint32_t value_heads,
    const std::uint32_t value_width,
    const std::uint32_t value_offset,
    std::uint32_t* numerical_status) {
    const std::size_t lane = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    const std::size_t tokens = static_cast<std::size_t>(token_count);
    const std::size_t heads = static_cast<std::size_t>(value_heads);
    const std::size_t width = static_cast<std::size_t>(value_width);
    const std::size_t elements = tokens * heads * width;
    if (lane >= elements) return;
    const std::size_t column = lane % width;
    const std::size_t head_token = lane / width;
    const std::size_t token = head_token % tokens;
    const std::size_t head = head_token / tokens;
    const std::size_t source = token * static_cast<std::size_t>(convolved_row_elements)
        + static_cast<std::size_t>(value_offset) + head * width + column;
    values[lane] = logismos_status_result(logismos_status_input(convolved[source], numerical_status), numerical_status);
}

__global__ void decoder_recurrent_values_to_token_major_f32_kernel(
    const float* __restrict__ head_major,
    float* __restrict__ token_major,
    const std::uint32_t token_count,
    const std::uint32_t value_heads,
    const std::uint32_t value_width,
    std::uint32_t* numerical_status) {
    const std::size_t lane = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    const std::size_t tokens = static_cast<std::size_t>(token_count);
    const std::size_t heads = static_cast<std::size_t>(value_heads);
    const std::size_t width = static_cast<std::size_t>(value_width);
    const std::size_t elements = tokens * heads * width;
    if (lane >= elements) return;
    const std::size_t column = lane % width;
    const std::size_t token_head = lane / width;
    const std::size_t head = token_head % heads;
    const std::size_t token = token_head / heads;
    const std::size_t source = (head * tokens + token) * width + column;
    token_major[lane] = logismos_status_result(logismos_status_input(head_major[source], numerical_status), numerical_status);
}

std::uint32_t layout_blocks(const std::size_t elements) {
    return static_cast<std::uint32_t>((elements - 1U) / kLayoutThreads + 1U);
}
}

extern "C" hipError_t logismos_launch_decoder_recurrent_values_to_head_major_f32(
    const void* convolved, void* values, const std::uint32_t token_count,
    const std::uint32_t convolved_row_elements, const std::uint32_t value_heads,
    const std::uint32_t value_width, const std::uint32_t value_offset,
    void* numerical_status, hipStream_t stream) {
    const std::size_t elements = static_cast<std::size_t>(token_count)
        * static_cast<std::size_t>(value_heads) * static_cast<std::size_t>(value_width);
    decoder_recurrent_values_to_head_major_f32_kernel<<<dim3(layout_blocks(elements)), dim3(kLayoutThreads), 0, stream>>>(
        reinterpret_cast<const float*>(convolved), reinterpret_cast<float*>(values), token_count,
        convolved_row_elements, value_heads, value_width, value_offset,
        reinterpret_cast<std::uint32_t*>(numerical_status));
    return hipGetLastError();
}

extern "C" hipError_t logismos_launch_decoder_recurrent_values_to_token_major_f32(
    const void* head_major, void* token_major, const std::uint32_t token_count,
    const std::uint32_t value_heads, const std::uint32_t value_width,
    void* numerical_status, hipStream_t stream) {
    const std::size_t elements = static_cast<std::size_t>(token_count)
        * static_cast<std::size_t>(value_heads) * static_cast<std::size_t>(value_width);
    decoder_recurrent_values_to_token_major_f32_kernel<<<dim3(layout_blocks(elements)), dim3(kLayoutThreads), 0, stream>>>(
        reinterpret_cast<const float*>(head_major), reinterpret_cast<float*>(token_major), token_count,
        value_heads, value_width, reinterpret_cast<std::uint32_t*>(numerical_status));
    return hipGetLastError();
}

#pragma clang fp contract(on)
#pragma clang fp reassociate(on)
