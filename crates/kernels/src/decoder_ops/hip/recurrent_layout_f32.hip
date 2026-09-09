// T=1 Qwen recurrent Q/K arrangement and L2 normalization.
//
// The grouped GDN primitive has a floor key-head mapping. Qwen recurrent
// semantics instead tile source heads in modulo order, so this producer emits
// explicitly tiled Q/K rows before GDN receives an equal-head configuration.

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

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
    const std::uint32_t source_key_heads,
    const std::uint32_t key_width,
    const float epsilon) {
    const std::uint32_t value_head = blockIdx.x;
    const std::uint32_t lane = threadIdx.x;
    if (lane >= kWaveSize) {
        return;
    }
    const std::uint32_t source_head = value_head % source_key_heads;
    const std::size_t width = static_cast<std::size_t>(key_width);
    const std::size_t source_elements = static_cast<std::size_t>(source_key_heads) * width;
    const std::size_t source_start = static_cast<std::size_t>(source_head) * width;
    const std::size_t target_start = static_cast<std::size_t>(value_head) * width;

    float query_sum = 0.0f;
    float key_sum = 0.0f;
    if (lane == 0) {
        for (std::size_t column = 0; column < width; ++column) {
            const float query_value = convolved[source_start + column];
            const float key_value = convolved[source_elements + source_start + column];
            query_sum = query_sum + query_value * query_value;
            key_sum = key_sum + key_value * key_value;
        }
    }
    query_sum = __shfl(query_sum, 0, kWaveSize);
    key_sum = __shfl(key_sum, 0, kWaveSize);
    const float query_denominator = fmaxf(sqrtf(query_sum), epsilon);
    const float key_denominator = fmaxf(sqrtf(key_sum), epsilon);
    for (std::size_t column = lane; column < width; column += kWaveSize) {
        query[target_start + column] = convolved[source_start + column] / query_denominator;
        key[target_start + column] = convolved[source_elements + source_start + column] / key_denominator;
    }
}
}

extern "C" hipError_t logismos_launch_decoder_recurrent_qk_l2_f32(
    const void* convolved,
    void* query,
    void* key,
    const std::uint32_t source_key_heads,
    const std::uint32_t value_heads,
    const std::uint32_t key_width,
    const float epsilon,
    hipStream_t stream) {
    decoder_recurrent_qk_l2_f32_kernel<<<dim3(value_heads), dim3(kWaveSize), 0, stream>>>(
        reinterpret_cast<const float*>(convolved),
        reinterpret_cast<float*>(query),
        reinterpret_cast<float*>(key),
        source_key_heads,
        key_width,
        epsilon);
    return hipGetLastError();
}

#pragma clang fp contract(on)
#pragma clang fp reassociate(on)
