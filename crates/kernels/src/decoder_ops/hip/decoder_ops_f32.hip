// Compact f32 decoder primitives. The build script compiles this source with
// no-fast-math and contraction disabled because its f32 operation order is a
// correctness witness rather than a throughput-specialized approximation.

#include <cstddef>
#include <hip/hip_runtime.h>

#include "numerical_status.h"

namespace {
constexpr unsigned int kWaveSize = 32;
constexpr unsigned int kElementwiseThreads = 256;

__global__ void decoder_rms_norm_f32_kernel(
    const float* input,
    const float* weight,
    float* output,
    const unsigned int rows,
    const unsigned int width,
    const float epsilon,
    std::uint32_t* numerical_status) {
    const unsigned int row = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (row >= rows || lane >= kWaveSize) return;
    const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(width);
    float partial = 0.0f;
    for (size_t column = lane; column < static_cast<size_t>(width); column += kWaveSize) {
        const float value = input[row_offset + column];
        logismos_status_input(value, numerical_status);
        partial = logismos_status_add(
            partial,
            logismos_status_mul(value, value, numerical_status),
            numerical_status);
    }
    const float shuffled16 = __shfl_down(partial, 16, kWaveSize);
    if (lane < 16) partial = logismos_status_add(partial, shuffled16, numerical_status);
    const float shuffled8 = __shfl_down(partial, 8, kWaveSize);
    if (lane < 8) partial = logismos_status_add(partial, shuffled8, numerical_status);
    const float shuffled4 = __shfl_down(partial, 4, kWaveSize);
    if (lane < 4) partial = logismos_status_add(partial, shuffled4, numerical_status);
    const float shuffled2 = __shfl_down(partial, 2, kWaveSize);
    if (lane < 2) partial = logismos_status_add(partial, shuffled2, numerical_status);
    const float shuffled1 = __shfl_down(partial, 1, kWaveSize);
    if (lane < 1) partial = logismos_status_add(partial, shuffled1, numerical_status);
    float inverse = 0.0f;
    if (lane == 0) {
        logismos_status_input(epsilon, numerical_status);
        const float mean = logismos_status_div(partial, static_cast<float>(width), numerical_status);
        const float variance = logismos_status_add(mean, epsilon, numerical_status);
        inverse = logismos_status_rsqrt(variance, numerical_status);
    }
    inverse = __shfl(inverse, 0, kWaveSize);
    for (size_t column = lane; column < static_cast<size_t>(width); column += kWaveSize) {
        const float value = input[row_offset + column];
        const float scale = weight[column];
        logismos_status_input(value, numerical_status);
        logismos_status_input(scale, numerical_status);
        output[row_offset + column] = logismos_status_mul(
            logismos_status_mul(value, inverse, numerical_status), scale, numerical_status);
    }
}

__global__ void decoder_rotary_half_split_f32_kernel(
    float* values,
    const float* cosine,
    const float* sine,
    const unsigned int heads,
    const unsigned int width,
    const unsigned int rotary_width,
    std::uint32_t* numerical_status) {
    const size_t pairs = static_cast<size_t>(rotary_width) / 2;
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t work_items = static_cast<size_t>(heads) * pairs;
    if (index >= work_items) return;
    const size_t head = index / pairs;
    const size_t pair = index % pairs;
    const size_t row = head * static_cast<size_t>(width);
    const size_t right = pair + pairs;
    const float left_value = values[row + pair];
    const float right_value = values[row + right];
    const float cos_value = cosine[pair];
    const float sin_value = sine[pair];
    logismos_status_input(left_value, numerical_status);
    logismos_status_input(right_value, numerical_status);
    logismos_status_input(cos_value, numerical_status);
    logismos_status_input(sin_value, numerical_status);
    values[row + pair] = logismos_status_sub(
        logismos_status_mul(left_value, cos_value, numerical_status),
        logismos_status_mul(right_value, sin_value, numerical_status),
        numerical_status);
    values[row + right] = logismos_status_add(
        logismos_status_mul(left_value, sin_value, numerical_status),
        logismos_status_mul(right_value, cos_value, numerical_status),
        numerical_status);
}

__global__ void decoder_split_q_gate_f32_kernel(
    const float* source,
    float* query,
    float* gate,
    const unsigned int heads,
    const unsigned int key_width,
    std::uint32_t* numerical_status) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t output_elements = static_cast<size_t>(heads) * static_cast<size_t>(key_width);
    if (index >= output_elements) return;
    const size_t head = index / static_cast<size_t>(key_width);
    const size_t column = index % static_cast<size_t>(key_width);
    const size_t source_start = head * static_cast<size_t>(key_width) * 2;
    const float query_value = source[source_start + column];
    const float gate_value = source[source_start + static_cast<size_t>(key_width) + column];
    logismos_status_input(query_value, numerical_status);
    logismos_status_input(gate_value, numerical_status);
    query[index] = query_value;
    gate[index] = gate_value;
}

__global__ void decoder_sigmoid_mul_f32_kernel(const float* value, const float* gate, float* output, const unsigned int elements, std::uint32_t* numerical_status) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<size_t>(elements)) return;
    const float value_input = value[index];
    const float gate_input = gate[index];
    logismos_status_input(value_input, numerical_status);
    logismos_status_input(gate_input, numerical_status);
    const float negated_gate = logismos_status_sub(0.0f, gate_input, numerical_status);
    const float denominator = logismos_status_add(
        1.0f, logismos_status_exp(negated_gate, numerical_status), numerical_status);
    const float sigmoid = logismos_status_div(1.0f, denominator, numerical_status);
    output[index] = logismos_status_mul(value_input, sigmoid, numerical_status);
}

__global__ void decoder_silu_f32_kernel(const float* input, float* output, const unsigned int elements, std::uint32_t* numerical_status) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<size_t>(elements)) return;
    const float value = input[index];
    logismos_status_input(value, numerical_status);
    const float negated_value = logismos_status_sub(0.0f, value, numerical_status);
    const float denominator = logismos_status_add(
        1.0f, logismos_status_exp(negated_value, numerical_status), numerical_status);
    output[index] = logismos_status_div(value, denominator, numerical_status);
}

__global__ void decoder_silu_mul_f32_kernel(const float* gate, const float* up, float* output, const unsigned int elements, std::uint32_t* numerical_status) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<size_t>(elements)) return;
    const float gate_value = gate[index];
    const float up_value = up[index];
    logismos_status_input(gate_value, numerical_status);
    logismos_status_input(up_value, numerical_status);
    const float negated_gate = logismos_status_sub(0.0f, gate_value, numerical_status);
    const float denominator = logismos_status_add(
        1.0f, logismos_status_exp(negated_gate, numerical_status), numerical_status);
    const float silu = logismos_status_div(gate_value, denominator, numerical_status);
    output[index] = logismos_status_mul(silu, up_value, numerical_status);
}

__global__ void decoder_residual_add_f32_kernel(const float* left, const float* right, float* output, const unsigned int elements, std::uint32_t* numerical_status) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<size_t>(elements)) return;
    const float left_value = left[index];
    const float right_value = right[index];
    logismos_status_input(left_value, numerical_status);
    logismos_status_input(right_value, numerical_status);
    output[index] = logismos_status_add(left_value, right_value, numerical_status);
}

unsigned int blocks_for(const size_t elements) {
    return static_cast<unsigned int>((elements + kElementwiseThreads - 1) / kElementwiseThreads);
}
}

extern "C" hipError_t logismos_launch_decoder_rms_norm_f32(const void* input, const void* weight, void* output, unsigned int rows, unsigned int width, float epsilon, void* numerical_status, hipStream_t stream) {
    decoder_rms_norm_f32_kernel<<<dim3(rows), dim3(kWaveSize), 0, stream>>>(reinterpret_cast<const float*>(input), reinterpret_cast<const float*>(weight), reinterpret_cast<float*>(output), rows, width, epsilon, reinterpret_cast<std::uint32_t*>(numerical_status));
    return hipGetLastError();
}

extern "C" hipError_t logismos_launch_decoder_rotary_half_split_f32(void* values, const void* cosine, const void* sine, unsigned int heads, unsigned int width, unsigned int rotary_width, void* numerical_status, hipStream_t stream) {
    const size_t elements = static_cast<size_t>(heads) * (static_cast<size_t>(rotary_width) / 2);
    decoder_rotary_half_split_f32_kernel<<<dim3(blocks_for(elements)), dim3(kElementwiseThreads), 0, stream>>>(reinterpret_cast<float*>(values), reinterpret_cast<const float*>(cosine), reinterpret_cast<const float*>(sine), heads, width, rotary_width, reinterpret_cast<std::uint32_t*>(numerical_status));
    return hipGetLastError();
}

extern "C" hipError_t logismos_launch_decoder_split_q_gate_f32(const void* source, void* query, void* gate, unsigned int heads, unsigned int key_width, void* numerical_status, hipStream_t stream) {
    const size_t elements = static_cast<size_t>(heads) * static_cast<size_t>(key_width);
    decoder_split_q_gate_f32_kernel<<<dim3(blocks_for(elements)), dim3(kElementwiseThreads), 0, stream>>>(reinterpret_cast<const float*>(source), reinterpret_cast<float*>(query), reinterpret_cast<float*>(gate), heads, key_width, reinterpret_cast<std::uint32_t*>(numerical_status));
    return hipGetLastError();
}

extern "C" hipError_t logismos_launch_decoder_sigmoid_mul_f32(const void* value, const void* gate, void* output, unsigned int elements, void* numerical_status, hipStream_t stream) {
    decoder_sigmoid_mul_f32_kernel<<<dim3(blocks_for(elements)), dim3(kElementwiseThreads), 0, stream>>>(reinterpret_cast<const float*>(value), reinterpret_cast<const float*>(gate), reinterpret_cast<float*>(output), elements, reinterpret_cast<std::uint32_t*>(numerical_status));
    return hipGetLastError();
}

extern "C" hipError_t logismos_launch_decoder_silu_f32(const void* input, void* output, unsigned int elements, void* numerical_status, hipStream_t stream) {
    decoder_silu_f32_kernel<<<dim3(blocks_for(elements)), dim3(kElementwiseThreads), 0, stream>>>(reinterpret_cast<const float*>(input), reinterpret_cast<float*>(output), elements, reinterpret_cast<std::uint32_t*>(numerical_status));
    return hipGetLastError();
}

extern "C" hipError_t logismos_launch_decoder_silu_mul_f32(const void* gate, const void* up, void* output, unsigned int elements, void* numerical_status, hipStream_t stream) {
    decoder_silu_mul_f32_kernel<<<dim3(blocks_for(elements)), dim3(kElementwiseThreads), 0, stream>>>(reinterpret_cast<const float*>(gate), reinterpret_cast<const float*>(up), reinterpret_cast<float*>(output), elements, reinterpret_cast<std::uint32_t*>(numerical_status));
    return hipGetLastError();
}

extern "C" hipError_t logismos_launch_decoder_residual_add_f32(const void* left, const void* right, void* output, unsigned int elements, void* numerical_status, hipStream_t stream) {
    decoder_residual_add_f32_kernel<<<dim3(blocks_for(elements)), dim3(kElementwiseThreads), 0, stream>>>(reinterpret_cast<const float*>(left), reinterpret_cast<const float*>(right), reinterpret_cast<float*>(output), elements, reinterpret_cast<std::uint32_t*>(numerical_status));
    return hipGetLastError();
}
