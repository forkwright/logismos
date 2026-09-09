// Compact f32 decoder primitives. The build script compiles this source with
// no-fast-math and contraction disabled because its f32 operation order is a
// correctness witness rather than a throughput-specialized approximation.

#include <cstddef>
#include <hip/hip_runtime.h>

namespace {
constexpr unsigned int kWaveSize = 32;
constexpr unsigned int kElementwiseThreads = 256;

__global__ void decoder_rms_norm_f32_kernel(
    const float* input,
    const float* weight,
    float* output,
    const unsigned int rows,
    const unsigned int width,
    const float epsilon) {
    const unsigned int row = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (row >= rows || lane >= kWaveSize) return;
    const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(width);
    float partial = 0.0f;
    for (size_t column = lane; column < static_cast<size_t>(width); column += kWaveSize) {
        const float value = input[row_offset + column];
        partial += value * value;
    }
    partial += __shfl_down(partial, 16, kWaveSize);
    partial += __shfl_down(partial, 8, kWaveSize);
    partial += __shfl_down(partial, 4, kWaveSize);
    partial += __shfl_down(partial, 2, kWaveSize);
    partial += __shfl_down(partial, 1, kWaveSize);
    float inverse = lane == 0 ? rsqrtf(partial / static_cast<float>(width) + epsilon) : 0.0f;
    inverse = __shfl(inverse, 0, kWaveSize);
    for (size_t column = lane; column < static_cast<size_t>(width); column += kWaveSize) {
        output[row_offset + column] = input[row_offset + column] * inverse * weight[column];
    }
}

__global__ void decoder_rotary_half_split_f32_kernel(
    float* values,
    const float* cosine,
    const float* sine,
    const unsigned int heads,
    const unsigned int width,
    const unsigned int rotary_width) {
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
    values[row + pair] = left_value * cos_value - right_value * sin_value;
    values[row + right] = left_value * sin_value + right_value * cos_value;
}

__global__ void decoder_split_q_gate_f32_kernel(
    const float* source,
    float* query,
    float* gate,
    const unsigned int heads,
    const unsigned int key_width) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t output_elements = static_cast<size_t>(heads) * static_cast<size_t>(key_width);
    if (index >= output_elements) return;
    const size_t head = index / static_cast<size_t>(key_width);
    const size_t column = index % static_cast<size_t>(key_width);
    const size_t source_start = head * static_cast<size_t>(key_width) * 2;
    query[index] = source[source_start + column];
    gate[index] = source[source_start + static_cast<size_t>(key_width) + column];
}

__global__ void decoder_sigmoid_mul_f32_kernel(const float* value, const float* gate, float* output, const unsigned int elements) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<size_t>(elements)) return;
    const float sigmoid = 1.0f / (1.0f + expf(-gate[index]));
    output[index] = value[index] * sigmoid;
}

__global__ void decoder_silu_mul_f32_kernel(const float* gate, const float* up, float* output, const unsigned int elements) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<size_t>(elements)) return;
    const float silu = gate[index] / (1.0f + expf(-gate[index]));
    output[index] = silu * up[index];
}

__global__ void decoder_residual_add_f32_kernel(const float* left, const float* right, float* output, const unsigned int elements) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<size_t>(elements)) return;
    output[index] = left[index] + right[index];
}

unsigned int blocks_for(const size_t elements) {
    return static_cast<unsigned int>((elements + kElementwiseThreads - 1) / kElementwiseThreads);
}
}

extern "C" hipError_t logismos_launch_decoder_rms_norm_f32(const void* input, const void* weight, void* output, unsigned int rows, unsigned int width, float epsilon, hipStream_t stream) {
    decoder_rms_norm_f32_kernel<<<dim3(rows), dim3(kWaveSize), 0, stream>>>(reinterpret_cast<const float*>(input), reinterpret_cast<const float*>(weight), reinterpret_cast<float*>(output), rows, width, epsilon);
    return hipGetLastError();
}

extern "C" hipError_t logismos_launch_decoder_rotary_half_split_f32(void* values, const void* cosine, const void* sine, unsigned int heads, unsigned int width, unsigned int rotary_width, hipStream_t stream) {
    const size_t elements = static_cast<size_t>(heads) * (static_cast<size_t>(rotary_width) / 2);
    decoder_rotary_half_split_f32_kernel<<<dim3(blocks_for(elements)), dim3(kElementwiseThreads), 0, stream>>>(reinterpret_cast<float*>(values), reinterpret_cast<const float*>(cosine), reinterpret_cast<const float*>(sine), heads, width, rotary_width);
    return hipGetLastError();
}

extern "C" hipError_t logismos_launch_decoder_split_q_gate_f32(const void* source, void* query, void* gate, unsigned int heads, unsigned int key_width, hipStream_t stream) {
    const size_t elements = static_cast<size_t>(heads) * static_cast<size_t>(key_width);
    decoder_split_q_gate_f32_kernel<<<dim3(blocks_for(elements)), dim3(kElementwiseThreads), 0, stream>>>(reinterpret_cast<const float*>(source), reinterpret_cast<float*>(query), reinterpret_cast<float*>(gate), heads, key_width);
    return hipGetLastError();
}

extern "C" hipError_t logismos_launch_decoder_sigmoid_mul_f32(const void* value, const void* gate, void* output, unsigned int elements, hipStream_t stream) {
    decoder_sigmoid_mul_f32_kernel<<<dim3(blocks_for(elements)), dim3(kElementwiseThreads), 0, stream>>>(reinterpret_cast<const float*>(value), reinterpret_cast<const float*>(gate), reinterpret_cast<float*>(output), elements);
    return hipGetLastError();
}

extern "C" hipError_t logismos_launch_decoder_silu_mul_f32(const void* gate, const void* up, void* output, unsigned int elements, hipStream_t stream) {
    decoder_silu_mul_f32_kernel<<<dim3(blocks_for(elements)), dim3(kElementwiseThreads), 0, stream>>>(reinterpret_cast<const float*>(gate), reinterpret_cast<const float*>(up), reinterpret_cast<float*>(output), elements);
    return hipGetLastError();
}

extern "C" hipError_t logismos_launch_decoder_residual_add_f32(const void* left, const void* right, void* output, unsigned int elements, hipStream_t stream) {
    decoder_residual_add_f32_kernel<<<dim3(blocks_for(elements)), dim3(kElementwiseThreads), 0, stream>>>(reinterpret_cast<const float*>(left), reinterpret_cast<const float*>(right), reinterpret_cast<float*>(output), elements);
    return hipGetLastError();
}
