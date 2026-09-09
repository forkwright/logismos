// WHY: T=1 recurrent scalars retain the CPU operation order exactly.

#include <cstddef>
#include <hip/hip_runtime.h>

namespace {
constexpr unsigned int kThreads = 256;

__device__ float stable_sigmoid(const float value) {
    if (value >= 0.0f) {
        return 1.0f / (1.0f + expf(-value));
    }
    const float exponential = expf(value);
    return exponential / (1.0f + exponential);
}

__device__ float stable_softplus(const float value) {
    return fmaxf(value, 0.0f) + log1pf(expf(-fabsf(value)));
}

__global__ void recurrent_scalars_f32_kernel(
    const float* alpha,
    const float* dt,
    const float* a,
    const float* beta_projection,
    float* beta,
    float* log_decay,
    const unsigned int value_heads) {
    const size_t head = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (head >= static_cast<size_t>(value_heads)) return;
    beta[head] = stable_sigmoid(beta_projection[head]);
    const float alpha_plus_dt = alpha[head] + dt[head];
    log_decay[head] = a[head] * stable_softplus(alpha_plus_dt);
}

unsigned int blocks_for(const size_t value_heads) {
    return static_cast<unsigned int>((value_heads + kThreads - 1) / kThreads);
}
}

extern "C" hipError_t logismos_launch_recurrent_scalars_f32(
    const void* alpha,
    const void* dt,
    const void* a,
    const void* beta_projection,
    void* beta,
    void* log_decay,
    const unsigned int value_heads,
    hipStream_t stream) {
    recurrent_scalars_f32_kernel<<<dim3(blocks_for(value_heads)), dim3(kThreads), 0, stream>>>(
        reinterpret_cast<const float*>(alpha),
        reinterpret_cast<const float*>(dt),
        reinterpret_cast<const float*>(a),
        reinterpret_cast<const float*>(beta_projection),
        reinterpret_cast<float*>(beta),
        reinterpret_cast<float*>(log_decay),
        value_heads);
    return hipGetLastError();
}
