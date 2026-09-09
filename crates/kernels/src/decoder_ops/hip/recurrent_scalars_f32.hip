// WHY: T=1 recurrent scalars retain the CPU operation order exactly.

#include <cstddef>
#include <hip/hip_runtime.h>

#include "numerical_status.h"

namespace {
constexpr unsigned int kThreads = 256;

__device__ float stable_sigmoid(const float value, std::uint32_t* numerical_status) {
    if (value >= 0.0f) {
        return logismos_status_div(
            1.0f,
            logismos_status_add(
                1.0f,
                logismos_status_exp(logismos_status_sub(0.0f, value, numerical_status), numerical_status),
                numerical_status),
            numerical_status);
    }
    const float exponential = logismos_status_exp(value, numerical_status);
    return logismos_status_div(
        exponential,
        logismos_status_add(1.0f, exponential, numerical_status),
        numerical_status);
}

__device__ float stable_softplus(const float value, std::uint32_t* numerical_status) {
    const float positive = fmaxf(value, 0.0f);
    const float magnitude = fabsf(value);
    logismos_status_result(positive, numerical_status);
    logismos_status_result(magnitude, numerical_status);
    return logismos_status_add(
        positive,
        logismos_status_log1p(
            logismos_status_exp(logismos_status_sub(0.0f, magnitude, numerical_status), numerical_status),
            numerical_status),
        numerical_status);
}

__global__ void recurrent_scalars_f32_kernel(
    const float* alpha,
    const float* dt,
    const float* a,
    const float* beta_projection,
    float* beta,
    float* log_decay,
    const unsigned int value_heads,
    std::uint32_t* numerical_status) {
    const size_t head = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (head >= static_cast<size_t>(value_heads)) return;
    const float alpha_value = alpha[head];
    const float dt_value = dt[head];
    const float a_value = a[head];
    const float beta_projection_value = beta_projection[head];
    logismos_status_input(alpha_value, numerical_status);
    logismos_status_input(dt_value, numerical_status);
    logismos_status_input(a_value, numerical_status);
    logismos_status_input(beta_projection_value, numerical_status);
    beta[head] = stable_sigmoid(beta_projection_value, numerical_status);
    const float alpha_plus_dt = logismos_status_add(alpha_value, dt_value, numerical_status);
    log_decay[head] = logismos_status_mul(
        a_value, stable_softplus(alpha_plus_dt, numerical_status), numerical_status);
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
    void* numerical_status,
    hipStream_t stream) {
    recurrent_scalars_f32_kernel<<<dim3(blocks_for(value_heads)), dim3(kThreads), 0, stream>>>(
        reinterpret_cast<const float*>(alpha),
        reinterpret_cast<const float*>(dt),
        reinterpret_cast<const float*>(a),
        reinterpret_cast<const float*>(beta_projection),
        reinterpret_cast<float*>(beta),
        reinterpret_cast<float*>(log_decay),
        value_heads,
        reinterpret_cast<std::uint32_t*>(numerical_status));
    return hipGetLastError();
}
